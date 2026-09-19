"""Quantum ESPRESSO ``pw.x``: input rendering, unit planning, output parsing, collection.

CONTRACTS.md row 6. Nothing here runs ``pw.x``; the bash snippet :data:`UNIT_SCRIPT` does, once
per unit, under either executor (``dft run``).

Unit conventions (binding for every consumer of the ``DFTFrame``s produced here)
--------------------------------------------------------------------------------
* Energies: Ry -> eV with ``ase.units.Ry``; forces: Ry/Bohr -> eV/Å with ``ase.units.Bohr``
  (the constants ``ase.io.espresso`` uses, so QE frames and ASE-read QE outputs agree bit for bit).
* Stress is read from the ``total   stress  (Ry/bohr**3)`` columns (8 decimals; the kbar columns
  carry only 2) and converted with Ry/Bohr³ -> eV/Å³. **Sign**: ``pw.x`` prints ``σ_QE`` with
  pressure ``P = Tr(σ_QE)/3`` (compressive = positive); ASE and MACE use ``σ = -σ_QE``
  (tensile = positive, ``P = -Tr(σ)/3``), the same flip ``ase.io.espresso.read_espresso_out``
  applies (``stress *= -1 * Ry / Bohr**3``). ``Frame.stress`` holds the ASE sign as Voigt-6
  ``[xx, yy, zz, yz, xz, xy]``; QE's pressure line is kept verbatim in ``info["pressure_kbar"]``.
* k-mesh: ``k_i = max(1, ceil(|b_i| / Δk))`` with ``|b_i| = 2π |a*_i|`` (reciprocal vectors
  *including* 2π), the convention of ASE ``kspacing`` and VASP ``KSPACING``; ``Δk`` is
  ``cfg.dft.k_spacing_inv_A`` (0.25 1/Å -> 6x6x6 for a 4.56 Å B20 cell). The mesh is unshifted
  (Γ-centred, ``0 0 0``): a Γ-shifted mesh has no symmetry advantage for the low-symmetry rattled
  and sheared cells, and odd/even meshes of the convergence scan stay comparable.
* Magnetisation: ``total``/``absolute magnetization`` in μB per cell (``Frame.total_magnetization``,
  ``DFTFrame.abs_magnetization``); ``Frame.magmoms`` are the ``magn=`` values of the LAST
  ``Magnetic moment per site`` block (μB integrated in atomic spheres of radius R, so they slightly
  under-count the true site moments); no block -> ``magmoms=None``. Non-spin-polarised runs give
  ``total_magnetization=None``.
* Convergence: ``converged`` is true only for ``convergence has been achieved in N iterations``.
  A missing, truncated or ``convergence NOT achieved`` output yields ``converged=False`` — never an
  exception — with ``energy`` taken from the final ``!    total energy`` line only (absent for an
  unconverged SCF), ``scf_steps`` from the convergence line or the last ``iteration #``.
* ``pseudo_dir`` is written verbatim when absolute, else resolved against the repository root at
  render time (the Tillicum overlay must therefore set an absolute cluster path, which
  ``cluster bootstrap`` does); ``outdir='./tmp'`` is per unit and removed after a converged run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import jinja2
import numpy as np
from ase import units
from ase.data import atomic_masses, atomic_numbers, chemical_symbols

from b20mlip.config import DFTConfig, Settings, load_config, repo_root
from b20mlip.executors import JobSpec, unit_slug
from b20mlip.models import DFTFrame, Frame

# --- constants ---------------------------------------------------------------------------------

RY_TO_EV = float(units.Ry)
BOHR_TO_A = float(units.Bohr)
RY_BOHR_TO_EV_A = RY_TO_EV / BOHR_TO_A
RY_BOHR3_TO_EV_A3 = RY_TO_EV / BOHR_TO_A**3
KBAR_TO_EV_A3 = 1e3 * float(units.bar)

TEMPLATE_NAME = "pw.in.j2"
PW_INPUT = "pw.in"
PW_OUTPUT = "pw.out"
UNIT_JSON = "unit.json"
UNITS_INDEX = "units.json"
UNIT_SCHEMA = "b20mlip.qe_unit.v1"
UNITS_SCHEMA = "b20mlip.qe_units.v1"
DONE_MARKER = ".done"
FAILED_MARKER = ".failed"
OUTDIR = "./tmp"
TM_NUMBERS = frozenset({25, 26, 27})  # Mn, Fe, Co: the atoms the magnetic-branch check averages
UnitStatus = Literal["done", "failed", "pending"]

OVERRIDE_KEYS = frozenset(
    {
        "ecut_ry",
        "ecut_rho",
        "k_spacing_inv_A",
        "kpoints",
        "nspin",
        "starting_magnetization",
        "smearing",
        "degauss_ry",
        "conv_thr",
        "mixing_beta",
        "electron_maxstep",
        "prefix",
        "pseudo_dir",
        "outdir",
    }
)

#: Bash snippet run once per unit by both executors (``JobSpec.script``). Reads ``B20_UNIT``,
#: ``B20_UNITS_ROOT`` (directory holding the unit dirs, local or the cluster mirror), ``QE_CMD``
#: (``cfg.cluster.qe_cmd``); ``B20_KEEP_TMP=1`` keeps the wavefunction directory. Success means
#: SCF convergence in ``pw.out`` (marker ``.done``), anything else writes ``.failed`` and exits 1;
#: a unit with ``.done`` is skipped, so resubmission under any job name is idempotent.
UNIT_SCRIPT = r"""# b20mlip QE unit: cd into the unit dir, run pw.x, succeed only on convergence.
unit_dir="${B20_UNITS_ROOT:?B20_UNITS_ROOT is unset}/${B20_UNIT:?B20_UNIT is unset}"
cd "$unit_dir"
if [ -f .done ]; then echo "unit $B20_UNIT already done (.done marker)"; exit 0; fi
rm -f .failed
: "${QE_CMD:?QE_CMD is unset (cfg.cluster.qe_cmd)}"
start=$(date +%s)
set +e
$QE_CMD -in pw.in > pw.out 2> pw.err
rc=$?
set -e
# Davidson occasionally dies with "too many bands are not converged" (c_bands) on rattled or
# hot frames: retry ONCE with the conjugate-gradient diagonaliser and a softer mixing before
# giving up (retry input kept as pw_retry.in; the failed attempt's output as pw_attempt1.out).
davidson='too many bands are not converged\|Error in routine c_bands'
if ! grep -q "convergence has been achieved" pw.out && grep -q "$davidson" pw.out; then
  cp pw.out pw_attempt1.out
  sed -e "s/^&ELECTRONS/\&ELECTRONS\n  diagonalization = 'cg'\n  diago_full_acc = .true./" \
      -e "s/mixing_beta = [0-9.]*/mixing_beta = 0.3/" pw.in > pw_retry.in
  rm -rf tmp
  set +e
  $QE_CMD -in pw_retry.in > pw.out 2>> pw.err
  rc=$?
  set -e
  echo "unit $B20_UNIT: Davidson failure, retried with diagonalization='cg' (exit $rc)" >&2
fi
wall=$(( $(date +%s) - start ))
if grep -q "convergence has been achieved" pw.out; then
  [ "${B20_KEEP_TMP:-0}" = "1" ] || rm -rf tmp
  printf '{"unit": "%s", "state": "done", "returncode": %d, "wall_seconds": %d}\n' \
    "$B20_UNIT" "$rc" "$wall" > .done
  exit 0
fi
[ -s pw.out ] || rm -f pw.out
printf '{"unit": "%s", "state": "failed", "returncode": %d, "wall_seconds": %d}\n' \
  "$B20_UNIT" "$rc" "$wall" > .failed
echo "unit $B20_UNIT: pw.x exit $rc, no SCF convergence in pw.out" >&2
exit 1
"""

# --- rendering ---------------------------------------------------------------------------------


def template_dir() -> Path:
    return repo_root() / "templates" / "qe"


def _jinja(directory: Path | None = None) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(directory or template_dir())),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # Fortran namelists, not HTML
    )


def species_order(numbers: Iterable[int]) -> list[str]:
    """Chemical symbols in order of first appearance (the ATOMIC_SPECIES / type index order)."""
    out: list[str] = []
    for z in numbers:
        symbol = chemical_symbols[int(z)]
        if symbol not in out:
            out.append(symbol)
    return out


def kmesh(cell: Sequence[Sequence[float]], spacing: float) -> tuple[int, int, int]:
    """``k_i = max(1, ceil(|b_i| / spacing))`` with ``|b_i| = 2π |a*_i|`` (see module docstring)."""
    if spacing <= 0:
        raise ValueError(f"k-spacing must be positive, got {spacing}")
    recip = 2.0 * math.pi * np.linalg.inv(np.asarray(cell, dtype=np.float64)).T
    lengths = np.linalg.norm(recip, axis=1)
    mesh = tuple(max(1, int(math.ceil(float(b) / spacing - 1e-8))) for b in lengths)
    return mesh[0], mesh[1], mesh[2]


def nspin_for(compound: str, cfg: Settings, numbers: Iterable[int] = ()) -> int:
    """``cfg.dft.nspin[compound]``; unknown compounds are spin-polarised iff a species has a
    non-zero ``starting_magnetization`` (a non-magnetic solution is a subset of nspin=2)."""
    if compound in cfg.dft.nspin:
        return int(cfg.dft.nspin[compound])
    symbols = species_order(numbers)
    magnetic = any(cfg.dft.starting_magnetization.get(s, 0.0) != 0.0 for s in symbols)
    return 2 if magnetic else 1


def resolve_pseudo_dir(pseudo_dir: str | Path) -> str:
    p = Path(pseudo_dir).expanduser()
    return str(p if p.is_absolute() else repo_root() / p)


def load_sssp(path: str | Path) -> dict[str, dict[str, Any]]:
    """Element -> ``{filename, md5, cutoff_wfc, cutoff_rho, pseudopotential}`` from the SSSP JSON
    (either the raw Materials Cloud file or ``configs/dft/sssp_efficiency_1.3_pbe.json``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    elements = data.get("elements", data) if isinstance(data, dict) else {}
    return {str(k): dict(v) for k, v in elements.items() if isinstance(v, Mapping)}


def sssp_cutoffs(symbols: Iterable[str], cfg: Settings) -> tuple[float, float, float]:
    """``(max cutoff_wfc, max cutoff_rho, max dual)`` over ``symbols`` from ``cfg.dft.sssp_json``;
    falls back to ``(cfg.dft.ecut_ry, cfg.dft.ecut_rho, ratio)`` when the file or an element is
    absent (never invents SSSP values)."""
    fallback = (cfg.dft.ecut_ry, cfg.dft.ecut_rho, cfg.dft.ecut_rho / cfg.dft.ecut_ry)
    path = Path(cfg.dft.sssp_json)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.is_file():
        return fallback
    table = load_sssp(path)
    wanted = list(dict.fromkeys(symbols))
    if not wanted or any(s not in table for s in wanted):
        return fallback
    wfc = max(float(table[s]["cutoff_wfc"]) for s in wanted)
    rho = max(float(table[s]["cutoff_rho"]) for s in wanted)
    dual = max(float(table[s]["cutoff_rho"]) / float(table[s]["cutoff_wfc"]) for s in wanted)
    return wfc, rho, dual


def _check_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    ov = dict(overrides or {})
    unknown = set(ov) - OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"unknown pw.in overrides {sorted(unknown)}; allowed: {sorted(OVERRIDE_KEYS)}"
        )
    return ov


def pw_parameters(
    frame: Frame,
    cfg: Settings,
    *,
    overrides: Mapping[str, Any] | None = None,
    unit_id: str | None = None,
) -> dict[str, Any]:
    """Resolve every pw.in value for ``frame`` (template context + raw ``values`` for unit.json).

    ``overrides`` may set ``ecut_ry``, ``ecut_rho``, ``k_spacing_inv_A``, ``kpoints`` (``"gamma"``
    or ``"automatic"``), ``nspin``, ``starting_magnetization`` (element -> value, merged over the
    config), ``smearing``, ``degauss_ry``, ``conv_thr``, ``mixing_beta``, ``electron_maxstep``,
    ``prefix``, ``pseudo_dir`` and ``outdir``.
    """
    ov = _check_overrides(overrides)
    dft = cfg.dft
    symbols = species_order(frame.numbers)
    missing = [s for s in symbols if s not in dft.pseudos]
    if missing:
        raise ValueError(
            f"no pseudopotential file configured for {missing} (cfg.dft.pseudos, SSSP metadata)"
        )
    ecut = float(ov.get("ecut_ry", dft.ecut_ry))
    ecutrho = float(ov.get("ecut_rho", dft.ecut_rho))
    spacing = float(ov.get("k_spacing_inv_A", dft.k_spacing_inv_A))
    kpoints = str(ov.get("kpoints", "automatic"))
    if kpoints not in ("automatic", "gamma"):
        raise ValueError(f"kpoints must be 'automatic' or 'gamma', got {kpoints!r}")
    nspin = int(ov.get("nspin", nspin_for(frame.compound, cfg, frame.numbers)))
    if nspin not in (1, 2):
        raise ValueError(f"nspin must be 1 or 2 (collinear only), got {nspin}")
    start_mag = {**dft.starting_magnetization, **dict(ov.get("starting_magnetization", {}))}
    smearing = str(ov.get("smearing", dft.smearing))
    degauss = float(ov.get("degauss_ry", dft.degauss_ry))
    conv_thr = float(ov.get("conv_thr", dft.conv_thr))
    mixing_beta = float(ov.get("mixing_beta", dft.mixing_beta))
    electron_maxstep = int(ov.get("electron_maxstep", dft.electron_maxstep))
    mesh = (1, 1, 1) if kpoints == "gamma" else kmesh(frame.cell, spacing)
    prefix = str(ov.get("prefix", unit_id or frame.frame_id))
    pseudo_dir = resolve_pseudo_dir(ov.get("pseudo_dir", dft.pseudo_dir))
    outdir = str(ov.get("outdir", OUTDIR))

    species = [
        {
            "symbol": s,
            "mass": f"{float(atomic_masses[atomic_numbers[s]]):.4f}",
            "pseudo": dft.pseudos[s],
            "starting_magnetization": f"{float(start_mag.get(s, 0.0)):.2f}",
        }
        for s in symbols
    ]
    cell_rows = [
        " ".join(f"{float(x):16.10f}" for x in row) for row in np.asarray(frame.cell, float)
    ]
    atom_rows = [
        f"{chemical_symbols[int(z)]:<2s} " + " ".join(f"{float(x):16.10f}" for x in pos)
        for z, pos in zip(frame.numbers, frame.positions, strict=True)
    ]
    values = {
        "ecut_ry": ecut,
        "ecut_rho": ecutrho,
        "k_spacing_inv_A": spacing,
        "kpoints": kpoints,
        "kmesh": list(mesh),
        "nspin": nspin,
        "starting_magnetization": {s: float(start_mag.get(s, 0.0)) for s in symbols},
        "smearing": smearing,
        "degauss_ry": degauss,
        "conv_thr": conv_thr,
        "mixing_beta": mixing_beta,
        "electron_maxstep": electron_maxstep,
        "pseudos": {s: dft.pseudos[s] for s in symbols},
        "pseudo_md5s": {s: dft.pseudo_md5s[s] for s in symbols if s in dft.pseudo_md5s},
        "pseudo_family": dft.pseudo_family,
        "pseudo_dir": pseudo_dir,
        "prefix": prefix,
        "m_ref_muB": dft.m_ref_muB.get(frame.compound),
        "branch_tol_muB": dft.branch_tol_muB,
    }
    return {
        "prefix": prefix,
        "pseudo_dir": pseudo_dir,
        "outdir": outdir,
        "nat": len(frame.numbers),
        "ntyp": len(species),
        "ecutwfc": f"{ecut:.1f}",
        "ecutrho": f"{ecutrho:.1f}",
        "smearing": smearing,
        "degauss": f"{degauss:g}",
        "nspin": nspin,
        "species": species,
        "conv_thr": f"{conv_thr:.1e}",
        "mixing_beta": f"{mixing_beta:g}",
        "electron_maxstep": electron_maxstep,
        "cell": cell_rows,
        "atoms": atom_rows,
        "kpoints": kpoints,
        "kmesh": mesh,
        "values": values,
    }


def render_pw_input(
    frame: Frame,
    cfg: Settings,
    *,
    overrides: Mapping[str, Any] | None = None,
    unit_id: str | None = None,
    template_path: str | Path | None = None,
) -> str:
    """Render ``templates/qe/pw.in.j2`` for ``frame`` (see :func:`pw_parameters`)."""
    params = pw_parameters(frame, cfg, overrides=overrides, unit_id=unit_id)
    directory = Path(template_path).parent if template_path else template_dir()
    name = Path(template_path).name if template_path else TEMPLATE_NAME
    return _jinja(directory).get_template(name).render(**params)


# --- units ------------------------------------------------------------------------------------


def unit_dir(frame: Frame, root: str | Path, unit_id: str | None = None) -> Path:
    """``root/<unit id>``; the unit id defaults to the frame id (the geometry hash)."""
    return Path(root) / (unit_id or frame.frame_id)


def unit_status(directory: str | Path) -> UnitStatus:
    d = Path(directory)
    if (d / DONE_MARKER).is_file():
        return "done"
    if (d / FAILED_MARKER).is_file():
        return "failed"
    return "pending"


def read_unit(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / UNIT_JSON).read_text(encoding="utf-8"))


def _write_index(root: Path, unit_ids: list[str]) -> None:
    index = root / UNITS_INDEX
    existing: list[str] = []
    if index.is_file():
        try:
            existing = [str(u) for u in json.loads(index.read_text(encoding="utf-8"))["units"]]
        except (ValueError, KeyError, TypeError):
            existing = []
    merged = list(dict.fromkeys([*existing, *unit_ids]))
    index.write_text(
        json.dumps(
            {"schema": UNITS_SCHEMA, "units": merged, "updated_at": datetime.now(UTC).isoformat()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def list_units(root: str | Path) -> list[str]:
    """Unit ids under ``root``: the ``units.json`` index order, else every dir with a unit.json."""
    r = Path(root)
    index = r / UNITS_INDEX
    if index.is_file():
        try:
            ids = [str(u) for u in json.loads(index.read_text(encoding="utf-8"))["units"]]
            return [u for u in ids if (r / u / UNIT_JSON).is_file()]
        except (ValueError, KeyError, TypeError):
            pass
    if not r.is_dir():
        return []
    return sorted(p.name for p in r.iterdir() if p.is_dir() and (p / UNIT_JSON).is_file())


def plan_units(
    frames: Sequence[Frame],
    root: str | Path,
    cfg: Settings | None = None,
    *,
    unit_ids: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None = None,
) -> list[str]:
    """Write ``pw.in`` + ``unit.json`` (frame, resolved parameters, ``cfg.dft`` snapshot) per unit.

    Unit ids default to frame ids; ``overrides`` is one mapping for all units or one per unit.
    Re-planning an existing unit with identical parameters leaves its outputs and markers alone;
    changed parameters clear stale ``pw.out``/markers so the unit reruns.
    """
    cfg = cfg if cfg is not None else load_config([], [])
    r = Path(root)
    r.mkdir(parents=True, exist_ok=True)
    ids = list(unit_ids) if unit_ids is not None else [f.frame_id for f in frames]
    if len(ids) != len(frames):
        raise ValueError(f"{len(ids)} unit ids for {len(frames)} frames")
    if len(set(ids)) != len(ids):
        dupes = sorted({u for u in ids if ids.count(u) > 1})
        raise ValueError(f"duplicate unit ids {dupes[:5]} (identical geometries?)")
    if any(u != unit_slug(u) for u in ids):
        bad = [u for u in ids if u != unit_slug(u)]
        raise ValueError(f"unit ids must be filesystem/marker safe, got {bad[:5]}")
    per_unit: list[Mapping[str, Any] | None]
    if overrides is None or isinstance(overrides, Mapping):
        per_unit = [overrides] * len(frames)
    else:
        per_unit = list(overrides)
        if len(per_unit) != len(frames):
            raise ValueError("one overrides mapping per frame expected")
    dft_snapshot = cfg.dft.model_dump(mode="json")
    for frame, uid, ov in zip(frames, ids, per_unit, strict=True):
        d = unit_dir(frame, r, uid)
        d.mkdir(parents=True, exist_ok=True)
        params = pw_parameters(frame, cfg, overrides=ov, unit_id=uid)
        record = {
            "schema": UNIT_SCHEMA,
            "unit_id": uid,
            "frame": frame.model_dump(mode="json"),
            "parameters": params["values"],
            "overrides": dict(ov or {}),
            "dft": dft_snapshot,
            "created_at": datetime.now(UTC).isoformat(),
        }
        unit_json = d / UNIT_JSON
        if unit_json.is_file():
            try:
                old = json.loads(unit_json.read_text(encoding="utf-8"))
            except ValueError:
                old = {}
            same = (
                old.get("frame") == record["frame"]
                and old.get("parameters") == record["parameters"]
            )
            if same:
                continue
            for stale in (PW_OUTPUT, "pw.err", DONE_MARKER, FAILED_MARKER):
                (d / stale).unlink(missing_ok=True)
        (d / PW_INPUT).write_text(
            render_pw_input(frame, cfg, overrides=ov, unit_id=uid), encoding="utf-8"
        )
        unit_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _write_index(r, ids)
    return ids


def pending_units(root: str | Path, unit_ids: Sequence[str] | None = None) -> list[str]:
    r = Path(root)
    ids = list(unit_ids) if unit_ids is not None else list_units(r)
    return [u for u in ids if unit_status(r / u) != "done"]


def job_name(root: str | Path) -> str:
    resolved = Path(root).resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"qe-{unit_slug(resolved.name)}-{digest}"


def job_spec(
    root: str | Path,
    unit_ids: Sequence[str],
    cfg: Settings,
    *,
    name: str | None = None,
    template: str = "qe_array",
    resources: Mapping[str, Any] | None = None,
    units_root: str | None = None,
    env: Mapping[str, str] | None = None,
) -> JobSpec:
    """The ``JobSpec`` that runs :data:`UNIT_SCRIPT` over ``unit_ids`` (``units_root`` is the
    directory the *executing* machine sees: the local root or its cluster mirror)."""
    r = Path(root)
    job_env: dict[str, str] = {"B20_UNITS_ROOT": units_root or str(r.resolve())}
    if cfg.cluster.qe_cmd:
        job_env["QE_CMD"] = cfg.cluster.qe_cmd
    job_env.update(env or {})
    return JobSpec(
        name=name or job_name(r),
        script=UNIT_SCRIPT,
        units=list(unit_ids),
        resources={
            "template": template,
            **dict(cfg.cluster.resources.get(template, {})),  # site defaults (cluster overlay)
            **dict(resources or {}),  # explicit per-call overrides win
        },
        env=job_env,
    )


# --- parsing -----------------------------------------------------------------------------------

_RE_VERSION = re.compile(r"Program PWSCF v\.(\S+)")
_RE_FINAL_ENERGY = re.compile(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", re.M)
_RE_FERMI = re.compile(r"the Fermi energy is\s+(-?\d+\.\d+)\s+ev")
_RE_TOTAL_MAG = re.compile(r"total magnetization\s+=\s+(-?\d+\.\d+)\s+Bohr mag/cell")
_RE_ABS_MAG = re.compile(r"absolute magnetization\s+=\s+(-?\d+\.\d+)\s+Bohr mag/cell")
_RE_CONVERGED = re.compile(r"convergence has been achieved in\s+(\d+)\s+iterations")
_RE_NOT_CONVERGED = re.compile(r"convergence NOT achieved after\s+(\d+)\s+iterations")
_RE_ITERATION = re.compile(r"iteration #\s*(\d+)")
_RE_FORCES_HEADER = re.compile(r"Forces acting on atoms \(cartesian axes, Ry/au\):")
_RE_FORCE_LINE = re.compile(
    r"^\s*atom\s+(\d+)\s+type\s+(\d+)\s+force =\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_RE_STRESS_HEADER = re.compile(r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?\d+\.\d+)")
_RE_SITE_HEADER = re.compile(r"Magnetic moment per site")
_RE_SITE_NEW = re.compile(
    r"^\s*atom\s+(\d+)\s+\(R=\s*([\d.]+)\)\s+charge=\s*(-?\d+\.\d+)\s+magn=\s*(-?\d+\.\d+)"
)
_RE_SITE_OLD = re.compile(r"^\s*atom:\s*(\d+)\s+charge:\s*(-?\d+\.\d+)\s+magn:\s*(-?\d+\.\d+)")
_RE_WALL = re.compile(r"PWSCF\s*:\s*(.+?)\s*CPU\s+(.+?)\s*WALL")
_RE_DURATION = re.compile(r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+(?:\.\d+)?)s)?")
_NUM = re.compile(r"-?\d+\.\d+")


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def parse_duration(text: str) -> float | None:
    """QE clock strings: ``16.12s``, ``2m30.12s``, ``1h 5m`` -> seconds."""
    m = _RE_DURATION.fullmatch(text.strip())
    if not m or not any(m.groups()):
        return None
    hours, minutes, seconds = m.groups()
    return float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)


def parse_forces_ry_bohr(text: str) -> list[list[float]] | None:
    """Rows of the LAST ``Forces acting on atoms`` block (Ry/Bohr), ordered by atom index."""
    headers = list(_RE_FORCES_HEADER.finditer(text))
    if not headers:
        return None
    rows: dict[int, list[float]] = {}
    for line in text[headers[-1].end() :].splitlines():
        if not line.strip():
            continue
        m = _RE_FORCE_LINE.match(line)
        if not m:
            break  # the decomposition blocks / "Total force" line end the block
        rows[int(m.group(1))] = [float(m.group(3)), float(m.group(4)), float(m.group(5))]
    if not rows:
        return None
    return [rows[i] for i in sorted(rows)]


def parse_stress_ry_bohr3(text: str) -> tuple[list[list[float]], float] | None:
    """``(3x3 σ_QE in Ry/bohr³, pressure in kbar)`` from the LAST ``total   stress`` block."""
    headers = list(_RE_STRESS_HEADER.finditer(text))
    if not headers:
        return None
    pressure = float(headers[-1].group(1))
    tail = text[headers[-1].end() :].splitlines()
    rows: list[list[float]] = []
    for line in tail[1:4] if tail and not tail[0].strip() else tail[:3]:
        nums = _NUM.findall(line)
        if len(nums) < 3:
            return None
        rows.append([float(x) for x in nums[:3]])
    if len(rows) != 3:
        return None
    return rows, pressure


def parse_site_moments(text: str) -> list[float] | None:
    """``magn=`` values of the LAST ``Magnetic moment per site`` block (new and old format)."""
    headers = list(_RE_SITE_HEADER.finditer(text))
    if not headers:
        return None
    moments: dict[int, float] = {}
    for line in text[headers[-1].end() :].splitlines()[1:]:
        m = _RE_SITE_NEW.match(line)
        if m:
            moments[int(m.group(1))] = float(m.group(4))
            continue
        m = _RE_SITE_OLD.match(line)
        if m:
            moments[int(m.group(1))] = float(m.group(3))
            continue
        if moments:
            break
        if line.strip():
            break
    if not moments:
        return None
    return [moments[i] for i in sorted(moments)]


def voigt6_ase_from_qe(sigma_qe: Sequence[Sequence[float]]) -> list[float]:
    """QE ``total stress`` (Ry/bohr³, compressive positive) -> ASE Voigt-6 in eV/Å³ (σ = -σ_QE)."""
    s = -np.asarray(sigma_qe, dtype=np.float64) * RY_BOHR3_TO_EV_A3
    return [
        float(s[0, 0]),
        float(s[1, 1]),
        float(s[2, 2]),
        float(s[1, 2]),
        float(s[0, 2]),
        float(s[0, 1]),
    ]


def parse_pw_text(text: str) -> dict[str, Any]:
    """Raw quantities of a ``pw.out`` text in QE units plus converted ones (module docstring)."""
    converged_m = _RE_CONVERGED.findall(text)
    not_converged_m = _RE_NOT_CONVERGED.findall(text)
    iterations = [int(x) for x in _RE_ITERATION.findall(text)]
    if converged_m:
        scf_steps = int(converged_m[-1])
    elif not_converged_m:
        scf_steps = int(not_converged_m[-1])
    else:
        scf_steps = max(iterations, default=0)
    energy_ry = _last_float(_RE_FINAL_ENERGY, text)
    forces_ry = parse_forces_ry_bohr(text)
    stress = parse_stress_ry_bohr3(text)
    wall = None
    wall_m = _RE_WALL.findall(text)
    if wall_m:
        wall = parse_duration(wall_m[-1][1])
    version_m = _RE_VERSION.search(text)
    return {
        "converged": bool(converged_m),
        "scf_steps": scf_steps,
        "energy_ry": energy_ry,
        "energy_eV": None if energy_ry is None else energy_ry * RY_TO_EV,
        "forces_ry_bohr": forces_ry,
        "forces_eV_A": None
        if forces_ry is None
        else (np.asarray(forces_ry) * RY_BOHR_TO_EV_A).tolist(),
        "stress_qe_ry_bohr3": None if stress is None else stress[0],
        "stress_voigt6_eV_A3": None if stress is None else voigt6_ase_from_qe(stress[0]),
        "pressure_kbar": None if stress is None else stress[1],
        "fermi_eV": _last_float(_RE_FERMI, text),
        "total_magnetization": _last_float(_RE_TOTAL_MAG, text),
        "abs_magnetization": _last_float(_RE_ABS_MAG, text),
        "magmoms": parse_site_moments(text),
        "wall_seconds": wall,
        "pw_version": version_m.group(1) if version_m else None,
        "job_done": "JOB DONE" in text,
    }


def _parameters_for(
    path: Path, dft: DFTConfig | Mapping[str, Any] | None, frame: Frame
) -> dict[str, Any]:
    if isinstance(dft, DFTConfig):
        symbols = species_order(frame.numbers)
        return {
            "ecut_ry": dft.ecut_ry,
            "ecut_rho": dft.ecut_rho,
            "k_spacing_inv_A": dft.k_spacing_inv_A,
            "kmesh": list(kmesh(frame.cell, dft.k_spacing_inv_A)),
            "nspin": nspin_for(
                frame.compound, Settings.model_validate({"dft": dft.model_dump()}), frame.numbers
            ),
            "smearing": dft.smearing,
            "degauss_ry": dft.degauss_ry,
            "pseudo_md5s": {s: dft.pseudo_md5s[s] for s in symbols if s in dft.pseudo_md5s},
            "m_ref_muB": dft.m_ref_muB.get(frame.compound),
            "branch_tol_muB": dft.branch_tol_muB,
        }
    if isinstance(dft, Mapping):
        return dict(dft)
    unit_json = path.parent / UNIT_JSON
    if unit_json.is_file():
        try:
            return dict(json.loads(unit_json.read_text(encoding="utf-8"))["parameters"])
        except (ValueError, KeyError, TypeError):
            pass
    return _parameters_for(path, DFTConfig(), frame)


def branch_ok(
    magmoms: Sequence[float] | None, numbers: Sequence[int], m_ref: float | None, tol: float
) -> bool | None:
    """Magnetic-branch check: mean |m| over the transition-metal atoms within ``tol`` of ``m_ref``.
    ``None`` when there is no reference moment or no site moments (the data-tier filter decides)."""
    if magmoms is None or m_ref is None:
        return None
    tm = [abs(m) for m, z in zip(magmoms, numbers, strict=True) if int(z) in TM_NUMBERS]
    if not tm:
        return None
    return abs(float(np.mean(tm)) - float(m_ref)) <= float(tol)


def parse_pw_output(
    path: str | Path,
    frame: Frame,
    *,
    dft: DFTConfig | Mapping[str, Any] | None = None,
    unit_id: str | None = None,
) -> DFTFrame:
    """Parse ``pw.out`` into a ``DFTFrame`` for ``frame`` (labels in ASE units, module docstring).

    ``dft`` supplies the run parameters (a ``DFTConfig``, the ``parameters`` mapping of a unit.json,
    or ``None`` to read ``unit.json`` next to ``path``). A missing or unconverged output gives
    ``converged=False`` rather than raising.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    parsed = parse_pw_text(text)
    params = _parameters_for(p, dft, frame)
    nat = len(frame.numbers)
    warnings: list[str] = []
    if not p.is_file():
        warnings.append("pw.out missing")
    forces = parsed["forces_eV_A"]
    if forces is not None and len(forces) != nat:
        warnings.append(f"force block has {len(forces)} rows for {nat} atoms")
        forces = None
    magmoms = parsed["magmoms"]
    if magmoms is not None and len(magmoms) != nat:
        warnings.append(f"site-moment block has {len(magmoms)} rows for {nat} atoms")
        magmoms = None
    if text and not parsed["converged"] and not parsed["job_done"]:
        warnings.append("output truncated (no JOB DONE)")

    base: dict[str, Any] = {name: getattr(frame, name) for name in Frame.model_fields}
    info = dict(frame.info)
    info.update(
        pw_version=parsed["pw_version"],
        pressure_kbar=parsed["pressure_kbar"],
        total_energy_ry=parsed["energy_ry"],
        ecut_rho=params.get("ecut_rho"),
        kmesh=list(params.get("kmesh") or []),
        pseudo_family=params.get("pseudo_family"),
    )
    if warnings:
        info["parse_warnings"] = "; ".join(warnings)
    info = {k: v for k, v in info.items() if v is not None}
    base.update(
        energy=parsed["energy_eV"],
        forces=forces,
        stress=parsed["stress_voigt6_eV_A3"],
        magmoms=magmoms,
        total_magnetization=parsed["total_magnetization"],
        label_source="qe",
        energy_scale="qe",
        info=info,
    )
    return DFTFrame(
        **base,
        code="qe",
        functional="PBE",
        pseudo_md5s=dict(params.get("pseudo_md5s") or {}),
        ecut_ry=float(params.get("ecut_ry", DFTConfig().ecut_ry)),
        k_spacing=float(params.get("k_spacing_inv_A", DFTConfig().k_spacing_inv_A)),
        nspin=int(params.get("nspin", 1)),
        smearing=str(params.get("smearing", DFTConfig().smearing)),
        degauss_ry=float(params.get("degauss_ry", DFTConfig().degauss_ry)),
        converged=bool(parsed["converged"]),
        scf_steps=int(parsed["scf_steps"]),
        abs_magnetization=parsed["abs_magnetization"],
        fermi_eV=parsed["fermi_eV"],
        branch_ok=branch_ok(
            magmoms,
            frame.numbers,
            params.get("m_ref_muB"),
            float(params.get("branch_tol_muB", DFTConfig().branch_tol_muB)),
        ),
        wall_seconds=float(parsed["wall_seconds"] or 0.0),
        unit_id=unit_id or p.parent.name,
    )


def collect(root: str | Path) -> tuple[list[DFTFrame], dict[str, int]]:
    """Parse every planned unit under ``root``.

    Returns all parsed frames (converged or not; check ``DFTFrame.converged``) and the counts
    ``planned`` (unit dirs), ``done`` (converged output), ``unconverged`` (output without SCF
    convergence), ``failed`` (``.failed`` marker and no output) and ``missing`` (no output, no
    marker: not run yet).
    """
    r = Path(root)
    ids = list_units(r)
    counts = {"planned": len(ids), "done": 0, "failed": 0, "unconverged": 0, "missing": 0}
    frames: list[DFTFrame] = []
    for uid in ids:
        d = r / uid
        out = d / PW_OUTPUT
        if not out.is_file():
            counts["failed" if unit_status(d) == "failed" else "missing"] += 1
            continue
        unit = read_unit(d)
        frame = Frame.model_validate(unit["frame"])
        df = parse_pw_output(out, frame, dft=unit.get("parameters"), unit_id=uid)
        frames.append(df)
        counts["done" if df.converged else "unconverged"] += 1
    return frames, counts


def to_plain_frame(df: DFTFrame) -> Frame:
    """``DFTFrame`` -> ``Frame`` with the DFT-only fields folded into ``info`` (``dft_*`` keys), so
    extxyz round trips (``io.write_frames``) keep them."""
    base = {name: getattr(df, name) for name in Frame.model_fields}
    extra = {
        "dft_code": df.code,
        "dft_functional": df.functional,
        "dft_pseudo_md5s": dict(df.pseudo_md5s),
        "dft_ecut_ry": df.ecut_ry,
        "dft_k_spacing": df.k_spacing,
        "dft_nspin": df.nspin,
        "dft_smearing": df.smearing,
        "dft_degauss_ry": df.degauss_ry,
        "dft_converged": df.converged,
        "dft_scf_steps": df.scf_steps,
        "dft_abs_magnetization": df.abs_magnetization,
        "dft_fermi_eV": df.fermi_eV,
        "dft_branch_ok": df.branch_ok,
        "dft_wall_seconds": df.wall_seconds,
        "dft_unit_id": df.unit_id,
    }
    base["info"] = {**df.info, **{k: v for k, v in extra.items() if v is not None}}
    return Frame(**base)


__all__ = [
    "BOHR_TO_A",
    "DONE_MARKER",
    "FAILED_MARKER",
    "KBAR_TO_EV_A3",
    "PW_INPUT",
    "PW_OUTPUT",
    "RY_BOHR3_TO_EV_A3",
    "RY_BOHR_TO_EV_A",
    "RY_TO_EV",
    "UNIT_JSON",
    "UNIT_SCRIPT",
    "branch_ok",
    "collect",
    "job_name",
    "job_spec",
    "kmesh",
    "list_units",
    "load_sssp",
    "nspin_for",
    "parse_duration",
    "parse_forces_ry_bohr",
    "parse_pw_output",
    "parse_pw_text",
    "parse_site_moments",
    "parse_stress_ry_bohr3",
    "pending_units",
    "plan_units",
    "pw_parameters",
    "read_unit",
    "render_pw_input",
    "resolve_pseudo_dir",
    "species_order",
    "sssp_cutoffs",
    "to_plain_frame",
    "unit_dir",
    "unit_status",
    "voigt6_ase_from_qe",
]
