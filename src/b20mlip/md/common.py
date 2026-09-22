"""Helpers shared by the MD tier: structures and supercells, model provenance, ``numbers.json``.

Conventions (binding for the sampling / agent / bench tiers that reuse them):

* **Supercell.** ``make_supercell(atoms, natoms)`` repeats the input cell isotropically
  ``reps x reps x reps`` with the smallest ``reps`` such that ``len(atoms) * reps**3 >= natoms``
  (an 8-atom B20 cell and ``natoms=64`` -> 2x2x2; 512 -> 4x4x4). The cubic lattice parameter
  reported everywhere is ``a = (V / reps**3)**(1/3)``, i.e. the edge of the *input* cell — the
  conventional B20 lattice parameter when the input is the conventional cell.
* **Structure.** ``load_structure(cfg, compound, structure)`` is the relaxed MPtrj parent of the
  compound (``data/raw/mptrj/b20_mptrj.extxyz``, lowest-energy frame; the same choice as the dft
  tier's ``dft.structures.reference_frame``) or the first/lowest-energy frame of ``--structure``.
* **Model provenance.** ``model_provenance(path)`` reads the train run's ``checkpoint.json`` when
  the model lives in a ``runs/train/<id>/models/`` directory (``e0_source``, ``energy_scale``,
  ``variant``, ``heads``), recognises the cached foundation files (``mace-mpa-0-medium.model`` ->
  bracket ``B0``) and otherwise labels the model by its file stem. ``default_label`` maps the
  variant to the SPEC.md bracket (``B1`` naive, ``B2`` replay, ``B3`` scratch, ``B4`` bootstrap).
* **numbers.json.** ``write_numbers(ctx, numbers)`` writes ``ctx.out_dir / numbers.json`` in the
  report tier's format (flat dotted keys plus ``<key>@meta``) and registers it as an output.
  ``number_meta(...)`` builds the metadata gate A3 needs: ``reference`` (``code="mace"``,
  ``functional=REFERENCE_FUNCTIONAL`` — the functional of the model's training labels, PBE for
  MPtrj/sAlex and for the in-house QE frames; gate A3 rejects ``null`` for ``code != experiment``),
  ``e0_source``, ``head``, ``n``, ``seed``, ``ci95`` (or ``null`` with ``ci95_reason``), plus
  ``model_label``, ``energy_scale``, ``engine``, ``ensemble``, ``T``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from numpy.typing import ArrayLike

from b20mlip.config import Settings
from b20mlip.io import contract_head, frame_to_atoms, read_frames
from b20mlip.models import CheckpointInfo, MDResult, StageResult, Status
from b20mlip.provenance import RunContext, sha256_file

NUMBERS_FILE = "numbers.json"
REFERENCE_FUNCTIONAL = "PBE"
# Room-temperature experimental B20 lattice parameters (Å) for the a(300 K) comparison of SPEC.md
# section 6. MnSi is the value SPEC.md quotes; the others are literature room-temperature values
# (FeSi 4.489, CoSi 4.444, FeGe 4.700) and are published with ``a_exp_source`` in their meta so a
# reader can check them before they enter a claim.
EXPERIMENTAL_A_A: dict[str, float] = {"MnSi": 4.558, "FeSi": 4.489, "CoSi": 4.444, "FeGe": 4.700}
EXPERIMENTAL_A_SOURCE = (
    "room-temperature literature lattice parameters; MnSi 4.558 A per SPEC.md section 6"
)
BRACKET_BY_VARIANT: dict[str, str] = {
    "naive": "B1",
    "replay": "B2",
    "scratch": "B3",
    "bootstrap": "B4",
}
FOUNDATION_LABELS: dict[str, str] = {
    "mace-mpa-0-medium.model": "B0",
    "2023-12-03-mace-128-L1_epoch-199.model": "B0p",
}
LAMMPS_SUFFIX = "-lammps.pt"
_KEY_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


# --- structures and supercells --------------------------------------------------------------------


def supercell_reps(n_cell: int, natoms: int) -> int:
    """Smallest ``reps`` with ``n_cell * reps**3 >= natoms`` (at least 1)."""
    if n_cell <= 0:
        raise ValueError("the input cell has no atoms")
    reps = max(1, int(math.ceil(round((max(1, natoms) / n_cell) ** (1.0 / 3.0), 9))))
    while n_cell * reps**3 < natoms:
        reps += 1
    return reps


def make_supercell(atoms: Atoms, natoms: int | None) -> tuple[Atoms, int]:
    """``(supercell, reps)``: the isotropic repetition reaching at least ``natoms`` atoms."""
    reps = 1 if natoms is None else supercell_reps(len(atoms), int(natoms))
    cell = atoms.repeat((reps, reps, reps)) if reps > 1 else atoms.copy()
    cell.info = {k: v for k, v in atoms.info.items()}
    return cell, reps


def lattice_parameter(volume: float, reps: int) -> float:
    """Cubic lattice parameter of the input cell from the supercell volume."""
    return float(volume / reps**3) ** (1.0 / 3.0)


def elements_of_atoms(atoms: Atoms | Iterable[int]) -> list[str]:
    """Element symbols by increasing atomic number (the LAMMPS type order used everywhere)."""
    numbers = atoms.numbers if isinstance(atoms, Atoms) else list(atoms)
    return [chemical_symbols[z] for z in sorted({int(z) for z in numbers})]


def compound_of(atoms: Atoms) -> str:
    return atoms.get_chemical_formula("metal", empirical=True)


def _default_structure_path(cfg: Settings) -> Path:
    data = Path(cfg.paths.data_dir)
    preferred = data / "frames" / "mptrj_b20.extxyz"  # the data tier's spglib-198 extract
    return preferred if preferred.is_file() else data / "raw" / "mptrj" / "b20_mptrj.extxyz"


def load_structure(cfg: Settings, compound: str, structure: str | Path | None = None) -> Atoms:
    """The relaxed reference cell of ``compound`` (see the module docstring) as bare ``Atoms``."""
    try:
        from b20mlip.dft.structures import reference_frame  # row 6: allowed, but optional

        frame = reference_frame(cfg, compound, structure)
    except ImportError:  # pragma: no cover - dft tier absent
        frame = _reference_frame_fallback(cfg, compound, structure)
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    atoms.info = {"compound": compound, "frame_id": frame.frame_id, "parent_id": frame.parent_id}
    return atoms


def _reference_frame_fallback(
    cfg: Settings, compound: str, structure: str | Path | None
) -> Any:  # pragma: no cover - exercised only without the dft tier
    path = Path(structure) if structure is not None else _default_structure_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"no structure file {path}; pass --structure")
    frames = [f for f in read_frames(path) if f.compound == compound] or read_frames(path)
    if not frames:
        raise ValueError(f"no frames in {path}")
    if any(f.energy is not None for f in frames):
        return min(frames, key=lambda f: f.energy if f.energy is not None else math.inf)
    return frames[-1]


def elements_of_compound(compound: str) -> list[str]:
    out: list[str] = []
    for symbol in _ELEMENT_RE.findall(compound):
        if symbol not in atomic_numbers:
            raise ValueError(f"unknown element {symbol!r} in compound {compound!r}")
        if symbol not in out:
            out.append(symbol)
    return out


# --- model provenance -----------------------------------------------------------------------------


def lammps_path_for(model_path: str | Path) -> Path:
    """``<model>-lammps.pt`` (what ``b20mlip export`` writes); a ``-lammps.pt`` maps to itself."""
    p = Path(model_path)
    return p if p.name.endswith(LAMMPS_SUFFIX) else p.with_name(p.name + LAMMPS_SUFFIX)


def base_model_for(path: str | Path) -> Path:
    """The ``.model`` a ``-lammps.pt`` was exported from (the file itself otherwise)."""
    p = Path(path)
    return p.with_name(p.name[: -len(LAMMPS_SUFFIX)]) if p.name.endswith(LAMMPS_SUFFIX) else p


def sanitize_key_segment(text: str) -> str:
    """A ``numbers.json`` key segment: ``[A-Za-z0-9_.-]`` only, never empty."""
    cleaned = _KEY_SEGMENT_RE.sub("-", str(text)).strip("-.") or "model"
    return cleaned


def default_label(provenance: Mapping[str, Any]) -> str:
    """SPEC.md bracket (``B0``..``B4``) when the variant is known, else the sanitized stem."""
    variant = provenance.get("variant")
    if variant == "zero_shot" and provenance.get("bracket"):
        return str(provenance["bracket"])
    if isinstance(variant, str) and variant in BRACKET_BY_VARIANT:
        return BRACKET_BY_VARIANT[variant]
    return sanitize_key_segment(Path(str(provenance.get("model_path", "model"))).stem)


def model_provenance(model_path: str | Path) -> dict[str, Any]:
    """What is known about a model file: hashes, e0 source, energy scale, heads, label."""
    base = base_model_for(model_path)
    lammps = lammps_path_for(base)
    info: dict[str, Any] = {
        "model_path": str(base),
        "sha256": sha256_file(base) if base.is_file() else None,
        "lammps_path": str(lammps) if lammps.is_file() else None,
        "lammps_sha256": sha256_file(lammps) if lammps.is_file() else None,
        "e0_source": "unknown",
        "energy_scale": "none",
        "variant": None,
        "heads": ["Default"],
        "bracket": None,
        "train_run_id": None,
        "checkpoint_json": None,
    }
    for cand in (base.parent / "checkpoint.json", base.parent.parent / "checkpoint.json"):
        if not cand.is_file():
            continue
        try:
            ckpt = CheckpointInfo.model_validate_json(cand.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if ckpt.sha256 != info["sha256"] and Path(ckpt.model_path).name != base.name:
            continue
        info.update(
            e0_source=ckpt.e0_source,
            energy_scale=ckpt.energy_scale,
            variant=ckpt.variant,
            heads=list(ckpt.heads),
            train_run_id=ckpt.train_run_id,
            checkpoint_json=str(cand),
        )
        if ckpt.variant == "zero_shot":
            info["bracket"] = FOUNDATION_LABELS.get(Path(ckpt.model_path).name)
        break
    else:
        if base.name in FOUNDATION_LABELS:
            info.update(
                e0_source="foundation",
                energy_scale="mp",
                variant="zero_shot",
                bracket=FOUNDATION_LABELS[base.name],
            )
    info["label"] = default_label(info)
    return info


# --- statistics -----------------------------------------------------------------------------------


def linear_fit(x: ArrayLike, y: ArrayLike) -> tuple[float, float, float | None]:
    """``(slope, intercept, slope standard error)``; the error is ``None`` below 3 points."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 2 or np.ptp(xs) == 0:
        raise ValueError("a linear fit needs at least two distinct abscissae")
    slope, intercept = np.polyfit(xs, ys, 1)
    if xs.size < 3:
        return float(slope), float(intercept), None
    resid = ys - (slope * xs + intercept)
    s2 = float(np.sum(resid**2) / (xs.size - 2))
    se = math.sqrt(s2 / float(np.sum((xs - xs.mean()) ** 2)))
    return float(slope), float(intercept), se


def block_ci95(values: ArrayLike, nblocks: int = 5) -> tuple[float, float] | None:
    """95 % interval of the mean from ``nblocks`` block averages (``None`` below 2 per block)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 * nblocks:
        return None
    blocks = np.array_split(arr, nblocks)
    means = np.array([b.mean() for b in blocks])
    se = float(means.std(ddof=1) / math.sqrt(nblocks))
    mean = float(arr.mean())
    return (mean - 1.96 * se, mean + 1.96 * se)


def fmt_T(T: float) -> str:
    """``300`` for 300.0, ``300.5`` for 300.5 (key segments must stay short and exact)."""
    return str(int(round(T))) if abs(T - round(T)) < 1e-9 else repr(float(T))


# --- numbers.json and results ---------------------------------------------------------------------


def number_meta(
    *,
    provenance: Mapping[str, Any],
    head: str,
    n: int,
    seed: int | None,
    engine: str,
    ensemble: str | None,
    T: float | None,
    ci95: tuple[float, float] | None,
    ci95_reason: str | None = None,
    label: str | None = None,
    reference_code: str = "mace",
    **extra: Any,
) -> dict[str, Any]:
    """Gate-A3 metadata for one published MD number (see the module docstring)."""
    functional = REFERENCE_FUNCTIONAL if reference_code != "experiment" else None
    meta: dict[str, Any] = {
        "reference": {
            "code": reference_code,
            "functional": functional,
            "pseudos": None,
            "e0_source": provenance.get("e0_source", "unknown"),
            "cross_functional": False,
        },
        "e0_source": provenance.get("e0_source", "unknown"),
        "head": contract_head(head),
        "n": int(n),
        "seed": int(seed) if seed is not None else 0,
        "ci95": [float(ci95[0]), float(ci95[1])] if ci95 is not None else None,
        "model_label": label or provenance.get("label", "model"),
        "energy_scale": provenance.get("energy_scale", "none"),
        "model_sha256": provenance.get("sha256"),
        "engine": engine,
        "ensemble": ensemble,
        "T": T,
    }
    if ci95 is None:
        meta["ci95_reason"] = ci95_reason or "single trajectory; too few samples for block errors"
    meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


def write_numbers(ctx: RunContext, numbers: Mapping[str, Any]) -> Path:
    """Write ``numbers.json`` into the run directory and register it (report tier convention)."""
    path = ctx.out_dir / NUMBERS_FILE
    path.write_text(json.dumps(dict(numbers), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.add_output(path, "json")
    return path


def write_result(ctx: RunContext, result: MDResult, name: str) -> Path:
    path = ctx.out_dir / name
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ctx.add_output(path, "json")
    return path


def stage_result(
    ctx: RunContext, status: Status, summary: Mapping[str, Any] | None = None
) -> StageResult:
    """A ``StageResult`` for the current context (booleans -> 0/1, containers -> JSON)."""
    clean: dict[str, float | int | str] = {}
    for key, value in (summary or {}).items():
        if isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, (int, float, str)):
            clean[key] = value
        elif value is not None:
            clean[key] = json.dumps(value, sort_keys=True, default=str)
    return StageResult(
        stage=ctx.stage,
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status=status,
        outputs=list(ctx.outputs),
        summary=clean,
    )


def relpath(path: str | Path, ctx: RunContext) -> str:
    """A path for JSON payloads: relative to the run directory when inside it."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ctx.out_dir.resolve()))
    except ValueError:
        return str(p)


__all__ = [
    "BRACKET_BY_VARIANT",
    "EXPERIMENTAL_A_A",
    "EXPERIMENTAL_A_SOURCE",
    "FOUNDATION_LABELS",
    "LAMMPS_SUFFIX",
    "NUMBERS_FILE",
    "REFERENCE_FUNCTIONAL",
    "base_model_for",
    "block_ci95",
    "compound_of",
    "default_label",
    "elements_of_atoms",
    "elements_of_compound",
    "fmt_T",
    "lammps_path_for",
    "lattice_parameter",
    "linear_fit",
    "load_structure",
    "make_supercell",
    "model_provenance",
    "number_meta",
    "relpath",
    "sanitize_key_segment",
    "stage_result",
    "supercell_reps",
    "write_numbers",
    "write_result",
]


# --- published numbers (shared by both engines) ---------------------------------------------------


def md_numbers(
    results: Sequence[MDResult],
    stats: Mapping[float, Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    label: str,
    seed: int | None,
    fit: Mapping[str, Any] | None = None,
    a_exp_A: float | None = None,
) -> dict[str, Any]:
    """The ``numbers.json`` payload of an MD stage (both engines; the module docstring lists the
    keys). ``stats[T]`` carries ``n_production``, ``a_ci95``, ``drift_ci95`` per temperature;
    ``fit`` is :func:`b20mlip.md.ase_md.thermal_expansion_fit` when >= 3 NPT temperatures ran.
    """
    numbers: dict[str, Any] = {}
    if not results:
        return numbers
    engine = results[0].engine
    compound = sanitize_key_segment(results[0].compound)
    label = sanitize_key_segment(label)
    head = results[0].head
    for res in results:
        st = stats.get(res.temperature_K, {})
        n = max(1, int(st.get("n_production", 1)))
        base = f"md.{engine}.{compound}.{label}.{res.ensemble}.{fmt_T(res.temperature_K)}"
        common: dict[str, Any] = dict(
            provenance=provenance, head=head, seed=seed, engine=engine, ensemble=res.ensemble,
            T=res.temperature_K, label=label, compound=res.compound, natoms=res.natoms,
            timestep_fs=res.timestep_fs, steps=res.steps,
        )  # fmt: skip
        if res.ensemble == "nve" and res.drift_meV_atom_ps is not None:
            meta = number_meta(
                n=n, ci95=st.get("drift_ci95"),
                ci95_reason="fewer than 3 thermo samples: no slope error", unit="meV/atom/ps",
                **common,
            )  # fmt: skip
            numbers[f"{base}.drift_meV_atom_ps"] = res.drift_meV_atom_ps
            numbers[f"{base}.drift_meV_atom_ps@meta"] = meta
            numbers[f"md.{engine}.{compound}.{label}.drift_meV_atom_ps"] = res.drift_meV_atom_ps
            numbers[f"md.{engine}.{compound}.{label}.drift_meV_atom_ps@meta"] = meta
        if res.ensemble == "npt" and res.a_mean_A is not None:
            meta = number_meta(
                n=n, ci95=st.get("a_ci95"),
                ci95_reason="fewer than 10 production samples: no block error", unit="A",
                **common,
            )  # fmt: skip
            numbers[f"{base}.a_mean_A"] = res.a_mean_A
            numbers[f"{base}.a_mean_A@meta"] = meta
            numbers[f"{base}.a_std_A"] = res.a_std_A
            numbers[f"{base}.a_std_A@meta"] = {**meta, "ci95": None, "ci95_reason": "a spread"}
            if abs(res.temperature_K - 300.0) < 1e-9:
                exp_meta = number_meta(
                    n=n, ci95=st.get("a_ci95"),
                    ci95_reason="fewer than 10 production samples: no block error", unit="A",
                    reference_code="experiment", a_exp_source=EXPERIMENTAL_A_SOURCE, **common,
                )  # fmt: skip
                numbers[f"md.{engine}.{compound}.{label}.a_300K_A"] = res.a_mean_A
                numbers[f"md.{engine}.{compound}.{label}.a_300K_A@meta"] = exp_meta
                exp = a_exp_A if a_exp_A is not None else EXPERIMENTAL_A_A.get(res.compound)
                if exp is not None:
                    numbers[f"md.{engine}.{compound}.{label}.a_exp_A"] = float(exp)
                    numbers[f"md.{engine}.{compound}.{label}.a_exp_A@meta"] = {
                        **exp_meta, "ci95": None, "ci95_reason": "tabulated experimental value",
                    }  # fmt: skip
                    numbers[f"md.{engine}.{compound}.{label}.a_dev_pct"] = (
                        100.0 * (res.a_mean_A - float(exp)) / float(exp)
                    )
                    numbers[f"md.{engine}.{compound}.{label}.a_dev_pct@meta"] = {
                        **exp_meta, "unit": "%", "ci95": None,
                        "ci95_reason": "derived from a_300K_A and a_exp_A",
                    }  # fmt: skip
    if fit is not None and fit.get("alpha_per_K") is not None:
        temps = [float(t) for t in fit["temperatures_K"]]
        alpha_meta = number_meta(
            provenance=provenance, head=head, n=len(temps), seed=seed, engine=engine,
            ensemble="npt", T=None, ci95=fit.get("alpha_ci95"),
            ci95_reason="fewer than 4 temperatures: no slope error", label=label,
            compound=results[0].compound, unit="1/K", temperatures_K=temps,
            T_ref_K=fit.get("T_ref_K"),
        )  # fmt: skip
        for res in results:
            base = f"md.{engine}.{compound}.{label}.npt.{fmt_T(res.temperature_K)}"
            numbers[f"{base}.alpha_per_K"] = fit["alpha_per_K"]
            numbers[f"{base}.alpha_per_K@meta"] = {**alpha_meta, "T": res.temperature_K}
        numbers[f"md.{engine}.{compound}.{label}.alpha_per_K"] = fit["alpha_per_K"]
        numbers[f"md.{engine}.{compound}.{label}.alpha_per_K@meta"] = {
            **alpha_meta,
            "reference": {**alpha_meta["reference"], "code": "experiment", "functional": None},
            "a_exp_source": EXPERIMENTAL_A_SOURCE,
        }
    return numbers


__all__.append("md_numbers")
