"""Model-vs-reference phonon comparison (CONTRACTS.md row 9, SPEC.md section 6).

Both inputs are :class:`~b20mlip.models.PhononResult` objects with the conventions of
:mod:`b20mlip.phonons.harmonic`: the same seekpath path sampled at the same ``npoints``
(default 100) q-points in reduced coordinates of the primitive cell, ``3 x N_primitive``
branches sorted ascending at every q-point, frequencies in meV.

* ``omega_mae(model, ref)`` — mean |ω_model − ω_ref| over all q-points and branches (meV),
  branch ``j`` of the model paired with branch ``j`` of the reference after sorting. The two
  results must have the same primitive-cell size (branch count) and the same reduced
  q-points (``atol`` 1e-6); otherwise ``ValueError``.
* ``softening_index(model, ref)`` — ``s = median(ω_model / ω_ref)`` over q-points and
  branches with ``|ω_ref| > 1 meV`` (acoustic branches near Γ are excluded by that cut);
  ``s < 1`` means the model's PES is softer than the reference.
* ``compare(model, ref)`` — the model result with ``omega_mae_meV`` and ``softening_index``
  filled and its ``reference`` replaced by the reference result's :class:`Reference`
  (so the compared result says what it was compared against; ``source_label`` stays the
  model's).
* ``imaginary_delta`` — model minus reference imaginary-mode count (ω < −0.4 meV).

Model-cell vs DFT-cell: the ``eval phonons`` stage computes the model at the reference
cell (``--cell dft``, the headline) or at the model-relaxed cell (``--cell model``,
``cell_source="model_relaxed"``); the reduced q-points of a uniformly relaxed cubic cell are
identical, so the comparison is still well defined and the ``cell_source`` is recorded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms

from b20mlip.config import Settings
from b20mlip.evaluate.discovery import relax
from b20mlip.evaluate.errors import (
    UNSPECIFIED_E0,
    label_for,
    make_calculator,
    model_energy_scale,
    read_checkpoint,
)
from b20mlip.models import PhononResult, Reference
from b20mlip.phonons import harmonic
from b20mlip.provenance import RunContext, find_runs, sha256_file

SOFTENING_MIN_REF_MEV = 1.0
QPOINT_ATOL = 1e-6
REFERENCES: tuple[str, ...] = ("qe", "phonondb103", "pbesol")
KEY_SUFFIX: dict[str, str] = {"qe": "", "phonondb103": "_phonondb103", "pbesol": "_pbesol"}
NUMBERS_FILE = "numbers.json"
MODEL_JSON = "phonons_model.json"
REFERENCE_JSON = "phonons_reference.json"
SUMMARY_JSON = "phonon_compare.json"
PHONONDB_STRUCTURES = Path("raw") / "references" / "phononDB-PBE-103-structures.extxyz"


def _frequencies(result: PhononResult) -> np.ndarray:
    arr = np.asarray(result.frequencies_meV, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            f"{result.source_label}: frequencies must be a non-empty (nq, nbranch) array"
        )
    return np.sort(arr, axis=1)


def check_comparable(model: PhononResult, ref: PhononResult) -> tuple[np.ndarray, np.ndarray]:
    """Sorted ``(ω_model, ω_ref)`` arrays after asserting the same cell size and q-points."""
    wm, wr = _frequencies(model), _frequencies(ref)
    if wm.shape[1] != wr.shape[1]:
        raise ValueError(
            f"branch count differs: model {wm.shape[1]} vs reference {wr.shape[1]} "
            "(different primitive cells)"
        )
    qm = np.asarray(model.qpoints, dtype=float)
    qr = np.asarray(ref.qpoints, dtype=float)
    if qm.shape != qr.shape:
        raise ValueError(f"q-point count differs: model {qm.shape[0]} vs reference {qr.shape[0]}")
    if not np.allclose(qm, qr, atol=QPOINT_ATOL):
        raise ValueError("q-points differ between model and reference (same path/npoints needed)")
    if wm.shape[0] != qm.shape[0] or wr.shape[0] != qr.shape[0]:
        raise ValueError("frequencies and q-points disagree in length")
    return wm, wr


def omega_mae(model: PhononResult, ref: PhononResult) -> float:
    """Branch-wise mean |Δω| (meV) over the shared q-points."""
    wm, wr = check_comparable(model, ref)
    return float(np.mean(np.abs(wm - wr)))


def softening_index(model: PhononResult, ref: PhononResult) -> float:
    """``median(ω_model / ω_ref)`` over entries with ``|ω_ref| > 1 meV``."""
    wm, wr = check_comparable(model, ref)
    mask = np.abs(wr) > SOFTENING_MIN_REF_MEV
    if not np.any(mask):
        raise ValueError("no reference frequency above 1 meV; softening index undefined")
    return float(np.median(wm[mask] / wr[mask]))


def imaginary_delta(model: PhononResult, ref: PhononResult) -> int:
    return int(model.imaginary_count) - int(ref.imaginary_count)


def compare(model: PhononResult, ref: PhononResult) -> PhononResult:
    """The model result with ``omega_mae_meV``/``softening_index`` filled (see module)."""
    mae = omega_mae(model, ref)
    s = softening_index(model, ref)
    return model.model_copy(
        update={"omega_mae_meV": mae, "softening_index": s, "reference": ref.reference}
    )


def load_reference(path: str | Path) -> PhononResult:
    """A reference :class:`PhononResult` JSON (``harmonic.to_json`` format)."""
    return harmonic.from_json(path)


def summary(model: PhononResult, ref: PhononResult) -> dict[str, Any]:
    """Plain-JSON comparison summary (what ``eval phonons`` records)."""
    compared = compare(model, ref)
    wm, wr = check_comparable(model, ref)
    return {
        "compound": model.compound,
        "model_label": model.source_label,
        "reference_label": ref.source_label,
        "reference": ref.reference.model_dump(mode="json"),
        "cell_source": model.cell_source,
        "n_qpoints": int(wm.shape[0]),
        "n_branches": int(wm.shape[1]),
        "omega_mae_meV": compared.omega_mae_meV,
        "omega_max_abs_delta_meV": float(np.max(np.abs(wm - wr))),
        "softening_index": compared.softening_index,
        "imaginary_count_model": int(model.imaginary_count),
        "imaginary_count_reference": int(ref.imaginary_count),
        "imaginary_delta": imaginary_delta(model, ref),
        "omega_max_model_meV": float(np.max(wm)),
        "omega_max_reference_meV": float(np.max(wr)),
    }


def write_summary(data: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return p


# --- stage ----------------------------------------------------------------------------------------


def cell_from_force_sets(path: str | Path) -> tuple[Atoms, np.ndarray]:
    """``(unit cell, supercell matrix)`` stored in a ``b20mlip.force_sets.v1`` document."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != harmonic.FORCE_SETS_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {harmonic.FORCE_SETS_SCHEMA!r}")
    uc = doc["unitcell"]
    atoms = Atoms(
        numbers=[int(z) for z in uc["numbers"]],
        positions=np.asarray(uc["positions"], dtype=float),
        cell=np.asarray(uc["cell"], dtype=float),
        pbc=True,
    )
    return atoms, harmonic.supercell_matrix(doc["supercell_matrix"])


def find_force_sets(cfg: Settings, compound: str, force_sets: str | Path | None = None) -> Path:
    """The QE ``force_sets.json`` of ``compound``: an explicit path, else
    ``<data_dir>/phonons/force_sets_<compound>.json`` (the dft stage's default output), else
    the newest ok ``dft.phonons`` run that produced a file of that name."""
    if force_sets is not None:
        path = Path(force_sets)
        if not path.is_file():
            raise FileNotFoundError(f"force sets file not found: {path}")
        return path
    default = Path(cfg.paths.data_dir) / "phonons" / f"force_sets_{compound}.json"
    if default.is_file():
        return default
    name = f"force_sets_{compound}.json"
    for manifest in reversed(find_runs("dft.phonons", cfg.paths.runs_dir)):
        if manifest.status != "ok":
            continue
        for art in manifest.outputs:
            if Path(art.path).name == name and Path(art.path).is_file():
                return Path(art.path)
            if Path(art.path).name == "force_sets.json" and Path(art.path).is_file():
                local = (
                    Path(cfg.paths.runs_dir) / "dft.phonons" / manifest.run_id / "force_sets.json"
                )
                if local.is_file():
                    return local
    raise FileNotFoundError(
        f"no QE force sets for {compound}: run `b20mlip dft phonons --compound {compound}` "
        "(cluster) or pass --force-sets"
    )


def reference_cell(cfg: Settings, compound: str, structure: str | Path | None) -> Atoms:
    from b20mlip.dft.structures import reference_frame  # noqa: PLC0415 - row 6, allowed
    from b20mlip.io import frame_to_atoms  # noqa: PLC0415

    atoms = frame_to_atoms(reference_frame(cfg, compound, structure))
    atoms.calc = None
    atoms.info = {}
    return atoms


def load_labelled_reference(reference: str, reference_json: str | Path) -> PhononResult:
    """A reference PhononResult JSON for ``phonondb103`` (VASP PBE) or ``pbesol``
    (flagged ``cross_functional=True``, gate A5)."""
    ref = load_reference(reference_json)
    if reference == "pbesol":
        flagged = ref.reference.model_copy(
            update={"functional": ref.reference.functional or "PBEsol", "cross_functional": True}
        )
        return ref.model_copy(update={"reference": flagged})
    return ref


def numbers_for_run(
    result: dict[str, Any],
    ref: Reference,
    *,
    compound: str,
    label: str,
    reference: str,
    head: str,
    e0_source: str,
    energy_scale: str | None,
    model_sha256: str,
    supercell: Sequence[int],
    displacement: float,
) -> dict[str, Any]:
    """``eval.phonons.<compound>.<label>.<metric><suffix>`` (suffix names the reference)."""
    suffix = KEY_SUFFIX[reference]
    base: dict[str, Any] = {
        "reference": ref.model_dump(mode="json"),
        "head": head,
        "n": int(result["n_qpoints"]),
        "seed": 0,
        "ci95": None,
        "ci95_reason": "deterministic harmonic quantity over a fixed q-path; no resampling",
        "e0_source": e0_source,
        "energy_scale": energy_scale,
        "model_label": label,
        "bracket": label,
        "tier": "phonons",
        "compound": compound,
        "reference_label": reference,
        "cell_source": result["cell_source"],
        "supercell": [int(x) for x in supercell],
        "displacement_A": float(displacement),
        "n_branches": int(result["n_branches"]),
        "model_sha256": model_sha256,
        "cross_functional": bool(ref.cross_functional),
    }
    out: dict[str, Any] = {}
    for name, unit in (
        ("omega_mae_meV", "meV"),
        ("softening_index", "ratio"),
        ("imaginary_count", "count"),
    ):
        value = result["imaginary_count_model"] if name == "imaginary_count" else result[name]
        key = f"eval.phonons.{compound}.{label}.{name}{suffix}"
        out[key] = float(value)
        out[f"{key}@meta"] = {**base, "unit": unit}
    return out


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    compound: str,
    reference: str = "qe",
    cell: str = "dft",
    reference_json: str | Path | None = None,
    force_sets: str | Path | None = None,
    structure: str | Path | None = None,
    supercell: Sequence[int] = (2, 2, 2),
    distance: float = harmonic.DEFAULT_DISTANCE,
    npoints: int = harmonic.DEFAULT_NPOINTS,
    mesh: int | None = harmonic.DEFAULT_MESH,
    head: str = "Default",
    label: str | None = None,
    checkpoint_json: str | Path | None = None,
    e0_source: str | None = None,
    calc: Any | None = None,
) -> dict[str, Any]:
    """Stage ``eval.phonons``: model phonons vs a labelled reference -> ω-MAE, softening index.

    ``reference="qe"`` reads the dft tier's ``force_sets.json`` (own QE PBE; the model uses the
    same unit cell and supercell). ``"phonondb103"`` needs ``reference_json`` (only the 76 KB
    structures file is local; the reference force sets are not downloaded) and ``"pbesol"``
    needs it too and is flagged cross-functional. ``cell="model"`` relaxes the cell with the
    model first (``cell_source="model_relaxed"``).
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
    if cell not in ("dft", "model"):
        raise ValueError(f"cell must be dft or model, got {cell!r}")
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    ctx.add_input(model_path, "model")
    model_sha = sha256_file(model_path)
    info = None
    if checkpoint_json is not None:
        info = read_checkpoint(checkpoint_json)
        ctx.add_input(Path(checkpoint_json), "json")
        if info.sha256 != model_sha:
            raise ValueError("checkpoint.json sha256 does not match the model file")
    if e0_source is None:
        e0_source = (
            "foundation" if head == "pt_head" else (info.e0_source if info else UNSPECIFIED_E0)
        )
    model_label = label or label_for(info, model_path)
    scale = model_energy_scale(info, head)

    matrix = harmonic.supercell_matrix(list(supercell))
    if reference == "qe":
        fs_path = find_force_sets(cfg, compound, force_sets)
        ctx.add_input(fs_path, "json")
        atoms, matrix = cell_from_force_sets(fs_path)
        ref = harmonic.from_force_sets(None, fs_path, None, npoints=npoints, mesh=mesh)
    else:
        if reference_json is None:
            if reference == "phonondb103":
                raise FileNotFoundError(
                    "phononDB-PBE-103 reference force sets not downloaded: only the structures "
                    f"file ({PHONONDB_STRUCTURES}) is local; pass --reference-json with a "
                    "PhononResult built from the phonopy_params.yaml.xz of this material"
                )
            raise FileNotFoundError("reference pbesol needs --reference-json (a PhononResult JSON)")
        ref_path = Path(reference_json)
        ctx.add_input(ref_path, "json")
        ref = load_labelled_reference(reference, ref_path)
        if len(ref.supercell) == 3:
            matrix = harmonic.supercell_matrix(ref.supercell)
        npoints = len(ref.qpoints)
        atoms = reference_cell(cfg, compound, structure)
        if structure is not None:
            ctx.add_input(Path(structure), "frames")
    if ref.compound != compound and reference == "qe":
        raise ValueError(f"force sets are for {ref.compound}, not {compound}")

    plan = {
        "model": str(model_path), "model_sha256": model_sha, "head": head,
        "model_label": model_label,
        "compound": compound, "reference": reference, "cell": cell, "supercell": matrix.tolist(),
        "displacement": float(distance), "npoints": int(npoints), "n_atoms": len(atoms),
        "dry_run": bool(ctx.dry_run),
    }  # fmt: skip
    (ctx.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / "plan.json", "json")
    if ctx.dry_run:
        ctx.log(
            model_sha256=model_sha, head=head, compound=compound, reference=reference, planned=True
        )
        return {
            "compound": compound,
            "model_label": model_label,
            "reference": reference,
            "planned": 1,
        }

    calc = (
        calc
        if calc is not None
        else make_calculator(model_path, head, device=cfg.compute.device, dtype=cfg.compute.dtype)
    )
    cell_source = "dft"
    relax_steps: int | None = None
    if cell == "model":
        relaxed, relax_steps = relax(
            atoms, calc, fmax=cfg.eval.fmax, steps=cfg.eval.max_steps, fix_symmetry=True
        )
        relaxed.calc = None
        relaxed.set_constraint()
        atoms = relaxed
        cell_source = "model_relaxed"
    model_result = harmonic.compute(
        atoms, calc, matrix, float(distance), cell_source=cast(Any, cell_source),
        npoints=int(npoints),
        mesh=mesh, source_label=model_label, run_id=ctx.run_id,
    )  # fmt: skip
    compared = compare(model_result, ref)
    result = summary(model_result, ref)
    result.update({"relax_steps": relax_steps, "a_model_A": float(np.cbrt(atoms.get_volume()))})
    harmonic.to_json(compared, ctx.out_dir / MODEL_JSON)
    harmonic.to_json(ref, ctx.out_dir / REFERENCE_JSON)
    write_summary(result, ctx.out_dir / SUMMARY_JSON)
    for name in (MODEL_JSON, REFERENCE_JSON, SUMMARY_JSON):
        ctx.add_output(ctx.out_dir / name, "json")
    numbers = numbers_for_run(
        result, ref.reference, compound=compound, label=model_label, reference=reference, head=head,
        e0_source=e0_source, energy_scale=scale, model_sha256=model_sha,
        supercell=harmonic.supercell_field(matrix), displacement=float(distance),
    )  # fmt: skip
    (ctx.out_dir / NUMBERS_FILE).write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / NUMBERS_FILE, "json")
    ctx.log(
        model_sha256=model_sha, head=head, tier="phonons", compound=compound, n=result["n_qpoints"],
        bootstrap_seed=None, reference=ref.reference.model_dump(mode="json"),
        reference_label=reference,
        e0_source=e0_source, energy_scale=scale, model_label=model_label, cell_source=cell_source,
        supercell=matrix.tolist(), displacement=float(distance),
    )  # fmt: skip
    return {
        "compound": compound,
        "model_label": model_label,
        "reference": reference,
        "cell_source": cell_source,
        "omega_mae_meV": round(float(result["omega_mae_meV"]), 4),
        "softening_index": round(float(result["softening_index"]), 4),
        "imaginary_count": int(result["imaginary_count_model"]),
        "imaginary_count_reference": int(result["imaginary_count_reference"]),
    }


__all__ = [
    "KEY_SUFFIX",
    "MODEL_JSON",
    "NUMBERS_FILE",
    "PHONONDB_STRUCTURES",
    "QPOINT_ATOL",
    "REFERENCES",
    "REFERENCE_JSON",
    "SUMMARY_JSON",
    "cell_from_force_sets",
    "find_force_sets",
    "load_labelled_reference",
    "numbers_for_run",
    "reference_cell",
    "run",
    "SOFTENING_MIN_REF_MEV",
    "check_comparable",
    "compare",
    "imaginary_delta",
    "load_reference",
    "omega_mae",
    "softening_index",
    "summary",
    "write_summary",
]
