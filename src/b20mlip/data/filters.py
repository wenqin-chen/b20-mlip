"""``data filter``: structure dedupe, force cap and the magnetic-branch filter.

* :func:`dedupe` uses pymatgen's ``StructureMatcher`` with *tight* tolerances
  (``ltol=0.002`` = 0.2 % on lattice lengths, ``stol=0.005`` of the mean free length per atom
  (~0.01 Å for these cells), ``angle_tol=0.5°``, ``scale=False``, no primitive reduction, no
  supercell search) so that strained (the EOS grid steps the lattice by ~0.5 %), sheared or
  rattled derivatives of one parent are kept while byte-identical or numerically jittered
  copies collapse; the first frame of each duplicate cluster survives (file order). Only
  frames with the same sorted atomic numbers are ever compared. Because the matcher works
  up to rotation, symmetry-equivalent copies (the x/y/z uniaxial strains or xy/xz/yz shears
  of a cubic parent) are also merged: they carry no information for an E(3)-equivariant
  model and would cost identical DFT runs.
* :func:`force_cap` drops labelled frames whose largest force norm exceeds ``cap`` (eV/Å);
  unlabelled frames pass.
* :func:`magnetic_branch` rejects unconverged SCF frames and frames on the wrong magnetic
  branch: ``m_per_TM = abs_magnetization / n_TM`` (QE's "absolute magnetization" in μB;
  ``|total_magnetization| / n_TM`` when the absolute value is missing) must satisfy
  ``|m_per_TM - m_ref[compound]| <= tol``. Frames without a transition metal, without a
  reference entry, or without any magnetization value are kept and counted.
  ``DFTFrame`` attributes are read first, then same-named ``info`` keys (the extxyz round
  trip keeps DFT metadata in ``info``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from b20mlip.config import Settings
from b20mlip.data._common import n_transition_metals
from b20mlip.io import frame_to_atoms, read_frames, write_frames
from b20mlip.models import DFTFrame, Frame
from b20mlip.provenance import RunContext

Log = Callable[..., None] | None
DEDUPE_LTOL = 0.002
DEDUPE_STOL = 0.005
DEDUPE_ANGLE_TOL = 0.5


def _composition_key(frame: Frame) -> tuple[int, ...]:
    return tuple(sorted(frame.numbers))


def dedupe(
    frames: Iterable[Frame],
    *,
    ltol: float = DEDUPE_LTOL,
    stol: float = DEDUPE_STOL,
    angle_tol: float = DEDUPE_ANGLE_TOL,
    log: Log = None,
    stats: dict[str, Any] | None = None,
) -> list[Frame]:
    """Keep the first frame of every duplicate cluster (see the module docstring)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )
    kept: list[Frame] = []
    seen_ids: set[str] = set()
    representatives: dict[tuple[int, ...], list[Any]] = {}
    n_in = 0
    n_id_dup = 0
    n_struct_dup = 0
    for frame in frames:
        n_in += 1
        if frame.frame_id in seen_ids:
            n_id_dup += 1
            continue
        atoms = frame_to_atoms(frame)
        atoms.calc = None
        structure = AseAtomsAdaptor.get_structure(atoms)
        key = _composition_key(frame)
        reps = representatives.setdefault(key, [])
        if any(matcher.fit(structure, other) for other in reps):
            n_struct_dup += 1
            continue
        reps.append(structure)
        seen_ids.add(frame.frame_id)
        kept.append(frame)
    counts = {
        "n_in": n_in,
        "n_kept": len(kept),
        "n_duplicate_ids": n_id_dup,
        "n_duplicate_structures": n_struct_dup,
        "ltol": ltol,
        "stol": stol,
        "angle_tol": angle_tol,
    }
    if stats is not None:
        stats.update(counts)
    if log is not None:
        log(dedupe=counts)
    return kept


def max_force_norm(frame: Frame) -> float | None:
    if frame.forces is None:
        return None
    arr = np.asarray(frame.forces, dtype=float)
    return float(np.linalg.norm(arr, axis=1).max()) if arr.size else 0.0


def force_cap(
    frames: Iterable[Frame], cap: float, *, stats: dict[str, Any] | None = None
) -> list[Frame]:
    """Drop frames with ``max ||F|| > cap`` (eV/Å); frames without forces are kept."""
    kept: list[Frame] = []
    dropped = 0
    unlabelled = 0
    for frame in frames:
        fmax = max_force_norm(frame)
        if fmax is None:
            unlabelled += 1
        elif fmax > cap:
            dropped += 1
            continue
        kept.append(frame)
    if stats is not None:
        stats.update(
            n_kept=len(kept), n_dropped_force_cap=dropped, n_unlabelled=unlabelled, cap_eVA=cap
        )
    return kept


def _field(frame: Frame, name: str) -> Any:
    value = getattr(frame, name, None) if isinstance(frame, DFTFrame) else None
    if value is None:
        value = frame.info.get(name)
    return value


def m_per_tm(frame: Frame) -> float | None:
    """Magnetic moment per transition-metal atom (μB), or ``None`` when unknown."""
    n_tm = n_transition_metals(frame.numbers)
    if n_tm == 0:
        return 0.0
    abs_m = _field(frame, "abs_magnetization")
    if abs_m is None:
        total = frame.total_magnetization
        if total is None:
            total = frame.info.get("total_magnetization")
        if total is None:
            return None
        return abs(float(total)) / n_tm
    return abs(float(abs_m)) / n_tm


def magnetic_branch(
    frames: Iterable[Frame],
    m_ref: Mapping[str, float],
    tol: float,
) -> tuple[list[Frame], dict[str, Any]]:
    """Reject unconverged SCF frames and wrong-branch moments (module docstring)."""
    kept: list[Frame] = []
    counts: dict[str, Any] = {
        "n_in": 0,
        "n_kept": 0,
        "n_unconverged": 0,
        "n_wrong_branch": 0,
        "n_no_reference": 0,
        "n_no_magnetization": 0,
        "n_no_transition_metal": 0,
        "tol_muB": float(tol),
        "m_ref": {k: float(v) for k, v in m_ref.items()},
        "rejected": [],
    }
    for frame in frames:
        counts["n_in"] += 1
        converged = _field(frame, "converged")
        if converged is not None and not bool(converged):
            counts["n_unconverged"] += 1
            counts["rejected"].append({"frame_id": frame.frame_id, "reason": "unconverged"})
            continue
        ref = m_ref.get(frame.compound)
        n_tm = n_transition_metals(frame.numbers)
        branch_ok: bool | None = None
        if n_tm == 0:
            counts["n_no_transition_metal"] += 1
        elif ref is None:
            counts["n_no_reference"] += 1
        else:
            m = m_per_tm(frame)
            if m is None:
                counts["n_no_magnetization"] += 1
            elif abs(m - float(ref)) > tol:
                counts["n_wrong_branch"] += 1
                counts["rejected"].append(
                    {
                        "frame_id": frame.frame_id,
                        "reason": "wrong_branch",
                        "m_per_TM": m,
                        "m_ref": float(ref),
                    }
                )
                continue
            else:
                branch_ok = True
        if isinstance(frame, DFTFrame):
            kept.append(frame.model_copy(update={"branch_ok": branch_ok}))
        elif branch_ok is not None:
            kept.append(frame.model_copy(update={"info": {**frame.info, "branch_ok": True}}))
        else:
            kept.append(frame)
    counts["n_kept"] = len(kept)
    return kept, counts


def load_m_ref(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object {{compound: muB per TM atom}}")
    return {str(k): float(v) for k, v in data.items()}


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    frames: str | Path,
    out: str | Path,
    m_ref: Mapping[str, float] | str | Path | None = None,
    cap: float | None = None,
    tol: float | None = None,
    do_dedupe: bool = True,
) -> dict[str, Any]:
    """Stage ``data.filter``: dedupe -> force cap -> magnetic branch (when ``m_ref`` is given)."""
    src = Path(frames)
    ctx.add_input(src, "frames")
    cap_value = cfg.data.force_cap_eVA if cap is None else float(cap)
    tol_value = cfg.dft.branch_tol_muB if tol is None else float(tol)
    refs: Mapping[str, float] | None
    refs = load_m_ref(m_ref) if isinstance(m_ref, (str, Path)) else m_ref
    if refs is None:  # the dft tier's config carries the reference moments once measured
        cfg_refs = getattr(cfg.dft, "m_ref_muB", None)
        refs = dict(cfg_refs) if cfg_refs else None
    if ctx.dry_run:
        ctx.log(plan={"frames": str(src), "out": str(out), "dedupe": do_dedupe, "cap": cap_value})
        return {"planned": 1}
    data = read_frames(src)
    summary: dict[str, Any] = {"n_in": len(data)}
    if do_dedupe:
        stats: dict[str, Any] = {}
        data = dedupe(data, stats=stats)
        ctx.log(dedupe=stats)
        summary["n_after_dedupe"] = len(data)
    fstats: dict[str, Any] = {}
    data = force_cap(data, cap_value, stats=fstats)
    ctx.log(force_cap=fstats)
    summary["n_after_force_cap"] = len(data)
    if refs is not None:
        data, mstats = magnetic_branch(data, refs, tol_value)
        ctx.log(magnetic_branch=mstats)
        summary["n_after_magnetic_branch"] = len(data)
        summary["n_wrong_branch"] = mstats["n_wrong_branch"]
        summary["n_unconverged"] = mstats["n_unconverged"]
    else:
        ctx.log(magnetic_branch="skipped (no m_ref given)")
    out_path = Path(out)
    write_frames(data, out_path)
    ctx.add_output(out_path, "frames")
    summary["n_out"] = len(data)
    return summary


__all__ = [
    "DEDUPE_ANGLE_TOL",
    "DEDUPE_LTOL",
    "DEDUPE_STOL",
    "dedupe",
    "force_cap",
    "load_m_ref",
    "m_per_tm",
    "magnetic_branch",
    "max_force_norm",
    "run",
]
