"""Build ``force_sets_<compound>.json`` from QE-labelled ``phonon_disp`` frames of the sampled
round-0 set (``data sample`` displacement supercells labelled through ``dft prep/run/collect``).

The frames carry ``phonopy_disp_number`` (-1 = undisplaced reference supercell),
``phonopy_disp_atom`` and ``phonopy_disp_vector``; the reference supercell's residual forces are
subtracted from every displaced set (the usual finite-displacement correction). Runs as the
``dft.phonons`` stage so the file has a manifest; the phonons tier reads the result unchanged.

    uv run scripts/force_sets_from_sampled.py --compound FeSi \\
        --frames data/frames/labelled_r0_large.extxyz --out data/phonons/force_sets_FeSi.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from b20mlip.config import load_config
from b20mlip.dft.phonons import build_force_sets, write_force_sets
from b20mlip.dft.structures import reference_frame
from b20mlip.io import read_frames
from b20mlip.provenance import run_stage


def stage(
    cfg,
    ctx,
    *,
    compound: str,
    frames: str,
    out: str,
    supercell=(2, 2, 2),
    distance=0.03,
    allow_partial: bool = False,
):
    path = Path(frames)
    ctx.add_input(path, "frames")
    disp = [
        f for f in read_frames(path)
        if f.compound == compound and f.config_type == "phonon_disp" and f.forces is not None
    ]  # fmt: skip
    by_n = {int(f.info["phonopy_disp_number"]): f for f in disp}
    if -1 not in by_n:
        raise ValueError(f"{compound}: no undisplaced reference supercell (phonopy_disp_number -1)")
    ref_forces = np.asarray(by_n[-1].forces, dtype=float)
    numbers = sorted(n for n in by_n if n >= 0)
    complete = numbers == list(range(len(numbers)))
    if not complete and not allow_partial:
        raise ValueError(f"{compound}: displacement set incomplete: have {numbers}")
    if not complete:
        # phonopy solves the force constants from whatever displacements the dataset holds;
        # with one (single-sided) displacement per inequivalent atom the result is valid but
        # lacks the +/- cancellation of odd anharmonic terms (O(u) bias, u = 0.03 A): labelled.
        atoms_covered = {int(by_n[n].info["phonopy_disp_atom"]) for n in numbers}
        ctx.log(partial_displacement_set=numbers, atoms_covered=sorted(atoms_covered))
    parent = reference_frame(cfg, compound)
    first_atoms, forces = [], []
    for n in numbers:
        f = by_n[n]
        vec = f.info["phonopy_disp_vector"]
        vec = json.loads(vec) if isinstance(vec, str) else list(vec)
        first_atoms.append({"number": int(f.info["phonopy_disp_atom"]), "displacement": vec})
        forces.append((np.asarray(f.forces, dtype=float) - ref_forces).tolist())
    dataset = {"natom": len(by_n[-1].numbers), "first_atoms": first_atoms}
    data = build_force_sets(
        parent, supercell, distance, dataset, forces, compound=compound, run_id=ctx.run_id, cfg=cfg
    )
    data["displacement_set"] = {
        "numbers": numbers,
        "complete": complete,
        "note": (
            None if complete else "single-sided displacements only for some atoms (partial set)"
        ),
    }
    data["residual_force_correction"] = {
        "reference_frame_id": by_n[-1].frame_id,
        "max_residual_force_eVA": float(np.abs(ref_forces).max()),
    }
    data["source_frames"] = {
        "path": str(path),
        "frame_ids": [by_n[n].frame_id for n in [-1, *numbers]],
    }
    local = write_force_sets(data, ctx.out_dir / "force_sets.json")
    ctx.add_output(local, "json")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_force_sets(data, out_path)
    ctx.add_output(out_path, "json")
    ctx.log(
        compound=compound,
        n_displacements=len(numbers),
        supercell=list(supercell),
        distance=distance,
    )
    return {
        "compound": compound,
        "n_displacements": len(numbers),
        "out": str(out_path),
        "max_residual_force_eVA": data["residual_force_correction"]["max_residual_force_eVA"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compound", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--allow-partial", action="store_true", help="accept an incomplete +/- set")
    a = ap.parse_args()
    cfg = load_config([Path(c) for c in a.config], [])
    result = run_stage(
        "dft.phonons",
        cfg,
        stage,
        compound=a.compound,
        frames=a.frames,
        out=a.out,
        allow_partial=a.allow_partial,
    )
    payload = {"status": result.status, "run_id": result.run_id, "summary": result.summary}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
