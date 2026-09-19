"""``dft phonons``: phonopy displacement sets labelled by QE, written as ``force_sets.json``.

Flow: reference cell (relaxed MPtrj parent or ``--structure``) -> ``Phonopy`` supercell
displacements (``distance`` Å, symmetry-reduced) -> one QE unit per displaced supercell
(``config_type="phonon_disp"``, unit ids ``<compound>_dispNNN``) -> forces (eV/Å) into the
phonopy dataset -> ``force_sets.json`` in exactly the schema the phonons tier reads::

    {"schema": "b20mlip.force_sets.v1", "compound": "FeSi",
     "unitcell": {"numbers": [...], "positions": [[...]], "cell": [[...]]},
     "supercell_matrix": [[2,0,0],[0,2,0],[0,0,2]], "displacement_distance": 0.03,
     "dataset": {"natom": N_super, "first_atoms": [{"number": i, "displacement": [dx,dy,dz],
                                                    "forces": [[fx,fy,fz], ...]}, ...]},
     "reference": {"code": "qe", "functional": "PBE", "pseudos": "SSSP-efficiency-1.3",
                   "e0_source": null, "cross_functional": false},
     "units": {"forces": "eV/A", "positions": "A"}, "source_run_id": "<run_id>"}

``phonopy_from_force_sets`` reads it back (``Phonopy.dataset = ...``) so the round trip is
tested without QE. Note: the reference cell is relaxed at the MP (VASP PBE) level, not by QE;
residual forces at the undisplaced geometry are not subtracted here (see contract notes).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from b20mlip.config import Settings
from b20mlip.dft import qe
from b20mlip.dft.stages import stage_result, submit_units, write_counts
from b20mlip.dft.structures import reference_frame
from b20mlip.io import frame_id_for
from b20mlip.models import Frame, Reference, StageResult, Status
from b20mlip.provenance import RunContext

FORCE_SETS_SCHEMA = "b20mlip.force_sets.v1"
PLAN_JSON = "phonon_plan.json"
FORCE_SETS_NAME = "force_sets.json"


def supercell_matrix(supercell: Sequence[int]) -> list[list[int]]:
    if len(supercell) != 3 or any(int(n) < 1 for n in supercell):
        raise ValueError(f"supercell must be three positive integers, got {list(supercell)}")
    return [[int(supercell[0]), 0, 0], [0, int(supercell[1]), 0], [0, 0, int(supercell[2])]]


def phonopy_unitcell(frame: Frame) -> Any:
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        numbers=[int(z) for z in frame.numbers],
        cell=np.asarray(frame.cell, dtype=float),
        positions=np.asarray(frame.positions, dtype=float),
    )


def unit_ids_for(compound: str, n: int) -> list[str]:
    return [f"{compound}_disp{i:03d}" for i in range(n)]


def displacement_frames(
    frame: Frame, supercell: Sequence[int], distance: float, *, compound: str | None = None
) -> tuple[list[Frame], dict[str, Any]]:
    """Displaced supercells as ``Frame``s (``phonon_disp``) plus the phonopy dataset (no forces)."""
    from phonopy import Phonopy

    compound = compound or frame.compound
    matrix = supercell_matrix(supercell)
    ph = Phonopy(phonopy_unitcell(frame), supercell_matrix=np.asarray(matrix))
    ph.generate_displacements(distance=float(distance))
    raw: dict[str, Any] = dict(ph.dataset or {})  # type-1 dataset: natom + first_atoms
    supercells = ph.supercells_with_displacements or []
    dataset: dict[str, Any] = {
        "natom": int(raw["natom"]),
        "first_atoms": [
            {"number": int(d["number"]), "displacement": [float(x) for x in d["displacement"]]}
            for d in raw["first_atoms"]
        ],
    }
    frames: list[Frame] = []
    for i, sc in enumerate(supercells):
        numbers = [int(z) for z in sc.numbers]
        positions = np.asarray(sc.positions, dtype=float).tolist()
        cell = np.asarray(sc.cell, dtype=float).tolist()
        d = dataset["first_atoms"][i]
        frames.append(
            Frame(
                frame_id=frame_id_for(numbers, positions, cell),
                group_id=f"{compound}/phonon_disp/{frame.frame_id}",
                compound=compound,
                config_type="phonon_disp",
                parent_id=frame.frame_id,
                numbers=numbers,
                positions=positions,
                cell=cell,
                info={
                    "disp_index": i,
                    "disp_atom": d["number"],
                    "displacement": d["displacement"],
                    "displacement_distance": float(distance),
                    "supercell": [int(n) for n in supercell],
                },
            )
        )
    return frames, dataset


def build_force_sets(
    frame: Frame,
    supercell: Sequence[int],
    distance: float,
    dataset: Mapping[str, Any],
    forces: Sequence[Sequence[Sequence[float]]],
    *,
    compound: str,
    run_id: str,
    cfg: Settings,
) -> dict[str, Any]:
    """Assemble the ``force_sets.json`` document (forces in eV/Å, one array per displacement)."""
    first_atoms = list(dataset["first_atoms"])
    if len(forces) != len(first_atoms):
        raise ValueError(f"{len(forces)} force sets for {len(first_atoms)} displacements")
    natom = int(dataset["natom"])
    entries = []
    for d, f in zip(first_atoms, forces, strict=True):
        arr = np.asarray(f, dtype=float)
        if arr.shape != (natom, 3):
            raise ValueError(f"force set shape {arr.shape} != ({natom}, 3)")
        entries.append(
            {
                "number": int(d["number"]),
                "displacement": [float(x) for x in d["displacement"]],
                "forces": arr.tolist(),
            }
        )
    reference = Reference(
        code="qe", functional="PBE", pseudos=cfg.dft.pseudo_family, e0_source=None,
        cross_functional=False,
    )  # fmt: skip
    return {
        "schema": FORCE_SETS_SCHEMA,
        "compound": compound,
        "unitcell": {
            "numbers": [int(z) for z in frame.numbers],
            "positions": np.asarray(frame.positions, dtype=float).tolist(),
            "cell": np.asarray(frame.cell, dtype=float).tolist(),
        },
        "supercell_matrix": supercell_matrix(supercell),
        "displacement_distance": float(distance),
        "dataset": {"natom": natom, "first_atoms": entries},
        "reference": reference.model_dump(mode="json"),
        "units": {"forces": "eV/A", "positions": "A"},
        "source_run_id": run_id,
    }


def write_force_sets(data: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(data), indent=1) + "\n", encoding="utf-8")
    return p


def read_force_sets(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != FORCE_SETS_SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r} != {FORCE_SETS_SCHEMA!r}")
    return data


def phonopy_from_force_sets(data: Mapping[str, Any]) -> Any:
    """A ``Phonopy`` with the dataset (displacements + forces) loaded and force constants built."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    uc = data["unitcell"]
    unitcell = PhonopyAtoms(
        numbers=[int(z) for z in uc["numbers"]],
        cell=np.asarray(uc["cell"], dtype=float),
        positions=np.asarray(uc["positions"], dtype=float),
    )
    ph = Phonopy(unitcell, supercell_matrix=np.asarray(data["supercell_matrix"]))
    ph.dataset = {
        "natom": int(data["dataset"]["natom"]),
        "first_atoms": [
            {
                "number": int(d["number"]),
                "displacement": np.asarray(d["displacement"], dtype=float),
                "forces": np.asarray(d["forces"], dtype=float),
            }
            for d in data["dataset"]["first_atoms"]
        ],
    }
    ph.produce_force_constants()
    return ph


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    compound: str,
    supercell: Sequence[int] = (2, 2, 2),
    distance: float = 0.03,
    structure: str | Path | None = None,
    units: str | Path | None = None,
    out: str | Path | None = None,
    wait: bool = True,
    collect_only: bool = False,
    template: str = "qe_phonons",
    resources: Mapping[str, Any] | None = None,
) -> StageResult:
    """Displacements -> QE units -> ``force_sets.json`` (in ``ctx.out_dir`` and at ``out``)."""
    root = Path(units) if units is not None else Path(cfg.paths.dft_dir) / f"phonons_{compound}"
    out_path = (
        Path(out)
        if out is not None
        else Path(cfg.paths.data_dir) / "phonons" / f"force_sets_{compound}.json"
    )
    ref = reference_frame(cfg, compound, structure)
    if structure is not None:
        ctx.add_input(Path(structure), "frames")
    frames, dataset = displacement_frames(ref, supercell, distance, compound=compound)
    ids = unit_ids_for(compound, len(frames))
    qe.plan_units(frames, root, cfg, unit_ids=ids)
    plan = {
        "compound": compound,
        "supercell": [int(n) for n in supercell],
        "displacement_distance": float(distance),
        "unitcell": ref.model_dump(mode="json"),
        "dataset": dataset,
        "units": ids,
    }
    Path(root, PLAN_JSON).write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        "compound": compound,
        "units_root": str(root),
        "n_displacements": len(ids),
        "natom_supercell": dataset["natom"],
        "frame_id": ref.frame_id,
    }
    if ctx.dry_run:
        ctx.log(n_units=len(ids), dry_run=True, units_root=str(root))
        return stage_result(ctx, "partial", summary)
    pending = qe.pending_units(root, ids)
    if pending and not collect_only:
        submit_units(cfg, ctx, root, pending, template=template, resources=resources, wait=wait)
    parsed, counts = qe.collect(root)
    write_counts(ctx, counts)
    by_id = {f.unit_id: f for f in parsed}
    ctx.log(
        n_units=len(ids),
        n_failed=counts["failed"] + counts["unconverged"],
        counts=dict(counts),
        units_root=str(root),
        pw_version=next(
            (f.info.get("pw_version") for f in parsed if f.info.get("pw_version")), None
        ),
        pseudo_md5s={
            s: cfg.dft.pseudo_md5s[s]
            for s in qe.species_order(ref.numbers)
            if s in cfg.dft.pseudo_md5s
        },
        supercell=[int(n) for n in supercell],
        displacement_distance=float(distance),
    )
    forces: list[list[list[float]]] = []
    for uid in ids:
        f = by_id.get(uid)
        if f is None or not f.converged or f.forces is None:
            status: Status = "partial" if counts["missing"] else "failed"
            return stage_result(ctx, status, {**summary, **counts, "first_missing_unit": uid})
        forces.append(f.forces)
    data = build_force_sets(
        ref, supercell, distance, dataset, forces, compound=compound, run_id=ctx.run_id, cfg=cfg
    )
    local = write_force_sets(data, ctx.out_dir / FORCE_SETS_NAME)
    ctx.add_output(local, "json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, out_path)
    ctx.add_output(out_path, "json")
    return stage_result(ctx, "ok", {**summary, **counts, "out": str(out_path)})


__all__ = [
    "FORCE_SETS_NAME",
    "FORCE_SETS_SCHEMA",
    "build_force_sets",
    "displacement_frames",
    "phonopy_from_force_sets",
    "phonopy_unitcell",
    "read_force_sets",
    "run",
    "supercell_matrix",
    "unit_ids_for",
    "write_force_sets",
]
