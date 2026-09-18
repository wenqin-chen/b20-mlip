"""Harmonic phonons (CONTRACTS.md row 8): finite displacements with phonopy, seekpath band path.

Conventions (binding for the evaluate tier's ``phonon_compare``):

* ``PhononResult.qpoints`` are reduced coordinates of the reciprocal lattice of phonopy's
  *primitive* cell (``primitive_matrix="auto"``); for B20 (simple cubic) the primitive cell is
  the 8-atom cell, so ``frequencies_meV`` has ``3 x 8 = 24`` branches. In general there are
  ``3 x N_primitive`` branches, sorted ascending at every q-point.
* The path is seekpath's standard path for the primitive cell (``qpath="seekpath"``), sampled
  with exactly ``npoints`` (default 100) q-points distributed over the segments in proportion
  to their Cartesian length (every segment keeps both end points, so a shared corner of two
  connected segments appears twice). ``qpath_labels`` has one entry per q-point: the seekpath
  label (``"GAMMA"``, ``"X"``, ...) at segment ends, ``""`` elsewhere. Two results computed on
  the same cell with the same ``npoints`` therefore share their q-points exactly.
* Frequencies are in meV (``THz x 4.135667696``); negative values are imaginary modes and
  ``imaginary_count`` counts branch values below ``-0.4`` meV along the path.
* ``dos_meV`` / ``dos`` is the total DOS on a Gamma-centred mesh (tetrahedron method), in
  states per meV per primitive cell (integrates to ``3 x N_primitive``).
* The acoustic sum rule is enforced after ``produce_force_constants``: phonopy's
  ``symmetrize_force_constants`` (translational + permutation symmetry) followed by an explicit
  final ``set_translational_invariance``, so the three acoustic branches vanish at Gamma.
* ``softening_index`` and ``omega_mae_meV`` are ``None`` here; ``evaluate.phonon_compare``
  fills them from a model/reference pair.

``from_force_sets`` reads the ``b20mlip.force_sets.v1`` JSON written by the dft tier (phonopy's
displacement dataset plus the unit cell, supercell matrix and reference metadata).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from phonopy import Phonopy
from phonopy.harmonic.force_constants import set_translational_invariance
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import PrimitiveMatrixAutoDefaultWarning

from b20mlip.models import PhononResult, Reference

THZ_TO_MEV = 4.135667696
IMAGINARY_TOL_MEV = -0.4
DEFAULT_NPOINTS = 100
DEFAULT_MESH = 20
DEFAULT_DISTANCE = 0.03
FORCE_SETS_SCHEMA = "b20mlip.force_sets.v1"
FORCE_SETS_UNITS: dict[str, str] = {
    "positions": "Angstrom",
    "cell": "Angstrom",
    "displacement": "Angstrom",
    "forces": "eV/Angstrom",
}
QPath = Literal["seekpath"]
CellSource = Literal["dft", "model_relaxed"]


# --- cells ---------------------------------------------------------------------------------------


def to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell[:], dtype=float),
        scaled_positions=atoms.get_scaled_positions(),
    )


def from_phonopy(cell: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=cell.symbols,
        cell=np.asarray(cell.cell, dtype=float),
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


SupercellSpec = int | Sequence[int] | Sequence[Sequence[int]] | np.ndarray


def supercell_matrix(supercell: SupercellSpec) -> np.ndarray:
    """``2`` -> ``2I``; ``[2, 2, 2]`` -> ``diag``; a 3x3 matrix -> itself (integers)."""
    arr = np.asarray(supercell)
    if arr.ndim == 0:
        arr = np.eye(3) * int(arr)
    elif arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"supercell needs 3 integers, got {arr.tolist()}")
        arr = np.diag(arr)
    if arr.shape != (3, 3) or not np.allclose(arr, np.rint(arr)):
        raise ValueError(f"supercell must be an int, 3 ints or a 3x3 integer matrix, got {arr}")
    return np.rint(arr).astype(int)


def supercell_field(matrix: np.ndarray) -> list[int]:
    """``PhononResult.supercell``: the diagonal for diagonal matrices, else the 9 entries."""
    m = supercell_matrix(matrix)
    if np.array_equal(m, np.diag(np.diag(m))):
        return [int(x) for x in np.diag(m)]
    return [int(x) for x in m.ravel()]


def new_phonopy(atoms: Atoms, supercell: SupercellSpec) -> Phonopy:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrimitiveMatrixAutoDefaultWarning)
        return Phonopy(
            to_phonopy(atoms),
            supercell_matrix=supercell_matrix(supercell),
            primitive_matrix="auto",
            log_level=0,
        )


def compound_of(atoms: Atoms) -> str:
    return str(atoms.get_chemical_formula("metal", empirical=True))


# --- band path -----------------------------------------------------------------------------------


def _npts_for_segments(lengths: Sequence[float], npoints: int) -> list[int]:
    """Distribute exactly ``npoints`` over segments in proportion to length (each >= 2)."""
    n_seg = len(lengths)
    npoints = max(int(npoints), 2 * n_seg)
    total = float(sum(lengths)) or 1.0
    npts = [max(2, int(round(npoints * length / total))) for length in lengths]
    order = sorted(range(n_seg), key=lambda i: -lengths[i])
    i = 0
    while sum(npts) != npoints:
        idx = order[i % n_seg]
        if sum(npts) > npoints:
            if npts[idx] > 2:
                npts[idx] -= 1
        else:
            npts[idx] += 1
        i += 1
    return npts


def seekpath_path(
    primitive: PhonopyAtoms, npoints: int = DEFAULT_NPOINTS
) -> tuple[list[np.ndarray], list[str], list[tuple[str, str]]]:
    """seekpath's standard path for ``primitive`` in *that cell's* reduced coordinates.

    Returns ``(segments, labels, pairs)``: per-segment q-point arrays, one label per q-point
    (``""`` inside a segment) and the ``(start, end)`` label pairs. seekpath may standardise
    (rotate/re-choose) the primitive cell; its coordinates are mapped back through the
    Cartesian frame, and the mapping is checked to be an integer unimodular basis change.
    """
    import seekpath  # noqa: PLC0415 - optional at import time, required here

    res = seekpath.get_path(
        (np.asarray(primitive.cell), np.asarray(primitive.scaled_positions), primitive.numbers)
    )
    a_seek = np.asarray(res["primitive_lattice"], dtype=float)
    rot = np.asarray(res.get("rotation_matrix", np.eye(3)), dtype=float)
    a_ph = np.asarray(primitive.cell, dtype=float)
    # q (seekpath fractional) -> k_std = q . inv(A_seek).T -> k_in = k_std . rot
    # -> f (phonopy fractional) = k_in . A_ph.T
    mapping = np.linalg.inv(a_seek).T @ rot @ a_ph.T
    if not np.allclose(mapping, np.rint(mapping), atol=1e-4) or not np.isclose(
        abs(np.linalg.det(mapping)), 1.0, atol=1e-4
    ):
        raise ValueError(
            "seekpath's primitive cell is not a unimodular re-basing of phonopy's primitive "
            f"cell (mapping {np.round(mapping, 4).tolist()}); cannot place the band path"
        )
    mapping = np.rint(mapping)
    coords = {k: np.asarray(v, dtype=float) @ mapping for k, v in res["point_coords"].items()}
    b_seek = np.linalg.inv(a_seek).T
    pairs: list[tuple[str, str]] = [(str(s), str(e)) for s, e in res["path"]]
    lengths = [
        float(
            np.linalg.norm((np.asarray(res["point_coords"][e]) - res["point_coords"][s]) @ b_seek)
        )
        for s, e in pairs
    ]
    npts = _npts_for_segments(lengths, npoints)
    segments: list[np.ndarray] = []
    labels: list[str] = []
    for (start, end), n in zip(pairs, npts, strict=True):
        q_s, q_e = coords[start], coords[end]
        seg = np.array([q_s + (q_e - q_s) * i / (n - 1) for i in range(n)])
        segments.append(seg)
        labels += [start] + [""] * (n - 2) + [end]
    return segments, labels, pairs


# --- force constants and spectra ------------------------------------------------------------------


def enforce_acoustic_sum_rule(phonon: Phonopy) -> None:
    """Translational + permutation symmetrisation, translational invariance applied last."""
    phonon.symmetrize_force_constants(level=1, show_drift=False)
    fc = np.array(phonon.force_constants, dtype=float, copy=True)
    set_translational_invariance(fc)
    phonon.force_constants = fc


def band_structure(
    phonon: Phonopy, qpath: str = "seekpath", npoints: int = DEFAULT_NPOINTS
) -> tuple[list[list[float]], list[str], list[list[float]]]:
    """``(qpoints, labels, frequencies_meV)`` along the path (frequencies sorted per q-point)."""
    if qpath != "seekpath":
        raise ValueError(f"qpath must be 'seekpath', got {qpath!r}")
    segments, labels, _ = seekpath_path(phonon.primitive, npoints)
    phonon.run_band_structure([seg.tolist() for seg in segments])
    bs = phonon.band_structure
    assert bs is not None
    qpoints = np.concatenate([np.asarray(q, dtype=float) for q in bs.qpoints])
    freqs = np.concatenate([np.asarray(f, dtype=float) for f in bs.frequencies]) * THZ_TO_MEV
    freqs = np.sort(freqs, axis=1)
    return qpoints.tolist(), labels, freqs.tolist()


def total_dos(
    phonon: Phonopy, mesh: int | Sequence[int] = DEFAULT_MESH
) -> tuple[list[float], list[float]]:
    """Tetrahedron total DOS on a Gamma-centred mesh: ``(frequency_points_meV, dos_per_meV)``."""
    mesh_arr = [int(mesh)] * 3 if isinstance(mesh, int) else [int(m) for m in mesh]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phonon.run_mesh(mesh_arr, is_gamma_center=True)
        phonon.run_total_dos(use_tetrahedron_method=True)
    dos = phonon.total_dos
    assert dos is not None
    freq = np.asarray(dos.frequency_points, dtype=float) * THZ_TO_MEV
    values = np.asarray(dos.dos, dtype=float) / THZ_TO_MEV
    return freq.tolist(), values.tolist()


def imaginary_count(frequencies_meV: Sequence[Sequence[float]]) -> int:
    return int(np.sum(np.asarray(frequencies_meV, dtype=float) < IMAGINARY_TOL_MEV))


def _finish(
    phonon: Phonopy,
    *,
    compound: str,
    source_label: str,
    reference: Reference,
    displacement: float,
    cell_source: CellSource,
    qpath: str,
    npoints: int,
    mesh: int | Sequence[int] | None,
    run_id: str,
) -> PhononResult:
    enforce_acoustic_sum_rule(phonon)
    qpoints, labels, freqs = band_structure(phonon, qpath, npoints)
    dos_x: list[float] | None = None
    dos_y: list[float] | None = None
    if mesh is not None:
        dos_x, dos_y = total_dos(phonon, mesh)
    return PhononResult(
        compound=compound,
        source_label=source_label,
        reference=reference,
        supercell=supercell_field(phonon.supercell_matrix),
        displacement=float(displacement),
        cell_source=cell_source,
        qpath_labels=labels,
        qpoints=qpoints,
        frequencies_meV=freqs,
        dos_meV=dos_x,
        dos=dos_y,
        imaginary_count=imaginary_count(freqs),
        softening_index=None,
        omega_mae_meV=None,
        run_id=run_id,
    )


def compute(
    atoms: Atoms,
    calc: Calculator,
    supercell: SupercellSpec,
    distance: float = DEFAULT_DISTANCE,
    *,
    cell_source: CellSource,
    qpath: str = "seekpath",
    npoints: int = DEFAULT_NPOINTS,
    mesh: int | Sequence[int] | None = DEFAULT_MESH,
    source_label: str = "mace",
    run_id: str = "",
) -> PhononResult:
    """Finite-displacement phonons of ``atoms`` with an ASE calculator (an MLIP).

    ``cell_source="dft"`` means ``atoms`` is the reference (DFT) cell as given;
    ``"model_relaxed"`` means the caller relaxed it with the same model first. The reference
    of the result is ``Reference(code="mace")`` (the model itself); comparison against DFT is
    the evaluate tier's job.
    """
    phonon = new_phonopy(atoms, supercell)
    phonon.generate_displacements(distance=float(distance))
    forces: list[np.ndarray] = []
    for cell in phonon.supercells_with_displacements or []:
        sc = from_phonopy(cell)
        sc.calc = calc
        forces.append(np.asarray(sc.get_forces(), dtype=float))
    phonon.forces = forces
    phonon.produce_force_constants()
    return _finish(
        phonon,
        compound=compound_of(atoms),
        source_label=source_label,
        reference=Reference(code="mace", functional=None, pseudos=None, e0_source=None),
        displacement=float(distance),
        cell_source=cell_source,
        qpath=qpath,
        npoints=npoints,
        mesh=mesh,
        run_id=run_id,
    )


# --- force sets (dft tier schema) -----------------------------------------------------------------


def force_sets_document(
    atoms: Atoms,
    phonon: Phonopy,
    *,
    compound: str | None = None,
    reference: Reference | Mapping[str, Any],
    source_run_id: str = "",
) -> dict[str, Any]:
    """Serialise a phonopy displacement dataset (with forces) in the ``force_sets.v1`` schema."""
    dataset = cast(dict[str, Any], phonon.dataset)
    ref = reference if isinstance(reference, Reference) else Reference.model_validate(reference)
    return {
        "schema": FORCE_SETS_SCHEMA,
        "compound": compound or compound_of(atoms),
        "unitcell": {
            "numbers": [int(z) for z in atoms.numbers],
            "positions": np.asarray(atoms.get_positions(), dtype=float).tolist(),
            "cell": np.asarray(atoms.cell[:], dtype=float).tolist(),
        },
        "supercell_matrix": supercell_matrix(phonon.supercell_matrix).tolist(),
        "displacement_distance": float(np.linalg.norm(dataset["first_atoms"][0]["displacement"])),
        "dataset": {
            "natom": int(dataset["natom"]),
            "first_atoms": [
                {
                    "number": int(d["number"]),
                    "displacement": np.asarray(d["displacement"], dtype=float).tolist(),
                    "forces": np.asarray(d["forces"], dtype=float).tolist(),
                }
                for d in dataset["first_atoms"]
            ],
        },
        "reference": ref.model_dump(mode="json"),
        "units": dict(FORCE_SETS_UNITS),
        "source_run_id": source_run_id,
    }


def write_force_sets(doc: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return p


def _load_force_sets(force_sets_json: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(force_sets_json, Mapping):
        doc = dict(force_sets_json)
    else:
        doc = json.loads(Path(force_sets_json).read_text(encoding="utf-8"))
    if doc.get("schema") != FORCE_SETS_SCHEMA:
        raise ValueError(f"force sets schema {doc.get('schema')!r} is not {FORCE_SETS_SCHEMA!r}")
    for key in ("unitcell", "supercell_matrix", "displacement_distance", "dataset", "reference"):
        if key not in doc:
            raise ValueError(f"force sets document lacks {key!r}")
    return doc


def _unitcell_from_doc(doc: Mapping[str, Any]) -> Atoms:
    uc = doc["unitcell"]
    return Atoms(
        numbers=[int(z) for z in uc["numbers"]],
        positions=np.asarray(uc["positions"], dtype=float),
        cell=np.asarray(uc["cell"], dtype=float),
        pbc=True,
    )


def from_force_sets(
    atoms: Atoms | None,
    force_sets_json: str | Path | Mapping[str, Any],
    supercell: SupercellSpec | None = None,
    *,
    qpath: str = "seekpath",
    npoints: int = DEFAULT_NPOINTS,
    mesh: int | Sequence[int] | None = DEFAULT_MESH,
    cell_source: CellSource = "dft",
    run_id: str | None = None,
) -> PhononResult:
    """Phonons from a reference displacement dataset (``b20mlip.force_sets.v1``, QE by default).

    ``atoms`` may be ``None`` (the unit cell stored in the file is used); when given it must
    match that cell. ``supercell`` may be ``None`` (the stored matrix is used) or must equal it.
    """
    doc = _load_force_sets(force_sets_json)
    stored = _unitcell_from_doc(doc)
    if atoms is None:
        atoms = stored
    elif list(atoms.numbers) != list(stored.numbers) or not np.allclose(
        atoms.cell[:], stored.cell[:], atol=1e-6
    ):
        raise ValueError("atoms do not match the unit cell stored in the force sets file")
    matrix = supercell_matrix(doc["supercell_matrix"])
    if supercell is not None and not np.array_equal(supercell_matrix(supercell), matrix):
        raise ValueError(
            f"supercell {supercell_matrix(supercell).tolist()} differs from the stored "
            f"{matrix.tolist()}"
        )
    phonon = new_phonopy(atoms, matrix)
    dataset = doc["dataset"]
    if int(dataset["natom"]) != len(phonon.supercell):
        raise ValueError(
            f"dataset natom {dataset['natom']} != supercell atoms {len(phonon.supercell)}"
        )
    phonon.dataset = {
        "natom": int(dataset["natom"]),
        "first_atoms": [
            {
                "number": int(d["number"]),
                "displacement": np.asarray(d["displacement"], dtype=float),
                "forces": np.asarray(d["forces"], dtype=float),
            }
            for d in dataset["first_atoms"]
        ],
    }
    phonon.produce_force_constants()
    reference = Reference.model_validate(doc["reference"])
    label = doc.get("source_label") or f"{reference.code}:{reference.functional or 'n/a'}"
    return _finish(
        phonon,
        compound=str(doc.get("compound") or compound_of(atoms)),
        source_label=str(label),
        reference=reference,
        displacement=float(doc["displacement_distance"]),
        cell_source=cell_source,
        qpath=qpath,
        npoints=npoints,
        mesh=mesh,
        run_id=run_id if run_id is not None else str(doc.get("source_run_id") or ""),
    )


# --- JSON -----------------------------------------------------------------------------------------


def to_json(result: PhononResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return p


def from_json(path: str | Path) -> PhononResult:
    return PhononResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def gamma_frequencies(result: PhononResult) -> list[float]:
    """Frequencies (meV) at the first Gamma point of the path (``[]`` if the path has none)."""
    for label, freqs in zip(result.qpath_labels, result.frequencies_meV, strict=True):
        if label == "GAMMA":
            return list(freqs)
    return []


__all__ = [
    "DEFAULT_DISTANCE",
    "DEFAULT_MESH",
    "DEFAULT_NPOINTS",
    "FORCE_SETS_SCHEMA",
    "IMAGINARY_TOL_MEV",
    "THZ_TO_MEV",
    "band_structure",
    "compound_of",
    "compute",
    "enforce_acoustic_sum_rule",
    "force_sets_document",
    "from_force_sets",
    "from_json",
    "from_phonopy",
    "gamma_frequencies",
    "imaginary_count",
    "new_phonopy",
    "seekpath_path",
    "supercell_field",
    "supercell_matrix",
    "to_json",
    "to_phonopy",
    "total_dos",
    "write_force_sets",
]
