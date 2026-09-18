"""harmonic.compute (tiny MACE on FeSi), from_force_sets (EMT Al), ASR, path, JSON round trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from scipy.spatial.transform import Rotation

from b20mlip.models import PhononResult, Reference
from b20mlip.phonons import harmonic as h

QE_REF = Reference(code="qe", functional="PBE", pseudos="SSSP-efficiency-1.3", e0_source=None)


def _al_document(al_atoms: Atoms, emt: EMT, **kw: Any) -> dict[str, Any]:
    phonon = h.new_phonopy(al_atoms, 2)
    phonon.generate_displacements(distance=0.03)
    forces = []
    for cell in phonon.supercells_with_displacements:
        sc = h.from_phonopy(cell)
        sc.calc = emt
        forces.append(sc.get_forces())
    phonon.forces = forces
    return h.force_sets_document(al_atoms, phonon, reference=QE_REF, source_run_id="qe-run-1", **kw)


# --- compute with the tiny model ------------------------------------------------------------------


def test_compute_fesi_with_tiny_model(fesi_atoms: Atoms, tiny_calc: Any, tmp_path: Path) -> None:
    result = h.compute(
        fesi_atoms, tiny_calc, 1, 0.03, cell_source="dft", npoints=24, mesh=4, run_id="r1"
    )
    assert isinstance(result, PhononResult)
    assert result.compound == "FeSi" and result.supercell == [1, 1, 1]
    assert result.displacement == 0.03 and result.cell_source == "dft" and result.run_id == "r1"
    assert result.reference == Reference(code="mace", functional=None, pseudos=None, e0_source=None)
    assert result.source_label == "mace"
    assert len(result.qpoints) == len(result.qpath_labels) == len(result.frequencies_meV) == 24
    assert all(len(branches) == 24 for branches in result.frequencies_meV)  # 3N = 24
    assert result.qpath_labels[0] == "GAMMA" and result.qpoints[0] == [0.0, 0.0, 0.0]
    gamma = np.asarray(h.gamma_frequencies(result))
    assert np.sum(np.abs(gamma) < 0.1) >= 3  # acoustic sum rule after enforcement
    assert all(np.all(np.diff(f) >= -1e-9) for f in result.frequencies_meV)  # sorted branches
    flat = np.asarray(result.frequencies_meV)
    assert result.imaginary_count == int(np.sum(flat < -0.4))
    assert result.dos_meV is not None and result.dos is not None
    assert len(result.dos_meV) == len(result.dos) > 0
    assert result.softening_index is None and result.omega_mae_meV is None

    path = h.to_json(result, tmp_path / "phonons" / "fesi.json")
    assert h.from_json(path) == result
    assert json.loads(path.read_text())["compound"] == "FeSi"


# --- from_force_sets (dft tier schema) ------------------------------------------------------------


def test_force_sets_document_matches_the_dft_schema(al_atoms: Atoms, emt: EMT) -> None:
    doc = _al_document(al_atoms, emt)
    assert set(doc) == {
        "schema", "compound", "unitcell", "supercell_matrix", "displacement_distance",
        "dataset", "reference", "units", "source_run_id",
    }  # fmt: skip
    assert doc["schema"] == "b20mlip.force_sets.v1" and doc["compound"] == "Al"
    assert set(doc["unitcell"]) == {"numbers", "positions", "cell"}
    assert doc["supercell_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert doc["displacement_distance"] == pytest.approx(0.03)
    assert doc["dataset"]["natom"] == 32 and len(doc["dataset"]["first_atoms"]) == 1
    first = doc["dataset"]["first_atoms"][0]
    assert set(first) == {"number", "displacement", "forces"}
    assert np.asarray(first["forces"]).shape == (32, 3) and len(first["displacement"]) == 3
    assert doc["reference"] == {
        "code": "qe", "functional": "PBE", "pseudos": "SSSP-efficiency-1.3",
        "e0_source": None, "cross_functional": False,
    }  # fmt: skip
    assert doc["source_run_id"] == "qe-run-1"
    json.dumps(doc)  # plain JSON types only


def test_from_force_sets_al_emt(al_atoms: Atoms, emt: EMT, tmp_path: Path) -> None:
    doc = _al_document(al_atoms, emt)
    path = h.write_force_sets(doc, tmp_path / "force_sets.json")
    result = h.from_force_sets(None, path, None, mesh=8)
    assert result.compound == "Al" and result.supercell == [2, 2, 2]
    assert result.reference == QE_REF and result.source_label == "qe:PBE"
    assert result.run_id == "qe-run-1" and result.cell_source == "dft"
    assert len(result.qpoints) == 100 and all(len(f) == 3 for f in result.frequencies_meV)
    assert result.imaginary_count == 0  # fcc Al under EMT is dynamically stable
    top = max(max(f) for f in result.frequencies_meV)
    assert 25.0 < top < 45.0  # EMT Al tops out near 33 meV (experiment ~40 meV)
    assert max(abs(x) for x in h.gamma_frequencies(result)) < 0.1
    assert np.trapezoid(result.dos, result.dos_meV) == pytest.approx(3.0, rel=0.02)
    labels = [lab for lab in result.qpath_labels if lab]
    assert labels[:2] == ["GAMMA", "X"] and "L" in labels

    # same spectrum as compute() with the same calculator; explicit inputs are checked
    direct = h.compute(al_atoms, emt, 2, 0.03, cell_source="dft", mesh=8)
    np.testing.assert_allclose(direct.frequencies_meV, result.frequencies_meV, atol=1e-6)
    same = h.from_force_sets(al_atoms, doc, [2, 2, 2], mesh=None, run_id="override")
    assert same.run_id == "override" and same.dos is None
    np.testing.assert_allclose(same.frequencies_meV, result.frequencies_meV, atol=1e-9)
    with pytest.raises(ValueError, match="differs from the stored"):
        h.from_force_sets(None, doc, 3)
    with pytest.raises(ValueError, match="do not match the unit cell"):
        h.from_force_sets(al_atoms.repeat((2, 1, 1)), doc)
    with pytest.raises(ValueError, match="schema"):
        h.from_force_sets(None, {**doc, "schema": "phonopy"})
    with pytest.raises(ValueError, match="lacks 'dataset'"):
        h.from_force_sets(None, {k: v for k, v in doc.items() if k != "dataset"})
    bad = json.loads(json.dumps(doc))
    bad["dataset"]["natom"] = 8
    with pytest.raises(ValueError, match="natom"):
        h.from_force_sets(None, bad)


def test_rotated_cell_gives_the_same_spectrum(al_atoms: Atoms, emt: EMT) -> None:
    reference = h.compute(al_atoms, emt, 2, 0.03, cell_source="dft", mesh=None)
    rot = Rotation.from_euler("zyx", [30, 20, 10], degrees=True).as_matrix()
    rotated = al_atoms.copy()
    rotated.set_cell(al_atoms.cell[:] @ rot.T, scale_atoms=True)
    result = h.compute(rotated, emt, 2, 0.03, cell_source="model_relaxed", mesh=None)
    assert result.cell_source == "model_relaxed"
    np.testing.assert_allclose(result.frequencies_meV, reference.frequencies_meV, atol=1e-4)
    assert result.qpath_labels == reference.qpath_labels


def test_acoustic_sum_rule_holds_even_for_garbage_forces(al_atoms: Atoms) -> None:
    phonon = h.new_phonopy(al_atoms, 2)
    phonon.generate_displacements(distance=0.03)
    rng = np.random.default_rng(0)
    phonon.forces = [rng.normal(0.0, 0.3, (32, 3)) for _ in phonon.supercells_with_displacements]
    phonon.produce_force_constants()
    h.enforce_acoustic_sum_rule(phonon)
    fc = np.asarray(phonon.force_constants)
    np.testing.assert_allclose(fc.sum(axis=1), 0.0, atol=1e-10)  # translational invariance
    _, labels, freqs = h.band_structure(phonon, "seekpath", 10)
    gamma = np.asarray(freqs[labels.index("GAMMA")])
    assert np.sum(np.abs(gamma) < 1e-3) >= 3
    with pytest.raises(ValueError, match="qpath"):
        h.band_structure(phonon, "auto", 10)


# --- helpers --------------------------------------------------------------------------------------


def test_seekpath_path_counts(al_atoms: Atoms, fesi_atoms: Atoms) -> None:
    for atoms, n_seg in ((al_atoms, 6), (fesi_atoms, 7)):
        primitive = h.new_phonopy(atoms, 1).primitive
        for npoints in (14, 37, 100):
            segments, labels, pairs = h.seekpath_path(primitive, npoints)
            assert len(pairs) == n_seg and len(segments) == n_seg
            assert sum(len(s) for s in segments) == len(labels) == npoints
            assert all(len(s) >= 2 for s in segments)
            assert labels[0] == pairs[0][0] == "GAMMA" and labels[-1] == pairs[-1][1]
    assert h._npts_for_segments([1.0, 1.0, 2.0], 3) == [2, 2, 2]  # never below 2 per segment
    assert h._npts_for_segments([1.0, 3.0], 12) == [3, 9]
    assert sum(h._npts_for_segments([0.3, 0.3, 0.4, 5.0], 25)) == 25


def test_supercell_helpers() -> None:
    np.testing.assert_array_equal(h.supercell_matrix(2), 2 * np.eye(3, dtype=int))
    np.testing.assert_array_equal(h.supercell_matrix([1, 2, 3]), np.diag([1, 2, 3]))
    off = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    np.testing.assert_array_equal(h.supercell_matrix(off), np.asarray(off))
    assert h.supercell_field(np.diag([2, 2, 2])) == [2, 2, 2]
    assert h.supercell_field(np.asarray(off)) == [1, 1, 0, -1, 1, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="3 integers"):
        h.supercell_matrix([2, 2])
    with pytest.raises(ValueError, match="integer"):
        h.supercell_matrix([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_imaginary_count_and_units() -> None:
    assert h.THZ_TO_MEV == pytest.approx(4.135667696)
    assert h.imaginary_count([[-1.0, -0.3, 0.0], [-0.5, 2.0, 3.0]]) == 2
    assert h.imaginary_count([[0.0, 1.0]]) == 0
    result = PhononResult(
        compound="X", source_label="s", reference=QE_REF, supercell=[1, 1, 1], displacement=0.01,
        cell_source="dft", qpath_labels=["X", "GAMMA"], qpoints=[[0.5, 0, 0], [0, 0, 0]],
        frequencies_meV=[[1.0, 2.0], [0.0, 3.0]], dos_meV=None, dos=None, imaginary_count=0,
        softening_index=None, omega_mae_meV=None, run_id="",
    )  # fmt: skip
    assert h.gamma_frequencies(result) == [0.0, 3.0]
    assert h.gamma_frequencies(result.model_copy(update={"qpath_labels": ["X", "M"]})) == []
    assert h.compound_of(Atoms("Fe4Si4")) == "FeSi"
