"""vacancy_hop_endpoints (deterministic, 8 -> 63 atoms), the hop CV (0/1 at the end states)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from b20mlip.sampling import vacancy as v


def test_endpoints_are_deterministic_and_span_the_cv(fesi_atoms: Atoms) -> None:
    initial, final, info = v.vacancy_hop_endpoints(fesi_atoms, (2, 2, 2), "Si")
    again, _, info2 = v.vacancy_hop_endpoints(fesi_atoms, (2, 2, 2), "Si")
    assert len(fesi_atoms) == 8 and len(initial) == len(final) == 63
    assert np.array_equal(initial.numbers, final.numbers)
    assert np.allclose(initial.positions, again.positions) and info == info2
    assert info["species"] == "Si" and info["compound"] == "FeSi" and info["supercell"] == [2, 2, 2]
    assert info["n_atoms"] == 63 and initial.get_chemical_symbols()[info["hop_index"]] == "Si"
    assert 2.0 < info["hop_distance"] < 3.5  # nearest Si-Si distance in B20 FeSi
    assert np.isclose(np.linalg.norm(info["hop_vector"]), info["hop_distance"])
    # only the hopping atom differs between the two end states
    moved = np.flatnonzero(np.linalg.norm(final.positions - initial.positions, axis=1) > 1e-9)
    assert moved.tolist() == [info["hop_index"]]
    cv = v.HopCV(info)
    assert v.hop_cv(initial, info) == pytest.approx(0.0, abs=1e-12)
    assert v.hop_cv(final, info) == pytest.approx(1.0, abs=1e-12)
    assert cv(v.interpolate_endpoints(initial, final, 0.5))[0] == pytest.approx(0.5)
    assert cv.scale_A == pytest.approx(info["hop_distance"])
    grad = cv.gradient(initial)
    assert grad.shape == (63, 3) and np.count_nonzero(grad.any(axis=1)) == 63
    assert np.abs(grad.sum(axis=0)).max() < 1e-12  # measured in the frame of the other atoms
    assert np.allclose(grad[info["hop_index"]], np.asarray(info["hop_vector"]) / cv.scale_A**2)
    assert np.allclose(grad[info["hop_index"]] @ np.asarray(info["hop_vector"]), 1.0)


def test_cv_is_minimum_image_safe(fesi_atoms: Atoms) -> None:
    initial, final, info = v.vacancy_hop_endpoints(fesi_atoms, (1, 1, 1), "Si")
    cv = v.HopCV(info)
    wrapped = final.copy()
    wrapped.positions[info["hop_index"]] += wrapped.cell[:].sum(axis=0)  # a lattice translation
    assert cv.value(wrapped) == pytest.approx(1.0, abs=1e-9)
    tiny = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(IndexError):
        cv.value(tiny)


def test_species_aliases_rng_and_errors(fesi_atoms: Atoms) -> None:
    _, _, tm = v.vacancy_hop_endpoints(fesi_atoms, 2, "TM")
    _, _, x = v.vacancy_hop_endpoints(fesi_atoms, [2, 2, 2], "X")
    assert tm["species"] == "Fe" and x["species"] == "Si"
    rng = np.random.default_rng(7)
    _, _, picked = v.vacancy_hop_endpoints(fesi_atoms, (2, 2, 2), "Si", rng=rng)
    assert picked["species"] == "Si" and picked["n_atoms"] == 63
    assert v.resolve_species(fesi_atoms, "Fe") == "Fe"
    with pytest.raises(ValueError, match="exactly one element"):
        v.resolve_species(fesi_atoms, "Ge")
    with pytest.raises(ValueError, match="supercell"):
        v.supercell_tuple((2, 2))
    with pytest.raises(ValueError, match="at least two"):
        v.vacancy_hop_endpoints(Atoms("Si", cell=[3, 3, 3], pbc=True), 1, "Si")
    with pytest.raises(ValueError, match="no tabulated"):
        v.b20_cell("NiSi")
    cell = v.b20_cell("MnSi")
    assert len(cell) == 8 and v.compound_of(cell) == "MnSi"
    with pytest.raises(ValueError, match="non-zero"):
        v.HopCV({"hop_index": 0, "initial_site": [0, 0, 0], "hop_vector": [0, 0, 0]})


def test_hop_info_from_endpoints(fesi_atoms: Atoms) -> None:
    initial, final, info = v.vacancy_hop_endpoints(fesi_atoms, (2, 2, 2), "Si")
    derived = v.hop_info_from_endpoints(initial, final, base={"compound": "FeSi"})
    assert derived["hop_index"] == info["hop_index"] and derived["species"] == "Si"
    assert derived["hop_distance"] == pytest.approx(info["hop_distance"])
    assert np.allclose(derived["hop_vector"], info["hop_vector"])
    assert derived["max_other_displacement_A"] == 0.0 and derived["compound"] == "FeSi"
    with pytest.raises(ValueError, match="coincide"):
        v.hop_info_from_endpoints(initial, initial)
    with pytest.raises(ValueError, match="same atoms"):
        v.hop_info_from_endpoints(initial, fesi_atoms)


def test_hop_cv_is_invariant_to_rigid_translation_of_the_crystal() -> None:
    """Regression (2026-09-19): under a Langevin thermostat the bias force translated the whole
    crystal instead of moving the atom over the saddle; the CV must not change when every atom
    (hopping atom included) is shifted rigidly, and its gradient must sum to zero."""
    from ase.spacegroup import crystal

    from b20mlip.sampling.vacancy import HopCV, vacancy_hop_endpoints

    prim = crystal(
        ["Fe", "Si"],
        basis=[(0.137, 0.137, 0.137), (0.842, 0.842, 0.842)],
        spacegroup=198,
        cellpar=[4.48, 4.48, 4.48, 90, 90, 90],
    )
    initial, final, info = vacancy_hop_endpoints(prim, supercell=(2, 2, 2), species="Si")
    cv = HopCV(info)
    v0, g0 = cv(initial)
    assert abs(v0) < 1e-9 and abs(cv.value(final) - 1.0) < 1e-9
    assert np.abs(g0.sum(axis=0)).max() < 1e-12  # no net force from the bias
    shifted = initial.copy()
    shifted.positions += np.array([0.7, -0.3, 1.1])  # rigid translation of everything
    assert abs(cv.value(shifted) - v0) < 1e-9
    # moving only the hopping atom half-way along the hop gives 0.5 (up to the tiny centre shift)
    half = initial.copy()
    half.positions[info["hop_index"]] += 0.5 * np.asarray(info["hop_vector"])
    assert abs(cv.value(half) - 0.5) < 0.02
