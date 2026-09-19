"""NEB on a Lennard-Jones vacancy hop (fcc 2x2x2, one vacancy): barrier > 0, converged, images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read

from b20mlip.sampling import neb, vacancy

from .conftest import LJHop, lj_calc


def test_lj_vacancy_hop_barrier(lj_hop: LJHop, tmp_path: Path) -> None:
    traj = tmp_path / "band" / "path.traj"
    result = neb.barrier(
        None, lj_hop.initial, lj_hop.final, images=7, calc_factory=lj_calc, path_traj=traj
    )
    assert result["converged"] and result["neb_converged"]
    assert result["relax"]["initial"]["converged"] and result["relax"]["final"]["converged"]
    assert result["n_images"] == 7 and len(result["images"]) == 7
    assert result["E_a_eV"] > 0.5 and result["E_a_reverse_eV"] > 0.5
    assert result["E_a_eV"] == pytest.approx(result["E_a_reverse_eV"], abs=0.05)  # symmetric hop
    assert result["dE_eV"] == pytest.approx(0.0, abs=0.05)
    assert result["top_image"] == 3 and result["images_relative_eV"][0] == 0.0
    energies = np.asarray(result["images"])
    assert energies.argmax() == 3 and result["E_a_eV"] == pytest.approx(
        energies.max() - energies[0]
    )
    assert 0 < result["steps"] <= 300 and result["fmax_eVA"] <= 0.05
    assert result["method"] == neb.METHOD and result["climb"] and result["spring_eVA2"] == 0.1
    assert result["path_traj"] == str(traj) and traj.is_file()
    band = ase_read(str(traj), index=":")
    assert len(band) == 7 and all(len(image) == 31 for image in band)
    assert [image.get_potential_energy() for image in band] == pytest.approx(result["images"])
    assert band[3].get_forces().shape == (31, 3)
    cv = vacancy.HopCV(lj_hop.info)
    along = [cv.value(image) for image in result["band"]]
    assert along[0] == pytest.approx(0.0, abs=0.05) and along[-1] == pytest.approx(1.0, abs=0.05)
    assert all(np.diff(along) > 0)  # the hopping atom advances monotonically along the band
    assert result["wall_seconds"] > 0


def test_step_budget_and_argument_errors(lj_hop: LJHop) -> None:
    capped = neb.barrier(
        None, lj_hop.initial, lj_hop.final, images=5, calc_factory=lj_calc, neb_steps=2,
        relax_steps=1,
    )  # fmt: skip
    assert not capped["converged"] and capped["steps"] == 2 and capped["n_images"] == 5
    assert capped["path_traj"] is None and capped["E_a_eV"] > 0
    with pytest.raises(ValueError, match="at least 3"):
        neb.barrier(None, lj_hop.initial, lj_hop.final, images=2, calc_factory=lj_calc)
    with pytest.raises(ValueError, match="same atoms"):
        neb.barrier(None, lj_hop.initial, lj_hop.initial[:-1], calc_factory=lj_calc)
    with pytest.raises(ValueError, match="model_path or a calc_factory"):
        neb.barrier(None, lj_hop.initial, lj_hop.final)


def test_relax_and_read_endpoints(lj_hop: LJHop, tmp_path: Path) -> None:
    from ase.io import write as ase_write

    atoms = lj_hop.initial.copy()
    atoms.rattle(0.02, seed=1)
    atoms.calc = lj_calc()
    info = neb.relax(atoms, fmax=0.05, steps=200)
    assert info["converged"] and info["steps"] > 0 and info["fmax_eVA"] <= 0.05
    ase_write(str(tmp_path / "a.extxyz"), lj_hop.initial, format="extxyz")
    ase_write(str(tmp_path / "b.extxyz"), lj_hop.final, format="extxyz")
    first, last = neb.read_endpoints(tmp_path / "a.extxyz", tmp_path / "b.extxyz")
    assert len(first) == len(last) == 31 and first.calc is None and first.info == {}
    assert np.allclose(last.positions, lj_hop.final.positions)
