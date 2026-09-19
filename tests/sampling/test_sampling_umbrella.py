"""HarmonicBias energy/force consistency (3-atom LJ + hop CV); run_windows with the tiny MACE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from b20mlip.sampling import umbrella, vacancy

from .conftest import lj_calc


def numerical_forces(atoms: Atoms, h: float = 1e-5) -> np.ndarray:
    out = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for d in range(3):
            pos = atoms.get_positions()
            pos[i, d] += h
            atoms.set_positions(pos)
            plus = atoms.get_potential_energy()
            pos[i, d] -= 2 * h
            atoms.set_positions(pos)
            minus = atoms.get_potential_energy()
            pos[i, d] += h
            atoms.set_positions(pos)
            out[i, d] = -(plus - minus) / (2 * h)
    return out


def test_harmonic_bias_matches_numerical_derivatives() -> None:
    atoms = Atoms(
        "Ar3", positions=[[0, 0, 0], [1.2, 0.1, 0.0], [0.5, 1.1, 0.2]], cell=[8, 8, 8], pbc=True
    )
    info = {"hop_index": 1, "initial_site": [1.0, 0.0, 0.0], "hop_vector": [0.9, 0.3, 0.1]}
    cv = vacancy.HopCV(info)
    base = lj_calc()
    bias = umbrella.HarmonicBias(base, cv, k=2.5, center=0.3)
    atoms.calc = bias
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces().copy()
    unbiased = Atoms("Ar3", positions=atoms.positions, cell=atoms.cell, pbc=True)
    unbiased.calc = lj_calc()
    value = cv.value(atoms)
    assert bias.cv_value == pytest.approx(value) and 0 < value < 1
    assert bias.bias_energy == pytest.approx(0.5 * 2.5 * (value - 0.3) ** 2)
    assert energy == pytest.approx(unbiased.get_potential_energy() + bias.bias_energy)
    assert bias.unbiased_energy == pytest.approx(unbiased.get_potential_energy())
    expected = unbiased.get_forces() - 2.5 * (value - 0.3) * cv.gradient(atoms)
    assert np.allclose(forces, expected, atol=1e-10)
    assert np.abs(forces - numerical_forces(atoms)).max() < 1e-6
    assert np.allclose(forces[[0, 2]], unbiased.get_forces()[[0, 2]])  # only atom 1 is biased
    assert atoms.get_potential_energy() == pytest.approx(energy)  # positions restored
    assert bias.results["free_energy"] == pytest.approx(energy)
    value2, bias_energy, bias_forces = bias.bias(atoms)
    assert value2 == pytest.approx(value) and bias_forces.shape == (3, 3)
    assert bias_energy == pytest.approx(bias.bias_energy)


def test_run_windows_with_tiny_mace_resumes(
    fesi_atoms: Atoms, tiny_mace: Any, tmp_path: Path
) -> None:
    initial, final, info = vacancy.vacancy_hop_endpoints(fesi_atoms, (1, 1, 1), "Si")
    cv = vacancy.HopCV(info)
    out = tmp_path / "windows"
    logs: list[str] = []
    files = umbrella.run_windows(
        tiny_mace.model_path, initial, cv, [0.0, 1.0], k=5.0 * cv.scale_A**2, ps=0.04, T=300.0,
        timestep_fs=2.0, seed=4, out_dir=out, calc_factory=tiny_mace.calculator, log=logs.append,
        log_every=2, index_extra={"compound": "FeSi"},
    )  # fmt: skip
    assert [Path(f).name for f in files] == ["window_00.npz", "window_01.npz"]
    assert all(Path(f).is_file() for f in files) and any("2/2 windows done" in m for m in logs)
    index = json.loads((out / umbrella.INDEX_NAME).read_text())
    assert index["schema"] == umbrella.INDEX_SCHEMA and index["files"] == files
    assert index["n_steps"] == 20 and index["centers"] == [0.0, 1.0] and index["compound"] == "FeSi"
    assert index["n_atoms"] == 7 and index["model"] == str(tiny_mace.model_path)
    data = umbrella.load_window(files[1])
    assert data["complete"] and data["n_steps"] == 20 and data["n_samples"] == 21
    assert data["n_equil"] == 4 and data["center"] == 1.0 and data["T"] == 300.0
    assert data["cv"].shape == (21,) and data["epot_eV"].shape == (21,)
    assert (
        data["positions"].shape == (7, 3) and data["numbers"].tolist() == initial.numbers.tolist()
    )
    assert np.all(np.isfinite(data["cv"])) and data["time_fs"][-1] == pytest.approx(40.0)
    assert data["seed"] == 5 and data["dt_fs"] == 2.0

    # resume: complete windows are skipped, nothing is recomputed
    logs.clear()
    again = umbrella.run_windows(
        tiny_mace.model_path, initial, cv, [0.0, 1.0], k=5.0 * cv.scale_A**2, ps=0.04, T=300.0,
        seed=4, out_dir=out, calc_factory=tiny_mace.calculator, log=logs.append,
    )  # fmt: skip
    assert again == files and all("skipping" in m for m in logs) and len(logs) == 2
    assert umbrella.load_window(files[1])["cv"].tolist() == data["cv"].tolist()
    # a window run with other settings is not "complete" for the new run
    assert not umbrella.window_complete(Path(files[1]), 1.0, n_steps=40)
    assert not umbrella.window_complete(Path(files[1]), 0.9, n_steps=20)
    assert not umbrella.window_complete(out / "absent.npz", 1.0, n_steps=20)


def test_run_windows_interp_and_argument_errors(tmp_path: Path) -> None:
    from ase.build import bulk

    prim = bulk("Ar", "fcc", a=1.55, cubic=True)
    initial, final, info = vacancy.vacancy_hop_endpoints(prim, (2, 2, 2), "Ar")
    cv = vacancy.HopCV(info)
    files = umbrella.run_windows(
        None, initial, cv, [0.25, 0.75], k=50.0, ps=0.02, T=30.0, timestep_fs=2.0, seed=0,
        out_dir=tmp_path / "interp", calc_factory=lj_calc, init="interp",
        endpoints=(initial, final), sample_every=2,
    )  # fmt: skip
    assert len(files) == 2
    first = umbrella.load_window(files[0])
    assert first["n_samples"] == 6 and first["sample_every"] == 2  # 10 steps, every 2nd + start
    assert abs(first["cv"][0] - 0.25) < 0.05  # starts at the interpolated structure
    index, index_path = umbrella.read_index(tmp_path / "interp")
    assert index["init"] == "interp" and index_path.name == umbrella.INDEX_NAME
    same, _ = umbrella.read_index(index_path)
    assert same["files"] == index["files"]
    kw: dict[str, Any] = dict(
        k=1.0, ps=0.01, T=30.0, out_dir=tmp_path / "err", calc_factory=lj_calc
    )
    with pytest.raises(ValueError, match="init must be"):
        umbrella.run_windows(None, initial, cv, [0.0], init="random", **kw)
    with pytest.raises(ValueError, match="endpoints"):
        umbrella.run_windows(None, initial, cv, [0.0], init="interp", **kw)
    with pytest.raises(ValueError, match="sample_every"):
        umbrella.run_windows(None, initial, cv, [0.0], sample_every=0, **kw)
    with pytest.raises(ValueError, match="model_path or a calc_factory"):
        umbrella.run_windows(None, initial, cv, [0.0], k=1.0, ps=0.01, T=30.0, out_dir=tmp_path)
    with pytest.raises(ValueError, match="less than one step"):
        umbrella.run_windows(None, initial, cv, [0.0], k=1.0, ps=1e-6, T=30.0, out_dir=tmp_path,
                             calc_factory=lj_calc)  # fmt: skip
    with pytest.raises(FileNotFoundError):
        umbrella.read_index(tmp_path / "nowhere")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError, match="schema"):
        umbrella.read_index(bad)
