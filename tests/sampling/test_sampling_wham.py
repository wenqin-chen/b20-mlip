"""free_energy on the analytic double well: MBAR and WHAM recover dF within 0.02 eV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from b20mlip.config import Settings
from b20mlip.provenance import run_stage
from b20mlip.sampling import umbrella, wham

from .conftest import DW_WINDOWS, DoubleWell


@pytest.mark.parametrize("method", ["mbar", "wham"])
def test_double_well_recovers_the_analytic_free_energy(
    double_well: DoubleWell, method: str
) -> None:
    result = wham.free_energy(double_well.windows, method=method)
    assert result["method"] == method and result["n_windows"] == DW_WINDOWS
    assert result["dF_eV"] == pytest.approx(double_well.dF_true, abs=0.02)
    assert result["dF_barrier_point_eV"] == pytest.approx(double_well.barrier_true, abs=0.03)
    assert result["cv_ts"] == pytest.approx(double_well.ts, abs=0.1)
    assert result["cv_initial"] == pytest.approx(double_well.minima[0], abs=0.1)
    assert result["cv_final"] == pytest.approx(double_well.minima[1], abs=0.1)
    assert 0 < result["dF_block_err_eV"] < 0.02 and result["n_blocks_valid"] == 5
    assert len(result["dF_blocks_eV"]) == 5 and result["barrier_found"]
    lo, hi = result["dF_ci95_eV"]
    assert lo < result["dF_eV"] < hi and hi - lo < 0.08
    assert result["overlap_ok"] and result["overlap_min"] > 0.05 and len(result["overlaps"]) == 11
    assert result["n_samples"] == sum(result["n_samples_per_window"])
    assert all(n == int(round(0.8 * len(w["cv"]))) for n, w in zip(
        result["n_samples_per_window"], double_well.windows, strict=True
    ))  # fmt: skip
    assert result["estimator_converged"] and result["T"] == pytest.approx(double_well.T)
    assert result["kT_eV"] == pytest.approx(double_well.kT, rel=1e-6)
    grid, F = np.asarray(result["cv_grid"]), np.asarray(result["F_eV"])
    assert len(grid) == len(F) == result["n_bins"] >= 50
    assert np.nanmin(F) == pytest.approx(0.0) and np.isfinite(F).sum() > 40
    assert result["dF_barrier_eV"] > result["dF_barrier_point_eV"] - 0.05
    assert result["k"] == [double_well.k] * DW_WINDOWS and len(result["f_k"]) == DW_WINDOWS
    if method == "wham":
        assert result["iterations"] > 1


def test_mbar_and_wham_agree_and_match_pymbar_fes(double_well: DoubleWell) -> None:
    mbar = wham.free_energy(double_well.windows, method="mbar", n_bins=80)
    plain = wham.free_energy(double_well.windows, method="wham", n_bins=80)
    assert mbar["dF_eV"] == pytest.approx(plain["dF_eV"], abs=0.01)
    both = np.isfinite(mbar["F_eV"]) & np.isfinite(plain["F_eV"])
    assert np.abs(np.asarray(mbar["F_eV"])[both] - np.asarray(plain["F_eV"])[both]).max() < 0.02
    # the MBAR histogram profile equals pymbar's own FES estimator on populated bins
    from pymbar import FES

    wins = wham.as_windows(double_well.windows)
    kT = wham.KB_EV * wins[0].T
    edges = np.asarray(mbar["bin_edges"])
    x_n = np.concatenate([w.cv for w in wins])
    N_k = np.array([w.n for w in wins])
    centers = np.array([w.center for w in wins])
    u_kn = 0.5 * double_well.k * (x_n[None, :] - centers[:, None]) ** 2 / kT
    fes = FES(u_kn, N_k)
    fes.generate_fes(
        np.zeros(len(x_n)), x_n, fes_type="histogram", histogram_parameters={"bin_edges": edges}
    )
    grid = np.asarray(mbar["cv_grid"])
    populated = np.isfinite(mbar["F_eV"])
    ref = fes.get_fes(grid[populated], reference_point="from-lowest")["f_i"] * kT
    assert np.abs(ref - np.asarray(mbar["F_eV"])[populated]).max() < 1e-6


def test_window_inputs_overrides_and_errors(double_well: DoubleWell, tmp_path: Path) -> None:
    wins = wham.as_windows(double_well.windows)
    assert [w.center for w in wins] == sorted(w["center"] for w in double_well.windows)
    assert wins[0].n_equil == int(round(0.2 * len(wins[0].cv_all))) and wins[0].n < len(
        wins[0].cv_all
    )
    # Window objects, k/T overrides, explicit equilibration fraction
    again = wham.as_windows(wins, k=2.0, T=100.0, equil_fraction=0.5)
    assert (
        again[0].k == 2.0 and again[0].T == 100.0 and again[0].n_equil == len(wins[0].cv_all) // 2
    )
    # k and T given explicitly (files without them)
    bare = [{"cv": w["cv"], "center": w["center"]} for w in double_well.windows]
    result = wham.free_energy(bare, k=double_well.k, T=double_well.T, method="wham", n_blocks=3)
    assert result["dF_eV"] == pytest.approx(double_well.dF_true, abs=0.02)
    assert result["n_blocks_valid"] == 3
    with pytest.raises(ValueError, match="bias constant"):
        wham.free_energy(bare, T=double_well.T)
    with pytest.raises(ValueError, match="temperature"):
        wham.free_energy(bare, k=double_well.k)
    with pytest.raises(ValueError, match="method"):
        wham.free_energy(double_well.windows, method="tram")
    with pytest.raises(ValueError, match="n_blocks"):
        wham.free_energy(double_well.windows, n_blocks=1)
    with pytest.raises(ValueError, match="no umbrella windows"):
        wham.as_windows([])
    with pytest.raises(TypeError):
        wham.as_windows([42])
    mixed = [dict(double_well.windows[0]), dict(double_well.windows[1], T=999.0)]
    with pytest.raises(ValueError, match="different temperatures"):
        wham.as_windows(mixed)
    short = [dict(w, cv=np.asarray(w["cv"])[:3]) for w in double_well.windows]
    with pytest.raises(ValueError, match="too few"):
        wham.free_energy(short, n_blocks=5)
    # npz files and a run dir (index) are accepted too
    root = tmp_path / "run"
    root.mkdir()
    files = []
    for i, w in enumerate(double_well.windows[:4]):
        path = umbrella.window_path(root, i)
        np.savez(path, cv=w["cv"], center=w["center"], k=w["k"], T=w["T"], n_equil=100)
        files.append(str(path))
    from_files = wham.as_windows(files)
    assert len(from_files) == 4 and from_files[0].n_equil == 100 and from_files[0].path == files[0]
    with pytest.raises(FileNotFoundError):
        wham.as_windows(root)


def test_basins_and_overlaps() -> None:
    grid = np.linspace(-1.0, 1.0, 41)
    kT = 0.05
    F = 0.3 * (1 + np.cos(np.pi * grid)) + 0.1 * grid  # minima at -1 and 1, maximum near 0
    out = wham.basins(grid, F, kT, -1.0, 1.0)
    assert out["barrier_found"] and abs(out["cv_ts"]) < 0.2
    assert out["dF_eV"] > 0 and out["dF_barrier_eV"] > 0.4
    fixed = wham.basins(grid, F, kT, -1.0, 1.0, split=0.0)
    assert fixed["cv_ts"] == 0.0 and fixed["dF_eV"] == pytest.approx(out["dF_eV"], abs=0.02)
    mono = wham.basins(grid, 0.5 * grid, kT, -1.0, 1.0)
    assert not mono["barrier_found"] and mono["cv_ts"] == 0.0 and mono["dF_eV"] > 0
    with pytest.raises(ValueError, match="both basins"):
        wham.basins(grid, np.where(grid <= 0, np.nan, F), kT, -1.0, 1.0)
    with pytest.raises(ValueError, match="divider"):
        wham.basins(grid, F, kT, -1.0, 1.0, split=2.0)
    a = wham.Window(center=0.0, k=1.0, T=300.0, cv=np.zeros(10), cv_all=np.zeros(10), n_equil=0)
    b = wham.Window(center=1.0, k=1.0, T=300.0, cv=np.ones(10), cv_all=np.ones(10), n_equil=0)
    c = wham.Window(center=0.0, k=1.0, T=300.0, cv=np.zeros(10), cv_all=np.zeros(10), n_equil=0)
    edges = np.linspace(-0.5, 1.5, 5)
    assert wham.adjacent_overlaps([a, b], edges) == [0.0]
    assert wham.adjacent_overlaps([a, c], edges) == [1.0]
    with pytest.raises(ValueError, match="empty CV range"):
        wham.make_edges([a], cv_range=(1.0, 1.0))
    assert len(wham.make_edges([a, b], n_bins=10)) == 11


def test_free_energy_from_a_run_dir_and_index_fallbacks(
    synthetic_run: Path, double_well: DoubleWell, settings: Settings
) -> None:
    from_dir = wham.free_energy(synthetic_run, method="wham")
    assert from_dir["n_windows"] == DW_WINDOWS and from_dir["T"] == pytest.approx(double_well.T)
    assert from_dir["dF_eV"] == pytest.approx(double_well.dF_true, abs=0.02)
    from_index = wham.free_energy(
        synthetic_run / umbrella.INDEX_NAME, method="wham", T=2 * double_well.T
    )
    assert from_index["T"] == pytest.approx(2 * double_well.T)  # explicit T overrides the index
    # an index without model provenance: the model file decides, else a placeholder label
    index_path = synthetic_run / umbrella.INDEX_NAME
    index = json.loads(index_path.read_text())
    del index["model_info"]
    index_path.write_text(json.dumps(index))
    result = run_stage("sampling.wham", settings, wham.run, run_dir=synthetic_run, method="wham")
    assert result.status == "ok" and result.summary["model_label"] == "unknown"
    model = synthetic_run / "synthetic.model"
    model.write_bytes(b"weights")
    index["model"] = str(model)
    index_path.write_text(json.dumps(index))
    result = run_stage("sampling.wham", settings, wham.run, run_dir=synthetic_run, method="wham")
    assert result.status == "ok" and result.summary["model_label"] == "synthetic"
    numbers = json.loads((Path(result.manifest_path).parent / "numbers.json").read_text())
    assert numbers["sampling.wham.FeSi.synthetic.dF_eV@meta"]["e0_source"] == "unknown"
