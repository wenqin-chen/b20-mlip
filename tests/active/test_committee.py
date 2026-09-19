"""Committee sigma_F mechanics (identical members -> 0), synthetic-sigma selection rules,
the active.select stage and its numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from b20mlip.active import committee
from b20mlip.config import Settings
from b20mlip.io import read_frames
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage


def test_sigma_from_forces_definition() -> None:
    # two members, two frames of two atoms: per-atom std of the force vector, max over atoms
    a = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros((2, 3))]
    b = [np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]), np.zeros((2, 3))]
    sigma = committee.sigma_from_forces([a, b])
    assert (
        sigma.shape == (2,) and sigma[0] == pytest.approx(1.0) and sigma[1] == 0.0
    )  # std of y: ±1
    three = committee.sigma_from_forces([a, b, a])
    assert 0 < three[0] < 1.0  # population std over three members
    with pytest.raises(ValueError, match="at least 2"):
        committee.sigma_from_forces([a])
    with pytest.raises(ValueError, match="same frames"):
        committee.sigma_from_forces([a, b[:1]])


def test_sigma_f_identical_members_is_zero(
    committee_paths: list[Path], tiny_frames: list[Frame], tiny_calc: Any
) -> None:
    frames = tiny_frames[:4]
    sigma = committee.sigma_f(committee_paths, frames)
    assert sigma.shape == (4,) and np.all(
        np.abs(sigma) < 1e-8
    )  # identical weights (float noise only)
    assert np.all(
        committee.sigma_f([], frames, calcs=[tiny_calc, tiny_calc]) == 0.0
    )  # same instance: exact
    assert committee.sigma_f(committee_paths, []).shape == (0,)
    with pytest.raises(ValueError, match="at least 2 models"):
        committee.sigma_f(committee_paths[:1], frames)
    with pytest.raises(ValueError, match="at least 2 calculators"):
        committee.sigma_f([], frames, calcs=[tiny_calc])
    with pytest.raises(FileNotFoundError):
        committee.sigma_f([committee_paths[0], Path("nope.model")], frames)
    forces = committee.committee_forces([], frames, calcs=[tiny_calc, tiny_calc])
    assert len(forces) == 2 and forces[0][0].shape == (8, 3)
    np.testing.assert_allclose(forces[0][1], forces[1][1])


def test_select_rules(tiny_frames: list[Frame]) -> None:
    frames = tiny_frames[:12]  # 4 compounds x 3 config types, one group per (compound, type)
    sigma = np.array([0.05, 0.9, 0.3, 0.9, 0.2, 0.7, 0.05, 0.4, 0.8, 0.6, 0.1, 0.9])
    chosen = committee.select(frames, sigma, 5)
    values = [f.info["committee_sigma_f"] for f in chosen]
    assert values == sorted(values, reverse=True) and values[:3] == [0.9, 0.9, 0.9]
    assert [f.info["committee_rank"] for f in chosen] == [0, 1, 2, 3, 4]
    # ties (0.9 x3) are broken by frame_id ascending
    tied = [f.frame_id for f in chosen[:3]]
    assert tied == sorted(tied)
    # sigma_min drops the low-uncertainty frames even when n is not reached
    assert len(committee.select(frames, sigma, 100, sigma_min=0.5)) == 6
    # per-group cap: every frame is its own group here, so cap 1 changes nothing ...
    assert len(committee.select(frames, sigma, 100, per_group_max=1)) == 12
    # ... but with shared groups it does
    same_group = [f.model_copy(update={"group_id": f"{f.compound}/all/p"}) for f in frames]
    capped = committee.select(same_group, sigma, 100, per_group_max=1)
    assert len(capped) == 4 and len({f.group_id for f in capped}) == 4
    capped2 = committee.select(same_group, sigma, 100, per_group_max=2)
    assert len(capped2) == 8
    assert committee.select(frames, sigma, 0) == []
    nan_sigma = sigma.copy()
    nan_sigma[1] = np.nan
    assert all(
        f.info["committee_sigma_f"] != 0.9 or f.frame_id != frames[1].frame_id
        for f in committee.select(frames, nan_sigma, 12)
    )
    assert len(committee.select(frames, nan_sigma, 12)) == 11
    assert frames[0].info.get("committee_sigma_f") is None  # inputs untouched
    with pytest.raises(ValueError, match="shape"):
        committee.select(frames, sigma[:3], 2)
    with pytest.raises(ValueError, match="n must be"):
        committee.select(frames, sigma, -1)
    with pytest.raises(ValueError, match="per_group_max"):
        committee.select(frames, sigma, 2, per_group_max=0)
    assert committee.parse_models("a.model, b.model,") == [Path("a.model"), Path("b.model")]


def test_run_stage_writes_candidates_and_numbers(
    committee_paths: list[Path], frames_path: Path, settings: Settings, tmp_path: Path
) -> None:
    out = tmp_path / "data" / "frames" / "candidates_r1.extxyz"
    result = run_stage(
        "active.select", settings, committee.run, models=committee_paths, frames_path=frames_path,
        n=6, out=out, per_group_max=1,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    assert (
        result.summary["n_candidates"] == 15
        and result.summary["n_selected"] == 6
        and result.summary["n_models"] == 3
    )
    assert (
        abs(result.summary["sigma_f_median"]) < 1e-8 and abs(result.summary["sigma_f_max"]) < 1e-8
    )
    chosen = read_frames(out)
    assert len(chosen) == 6 and all(abs(f.info["committee_sigma_f"]) < 1e-8 for f in chosen)
    assert [f.info["committee_rank"] for f in chosen] == list(range(6))
    assert len({f.group_id for f in chosen}) == 6
    run_dir = Path(result.manifest_path).parent
    numbers = json.loads((run_dir / "numbers.json").read_text())
    assert numbers["active.n_selected"] == 6 and numbers["active.selected_n"] == 6
    assert numbers["active.n_candidates"] == 15 and abs(numbers["active.sigma_f_median"]) < 1e-8
    assert (
        numbers["active.sigma_f_median@meta"]["unit"] == "eV/Å"
        and numbers["active.n_selected@meta"]["n_models"] == 3
    )
    sigma_doc = json.loads((run_dir / "sigma_f.json").read_text())
    assert len(sigma_doc["sigma_f"]) == 15 and len(sigma_doc["selected"]) == 6
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["n_selected"] == 6 and len(manifest.extras["models_sha256"]) == 3
    assert {Path(a.path).name for a in manifest.outputs} >= {
        "candidates_r1.extxyz",
        "numbers.json",
        "sigma_f.json",
        "plan.json",
    }
    assert len([a for a in manifest.inputs if a.kind == "model"]) == 3
    # sigma_min above every sigma -> nothing selected, still ok
    none = run_stage(
        "active.select",
        settings,
        committee.run,
        models=",".join(str(p) for p in committee_paths),
        frames_path=frames_path,
        n=5,
        out=tmp_path / "none.extxyz",
        sigma_min=0.1,
    )
    assert (
        none.status == "ok"
        and none.summary["n_selected"] == 0
        and read_frames(tmp_path / "none.extxyz") == []
    )


def test_run_dry_run_and_failures(
    committee_paths: list[Path], frames_path: Path, settings: Settings, tmp_path: Path
) -> None:
    dry = run_stage(
        "active.select",
        settings,
        committee.run,
        dry_run=True,
        models=committee_paths,
        frames_path=frames_path,
        out=tmp_path / "x.extxyz",
    )
    assert dry.status == "partial" and dry.summary["planned"] == 1 and dry.outputs == []
    r = run_stage(
        "active.select",
        settings,
        committee.run,
        models=committee_paths[:1],
        frames_path=frames_path,
        out=tmp_path / "x.extxyz",
    )
    assert r.status == "failed" and "at least 2" in r.summary["error"]
    r = run_stage(
        "active.select",
        settings,
        committee.run,
        models=[committee_paths[0], tmp_path / "nope.model"],
        frames_path=frames_path,
        out=tmp_path / "x.extxyz",
    )
    assert r.status == "failed" and "not found" in r.summary["error"]
    r = run_stage(
        "active.select",
        settings,
        committee.run,
        models=committee_paths,
        frames_path=tmp_path / "nope.extxyz",
        out=tmp_path / "x.extxyz",
    )
    assert r.status == "failed" and "frames file not found" in r.summary["error"]
