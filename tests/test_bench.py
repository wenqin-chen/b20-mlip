"""Bench tier (SPEC.md section 15, CONTRACTS.md row 15): quick mode with the tiny model.

Everything runs offline under ``tmp_path``: the fixture frames stand in for the MPtrj B20
extract, a two-member zip for the WBM initial structures. ``bench.quick`` keeps every
measurement at seconds (the fine-tune sub-task spawns two ``mace_run_train`` processes of a
few seconds each with the tiny model as its foundation).
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase.io import write as ase_write

from b20mlip import bench
from b20mlip.config import Settings
from b20mlip.io import frame_to_atoms, write_frames
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage
from b20mlip.report.numbers import parse_numbers_file

WBM_IDS = ("wbm-1-7", "wbm-2-9")
TINY_BENCH: dict[str, Any] = {
    "quick": True,
    "md_natoms": [8, 64],
    "n_relax": 2,
}


def _write_wbm(root: Path, frames: list[Frame]) -> None:
    """A WBM sample JSON (the keys ``wbm.load_sample`` checks) and a zip of initial cells."""
    (root / "wbm").mkdir(parents=True, exist_ok=True)
    (root / "wbm" / "sample_1000_s0.json").write_text(
        json.dumps(
            {
                "ids": list(WBM_IDS),
                "prevalence": 0.5,
                "in_family_ids": list(WBM_IDS),
                "truth": {},
                "checksums": {},
                "seed": 0,
                "n": len(WBM_IDS),
            }
        ),
        encoding="utf-8",
    )
    refs = root / "raw" / "references"
    refs.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(refs / "wbm-initial-atoms.extxyz.zip", "w") as zf:
        for wid, frame in zip(WBM_IDS, frames, strict=False):
            atoms = frame_to_atoms(frame)
            atoms.calc = None
            atoms.info = {}
            buf = io.StringIO()
            ase_write(buf, atoms, format="extxyz")
            zf.writestr(f"{wid}.extxyz", buf.getvalue())


@pytest.fixture
def bench_root(tmp_path: Path, tiny_frames: list[Frame]) -> Path:
    data = tmp_path / "data"
    (data / "frames").mkdir(parents=True)
    write_frames(tiny_frames, data / "frames" / "mptrj_b20.extxyz")
    _write_wbm(data, [f for f in tiny_frames if f.config_type == "rattle"][:2])
    return tmp_path


@pytest.fixture
def bench_cfg(bench_root: Path) -> Settings:
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(bench_root / "data"),
                "runs_dir": str(bench_root / "runs"),
                "models_dir": str(bench_root / "models"),
            },
            "compute": {"threads": 2},
            "train": {"extra_args": []},
            "eval": {"fmax": 1e-8, "max_steps": 5},  # every relaxation hits the cap: 5 steps
            "bench": dict(TINY_BENCH),
        }
    )


def _state(out: Path) -> dict[str, Any]:
    return json.loads((out / bench.BENCH_JSON).read_text(encoding="utf-8"))


# --- pure helpers ---------------------------------------------------------------------------------


def test_parse_tiers_letters_names_order_and_errors() -> None:
    assert bench.parse_tiers(None) == ["a", "b", "c", "d", "e"]
    assert bench.parse_tiers("md, c , E") == ["b", "c", "e"]
    assert bench.parse_tiers(["phonons", "a", "a"]) == ["a", "d"]
    assert bench.parse_tiers("") == []
    with pytest.raises(ValueError, match="unknown bench tier"):
        bench.parse_tiers("a,z")


def test_bench_params_quick_scaling(bench_cfg: Settings) -> None:
    full = bench.bench_params(bench_cfg, quick=False)
    quick = bench.bench_params(bench_cfg, quick=True)
    assert full["n_frames_8atom"] == 80 and full["md_steps"] == 200 and full["quick"] is False
    assert quick["n_frames_8atom"] == 8 and quick["n_frames_64atom"] == 5
    assert quick["md_steps"] == 20 and quick["phonon_supercell"] == 1 and quick["quick"] is True
    assert quick["md_natoms"] == [8, 64]  # not scaled: the caller chooses the sizes
    assert bench._params_subset(quick, "phonons") == {
        "compound": "FeSi", "phonon_supercell": 1, "phonon_distance": 0.03,
    }  # fmt: skip


def test_split_frames_train_is_a_multiple_of_the_batch(tiny_frames: list[Frame]) -> None:
    frames = list(tiny_frames)
    cycled = [frames[i % len(frames)] for i in range(80)]
    train, valid = bench.split_frames(cycled, 4)
    assert (len(train), len(valid)) == (72, 8)
    train, valid = bench.split_frames(cycled[:20], 4)
    assert (len(train), len(valid)) == (16, 4)
    train, valid = bench.split_frames(cycled[:8], 4)
    assert (len(train), len(valid)) == (4, 4)
    train, valid = bench.split_frames(cycled[:5], 4)
    assert (len(train), len(valid)) == (4, 1)
    with pytest.raises(ValueError, match="batch_size \\+ 1"):
        bench.split_frames(cycled[:4], 4)


def test_parse_epoch_walls_and_step_times(tmp_path: Path) -> None:
    log_text = "\n".join(
        [
            "2026-09-18 20:00:00.000 INFO: Started training, reporting errors on validation set",
            "2026-09-18 20:00:01.500 INFO: Initial: head: Default, loss=1.0, RMSE_F=1.0 meV / A",
            "2026-09-18 20:00:08.000 INFO: Epoch 0: head: Default, loss=0.5, RMSE_F=1.0 meV / A",
            "2026-09-18 20:00:08.100 INFO: Epoch 0: head: pt_head, loss=0.5, RMSE_F=1.0 meV / A",
            "2026-09-18 20:00:14.250 INFO: Epoch 1: head: Default, loss=0.4, RMSE_F=1.0 meV / A",
            "2026-09-18 20:00:15.000 INFO: Training complete",
        ]
    )
    parsed = bench.parse_epoch_walls(log_text)
    assert parsed["initial_ts"] is not None
    assert parsed["epoch_wall_s"] == pytest.approx({0: 6.5, 1: 6.25})
    assert bench.parse_epoch_walls("nothing here") == {"initial_ts": None, "epoch_wall_s": {}}

    results = tmp_path / "x_run-0_train.txt"
    results.write_text(
        "\n".join(
            [
                json.dumps({"mode": "eval", "epoch": None, "loss": 1.0}),
                json.dumps({"mode": "opt", "epoch": 0, "loss": 0.9, "time": 0.25}),
                json.dumps({"mode": "opt", "epoch": 0, "loss": 0.8, "time": 0.35}),
                "not json",
                json.dumps({"mode": "opt", "epoch": 1, "loss": 0.7, "time": 0.5}),
                json.dumps({"mode": "opt", "epoch": 1, "loss": 0.6}),  # no time: skipped
            ]
        )
    )
    steps = bench.parse_step_times(results)
    assert steps == {0: {"steps_s": pytest.approx(0.6), "n_batches": 2},
                     1: {"steps_s": pytest.approx(0.5), "n_batches": 1}}  # fmt: skip
    assert bench.parse_step_times(tmp_path / "missing.txt") == {}


def test_ru_maxrss_units(monkeypatch: pytest.MonkeyPatch) -> None:
    native = bench.ru_maxrss_bytes()
    assert native > 0
    monkeypatch.setattr(bench.sys, "platform", "linux")
    linux = bench.ru_maxrss_bytes()
    monkeypatch.setattr(bench.sys, "platform", "darwin")
    darwin = bench.ru_maxrss_bytes()
    assert linux == darwin * 1024
    info = bench.host_info()
    assert info["hostname"] and info["cpu_count"] and "torch" in info["versions"]


def test_finetune_frames_labels_and_sizes(tiny_frames: list[Frame], tiny_mace: Any) -> None:
    calc = tiny_mace.calculator()
    mp_frames = [
        f.model_copy(update={"energy_scale": "mp", "label_source": "mptrj"})
        for f in tiny_frames[:3]
    ]
    source = mp_frames + list(tiny_frames[3:6])  # 3 with mp labels, 3 unlabelled scale
    classes = bench.finetune_frames(source, calc, 8, 5, seed=3)
    eight, big = classes["8atom"], classes["64atom"]
    assert len(eight) == 8 and len(big) == 5
    assert eight[:3] == mp_frames  # mp-scale DFT labels are kept verbatim
    for frame in eight[3:] + big:
        assert frame.label_source == "mace_zero_shot" and frame.energy_scale == "mp"
        assert frame.energy is not None and frame.forces is not None and frame.stress is not None
    assert {len(f.numbers) for f in eight} == {8} and {len(f.numbers) for f in big} == {64}
    assert len({f.frame_id for f in eight + big}) == 13  # rattled copies are distinct frames
    again = bench.finetune_frames(source, calc, 8, 5, seed=3)
    assert [f.frame_id for f in again["64atom"]] == [f.frame_id for f in big]  # seeded
    with pytest.raises(ValueError, match="source frame"):
        bench.finetune_frames([], calc, 8, 5, seed=0)


def test_zero_shot_label_without_stress(tiny_frames: list[Frame]) -> None:
    from ase.calculators.lj import LennardJones

    atoms = frame_to_atoms(tiny_frames[0])
    labelled = bench.zero_shot_label(atoms, LennardJones())  # LJ has stress; still a copy
    assert labelled is not atoms and labelled.calc.results["forces"].shape == (8, 3)
    assert labelled.info == {}


# --- the stage: quick tiers b, d, e -----------------------------------------------------------


def test_run_quick_tiers_b_d_e(bench_cfg: Settings, tiny_mace: Any, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run_stage(
        "bench", bench_cfg, bench.run, seed=1, out=out, model=tiny_mace.model_path, tiers="b,d,e"
    )
    assert result.status == "ok", result.summary
    assert result.summary["done"] == "md,phonons,fwbw" and result.summary["skipped"] == ""
    state = _state(out)
    assert state["schema"] == bench.SCHEMA and state["quick"] is True
    assert state["status"] == "partial"  # the bench file is ok only once all five are measured
    assert result.summary["status"] == "ok" and result.summary["bench_status"] == "partial"
    assert state["runs"][-1]["status"] == "ok"
    assert state["model"]["sha256"] == tiny_mace.sha256 and state["model"]["head"] == "Default"
    m = state["measurements"]
    assert set(m) == {"md", "phonons", "fwbw"}
    md = m["md"]
    assert set(md["sizes"]) == {"8", "64"} and md["s_per_step_64"] > 0
    assert md["sizes"]["64"]["steps_timed"] == 18 and md["sizes"]["64"]["reps"] == 2
    assert md["params"] == {"compound": "FeSi", "md_natoms": [8, 64], "md_steps": 20,
                            "md_warmup_steps": 2, "md_T": 300.0}  # fmt: skip
    ph = m["phonons"]
    assert ph["supercell"] == [1, 1, 1] and ph["n_displacements"] > 0 and ph["s"] > 0
    assert ph["n_branches"] == 24 and ph["compound"] == "FeSi"
    fw = m["fwbw"]
    assert fw["process"] == "child" and fw["batch_size"] == 4 and fw["n_atoms"] == 256
    assert fw["peak_rss_gb"] >= fw["rss_loaded_gb"] >= fw["rss_before_gb"] > 0
    assert fw["repeats"] == 1 and len(fw["s_per_batch_all"]) == 2 and fw["s_per_batch"] > 0
    assert Path(fw["log"]).is_file() and Path(fw["structures"]).is_file()
    for entry in m.values():
        assert entry["run_id"] == result.run_id and entry["wall_s"] > 0

    schedule = state["schedule"]
    assert schedule["md_nvt_64_40ps_per_T_h"]["value"] == pytest.approx(
        20000 * md["s_per_step_64"] / 3600
    )
    assert schedule["phonondb103_h"]["value"] == pytest.approx(103 * ph["s"] / 3600)
    assert "error" in schedule["finetune_round0_epoch_s"]
    assert "s_per_structure" in schedule["wbm_relax_h"]["error"]
    assert "value" in schedule["fwbw_ram_headroom_gib"]
    assert all("formula" in entry and "spec" in entry for entry in schedule.values())

    # numbers.json (report-tier format) and the manifest
    run_dir = Path(result.manifest_path).parent
    numbers = parse_numbers_file(json.loads((run_dir / "numbers.json").read_text()))
    keys = set(numbers)
    assert {"bench.md.s_per_step_8", "bench.md.s_per_step_64", "bench.phonons_fesi_111.s",
            "bench.fwbw.peak_rss_gb", "bench.fwbw.s_per_batch",
            "bench.schedule.phonondb103_h"} <= keys  # fmt: skip
    assert not any(k.startswith("bench.finetune") or k.startswith("bench.relax") for k in keys)
    value, meta = numbers["bench.md.s_per_step_64"]
    assert value == pytest.approx(md["s_per_step_64"])
    assert meta["reference"] == {"code": "mace", "functional": "PBE", "pseudos": None,
                                 "e0_source": "foundation"}  # fmt: skip
    assert meta["head"] == "Default" and meta["n"] == 18 and meta["seed"] == 1
    assert meta["ci95"] is None and meta["ci95_reason"] == "single timing run"
    assert meta["threads"] == 2 and meta["dtype"] == "float64" and meta["host"]
    assert meta["model_sha256"] == tiny_mace.sha256 and meta["unit"] == "s/step"
    manifest = read_manifest(result.manifest_path)
    assert manifest.status == "ok" and manifest.stage == "bench" and manifest.seed == 1
    names = {Path(a.path).name for a in manifest.outputs}
    assert {"bench.json", "numbers.json", "plan.json", "bench_report.md"} <= names
    assert manifest.extras["tiers"] == ["b", "d", "e"] and manifest.extras["model_head"]
    assert manifest.extras["bench_status"] == "partial" and manifest.extras["run_status"] == "ok"
    assert m["fwbw"]["heads"] == ["Default"]
    assert manifest.extras["tiers_done"] == ["md", "phonons", "fwbw"]
    assert manifest.extras["numbers_keys"] == sorted(keys)
    assert any(Path(a.path).name == "tiny_b20.model" for a in manifest.inputs)

    report = bench.bench_report(out / bench.BENCH_JSON)
    assert "| (b) md | s_per_step_64 |" in report and "| phonondb103_h |" in report
    assert "not measured" in report and "n/a (missing measurement" in report
    assert report == (run_dir / "bench_report.md").read_text()


def test_failing_tier_is_recorded_and_others_survive(
    bench_cfg: Settings, tiny_mace: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cfg: Settings, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("no MD today")

    monkeypatch.setattr(bench, "bench_md", boom)
    out = tmp_path / "out"
    result = run_stage(
        "bench", bench_cfg, bench.run, seed=0, out=out, model=tiny_mace.model_path, tiers="b,d"
    )
    assert result.status == "partial" and result.summary["failed"] == "md"
    assert result.summary["done"] == "phonons"
    state = _state(out)
    assert state["status"] == "partial"
    assert "no MD today" in state["measurements"]["md"]["error"]
    assert "Traceback" in state["measurements"]["md"]["traceback"]
    assert "error" not in state["measurements"]["phonons"]
    assert "error" in state["schedule"]["md_nvt_64_40ps_per_T_h"]
    assert "value" in state["schedule"]["phonondb103_h"]
    run_dir = Path(result.manifest_path).parent
    numbers = parse_numbers_file(json.loads((run_dir / "numbers.json").read_text()))
    assert not any(k.startswith("bench.md.") for k in numbers)
    assert "bench.phonons_fesi_111.s" in numbers
    assert read_manifest(result.manifest_path).status == "partial"
    report = bench.bench_report(state)
    assert "| (b) md | error |" in report and "no MD today" in report


def test_resume_skips_done_tiers_force_and_param_changes(
    bench_cfg: Settings, tiny_mace: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    first = run_stage(
        "bench", bench_cfg, bench.run, seed=0, out=out, model=tiny_mace.model_path, tiers="b,d"
    )
    assert first.status == "ok"
    cached = _state(out)["measurements"]
    calls: list[str] = []

    def fake(name: str) -> Any:
        def fn(cfg: Settings, **kw: Any) -> dict[str, Any]:
            calls.append(name)
            entry = {k: v for k, v in cached[name].items()
                     if k not in ("params", "wall_s", "measured_at", "run_id")}  # fmt: skip
            return {**entry, "fake": True}

        return fn

    monkeypatch.setattr(bench, "bench_md", fake("md"))
    monkeypatch.setattr(bench, "bench_phonons", fake("phonons"))
    second = run_stage(
        "bench", bench_cfg, bench.run, seed=0, out=out, model=tiny_mace.model_path, tiers="b,d"
    )
    assert second.status == "ok" and calls == []
    assert second.summary["skipped"] == "md,phonons" and second.summary["done"] == ""
    state = _state(out)
    assert [r["skipped"] for r in state["runs"]] == [[], ["md", "phonons"]]
    assert "fake" not in state["measurements"]["md"]

    third = run_stage(
        "bench", bench_cfg, bench.run, seed=0, out=out, model=tiny_mace.model_path,
        tiers="b,d", force=True,
    )  # fmt: skip
    assert third.status == "ok" and calls == ["md", "phonons"]
    assert _state(out)["measurements"]["md"]["fake"] is True

    calls.clear()
    changed = bench_cfg.model_copy(  # md_T is not one of the quick-mode overrides
        update={"bench": bench_cfg.bench.model_copy(update={"md_T": 350.0})}
    )
    fourth = run_stage(
        "bench", changed, bench.run, seed=0, out=out, model=tiny_mace.model_path, tiers="b,d"
    )
    assert fourth.status == "ok" and calls == ["md"]  # md params changed, phonons cached
    assert fourth.summary["skipped"] == "phonons"

    # a failed entry is never treated as done
    calls.clear()
    state = _state(out)
    ph_params = state["measurements"]["phonons"]["params"]
    state["measurements"]["phonons"] = {"error": "x", "params": ph_params}
    (out / bench.BENCH_JSON).write_text(json.dumps(state))
    fifth = run_stage(
        "bench", changed, bench.run, seed=0, out=out, model=tiny_mace.model_path, tiers="b,d"
    )
    assert fifth.status == "ok" and calls == ["phonons"]

    # a different model (or quick flag) resets the cache and keeps the stale file
    calls.clear()
    state = _state(out)
    state["model"]["sha256"] = "deadbeef"
    (out / bench.BENCH_JSON).write_text(json.dumps(state))
    sixth = run_stage(
        "bench", changed, bench.run, seed=0, out=out, model=tiny_mace.model_path, tiers="b,d"
    )
    assert sixth.status == "ok" and calls == ["md", "phonons"]
    assert list(out.glob("bench.stale-*.json")) and _state(out)["notes"]
    assert bench.load_state(out / "nonexistent.json") is None
    (out / "bad.json").write_text("{not json")
    assert bench.load_state(out / "bad.json") is None
    (out / "foreign.json").write_text(json.dumps({"schema": "other"}))
    assert bench.load_state(out / "foreign.json") is None


# --- the stage: quick tiers a and c (mace_run_train with the tiny model as foundation) --------


def test_run_quick_tiers_a_and_c(
    bench_cfg: Settings, tiny_mace: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path("out")  # relative, like the CLI default `--out runs/bench/`
    result = run_stage(
        "bench", bench_cfg, bench.run, seed=2, out=out, model=tiny_mace.model_path, tiers="a,c"
    )
    assert result.status == "ok", result.summary
    state = _state(out)
    ft = state["measurements"]["finetune"]
    assert ft["variant"] == "naive" and ft["e0s"] == "foundation" and ft["n_source"] == 15
    assert set(ft["classes"]) == {"8atom", "64atom"}
    for cls, natoms, n_train in (("8atom", 8, 4), ("64atom", 64, 4)):
        c = ft["classes"][cls]
        assert c["natoms"] == natoms and c["n_train"] == n_train and c["n_train"] % 4 == 0
        assert c["n_batches"] == 1 and c["batch_size"] == 4 and c["measured_epoch"] == 0
        assert c["s_per_epoch_wall"] > 0 and c["s_per_frame"] == c["s_per_epoch_wall"] / n_train
        assert c["s_per_epoch_steps"] > 0 and c["s_per_frame_steps"] < c["s_per_frame"]
        assert c["startup_s"] > 0 and c["subprocess_wall_s"] > c["s_per_epoch_wall"]
        assert c["returncode"] == 0 and Path(c["log"]).is_file() and Path(c["results"]).is_file()
        assert "--foundation_model" in c["argv"] and "--E0s" in c["argv"]
        assert c["argv"][c["argv"].index("--E0s") + 1] == "foundation"
        assert c["argv"][c["argv"].index("--max_num_epochs") + 1] == "1"
        assert not (Path(c["log"]).parent / "models").exists()  # weights cleaned up
    assert ft["classes"]["8atom"]["label_sources"] == {"mace_zero_shot": 8}
    assert ft["s_per_frame_8atom"] > 0 and ft["s_per_frame_64atom"] > 0 and ft["startup_s"] > 0
    relax = state["measurements"]["relax"]
    assert relax["n"] == 2 and relax["max_steps"] == 5 and relax["n_capped"] == 2
    assert relax["steps_per_structure"] == 5 and relax["s_per_structure"] > 0
    assert [r["id"] for r in relax["structures"]] == list(WBM_IDS)
    assert relax["source"]["ids"] == list(WBM_IDS) and relax["s_per_relax_step"] > 0
    sched = state["schedule"]
    assert sched["finetune_round0_epoch_s"]["value"] == pytest.approx(
        600 * ft["s_per_frame_8atom"] + 60 * ft["s_per_frame_64atom"]
    )
    assert sched["finetune_round0_3seeds_h"]["value"] > 0 and sched["wbm_relax_h"]["value"] > 0
    assert sched["wbm_relax_cap_h"]["value"] == pytest.approx(
        1000 * 5 * relax["s_per_relax_step"] / 3600
    )
    run_dir = Path(result.manifest_path).parent
    numbers = parse_numbers_file(json.loads((run_dir / "numbers.json").read_text()))
    assert {"bench.finetune.s_per_frame_8atom", "bench.finetune.s_per_frame_64atom",
            "bench.finetune.startup_s", "bench.relax.s_per_structure",
            "bench.relax.steps_per_structure", "bench.schedule.finetune_round0_3seeds_h"} <= set(
        numbers
    )  # fmt: skip
    assert numbers["bench.finetune.s_per_frame_8atom"][1]["n"] == 4
    assert numbers["bench.relax.steps_per_structure"][1]["n"] == 2
    report = bench.bench_report(state)
    assert "| (a) finetune | s_per_frame_64atom |" in report
    assert "| (c) relax | steps_per_structure | 5 |" in report
    assert "| finetune_round0_3seeds_h |" in report


def test_bench_finetune_reuses_a_finished_class(
    bench_cfg: Settings, tiny_frames: list[Frame], tiny_mace: Any, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    seen: list[str] = []

    def fake_time(cfg: Settings, model_path: Any, frames: Any, work_dir: Any, *, name: str,
                  **kw: Any) -> dict[str, Any]:  # fmt: skip
        seen.append(name)
        if name == "bench_64atom":
            raise RuntimeError("timeout")
        return {"s_per_frame": 0.1, "s_per_epoch_wall": 0.4, "startup_s": 1.0, "n_train": 4}

    monkeypatch.setattr(bench, "time_finetune_epochs", fake_time)
    params = bench.bench_params(bench_cfg, quick=True)
    calc = tiny_mace.calculator()
    first = bench.bench_finetune(
        bench_cfg, model_path=tiny_mace.model_path, calc=calc, params=params,
        work_dir=tmp_path / "ft", seed=0, source=list(tiny_frames),
    )  # fmt: skip
    assert seen == ["bench_8atom", "bench_64atom"] and "64atom: RuntimeError" in first["error"]
    assert first["s_per_frame_8atom"] == 0.1 and "s_per_frame_64atom" not in first
    assert "error" in first["classes"]["64atom"] and first["startup_s"] == 1.0

    seen.clear()
    monkeypatch.setattr(
        bench, "time_finetune_epochs",
        lambda *a, name, **k: seen.append(name) or {"s_per_frame": 0.2, "s_per_epoch_wall": 0.8,
                                                    "startup_s": 2.0, "n_train": 4},
    )  # fmt: skip
    second = bench.bench_finetune(
        bench_cfg, model_path=tiny_mace.model_path, calc=calc, params=params,
        work_dir=tmp_path / "ft", seed=0, source=list(tiny_frames), previous=first,
    )  # fmt: skip
    assert seen == ["bench_64atom"] and "error" not in second
    assert second["s_per_frame_8atom"] == 0.1 and second["s_per_frame_64atom"] == 0.2
    assert second["startup_s"] == pytest.approx(1.5)


def test_time_finetune_epochs_reports_mace_failures(
    bench_cfg: Settings, tiny_frames: list[Frame], tiny_mace: Any, tmp_path: Path
) -> None:
    frames = list(tiny_frames)[:5]
    cfg = bench_cfg.model_copy(
        update={"train": bench_cfg.train.model_copy(update={"extra_args": ["--no_such_flag"]})}
    )
    with pytest.raises(RuntimeError, match="exited with"):
        bench.time_finetune_epochs(
            cfg, tiny_mace.model_path, frames, tmp_path / "bad", name="bad", epochs=1, seed=0,
            timeout_s=600,
        )  # fmt: skip
    assert (tmp_path / "bad" / "mace_stdout.log").is_file()
    with pytest.raises(RuntimeError, match="exceeded"):
        bench.time_finetune_epochs(
            bench_cfg, tiny_mace.model_path, frames, tmp_path / "slow", name="slow", epochs=1,
            seed=0, timeout_s=0.01,
        )  # fmt: skip


def test_wbm_structures_and_source_frames_errors(
    bench_cfg: Settings, tiny_frames: list[Frame], tmp_path: Path
) -> None:
    structures, src = bench.wbm_structures(bench_cfg, 1)
    assert list(structures) == [WBM_IDS[0]] and src["ids"] == [WBM_IDS[0]]
    frames, path = bench.source_frames(bench_cfg)
    assert len(frames) == 15 and path.name == "mptrj_b20.extxyz"

    empty = Settings.model_validate({"paths": {"data_dir": str(tmp_path / "nodata")}})
    with pytest.raises(FileNotFoundError, match="WBM sample"):
        bench.wbm_structures(empty, 1)
    with pytest.raises(FileNotFoundError, match="MPtrj B20"):
        bench.source_frames(empty)
    (tmp_path / "nodata" / "wbm").mkdir(parents=True)
    (tmp_path / "nodata" / "wbm" / "sample_1000_s0.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="initial structures"):
        bench.wbm_structures(empty, 1)
    # the dft tier's by-mp-id extract is the fallback source
    raw = tmp_path / "nodata" / "raw" / "mptrj"
    raw.mkdir(parents=True)
    write_frames(tiny_frames[:2], raw / "b20_mptrj.extxyz")
    frames, path = bench.source_frames(empty)
    assert len(frames) == 2 and path.name == "b20_mptrj.extxyz"
    with pytest.raises(ValueError, match="no structures"):
        bench.bench_relax(bench_cfg, calc=None, structures={})


# --- fwbw in-process and the child entry point ------------------------------------------------


def test_fwbw_measure_and_child(
    tiny_mace: Any, tiny_frames: list[Frame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    cells = []
    for frame in tiny_frames[:2]:
        atoms = frame_to_atoms(frame)
        atoms.calc = None
        atoms.info = {}
        cells.append(atoms)
    res = bench.fwbw_measure(tiny_mace.model_path, cells, threads=1, batch_size=4, repeats=2)
    assert res["batch_size"] == 2 and res["n_atoms"] == 16 and res["n_edges"] > 0
    assert len(res["s_per_batch_all"]) == 3 and res["s_per_batch"] > 0
    assert res["peak_rss_gb"] >= res["rss_before_gb"] > 0 and res["rss_unit"] == "GiB"
    assert res["threads"] == 1 and res["dtype"] == "float64" and res["model_r_max"] == 4.0

    structures = tmp_path / "batch.extxyz"
    ase_write(structures, cells, format="extxyz")
    spec = {"model": str(tiny_mace.model_path), "structures": str(structures), "threads": 1,
            "dtype": "float64", "batch": 4, "repeats": 1}  # fmt: skip
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(spec)))
    bench._fwbw_child_main()
    out_lines = capsys.readouterr().out.splitlines()
    line = next(ln for ln in out_lines if ln.startswith(bench.FWBW_MARKER))
    child = json.loads(line[len(bench.FWBW_MARKER) :])
    assert child["batch_size"] == 2 and child["repeats"] == 1 and child["n_atoms"] == 16


def test_bench_fwbw_child_failure_paths(
    bench_cfg: Settings, tiny_mace: Any, tiny_frames: list[Frame], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    atoms = frame_to_atoms(tiny_frames[0])
    atoms.calc = None
    params = bench.bench_params(bench_cfg, quick=True)
    kw = dict(cfg=bench_cfg, model_path=tiny_mace.model_path, atoms=atoms, params=params,
              work_dir=tmp_path / "fw", seed=0)  # fmt: skip
    with pytest.raises(RuntimeError, match="exceeded"):
        bench.bench_fwbw(timeout_s=0.01, **kw)

    import subprocess

    class Proc:
        def __init__(self, rc: int, out: str) -> None:
            self.returncode, self.stdout, self.stderr = rc, out, "err"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc(3, "boom"))
    with pytest.raises(RuntimeError, match="exited with 3"):
        bench.bench_fwbw(**kw)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc(0, "no marker here"))
    with pytest.raises(RuntimeError, match="no result line"):
        bench.bench_fwbw(**kw)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: Proc(0, "notice\n" + bench.FWBW_MARKER + json.dumps({"peak_rss_gb": 1.0})),
    )  # fmt: skip
    res = bench.bench_fwbw(**kw)
    assert res["peak_rss_gb"] == 1.0 and res["process"] == "child" and res["reps"] == 2


# --- dry run, model resolution, schedule and report edge cases --------------------------------


def test_dry_run_plans_only(bench_cfg: Settings, tiny_mace: Any, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cfg = bench_cfg.model_copy(
        update={"bench": bench_cfg.bench.model_copy(
            update={"model": str(tiny_mace.model_path), "tiers": "c,e"})}
    )  # fmt: skip
    result = run_stage("bench", cfg, bench.run, seed=0, dry_run=True, out=out)
    assert result.status == "partial" and result.summary == {"planned": 2, "tiers": "c,e"}
    run_dir = Path(result.manifest_path).parent
    plan = json.loads((run_dir / bench.PLAN_JSON).read_text())
    assert plan["dry_run"] is True and plan["tiers"] == ["c", "e"] and plan["quick"] is True
    assert plan["model"] == str(tiny_mace.model_path.resolve())
    assert plan["measurements"] == ["relax", "fwbw"] and plan["params"]["n_relax"] == 2
    assert not (out / bench.BENCH_JSON).exists()
    manifest = read_manifest(result.manifest_path)
    assert manifest.status == "partial" and manifest.outputs == []
    assert manifest.extras["model_sha256"] == tiny_mace.sha256


def test_resolve_model_and_tier_validation(
    bench_cfg: Settings, tiny_mace: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bench.resolve_model(bench_cfg, tiny_mace.model_path) == tiny_mace.model_path.resolve()
    with pytest.raises(FileNotFoundError, match="bench model not found"):
        bench.resolve_model(bench_cfg, tmp_path / "nope.model")
    # no model anywhere: the foundation lookup of the train tier decides (it raises when the
    # MPA-0 file is not local; on a developer Mac it may well be, so the lookup is stubbed)
    monkeypatch.setattr(
        bench.finetune, "resolve_foundation",
        lambda cfg, name=None: (_ for _ in ()).throw(FileNotFoundError("no foundation")),
    )  # fmt: skip
    with pytest.raises(FileNotFoundError, match="no foundation"):
        bench.resolve_model(bench_cfg, None)
    monkeypatch.setattr(
        bench.finetune, "resolve_foundation", lambda cfg, name=None: tiny_mace.model_path
    )
    assert bench.resolve_model(bench_cfg, None) == tiny_mace.model_path
    result = run_stage(
        "bench", bench_cfg, bench.run, seed=0, out=tmp_path / "o", model=tiny_mace.model_path,
        tiers="",
    )  # fmt: skip
    assert result.status == "failed" and "no bench tiers" in str(result.summary["error"])
    result = run_stage(
        "bench", bench_cfg, bench.run, seed=0, out=tmp_path / "o2", model=tiny_mace.model_path,
        tiers="b", quick=True,
    )  # fmt: skip
    assert result.status == "ok"


def test_bench_md_rejects_too_few_steps(bench_cfg: Settings, tiny_frames: list[Frame]) -> None:
    params = {**bench.bench_params(bench_cfg, quick=True), "md_steps": 2, "md_warmup_steps": 2}
    with pytest.raises(ValueError, match="md_steps"):
        bench.bench_md(bench_cfg, calc=None, atoms=frame_to_atoms(tiny_frames[0]), params=params,
                       seed=0)  # fmt: skip


def test_derive_schedule_and_report_without_measurements(bench_cfg: Settings) -> None:
    schedule = bench.derive_schedule({}, bench_cfg)
    assert set(schedule) == {
        "finetune_round0_epoch_s", "finetune_round0_per_seed_min", "finetune_round0_3seeds_h",
        "md_nvt_64_40ps_per_T_h", "md_nvt_64_40ps_3T_h", "md_512_100ps_ase_h",
        "umbrella_windows_64_h", "wbm_relax_h", "wbm_relax_cap_h", "phonondb103_h",
        "fwbw_ram_headroom_gib",
    }  # fmt: skip
    assert all("error" in e and "value" not in e for e in schedule.values())
    measurements = {
        "finetune": {"s_per_frame_8atom": 0.12, "s_per_frame_64atom": 3.5, "startup_s": 60.0,
                     "classes": {}},
        "md": {"s_per_step_64": 0.4, "s_per_step_512": 3.2, "sizes": {}, "ensemble": "nvt",
               "T_K": 300.0, "timestep_fs": 2.0},
        "relax": {"s_per_structure": 20.0, "s_per_relax_step": 0.1, "n": 20,
                  "steps_per_structure": 200.0},
        "phonons": {"s": 30.0},
        "fwbw": {"peak_rss_gb": 4.5, "error": "n/a"},
    }  # fmt: skip
    schedule = bench.derive_schedule(measurements, bench_cfg)
    assert schedule["finetune_round0_epoch_s"]["value"] == pytest.approx(600 * 0.12 + 60 * 3.5)
    assert schedule["finetune_round0_per_seed_min"]["value"] == pytest.approx(
        (30 * (600 * 0.12 + 60 * 3.5) + 60.0) / 60
    )
    assert schedule["finetune_round0_3seeds_h"]["value"] == pytest.approx(
        3 * schedule["finetune_round0_per_seed_min"]["value"] / 60
    )
    assert schedule["md_nvt_64_40ps_3T_h"]["value"] == pytest.approx(3 * 20000 * 0.4 / 3600)
    assert schedule["md_512_100ps_ase_h"]["value"] == pytest.approx(50000 * 3.2 / 3600)
    assert schedule["umbrella_windows_64_h"]["value"] == pytest.approx(12 * 7500 * 0.4 / 3600)
    assert schedule["wbm_relax_h"]["value"] == pytest.approx(1000 * 20 / 3600)
    assert schedule["wbm_relax_cap_h"]["value"] == pytest.approx(
        1000 * bench_cfg.eval.max_steps * 0.1 / 3600
    )
    assert schedule["phonondb103_h"]["value"] == pytest.approx(103 * 30 / 3600)
    assert "error" in schedule["fwbw_ram_headroom_gib"]  # fwbw carries an error: not used

    state = {"model": {}, "host": {}, "measurements": {}, "schedule": {}}
    report = bench.bench_report(state)
    assert report.count("not measured") == 5 and "| Schedule entry |" not in report
    numbers = bench.bench_numbers({"measurements": {}, "schedule": {}}, seed=0)
    assert numbers == {}
    assert bench._fmt(None) == "n/a" and bench._fmt(True) == "True" and bench._fmt(3) == "3"
    assert bench._fmt(np.float64(0.123456)) == "0.123" and bench._fmt("x") == "x"
