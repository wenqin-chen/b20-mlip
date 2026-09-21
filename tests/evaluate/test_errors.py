"""Error tables with the tiny model: metrics, CIs, by_config_type, R3 energy gate, T3 floor,
numbers.json meta, the run() stage (dry-run, split tiers, checkpoint bookkeeping)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from b20mlip.config import Settings
from b20mlip.evaluate import errors
from b20mlip.io import write_frames
from b20mlip.models import CheckpointInfo, ErrorTable, Frame, Reference, Split
from b20mlip.provenance import read_manifest, run_stage, sha256_file, sha256_frames

from .conftest import BOOT_N, retag


def _split(frames: list[Frame], tiers: dict[str, list[str]]) -> Split:
    groups: dict[str, list[str]] = {}
    for f in frames:
        groups.setdefault(f.group_id, []).append(f.frame_id)
    return Split(
        split_id="eval-split", seed=0, frames_sha256=sha256_frames(frames),
        train=[], val=[], test=[f.frame_id for f in frames], tiers=tiers, groups=groups,  # type: ignore[arg-type]
    )  # fmt: skip


def _checkpoint(tiny_mace: Any, path: Path, **update: Any) -> Path:
    info = CheckpointInfo(
        model_path=str(tiny_mace.model_path), sha256=tiny_mace.sha256, lammps_path=None,
        variant="scratch", foundation=None, foundation_sha256=None, heads=["Default"],
        e0_source="estimated", energy_scale="qe", seed=0, epochs=1, lr=1e-4, batch_size=4,
        split_id=None, replay_samples=None, train_run_id=None,
    ).model_copy(update=update)  # fmt: skip
    path.write_text(info.model_dump_json(indent=2), encoding="utf-8")
    return path


# --- evaluate() -----------------------------------------------------------------------------------


def test_evaluate_metrics_cis_and_breakdown(
    tiny_mace: Any, tiny_calc: Any, qe_frames: list[Frame]
) -> None:
    table = errors.evaluate(
        tiny_mace.model_path, "Default", qe_frames, "T0", None, BOOT_N, 0,
        energy_scale="qe", model_label="B3", split_id="s", run_id="r", calc=tiny_calc,
    )  # fmt: skip
    assert isinstance(table, ErrorTable)
    assert set(table.metrics) == {"mae_e", "rmse_e", "mae_f", "rmse_f", "mae_s"}
    assert table.n_frames == 15 and table.n_atoms == 120
    assert table.model_sha256 == tiny_mace.sha256 and table.model_label == "B3"
    assert table.reference == Reference(
        code="qe", functional="PBE", pseudos="SSSP-efficiency-1.3", e0_source=None
    )
    assert table.bootstrap_n == BOOT_N and table.bootstrap_seed == 0 and table.run_id == "r"
    for name, metric in table.metrics.items():
        assert np.isfinite(metric.value) and metric.value >= 0.0, name
        assert metric.ci95 is not None and metric.ci95[0] <= metric.value <= metric.ci95[1], name
        assert metric.n == 15 and metric.unit == errors.UNITS[name]
    assert table.metrics["rmse_f"].value >= table.metrics["mae_f"].value
    assert table.metrics["rmse_e"].value >= table.metrics["mae_e"].value
    assert set(table.by_config_type) == {"relax", "strain", "rattle", "noise_floor"}
    for sub in table.by_config_type.values():
        assert {"mae_f", "rmse_f", "mae_e"} <= set(sub)
    # 3 noise_floor frames = 3 groups -> a CI; the rattle subset has 4 groups too
    assert table.by_config_type["noise_floor"]["mae_f"].n == 3
    assert table.by_config_type["noise_floor"]["mae_f"].ci95 is not None
    # bit-for-bit deterministic for the same seed
    again = errors.evaluate(
        tiny_mace.model_path,
        "Default",
        qe_frames,
        "T0",
        None,
        BOOT_N,
        0,
        energy_scale="qe",
        calc=tiny_calc,
    )
    assert again.metrics["mae_f"] == table.metrics["mae_f"]


def test_metrics_match_a_direct_computation(
    tiny_calc: Any, qe_frames: list[Frame], tiny_mace: Any
) -> None:
    frames = qe_frames[:4]
    res = errors.residuals(tiny_calc, frames, with_energy=True)
    metrics = errors.metrics_from_residuals(res, BOOT_N, 0)
    df = np.concatenate([r.df for r in res])
    assert metrics["mae_f"].value == pytest.approx(np.mean(np.abs(df)) * 1e3)
    assert metrics["rmse_f"].value == pytest.approx(np.sqrt(np.mean(df**2)) * 1e3)
    de = np.array([r.de_per_atom for r in res])
    assert metrics["mae_e"].value == pytest.approx(np.mean(np.abs(de)) * 1e3)
    ds = np.concatenate([r.ds for r in res])
    assert metrics["mae_s"].value == pytest.approx(np.mean(np.abs(ds)) * 1e3)
    # the residual of one frame equals prediction minus label
    e, f, s = errors.predict(tiny_calc, frames[0])
    np.testing.assert_allclose(res[0].df, (f - np.asarray(frames[0].forces)).ravel())
    assert res[0].de_per_atom == pytest.approx((e - frames[0].energy) / 8)
    assert s is not None and s.shape == (6,)
    # a single group gives no CI
    one = errors.metrics_from_residuals(res[:1], BOOT_N, 0)
    assert one["mae_f"].ci95 is None and one["mae_f"].n == 1
    with pytest.raises(ValueError, match="no residuals"):
        errors.metrics_from_residuals([], BOOT_N, 0)


def test_energy_scale_gate_rule_r3(
    tiny_mace: Any, tiny_calc: Any, qe_frames: list[Frame], tiny_frames: list[Frame]
) -> None:
    assert errors.energy_metrics_allowed("qe", qe_frames) == (True, None)
    ok, reason = errors.energy_metrics_allowed("mp", qe_frames)
    assert not ok and "rule R3" in (reason or "")
    ok, reason = errors.energy_metrics_allowed(None, qe_frames)
    assert not ok and "unknown" in (reason or "")
    mixed = retag(qe_frames[:5], "mp", "mptrj") + qe_frames[5:]
    assert errors.frames_energy_scale(mixed) == "mixed"
    assert errors.energy_metrics_allowed("qe", mixed)[1] == "frames mix energy scales"
    unlabelled = [f.model_copy(update={"energy": None}) for f in qe_frames]
    assert errors.energy_metrics_allowed("qe", unlabelled)[1] == "no energy labels in the frames"
    assert errors.frames_energy_scale(tiny_frames) == "qe"  # 3 qe frames among unlabelled ones
    table = errors.evaluate(
        tiny_mace.model_path,
        "Default",
        qe_frames,
        "T4a",
        None,
        BOOT_N,
        0,
        energy_scale="mp",
        calc=tiny_calc,
    )
    assert set(table.metrics) == {"mae_f", "rmse_f", "mae_s"}  # energies skipped, not fabricated
    assert all("mae_e" not in sub for sub in table.by_config_type.values())


def test_evaluate_validation_and_reference_inference(
    tiny_mace: Any, tiny_calc: Any, qe_frames: list[Frame]
) -> None:
    with pytest.raises(ValueError, match="no frames"):
        errors.evaluate(tiny_mace.model_path, "Default", [], "T0", None, 10, 0, calc=tiny_calc)
    with pytest.raises(ValueError, match="unknown tier"):
        errors.evaluate(
            tiny_mace.model_path, "Default", qe_frames, "T9", None, 10, 0, calc=tiny_calc
        )
    with pytest.raises(ValueError, match="head"):
        errors.evaluate(tiny_mace.model_path, "other", qe_frames, "T0", None, 10, 0, calc=tiny_calc)
    with pytest.raises(ValueError, match="no force labels"):
        errors.evaluate(
            tiny_mace.model_path,
            "Default",
            [qe_frames[0].model_copy(update={"forces": None})],
            "T0",
            None,
            10,
            0,
            calc=tiny_calc,
        )
    assert errors.reference_for(retag(qe_frames, "mp", "mptrj")).code == "vasp"
    assert errors.reference_for(retag(qe_frames, "omat24", "omat24")).pseudos == "PAW (OMat24)"
    assert errors.reference_for(qe_frames, pseudos="SSSP-x", e0_source="E0s_qe.json") == Reference(
        code="qe", functional="PBE", pseudos="SSSP-x", e0_source="E0s_qe.json"
    )
    with pytest.raises(ValueError, match="several label sources"):
        errors.reference_for(qe_frames[:1] + retag(qe_frames[1:2], "mp", "mptrj"))
    with pytest.raises(ValueError, match="no reference known"):
        errors.reference_for(retag(qe_frames, "none", "none"))
    with pytest.raises(ValueError, match="heads"):
        errors.make_calculator(tiny_mace.model_path, "pt_head")


def test_labels_scales_and_tiers(tiny_mace: Any, tmp_path: Path) -> None:
    assert errors.label_for(None, "runs/x/b20_naive.model") == "b20_naive"
    ckpt = CheckpointInfo.model_validate_json(
        _checkpoint(tiny_mace, tmp_path / "c.json").read_text()
    )
    assert errors.label_for(ckpt, "m") == "B3"
    assert errors.label_for(ckpt.model_copy(update={"variant": "naive"}), "m") == "B1"
    assert errors.label_for(ckpt.model_copy(update={"variant": "replay"}), "m") == "B2"
    zero = ckpt.model_copy(update={"variant": "zero_shot", "foundation": "medium-mpa-0"})
    assert errors.label_for(zero, "m") == "B0"
    assert errors.label_for(zero.model_copy(update={"foundation": "medium"}), "m") == "B0p"
    assert errors.model_energy_scale(ckpt, "Default") == "qe"
    assert errors.model_energy_scale(ckpt, "pt_head") == "mp"  # rule R2
    assert errors.model_energy_scale(None, "Default") is None
    assert errors.parse_tiers("T0, T3,") == ["T0", "T3"] and errors.parse_tiers(None) is None
    assert errors.parse_tiers(["T1"]) == ["T1"]
    with pytest.raises(ValueError, match="unknown tiers"):
        errors.parse_tiers("T0,T7")


def test_noise_floor_from_frames(qe_frames: list[Frame]) -> None:
    rng = np.random.default_rng(0)
    omat = retag(qe_frames[:4], "omat24", "omat24")
    relabelled = [
        f.model_copy(
            update={"forces": (np.asarray(f.forces) + rng.normal(0.0, 0.01, (8, 3))).tolist()}
        )
        for f in omat[:3]
    ]
    floor = errors.noise_floor_from_frames(relabelled, omat)
    assert floor["n_frames"] == 3 and floor["n_components"] == 72 and floor["unit"] == "meV/Å"
    assert 5.0 < floor["noise_floor_f"] < 20.0 and floor["mae_f"] < floor["noise_floor_f"]
    with pytest.raises(ValueError, match="no frame_id pairs"):
        errors.noise_floor_from_frames(relabelled, qe_frames[5:])


def test_numbers_for_table_meta_fields(
    tiny_mace: Any, tiny_calc: Any, qe_frames: list[Frame]
) -> None:
    t3 = retag(qe_frames, "omat24", "omat24")
    table = errors.evaluate(
        tiny_mace.model_path,
        "Default",
        t3,
        "T3",
        None,
        BOOT_N,
        1,
        energy_scale="qe",
        model_label="B1",
        calc=tiny_calc,
    )
    assert table.noise_floor_f is None and "mae_e" not in table.metrics
    numbers = errors.numbers_for_table(table, e0_source="E0s_qe.json", energy_scale="qe")
    keys = [k for k in numbers if not k.endswith("@meta")]
    assert keys == [
        "eval.errors.T3.B1.mae_f",
        "eval.errors.T3.B1.rmse_f",
        "eval.errors.T3.B1.mae_s",
    ]
    meta = numbers["eval.errors.T3.B1.mae_f@meta"]
    assert meta["reference"] == {
        "code": "vasp",
        "functional": "PBE",
        "pseudos": "PAW (OMat24)",
        "e0_source": "E0s_qe.json",
        "cross_functional": False,
    }
    assert meta["head"] == "Default" and meta["n"] == 15 and meta["seed"] == 1
    assert meta["ci95"] and meta["e0_source"] == "E0s_qe.json" and meta["energy_scale"] == "qe"
    assert meta["model_label"] == meta["bracket"] == "B1" and meta["tier"] == "T3"
    assert meta["noise_floor_f"] is None and "noise_floor_reason" in meta
    with_floor = errors.numbers_for_table(
        table.model_copy(update={"noise_floor_f": 8.5}), e0_source="x", energy_scale="qe"
    )
    assert with_floor["eval.errors.T3.B1.mae_f@meta"]["noise_floor_f"] == 8.5
    assert "noise_floor_reason" not in with_floor["eval.errors.T3.B1.mae_f@meta"]
    t0 = errors.numbers_for_table(
        table.model_copy(update={"tier": "T0"}), e0_source="x", energy_scale="qe"
    )
    assert "noise_floor_f" not in t0["eval.errors.T0.B1.mae_f@meta"]


# --- run() ----------------------------------------------------------------------------------------


def test_run_with_split_tiers_and_checkpoint(
    tiny_mace: Any, qe_frames: list[Frame], eval_settings: Settings, tmp_path: Path
) -> None:
    t3 = retag(qe_frames[:4], "omat24", "omat24")
    frames = qe_frames[4:] + t3
    frames_path = tmp_path / "frames.extxyz"
    write_frames(frames, frames_path)
    split = _split(
        frames,
        {"T0": [f.frame_id for f in qe_frames[4:10]], "T3": [f.frame_id for f in t3], "T4b": []},
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    ckpt = _checkpoint(tiny_mace, tmp_path / "checkpoint.json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_stage(
            "eval.errors", eval_settings, errors.run, seed=2,
            model=tiny_mace.model_path, split=split_path, frames_path=frames_path,
            checkpoint_json=ckpt,
        )  # fmt: skip
    assert result.status == "ok", result.summary
    assert any("noise floor" in str(w.message) for w in caught)  # T3 without a floor warns
    assert result.summary["tiers"] == "T0,T3" and result.summary["model_label"] == "B3"
    assert result.summary["T0.energy_metrics"] == "ok"
    assert result.summary["T3.energy_metrics"].startswith(
        "skipped: model energy scale 'qe' != frames' 'omat24'"
    )
    assert result.summary["noise_floor_f"] == "missing"
    run_dir = Path(result.manifest_path).parent
    names = {Path(a.path).name for a in result.outputs}
    assert {"errors_T0.json", "errors_T3.json", "numbers.json", "plan.json"} <= names
    t0 = ErrorTable.model_validate_json((run_dir / "errors_T0.json").read_text())
    assert (
        t0.n_frames == 6
        and t0.split_id == "eval-split"
        and t0.bootstrap_seed == 2
        and "mae_e" in t0.metrics
    )
    t3_table = ErrorTable.model_validate_json((run_dir / "errors_T3.json").read_text())
    assert (
        t3_table.reference.code == "vasp"
        and t3_table.noise_floor_f is None
        and "mae_e" not in t3_table.metrics
    )
    numbers = json.loads((run_dir / "numbers.json").read_text())
    assert "eval.errors.T0.B3.mae_e" in numbers and "eval.errors.T3.B3.mae_e" not in numbers
    assert numbers["eval.errors.T3.B3.mae_f@meta"]["noise_floor_f"] is None
    assert numbers["eval.errors.T0.B3.mae_f@meta"]["e0_source"] == "estimated"
    manifest = read_manifest(result.manifest_path)
    assert (
        manifest.extras["model_sha256"] == tiny_mace.sha256 and manifest.extras["head"] == "Default"
    )
    assert manifest.extras["tier"] == ["T0", "T3"] and manifest.extras["n"] == {"T0": 6, "T3": 4}
    assert (
        manifest.extras["bootstrap_seed"] == 2
        and manifest.extras["reference"]["T3"]["code"] == "vasp"
    )
    assert {Path(a.path).name for a in manifest.inputs} >= {
        "frames.extxyz",
        "split.json",
        "checkpoint.json",
    }


def test_run_noise_floor_frames_and_explicit_floor(
    tiny_mace: Any, qe_frames: list[Frame], eval_settings: Settings, tmp_path: Path
) -> None:
    omat = retag(qe_frames[:6], "omat24", "omat24")
    frames_path = tmp_path / "omat.extxyz"
    write_frames(omat, frames_path)
    rng = np.random.default_rng(1)
    relabelled = [
        f.model_copy(
            update={
                "forces": (np.asarray(f.forces) + rng.normal(0, 0.02, (8, 3))).tolist(),
                "label_source": "qe",
                "energy_scale": "qe",
            }
        )
        for f in omat[:3]
    ]
    qe_path = tmp_path / "relabelled.extxyz"
    write_frames(relabelled, qe_path)
    result = run_stage(
        "eval.errors", eval_settings, errors.run, model=tiny_mace.model_path,
        frames_path=frames_path, tier="T3", energy_scale="omat24", label="B0",
        e0_source="foundation", noise_floor_frames=qe_path,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    assert (
        isinstance(result.summary["noise_floor_f"], float)
        and result.summary["T3.energy_metrics"] == "ok"
    )
    run_dir = Path(result.manifest_path).parent
    numbers = json.loads((run_dir / "numbers.json").read_text())
    meta = numbers["eval.errors.T3.B0.mae_f@meta"]
    assert meta["noise_floor_f"] == pytest.approx(result.summary["noise_floor_f"], abs=1e-3)
    assert meta["e0_source"] == "foundation" and meta["energy_scale"] == "omat24"
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["noise_floor"]["n_frames"] == 3
    explicit = run_stage(
        "eval.errors", eval_settings, errors.run, model=tiny_mace.model_path,
        frames_path=frames_path, tier="T3", energy_scale="omat24", noise_floor_f=12.5,
        bootstrap_n=20,
    )  # fmt: skip
    assert explicit.status == "ok" and explicit.summary["noise_floor_f"] == 12.5
    table = ErrorTable.model_validate_json(
        (Path(explicit.manifest_path).parent / "errors_T3.json").read_text()
    )
    assert (
        table.noise_floor_f == 12.5 and table.bootstrap_n == 20 and table.model_label == "tiny_b20"
    )


def test_run_dry_run_and_failures(
    tiny_mace: Any, qe_frames_path: Path, eval_settings: Settings, tmp_path: Path
) -> None:
    dry = run_stage(
        "eval.errors",
        eval_settings,
        errors.run,
        dry_run=True,
        model=tiny_mace.model_path,
        frames_path=qe_frames_path,
        energy_scale="qe",
    )
    assert dry.status == "partial" and dry.outputs == [] and dry.summary["n_frames"] == 15
    assert read_manifest(dry.manifest_path).extras["planned"] is True

    def failing(**kw: Any) -> Any:
        return run_stage("eval.errors", eval_settings, errors.run, model=tiny_mace.model_path, **kw)

    r = failing(frames_path=tmp_path / "nope.extxyz")
    assert r.status == "failed" and "not found" in r.summary["error"]
    r = run_stage(
        "eval.errors",
        eval_settings,
        errors.run,
        model=tmp_path / "absent.model",
        frames_path=qe_frames_path,
    )
    assert r.status == "failed" and "model not found" in r.summary["error"]
    r = failing()
    assert r.status == "failed" and "--split" in r.summary["error"]
    r = failing(frames_path=qe_frames_path, head="pt_head")
    assert r.status == "failed"
    r = failing(frames_path=qe_frames_path, energy_scale="vasp")
    assert r.status == "failed" and "energy_scale" in r.summary["error"]
    bad = _checkpoint(tiny_mace, tmp_path / "bad.json", sha256="0" * 64)
    r = failing(frames_path=qe_frames_path, checkpoint_json=bad)
    assert r.status == "failed" and "does not match" in r.summary["error"]
    other_head = _checkpoint(tiny_mace, tmp_path / "heads.json", heads=["pt_head"])
    r = failing(frames_path=qe_frames_path, checkpoint_json=other_head)
    assert r.status == "failed" and "heads" in r.summary["error"]
    r = failing(frames_path=qe_frames_path, tier="T8")
    assert r.status == "failed" and "unknown tier" in r.summary["error"]
    assert sha256_file(tiny_mace.model_path) == tiny_mace.sha256


def test_aggregate_means_over_seeds_and_keeps_provenance() -> None:
    """Seed aggregation: value = mean, ci95 = min-max over seeds, sources recorded."""
    from b20mlip.evaluate.aggregate import aggregate

    def numbers(seed: int, mae: float) -> dict:
        return {
            "eval.errors.T0.B1.mae_f": mae,
            "eval.errors.T0.B1.mae_f@meta": {"seed": seed, "n": 147, "ci95": [mae - 5, mae + 5],
                                             "head": "Default", "tier": "T0"},
        }  # fmt: skip

    out = aggregate({"r0": numbers(0, 63.0), "r1": numbers(1, 66.0), "r2": numbers(2, 69.0)}, "B1")
    assert out["eval.errors.T0.B1.mae_f"] == pytest.approx(66.0)
    meta = out["eval.errors.T0.B1.mae_f@meta"]
    assert meta["ci95"] == [63.0, 69.0] and meta["seed"] == "0,1,2" and meta["n_seeds"] == 3
    assert [p["run_id"] for p in meta["per_seed"]] == ["r0", "r1", "r2"]
    assert meta["n"] == 147 and meta["aggregate"] == "mean_over_seeds"
    single = aggregate({"r0": numbers(0, 63.0)}, "B1")
    assert single["eval.errors.T0.B1.mae_f@meta"]["ci95"] == [58.0, 68.0]  # per-seed CI kept
