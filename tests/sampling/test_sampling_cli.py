"""`b20mlip sampling neb|umbrella|wham` through the root CLI, and the stage functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from b20mlip.cli import REGISTERED, app
from b20mlip.config import Settings
from b20mlip.provenance import read_manifest, run_stage
from b20mlip.sampling import umbrella, wham

from .conftest import DW_WINDOWS, DoubleWell

runner = CliRunner()
EXPECTED_NEB_META = {"reference", "e0_source", "head", "n", "seed", "ci95", "model_label", "T"}


def _payload(output: str) -> dict[str, Any]:
    """The StageResult JSON; MACE prints a one-line cuequivariance notice before it."""
    return json.loads(output[output.index("{") :])


def _runs(tmp_path: Path) -> list[str]:
    return [
        "--set",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "--set",
        f"paths.data_dir={tmp_path / 'data'}",
    ]


def _gate_a3(runs_dir: Path, tmp_path: Path) -> list[str]:
    """Harvest ``runs_dir`` like ``report build`` does and run the real gate A3 on it."""
    audit = pytest.importorskip("b20mlip.report.audit")
    numbers = pytest.importorskip("b20mlip.report.numbers")
    harvest = numbers.harvest(runs_dir)
    entries = {k: v for k, v in harvest.as_json().items() if not k.startswith("@")}
    readme = tmp_path / "README.md"
    readme.write_text("# stub\n", encoding="utf-8")
    view = audit.Audit(
        readme_path=readme, text="# stub\n", entries=entries, stale=[], numbers_text="{}",
        runs_dir=runs_dir, strict=True,
    )  # fmt: skip
    return audit.gate_a3(view)


def test_commands_are_registered_with_real_options() -> None:
    assert REGISTERED["sampling"] == {"neb", "umbrella", "wham"}
    for command, option in (("neb", "--images"), ("umbrella", "--windows"), ("wham", "--run")):
        result = runner.invoke(app, ["sampling", command, "--help"])
        assert result.exit_code == 0 and option in result.output and "[stub]" not in result.output


def test_wham_cli_on_a_synthetic_umbrella_run(
    synthetic_run: Path, double_well: DoubleWell, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, [*_runs(tmp_path), "--seed", "3", "sampling", "wham", "--run", str(synthetic_run)]
    )
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "sampling.wham" and payload["status"] == "ok"
    summary = payload["summary"]
    assert summary["dF_eV"] == pytest.approx(double_well.dF_true, abs=0.02)
    assert summary["dF_wham_eV"] == pytest.approx(summary["dF_eV"], abs=0.01)
    assert summary["n_windows"] == DW_WINDOWS and summary["method"] == "mbar"
    assert summary["overlap_ok"] == 1 and summary["compound"] == "FeSi"
    run_dir = Path(payload["manifest"]).parent
    names = {Path(p).name for p in payload["outputs"]}
    assert names == {"pmf.json", "numbers.json"}
    manifest = read_manifest(payload["manifest"])
    assert manifest.status == "ok" and manifest.seed == 3 and manifest.stage == "sampling.wham"
    assert manifest.extras["method"] == "mbar" and manifest.extras["crosscheck_method"] == "wham"
    assert manifest.extras["model_label"] == "B1" and manifest.extras["n_windows"] == DW_WINDOWS
    assert {Path(a.path).name for a in manifest.inputs} == {"windows.json"} | {
        f"window_{i:02d}.npz" for i in range(DW_WINDOWS)
    }
    pmf = json.loads((run_dir / "pmf.json").read_text())
    assert pmf["crosscheck"]["method"] == "wham" and len(pmf["F_eV"]) == pmf["n_bins"]
    assert pmf["model_info"]["label"] == "B1" and pmf["umbrella_index"]["compound"] == "FeSi"
    assert all(v is None or np.isfinite(v) for v in pmf["F_eV"])  # NaN never reaches JSON

    numbers = json.loads((run_dir / "numbers.json").read_text())
    keys = {k for k in numbers if not k.endswith("@meta")}
    assert keys == {
        "sampling.wham.FeSi.B1.dF_eV",
        "sampling.wham.FeSi.B1.dF_block_err_eV",
        "sampling.wham.FeSi.B1.n_windows",
        "sampling.wham.FeSi.B1.dF_barrier_eV",
        "sampling.umbrella.FeSi.dF_eV",
        "sampling.umbrella.FeSi.dF_err_eV",
    }
    assert numbers["sampling.wham.FeSi.B1.dF_eV"] == summary["dF_eV"]
    assert numbers["sampling.umbrella.FeSi.dF_eV"] == summary["dF_eV"]
    assert numbers["sampling.wham.FeSi.B1.n_windows"] == DW_WINDOWS
    meta = numbers["sampling.wham.FeSi.B1.dF_eV@meta"]
    assert meta["reference"] == {
        "code": "mace", "functional": "PBE", "pseudos": None, "e0_source": "E0s_qe.json",
        "cross_functional": False,
    }  # fmt: skip
    assert meta["e0_source"] == "E0s_qe.json" and meta["head"] == "Default" and meta["seed"] == 3
    assert meta["n"] == summary["n_samples"] and meta["method"] == "mbar"
    assert meta["T"] == pytest.approx(double_well.T) and meta["energy_scale"] == "qe"
    lo, hi = meta["ci95"]
    assert lo < summary["dF_eV"] < hi and meta["n_windows"] == DW_WINDOWS
    err_meta = numbers["sampling.wham.FeSi.B1.dF_block_err_eV@meta"]
    assert err_meta["ci95"] is None and "block" in err_meta["ci95_reason"]
    assert numbers["sampling.wham.FeSi.B1.n_windows@meta"]["ci95_reason"] == "count"
    assert _gate_a3(tmp_path / "runs", tmp_path) == []

    # the other estimator as primary, and a dry run
    result = runner.invoke(
        app, [*_runs(tmp_path), "sampling", "wham", "--run", str(synthetic_run), "--method", "wham",
              "--blocks", "4", "--bins", "70"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = _payload(result.output)["summary"]
    assert summary["method"] == "wham" and "dF_mbar_eV" in summary
    assert summary["dF_eV"] == pytest.approx(double_well.dF_true, abs=0.02)
    result = runner.invoke(
        app, [*_runs(tmp_path), "--dry-run", "sampling", "wham", "--run", str(synthetic_run)]
    )
    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["status"] == "partial" and payload["outputs"] == []
    assert payload["summary"] == {
        "planned": 1, "n_windows": DW_WINDOWS, "method": "mbar", "compound": "FeSi"
    }  # fmt: skip
    result = runner.invoke(
        app, [*_runs(tmp_path), "sampling", "wham", "--run", str(tmp_path / "absent")]
    )
    assert result.exit_code == 1 and _payload(result.output)["status"] == "failed"


def test_wham_stage_flags_poor_overlap(synthetic_run: Path, settings: Settings) -> None:
    result = run_stage(
        "sampling.wham", settings, wham.run, seed=0, run_dir=synthetic_run, min_overlap=0.99
    )
    assert result.status == "partial" and "overlap" in str(result.summary["note"])
    assert {Path(a.path).name for a in result.outputs} == {"pmf.json", "numbers.json"}
    bad = run_stage("sampling.wham", settings, wham.run, run_dir=synthetic_run, method="tram")
    assert bad.status == "failed" and "method" in str(bad.summary["error"])
    empty = synthetic_run.parent / "empty"
    empty.mkdir()
    index = json.loads((synthetic_run / umbrella.INDEX_NAME).read_text())
    index["files"] = []
    (empty / umbrella.INDEX_NAME).write_text(json.dumps(index))
    failed = run_stage("sampling.wham", settings, wham.run, run_dir=empty)
    assert failed.status == "failed" and "no completed windows" in str(failed.summary["error"])


def test_neb_cli_with_the_tiny_model(tiny_mace: Any, tmp_path: Path) -> None:
    model = str(tiny_mace.model_path)
    result = runner.invoke(
        app, [*_runs(tmp_path), "--seed", "2", "sampling", "neb", "--model", model,
              "--compound", "FeSi", "--supercell", "1", "1", "1", "--images", "3",
              "--relax-steps", "2", "--neb-steps", "2", "--label", "B3"],
    )  # fmt: skip
    assert result.exit_code in (0, 1), result.output
    payload = _payload(result.output)
    assert payload["stage"] == "sampling.neb"
    summary = payload["summary"]
    assert summary["n_images"] == 3 and summary["compound"] == "FeSi" and summary["steps"] <= 2
    assert summary["model_label"] == "B3" and isinstance(summary["E_a_eV"], float)
    converged = bool(summary["converged"])
    assert payload["status"] == ("ok" if converged else "partial")
    assert result.exit_code == (0 if converged else 1)
    names = {Path(p).name for p in payload["outputs"]}
    assert names == {"initial.extxyz", "final.extxyz", "neb_path.traj", "neb.json", "numbers.json"}
    run_dir = Path(payload["manifest"]).parent
    manifest = read_manifest(payload["manifest"])
    assert manifest.seed == 2 and manifest.extras["model_sha256"] == tiny_mace.sha256
    assert manifest.extras["hop"]["species"] == "Si" and manifest.extras["n_images"] == 3
    assert manifest.extras["structure_source"] == "tabulated"
    assert [Path(a.path).name for a in manifest.inputs] == [Path(model).name]
    document = json.loads((run_dir / "neb.json").read_text())
    assert document["n_atoms"] == 7 and len(document["images"]) == 3
    assert len(document["cv_along_path"]) == 3 and document["model_info"]["label"] == "B3"
    numbers = json.loads((run_dir / "numbers.json").read_text())
    keys = {k for k in numbers if not k.endswith("@meta")}
    assert keys == {
        "sampling.neb.FeSi.B3.E_a_eV",
        "sampling.neb.FeSi.B3.E_a_reverse_eV",
        "sampling.neb.FeSi.B3.dE_eV",
        "sampling.neb.FeSi.Ea_eV",
    }
    meta = numbers["sampling.neb.FeSi.B3.E_a_eV@meta"]
    assert EXPECTED_NEB_META <= set(meta)
    assert meta["ci95"] is None and "NEB" in meta["ci95_reason"] and meta["n"] == 3
    assert meta["T"] == 0.0 and meta["reference"]["code"] == "mace" and meta["seed"] == 2
    assert meta["e0_source"] == "unknown"  # no checkpoint.json beside the in-test model
    assert _gate_a3(tmp_path / "runs", tmp_path) == []

    # explicit end states from the files the first run wrote
    result = runner.invoke(
        app, [*_runs(tmp_path), "--dry-run", "sampling", "neb", "--model", model,
              "--initial", str(run_dir / "initial.extxyz"),
              "--final", str(run_dir / "final.extxyz")],
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    payload = _payload(result.output)
    assert payload["status"] == "partial" and payload["summary"]["planned"] == 1
    assert payload["summary"]["compound"] == "FeSi" and payload["summary"]["n_atoms"] == 7
    for argv in (
        ["sampling", "neb", "--model", model],  # neither compound nor end states
        ["sampling", "neb", "--model", model, "--initial", str(run_dir / "initial.extxyz")],
        ["sampling", "neb", "--model", str(tmp_path / "nope.model"), "--compound", "FeSi"],
    ):
        result = runner.invoke(app, [*_runs(tmp_path), *argv])
        assert result.exit_code == 1 and _payload(result.output)["status"] == "failed"


def test_umbrella_then_wham_with_the_tiny_model(tiny_mace: Any, tmp_path: Path) -> None:
    model = str(tiny_mace.model_path)
    result = runner.invoke(
        app, [*_runs(tmp_path), "--seed", "1", "sampling", "umbrella", "--model", model,
              "--compound", "FeSi", "--supercell", "1", "1", "1", "--windows", "3", "--ps", "0.04",
              "--T", "300", "--label", "B3", "--init", "interp"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "sampling.umbrella" and payload["status"] == "ok"
    summary = payload["summary"]
    assert summary["n_windows"] == 3 and summary["steps_per_window"] == 20 and summary["T"] == 300.0
    assert summary["k_eVA2"] == 5.0 and summary["k_cv_eV"] == pytest.approx(
        5.0 * summary["hop_distance_A"] ** 2
    )
    assert summary["n_atoms"] == 7 and summary["init"] == "interp" and summary["n_files"] == 3
    run_dir = Path(payload["manifest"]).parent
    names = {Path(p).name for p in payload["outputs"]}
    assert names == {"initial.extxyz", "final.extxyz", "windows.json", "window_00.npz",
                     "window_01.npz", "window_02.npz"}  # fmt: skip
    manifest = read_manifest(payload["manifest"])
    assert manifest.status == "ok" and manifest.seed == 1
    assert manifest.extras["thermostat"] == "langevin" and manifest.extras["timestep_fs"] == 2.0
    assert manifest.extras["friction_per_fs"] == 0.01 and manifest.extras["engine"] == "ase"
    assert manifest.extras["model_label"] == "B3" and "progress" in manifest.extras
    index = json.loads((run_dir / "windows.json").read_text())
    assert index["compound"] == "FeSi" and index["model_info"]["label"] == "B3"
    assert index["run_id"] == manifest.run_id and index["k"] == pytest.approx(summary["k_cv_eV"])
    assert index["centers"] == [0.0, 0.5, 1.0] and len(index["files"]) == 3

    result = runner.invoke(
        app, [*_runs(tmp_path), "sampling", "wham", "--run", str(run_dir), "--blocks", "3",
              "--min-overlap", "0"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["status"] == "ok" and payload["summary"]["n_windows"] == 3
    numbers = json.loads((Path(payload["manifest"]).parent / "numbers.json").read_text())
    assert "sampling.wham.FeSi.B3.dF_eV" in numbers and "sampling.umbrella.FeSi.dF_eV" in numbers
    meta = numbers["sampling.wham.FeSi.B3.dF_eV@meta"]
    assert meta["model_sha256"] == tiny_mace.sha256 and meta["T"] == 300.0 and meta["seed"] == 1
    assert meta["k_eVA2"] == 5.0 and meta["ps_per_window"] == 0.04
    assert _gate_a3(tmp_path / "runs", tmp_path) == []

    # dry run plans only; bad arguments fail cleanly
    result = runner.invoke(
        app, [*_runs(tmp_path), "--dry-run", "sampling", "umbrella", "--model", model,
              "--compound", "MnSi", "--windows", "4", "--ps", "0.01"],
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    payload = _payload(result.output)
    assert payload["status"] == "partial" and payload["outputs"] == []
    assert payload["summary"]["planned"] == 1 and payload["summary"]["n_atoms"] == 63
    assert payload["summary"]["compound"] == "MnSi" and payload["summary"]["n_windows"] == 4
    result = runner.invoke(
        app, [*_runs(tmp_path), "sampling", "umbrella", "--model", model, "--windows", "1"]
    )
    assert (
        result.exit_code == 1
        and "two umbrella windows" in _payload(result.output)["summary"]["error"]
    )
    result = runner.invoke(
        app, [*_runs(tmp_path), "sampling", "umbrella", "--model", str(tmp_path / "none.model")]
    )
    assert result.exit_code == 1 and _payload(result.output)["status"] == "failed"


def test_umbrella_stage_resumes_a_crashed_run(tiny_mace: Any, settings: Settings) -> None:
    kw = dict(model=tiny_mace.model_path, compound="FeSi", windows=2, ps=0.02, supercell=(1, 1, 1))
    first = run_stage("sampling.umbrella", settings, umbrella.run, seed=0, **kw)
    assert first.status == "ok" and len(first.outputs) == 5
    run_dir = Path(first.manifest_path).parent
    (run_dir / "manifest.json").unlink()  # a run that died before its manifest: resumable
    (run_dir / "window_01.npz").unlink()
    second = run_stage("sampling.umbrella", settings, umbrella.run, seed=0, resume=True, **kw)
    assert second.status == "ok" and second.run_id == first.run_id
    manifest = read_manifest(second.manifest_path)
    progress = manifest.extras["progress"]
    assert len(progress) == 2 and "skipping" in progress[0] and "20 steps" not in progress[0]
    assert "window 1" in progress[1] and "skipping" not in progress[1]
    assert (run_dir / "window_01.npz").is_file() and second.manifest_path == first.manifest_path
    bad = run_stage("sampling.umbrella", settings, umbrella.run, seed=0, init="random", **kw)
    assert bad.status == "failed" and "init must be" in str(bad.summary["error"])
