"""`b20mlip train` / `b20mlip export` through the root CLI (plugin hook)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.models import CheckpointInfo
from b20mlip.provenance import read_manifest, sha256_file

from .conftest import TINY_TRAIN

runner = CliRunner()


def _payload(output: str) -> dict:
    """The StageResult JSON; MACE prints a one-line cuequivariance notice before it."""
    return json.loads(output[output.index("{") :])


def _overlay(tmp_path: Path, **train: object) -> Path:
    data = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "runs_dir": str(tmp_path / "runs"),
            "models_dir": str(tmp_path / "models"),
        },
        "train": {**TINY_TRAIN, **train},
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_help_shows_train_and_export_as_real_commands() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0 and "--variant" in result.output and "[stub]" not in result.output
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0 and "--model" in result.output and "[stub]" not in result.output


def test_train_scratch_end_to_end(tmp_path: Path, tiny_b20_path: Path) -> None:
    overlay = _overlay(tmp_path)
    out = tmp_path / "final"
    result = runner.invoke(
        app,
        ["--config", str(overlay), "train", "--variant", "scratch", "--frames", str(tiny_b20_path),
         "--seed", "1", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "train" and payload["status"] == "ok"
    assert payload["summary"]["variant"] == "scratch"
    assert payload["summary"]["e0_source"] == "estimated"
    assert payload["summary"]["epochs"] == 1 and "val_rmse_f_meV_A" in payload["summary"]

    manifest = read_manifest(payload["manifest"])
    run_dir = Path(payload["manifest"]).parent
    assert manifest.status == "ok" and manifest.seed == 1 and manifest.stage == "train"
    assert manifest.extras["heads"] == ["Default"]
    assert manifest.extras["e0_source"] == "estimated"
    assert manifest.extras["energy_scale"] == "qe"  # 3 qe-tagged frames among unlabelled ones
    assert manifest.extras["foundation_sha256"] is None
    assert manifest.extras["mace_returncode"] == 0
    info = CheckpointInfo.model_validate_json((run_dir / "checkpoint.json").read_text())
    assert info.model_path == str(run_dir / "models" / "b20_scratch.model")
    assert info.sha256 == sha256_file(info.model_path) == payload["summary"]["sha256"]
    assert info.train_run_id == manifest.run_id and info.lammps_path is None
    assert info.seed == 1 and info.lr == 1e-4 and info.batch_size == 4 and info.split_id is None
    assert info.val_metrics["rmse_f_meV_A"] > 0 and "rmse_e_per_atom_meV" in info.val_metrics
    names = {Path(a.path).name for a in manifest.outputs}
    assert {"b20_scratch.model", "checkpoint.json", "plan.json", "train.extxyz", "valid.extxyz",
            "mace_stdout.log", "b20_scratch_run-1_train.txt"} <= names  # fmt: skip
    assert (out / "b20_scratch.model").is_file() and (out / "checkpoint.json").is_file()
    assert sha256_file(out / "b20_scratch.model") == info.sha256


def test_train_dry_run_and_failures(tmp_path: Path, tiny_b20_path: Path) -> None:
    overlay = _overlay(tmp_path)
    result = runner.invoke(
        app, ["--config", str(overlay), "--dry-run", "train", "--variant", "scratch",
              "--frames", str(tiny_b20_path)],
    )  # fmt: skip
    assert result.exit_code == 1 and _payload(result.output)["status"] == "partial"
    assert _payload(result.output)["summary"] == {"planned": 1, "variant": "scratch"}

    # naive without local foundation weights fails before touching MACE
    no_weights = _overlay(tmp_path, foundation=str(tmp_path / "absent-foundation.model"))
    result = runner.invoke(
        app, ["--config", str(no_weights), "train", "--variant", "naive", "--frames",
              str(tiny_b20_path)],
    )  # fmt: skip
    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["status"] == "failed" and "not found locally" in payload["summary"]["error"]

    result = runner.invoke(app, ["--config", str(overlay), "train"])
    assert result.exit_code == 1 and "--split" in _payload(result.output)["summary"]["error"]
    result = runner.invoke(app, ["train", "--variant", "lora"])
    assert result.exit_code == 2  # typer rejects the enum value


def test_export_cli_dry_run_and_missing_model(tmp_path: Path, tiny_mace: object) -> None:
    overlay = _overlay(tmp_path)
    model = getattr(tiny_mace, "model_path")  # noqa: B009
    result = runner.invoke(
        app, ["--config", str(overlay), "--dry-run", "export", "--model", str(model)]
    )
    assert result.exit_code == 1, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "export" and payload["status"] == "partial"
    assert payload["summary"]["lammps_sha256"] == ""
    result = runner.invoke(
        app, ["--config", str(overlay), "export", "--model", str(tmp_path / "nope.model")]
    )
    assert result.exit_code == 1 and _payload(result.output)["status"] == "failed"


def test_train_with_relative_runs_dir(
    tmp_path: Path, tiny_b20_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (2026-09-19): with the default relative ``paths.runs_dir`` ("runs") the MACE
    subprocess ran with ``cwd=out_dir`` and could not find ``runs/train/<id>/data/train.extxyz``."""
    monkeypatch.chdir(tmp_path)
    data = {
        "paths": {"data_dir": "data", "runs_dir": "runs", "models_dir": "models"},
        "train": {**TINY_TRAIN},
    }
    overlay = tmp_path / "rel.yaml"
    overlay.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = runner.invoke(
        app,
        ["--config", str(overlay), "train", "--variant", "scratch", "--frames", str(tiny_b20_path),
         "--seed", "2"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["status"] == "ok"
    assert Path(payload["manifest"]).parent.parts[:2] == ("runs", "train") or (
        Path(payload["manifest"]).is_absolute()
    )
