"""`b20mlip active select` through the root CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from b20mlip.active.cli import COMMANDS
from b20mlip.cli import REGISTERED, app
from b20mlip.io import read_frames
from b20mlip.report import numbers as nums

runner = CliRunner()


def _payload(output: str) -> dict[str, Any]:
    obj, _ = json.JSONDecoder().raw_decode(output[output.index("{") :])
    return obj


def test_registration_and_help() -> None:
    assert REGISTERED["active"] == set(COMMANDS) == {"select"}
    result = runner.invoke(app, ["active", "select", "--help"])
    assert result.exit_code == 0 and "--models" in result.output and "[stub]" not in result.output
    assert runner.invoke(app, ["active", "select"]).exit_code == 2  # --models/--frames required


def test_select_end_to_end(
    committee_paths: list[Path], frames_path: Path, overlay: Path, tmp_path: Path
) -> None:
    out = tmp_path / "candidates_r1.extxyz"
    result = runner.invoke(
        app, ["--config", str(overlay), "active", "select",
              "--models", ",".join(str(p) for p in committee_paths),
              "--frames", str(frames_path), "--n", "4", "--out", str(out), "--per-group-max", "1"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert (
        payload["stage"] == "active.select"
        and payload["status"] == "ok"
        and payload["summary"]["n_selected"] == 4
    )
    assert len(read_frames(out)) == 4
    harvest = nums.harvest(tmp_path / "runs")
    assert harvest.entries["active.n_selected"].value == 4.0
    assert (
        abs(harvest.entries["active.sigma_f_median"].value) < 1e-8
    )  # identical weights, float noise
    dry = runner.invoke(
        app,
        [
            "--config",
            str(overlay),
            "--dry-run",
            "active",
            "select",
            "--models",
            f"{committee_paths[0]},{committee_paths[1]}",
            "--frames",
            str(frames_path),
        ],
    )
    assert dry.exit_code == 1 and _payload(dry.output)["status"] == "partial"
    bad = runner.invoke(
        app,
        [
            "--config",
            str(overlay),
            "active",
            "select",
            "--models",
            str(committee_paths[0]),
            "--frames",
            str(frames_path),
        ],
    )
    assert bad.exit_code == 1 and "at least 2" in _payload(bad.output)["summary"]["error"]
