"""`b20mlip agent run|eval` and `b20mlip screen` through the root CLI (smoke), including the
anthropic backend refusing to start without ANTHROPIC_API_KEY (no client is constructed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from b20mlip.agent.cli import COMMANDS, TOPLEVEL_COMMANDS
from b20mlip.cli import REGISTERED, app
from b20mlip.cli import _toplevel_registered as TOPLEVEL_REGISTERED
from b20mlip.provenance import read_manifest

from .conftest import BOOT_N, FAST_MD, TASKS_PATH, populate

runner = CliRunner()


def _payload(output: str) -> dict[str, Any]:
    obj, _ = json.JSONDecoder().raw_decode(output[output.index("{") :])
    return obj


@pytest.fixture
def overlay(tmp_path: Path, tiny_mace: Any, tiny_frames: Any, refs_dir: Path) -> Path:
    populate(tmp_path, tiny_frames, Path(tiny_mace.model_path))
    data = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "runs_dir": str(tmp_path / "runs"),
            "models_dir": str(tmp_path / "models"),
            "dft_dir": str(tmp_path / "dft"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "agent": {
            "phonon_supercell": [1, 1, 1],
            "refs_dir": str(refs_dir),
            "trace_dir": str(tmp_path / "traces"),
            "tasks_path": str(TASKS_PATH),
        },
        "eval": {"bootstrap_n": BOOT_N, "max_steps": 20},
        "md": dict(FAST_MD),
    }
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_registration_and_help() -> None:
    assert REGISTERED["agent"] == set(COMMANDS) == {"run", "eval"}
    assert TOPLEVEL_COMMANDS <= TOPLEVEL_REGISTERED
    for argv, needle in (
        (["agent", "run", "--help"], "--backend"),
        (["agent", "eval", "--help"], "--gold"),
        (["screen", "--help"], "--compounds"),
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0 and needle in result.output and "[stub]" not in result.output
    result = runner.invoke(app, ["agent", "run", "--backend", "scripted"])
    assert result.exit_code != 0 and "exactly one of" in result.output


def test_run_scripted_prompt_and_record(overlay: Path, tiny_mace: Any, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--config", str(overlay), "--seed", "2", "agent", "run", "--backend", "scripted",
         "--prompt", "Relax FeSi with the configured MLIP and report a_A",
         "--model", str(tiny_mace.model_path), "--budget", "ci", "--record"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "agent.run" and payload["status"] == "ok"
    summary = payload["summary"]
    assert (
        summary["backend"] == "scripted" and summary["tool_calls"] == 2 and summary["grounded"] == 1
    )
    assert json.loads(summary["answer"])["a_A"] == pytest.approx(4.48, abs=0.05)
    recorded = Path(summary["recorded_trace"])
    assert recorded.parent == tmp_path / "traces" and recorded.is_file()
    manifest = read_manifest(Path(payload["manifest"]))
    assert manifest.extras["backend"] == "scripted" and manifest.seed == 2
    assert any(p.endswith("numbers.json") for p in payload["outputs"])
    # the recorded trace replays through the mock backend
    task_id = recorded.stem
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "run", "--backend", "mock", "--trace", str(recorded),
         "--prompt", "Relax FeSi with the configured MLIP and report a_A",
         "--model", str(tiny_mace.model_path)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert _payload(result.output)["summary"]["backend"] == "mock" and task_id.startswith("prompt-")
    # a dry run plans only and exits 1 (status partial)
    result = runner.invoke(
        app,
        ["--config", str(overlay), "--dry-run", "agent", "run", "--backend", "scripted",
         "--prompt", "Relax FeSi", "--model", str(tiny_mace.model_path)],
    )  # fmt: skip
    assert result.exit_code == 1 and _payload(result.output)["status"] == "partial"


def test_run_task_from_file_with_budget_json(overlay: Path, tiny_mace: Any) -> None:
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "run", "--backend", "scripted",
         "--task", f"{TASKS_PATH}:t11-inventory-relax-report-MnSi-budget",
         "--model", str(tiny_mace.model_path), "--budget", '{"max_calls": 6}', "--approve-cluster"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = _payload(result.output)["summary"]
    assert summary["task_id"] == "t11-inventory-relax-report-MnSi-budget"
    assert (
        summary["violations"] == 1 and summary["invalid_calls"] == 0
    )  # the injected budget still bites (a violation), but is not the agent's invalid call
    result = runner.invoke(
        app,
        [
            "--config",
            str(overlay),
            "agent",
            "run",
            "--backend",
            "scripted",
            "--task",
            f"{TASKS_PATH}",
        ],
    )
    assert result.exit_code != 0 and "TASKS.jsonl:ID" in result.output


def test_screen_then_eval_mock_limit_2(
    overlay: Path, tiny_mace: Any, traces_dir: Path, tmp_path: Path
) -> None:
    gold = tmp_path / "gold_tiny_b20.json"
    result = runner.invoke(
        app,
        ["--config", str(overlay), "screen", "--compounds", "FeSi,MnSi",
         "--model", str(tiny_mace.model_path), "--tasks", str(TASKS_PATH), "--out", str(gold),
         "--limit", "2"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "agent.eval" and payload["summary"]["n_gold"] == 2
    assert payload["summary"]["model_label"] == "tiny_b20"
    data = json.loads(gold.read_text())
    assert data["schema"] == "b20mlip.agent.gold.v1" and set(data["tasks"]) == {
        "t01-phonons-imaginary-FeSi", "t02-relax-a0-MnSi",
    }  # fmt: skip
    assert read_manifest(Path(payload["manifest"])).extras["screen"] is True
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "eval", "--backend", "mock", "--limit", "2",
         "--trace-dir", str(traces_dir), "--gold", str(gold), "--model", str(tiny_mace.model_path),
         "--bootstrap-n", "20"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    summary = payload["summary"]
    assert payload["status"] == "ok" and summary["n_tasks"] == 2 and summary["backend"] == "mock"
    assert (
        summary["accuracy"] == 1.0
        and summary["provenance_rate"] == 1.0
        and summary["tokens_in"] > 0
    )
    assert any(p.endswith("numbers.json") for p in payload["outputs"])
    # the default gold path lives next to the task file and the default trace dir is the config's
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "eval", "--backend", "mock", "--limit", "1",
         "--model", str(tiny_mace.model_path)],
    )  # fmt: skip
    assert (
        result.exit_code == 1 and _payload(result.output)["status"] == "failed"
    )  # no traces recorded there


def test_anthropic_backend_needs_a_key_and_never_builds_a_client(
    overlay: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anthropic

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("anthropic.Anthropic() must not be constructed")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", forbidden)
    for argv in (
        ["agent", "run", "--backend", "anthropic", "--prompt", "Relax FeSi"],
        ["agent", "eval", "--backend", "anthropic", "--limit", "1"],
    ):
        result = runner.invoke(app, ["--config", str(overlay), *argv])
        assert result.exit_code == 2, result.output
        assert "ANTHROPIC_API_KEY is not set" in result.output and "--backend mock" in result.output
    assert not (Path(overlay).parent / "runs").exists()


def test_run_with_registered_dataset_and_gold(
    overlay: Path, tiny_mace: Any, tmp_path: Path, tiny_frames: Any
) -> None:
    from b20mlip.io import write_frames

    extra = tmp_path / "extra_frames.extxyz"
    write_frames(
        [
            f.model_copy(update={"energy_scale": "qe", "label_source": "qe"})
            for f in tiny_frames[:6]
        ],
        extra,
    )
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {"schema": "b20mlip.agent.gold.v1", "tasks": {"prompt-mae": {"gold": {"n_frames": 6}}}}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "run", "--backend", "scripted",
         "--prompt", "Evaluate the force MAE on tier T0 of dataset extra and report mae_f",
         "--model", str(tiny_mace.model_path), "--frames", f"extra={extra}",
         "--gold", str(gold)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = _payload(result.output)["summary"]
    assert json.loads(summary["answer"])["n_frames"] == 6
    assert summary["task_id"] != "prompt-mae" and "accuracy" not in summary  # gold keyed by id
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "run", "--backend", "scripted", "--prompt", "x",
         "--frames", "=nolabel"],
    )  # fmt: skip
    assert result.exit_code != 0 and "LABEL=PATH" in result.output


def test_run_scores_against_gold_by_task_id(overlay: Path, tiny_mace: Any, tmp_path: Path) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "schema": "b20mlip.agent.gold.v1",
                "tasks": {"t02-relax-a0-MnSi": {"gold": {"a_A": 4.56}}},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["--config", str(overlay), "agent", "run", "--backend", "scripted",
         "--task", f"{TASKS_PATH}:t02-relax-a0-MnSi", "--model", str(tiny_mace.model_path),
         "--gold", str(gold)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = _payload(result.output)["summary"]
    assert summary["accuracy"] == 1.0 and summary["correct"] == 1
