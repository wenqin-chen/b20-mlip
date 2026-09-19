"""Eval: per-task scoring, the aggregate numbers with bootstrap CIs and A10 meta, ``eval.run``
on three tasks with the mock backend against gold from ``screen``, task-file helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from b20mlip.agent import eval as eval_mod
from b20mlip.agent import screen, tasks
from b20mlip.agent.backends import MockBackend, ScriptedBackend
from b20mlip.agent.eval import METRICS, aggregate, score
from b20mlip.config import Settings
from b20mlip.models import AgentReport, AgentTask, Budget
from b20mlip.provenance import run_stage
from b20mlip.report import audit
from b20mlip.report import numbers as nums


def make_report(answer: dict[str, Any], **kw: Any) -> AgentReport:
    base: dict[str, Any] = dict(
        task_id="t", backend="scripted", model_id="scripted", answer=answer, numbers=[],
        tool_calls=4,
        invalid_calls=1, dag_valid=True, grounded=True, violations=[], tokens_in=100, tokens_out=50,
        cost_usd=0.002, wall_s=1.0, trace_path="trace.jsonl",
    )  # fmt: skip
    base.update(kw)
    return AgentReport(**base)


def test_score_per_key_tolerances() -> None:
    task = AgentTask(
        task_id="t", prompt="p", tools_allowed=[],
        gold={"a_A": 4.48, "n": 3, "compound": "FeSi", "flag": True, "e": 1.0},
        tolerance={"a_A": 0.01, "n": 0}, injected_failure="tool_error",
    )  # fmt: skip
    s = score(
        make_report({"a_A": 4.485, "n": 3, "compound": "FeSi", "flag": 1, "e": 1.0 + 5e-7}), task
    )
    assert s["accuracy"] == 1.0 and s["correct"] is True and s["recovery"] is True
    assert s["invalid_call_rate"] == 0.25 and s["tokens"] == 150 and s["usd"] == 0.002
    assert s["dag_valid"] and s["provenance"]
    assert s["per_key"] == {"a_A": True, "n": True, "compound": True, "flag": True, "e": True}
    s = score(make_report({"a_A": 4.5, "n": 3, "compound": "CoSi", "e": 1.0}), task)
    assert s["accuracy"] == 0.4 and s["correct"] is False and s["recovery"] is False
    assert s["per_key"]["a_A"] is False and s["per_key"]["flag"] is False
    s = score(
        make_report({"a_A": "abc", "n": 3.0000001, "compound": "FeSi", "flag": True, "e": 1.01}),
        task,
    )
    assert s["per_key"] == {"a_A": False, "n": False, "compound": True, "flag": True, "e": False}
    no_gold = AgentTask(task_id="t", prompt="p", tools_allowed=[])
    s = score(make_report({"a_A": 1.0}, tool_calls=0, invalid_calls=0), no_gold)
    assert s["accuracy"] is None and s["correct"] is None and s["recovery"] is None
    assert s["invalid_call_rate"] == 0.0


def test_aggregate_numbers_and_ci() -> None:
    scores = [
        {
            "accuracy": 1.0,
            "invalid_call_rate": 0.0,
            "dag_valid": True,
            "provenance": True,
            "recovery": None,
            "tokens": 100,
            "usd": 0.01,
        },
        {
            "accuracy": 0.5,
            "invalid_call_rate": 0.25,
            "dag_valid": False,
            "provenance": True,
            "recovery": True,
            "tokens": 300,
            "usd": 0.03,
        },
        {
            "accuracy": None,
            "invalid_call_rate": 0.0,
            "dag_valid": True,
            "provenance": False,
            "recovery": False,
            "tokens": 200,
            "usd": 0.02,
        },
    ]
    numbers = aggregate(
        scores,
        backend="mock",
        model_id="claude-opus-5",
        trace_path="traces.jsonl",
        seed=1,
        n_boot=50,
    )
    parsed = nums.parse_numbers_file(numbers)
    assert set(parsed) == {f"agent.eval.{m}" for m in METRICS}
    assert parsed["agent.eval.accuracy"][0] == 0.75 and parsed["agent.eval.accuracy"][1]["n"] == 2
    assert parsed["agent.eval.recovery_rate"][0] == 0.5 and parsed["agent.eval.dag_valid_rate"][
        0
    ] == pytest.approx(2 / 3)
    assert parsed["agent.eval.provenance_rate"][0] == pytest.approx(2 / 3)
    assert parsed["agent.eval.tokens_per_task"][0] == 200 and parsed["agent.eval.usd_per_task"][
        0
    ] == pytest.approx(0.02)
    assert parsed["agent.eval.n_tasks"][0] == 3
    meta = parsed["agent.eval.tokens_per_task"][1]
    assert (
        meta["backend"] == "mock"
        and meta["model_id"] == "claude-opus-5"
        and meta["trace_path"] == "traces.jsonl"
    )
    lo, hi = meta["ci95"]
    assert (
        100 <= lo <= 200 <= hi <= 300 and meta["seed"] == 1 and meta["reference"]["code"] == "mace"
    )
    assert (
        parsed["agent.eval.n_tasks"][1]["ci95"] is None
        and parsed["agent.eval.n_tasks"][1]["ci95_reason"]
    )
    again = aggregate(
        scores, backend="mock", model_id="claude-opus-5", trace_path="t", seed=1, n_boot=50
    )
    assert again["agent.eval.tokens_per_task@meta"]["ci95"] == meta["ci95"]  # seeded
    single = aggregate(scores[:1], backend="scripted", model_id="scripted", trace_path="t")
    assert (
        single["agent.eval.accuracy@meta"]["ci95"] is None
        and "fewer than two" in single["agent.eval.accuracy@meta"]["ci95_reason"]
    )
    assert "agent.eval.recovery_rate" not in single  # no injected failures: nothing to publish


def test_task_file_helpers(task_specs: list[tasks.TaskSpec], tmp_path: Path) -> None:
    assert len(task_specs) == 12 and len({s.task.task_id for s in task_specs}) == 12
    kinds = {s.kind for s in task_specs}
    assert kinds <= set(tasks.KINDS) and all(
        s.gold_from == "screen" and s.task.gold is None for s in task_specs
    )
    injected = [s.task.injected_failure for s in task_specs if s.task.injected_failure]
    assert sorted(injected) == ["bad_structure", "budget_exhausted", "sum_rule", "tool_error"]
    assert sum(1 for s in task_specs if s.kind in ("relax_report", "inventory_relax_report")) == 2
    assert all(s.task.tolerance for s in task_specs)
    from b20mlip.agent.tools import TOOL_NAMES

    for spec in task_specs:
        assert set(spec.task.tools_allowed) <= set(TOOL_NAMES) and spec.task.tools_allowed
        assert spec.plan is not None and spec.plan.kind == spec.kind
    copy = tmp_path / "tasks.jsonl"
    tasks.write_tasks(task_specs, copy)
    assert [s.record() for s in tasks.load_tasks(copy)] == [s.record() for s in task_specs]
    assert tasks.parse_task_ref("evals/agent_tasks.jsonl:t01") == (
        Path("evals/agent_tasks.jsonl"),
        "t01",
    )
    assert tasks.parse_task_ref("evals/agent_tasks.jsonl") == (
        Path("evals/agent_tasks.jsonl"),
        None,
    )
    with pytest.raises(KeyError):
        tasks.select_task(task_specs, "nope")
    with pytest.raises(ValueError, match="unknown kind"):
        tasks.parse_task({"task_id": "x", "prompt": "p", "tools_allowed": [], "kind": "teleport"})
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"task_id": "a", "prompt": "p", "tools_allowed": []}\n'
        '{"task_id": "a", "prompt": "p", "tools_allowed": []}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        tasks.load_tasks(bad)
    bad.write_text("# comment only\n\n")
    with pytest.raises(ValueError, match="no tasks"):
        tasks.load_tasks(bad)
    spec = tasks.task_from_prompt("Relax FeSi", budget=Budget(max_calls=4))
    assert (
        spec.task.task_id.startswith("prompt-")
        and spec.task.budget.max_calls == 4
        and spec.plan is None
    )
    assert tasks.budget_from_spec(None) == Budget() and tasks.budget_from_spec("ci").max_calls == 8
    assert tasks.budget_from_spec('{"max_calls": 2}', approve_cluster=True) == Budget(
        max_calls=2, approve_cluster=True
    )
    with pytest.raises(ValueError, match="preset"):
        tasks.budget_from_spec("nope")
    with pytest.raises(ValueError, match="object"):
        tasks.budget_from_spec("[1]")
    gold = {"t01-phonons-imaginary-FeSi": {"gold": {"imaginary_count": 3}}}
    filled = tasks.apply_gold(task_specs[:2], gold)
    assert filled[0].task.gold == {"imaginary_count": 3} and filled[1].task.gold is None
    assert tasks.apply_gold(task_specs[:1], None)[0] is task_specs[0]
    not_gold = tmp_path / "not_gold.json"
    not_gold.write_text('{"schema": "other", "tasks": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="gold file"):
        tasks.load_gold(not_gold)


def test_screen_then_eval_with_mock_and_scripted(
    agent_settings: Settings,
    task_specs: list[tasks.TaskSpec],
    traces_dir: Path,
    tool_kwargs: dict[str, Any],
    tmp_path: Path,
) -> None:
    chosen = ["t01-phonons-imaginary-FeSi", "t02-relax-a0-MnSi", "t09-relax-a0-FeSi-tool-error"]
    specs = [tasks.select_task(task_specs, t) for t in chosen]
    gold_path = tmp_path / "gold_tiny.json"
    res = run_stage(
        "agent.eval", agent_settings, screen.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=specs, compounds=["FeSi", "MnSi"], out=gold_path, **tool_kwargs,
    )  # fmt: skip
    assert res.status == "ok" and res.summary["n_tasks"] == 3 and res.summary["n_gold"] == 3
    gold = tasks.load_gold(gold_path)
    assert set(gold) == set(chosen) and gold[chosen[0]]["gold"].keys() == {
        "imaginary_count",
        "omega_max_meV",
    }
    assert (
        gold[chosen[2]]["kind"] == "relax_a0" and gold[chosen[2]]["invalid_calls"] == 0
    )  # screen runs clean
    assert screen.filter_tasks(task_specs, ["FeSi"]) and all(
        set(screen.task_compounds(s)) <= {"FeSi"} for s in screen.filter_tasks(task_specs, ["FeSi"])
    )
    # the mock backend replays the three fixture traces and is scored against the gold
    res = run_stage(
        "agent.eval", agent_settings, eval_mod.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=specs, backend=MockBackend(traces_dir), gold=gold, n_boot=40, **tool_kwargs,
    )  # fmt: skip
    assert res.status == "ok", res.summary
    summary = res.summary
    assert (
        summary["n_tasks"] == 3 and summary["accuracy"] == 1.0 and summary["provenance_rate"] == 1.0
    )
    assert summary["dag_valid_rate"] == 1.0 and summary["recovery_rate"] == 1.0
    assert summary["invalid_call_rate"] == pytest.approx(1 / 9) and summary["tokens_in"] > 0
    run_dir = Path(res.manifest_path).parent
    payload = json.loads((run_dir / eval_mod.EVAL_FILE).read_text())
    assert [s["task_id"] for s in payload["scores"]] == chosen and payload["backend"] == "mock"
    assert all(s["status"] == "ok" for s in payload["scores"])
    traces = [
        json.loads(line) for line in (run_dir / eval_mod.TRACES_FILE).read_text().splitlines()
    ]
    assert len(traces) == 3 and all(Path(t["trace_path"]).is_file() for t in traces)
    harvest = nums.harvest(agent_settings.paths.runs_dir)
    entry = harvest.entries["agent.eval.accuracy"]
    assert (
        entry.value == 1.0
        and entry.meta["backend"] == "mock"
        and entry.meta["model_id"] == "claude-opus-5"
    )
    assert entry.meta["trace_path"].endswith(eval_mod.TRACES_FILE) and entry.meta["n"] == 3
    assert entry.meta["ci95"] == [1.0, 1.0] and harvest.entries["agent.eval.usd_per_task"].value > 0
    # gate A10: mock numbers carry the meta but may never be cited in a README
    readme = tmp_path / "README.md"
    readme.write_text(
        "# t\n\naccuracy <!-- num:agent.eval.accuracy -->1<!-- /num -->\n", encoding="utf-8"
    )
    violations = audit.run(readme, harvest.as_json(), agent_settings.paths.runs_dir)
    assert any("A10: mock-backend number agent.eval.accuracy is cited" in v for v in violations)
    readme.write_text("# t\n\nno numbers\n", encoding="utf-8")
    assert [
        v for v in audit.run(readme, harvest.as_json(), agent_settings.paths.runs_dir) if "A10" in v
    ] == []
    # the scripted backend reproduces its own gold
    res = run_stage(
        "agent.eval", agent_settings, eval_mod.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=specs, backend=ScriptedBackend(), gold=gold, limit=2, n_boot=40, **tool_kwargs,
    )  # fmt: skip
    assert res.status == "ok" and res.summary["n_tasks"] == 2 and res.summary["accuracy"] == 1.0
    assert "recovery_rate" not in res.summary  # the two clean tasks carry no injected failure
    # without gold there is no accuracy; a dry run plans only
    res = run_stage(
        "agent.eval", agent_settings, eval_mod.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=specs[:1], backend=ScriptedBackend(), n_boot=40, **tool_kwargs,
    )  # fmt: skip
    assert (
        res.status == "ok"
        and "accuracy" not in res.summary
        and res.summary["dag_valid_rate"] == 1.0
    )
    res = run_stage(
        "agent.eval", agent_settings, eval_mod.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        dry_run=True, tasks=specs, backend=ScriptedBackend(), **tool_kwargs,
    )  # fmt: skip
    assert res.status == "partial" and res.summary["planned"] == 3


def test_eval_survives_a_task_that_cannot_run(
    agent_settings: Settings, task_specs: list[tasks.TaskSpec], tool_kwargs: dict[str, Any]
) -> None:
    spec = tasks.select_task(task_specs, "t03-softening-rank")
    kwargs = dict(tool_kwargs, refs_dir=Path("/nonexistent/refs"))  # no references: compare fails
    res = run_stage(
        "agent.eval", agent_settings, eval_mod.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=[spec], backend=ScriptedBackend(),
        gold={spec.task.task_id: {"gold": {"compound": "FeSi"}}},
        n_boot=10, **kwargs,
    )  # fmt: skip
    assert (
        res.status == "ok"
        and res.summary["accuracy"] == 0.0
        and res.summary["provenance_rate"] == 0.0
    )
    assert res.summary["invalid_call_rate"] == pytest.approx(3 / 9)


def test_screen_all_twelve_tasks(
    agent_settings: Settings,
    task_specs: list[tasks.TaskSpec],
    tool_kwargs: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Every task of evals/agent_tasks.jsonl yields grounded, DAG-valid gold with the tiny model
    (1x1x1 phonons, 64-atom 1 ps MD; ~12 s) and the scripted eval reproduces it exactly."""
    gold_path = tmp_path / "gold_all.json"
    res = run_stage(
        "agent.eval", agent_settings, screen.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=task_specs, out=gold_path, **tool_kwargs,
    )  # fmt: skip
    assert res.status == "ok" and res.summary["n_tasks"] == 12 and res.summary["n_gold"] == 12
    gold = tasks.load_gold(gold_path)
    for spec in task_specs:
        entry = gold[spec.task.task_id]
        assert entry["grounded"] and entry["dag_valid"] and entry["invalid_calls"] == 0, (
            spec.task.task_id
        )
        assert set(entry["gold"]) == set(spec.task.tolerance), spec.task.task_id
    res = run_stage(
        "agent.eval", agent_settings, eval_mod.run, seed=0, runs_dir=agent_settings.paths.runs_dir,
        tasks=task_specs, backend=ScriptedBackend(), gold=gold, n_boot=40, **tool_kwargs,
    )  # fmt: skip
    assert (
        res.status == "ok"
        and res.summary["accuracy"] == 1.0
        and res.summary["recovery_rate"] == 1.0
    )
    assert res.summary["provenance_rate"] == 1.0 and res.summary["dag_valid_rate"] == 1.0
