"""Runner: DAG order, grounding, the four injected failures with recovery scoring, the dispatcher
feeding guard violations back as tool results, report/numbers files, dry runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from b20mlip.agent import eval as eval_mod
from b20mlip.agent import runner, tasks
from b20mlip.agent.backends import BackendResult, ScriptedBackend, format_answer
from b20mlip.agent.runner import CallRecord, Dispatcher, Injection, dag_valid, ground
from b20mlip.agent.trace import Trace
from b20mlip.config import Settings
from b20mlip.models import AgentReport, AgentTask, Budget
from b20mlip.provenance import RunContext, read_manifest
from b20mlip.report import numbers as nums


def rec(
    i: int, name: str, run_id: str | None = None, blocked: bool = False, **args: Any
) -> CallRecord:
    return CallRecord(i, name, dict(args), {}, blocked=blocked, run_id=run_id)


def test_dag_valid_true_cases() -> None:
    ok, reasons = dag_valid(
        [
            rec(0, "get_structure"),
            rec(1, "relax"),
            rec(2, "phonons", "p1"),
            rec(3, "compare_phonons", run_id_model="p1"),
            rec(4, "write_report"),
        ]
    )
    assert ok and reasons == []
    ok, _ = dag_valid(
        [rec(0, "list_data"), rec(1, "evaluate_errors"), rec(2, "select_frames"), rec(3, "run_md")]
    )
    assert ok
    assert dag_valid([])[0] and dag_valid([rec(0, "list_data")])[0]
    # a blocked call does not count: a blocked write_report may be followed by nothing at all
    ok, _ = dag_valid(
        [rec(0, "get_structure"), rec(1, "relax"), rec(2, "write_report", blocked=True)]
    )
    assert ok


def test_dag_valid_false_cases() -> None:
    ok, reasons = dag_valid([rec(0, "relax")])
    assert not ok and "before any get_structure" in reasons[0]
    ok, reasons = dag_valid(
        [rec(0, "get_structure"), rec(1, "compare_phonons", run_id_model="nope")]
    )
    assert not ok and "no earlier phonons run" in reasons[0]
    ok, reasons = dag_valid([rec(0, "list_data"), rec(1, "select_frames")])
    assert not ok and "before evaluate_errors" in reasons[0]
    ok, reasons = dag_valid([rec(0, "get_structure"), rec(1, "write_report"), rec(2, "relax")])
    assert not ok and "not the last call" in reasons[0]
    ok, reasons = dag_valid([rec(0, "phonons", "p"), rec(1, "run_md")])
    assert not ok and len(reasons) == 2


class NaiveBackend:
    """Calls whatever it is told, in order; answers with the last numeric result (for tests)."""

    name = "scripted"
    model_id = "naive"

    def __init__(
        self, calls: list[tuple[str, dict[str, Any]]], answer_key: str | None = None
    ) -> None:
        self.calls = calls
        self.answer_key = answer_key

    def run(self, task: AgentTask, tools: Any) -> BackendResult:
        results = []
        state: dict[str, str] = {}
        for name, args in self.calls:
            args = {k: state.get(v, v) if isinstance(v, str) else v for k, v in args.items()}
            res = tools.call(name, args)
            results.append(res)
            if isinstance(res.get("frame_id"), str):
                state["$frame"] = res["frame_id"]
        answer: dict[str, Any] = {}
        cites: list[dict[str, Any]] = []
        if self.answer_key:
            for res in reversed(results):
                if self.answer_key in res and "run_id" in res:
                    answer[self.answer_key] = res[self.answer_key]
                    cites.append(
                        {
                            "key": self.answer_key,
                            "value": res[self.answer_key],
                            "run_id": res["run_id"],
                        }
                    )
                    break
        tools.trace.message("assistant", format_answer(answer, cites, "naive"))
        return BackendResult(answer=answer, citations=cites, model_id=self.model_id, tool_calls=[])


def test_dispatcher_blocks_unknown_tools_and_feeds_violations_back(
    agent_settings: Settings, tool_kwargs: dict[str, Any]
) -> None:
    task = AgentTask(
        task_id="naive",
        prompt="x",
        tools_allowed=["get_structure", "relax"],
        budget=Budget(max_calls=3),
    )
    backend = NaiveBackend(
        [("shell", {"cmd": "ls"}), ("get_structure", {"compound": "FeSi"}),
         ("relax", {"compound": "FeSi", "frame_id": "$frame", "fmax": 0.0, "steps": 0}),
         ("get_structure", {"compound": "CoSi"})],
        answer_key="a_A",
    )  # fmt: skip
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(task, backend, agent_settings, ctx, **tool_kwargs)
    assert report.tool_calls == 4 and report.invalid_calls == 2
    assert [v.split(":")[0] for v in report.violations] == [
        "guard violation (unknown_tool)", "guard violation (max_calls)",
    ]  # fmt: skip
    assert report.grounded and report.dag_valid and report.answer["a_A"] > 0
    events = Trace.read(report.trace_path)
    assert [e["kind"] for e in events][:3] == ["message", "call", "violation"]
    assert events[2]["message"].startswith("guard violation (unknown_tool)")
    assert json.loads(events[-1]["text"])["answer"] == report.answer


def test_unknown_tools_allowed_is_rejected(
    agent_settings: Settings, tool_kwargs: dict[str, Any]
) -> None:
    task = AgentTask(task_id="bad", prompt="x", tools_allowed=["teleport"])
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    with pytest.raises(ValueError, match="unknown tools_allowed"):
        runner.run(task, ScriptedBackend(), agent_settings, ctx, **tool_kwargs)


def test_grounding(agent_settings: Settings, tool_kwargs: dict[str, Any]) -> None:
    task = AgentTask(task_id="g", prompt="x", tools_allowed=[])
    backend = NaiveBackend([("get_structure", {"compound": "FeSi"})], answer_key="a_A")
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(task, backend, agent_settings, ctx, **tool_kwargs)
    assert report.grounded and len(report.numbers) == 1
    run_id = report.numbers[0].run_id
    runs_dir = agent_settings.paths.runs_dir
    assert ground(
        {"a_A": 4.48, "note": "text"}, [{"key": "a_A", "value": 4.48, "run_id": run_id}], runs_dir
    )[0]
    ok, refs, problems = ground({"a_A": 4.48}, [], runs_dir)
    assert not ok and refs == [] and "without a resolving citation" in problems[0]
    ok, _, problems = ground(
        {"a_A": 4.48}, [{"key": "a_A", "value": 4.48, "run_id": "ghost"}], runs_dir
    )
    assert not ok and "has no manifest" in problems[0]
    ok, _, problems = ground(
        {"a_A": 4.49}, [{"key": "a_A", "value": 4.48, "run_id": run_id}], runs_dir
    )
    assert not ok and "differs" in problems[0]
    assert not ground({}, [], runs_dir)[0]  # an empty answer is never grounded
    assert ground({"compound": "FeSi"}, [], runs_dir)[0]  # strings need no citation


def test_backend_crash_is_an_invalid_run(
    agent_settings: Settings, tool_kwargs: dict[str, Any]
) -> None:
    class Crashing:
        name = "scripted"
        model_id = "crash"

        def run(self, task: AgentTask, tools: Any) -> BackendResult:
            raise RuntimeError("boom")

    task = AgentTask(task_id="crash", prompt="x", tools_allowed=[])
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(task, Crashing(), agent_settings, ctx, **tool_kwargs)
    assert report.answer == {} and not report.grounded and report.tool_calls == 0
    diagnostics = json.loads((ctx.out_dir / runner.REPORT_FILE).read_text())["diagnostics"]
    assert "boom" in diagnostics["backend_error"]


@pytest.mark.parametrize(
    "task_id, failure",
    [
        ("t09-relax-a0-FeSi-tool-error", "tool_error"),
        ("t10-relax-a0-CoSi-bad-structure", "bad_structure"),
        ("t11-inventory-relax-report-MnSi-budget", "budget_exhausted"),
        ("t12-phonons-imaginary-CoSi-sum-rule", "sum_rule"),
    ],
)
def test_injected_failures_and_recovery(
    task_id: str,
    failure: str,
    agent_settings: Settings,
    task_specs: list[tasks.TaskSpec],
    tool_kwargs: dict[str, Any],
) -> None:
    spec = tasks.select_task(task_specs, task_id)
    assert spec.task.injected_failure == failure
    backend = ScriptedBackend(tasks.plans_of(task_specs))
    # gold = the same task without the injection (what `screen` computes)
    clean = spec.task.model_copy(update={"injected_failure": None, "task_id": f"{task_id}-clean"})
    ctx0 = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    gold = runner.run(clean, backend, agent_settings, ctx0, **tool_kwargs)
    assert gold.grounded and gold.invalid_calls == 0
    graded = spec.task.model_copy(update={"gold": {k: gold.answer[k] for k in spec.task.tolerance}})
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(
        graded, ScriptedBackend(tasks.plans_of(task_specs)), agent_settings, ctx, **tool_kwargs
    )
    diagnostics = json.loads((ctx.out_dir / runner.REPORT_FILE).read_text())["diagnostics"]
    assert diagnostics["injection_fired"] is True
    assert report.grounded and report.dag_valid
    s = eval_mod.score(report, graded)
    assert s["recovery"] is True and s["correct"] is True and s["accuracy"] == 1.0
    calls = diagnostics["calls"]
    if failure == "tool_error":
        assert report.invalid_calls == 1 and [c["name"] for c in calls] == [
            "get_structure",
            "relax",
            "relax",
        ]
        assert calls[1]["error"] and calls[1]["injected"] == "tool_error" and not calls[2]["error"]
    elif failure == "bad_structure":
        assert [c["name"] for c in calls] == ["get_structure", "get_structure", "relax"]
        assert calls[0]["injected"] == "bad_structure" and report.invalid_calls == 0
    elif failure == "budget_exhausted":
        assert diagnostics["effective_budget"]["max_calls"] == 3
        assert [c["name"] for c in calls] == ["list_data", "get_structure", "relax", "write_report"]
        assert calls[3]["blocked"] and report.invalid_calls == 1 and len(report.violations) == 1
        assert "max_calls" in report.violations[0] and "n_models" in report.answer
    elif failure == "sum_rule":
        assert [c["name"] for c in calls] == ["get_structure", "phonons", "phonons"]
        assert calls[1]["injected"] == "sum_rule" and calls[2]["injected"] is None
        events = Trace.read(report.trace_path)
        results = [e["result"] for e in events if e["kind"] == "result" and e["tool"] == "phonons"]
        assert results[0]["asr_violation"] == 1 and results[0]["asr_applied"] == 0
        assert results[1]["asr_violation"] == 0 and results[1]["asr_applied"] == 1


def test_bad_structure_is_blocked_by_the_guard_for_a_naive_agent(
    agent_settings: Settings, tool_kwargs: dict[str, Any]
) -> None:
    task = AgentTask(
        task_id="naive-bad", prompt="x", tools_allowed=[], injected_failure="bad_structure"
    )
    backend = NaiveBackend(
        [("get_structure", {"compound": "FeSi"}),
         ("relax", {"compound": "FeSi", "frame_id": "$frame", "fmax": 0.0, "steps": 0})],
        answer_key="energy_eV",  # only a successful relax provides it
    )  # fmt: skip
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(task, backend, agent_settings, ctx, **tool_kwargs)
    assert (
        report.invalid_calls == 1
        and "allowlist" in report.violations[0]
        and "Cu" in report.violations[0]
    )
    assert report.answer == {} and not report.grounded
    events = Trace.read(report.trace_path)
    fetched = next(
        e["result"] for e in events if e["kind"] == "result" and e["tool"] == "get_structure"
    )
    assert fetched["elements"] == ["Cu", "Si"] and fetched["formula"] == "CuSi"


def test_injection_budget_and_no_op() -> None:
    assert Injection("budget_exhausted").budget(Budget()).max_calls == 3
    none = Injection(None)
    assert none.budget(Budget()).max_calls == 25 and not none.fired


def test_stage_files_numbers_and_dry_run(
    agent_settings: Settings, task_specs: list[tasks.TaskSpec], tool_kwargs: dict[str, Any]
) -> None:
    spec = tasks.select_task(task_specs, "t02-relax-a0-MnSi")
    backend = ScriptedBackend(tasks.plans_of(task_specs))
    report, res = runner.run_task_stage(
        agent_settings,
        spec.task,
        backend,
        seed=3,
        runs_dir=agent_settings.paths.runs_dir,
        **tool_kwargs,
    )
    assert res.status == "ok" and report is not None and report.task_id == spec.task.task_id
    assert AgentReport.model_validate(report.model_dump()) == report
    run_dir = Path(res.manifest_path).parent
    manifest = read_manifest(run_dir)
    assert manifest.stage == "agent.run" and manifest.seed == 3
    assert manifest.extras["backend"] == "scripted" and manifest.extras["model_id"] == "scripted"
    assert manifest.extras["tokens_in"] == 0 and manifest.extras["cost_usd"] == 0.0
    names = {Path(a.path).name for a in manifest.outputs}
    assert {"trace.jsonl", "report.json", "numbers.json", "plan.json"} <= names
    assert Path(report.trace_path) == run_dir / "trace.jsonl"
    harvest = nums.harvest(agent_settings.paths.runs_dir)
    key = "agent.run.t02-relax-a0-MnSi.grounded"
    assert harvest.entries[key].value == 1.0 and harvest.stale == []
    meta = harvest.entries[key].meta
    assert meta["backend"] == "scripted" and meta["model_id"] == "scripted" and meta["trace_path"]
    assert meta["ci95"] is None and meta["ci95_reason"] and meta["reference"]["code"] == "mace"
    assert meta["head"] == "Default" and meta["e0_source"] == "foundation" and meta["seed"] == 3
    assert (
        res.summary["tool_calls"] == 2
        and json.loads(res.summary["answer"])["a_A"] == report.answer["a_A"]
    )
    # sub-runs of the tools sit next to the agent run and never publish numbers
    dirs = [p for p in (Path(agent_settings.paths.runs_dir) / "agent.run").iterdir() if p.is_dir()]
    assert len(dirs) == 3 and sum((d / "numbers.json").is_file() for d in dirs) == 1
    # dry run: plan only
    report_dry, res_dry = runner.run_task_stage(
        agent_settings, spec.task, backend, seed=3, runs_dir=agent_settings.paths.runs_dir,
        dry_run=True,
        **tool_kwargs,
    )  # fmt: skip
    assert res_dry.status == "partial" and report_dry is None and res_dry.summary["planned"] == 1
    assert (Path(res_dry.manifest_path).parent / "plan.json").is_file()


def test_run_numbers_keys_are_valid(task_specs: list[tasks.TaskSpec]) -> None:
    report = AgentReport(
        task_id="t01-phonons-imaginary-FeSi", backend="mock", model_id="claude-opus-5",
        answer={"x": 1},
        numbers=[], tool_calls=2, invalid_calls=0, dag_valid=True, grounded=True, violations=[],
        tokens_in=10, tokens_out=5, cost_usd=0.001, wall_s=1.0, trace_path="t.jsonl",
    )  # fmt: skip
    numbers = runner.run_numbers(report, task_specs[0].task, seed=0)
    parsed = nums.parse_numbers_file(numbers)
    assert set(parsed) == {f"agent.run.t01-phonons-imaginary-FeSi.{m}" for m in runner.RUN_METRICS}
    assert parsed["agent.run.t01-phonons-imaginary-FeSi.cost_usd"][1]["unit"] == "USD"


def test_dispatcher_beta_tools_are_guarded(
    agent_settings: Settings, tool_kwargs: dict[str, Any]
) -> None:
    from b20mlip.agent.guard import Guard
    from b20mlip.agent.tools import ToolContext, use_context

    tc = ToolContext(agent_settings, runs_dir=agent_settings.paths.runs_dir, **tool_kwargs)
    trace = Trace(Path(agent_settings.paths.runs_dir) / "trace.jsonl")
    dispatcher = Dispatcher(
        tc, Guard(Budget(max_calls=3)), trace, allowed=["get_structure", "list_data"]
    )
    tools_ = dispatcher.beta_tools(["get_structure"])
    assert [t.name for t in tools_] == ["get_structure"] and tools_[0].to_dict()["strict"] is True
    assert [t.name for t in dispatcher.beta_tools()] == ["list_data", "get_structure"]
    with use_context(tc):
        first = json.loads(tools_[0].call({"compound": "FeSi"}))
        bad = json.loads(
            tools_[0].call({"compound": 3})
        )  # invalid arguments: an error, not a crash
        third = json.loads(tools_[0].call({"compound": "CoSi"}))
        blocked = json.loads(tools_[0].call({"compound": "CoSi"}))
    assert first["formula"] == "FeSi" and "invalid arguments" in bad["error"]
    assert third["formula"] == "CoSi" and "max_calls" in blocked["error"]
    assert dispatcher.invalid_calls == 2 and len(dispatcher.records()) == 4
    assert dispatcher.records()[0]["args_sha256"] and dispatcher.records()[3]["blocked"]
