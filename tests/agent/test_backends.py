"""Backends: answer parsing, prices, the scripted planner (gold on 3 tasks), mock replay of the
fixture traces (fresh and stored results), and the anthropic backend refusing to exist without
a key (no client constructed)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from b20mlip.agent import backends, runner, tasks
from b20mlip.agent.backends import (
    PRICES_USD_PER_MTOK,
    AnthropicBackend,
    BackendUnavailable,
    MockBackend,
    ScriptedBackend,
    TaskPlan,
    build_plan,
    compose_answer,
    cost_usd,
    format_answer,
    infer_kind,
    make_backend,
    parse_answer,
    parse_params,
    system_prompt,
)
from b20mlip.agent.tools import ToolContext
from b20mlip.agent.trace import Trace
from b20mlip.config import Settings
from b20mlip.models import AgentTask, Budget
from b20mlip.provenance import RunContext


def test_parse_answer_variants() -> None:
    text = (
        'Here it is:\n```json\n{"answer": {"a_A": 4.5, "compound": "FeSi"}, '
        '"numbers": [{"key": "a_A", "value": 4.5, "run_id": "r1"}], "text": "done"}\n```'
    )
    answer, cites, body = parse_answer(text)
    assert answer == {"a_A": 4.5, "compound": "FeSi"} and cites == [
        {"key": "a_A", "value": 4.5, "run_id": "r1"}
    ]
    assert body == "done"
    answer, cites, _ = parse_answer(
        'prose {"a_A": 4.5, "numbers": [{"key": "a_A", "value": 4.5, "run_id": "r"}]} trailing'
    )
    assert answer == {"a_A": 4.5} and cites[0]["run_id"] == "r"
    assert parse_answer("no json here") == ({}, [], "no json here")
    assert parse_answer("[1, 2]") == ({}, [], "[1, 2]")
    assert parse_answer('{"answer": {"x": 1}, "numbers": [{"key": "x"}]}')[1] == []
    rebuilt = json.loads(format_answer({"x": 1}, [{"key": "x", "value": 1, "run_id": "r"}], "t"))
    assert rebuilt == {
        "answer": {"x": 1},
        "numbers": [{"key": "x", "value": 1, "run_id": "r"}],
        "text": "t",
    }


def test_prices() -> None:
    assert PRICES_USD_PER_MTOK["claude-opus-5"] == (5.0, 25.0)
    assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
    assert cost_usd("claude-opus-5", 2000, 100) == pytest.approx(0.0125)
    assert cost_usd("scripted", 10, 10) == 0.0 and cost_usd(None, 10, 10) == 0.0


def test_kind_inference_and_params() -> None:
    assert infer_kind("How many imaginary phonon modes for FeSi?") == "phonons_imaginary"
    assert infer_kind("Relax MnSi and report a0") == "relax_a0"
    assert (
        infer_kind("Which of FeSi/CoSi has the largest softening index vs the qe reference?")
        == "softening_rank"
    )
    assert infer_kind("Evaluate the force MAE on tier T0 for model B1") == "force_mae"
    assert infer_kind("Select the 5 highest-uncertainty frames") == "select_frames"
    assert infer_kind("Plan a QE round for 10 frames: how many node-hours?") == "dft_plan"
    assert infer_kind("Run 1 ps NVT MD of CoSi at 300 K and report drift") == "md_drift"
    assert infer_kind("Relax FeSi and write a report") == "relax_report"
    assert infer_kind("List the models, relax FeSi and write a report") == "inventory_relax_report"
    with pytest.raises(ValueError, match="cannot infer"):
        infer_kind("What is the meaning of life?")
    p = parse_params(
        "Run 1 ps NVT MD of CoSi at 300 K with 64 atoms on tier T1 with model B2 of dataset "
        "labelled_r0, the 5 highest"
    )
    assert p["compound"] == "CoSi" and p["ensemble"] == "nvt" and p["ps"] == 1.0 and p["T"] == 300.0
    assert (
        p["natoms"] == 64
        and p["tier"] == "T1"
        and p["model"] == "B2"
        and p["frames"] == "labelled_r0"
    )
    assert p["n"] == 5
    assert parse_params("FeSi, CoSi and MnSi vs qe")["compounds"] == ["FeSi", "CoSi", "MnSi"]
    assert parse_params("nothing") == {}


def test_build_plan_covers_every_kind(agent_settings: Settings) -> None:
    tc = ToolContext(agent_settings)
    for kind in backends.KINDS:
        params = {"compound": "FeSi", "compounds": ["FeSi", "CoSi"], "frames": "d", "n": 3}
        steps = build_plan(kind, params, tc)
        assert steps and all(s.tool for s in steps)
        for step in steps:
            args = step.args({})
            assert isinstance(args, dict)
    names = [s.tool for s in build_plan("softening_rank", {"compounds": ["FeSi", "CoSi"]}, tc)]
    assert names == ["get_structure", "phonons", "compare_phonons"] * 2
    assert [s.tool for s in build_plan("select_frames", {"frames": "d"}, tc)] == [
        "list_data", "list_data", "evaluate_errors", "select_frames",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="unknown task kind"):
        build_plan("teleport", {}, tc)
    answer, cites, text = compose_answer("phonons_imaginary", {"compound": "FeSi"}, {})
    assert answer == {} and cites == [] and "FeSi" in text
    answer, cites, _ = compose_answer(
        "softening_rank", {"compounds": ["FeSi", "CoSi"]},
        {
            "compare:FeSi": {"softening_index": 0.9, "run_id": "a"},
            "compare:CoSi": {"softening_index": 1.1, "run_id": "b"},
        },
    )  # fmt: skip
    assert answer["compound"] == "CoSi" and answer["softening_index"] == 1.1
    assert {c["key"] for c in cites} == {
        "softening_index_FeSi",
        "softening_index_CoSi",
        "softening_index",
    }


def test_system_prompt_mentions_budget(agent_settings: Settings) -> None:
    text = system_prompt(ToolContext(agent_settings), Budget(max_calls=7, approve_cluster=True))
    assert "7 tool calls" in text and "is approved" in text and "[1, 1, 1]" in text
    assert "JSON" in text and "run_id" in text


@pytest.mark.parametrize(
    "task_id", ["t01-phonons-imaginary-FeSi", "t02-relax-a0-MnSi", "t04-force-mae-T0"]
)
def test_scripted_gold_on_three_tasks(
    task_id: str,
    agent_settings: Settings,
    task_specs: list[tasks.TaskSpec],
    tool_kwargs: dict[str, Any],
) -> None:
    spec = tasks.select_task(task_specs, task_id)
    backend = ScriptedBackend(tasks.plans_of(task_specs))
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(spec.task, backend, agent_settings, ctx, **tool_kwargs)
    assert report.backend == "scripted" and report.model_id == "scripted"
    assert report.grounded and report.dag_valid and report.invalid_calls == 0
    assert set(spec.task.tolerance) <= set(report.answer), report.answer
    assert report.numbers and all(n.manifest_sha256 for n in report.numbers)
    # the gold script is deterministic: a second run reproduces every graded value
    ctx2 = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    again = runner.run(
        spec.task, ScriptedBackend(tasks.plans_of(task_specs)), agent_settings, ctx2, **tool_kwargs
    )
    for key in spec.task.tolerance:
        assert again.answer[key] == pytest.approx(report.answer[key])
    if task_id == "t01-phonons-imaginary-FeSi":
        assert report.tool_calls == 2 and isinstance(report.answer["imaginary_count"], int)
    if task_id == "t04-force-mae-T0":
        assert report.answer["n_frames"] == 15 and report.answer["mae_f"] > 0


def test_scripted_infers_a_plan_from_the_prompt(
    agent_settings: Settings, tool_kwargs: dict[str, Any]
) -> None:
    spec = tasks.task_from_prompt("Relax CoSi with the model and report a_A")
    backend = ScriptedBackend()
    assert backend.plan_for(spec.task).kind == "relax_a0"
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(spec.task, backend, agent_settings, ctx, **tool_kwargs)
    assert report.grounded and report.answer["a_A"] == pytest.approx(4.44, abs=0.05)
    explicit = ScriptedBackend(
        {spec.task.task_id: TaskPlan("phonons_imaginary", {"compound": "CoSi"})}
    )
    assert explicit.plan_for(spec.task).kind == "phonons_imaginary"


@pytest.mark.parametrize("task_id", ["t01-phonons-imaginary-FeSi", "t09-relax-a0-FeSi-tool-error"])
def test_mock_replays_fixture_traces(
    task_id: str,
    agent_settings: Settings,
    task_specs: list[tasks.TaskSpec],
    traces_dir: Path,
    tool_kwargs: dict[str, Any],
) -> None:
    spec = tasks.select_task(task_specs, task_id)
    recorded = Trace.read(traces_dir / f"{task_id}.jsonl")
    n_calls = sum(1 for e in recorded if e["kind"] == "call")
    backend = MockBackend(traces_dir)
    assert backend.trace_for(spec.task) == traces_dir / f"{task_id}.jsonl"
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(spec.task, backend, agent_settings, ctx, **tool_kwargs)
    assert report.backend == "mock" and report.model_id == "claude-opus-5"
    assert report.tool_calls == n_calls and report.grounded and report.dag_valid
    tokens_in, tokens_out = Trace.usage_totals(recorded)
    assert report.tokens_in == tokens_in and report.tokens_out == tokens_out
    assert report.cost_usd == pytest.approx(cost_usd("claude-opus-5", tokens_in, tokens_out))
    fresh = Trace.read(report.trace_path)
    fresh_calls = [e for e in fresh if e["kind"] == "call"]
    assert [e["tool"] for e in fresh_calls] == [e["tool"] for e in recorded if e["kind"] == "call"]
    # citations were re-mapped to the fresh runs and their values refreshed
    for ref in report.numbers:
        assert ref.run_id in {e["run_id"] for e in fresh if e["kind"] == "result"}
    if task_id.endswith("tool-error"):
        assert report.invalid_calls == 0 and report.answer["a_A"] == pytest.approx(4.48, abs=0.05)


def test_mock_with_stored_results_computes_nothing(
    agent_settings: Settings,
    task_specs: list[tasks.TaskSpec],
    traces_dir: Path,
    tool_kwargs: dict[str, Any],
) -> None:
    spec = tasks.select_task(task_specs, "t02-relax-a0-MnSi")
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(
        spec.task, MockBackend(traces_dir, replay_results=True), agent_settings, ctx, **tool_kwargs
    )
    assert report.tool_calls == 2 and report.answer["a_A"] == pytest.approx(4.56, abs=0.05)
    assert not report.grounded and report.numbers == []  # stored run ids do not exist here
    runs = Path(agent_settings.paths.runs_dir) / "agent.run"
    assert len([p for p in runs.iterdir() if p.is_dir()]) == 1  # only the agent run itself


def test_mock_needs_a_trace(
    agent_settings: Settings, tmp_path: Path, tool_kwargs: dict[str, Any]
) -> None:
    task = AgentTask(task_id="nope", prompt="x", tools_allowed=[])
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    with pytest.raises(BackendUnavailable, match="no recorded trace"):
        runner.run(task, MockBackend(tmp_path / "traces"), agent_settings, ctx, **tool_kwargs)
    with pytest.raises(BackendUnavailable, match="needs a recorded trace"):
        make_backend("mock")
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("gpt")


def test_anthropic_backend_never_builds_a_client_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("anthropic.Anthropic() must not be constructed without a key")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", forbidden)
    with pytest.raises(BackendUnavailable, match="ANTHROPIC_API_KEY is not set"):
        AnthropicBackend()
    with pytest.raises(BackendUnavailable):
        make_backend("anthropic")
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_anthropic_backend_with_an_injected_fake_client(
    agent_settings: Settings, tool_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live loop, driven by a fake tool_runner: guarded tools run, usage is traced."""
    import anthropic

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no client"))
    )
    seen: dict[str, Any] = {}

    class Usage:
        def __init__(self, i: int, o: int) -> None:
            self.input_tokens, self.output_tokens = i, o

    class Block:
        def __init__(self, text: str) -> None:
            self.type, self.text = "text", text

    class Message:
        def __init__(self, text: str, usage: Usage, stop: str) -> None:
            self.content, self.usage, self.stop_reason = [Block(text)], usage, stop

    class FakeRunner:
        def __init__(self, tools: list[Any]) -> None:
            self.tools = {t.name: t for t in tools}

        def __iter__(self) -> Any:
            s = json.loads(self.tools["get_structure"].call({"compound": "FeSi"}))
            yield Message("", Usage(1000, 50), "tool_use")
            r = json.loads(
                self.tools["relax"].call(
                    {"compound": "FeSi", "frame_id": s["frame_id"], "fmax": 0.0, "steps": 0}
                )
            )
            yield Message("", Usage(1500, 60), "tool_use")
            final = format_answer(
                {"a_A": r["a_A"]}, [{"key": "a_A", "value": r["a_A"], "run_id": r["run_id"]}], "ok"
            )
            yield Message(final, Usage(1800, 120), "end_turn")

    class Messages:
        def tool_runner(self, **kw: Any) -> FakeRunner:
            seen.update(kw)
            return FakeRunner(list(kw["tools"]))

    class Beta:
        messages = Messages()

    class Client:
        beta = Beta()

    backend = AnthropicBackend("claude-opus-5", client=Client())
    task = AgentTask(
        task_id="live-fake",
        prompt="Relax FeSi and report a_A",
        tools_allowed=["get_structure", "relax"],
    )
    ctx = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    report = runner.run(task, backend, agent_settings, ctx, **tool_kwargs)
    assert seen["model"] == "claude-opus-5" and seen["max_tokens"] == 16000
    assert seen["fallbacks"] == "default" and seen["betas"] == ["server-side-fallback-2026-07-01"]
    assert [t.name for t in seen["tools"]] == ["get_structure", "relax"]
    assert all(t.to_dict()["strict"] is True for t in seen["tools"])
    assert seen["max_iterations"] == 27 and "materials scientist" in seen["system"]
    assert (
        report.backend == "anthropic"
        and report.tool_calls == 2
        and report.grounded
        and report.dag_valid
    )
    assert report.tokens_in == 4300 and report.tokens_out == 230
    assert report.cost_usd == pytest.approx(cost_usd("claude-opus-5", 4300, 230))
    events = Trace.read(report.trace_path)
    assert sum(1 for e in events if e["kind"] == "usage") == 3
    assert events[-1]["kind"] == "message" and events[-1]["role"] == "assistant"
