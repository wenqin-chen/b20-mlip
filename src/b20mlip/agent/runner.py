"""Agent runner (CONTRACTS.md row 13): ``run(task, backend, cfg, ctx) -> AgentReport``.

The runner owns everything a backend must not: it sets the :class:`~b20mlip.agent.tools.
ToolContext`, wraps every tool call in the :class:`~b20mlip.agent.guard.Guard` and the
:class:`~b20mlip.agent.trace.Trace` (the :class:`Dispatcher`), injects the task's failure,
and scores the run's structure:

* ``tool_calls`` — every attempted call (blocked ones included);
* ``invalid_calls`` — guard violations plus tool errors (``{"error": ...}`` results);
* ``dag_valid`` — the executed calls respect the tool DAG: ``get_structure``/``list_data``
  before ``relax``/``phonons``/``run_md``; ``compare_phonons`` after the ``phonons`` run it
  cites; ``select_frames`` after ``evaluate_errors``; ``write_report`` last;
* ``grounded`` — the answer is non-empty and every numeric value in it is cited with a run id
  whose manifest is ``ok`` and whose ``result.json`` holds that value
  (:func:`~b20mlip.agent.tools.resolve_number`);
* ``tokens`` / ``cost_usd`` — from the backend
  (:data:`~b20mlip.agent.backends.PRICES_USD_PER_MTOK`).

Injected failures (``AgentTask.injected_failure``; each fires once):

* ``tool_error`` — the first ``relax`` call returns a transient error without running;
* ``bad_structure`` — the first ``get_structure`` result is a copy of the reference cell with
  the transition metal replaced by Cu (outside the allowlist); the guard blocks any later call
  that uses that ``frame_id`` and the agent must fetch the structure again;
* ``budget_exhausted`` — the guard runs with ``max_calls=3``;
* ``sum_rule`` — the first ``phonons`` call runs without the acoustic sum rule and is flagged
  ``asr_violation=true``; the agent must re-run it with ``asr=true``.

Outputs under ``ctx.out_dir``: ``trace.jsonl``, ``report.json`` (the :class:`AgentReport`
plus diagnostics) and ``numbers.json`` (``agent.run.<task_id>.*`` with the A10 meta:
``backend``, ``model_id``, ``trace_path``). Manifest extras: ``backend``, ``model_id``,
``tokens_in``, ``tokens_out``, ``cost_usd`` (CONTRACTS section 5).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anthropic import beta_tool

from b20mlip.agent import tools as tools_mod
from b20mlip.agent.backends import BackendResult, BackendUnavailable
from b20mlip.agent.guard import DEFAULT_ALLOWLIST, Guard, GuardViolation
from b20mlip.agent.tools import ToolContext, resolve_number, use_context
from b20mlip.agent.trace import ToolCall, Trace, sha256_json
from b20mlip.config import Settings
from b20mlip.io import frame_id_for
from b20mlip.md.common import sanitize_key_segment
from b20mlip.models import AgentReport, AgentTask, Budget, NumberRef, StageResult
from b20mlip.provenance import RunContext, run_stage, sha256_file

TRACE_FILE = "trace.jsonl"
REPORT_FILE = "report.json"
NUMBERS_FILE = "numbers.json"
PLAN_FILE = "plan.json"
INJECTED_BUDGET_CALLS = 3
TM_NUMBERS: tuple[int, ...] = (25, 26, 27)  # Mn, Fe, Co -> replaced by Cu in bad_structure
BAD_ELEMENT_Z = 29
CONTEXT_TOOLS: frozenset[str] = frozenset({"get_structure", "list_data"})
STRUCTURE_TOOLS: frozenset[str] = frozenset({"relax", "phonons", "run_md"})
RUN_METRICS: tuple[str, ...] = (
    "tool_calls",
    "invalid_calls",
    "dag_valid",
    "grounded",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "wall_s",
)
REFERENCE_META: dict[str, Any] = {
    "code": "mace",
    "functional": "PBE",
    "pseudos": None,
    "e0_source": "foundation",
    "cross_functional": False,
}


# --- call records and injected failures ---------------------------------------------------------


@dataclass
class CallRecord:
    index: int
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    blocked: bool = False
    error: bool = False
    run_id: str | None = None
    manifest_sha256: str | None = None
    violation: str | None = None
    injected: str | None = None
    wall_s: float = 0.0

    @property
    def invalid(self) -> bool:
        return self.blocked or self.error


class Injection:
    """The task's injected failure (see the module docstring); ``fired`` once it happened."""

    def __init__(self, kind: str | None) -> None:
        self.kind = kind
        self.fired = False
        self._pending_sum_rule = False

    def budget(self, budget: Budget) -> Budget:
        if self.kind == "budget_exhausted":
            self.fired = True
            return budget.model_copy(update={"max_calls": INJECTED_BUDGET_CALLS})
        return budget

    def before(self, call: ToolCall) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
        """``(result override, args, fired now)``: an override skips the tool entirely."""
        args = dict(call.args)
        if self.fired or self.kind is None:
            return None, args, False
        if self.kind == "tool_error" and call.name == "relax":
            self.fired = True
            return (
                {
                    "error": "RuntimeError: injected transient failure: the relaxation backend was "
                    "unavailable; retry the call"
                },
                args,
                True,
            )
        if self.kind == "sum_rule" and call.name == "phonons":
            self.fired = True
            self._pending_sum_rule = True
            args["asr"] = False
            return None, args, True
        return None, args, False

    def after(
        self, call: ToolCall, result: dict[str, Any], tc: ToolContext
    ) -> tuple[dict[str, Any], bool]:
        """``(result, fired now)``: rewrite a result after the tool ran."""
        if self._pending_sum_rule and call.name == "phonons":
            self._pending_sum_rule = False
            if "error" not in result:
                return {**result, "asr_applied": False, "asr_violation": True}, True
            return result, True
        if self.kind == "bad_structure" and call.name == "get_structure" and not self.fired:
            if "error" in result or not result.get("frame_id"):
                return result, False
            entry = tc.structures.get(str(result["frame_id"]))
            if entry is None:
                return result, False
            self.fired = True
            frame = entry[0]
            numbers = [BAD_ELEMENT_Z if z in TM_NUMBERS else z for z in frame.numbers]
            bad = frame.model_copy(
                update={
                    "numbers": numbers,
                    "frame_id": frame_id_for(numbers, frame.positions, frame.cell),
                    "compound": "CuSi" if 14 in numbers else "CuGe",
                }
            )
            tc.register_structure(bad, "injected")
            elements = sorted({tools_mod.elements_of_number(z) for z in numbers})
            return {
                **result,
                "frame_id": bad.frame_id,
                "formula": bad.compound,
                "elements": elements,
                "source": "reference",
            }, True
        return result, False


# --- the dispatcher -----------------------------------------------------------------------------


class Dispatcher:
    """Guard + trace + injection around every tool call, for every backend.

    ``call(name, args)`` returns the tool's result dict (``{"error": ...}`` for blocked calls and
    failures); ``beta_tools(allowed)`` returns guarded ``@beta_tool`` objects for the live
    backend (same schemas as :data:`~b20mlip.agent.tools.TOOLS`).
    """

    def __init__(
        self,
        context: ToolContext,
        guard: Guard,
        trace: Trace,
        *,
        injection: Injection | None = None,
        allowed: Iterable[str] | None = None,
    ) -> None:
        self.context = context
        self.guard = guard
        self.trace = trace
        self.injection = injection or Injection(None)
        self.allowed = frozenset(allowed) if allowed else frozenset(tools_mod.TOOL_NAMES)
        self.calls: list[CallRecord] = []

    @property
    def names(self) -> list[str]:
        return [n for n in tools_mod.TOOL_NAMES if n in self.allowed]

    def call(
        self, name: str, args: Mapping[str, Any], *, stored_result: Any = None
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        call = ToolCall(str(name), dict(args), len(self.calls))
        self.trace.call(call)
        record = CallRecord(call.index, call.name, call.args, {})
        try:
            self.guard.check(call)
        except GuardViolation as exc:
            record.blocked = True
            record.violation = str(exc)
            record.result = {"error": str(exc), "rule": exc.rule}
            self.trace.violation(call, str(exc))
            record.wall_s = time.perf_counter() - t0
            self.calls.append(record)
            return dict(record.result)
        override, run_args, fired_before = self.injection.before(call)
        if override is not None:
            result = dict(override)
        elif stored_result is not None:
            result = dict(stored_result)
        else:
            result = self._execute(call.name, run_args)
        result, fired_after = self.injection.after(call, result, self.context)
        if fired_before or fired_after:
            record.injected = self.injection.kind
        record.result = result
        record.error = "error" in result
        run_id = result.get("run_id")
        record.run_id = str(run_id) if isinstance(run_id, str) else None
        manifest_path = self.context.runs.get(record.run_id or "")
        if manifest_path is not None and Path(manifest_path).is_file():
            record.manifest_sha256 = sha256_file(manifest_path)
        self.trace.result(call, result, manifest_sha=record.manifest_sha256, run_id=record.run_id)
        record.wall_s = time.perf_counter() - t0
        self.calls.append(record)
        return dict(result)

    def _execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = tools_mod.TOOLS_BY_NAME[name]
        try:
            text = tool.call(args)
        except Exception as exc:  # noqa: BLE001 - invalid arguments are a tool error, not a crash
            return {"error": f"{type(exc).__name__}: invalid arguments for {name}: {exc}"}
        try:
            return tools_mod.parse_result(str(text))
        except (ValueError, TypeError) as exc:
            return {"error": f"tool {name} returned no JSON object: {exc}"}

    def beta_tools(self, allowed: Iterable[str] | None = None) -> list[Any]:
        names = [n for n in self.names if allowed is None or n in set(allowed)]
        out: list[Any] = []
        for name in names:
            out.append(self._guarded(tools_mod.TOOLS_BY_NAME[name]))
        return out

    def _guarded(self, tool: Any) -> Any:
        dispatcher = self

        def wrapped(**kwargs: Any) -> str:
            return json.dumps(dispatcher.call(tool.name, kwargs), sort_keys=True, default=str)

        wrapped.__name__ = tool.name
        wrapped.__doc__ = tool.description
        return beta_tool(
            wrapped,
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            strict=True,
        )

    # -- summaries -------------------------------------------------------------------------------

    @property
    def invalid_calls(self) -> int:
        return sum(1 for c in self.calls if c.invalid)

    def records(self) -> list[dict[str, Any]]:
        out = []
        for c in self.calls:
            d = asdict(c)
            d["args_sha256"] = sha256_json(c.args)
            d.pop("result", None)
            d["result_keys"] = sorted(c.result)
            out.append(d)
        return out


# --- DAG order and grounding --------------------------------------------------------------------


def dag_valid(calls: Sequence[CallRecord]) -> tuple[bool, list[str]]:
    """Whether the executed (non-blocked) calls respect the tool DAG (module docstring)."""
    executed = [c for c in calls if not c.blocked]
    reasons: list[str] = []
    seen_context = False
    seen_eval = False
    phonon_runs: set[str] = set()
    report_at: int | None = None
    for i, c in enumerate(executed):
        if c.name in CONTEXT_TOOLS:
            seen_context = True
        if c.name in STRUCTURE_TOOLS and not seen_context:
            reasons.append(f"call {c.index}: {c.name} before any get_structure/list_data")
        if c.name == "compare_phonons":
            wanted = str(c.args.get("run_id_model", ""))
            if wanted not in phonon_runs:
                reasons.append(f"call {c.index}: compare_phonons cites no earlier phonons run")
        if c.name == "phonons" and c.run_id and not c.error:
            phonon_runs.add(c.run_id)
        if c.name == "evaluate_errors":
            seen_eval = True
        if c.name == "select_frames" and not seen_eval:
            reasons.append(f"call {c.index}: select_frames before evaluate_errors")
        if c.name == "write_report":
            report_at = i
    if report_at is not None and report_at != len(executed) - 1:
        reasons.append("write_report was not the last call")
    return not reasons, reasons


def ground(
    answer: Mapping[str, Any], citations: Sequence[Mapping[str, Any]], runs_dir: str | Path
) -> tuple[bool, list[NumberRef], list[str]]:
    """``(grounded, resolved refs, problems)``: every numeric answer value must be cited by a
    resolving run (:func:`resolve_number`); string values need no citation."""
    refs: list[NumberRef] = []
    problems: list[str] = []
    resolved: dict[str, NumberRef] = {}
    for cite in citations:
        try:
            ref = resolve_number(runs_dir, str(cite["key"]), cite["value"], str(cite["run_id"]))
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"citation {cite}: {exc}")
            continue
        refs.append(ref)
        resolved.setdefault(ref.key, ref)
    for key, value in answer.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        cited = resolved.get(key)
        if cited is None:
            problems.append(f"{key}: numeric answer without a resolving citation")
        elif not math.isclose(cited.value, float(value), rel_tol=1e-6, abs_tol=1e-9):
            problems.append(f"{key}: answer {value} differs from the cited {cited.value}")
    grounded = bool(answer) and not problems
    return grounded, refs, problems


# --- the run ------------------------------------------------------------------------------------


def run(
    task: AgentTask,
    backend: Any,
    cfg: Settings,
    ctx: RunContext,
    *,
    tool_context: ToolContext | None = None,
    allowlist: Iterable[str] = DEFAULT_ALLOWLIST,
    **tc_kwargs: Any,
) -> AgentReport:
    """Run one task through ``backend`` under guard + trace; write the run's files; report.

    ``tc_kwargs`` (``model``, ``committee``, ``frames``, ``splits``, ``structure``, ``refs_dir``,
    ``calc``, ``head``) build the :class:`ToolContext` when none is given.
    """
    t0 = time.perf_counter()
    seed = int(ctx.seed if ctx.seed is not None else 0)
    if tool_context is None:
        tool_context = ToolContext(cfg, ctx=ctx, seed=seed, runs_dir=ctx.runs_dir, **tc_kwargs)
    tc = tool_context
    tc.ctx = ctx
    tc.runs_dir = Path(ctx.runs_dir)
    tc.dry_run = bool(ctx.dry_run)
    injection = Injection(task.injected_failure)
    budget = injection.budget(task.budget)
    tc.budget = budget
    allowed = (
        frozenset(task.tools_allowed) if task.tools_allowed else frozenset(tools_mod.TOOL_NAMES)
    )
    unknown = sorted(allowed - set(tools_mod.TOOL_NAMES))
    if unknown:
        raise ValueError(f"task {task.task_id}: unknown tools_allowed {unknown}")
    trace = Trace(ctx.out_dir / TRACE_FILE)
    model_id = getattr(backend, "model_id", None)
    trace.message(
        "system",
        f"agent run {ctx.run_id}",
        backend=backend.name,
        model_id=model_id,
        task_id=task.task_id,
        budget=budget.model_dump(mode="json"),
        injected_failure=task.injected_failure,
        tools_allowed=sorted(allowed),
    )
    guard = Guard(
        budget,
        frozenset(allowlist),
        tool_names=allowed,
        frames_resolver=tc.frames_natoms,
        frames_elements_resolver=tc.frames_elements,
        structure_resolver=tc.frame_elements,
        structure_natoms_resolver=tc.frame_natoms,
    )
    dispatcher = Dispatcher(tc, guard, trace, injection=injection, allowed=allowed)
    with use_context(tc):
        try:
            result: BackendResult = backend.run(task, dispatcher)
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a crashed backend is an invalid run, not a crash
            result = BackendResult(
                answer={}, model_id=model_id, error=f"{type(exc).__name__}: {exc}"
            )
            trace.message("assistant", "", error=result.error)
    ok_dag, dag_reasons = dag_valid(dispatcher.calls)
    grounded, refs, ground_problems = ground(result.answer, result.citations, tc.runs_dir)
    wall = time.perf_counter() - t0
    report = AgentReport(
        task_id=task.task_id,
        backend=backend.name,
        model_id=str(result.model_id) if result.model_id is not None else None,
        answer=dict(result.answer),
        numbers=refs,
        tool_calls=len(dispatcher.calls),
        invalid_calls=dispatcher.invalid_calls,
        dag_valid=ok_dag,
        grounded=grounded,
        violations=list(guard.violations),
        tokens_in=int(result.tokens_in),
        tokens_out=int(result.tokens_out),
        cost_usd=float(result.cost_usd),
        wall_s=float(wall),
        trace_path=str(trace.path),
    )
    diagnostics = {
        "dag_reasons": dag_reasons,
        "grounding_problems": ground_problems,
        "citations": list(result.citations),
        "text": result.text,
        "stop_reason": result.stop_reason,
        "backend_error": result.error,
        "injected_failure": task.injected_failure,
        "injection_fired": injection.fired,
        "effective_budget": budget.model_dump(mode="json"),
        "calls": dispatcher.records(),
    }
    write_run_files(ctx, report, task, diagnostics, seed=seed)
    ctx.log(
        backend=backend.name,
        model_id=report.model_id,
        tokens_in=report.tokens_in,
        tokens_out=report.tokens_out,
        cost_usd=report.cost_usd,
        task_id=task.task_id,
        trace_path=str(trace.path),
        tool_calls=report.tool_calls,
        invalid_calls=report.invalid_calls,
        dag_valid=report.dag_valid,
        grounded=report.grounded,
        injected_failure=task.injected_failure,
    )
    return report


def run_numbers(report: AgentReport, task: AgentTask, *, seed: int) -> dict[str, Any]:
    """``agent.run.<task_id>.<metric>`` entries with the A10 meta (mock ones never reach README)."""
    base = f"agent.run.{sanitize_key_segment(task.task_id)}"
    meta = {
        "backend": report.backend,
        "model_id": report.model_id or report.backend,
        "trace_path": report.trace_path,
        "n": 1,
        "seed": int(seed),
        "ci95": None,
        "ci95_reason": "a single agent run; no resampling",
        "reference": dict(REFERENCE_META),
        "head": "Default",
        "e0_source": "foundation",
        "task_id": task.task_id,
        "injected_failure": task.injected_failure,
    }
    values: dict[str, Any] = {
        "tool_calls": report.tool_calls,
        "invalid_calls": report.invalid_calls,
        "dag_valid": int(report.dag_valid),
        "grounded": int(report.grounded),
        "tokens_in": report.tokens_in,
        "tokens_out": report.tokens_out,
        "cost_usd": report.cost_usd,
        "wall_s": report.wall_s,
    }
    units = {"cost_usd": "USD", "wall_s": "s"}
    out: dict[str, Any] = {}
    for name in RUN_METRICS:
        out[f"{base}.{name}"] = values[name]
        out[f"{base}.{name}@meta"] = {**meta, "unit": units.get(name, "count")}
    return out


def write_run_files(
    ctx: RunContext,
    report: AgentReport,
    task: AgentTask,
    diagnostics: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    payload = {
        "schema": "b20mlip.agent.report.v1",
        "report": report.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "diagnostics": dict(diagnostics),
    }
    report_path = ctx.out_dir / REPORT_FILE
    report_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    ctx.add_output(report_path, "json")
    numbers_path = ctx.out_dir / NUMBERS_FILE
    numbers_path.write_text(
        json.dumps(run_numbers(report, task, seed=seed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ctx.add_output(numbers_path, "json")
    if Path(report.trace_path).is_file():
        ctx.add_output(report.trace_path, "json")


def read_report(run_dir: str | Path) -> AgentReport:
    data = json.loads((Path(run_dir) / REPORT_FILE).read_text(encoding="utf-8"))
    return AgentReport.model_validate(data["report"])


# --- stage wrappers -----------------------------------------------------------------------------


def stage(
    cfg: Settings, ctx: RunContext, *, task: AgentTask, backend: Any, **tc_kwargs: Any
) -> dict[str, Any]:
    """Stage ``agent.run`` (``b20mlip agent run``): plan only under ``--dry-run``."""
    plan = {
        "task_id": task.task_id,
        "backend": backend.name,
        "model_id": getattr(backend, "model_id", None),
        "budget": task.budget.model_dump(mode="json"),
        "tools_allowed": list(task.tools_allowed),
        "injected_failure": task.injected_failure,
        "model": str(tc_kwargs.get("model")) if tc_kwargs.get("model") else None,
        "dry_run": bool(ctx.dry_run),
    }
    (ctx.out_dir / PLAN_FILE).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / PLAN_FILE, "json")
    if ctx.dry_run:
        ctx.log(backend=backend.name, model_id=plan["model_id"], task_id=task.task_id, planned=True)
        return {"planned": 1, "task_id": task.task_id, "backend": backend.name}
    report = run(task, backend, cfg, ctx, **tc_kwargs)
    scored: dict[str, Any] = {}
    if task.gold:
        from b20mlip.agent.eval import score  # noqa: PLC0415 - eval builds on the runner

        s = score(report, task)
        scored = {"accuracy": float(s["accuracy"]), "correct": int(bool(s["correct"]))}
    return {
        **scored,
        "task_id": report.task_id,
        "backend": report.backend,
        "model_id": report.model_id or "",
        "tool_calls": report.tool_calls,
        "invalid_calls": report.invalid_calls,
        "dag_valid": int(report.dag_valid),
        "grounded": int(report.grounded),
        "violations": len(report.violations),
        "tokens_in": report.tokens_in,
        "tokens_out": report.tokens_out,
        "cost_usd": report.cost_usd,
        "wall_s": round(report.wall_s, 3),
        "answer": json.dumps(report.answer, sort_keys=True, default=str),
    }


def run_task_stage(
    cfg: Settings,
    task: AgentTask,
    backend: Any,
    *,
    seed: int | None = None,
    runs_dir: str | Path | None = None,
    executor: Any | None = None,
    dry_run: bool = False,
    resume: bool = False,
    **tc_kwargs: Any,
) -> tuple[AgentReport | None, StageResult]:
    """One task as its own ``agent.run`` stage (manifest written even on failure)."""
    res = run_stage(
        "agent.run",
        cfg,
        stage,
        seed=seed,
        resume=resume,
        executor=executor,
        dry_run=dry_run,
        runs_dir=runs_dir,
        task=task,
        backend=backend,
        **tc_kwargs,
    )
    run_dir = Path(res.manifest_path).parent
    report = read_report(run_dir) if (run_dir / REPORT_FILE).is_file() else None
    return report, res


__all__ = [
    "CONTEXT_TOOLS",
    "INJECTED_BUDGET_CALLS",
    "NUMBERS_FILE",
    "PLAN_FILE",
    "REFERENCE_META",
    "REPORT_FILE",
    "RUN_METRICS",
    "STRUCTURE_TOOLS",
    "TRACE_FILE",
    "CallRecord",
    "Dispatcher",
    "Injection",
    "dag_valid",
    "ground",
    "read_report",
    "run",
    "run_numbers",
    "run_task_stage",
    "stage",
    "write_run_files",
]
