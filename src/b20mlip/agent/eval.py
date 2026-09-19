"""Agent eval (SPEC.md section 7): score one run, aggregate the 12-task eval, publish numbers.

``score(report, task)`` (CONTRACTS row 13)

* ``accuracy`` — the fraction of gold keys whose answer value lies within the task's per-key
  ``tolerance`` (absolute; default :data:`DEFAULT_TOLERANCE`); strings must match exactly;
  a missing key counts as wrong; ``None`` when the task carries no gold;
* ``correct`` — every gold key matched;
* ``invalid_call_rate`` — ``invalid_calls / tool_calls`` (0 when no call was made);
* ``dag_valid``, ``provenance`` (= ``AgentReport.grounded``);
* ``recovery`` — for tasks with an injected failure: ``correct`` (``None`` otherwise);
* ``tokens`` (in + out) and ``usd``.

``run(cfg, ctx, tasks=..., backend=...)`` runs every task as its own ``agent.run`` stage,
writes ``eval.json`` (per-task scores and reports), ``traces.jsonl`` (task -> run -> trace) and
``numbers.json`` with ``agent.eval.{accuracy, invalid_call_rate, dag_valid_rate,
provenance_rate, recovery_rate, tokens_per_task, usd_per_task, n_tasks}``: means over tasks
with a seeded percentile bootstrap over tasks (``ci95``; ``null`` with a reason below two
tasks) and the A10 meta (``backend``, ``model_id``, ``trace_path``; mock numbers are never
rendered into the README by the report tier).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from b20mlip.agent import runner
from b20mlip.agent.backends import BackendUnavailable
from b20mlip.agent.runner import REFERENCE_META
from b20mlip.agent.tasks import TaskSpec, apply_gold, plans_of
from b20mlip.config import Settings
from b20mlip.evaluate import bootstrap
from b20mlip.models import AgentReport, AgentTask
from b20mlip.provenance import RunContext

DEFAULT_TOLERANCE = 1e-6
EVAL_FILE = "eval.json"
TRACES_FILE = "traces.jsonl"
NUMBERS_FILE = "numbers.json"
PLAN_FILE = "plan.json"
METRICS: tuple[str, ...] = (
    "accuracy",
    "invalid_call_rate",
    "dag_valid_rate",
    "provenance_rate",
    "recovery_rate",
    "tokens_per_task",
    "usd_per_task",
    "n_tasks",
)
UNITS: dict[str, str] = {
    "accuracy": "fraction",
    "invalid_call_rate": "fraction",
    "dag_valid_rate": "fraction",
    "provenance_rate": "fraction",
    "recovery_rate": "fraction",
    "tokens_per_task": "tokens",
    "usd_per_task": "USD",
    "n_tasks": "count",
}


def _matches(answer: Any, gold: Any, tol: float) -> bool:
    if isinstance(gold, bool) or isinstance(answer, bool):
        return bool(answer) == bool(gold)
    if isinstance(gold, (int, float)):
        try:
            return abs(float(answer) - float(gold)) <= float(tol)
        except (TypeError, ValueError):
            return False
    return str(answer) == str(gold)


def score(
    report: AgentReport, task: AgentTask, *, default_tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, Any]:
    """Per-task scores (see the module docstring)."""
    per_key: dict[str, bool] = {}
    accuracy: float | None = None
    correct: bool | None = None
    if task.gold:
        for key, gold in task.gold.items():
            tol = float(task.tolerance.get(key, default_tolerance))
            per_key[key] = key in report.answer and _matches(report.answer[key], gold, tol)
        accuracy = sum(per_key.values()) / len(per_key)
        correct = all(per_key.values())
    rate = report.invalid_calls / report.tool_calls if report.tool_calls else 0.0
    return {
        "task_id": task.task_id,
        "accuracy": accuracy,
        "correct": correct,
        "per_key": per_key,
        "invalid_call_rate": float(rate),
        "dag_valid": bool(report.dag_valid),
        "provenance": bool(report.grounded),
        "recovery": (bool(correct) if task.injected_failure else None),
        "injected_failure": task.injected_failure,
        "tokens": int(report.tokens_in + report.tokens_out),
        "usd": float(report.cost_usd),
        "tool_calls": report.tool_calls,
        "invalid_calls": report.invalid_calls,
        "violations": len(report.violations),
        "wall_s": report.wall_s,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _ci(values: Sequence[float], seed: int, n_boot: int) -> tuple[list[float] | None, str | None]:
    if len(values) < 2:
        return None, "fewer than two tasks; no bootstrap over tasks"
    groups = {f"task{i}": np.asarray([v], dtype=float) for i, v in enumerate(values)}
    lo, hi = bootstrap.ci(groups, n_boot, seed, np.mean)
    return [float(lo), float(hi)], None


def aggregate(
    scores: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    model_id: str,
    trace_path: str,
    seed: int = 0,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """``agent.eval.*`` numbers with ``@meta`` (A10: backend, model_id, trace_path)."""
    series: dict[str, list[float]] = {
        "accuracy": [float(s["accuracy"]) for s in scores if s.get("accuracy") is not None],
        "invalid_call_rate": [float(s["invalid_call_rate"]) for s in scores],
        "dag_valid_rate": [1.0 if s["dag_valid"] else 0.0 for s in scores],
        "provenance_rate": [1.0 if s["provenance"] else 0.0 for s in scores],
        "recovery_rate": [
            1.0 if s["recovery"] else 0.0 for s in scores if s.get("recovery") is not None
        ],
        "tokens_per_task": [float(s["tokens"]) for s in scores],
        "usd_per_task": [float(s["usd"]) for s in scores],
    }
    base_meta = {
        "backend": backend,
        "model_id": model_id,
        "trace_path": trace_path,
        "seed": int(seed),
        "reference": dict(REFERENCE_META),
        "head": "Default",
        "e0_source": "foundation",
        "bootstrap_n": int(n_boot),
    }
    out: dict[str, Any] = {}
    for name in METRICS:
        if name == "n_tasks":
            out["agent.eval.n_tasks"] = len(scores)
            out["agent.eval.n_tasks@meta"] = {
                **base_meta, "n": len(scores), "ci95": None,
                "ci95_reason": "a count", "unit": UNITS[name],
            }  # fmt: skip
            continue
        values = series[name]
        mean = _mean(values)
        if mean is None or not math.isfinite(mean):
            continue  # nothing to publish (no gold / no injected failures)
        ci, reason = _ci(values, seed, n_boot)
        meta = {**base_meta, "n": len(values), "ci95": ci, "unit": UNITS[name]}
        if reason:
            meta["ci95_reason"] = reason
        out[f"agent.eval.{name}"] = mean
        out[f"agent.eval.{name}@meta"] = meta
    return out


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    tasks: Sequence[TaskSpec],
    backend: Any,
    gold: Mapping[str, dict[str, Any]] | None = None,
    limit: int | None = None,
    n_boot: int | None = None,
    **tc_kwargs: Any,
) -> dict[str, Any]:
    """Stage ``agent.eval`` (``b20mlip agent eval``)."""
    specs = apply_gold(list(tasks)[: limit if limit else None], dict(gold) if gold else None)
    if hasattr(backend, "plans"):
        for task_id, task_plan in plans_of(specs).items():
            backend.plans.setdefault(task_id, task_plan)
    seed = int(ctx.seed if ctx.seed is not None else 0)
    boot = int(n_boot if n_boot is not None else cfg.eval.bootstrap_n)
    plan: dict[str, Any] = {
        "backend": backend.name,
        "model_id": getattr(backend, "model_id", None),
        "n_tasks": len(specs),
        "task_ids": [s.task.task_id for s in specs],
        "with_gold": sum(1 for s in specs if s.task.gold),
        "seed": seed,
        "dry_run": bool(ctx.dry_run),
    }
    (ctx.out_dir / PLAN_FILE).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / PLAN_FILE, "json")
    if ctx.dry_run:
        ctx.log(backend=backend.name, model_id=plan["model_id"], n_tasks=len(specs), planned=True)
        return {"planned": len(specs), "backend": backend.name}

    scores: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    for spec in specs:
        report, res = runner.run_task_stage(
            cfg, spec.task, backend, seed=seed, runs_dir=ctx.runs_dir, executor=ctx.executor,
            **tc_kwargs,
        )  # fmt: skip
        if report is None and "BackendUnavailable" in str(res.summary.get("error", "")):
            raise BackendUnavailable(
                str(res.summary["error"])
            )  # a setup error, not an agent failure
        if report is None:  # the task stage crashed before writing its report
            report = AgentReport(
                task_id=spec.task.task_id, backend=backend.name,
                model_id=getattr(backend, "model_id", None), answer={}, numbers=[], tool_calls=0,
                invalid_calls=0, dag_valid=False, grounded=False,
                violations=[str(res.summary.get("error", "run failed"))], tokens_in=0,
                tokens_out=0, cost_usd=0.0, wall_s=0.0, trace_path="",
            )  # fmt: skip
        s = score(report, spec.task)
        s["run_id"] = res.run_id
        s["status"] = res.status
        scores.append(s)
        reports.append(report.model_dump(mode="json"))
        trace_rows.append(
            {"task_id": spec.task.task_id, "run_id": res.run_id, "trace_path": report.trace_path}
        )
        if report.model_id:
            model_ids.add(report.model_id)

    traces_path = ctx.out_dir / TRACES_FILE
    traces_path.write_text("".join(json.dumps(r) + "\n" for r in trace_rows), encoding="utf-8")
    ctx.add_output(traces_path, "json")
    model_id = getattr(backend, "model_id", None) or (
        sorted(model_ids)[0] if model_ids else backend.name
    )
    numbers = aggregate(
        scores, backend=backend.name, model_id=str(model_id), trace_path=str(traces_path),
        seed=seed, n_boot=boot,
    )  # fmt: skip
    (ctx.out_dir / NUMBERS_FILE).write_text(
        json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ctx.add_output(ctx.out_dir / NUMBERS_FILE, "json")
    payload = {
        "schema": "b20mlip.agent.eval.v1",
        "backend": backend.name,
        "model_id": model_id,
        "seed": seed,
        "scores": scores,
        "reports": reports,
        "numbers": {k: v for k, v in numbers.items() if not k.endswith("@meta")},
    }
    (ctx.out_dir / EVAL_FILE).write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    ctx.add_output(ctx.out_dir / EVAL_FILE, "json")
    tokens_in = sum(int(r["tokens_in"]) for r in reports)
    tokens_out = sum(int(r["tokens_out"]) for r in reports)
    cost = sum(float(r["cost_usd"]) for r in reports)
    ctx.log(
        backend=backend.name, model_id=model_id, tokens_in=tokens_in, tokens_out=tokens_out,
        cost_usd=cost, n_tasks=len(scores), trace_path=str(traces_path),
        task_ids=[s["task_id"] for s in scores],
    )  # fmt: skip
    summary: dict[str, Any] = {
        "backend": backend.name,
        "model_id": str(model_id),
        "n_tasks": len(scores),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
    }
    for name in METRICS:
        key = f"agent.eval.{name}"
        if key in numbers:
            summary[name] = round(float(numbers[key]), 6)
    return summary


__all__ = [
    "DEFAULT_TOLERANCE",
    "EVAL_FILE",
    "METRICS",
    "NUMBERS_FILE",
    "PLAN_FILE",
    "TRACES_FILE",
    "UNITS",
    "aggregate",
    "run",
    "score",
]
