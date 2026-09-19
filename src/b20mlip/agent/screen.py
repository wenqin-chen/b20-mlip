"""``b20mlip screen``: the deterministic gold script (CONTRACTS.md section 3).

Runs every task of a task file through the :class:`~b20mlip.agent.backends.ScriptedBackend`
(the canonical DAG plan per task kind; the task's injected failure is stripped, gold is the
clean value) with the model under test and
writes a gold file (``evals/gold_<model_label>.json``, schema ``b20mlip.agent.gold.v1``) that
``agent eval --gold`` reads. Gold is recomputed rather than stored in the JSONL because every
value depends on the model, the configuration (supercell, fmax, seed) and the references.

``--compounds`` restricts the run to tasks whose compounds are all in the list (tasks without
a compound always run). The stage is recorded as ``agent.eval`` with ``extras.screen = true``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from b20mlip.agent import runner
from b20mlip.agent.backends import ScriptedBackend, parse_params
from b20mlip.agent.tasks import GOLD_SCHEMA, TaskSpec, plans_of
from b20mlip.config import Settings
from b20mlip.md.common import model_provenance
from b20mlip.provenance import RunContext, sha256_file

GOLD_FILE = "gold.json"


def task_compounds(spec: TaskSpec) -> list[str]:
    params = {**parse_params(spec.task.prompt), **spec.params}
    compounds = params.get("compounds") or ([params["compound"]] if params.get("compound") else [])
    return [str(c) for c in compounds]


def filter_tasks(specs: Iterable[TaskSpec], compounds: Iterable[str] | None) -> list[TaskSpec]:
    allowed = {c.strip() for c in compounds if c.strip()} if compounds else None
    if not allowed:
        return list(specs)
    return [s for s in specs if set(task_compounds(s)) <= allowed]


def graded_keys(spec: TaskSpec, answer: dict[str, Any]) -> list[str]:
    keys = [k for k in spec.task.tolerance if k in answer]
    return keys or sorted(answer)


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    tasks: Sequence[TaskSpec],
    compounds: Iterable[str] | None = None,
    out: str | Path | None = None,
    limit: int | None = None,
    **tc_kwargs: Any,
) -> dict[str, Any]:
    """Stage ``agent.eval`` in screen mode: scripted gold for every (filtered) task."""
    specs = filter_tasks(tasks, compounds)[: limit if limit else None]
    model = tc_kwargs.get("model")
    prov = model_provenance(model) if model else {"label": "none", "sha256": None}
    ctx.log(screen=True, backend="scripted", model_id="scripted", n_tasks=len(specs))
    if ctx.dry_run:
        return {"planned": len(specs), "model_label": str(prov["label"])}
    backend = ScriptedBackend(plans_of(specs))
    seed = int(ctx.seed if ctx.seed is not None else 0)
    entries: dict[str, dict[str, Any]] = {}
    n_ok = 0
    for spec in specs:
        clean = spec.task.model_copy(update={"injected_failure": None})  # gold is never injected
        report, res = runner.run_task_stage(
            cfg, clean, backend, seed=seed, runs_dir=ctx.runs_dir, executor=ctx.executor,
            **tc_kwargs,
        )  # fmt: skip
        answer = dict(report.answer) if report is not None else {}
        keys = graded_keys(spec, answer)
        ok = bool(report is not None and report.grounded and answer and res.status == "ok")
        n_ok += int(ok)
        entries[spec.task.task_id] = {
            "kind": spec.kind or backend.plan_for(spec.task).kind,
            "injected_failure": spec.task.injected_failure,
            "gold": {k: answer[k] for k in keys} if ok else None,
            "answer": answer,
            "graded_keys": keys,
            "run_id": res.run_id,
            "status": res.status,
            "tool_calls": report.tool_calls if report is not None else 0,
            "invalid_calls": report.invalid_calls if report is not None else 0,
            "dag_valid": bool(report.dag_valid) if report is not None else False,
            "grounded": bool(report.grounded) if report is not None else False,
        }
    payload = {
        "schema": GOLD_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "model": str(model) if model else None,
        "model_sha256": prov.get("sha256"),
        "model_label": str(prov["label"]),
        "config_sha256": cfg.sha256(),
        "seed": seed,
        "screen_run_id": ctx.run_id,
        "tasks": entries,
    }
    gold_path = ctx.out_dir / GOLD_FILE
    gold_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.add_output(gold_path, "json")
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(gold_path.read_text(encoding="utf-8"), encoding="utf-8")
        ctx.add_output(out_path, "json")
    ctx.log(
        tokens_in=0, tokens_out=0, cost_usd=0.0, n_gold=n_ok, gold_sha256=sha256_file(gold_path),
        task_ids=list(entries),
    )  # fmt: skip
    return {
        "backend": "scripted",
        "model_label": str(prov["label"]),
        "n_tasks": len(specs),
        "n_gold": n_ok,
        "out": str(out) if out is not None else str(gold_path),
    }


__all__ = ["GOLD_FILE", "filter_tasks", "graded_keys", "run", "task_compounds"]
