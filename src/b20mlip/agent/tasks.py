"""Task files (``evals/agent_tasks.jsonl``), gold files and budget presets.

One JSON object per line: the :class:`~b20mlip.models.AgentTask` fields (``task_id``,
``prompt``, ``tools_allowed``, ``budget``, ``gold``, ``tolerance``, ``injected_failure``) plus
the planner keys the scripted backend uses — ``kind`` (one of
:data:`~b20mlip.agent.backends.KINDS`), ``params`` (compound, dataset labels, n, T, ps, ...) —
and ``gold_from: "screen"`` marking that the gold values are recomputed by ``b20mlip screen``
with the model under test (the JSONL itself keeps ``gold: null``). ``note`` is free text.

Gold files (``evals/gold_<model_label>.json``) map ``task_id`` to ``{"kind", "gold", "answer",
"run_id", ...}``; ``agent eval --gold`` reads them and fills ``AgentTask.gold`` before scoring.
The graded keys are the task's ``tolerance`` keys (every answer key when ``tolerance`` is empty).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from b20mlip.agent.backends import KINDS, TaskPlan
from b20mlip.models import AgentTask, Budget

EXTRA_KEYS: tuple[str, ...] = ("kind", "params", "gold_from", "note")
GOLD_SCHEMA = "b20mlip.agent.gold.v1"
BUDGET_PRESETS: dict[str, Budget] = {
    "default": Budget(),
    "ci": Budget(
        max_calls=8,
        max_wall_s=600,
        max_dft_frames=20,
        max_node_hours=1.0,
        max_atoms=64,
        max_md_ps=1.0,
    ),
    "tight": Budget(
        max_calls=6,
        max_wall_s=300,
        max_dft_frames=10,
        max_node_hours=0.5,
        max_atoms=64,
        max_md_ps=0.5,
    ),
    "generous": Budget(max_calls=40, max_wall_s=3600, max_dft_frames=400, max_node_hours=20.0),
}


@dataclass
class TaskSpec:
    """An :class:`AgentTask` plus the planner keys the JSONL carries."""

    task: AgentTask
    kind: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    gold_from: str | None = None
    note: str | None = None

    @property
    def plan(self) -> TaskPlan | None:
        return TaskPlan(self.kind, dict(self.params)) if self.kind else None

    def record(self) -> dict[str, Any]:
        out = self.task.model_dump(mode="json")
        if self.kind:
            out["kind"] = self.kind
        if self.params:
            out["params"] = dict(self.params)
        if self.gold_from:
            out["gold_from"] = self.gold_from
        if self.note:
            out["note"] = self.note
        return out


def parse_task(record: dict[str, Any]) -> TaskSpec:
    data = dict(record)
    kind = data.pop("kind", None)
    params = data.pop("params", None) or {}
    gold_from = data.pop("gold_from", None)
    note = data.pop("note", None)
    if kind is not None and kind not in KINDS:
        raise ValueError(f"task {data.get('task_id')!r}: unknown kind {kind!r}; expected {KINDS}")
    if not isinstance(params, dict):
        raise ValueError(f"task {data.get('task_id')!r}: params must be an object")
    return TaskSpec(AgentTask.model_validate(data), kind, dict(params), gold_from, note)


def load_tasks(path: str | Path) -> list[TaskSpec]:
    """Every task of a JSONL file (blank lines and ``#`` comment lines skipped)."""
    p = Path(path)
    specs: list[TaskSpec] = []
    seen: set[str] = set()
    for number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{number}: {exc}") from exc
        spec = parse_task(record)
        if spec.task.task_id in seen:
            raise ValueError(f"{p}:{number}: duplicate task_id {spec.task.task_id!r}")
        seen.add(spec.task.task_id)
        specs.append(spec)
    if not specs:
        raise ValueError(f"{p}: no tasks")
    return specs


def write_tasks(specs: Iterable[TaskSpec], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(s.record(), sort_keys=True) + "\n" for s in specs), encoding="utf-8"
    )
    return p


def select_task(specs: Sequence[TaskSpec], task_id: str) -> TaskSpec:
    for spec in specs:
        if spec.task.task_id == task_id:
            return spec
    raise KeyError(f"no task {task_id!r}; available: {[s.task.task_id for s in specs]}")


def parse_task_ref(text: str) -> tuple[Path, str | None]:
    """``TASKS.jsonl:ID`` -> ``(path, id)``; a bare path means every task."""
    raw = str(text)
    if ":" in raw:
        head, _, tail = raw.rpartition(":")
        if head and not tail.startswith(("/", "\\")) and Path(head).suffix:
            return Path(head), tail or None
    return Path(raw), None


def task_from_prompt(
    prompt: str,
    *,
    budget: Budget | None = None,
    tools_allowed: Sequence[str] | None = None,
    task_id: str | None = None,
    injected_failure: str | None = None,
) -> TaskSpec:
    """An ad-hoc task for ``agent run --prompt`` (id = ``prompt-<sha256[:8]>``)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    task = AgentTask(
        task_id=task_id or f"prompt-{digest}",
        prompt=prompt,
        tools_allowed=list(tools_allowed or []),
        budget=budget or Budget(),
        injected_failure=injected_failure,  # type: ignore[arg-type]
    )
    return TaskSpec(task)


def budget_from_spec(text: str | None, *, approve_cluster: bool | None = None) -> Budget:
    """A preset name (``default``, ``ci``, ``tight``, ``generous``) or a JSON object."""
    if not text:
        budget = Budget()
    elif text in BUDGET_PRESETS:
        budget = BUDGET_PRESETS[text].model_copy()
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--budget must be a preset {sorted(BUDGET_PRESETS)} or a JSON object: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError("--budget JSON must be an object")
        budget = Budget.model_validate(data)
    if approve_cluster is not None:
        budget = budget.model_copy(update={"approve_cluster": bool(approve_cluster)})
    return budget


def load_gold(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != GOLD_SCHEMA:
        raise ValueError(f"{path}: not a {GOLD_SCHEMA} gold file")
    tasks = data.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"{path}: 'tasks' must be an object")
    return {str(k): dict(v) for k, v in tasks.items()}


def apply_gold(specs: Sequence[TaskSpec], gold: dict[str, dict[str, Any]] | None) -> list[TaskSpec]:
    """Copies of ``specs`` with ``task.gold`` filled from a gold file where available."""
    if not gold:
        return list(specs)
    out: list[TaskSpec] = []
    for spec in specs:
        entry = gold.get(spec.task.task_id)
        if entry is not None and isinstance(entry.get("gold"), dict):
            task = spec.task.model_copy(update={"gold": dict(entry["gold"])})
            out.append(TaskSpec(task, spec.kind, dict(spec.params), spec.gold_from, spec.note))
        else:
            out.append(spec)
    return out


def plans_of(specs: Iterable[TaskSpec]) -> dict[str, TaskPlan]:
    out: dict[str, TaskPlan] = {}
    for spec in specs:
        plan = spec.plan
        if plan is not None:
            out[spec.task.task_id] = plan
    return out


__all__ = [
    "BUDGET_PRESETS",
    "EXTRA_KEYS",
    "GOLD_SCHEMA",
    "TaskSpec",
    "apply_gold",
    "budget_from_spec",
    "load_gold",
    "load_tasks",
    "parse_task",
    "parse_task_ref",
    "plans_of",
    "select_task",
    "task_from_prompt",
    "write_tasks",
]
