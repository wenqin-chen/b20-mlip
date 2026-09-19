"""``b20mlip agent run|eval`` and the top-level ``b20mlip screen`` (CONTRACTS.md section 3).

The root CLI imports this module at start-up, so it stays cheap: the agent modules (and the
``anthropic`` SDK) are imported inside the command bodies. ``--backend anthropic`` without
``ANTHROPIC_API_KEY`` exits 2 with a clear message and never constructs a client.
"""

from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip.config import Settings
from b20mlip.models import StageName
from b20mlip.provenance import run_stage

COMMANDS: frozenset[str] = frozenset({"run", "eval"})
TOPLEVEL_COMMANDS: frozenset[str] = frozenset({"screen"})
BACKEND_EXIT = 2


class BackendOpt(StrEnum):
    anthropic = "anthropic"
    mock = "mock"
    scripted = "scripted"


def _labelled(items: list[str] | None, what: str) -> dict[str, Path]:
    """``LABEL=PATH`` options -> ``{label: path}`` (a bare path is labelled by its stem)."""
    out: dict[str, Path] = {}
    for item in items or []:
        label, sep, raw = item.partition("=")
        path = Path(raw if sep else item)
        name = label if sep else path.stem
        if not name:
            raise typer.BadParameter(f"--{what} expects LABEL=PATH, got {item!r}")
        out[name] = path
    return out


def _tool_kwargs(
    model: Path | None,
    head: str,
    committee: str | None,
    structure: Path | None,
    refs: Path | None,
    frames: list[str] | None,
    splits: list[str] | None,
) -> dict[str, Any]:
    kw: dict[str, Any] = {"head": head}
    if model is not None:
        kw["model"] = model
    if committee:
        kw["committee"] = [Path(p.strip()) for p in committee.split(",") if p.strip()]
    if structure is not None:
        kw["structure_path"] = structure
    if refs is not None:
        kw["refs_dir"] = refs
    if frames:
        kw["frames"] = _labelled(frames, "frames")
    if splits:
        kw["splits"] = _labelled(splits, "split")
    return kw


def _backend(
    name: BackendOpt,
    cfg: Settings,
    *,
    model_id: str | None,
    trace: Path | None,
    trace_dir: Path | None,
    replay_results: bool,
    plans: Any,
    max_tokens: int | None,
) -> Any:
    from b20mlip.agent.backends import BackendUnavailable, make_backend

    trace_path = trace if trace is not None else (trace_dir or Path(cfg.agent.trace_dir))
    try:
        return make_backend(
            name.value,
            model_id=model_id or cfg.agent.model_id,
            trace_path=trace_path if name == BackendOpt.mock else None,
            replay_results=replay_results,
            plans=plans,
            max_tokens=max_tokens or cfg.agent.max_tokens,
        )
    except BackendUnavailable as exc:
        typer.echo(f"b20mlip agent: {exc}", err=True)
        raise typer.Exit(BACKEND_EXIT) from None


def _run(ctx: typer.Context, stage: StageName, fn: Any, **kw: Any) -> None:
    from b20mlip.cli import finish, state_of  # lazy: the root CLI imports this module

    state = state_of(ctx)
    cfg = state.settings()
    result = run_stage(
        stage,
        cfg,
        fn,
        seed=state.seed,
        resume=state.resume,
        executor=state.make_executor(cfg),
        dry_run=state.dry_run,
        **kw,
    )
    finish(result)


def register(group: typer.Typer) -> set[str]:
    """Add ``run`` and ``eval`` to the ``agent`` group; returns the names registered."""

    @group.command("run")
    def run_cmd(
        ctx: typer.Context,
        task: Annotated[
            str | None, typer.Option("--task", help="TASKS.jsonl:ID (a task of a task file).")
        ] = None,
        prompt: Annotated[
            str | None, typer.Option("--prompt", help="An ad-hoc task prompt.")
        ] = None,
        backend: Annotated[
            BackendOpt | None, typer.Option("--backend", help="anthropic | mock | scripted.")
        ] = None,
        budget: Annotated[
            str | None,
            typer.Option("--budget", help="Preset (default|ci|tight|generous) or a JSON object."),
        ] = None,
        approve_cluster: Annotated[
            bool, typer.Option("--approve-cluster", help="Allow submit_dft to submit.")
        ] = False,
        model: Annotated[
            Path | None, typer.Option("--model", help="MLIP .model the agent drives.")
        ] = None,
        head: Annotated[str, typer.Option("--head", help="Model head.")] = "Default",
        committee: Annotated[
            str | None, typer.Option("--committee", help="Comma list of committee .model files.")
        ] = None,
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the reference cells.")
        ] = None,
        refs: Annotated[
            Path | None, typer.Option("--refs", help="Directory of phonons_<compound>_<ref>.json.")
        ] = None,
        frames: Annotated[
            list[str] | None, typer.Option("--frames", help="LABEL=PATH dataset (repeatable).")
        ] = None,
        splits: Annotated[
            list[str] | None, typer.Option("--split", help="LABEL=PATH split (repeatable).")
        ] = None,
        model_id: Annotated[
            str | None, typer.Option("--model-id", help="Claude model id (anthropic backend).")
        ] = None,
        max_tokens: Annotated[
            int | None, typer.Option("--max-tokens", help="max_tokens per turn (anthropic).")
        ] = None,
        trace: Annotated[
            Path | None, typer.Option("--trace", help="Trace to replay (mock).")
        ] = None,
        trace_dir: Annotated[
            Path | None,
            typer.Option("--trace-dir", help="Directory of <task_id>.jsonl traces (mock)."),
        ] = None,
        replay_results: Annotated[
            bool,
            typer.Option("--replay-results", help="Mock: return stored results, compute nothing."),
        ] = False,
        record: Annotated[
            bool,
            typer.Option("--record", help="Copy the trace to agent.trace_dir/<task_id>.jsonl."),
        ] = False,
        gold: Annotated[Path | None, typer.Option("--gold", help="Gold file for scoring.")] = None,
        injected_failure: Annotated[
            str | None,
            typer.Option(
                "--injected-failure",
                help="tool_error|bad_structure|budget_exhausted|sum_rule (prompt tasks).",
            ),
        ] = None,
    ) -> None:
        """Run one agent task under the guard and the trace; writes runs/agent.run/<run_id>/."""
        from b20mlip.agent import runner, tasks
        from b20mlip.cli import state_of

        if (task is None) == (prompt is None):
            raise typer.BadParameter("give exactly one of --task TASKS.jsonl:ID or --prompt TEXT")
        state = state_of(ctx)
        cfg = state.settings()
        if task is not None:
            path, task_id = tasks.parse_task_ref(task)
            specs = tasks.load_tasks(path)
            if task_id is None:
                raise typer.BadParameter("--task needs TASKS.jsonl:ID")
            spec = tasks.select_task(specs, task_id)
            if injected_failure:
                spec.task = spec.task.model_copy(update={"injected_failure": injected_failure})
        else:
            spec = tasks.task_from_prompt(str(prompt), injected_failure=injected_failure)
        if budget is not None or approve_cluster:
            new_budget = (
                tasks.budget_from_spec(budget, approve_cluster=approve_cluster or None)
                if budget is not None
                else spec.task.budget.model_copy(update={"approve_cluster": True})
            )
            spec.task = spec.task.model_copy(update={"budget": new_budget})
        if gold is not None:
            spec = tasks.apply_gold([spec], tasks.load_gold(gold))[0]
        name = backend or BackendOpt(cfg.agent.backend)
        be = _backend(
            name, cfg, model_id=model_id, trace=trace, trace_dir=trace_dir,
            replay_results=replay_results, plans=tasks.plans_of([spec]), max_tokens=max_tokens,
        )  # fmt: skip
        kw = _tool_kwargs(model, head, committee, structure, refs, frames, splits)

        def stage(cfg_: Settings, run_ctx: Any, **inner: Any) -> dict[str, Any]:
            summary = runner.stage(cfg_, run_ctx, task=spec.task, backend=be, **inner)
            if record and not run_ctx.dry_run:
                src = run_ctx.out_dir / runner.TRACE_FILE
                dst = Path(cfg_.agent.trace_dir) / f"{spec.task.task_id}.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                summary["recorded_trace"] = str(dst)
            return summary

        _run(ctx, "agent.run", stage, **kw)

    @group.command("eval")
    def eval_cmd(
        ctx: typer.Context,
        tasks_path: Annotated[
            Path | None, typer.Option("--tasks", help="Task file (default agent.tasks_path).")
        ] = None,
        backend: Annotated[
            BackendOpt | None, typer.Option("--backend", help="mock | anthropic | scripted.")
        ] = None,
        gold: Annotated[
            Path | None, typer.Option("--gold", help="Gold file from `screen`.")
        ] = None,
        limit: Annotated[
            int | None, typer.Option("--limit", help="Only the first N tasks.")
        ] = None,
        model: Annotated[
            Path | None, typer.Option("--model", help="MLIP .model the agent drives.")
        ] = None,
        head: Annotated[str, typer.Option("--head", help="Model head.")] = "Default",
        committee: Annotated[
            str | None, typer.Option("--committee", help="Comma list of .model files.")
        ] = None,
        structure: Annotated[
            Path | None, typer.Option("--structure", help="Reference cells extxyz.")
        ] = None,
        refs: Annotated[
            Path | None, typer.Option("--refs", help="Phonon references directory.")
        ] = None,
        frames: Annotated[
            list[str] | None, typer.Option("--frames", help="LABEL=PATH dataset.")
        ] = None,
        splits: Annotated[
            list[str] | None, typer.Option("--split", help="LABEL=PATH split.")
        ] = None,
        model_id: Annotated[str | None, typer.Option("--model-id", help="Claude model id.")] = None,
        max_tokens: Annotated[
            int | None, typer.Option("--max-tokens", help="max_tokens per turn.")
        ] = None,
        trace_dir: Annotated[
            Path | None,
            typer.Option("--trace-dir", help="Directory of <task_id>.jsonl traces (mock)."),
        ] = None,
        replay_results: Annotated[
            bool, typer.Option("--replay-results", help="Mock: return stored results only.")
        ] = False,
        bootstrap_n: Annotated[
            int | None, typer.Option("--bootstrap-n", help="Resamples for the CI over tasks.")
        ] = None,
    ) -> None:
        """Score the task file (accuracy, invalid calls, DAG order, provenance, recovery, cost)."""
        from b20mlip.agent import eval as eval_mod
        from b20mlip.agent import tasks
        from b20mlip.cli import state_of

        state = state_of(ctx)
        cfg = state.settings()
        specs = tasks.load_tasks(tasks_path or cfg.agent.tasks_path)
        name = backend or BackendOpt(cfg.agent.backend)
        be = _backend(
            name, cfg, model_id=model_id, trace=None, trace_dir=trace_dir,
            replay_results=replay_results, plans=tasks.plans_of(specs), max_tokens=max_tokens,
        )  # fmt: skip
        gold_data = tasks.load_gold(gold) if gold is not None else None
        kw = _tool_kwargs(model, head, committee, structure, refs, frames, splits)
        _run(
            ctx, "agent.eval", eval_mod.run, tasks=specs, backend=be, gold=gold_data, limit=limit,
            n_boot=bootstrap_n, **kw,
        )  # fmt: skip

    return set(COMMANDS)


def register_toplevel(app: typer.Typer) -> set[str]:
    @app.command("screen")
    def screen_cmd(
        ctx: typer.Context,
        compounds: Annotated[
            str | None,
            typer.Option("--compounds", help="Comma list; tasks outside it are skipped."),
        ] = None,
        model: Annotated[
            Path | None, typer.Option("--model", help="MLIP .model the gold uses.")
        ] = None,
        tasks_path: Annotated[
            Path | None, typer.Option("--tasks", help="Task file (default agent.tasks_path).")
        ] = None,
        out: Annotated[
            Path | None, typer.Option("--out", help="Gold file (default evals/gold_<label>.json).")
        ] = None,
        limit: Annotated[
            int | None, typer.Option("--limit", help="Only the first N tasks.")
        ] = None,
        head: Annotated[str, typer.Option("--head", help="Model head.")] = "Default",
        committee: Annotated[
            str | None, typer.Option("--committee", help="Comma list of .model files.")
        ] = None,
        structure: Annotated[
            Path | None, typer.Option("--structure", help="Reference cells extxyz.")
        ] = None,
        refs: Annotated[
            Path | None, typer.Option("--refs", help="Phonon references directory.")
        ] = None,
        frames: Annotated[
            list[str] | None, typer.Option("--frames", help="LABEL=PATH dataset.")
        ] = None,
        splits: Annotated[
            list[str] | None, typer.Option("--split", help="LABEL=PATH split.")
        ] = None,
    ) -> None:
        """Deterministic gold script: scripted plans for every task -> gold for `agent eval`."""
        from b20mlip.agent import screen, tasks
        from b20mlip.cli import state_of
        from b20mlip.md.common import model_provenance

        state = state_of(ctx)
        cfg = state.settings()
        specs = tasks.load_tasks(tasks_path or cfg.agent.tasks_path)
        kw = _tool_kwargs(model, head, committee, structure, refs, frames, splits)
        if out is None and model is not None:
            label = model_provenance(model)["label"]
            out = Path(cfg.agent.tasks_path).parent / f"gold_{label}.json"
        wanted = [c.strip() for c in compounds.split(",")] if compounds else None
        _run(
            ctx, "agent.eval", screen.run, tasks=specs, compounds=wanted, out=out, limit=limit, **kw
        )

    return set(TOPLEVEL_COMMANDS)


__all__ = [
    "BACKEND_EXIT",
    "COMMANDS",
    "TOPLEVEL_COMMANDS",
    "BackendOpt",
    "register",
    "register_toplevel",
]
