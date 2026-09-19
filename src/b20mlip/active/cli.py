"""``b20mlip active select`` (CONTRACTS.md section 3), plugged into the root CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip.provenance import run_stage

COMMANDS: frozenset[str] = frozenset({"select"})


def _run(ctx: typer.Context, stage: Any, fn: Any, **kw: Any) -> None:
    from b20mlip.cli import finish, state_of  # noqa: PLC0415 - lazy: the root CLI imports us

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
    """Add ``select`` to the ``active`` group; returns the names registered."""

    @group.command("select")
    def select(
        ctx: typer.Context,
        models: Annotated[
            str,
            typer.Option(
                "--models", help="Comma list of >= 2 committee .model files (e.g. three seeds)."
            ),
        ],
        frames: Annotated[Path, typer.Option("--frames", help="Candidate frames extxyz.")],
        n: Annotated[int, typer.Option("--n", help="Maximum number of frames to select.")] = 100,
        out: Annotated[Path, typer.Option("--out", help="Selected candidates extxyz.")] = Path(
            "data/frames/candidates_r1.extxyz"
        ),
        sigma_min: Annotated[
            float, typer.Option("--sigma-min", help="Skip frames with sigma_F below this (eV/Å).")
        ] = 0.0,
        per_group_max: Annotated[
            int | None,
            typer.Option("--per-group-max", help="At most this many frames per group_id."),
        ] = None,
        head: Annotated[
            str, typer.Option("--head", help="MACE head used by every member.")
        ] = "Default",
    ) -> None:
        """Committee sigma_F selection: highest-uncertainty frames -> candidates + numbers.json."""
        from b20mlip.active import committee as mod  # noqa: PLC0415

        _run(
            ctx, "active.select", mod.run, models=models, frames_path=frames, n=n, out=out,
            sigma_min=sigma_min, per_group_max=per_group_max, head=head,
        )  # fmt: skip

    return set(COMMANDS)


__all__ = ["COMMANDS", "register"]
