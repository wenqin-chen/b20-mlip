"""``b20mlip dft ...`` commands (CONTRACTS.md section 3), plugged into the root CLI group.

The root ``b20mlip.cli`` imports this module at import time and calls ``register(group)``; the
command bodies import ``finish``/``state_of`` lazily to avoid the cycle. Every command runs its
stage under ``provenance.run_stage`` and exits 0 only when the ``StageResult`` is ``ok``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip.models import StageName
from b20mlip.provenance import run_stage

COMMANDS: frozenset[str] = frozenset(
    {"converge", "prep", "run", "collect", "phonons", "e0s", "offsets"}
)


def _run(ctx: typer.Context, stage: StageName, fn: Callable[..., Any], **kw: Any) -> None:
    from b20mlip.cli import finish, state_of  # lazy: the root CLI imports this module

    state = state_of(ctx)
    cfg = state.settings()
    executor = state.make_executor(cfg)
    result = run_stage(
        stage,
        cfg,
        fn,
        seed=state.seed,
        resume=state.resume,
        executor=executor,
        dry_run=state.dry_run,
        **kw,
    )
    finish(result)


def _elements(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [t.strip() for t in text.split(",") if t.strip()]


def register(group: typer.Typer) -> set[str]:
    """Add the dft commands to ``group`` and return their names."""

    @group.command("converge")
    def converge(
        ctx: typer.Context,
        compound: Annotated[str, typer.Option("--compound", help="B20 compound, e.g. MnSi.")],
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the reference cell.")
        ] = None,
        units: Annotated[
            Path | None, typer.Option("--units", help="Scan root (default dft/converge_<C>).")
        ] = None,
        out: Annotated[
            Path | None,
            typer.Option("--out", help="Result JSON (default configs/dft/qe_<C>_converged.json)."),
        ] = None,
        collect: Annotated[
            bool, typer.Option("--collect", help="Only collect/analyse existing outputs.")
        ] = False,
        wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job.")] = True,
    ) -> None:
        """Cutoff x k-spacing scan (24 units); cheapest point within cfg.dft.thresholds."""
        from b20mlip.dft import converge as mod

        _run(
            ctx, "dft.converge", mod.run, compound=compound, structure=structure, units=units,
            out=out, wait=wait, collect_only=collect,
        )  # fmt: skip

    @group.command("prep")
    def prep(
        ctx: typer.Context,
        frames: Annotated[Path, typer.Option("--frames", help="Input frames (extxyz).")],
        out: Annotated[Path, typer.Option("--out", help="Units root, one dir per frame.")],
    ) -> None:
        """One QE unit directory (pw.in, unit.json) per frame."""
        from b20mlip.dft import stages

        _run(ctx, "dft.prep", stages.prep, frames=frames, out=out)

    @group.command("run")
    def run(
        ctx: typer.Context,
        units: Annotated[Path, typer.Option("--units", help="Units root from `dft prep`.")],
        limit: Annotated[
            int | None, typer.Option("--limit", help="Run at most N pending units.")
        ] = None,
        wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job.")] = True,
        template: Annotated[
            str, typer.Option("--template", help="SLURM template (qe_array|qe_phonons).")
        ] = "qe_array",
        parallel: Annotated[
            int, typer.Option("--parallel", help="Concurrent units (local executor).")
        ] = 1,
    ) -> None:
        """Run pending QE units (per-unit resume; SLURM array or local)."""
        from b20mlip.dft import stages

        _run(
            ctx, "dft.run", stages.run, units=units, limit=limit, wait=wait, template=template,
            parallel=parallel,
        )  # fmt: skip

    @group.command("collect")
    def collect(
        ctx: typer.Context,
        units: Annotated[Path, typer.Option("--units", help="Units root.")],
        out: Annotated[Path, typer.Option("--out", help="Labelled frames (extxyz).")],
    ) -> None:
        """Parse pw.out of every unit into QE-labelled frames."""
        from b20mlip.dft import stages

        _run(ctx, "dft.collect", stages.collect, units=units, out=out)

    @group.command("phonons")
    def phonons(
        ctx: typer.Context,
        compound: Annotated[str, typer.Option("--compound", help="B20 compound, e.g. FeSi.")],
        supercell: Annotated[
            tuple[int, int, int], typer.Option("--supercell", help="Supercell repetitions.")
        ] = (2, 2, 2),
        distance: Annotated[
            float, typer.Option("--distance", help="Displacement amplitude (A).")
        ] = 0.03,
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the reference cell.")
        ] = None,
        units: Annotated[
            Path | None, typer.Option("--units", help="Units root (default dft/phonons_<C>).")
        ] = None,
        out: Annotated[
            Path | None, typer.Option("--out", help="force_sets.json destination.")
        ] = None,
        collect: Annotated[
            bool, typer.Option("--collect", help="Only collect existing outputs.")
        ] = False,
        wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job.")] = True,
    ) -> None:
        """phonopy displacement set -> QE units -> force_sets.json."""
        from b20mlip.dft import phonons as mod

        _run(
            ctx, "dft.phonons", mod.run, compound=compound, supercell=tuple(supercell),
            distance=distance, structure=structure, units=units, out=out, wait=wait,
            collect_only=collect,
        )  # fmt: skip

    @group.command("e0s")
    def e0s(
        ctx: typer.Context,
        out: Annotated[Path, typer.Option("--out", help="MACE --E0s JSON.")] = Path(
            "configs/dft/E0s_qe.json"
        ),
        elements: Annotated[
            str | None, typer.Option("--elements", help="Comma list (default: from compounds).")
        ] = None,
        units: Annotated[
            Path | None, typer.Option("--units", help="Units root (default dft/e0s).")
        ] = None,
        wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job.")] = True,
    ) -> None:
        """Isolated-atom QE energies -> E0s_qe.json."""
        from b20mlip.dft import e0s as mod

        _run(ctx, "dft.e0s", mod.run, out=out, elements=_elements(elements), units=units, wait=wait)

    @group.command("offsets")
    def offsets(
        ctx: typer.Context,
        qe: Annotated[Path, typer.Option("--qe", help="QE-labelled frames (extxyz).")],
        mp: Annotated[Path, typer.Option("--mp", help="MPtrj frames (extxyz).")],
        out: Annotated[Path, typer.Option("--out", help="offsets.json.")] = Path(
            "configs/dft/offsets.json"
        ),
    ) -> None:
        """Per-element QE -> MP energy offsets and the residual gate."""
        from b20mlip.dft import offsets as mod

        _run(ctx, "dft.offsets", mod.run, qe=qe, mp=mp, out=out)

    return set(COMMANDS)


__all__ = ["COMMANDS", "register"]
