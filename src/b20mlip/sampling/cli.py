"""``b20mlip sampling neb|umbrella|wham`` (CONTRACTS.md section 3), plugged into the root CLI.

The root ``b20mlip.cli`` imports this module at import time and calls ``register(group)``;
command bodies import ``finish``/``state_of`` lazily to avoid the cycle. Every command runs its
stage under ``provenance.run_stage`` and exits 0 only when the ``StageResult`` is ``ok``.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip.models import StageName
from b20mlip.provenance import run_stage

COMMANDS: frozenset[str] = frozenset({"neb", "umbrella", "wham"})


class MethodOpt(StrEnum):
    mbar = "mbar"
    wham = "wham"


class InitOpt(StrEnum):
    sequential = "sequential"
    interp = "interp"


def _run(ctx: typer.Context, stage: StageName, fn: Callable[..., Any], **kw: Any) -> None:
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
    """Add the sampling commands to ``group`` and return their names."""

    @group.command("neb")
    def neb(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        images: Annotated[int, typer.Option("--images", help="Images in the band.")] = 7,
        compound: Annotated[
            str | None,
            typer.Option("--compound", help="B20 compound for the automatic vacancy hop."),
        ] = None,
        initial: Annotated[
            Path | None, typer.Option("--initial", help="Initial state (extxyz).")
        ] = None,
        final: Annotated[Path | None, typer.Option("--final", help="Final state (extxyz).")] = None,
        species: Annotated[
            str, typer.Option("--species", help="Vacancy sublattice: element, TM or X.")
        ] = "Si",
        supercell: Annotated[
            tuple[int, int, int], typer.Option("--supercell", help="Supercell repetitions.")
        ] = (2, 2, 2),
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the reference cell.")
        ] = None,
        label: Annotated[
            str | None, typer.Option("--label", help="Model label in numbers keys (B0..B4).")
        ] = None,
        head: Annotated[str | None, typer.Option("--head", help="Model head.")] = None,
        fmax: Annotated[
            float | None, typer.Option("--fmax", help="Force criterion (default eval.fmax).")
        ] = None,
        relax_steps: Annotated[
            int | None, typer.Option("--relax-steps", help="Endpoint FIRE steps (default 500).")
        ] = None,
        neb_steps: Annotated[
            int | None, typer.Option("--neb-steps", help="Band FIRE steps (default 300).")
        ] = None,
    ) -> None:
        """Climbing-image NEB barrier of a vacancy hop (writes neb.json, the band, numbers.json)."""
        from b20mlip.sampling import neb as mod

        _run(
            ctx, "sampling.neb", mod.run, model=model, compound=compound, initial=initial,
            final=final, images=images, species=species, supercell=tuple(supercell),
            structure=structure, label=label, head=head, fmax=fmax, relax_steps=relax_steps,
            neb_steps=neb_steps,
        )  # fmt: skip

    @group.command("umbrella")
    def umbrella(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        windows: Annotated[
            int | None, typer.Option("--windows", help="Windows (default sampling.windows).")
        ] = None,
        ps: Annotated[
            float | None,
            typer.Option("--ps", help="ps per window (default sampling.ps_per_window)."),
        ] = None,
        k: Annotated[
            float | None, typer.Option("--k", help="Bias eV/A^2 (default sampling.k_eVA2).")
        ] = None,
        T: Annotated[float | None, typer.Option("--T", help="Kelvin (default sampling.T).")] = None,
        compound: Annotated[str, typer.Option("--compound", help="B20 compound.")] = "FeSi",
        species: Annotated[
            str, typer.Option("--species", help="Vacancy sublattice: element, TM or X.")
        ] = "Si",
        supercell: Annotated[
            tuple[int, int, int], typer.Option("--supercell", help="Supercell repetitions.")
        ] = (2, 2, 2),
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the reference cell.")
        ] = None,
        init: Annotated[
            InitOpt, typer.Option("--init", help="Window start: previous frame or interpolation.")
        ] = InitOpt.sequential,
        timestep_fs: Annotated[
            float | None, typer.Option("--timestep-fs", help="Default md.timestep_fs.")
        ] = None,
        label: Annotated[
            str | None, typer.Option("--label", help="Model label in numbers keys (B0..B4).")
        ] = None,
        head: Annotated[str | None, typer.Option("--head", help="Model head.")] = None,
    ) -> None:
        """Langevin umbrella windows along the vacancy-hop CV (resumable with --resume)."""
        from b20mlip.sampling import umbrella as mod

        _run(
            ctx, "sampling.umbrella", mod.run, model=model, compound=compound, windows=windows,
            ps=ps, k=k, T=T, species=species, supercell=tuple(supercell), structure=structure,
            init=init.value, timestep_fs=timestep_fs, label=label, head=head,
        )  # fmt: skip

    @group.command("wham")
    def wham(
        ctx: typer.Context,
        run: Annotated[Path, typer.Option("--run", help="Umbrella run dir (windows.json).")],
        method: Annotated[
            MethodOpt, typer.Option("--method", help="Primary estimator; the other cross-checks.")
        ] = MethodOpt.mbar,
        blocks: Annotated[int, typer.Option("--blocks", help="Blocks for the error.")] = 5,
        bins: Annotated[
            int | None, typer.Option("--bins", help="Histogram bins (default max(50, 5 K)).")
        ] = None,
        min_overlap: Annotated[
            float, typer.Option("--min-overlap", help="Neighbour histogram overlap gate.")
        ] = 0.05,
    ) -> None:
        """MBAR/WHAM free-energy profile, dF with block error (pmf.json, numbers.json)."""
        from b20mlip.sampling import wham as mod

        _run(
            ctx, "sampling.wham", mod.run, run_dir=run, method=method.value, n_blocks=blocks,
            n_bins=bins, min_overlap=min_overlap,
        )  # fmt: skip

    return set(COMMANDS)


__all__ = ["COMMANDS", "InitOpt", "MethodOpt", "register"]
