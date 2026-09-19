"""``b20mlip md ase|lammps|parity`` (CONTRACTS.md section 3), plugged into the root CLI group.

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

COMMANDS: frozenset[str] = frozenset({"ase", "lammps", "parity"})


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


def register(group: typer.Typer) -> set[str]:
    """Add the md commands to ``group`` and return their names."""

    @group.command("ase")
    def ase(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        compound: Annotated[str, typer.Option("--compound", help="B20 compound, e.g. MnSi.")],
        ensemble: Annotated[
            str, typer.Option("--ensemble", help="nve | nvt | npt (ASE NPT: isotropic, 0 GPa).")
        ] = "nvt",
        T: Annotated[
            list[float] | None,
            typer.Option("--T", help="Temperature(s) in K; repeat for a(T) (>= 3 -> alpha)."),
        ] = None,
        ps: Annotated[
            float, typer.Option("--ps", help="Length in ps (after equilibration).")
        ] = 40.0,
        natoms: Annotated[
            int | None, typer.Option("--natoms", help="Supercell size (default cfg.md.natoms).")
        ] = None,
        head: Annotated[
            str, typer.Option("--head", help="Model head: Default | pt_head.")
        ] = "Default",
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the input cell.")
        ] = None,
        label: Annotated[
            str | None, typer.Option("--label", help="numbers.json model label (default bracket).")
        ] = None,
        a_exp: Annotated[
            float | None, typer.Option("--a-exp", help="Experimental a (A) for a_dev_pct.")
        ] = None,
    ) -> None:
        """ASE MD with the MACE calculator: trajectory, thermo CSV, VDOS, RDF, a(T), NVE drift."""
        from b20mlip.md import ase_md

        _run(
            ctx, "md.ase", ase_md.stage, model=model, compound=compound, ensemble=ensemble,
            T=list(T) if T else [300.0], ps=ps, natoms=natoms, head=head, structure=structure,
            label=label, a_exp_A=a_exp,
        )  # fmt: skip

    @group.command("lammps")
    def lammps(
        ctx: typer.Context,
        model: Annotated[
            Path, typer.Option("--model", help="MACE .model (needs its -lammps.pt) or -lammps.pt.")
        ],
        compound: Annotated[str, typer.Option("--compound", help="B20 compound, e.g. MnSi.")],
        T: Annotated[float, typer.Option("--T", help="Temperature in K.")] = 300.0,
        ps: Annotated[float, typer.Option("--ps", help="Production length in ps.")] = 100.0,
        natoms: Annotated[int | None, typer.Option("--natoms", help="Supercell size.")] = None,
        ensemble: Annotated[str, typer.Option("--ensemble", help="npt | nvt | nve.")] = "npt",
        head: Annotated[str, typer.Option("--head", help="Head the export used.")] = "Default",
        structure: Annotated[
            Path | None, typer.Option("--structure", help="extxyz with the input cell.")
        ] = None,
        label: Annotated[str | None, typer.Option("--label", help="numbers.json label.")] = None,
        wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job.")] = True,
        gpu: Annotated[
            bool | None,
            typer.Option(
                "--gpu/--cpu", help="Kokkos GPU flags (default: SLURM with a GPU partition)."
            ),
        ] = None,
        a_exp: Annotated[float | None, typer.Option("--a-exp", help="Experimental a (A).")] = None,
    ) -> None:
        """LAMMPS `pair_style mace` MD through the executor (cluster GPU); parses log.lammps."""
        from b20mlip.md import lammps as mod

        _run(
            ctx, "md.lammps", mod.run, model=model, compound=compound, T=T, ps=ps, natoms=natoms,
            ensemble=ensemble, structure=structure, head=head, label=label, wait=wait, gpu=gpu,
            a_exp_A=a_exp,
        )  # fmt: skip

    @group.command("parity")
    def parity(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model (with its -lammps.pt).")],
        frames: Annotated[Path, typer.Option("--frames", help="extxyz with the 20 gate frames.")],
        lammps_json: Annotated[
            Path | None,
            typer.Option("--lammps-json", help="{frame_id: {energy, forces}} from LAMMPS."),
        ] = None,
        head: Annotated[str, typer.Option("--head", help="Model head.")] = "Default",
        tol_f: Annotated[float, typer.Option("--tol-f", help="max|dF| gate (eV/A).")] = 1e-3,
        tol_e: Annotated[float, typer.Option("--tol-e", help="|dE| gate (eV/atom).")] = 1e-4,
        wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job.")] = True,
        gpu: Annotated[
            bool | None, typer.Option("--gpu/--cpu", help="Kokkos GPU flags for the job.")
        ] = None,
        label: Annotated[str | None, typer.Option("--label", help="numbers.json label.")] = None,
    ) -> None:
        """20-frame ASE-vs-LAMMPS parity gate (A6); stages the LAMMPS single points if needed."""
        from b20mlip.md import parity as mod

        _run(
            ctx, "md.parity", mod.run, model=model, frames=frames, lammps_json=lammps_json,
            head=head, tol_f=tol_f, tol_e=tol_e, wait=wait, gpu=gpu, label=label,
        )  # fmt: skip

    return set(COMMANDS)


__all__ = ["COMMANDS", "register"]
