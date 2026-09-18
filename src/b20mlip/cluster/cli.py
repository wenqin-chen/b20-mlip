"""``b20mlip cluster bootstrap | sync | status`` — plugin hook for the root CLI.

The root ``b20mlip.cli`` imports this module at import time and calls :func:`register`; the
command bodies import ``finish`` / ``state_of`` lazily to avoid the import cycle. The real
runner (``subprocess_runner``: ssh/rsync over the ControlMaster socket) comes from
:data:`RUNNER_FACTORY`, which tests replace with a fake.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from b20mlip.cluster import discover
from b20mlip.cluster import status as status_mod
from b20mlip.cluster import sync as sync_mod
from b20mlip.executors import ClusterUnreachable, Runner, subprocess_runner
from b20mlip.models import StageResult
from b20mlip.provenance import read_manifest, run_stage


def default_runner() -> Runner:
    return subprocess_runner


RUNNER_FACTORY: Callable[[], Runner] = default_runner


def unreachable_hint(result: StageResult) -> None:
    """Print the MFA instruction when a failed stage recorded a dead socket."""
    if result.status != "failed":
        return
    try:
        extras = read_manifest(result.manifest_path).extras
    except (OSError, ValueError):
        return
    instruction = extras.get("mfa_instruction")
    if instruction and "ClusterUnreachable" in str(extras.get("error", "")):
        typer.echo(f"cluster unreachable: {instruction}", err=True)


def register(group: typer.Typer) -> set[str]:
    @group.command("bootstrap")
    def bootstrap(
        ctx: typer.Context,
        build_lammps: Annotated[
            bool,
            typer.Option(
                "--build-lammps",
                help="Submit templates/slurm/build_lammps.sbatch.j2 (ACEsuit/lammps mace).",
            ),
        ] = False,
        install_qe: Annotated[
            bool,
            typer.Option(
                "--install-qe",
                help="Run the micromamba QE fallback when no QE module exists "
                "(default: only planned; same as --set cluster.install_qe=true).",
            ),
        ] = False,
        skip: Annotated[
            list[str] | None,
            typer.Option("--skip", help="Skip a step (repeatable): " + ", ".join(discover.STEPS)),
        ] = None,
        partition: Annotated[
            str | None,
            typer.Option(
                "--partition", help="Partition for the LAMMPS build job (else discovered)."
            ),
        ] = None,
        config_out: Annotated[
            Path | None,
            typer.Option(
                "--config-out", help="YAML to write (default configs/cluster/<alias>.yaml)."
            ),
        ] = None,
    ) -> None:
        """Discover account/partitions/QOS/scratch/modules, sync the repo, check QE (SPEC 8)."""
        from b20mlip.cli import finish, state_of

        state = state_of(ctx)
        cfg = state.settings()
        result = run_stage(
            "cluster.bootstrap",
            cfg,
            discover.bootstrap,
            seed=state.seed,
            resume=state.resume,
            dry_run=state.dry_run,
            runner=RUNNER_FACTORY(),
            build_lammps=build_lammps,
            install_qe=True if install_qe else None,
            skip=tuple(skip or ()),
            partition=partition,
            yaml_path=config_out,
        )
        unreachable_hint(result)
        finish(result)

    @group.command("sync")
    def sync(
        ctx: typer.Context,
        jobs: Annotated[
            str | None,
            typer.Option(
                "--jobs",
                help="Comma-separated job names (default: every <scratch>/b20-mlip/jobs/*).",
            ),
        ] = None,
    ) -> None:
        """Pull markers, logs and results of cluster jobs into runs/slurm/<job>/ (idempotent)."""
        from b20mlip.cli import finish, state_of

        state = state_of(ctx)
        cfg = state.settings()
        result = run_stage(
            "cluster.sync",
            cfg,
            sync_mod.pull,
            seed=state.seed,
            resume=state.resume,
            dry_run=state.dry_run,
            runner=RUNNER_FACTORY(),
            jobs=[j.strip() for j in jobs.split(",") if j.strip()] if jobs else None,
        )
        unreachable_hint(result)
        finish(result)

    @group.command("status")
    def status(
        ctx: typer.Context,
        as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    ) -> None:
        """squeue for the user plus local unit-marker counts (no manifest)."""
        from b20mlip.cli import state_of

        cfg = state_of(ctx).settings()
        try:
            report = status_mod.show(cfg, runner=RUNNER_FACTORY())
        except ClusterUnreachable as exc:
            typer.echo(f"cluster status: {exc}", err=True)
            raise typer.Exit(1) from None
        if as_json:
            typer.echo(json.dumps({k: v for k, v in report.items() if k != "commands"}, indent=2))
        else:
            typer.echo(status_mod.format_table(report))

    return {"bootstrap", "sync", "status"}


__all__ = ["RUNNER_FACTORY", "default_runner", "register", "unreachable_hint"]
