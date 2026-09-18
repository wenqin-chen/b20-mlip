"""Typer CLI ``b20mlip`` (CONTRACTS.md section 3).

Global options come before the sub-command::

    b20mlip --config configs/cluster/tillicum.yaml --set compute.threads=4 --seed 0 report audit

This tier wires ``--version``, ``bench`` and ``report audit`` to their contract signatures; the
implementing modules (``b20mlip.bench``, ``b20mlip.report.audit``) belong to later build rows
and are imported lazily, so until they exist those commands exit 2 with a clear message. Every
other command is registered as a stub that exits 2 ("not implemented in this tier"); nothing
here fakes behaviour. A command exits 0 only when its ``StageResult.status == "ok"``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip import __version__
from b20mlip.config import Settings, load_config
from b20mlip.executors import Executor, get_executor
from b20mlip.models import StageResult
from b20mlip.provenance import run_stage

NOT_IMPLEMENTED_EXIT = 2
_STUB_SETTINGS = {"ignore_unknown_options": True, "allow_extra_args": True}


class ExecutorKind(StrEnum):
    local = "local"
    slurm = "slurm"


@dataclass
class CLIState:
    """Global options, resolved lazily into ``Settings`` and an ``Executor``."""

    config_paths: list[Path] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    seed: int | None = None
    resume: bool = False
    executor: ExecutorKind = ExecutorKind.local
    dry_run: bool = False

    def settings(self) -> Settings:
        return load_config(self.config_paths, self.overrides)

    def make_executor(self, cfg: Settings) -> Executor:
        return get_executor(cfg, self.executor.value)


app = typer.Typer(
    name="b20mlip",
    help="Fine-tune MACE-MPA-0 on QE frames of B20 skyrmion hosts (see SPEC.md / CONTRACTS.md).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"b20mlip {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        list[Path] | None,
        typer.Option(
            "--config", "-c", help="YAML overlay(s), merged in order after configs/default.yaml."
        ),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override key=value (dotted keys), applied last."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed recorded in the manifest.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Reuse the last failed/partial run of this config.")
    ] = False,
    executor: Annotated[
        ExecutorKind, typer.Option("--executor", help="Where units run.")
    ] = ExecutorKind.local,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan only: manifest with status=partial, no outputs.")
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Print version."),
    ] = None,
) -> None:
    ctx.obj = CLIState(
        config_paths=list(config or []),
        overrides=list(set_ or []),
        seed=seed,
        resume=resume,
        executor=executor,
        dry_run=dry_run,
    )


def state_of(ctx: typer.Context) -> CLIState:
    """Global options of this invocation (tier CLIs use this; safe outside a CLI too)."""
    obj = ctx.find_object(CLIState)
    return obj if obj is not None else CLIState()


_state = state_of


def _not_implemented(command: str, module: str | None = None) -> None:
    where = f" (module {module} is a later build row)" if module else ""
    typer.echo(
        f"b20mlip {command}: not implemented in this tier{where}; see CONTRACTS.md section 1.",
        err=True,
    )
    raise typer.Exit(NOT_IMPLEMENTED_EXIT)


def _import_or_exit(module: str, command: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        _not_implemented(command, module)
        raise AssertionError("unreachable") from None  # pragma: no cover


def finish(result: StageResult) -> None:
    """Print the StageResult as JSON and exit 0 only when ``status == "ok"``."""
    typer.echo(
        json.dumps(
            {
                "stage": result.stage,
                "run_id": result.run_id,
                "status": result.status,
                "manifest": result.manifest_path,
                "outputs": [a.path for a in result.outputs],
                "summary": result.summary,
            },
            indent=2,
        )
    )
    raise typer.Exit(0 if result.status == "ok" else 1)


_finish = finish


# --- stub registration -------------------------------------------------------------------------

STUB_COMMANDS: dict[str | None, list[tuple[str, str]]] = {
    "data": [
        ("pull", "Download MPtrj/OMat24/WBM/phononDB sources."),
        ("sample", "Generate round-0 candidate frames."),
        ("filter", "Dedupe, force cap, magnetic-branch filter."),
        ("split", "Group-hash 80/10/10 split and tiers."),
    ],
    "dft": [
        ("converge", "QE convergence scan for one compound."),
        ("prep", "One QE unit dir per frame."),
        ("run", "Run QE units (per-unit resume; SLURM array or local)."),
        ("collect", "Parse QE outputs into labelled frames."),
        ("phonons", "QE phonopy displacement sets."),
        ("e0s", "Isolated-atom QE energies -> E0s_qe.json."),
        ("offsets", "Per-element QE->MP offset map."),
    ],
    "eval": [
        ("errors", "E/F/S errors per tier with bootstrap CIs."),
        ("discovery", "Labelled WBM-1000 sample, paired dF1 vs B0."),
        ("phonons", "Phonon omega-MAE / softening index vs a reference."),
        ("elastic", "C11/C12/C44/B and EOS."),
    ],
    "md": [
        ("ase", "ASE MD (nve|nvt|npt)."),
        ("lammps", "LAMMPS pair_style mace MD (cluster)."),
        ("parity", "20-frame ASE vs LAMMPS parity gate."),
    ],
    "sampling": [
        ("neb", "NEB vacancy-hop barrier."),
        ("umbrella", "Umbrella-sampling windows."),
        ("wham", "MBAR/WHAM free energy from windows."),
    ],
    "active": [("select", "Committee sigma_F frame selection.")],
    "agent": [("run", "Run one agent task."), ("eval", "Score the 12-task agent eval.")],
    "cluster": [
        ("bootstrap", "Discover account/partitions, sync repo, build LAMMPS."),
        ("sync", "Pull cluster results."),
        ("status", "Show cluster jobs."),
    ],
    None: [
        ("train", "Fine-tune (naive|replay|scratch|bootstrap)."),
        ("export", "mace_create_lammps_model."),
        ("screen", "Deterministic gold script for agent tasks."),
    ],
}

GROUP_HELP = {
    "data": "Datasets: pull, sample, filter, split.",
    "dft": "Quantum ESPRESSO stages.",
    "eval": "Evaluation tiers, discovery, phonons, elastic.",
    "md": "Molecular dynamics (ASE, LAMMPS, parity).",
    "sampling": "NEB, umbrella sampling, WHAM.",
    "active": "Active learning.",
    "agent": "Tool-calling agent.",
    "cluster": "Tillicum hand-off.",
    "report": "numbers.json, README, honesty audit.",
}


def _make_stub(command: str) -> Callable[[typer.Context], None]:
    def stub(ctx: typer.Context) -> None:
        _not_implemented(command)

    stub.__doc__ = f"{command}: not implemented in this tier."
    return stub


# Tier packages plug their commands in without editing this file: ``b20mlip.<pkg>.cli`` may
# expose ``register(group: typer.Typer) -> set[str]`` (group commands) and/or
# ``register_toplevel(app: typer.Typer) -> set[str]`` (top-level commands such as ``train``);
# each returns the command names it registered. Whatever is not registered stays a stub.
GROUP_PACKAGES: dict[str, str] = {
    "data": "b20mlip.data.cli",
    "dft": "b20mlip.dft.cli",
    "eval": "b20mlip.evaluate.cli",
    "md": "b20mlip.md.cli",
    "sampling": "b20mlip.sampling.cli",
    "active": "b20mlip.active.cli",
    "agent": "b20mlip.agent.cli",
    "cluster": "b20mlip.cluster.cli",
}
TOPLEVEL_PACKAGES: tuple[str, ...] = ("b20mlip.train.cli", "b20mlip.agent.cli")


def _optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # missing tier or a broken optional dependency
        if exc.name is not None and not name.startswith(exc.name.split(".")[0]):
            raise  # a genuine third-party import error inside an existing tier
        return None


GROUPS: dict[str, typer.Typer] = {}
for _group_name, _help in GROUP_HELP.items():
    GROUPS[_group_name] = typer.Typer(help=_help, no_args_is_help=True)
    app.add_typer(GROUPS[_group_name], name=_group_name)

REGISTERED: dict[str, set[str]] = {}
for _group_name, _module_name in GROUP_PACKAGES.items():
    _module = _optional_module(_module_name)
    if _module is not None and hasattr(_module, "register"):
        REGISTERED[_group_name] = set(_module.register(GROUPS[_group_name]))
_toplevel_registered: set[str] = set()
for _module_name in TOPLEVEL_PACKAGES:
    _module = _optional_module(_module_name)
    if _module is not None and hasattr(_module, "register_toplevel"):
        _toplevel_registered |= set(_module.register_toplevel(app))

for _group, _commands in STUB_COMMANDS.items():
    _target = app if _group is None else GROUPS[_group]
    _done = _toplevel_registered if _group is None else REGISTERED.get(_group, set())
    for _name, _doc in _commands:
        if _name in _done:
            continue
        _full = _name if _group is None else f"{_group} {_name}"
        _target.command(_name, help=f"{_doc} [stub]", context_settings=_STUB_SETTINGS)(
            _make_stub(_full)
        )

report_app = GROUPS["report"]


@report_app.command("build", context_settings=_STUB_SETTINGS)
def report_build(ctx: typer.Context) -> None:
    """Collect numbers.json from manifests and render README. [stub]"""
    _not_implemented("report build")


@report_app.command("audit")
def report_audit(
    ctx: typer.Context,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Also fail on status=partial manifests cited by README."),
    ] = False,
    readme: Annotated[Path, typer.Option("--readme", help="README to audit.")] = Path("README.md"),
) -> None:
    """Honesty gates A1-A11 (CONTRACTS.md section 8); exit 1 with a JSON list of violations."""
    state = _state(ctx)
    audit = _import_or_exit("b20mlip.report.audit", "report audit")
    cfg = state.settings()
    violations: list[str] = audit.run(
        readme=readme,
        numbers=Path(cfg.report.numbers_path),
        runs_dir=Path(cfg.paths.runs_dir),
        strict=strict,
    )
    if violations:
        typer.echo(json.dumps(violations, indent=2))
        raise typer.Exit(1)
    typer.echo("report audit: OK (no violations)")


@app.command()
def bench(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("runs/bench"),
) -> None:
    """Day-1 timing task (SPEC.md section 15): writes bench.json plus its manifest."""
    state = _state(ctx)
    bench_module = _import_or_exit("b20mlip.bench", "bench")
    cfg = state.settings()
    result = run_stage(
        "bench",
        cfg,
        bench_module.run,
        seed=state.seed,
        resume=state.resume,
        executor=state.make_executor(cfg),
        dry_run=state.dry_run,
        out=out,
    )
    _finish(result)


def run() -> None:  # pragma: no cover - console entry point helper
    app()


__all__ = [
    "CLIState",
    "ExecutorKind",
    "GROUPS",
    "NOT_IMPLEMENTED_EXIT",
    "app",
    "finish",
    "run",
    "state_of",
]
