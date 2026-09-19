"""``b20mlip data`` commands (CONTRACTS.md section 3), plugged into the root CLI.

The root CLI imports this module at import time and calls :func:`register`; the command
bodies import ``finish``/``state_of`` lazily to avoid the circular import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from b20mlip.provenance import run_stage

COMMANDS: frozenset[str] = frozenset({"pull", "sample", "filter", "split"})


def _split_csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [t.strip() for t in text.split(",") if t.strip()]


def pull(
    ctx: typer.Context,
    sources: Annotated[
        str, typer.Option("--sources", help="Comma list of mptrj,omat24,wbm,phonondb (or all).")
    ] = "mptrj,omat24,wbm,phonondb",
    extract: Annotated[
        bool,
        typer.Option(
            "--extract/--no-extract",
            help="After verifying the raw files, run the local extractions: MPtrj family/B20 "
            "frames, OMat24 in-family frames, WBM sample (data/frames, data/wbm).",
        ),
    ] = False,
    delete_raw: Annotated[
        bool,
        typer.Option(
            "--delete-raw/--keep-raw",
            help="Delete the OMat24 raw tarball/shards after a successful filter (never default).",
        ),
    ] = False,
    force_cap: Annotated[
        float | None, typer.Option("--force-cap", help="OMat24 max ||F|| cap (eV/Å).")
    ] = None,
) -> None:
    """Download or verify MPtrj/OMat24/WBM/phononDB raw sources; write sources.json."""
    from b20mlip.cli import finish, state_of
    from b20mlip.data import pull as pull_module

    state = state_of(ctx)
    cfg = state.settings()
    result = run_stage(
        "data.pull",
        cfg,
        pull_module.fetch,
        seed=state.seed,
        resume=state.resume,
        executor=state.make_executor(cfg),
        dry_run=state.dry_run,
        sources=sources,
        extract=extract,
        delete_raw=delete_raw,
        force_cap=force_cap,
    )
    finish(result)


def sample(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option("--out", help="Output extxyz.")] = Path(
        "data/frames/candidates_r0.extxyz"
    ),
    parents: Annotated[
        Path | None,
        typer.Option("--parents", help="Relaxed parents (default data/frames/mptrj_b20.extxyz)."),
    ] = None,
    md: Annotated[
        bool,
        typer.Option(
            "--md/--no-md", help="Also run zero-shot MACE-MPA-0 NVT snapshots (takes hours)."
        ),
    ] = False,
    compounds: Annotated[
        str | None, typer.Option("--compounds", help="Comma list (default cfg.data.compounds).")
    ] = None,
    config_types: Annotated[
        str | None,
        typer.Option(
            "--config-types",
            help="Comma subset of strain,shear,rattle,eos,vacancy,phonon_disp,md.",
        ),
    ] = None,
) -> None:
    """Generate round-0 candidate frames: strain, shear, rattle, eos, vacancy, phonon_disp, md."""
    from b20mlip.cli import finish, state_of
    from b20mlip.data import sample as sample_module

    state = state_of(ctx)
    cfg = state.settings()
    options = None
    cts = _split_csv(config_types)
    if cts:
        unknown = sorted(set(cts) - set(sample_module.ALL_CONFIG_TYPES))
        if unknown:
            raise typer.BadParameter(f"unknown config types {unknown}")
        options = sample_module.SampleOptions(config_types=tuple(cts))
    result = run_stage(
        "data.sample",
        cfg,
        sample_module.run,
        seed=state.seed if state.seed is not None else 0,
        resume=state.resume,
        executor=state.make_executor(cfg),
        dry_run=state.dry_run,
        out=out,
        parents=parents,
        md=md,
        compounds=_split_csv(compounds),
        options=options,
    )
    finish(result)


def filter_cmd(
    ctx: typer.Context,
    frames: Annotated[Path, typer.Option("--frames", help="Input extxyz.")],
    out: Annotated[Path, typer.Option("--out", help="Output extxyz.")],
    m_ref: Annotated[
        Path | None,
        typer.Option(
            "--m-ref",
            help="JSON {compound: reference muB per TM atom}; enables the magnetic-branch "
            "filter (skipped when absent).",
        ),
    ] = None,
    cap: Annotated[
        float | None,
        typer.Option("--cap", help="Force cap eV/Å (default cfg.data.force_cap_eVA)."),
    ] = None,
    tol: Annotated[
        float | None,
        typer.Option("--tol", help="Branch tolerance muB (default cfg.dft.branch_tol_muB)."),
    ] = None,
    dedupe: Annotated[
        bool, typer.Option("--dedupe/--no-dedupe", help="StructureMatcher dedupe.")
    ] = True,
) -> None:
    """Dedupe, force cap and (with --m-ref) magnetic-branch filter."""
    from b20mlip.cli import finish, state_of
    from b20mlip.data import filters as filters_module

    state = state_of(ctx)
    cfg = state.settings()
    result = run_stage(
        "data.filter",
        cfg,
        filters_module.run,
        seed=state.seed,
        resume=state.resume,
        executor=state.make_executor(cfg),
        dry_run=state.dry_run,
        frames=frames,
        out=out,
        m_ref=m_ref,
        cap=cap,
        tol=tol,
        do_dedupe=dedupe,
    )
    finish(result)


def split(
    ctx: typer.Context,
    frames: Annotated[Path, typer.Option("--frames", help="Input extxyz.")],
    out: Annotated[
        Path, typer.Option("--out", help="Split JSON path, or a directory for <split_id>.json.")
    ] = Path("data/splits"),
    wbm_sample: Annotated[
        Path | None, typer.Option("--wbm-sample", help="WBM sample JSON whose ids become T4b.")
    ] = None,
    fractions: Annotated[str, typer.Option("--fractions", help="train,val,test.")] = "0.8,0.1,0.1",
    holdout: Annotated[
        str, typer.Option("--holdout", help="Compounds never in train.")
    ] = "FeGe,MnGe",
    max_train_t: Annotated[
        float, typer.Option("--max-train-T", help="Frames hotter than this leave train (T1).")
    ] = 600.0,
    train_sources: Annotated[
        str, typer.Option("--train-sources", help="label_source values allowed in train.")
    ] = "qe",
) -> None:
    """Group-hash 80/10/10 split with tiers T0-T4b; writes data/splits/<split_id>.json."""
    from b20mlip.cli import finish, state_of
    from b20mlip.data import split as split_module

    state = state_of(ctx)
    cfg = state.settings()
    fr = [float(x) for x in fractions.split(",")]
    result = run_stage(
        "data.split",
        cfg,
        split_module.run,
        seed=state.seed if state.seed is not None else 0,
        resume=state.resume,
        executor=state.make_executor(cfg),
        dry_run=state.dry_run,
        frames=frames,
        out=out,
        wbm_sample=wbm_sample,
        fractions=fr,
        holdout_compounds=_split_csv(holdout) or [],
        max_train_T=max_train_t,
        train_sources=_split_csv(train_sources) or [],
    )
    finish(result)


def register(group: typer.Typer) -> set[str]:
    """Add the four data commands to the ``data`` group; returns their names."""
    group.command("pull")(pull)
    group.command("sample")(sample)
    group.command("filter")(filter_cmd)
    group.command("split")(split)
    return set(COMMANDS)


__all__ = ["COMMANDS", "filter_cmd", "pull", "register", "sample", "split"]
