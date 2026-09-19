"""``b20mlip eval`` commands (CONTRACTS.md section 3), plugged into the root CLI.

``register(group)`` adds ``errors``, ``discovery``, ``phonons`` and ``elastic``. The root CLI
imports this module at import time; command bodies import ``finish``/``state_of`` lazily to
avoid the circular import, resolve the settings, run the stage under ``run_stage`` and exit 0
only when the ``StageResult`` status is ``ok``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip.provenance import run_stage

COMMANDS: frozenset[str] = frozenset({"errors", "discovery", "phonons", "elastic"})
HEAD_HELP = "MACE head: Default (QE scale for fine-tuned models) or pt_head (MP scale, B2 replay)."
CKPT_HELP = "checkpoint.json of the train stage (energy scale, E0 source, bracket label)."
SCALE_HELP = "Model energy scale mp|omat24|qe|none when no checkpoint.json is given."
LABEL_HELP = (
    "Bracket label used in numbers.json keys (B0, B0p, B1, B2, B3, B4); "
    "default from the checkpoint."
)


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
    """Add the eval commands to ``group``; returns the names registered."""

    @group.command("errors")
    def errors(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        head: Annotated[str, typer.Option("--head", help=HEAD_HELP)] = "Default",
        split: Annotated[
            Path | None, typer.Option("--split", help="Split JSON (data split).")
        ] = None,
        tiers: Annotated[
            str | None,
            typer.Option("--tiers", help="Comma list of T0,T1,T2,T3,T4a (default: all present)."),
        ] = None,
        frames: Annotated[
            Path | None, typer.Option("--frames", help="Frames extxyz (default: the split's file).")
        ] = None,
        tier: Annotated[
            str | None,
            typer.Option("--tier", help="Without --split: evaluate the whole file as this tier."),
        ] = None,
        checkpoint_json: Annotated[
            Path | None, typer.Option("--checkpoint-json", help=CKPT_HELP)
        ] = None,
        energy_scale: Annotated[str | None, typer.Option("--energy-scale", help=SCALE_HELP)] = None,
        label: Annotated[str | None, typer.Option("--label", help=LABEL_HELP)] = None,
        e0_source: Annotated[
            str | None,
            typer.Option(
                "--e0-source", help="E0 source recorded in the meta (default: checkpoint)."
            ),
        ] = None,
        noise_floor: Annotated[
            float | None,
            typer.Option("--noise-floor", help="T3 force noise floor (meV/Å, rule R5)."),
        ] = None,
        noise_floor_frames: Annotated[
            Path | None,
            typer.Option(
                "--noise-floor-frames",
                help="QE re-labels of OMat24 frames; the floor is computed from them.",
            ),
        ] = None,
        bootstrap_n: Annotated[
            int | None, typer.Option("--bootstrap-n", help="Resamples (cfg.eval).")
        ] = None,
    ) -> None:
        """E/F/S error tables per tier with 95 % group-bootstrap CIs -> errors_<tier>.json."""
        from b20mlip.evaluate import errors as mod  # noqa: PLC0415

        _run(
            ctx, "eval.errors", mod.run, model=model, head=head, split=split, tiers=tiers,
            frames_path=frames, tier=tier, checkpoint_json=checkpoint_json,
            energy_scale=energy_scale, label=label, e0_source=e0_source,
            noise_floor_f=noise_floor, noise_floor_frames=noise_floor_frames,
            bootstrap_n=bootstrap_n,
        )  # fmt: skip

    @group.command("discovery")
    def discovery(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        sample: Annotated[Path, typer.Option("--sample", help="WBM sample JSON (data sample).")],
        head: Annotated[str, typer.Option("--head", help=HEAD_HELP)] = "pt_head",
        atoms_zip: Annotated[
            Path | None,
            typer.Option(
                "--atoms-zip", help="wbm-initial-atoms.extxyz.zip (default data/raw/references)."
            ),
        ] = None,
        baseline_run: Annotated[
            Path | None,
            typer.Option(
                "--baseline-run", help="B0 discovery run dir to pair against (paired ΔF1)."
            ),
        ] = None,
        checkpoint_json: Annotated[
            Path | None, typer.Option("--checkpoint-json", help=CKPT_HELP)
        ] = None,
        energy_scale: Annotated[str | None, typer.Option("--energy-scale", help=SCALE_HELP)] = None,
        offsets: Annotated[
            Path | None,
            typer.Option(
                "--offsets", help="offsets.json (dft offsets) for QE-scale models, rule R3."
            ),
        ] = None,
        label: Annotated[str | None, typer.Option("--label", help=LABEL_HELP)] = None,
        e0_source: Annotated[
            str | None, typer.Option("--e0-source", help="E0 source recorded in the meta.")
        ] = None,
        limit: Annotated[
            int | None, typer.Option("--limit", help="Only the first N ids (smoke runs).")
        ] = None,
        fmax: Annotated[
            float | None, typer.Option("--fmax", help="Relaxation fmax eV/Å (cfg.eval.fmax).")
        ] = None,
        max_steps: Annotated[
            int | None, typer.Option("--max-steps", help="Relaxation step cap (cfg.eval).")
        ] = None,
        bootstrap_n: Annotated[
            int | None, typer.Option("--bootstrap-n", help="Resamples (cfg.eval).")
        ] = None,
    ) -> None:
        """Relax the labelled WBM sample; vendored stable metrics; paired ΔF1 vs B0 (resumable)."""
        from b20mlip.evaluate import discovery as mod  # noqa: PLC0415

        _run(
            ctx, "eval.discovery", mod.run, model=model, head=head, sample=sample,
            atoms_zip=atoms_zip, baseline_run=baseline_run, checkpoint_json=checkpoint_json,
            energy_scale=energy_scale, offsets=offsets, label=label, e0_source=e0_source,
            limit=limit, fmax=fmax, max_steps=max_steps, bootstrap_n=bootstrap_n,
        )  # fmt: skip

    @group.command("phonons")
    def phonons(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        compound: Annotated[str, typer.Option("--compound", help="B20 compound, e.g. FeSi.")],
        reference: Annotated[
            str,
            typer.Option(
                "--reference", help="qe (own force sets) | phonondb103 | pbesol (cross-functional)."
            ),
        ] = "qe",
        cell: Annotated[
            str, typer.Option("--cell", help="dft (reference cell) | model (model-relaxed cell).")
        ] = "dft",
        reference_json: Annotated[
            Path | None,
            typer.Option("--reference-json", help="PhononResult JSON for phonondb103/pbesol."),
        ] = None,
        force_sets: Annotated[
            Path | None,
            typer.Option(
                "--force-sets",
                help="QE force_sets.json (default: data/phonons or the latest ok dft.phonons run).",
            ),
        ] = None,
        structure: Annotated[
            Path | None,
            typer.Option("--structure", help="extxyz with the reference cell (non-qe references)."),
        ] = None,
        supercell: Annotated[
            tuple[int, int, int],
            typer.Option("--supercell", help="Supercell for non-qe references."),
        ] = (2, 2, 2),
        distance: Annotated[
            float, typer.Option("--distance", help="Displacement amplitude (Å).")
        ] = 0.03,
        npoints: Annotated[
            int, typer.Option("--npoints", help="q-points along the seekpath path.")
        ] = 100,
        mesh: Annotated[int, typer.Option("--mesh", help="DOS mesh (0 = no DOS).")] = 20,
        head: Annotated[str, typer.Option("--head", help=HEAD_HELP)] = "Default",
        label: Annotated[str | None, typer.Option("--label", help=LABEL_HELP)] = None,
        checkpoint_json: Annotated[
            Path | None, typer.Option("--checkpoint-json", help=CKPT_HELP)
        ] = None,
        e0_source: Annotated[
            str | None, typer.Option("--e0-source", help="E0 source recorded in the meta.")
        ] = None,
    ) -> None:
        """Model phonons vs a labelled reference: ω-MAE (meV), softening index, imaginary count."""
        from b20mlip.evaluate import phonon_compare as mod  # noqa: PLC0415

        _run(
            ctx, "eval.phonons", mod.run, model=model, compound=compound, reference=reference,
            cell=cell, reference_json=reference_json, force_sets=force_sets, structure=structure,
            supercell=tuple(supercell), distance=distance, npoints=npoints,
            mesh=(mesh if mesh > 0 else None), head=head, label=label,
            checkpoint_json=checkpoint_json, e0_source=e0_source,
        )  # fmt: skip

    @group.command("elastic")
    def elastic(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="MACE .model file.")],
        compound: Annotated[str, typer.Option("--compound", help="B20 compound, e.g. FeSi.")],
        structure: Annotated[
            Path | None,
            typer.Option(
                "--structure", help="extxyz with the reference cell (default MPtrj B20 extract)."
            ),
        ] = None,
        cell: Annotated[
            str, typer.Option("--cell", help="dft (reference cell) | model (model-relaxed cell).")
        ] = "dft",
        strains: Annotated[
            str, typer.Option("--strains", help="Comma list of strain magnitudes.")
        ] = "0.005,0.01",
        pct: Annotated[float, typer.Option("--pct", help="EOS volume range ±pct %.")] = 8.0,
        npoints: Annotated[int, typer.Option("--npoints", help="EOS points.")] = 9,
        head: Annotated[str, typer.Option("--head", help=HEAD_HELP)] = "Default",
        label: Annotated[str | None, typer.Option("--label", help=LABEL_HELP)] = None,
        checkpoint_json: Annotated[
            Path | None, typer.Option("--checkpoint-json", help=CKPT_HELP)
        ] = None,
        e0_source: Annotated[
            str | None, typer.Option("--e0-source", help="E0 source recorded in the meta.")
        ] = None,
        constants: Annotated[
            bool, typer.Option("--constants/--no-constants", help="Compute C11/C12/C44/B.")
        ] = True,
        eos: Annotated[
            bool, typer.Option("--eos/--no-eos", help="Compute the Birch–Murnaghan EOS.")
        ] = True,
        full: Annotated[
            bool, typer.Option("--full", help="All six Voigt modes (symmetry check).")
        ] = False,
    ) -> None:
        """Cubic elastic constants (GPa) and EOS (a0, B0) of a compound with the model."""
        from b20mlip.evaluate import elastic as mod  # noqa: PLC0415

        try:
            mags = tuple(float(s) for s in strains.split(",") if s.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"--strains must be a comma list of floats, got {strains!r}"
            ) from exc
        _run(
            ctx, "eval.elastic", mod.run, model=model, compound=compound, structure=structure,
            cell=cell, strains=mags, pct=pct, npoints=npoints, head=head, label=label,
            checkpoint_json=checkpoint_json, e0_source=e0_source, do_constants=constants,
            do_eos=eos, full=full,
        )  # fmt: skip

    return set(COMMANDS)


__all__ = ["COMMANDS", "register"]
