"""Top-level ``train`` and ``export`` commands, plugged into the root CLI (``register_toplevel``).

The root CLI imports this module at start-up, so it must stay cheap: ``finish``/``state_of``
and the stage modules are imported inside the command bodies.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from b20mlip.config import Settings
from b20mlip.provenance import RunContext, run_stage


class VariantOpt(StrEnum):
    naive = "naive"
    replay = "replay"
    scratch = "scratch"
    bootstrap = "bootstrap"


def _train_stage(cfg: Settings, ctx: RunContext, **kw: Any) -> dict[str, Any]:
    """``finetune.run`` adapted to ``run_stage`` (which summarises dicts, not CheckpointInfo)."""
    from b20mlip.train import finetune

    info = finetune.run(cfg, ctx, **kw)
    if info is None:  # dry run: plan only
        return {"planned": 1, "variant": str(kw.get("variant", ""))}
    summary: dict[str, Any] = {
        "model_path": info.model_path,
        "sha256": info.sha256,
        "variant": info.variant,
        "e0_source": info.e0_source,
        "energy_scale": info.energy_scale,
        "epochs": info.epochs,
    }
    summary.update({f"val_{k}": v for k, v in info.val_metrics.items()})
    return summary


def register_toplevel(app: typer.Typer) -> set[str]:
    @app.command("train")
    def train(
        ctx: typer.Context,
        variant: Annotated[
            VariantOpt, typer.Option("--variant", help="Bracket: naive (B1), replay (B2), "
                                                      "scratch (B3), bootstrap (B4).")
        ] = VariantOpt.naive,
        split: Annotated[
            Path | None, typer.Option("--split", help="Split JSON (data/splits/v1.json).")
        ] = None,
        seed: Annotated[int | None, typer.Option("--seed", help="MACE seed (default 0).")] = None,
        lr: Annotated[float | None, typer.Option("--lr", help="Override train.lr.")] = None,
        epochs: Annotated[
            int | None, typer.Option("--epochs", help="Override train.epochs.")
        ] = None,
        frames: Annotated[
            Path | None,
            typer.Option("--frames", help="Frames extxyz; with no --split an in-function 90/10 "
                                          "group split is used."),
        ] = None,
        out: Annotated[
            Path | None,
            typer.Option("--out", help="Also copy the final .model (and checkpoint.json) here."),
        ] = None,
    ) -> None:  # fmt: skip
        """Fine-tune MACE (naive|replay|scratch|bootstrap); writes runs/train/<run_id>/."""
        import shutil

        from b20mlip.cli import finish, state_of

        state = state_of(ctx)
        cfg = state.settings()
        seed_value = seed if seed is not None else (state.seed if state.seed is not None else 0)
        result = run_stage(
            "train",
            cfg,
            _train_stage,
            seed=seed_value,
            resume=state.resume,
            executor=state.make_executor(cfg),
            dry_run=state.dry_run,
            variant=variant.value,
            split=split,
            frames=frames,
            lr=lr,
            epochs=epochs,
        )
        if out is not None and result.status == "ok":
            out.mkdir(parents=True, exist_ok=True)
            for art in result.outputs:
                if art.kind == "model" or Path(art.path).name == "checkpoint.json":
                    shutil.copy2(art.path, out / Path(art.path).name)
        finish(result)

    @app.command("export")
    def export(
        ctx: typer.Context,
        model: Annotated[Path, typer.Option("--model", help="Trained .model to export.")],
        head: Annotated[
            str | None, typer.Option("--head", help="Head to export (default: Default).")
        ] = None,
    ) -> None:
        """mace_create_lammps_model: <model>-lammps.pt plus a manifest with both hashes."""
        from b20mlip.cli import finish, state_of
        from b20mlip.train import export as export_module

        state = state_of(ctx)
        cfg = state.settings()
        result = run_stage(
            "export",
            cfg,
            export_module.run,
            seed=state.seed,
            resume=state.resume,
            executor=state.make_executor(cfg),
            dry_run=state.dry_run,
            model=model,
            head=head,
        )
        finish(result)

    return {"train", "export"}


__all__ = ["VariantOpt", "register_toplevel"]
