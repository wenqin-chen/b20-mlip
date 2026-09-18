"""LAMMPS export (CONTRACTS.md row 7): ``to_lammps`` wraps ``mace_create_lammps_model``.

``mace_create_lammps_model <model> --head <head> --dtype float64`` (mace-torch 0.3.16) wraps the
model in ``LAMMPS_MACE``, TorchScript-compiles it and writes ``<model>-lammps.pt`` next to the
input. The head is always passed explicitly: with several heads (a replay checkpoint) the tool
would otherwise prompt on stdin. The ``export`` stage records both files' SHA-256 in
``export.json`` and the run manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from b20mlip.config import Settings
from b20mlip.provenance import RunContext, sha256_file
from b20mlip.train.finetune import inspect_model

MACE_EXPORT_SCRIPT = "mace_create_lammps_model"
MACE_EXPORT_MODULE = "mace.cli.create_lammps_model"
LAMMPS_SUFFIX = "-lammps.pt"
EXPORT_JSON = "export.json"


def lammps_path_for(model_path: str | Path) -> Path:
    """Where ``mace_create_lammps_model`` writes: ``<model path>-lammps.pt``."""
    p = Path(model_path)
    return p.with_name(p.name + LAMMPS_SUFFIX)


def export_command(model_path: str | Path, head: str, dtype: str = "float64") -> list[str]:
    return [
        sys.executable, "-m", MACE_EXPORT_MODULE, str(model_path),
        "--head", head, "--dtype", dtype, "--format", "libtorch",
    ]  # fmt: skip


def choose_head(model_path: str | Path, head: str | None = None) -> str:
    """``head`` if given, else ``Default`` when the model has it, else its last head."""
    heads = inspect_model(model_path)["heads"]
    if head is not None:
        if head not in heads:
            raise ValueError(f"model {model_path} has heads {heads}, not {head!r}")
        return head
    return "Default" if "Default" in heads else heads[-1]


def to_lammps(model_path: str | Path, head: str | None = None, *, dtype: str = "float64") -> str:
    """Run ``mace_create_lammps_model`` and return the ``-lammps.pt`` path."""
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    head = choose_head(model, head)
    proc = subprocess.run(
        export_command(model, head, dtype),
        cwd=model.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    out = lammps_path_for(model)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"{MACE_EXPORT_SCRIPT} failed (exit {proc.returncode}) for {model}:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return str(out)


def run(
    cfg: Settings, ctx: RunContext, model: str | Path, head: str | None = None
) -> dict[str, Any]:
    """Stage ``export``: ``<model>-lammps.pt`` plus ``export.json`` with both SHA-256 hashes."""
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    ctx.add_input(model_path, "model")
    record: dict[str, Any] = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "head": head,
        "lammps_path": str(lammps_path_for(model_path)),
        "lammps_sha256": None,
        "dtype": cfg.compute.dtype,
        "dry_run": bool(ctx.dry_run),
    }
    if not ctx.dry_run:
        record["head"] = choose_head(model_path, head)
        lammps = Path(to_lammps(model_path, record["head"], dtype=cfg.compute.dtype))
        record["lammps_path"] = str(lammps)
        record["lammps_sha256"] = sha256_file(lammps)
        ctx.add_output(lammps, "model")
    out = ctx.out_dir / EXPORT_JSON
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(out, "json")
    ctx.log(
        model_sha256=record["model_sha256"],
        lammps_sha256=record["lammps_sha256"],
        head=record["head"],
    )
    return {
        "model_sha256": record["model_sha256"],
        "lammps_path": record["lammps_path"],
        "lammps_sha256": record["lammps_sha256"] or "",
    }


__all__ = [
    "EXPORT_JSON",
    "LAMMPS_SUFFIX",
    "MACE_EXPORT_MODULE",
    "MACE_EXPORT_SCRIPT",
    "choose_head",
    "export_command",
    "lammps_path_for",
    "run",
    "to_lammps",
]
