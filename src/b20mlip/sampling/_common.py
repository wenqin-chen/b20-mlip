"""Helpers shared by the sampling stages: model metadata, calculators, ``numbers.json`` meta.

Numbers convention (consumed by ``b20mlip.report``): every stage writes ``numbers.json`` in its
run directory with flat dotted keys and a ``<key>@meta`` object per key carrying the gate-A3
fields (``reference{code, functional, pseudos, e0_source, cross_functional}``, ``e0_source``,
``head``, ``n``, ``seed``, ``ci95`` or ``null`` with ``ci95_reason``) plus ``model_label``,
``energy_scale``, ``T`` and ``method``. Sampling numbers are model-vs-model self-consistency
numbers: ``reference.code`` is ``"mace"`` and ``reference.functional`` the training functional
of the model (PBE for MACE-MPA-0 and for every QE-fine-tuned bracket).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from b20mlip.config import Settings
from b20mlip.models import CheckpointInfo
from b20mlip.provenance import RunContext, sha256_file
from b20mlip.sampling.vacancy import b20_cell

NUMBERS_FILE = "numbers.json"
REFERENCE_CODE = "mace"
TRAINING_FUNCTIONAL = "PBE"
CHECKPOINT_JSON = "checkpoint.json"
BRACKET_BY_VARIANT: dict[str, str] = {
    "zero_shot": "B0",
    "naive": "B1",
    "replay": "B2",
    "scratch": "B3",
    "bootstrap": "B4",
}
# foundation checkpoint basenames (b20mlip.train.finetune.FOUNDATION_URLS) -> bracket label
FOUNDATION_LABELS: dict[str, str] = {
    "mace-mpa-0-medium.model": "B0",
    "macempa0mediummodel": "B0",  # MACE's cache name of the same file
    "2023-12-03-mace-128-L1_epoch-199.model": "B0p",
    "20231203mace128L1_epoch199model": "B0p",
}
_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitise_label(text: str) -> str:
    """A key segment: ``[A-Za-z0-9_-]`` only, never empty, never starting with ``-``/``_``."""
    label = _LABEL_RE.sub("_", str(text).strip()).strip("_-")
    return label or "model"


@dataclass(frozen=True)
class ModelInfo:
    """What the numbers meta needs to know about the model that produced a sampling number."""

    path: str
    sha256: str
    label: str
    e0_source: str
    energy_scale: str
    heads: list[str]
    variant: str | None
    checkpoint: str | None

    def reference(self) -> dict[str, Any]:
        return {
            "code": REFERENCE_CODE,
            "functional": TRAINING_FUNCTIONAL,
            "pseudos": None,
            "e0_source": self.e0_source,
            "cross_functional": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_checkpoint(model_path: Path) -> Path | None:
    """``checkpoint.json`` beside the model or one level up (the ``runs/train/<id>`` layout)."""
    for directory in (model_path.parent, model_path.parent.parent):
        cand = directory / CHECKPOINT_JSON
        if cand.is_file():
            try:
                info = CheckpointInfo.model_validate_json(cand.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if Path(info.model_path).name == model_path.name:
                return cand
    return None


def model_info(model_path: str | Path, label: str | None = None) -> ModelInfo:
    """Hash the model and pick up its training provenance (``checkpoint.json``) when present."""
    path = Path(model_path)
    sha = sha256_file(path)
    checkpoint = find_checkpoint(path)
    variant: str | None = None
    if checkpoint is not None:
        info = CheckpointInfo.model_validate_json(checkpoint.read_text(encoding="utf-8"))
        variant = info.variant
        e0_source: str = info.e0_source
        energy_scale: str = info.energy_scale
        heads = [str(h) for h in info.heads]
        default_label = BRACKET_BY_VARIANT.get(info.variant, sanitise_label(path.stem))
    elif path.name in FOUNDATION_LABELS:
        e0_source, energy_scale, heads = "foundation", "mp", ["Default"]
        variant = "zero_shot"
        default_label = FOUNDATION_LABELS[path.name]
    else:
        e0_source, energy_scale, heads = "unknown", "none", ["Default"]
        default_label = sanitise_label(path.stem)
    return ModelInfo(
        path=str(path),
        sha256=sha,
        label=sanitise_label(label) if label else default_label,
        e0_source=e0_source,
        energy_scale=energy_scale,
        heads=heads,
        variant=variant,
        checkpoint=None if checkpoint is None else str(checkpoint),
    )


def make_calculator(model_path: str | Path, cfg: Settings | None = None, head: str | None = None):
    """A ``MACECalculator`` for ``model_path`` on ``cfg.compute`` (CPU float64 by default)."""
    import torch
    from mace.calculators import MACECalculator

    device = cfg.compute.device if cfg is not None else "cpu"
    dtype = cfg.compute.dtype if cfg is not None else "float64"
    if cfg is not None and device == "cpu":
        torch.set_num_threads(int(cfg.compute.threads))
    kwargs: dict[str, Any] = {}
    if head is not None:
        kwargs["head"] = head
    return MACECalculator(model_paths=str(model_path), device=device, default_dtype=dtype, **kwargs)


def reference_cell(
    cfg: Settings, compound: str, structure: str | Path | None = None
) -> tuple[Atoms, str]:
    """``(cell, source)``: the reference B20 cell of ``compound``.

    From ``structure`` (extxyz; lowest-energy frame of the compound) when given, else from the
    MPtrj extract through the dft tier's ``reference_frame`` when that file exists, else the
    tabulated experimental cell (``vacancy.B20_PARAMS``).
    """
    from b20mlip.io import frame_to_atoms

    if structure is not None:
        from b20mlip.io import read_frames

        frames = [f for f in read_frames(structure) if f.compound == compound]
        if not frames:
            raise ValueError(f"no frame of compound {compound!r} in {structure}")
        best = (
            min(frames, key=lambda f: f.energy if f.energy is not None else math.inf)
            if any(f.energy is not None for f in frames)
            else frames[-1]
        )
        atoms = frame_to_atoms(best)
        source = str(structure)
    else:
        try:
            from b20mlip.dft.structures import reference_frame

            frame = reference_frame(cfg, compound)
        except (ImportError, FileNotFoundError, ValueError):
            atoms, source = b20_cell(compound), "tabulated"
        else:
            atoms = frame_to_atoms(frame)
            source = str(frame.info.get("reference_source", "mptrj"))
    atoms.calc = None
    atoms.info = {}
    return atoms, source


def numbers_meta(
    info: ModelInfo,
    *,
    head: str,
    n: int,
    seed: int,
    T: float,
    method: str,
    ci95: tuple[float, float] | list[float] | None,
    ci95_reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The per-key meta object of the numbers convention (gate A3 fields first)."""
    meta: dict[str, Any] = {
        "reference": info.reference(),
        "e0_source": info.e0_source,
        "head": head,
        "n": int(n),
        "seed": int(seed),
        "ci95": None if ci95 is None else [float(ci95[0]), float(ci95[1])],
        "model_label": info.label,
        "model_sha256": info.sha256,
        "energy_scale": info.energy_scale,
        "T": float(T),
        "method": method,
    }
    if ci95 is None:
        meta["ci95_reason"] = ci95_reason or "not a sampled quantity"
    meta.update(extra)
    return meta


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and paths; non-finite floats become ``None``."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(obj), indent=2) + "\n", encoding="utf-8")
    return p


def write_numbers(ctx: RunContext, numbers: Mapping[str, Any]) -> Path:
    """Write ``numbers.json`` into the run dir and register it as an output."""
    path = write_json(ctx.out_dir / NUMBERS_FILE, dict(numbers))
    ctx.add_output(path, "json")
    return path


def head_label(head: str | None) -> str:
    """The head name recorded in the numbers meta (gate A3 accepts ``Default``/``pt_head``)."""
    return head if head in ("Default", "pt_head") else "Default"


__all__ = [
    "BRACKET_BY_VARIANT",
    "FOUNDATION_LABELS",
    "NUMBERS_FILE",
    "REFERENCE_CODE",
    "TRAINING_FUNCTIONAL",
    "ModelInfo",
    "find_checkpoint",
    "head_label",
    "json_safe",
    "make_calculator",
    "model_info",
    "numbers_meta",
    "reference_cell",
    "sanitise_label",
    "write_json",
    "write_numbers",
]
