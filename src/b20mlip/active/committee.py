"""Committee-uncertainty frame selection (CONTRACTS.md row 12, SPEC.md G5).

* ``sigma_f(model_paths, frames)`` — per-frame committee uncertainty: every model of the
  committee (>= 2 ``.model`` files, e.g. the three seeds of one bracket) predicts the forces
  of every frame; per atom the standard deviation of the force vector across models is
  ``sqrt(Σ_α Var_models(F_α))`` (population std over the committee, ``ddof=0``, summed over
  the three Cartesian components), and the frame's ``σ_F`` is the **maximum over atoms**
  (eV/Å). Identical models give exactly 0.
* ``select(frames, sigma, n, sigma_min, per_group_max)`` — highest ``σ_F`` first; frames with
  ``σ_F < sigma_min`` are skipped; at most ``per_group_max`` frames per ``group_id`` (so one
  badly described parent cannot fill the whole batch); ties broken by ``frame_id`` so the
  choice is deterministic; at most ``n`` frames.
* ``run(cfg, ctx, ...)`` writes the candidates as extxyz with ``info["committee_sigma_f"]``
  (and ``committee_rank``), ``sigma_f.json`` with every frame's σ and ``numbers.json`` with
  ``active.n_selected`` (the README key; ``active.selected_n`` is published too),
  ``active.sigma_f_median`` and ``active.sigma_f_max`` (eV/Å).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from b20mlip.config import Settings
from b20mlip.io import read_frames, write_frames
from b20mlip.models import Frame
from b20mlip.provenance import RunContext, sha256_file

SIGMA_KEY = "committee_sigma_f"
RANK_KEY = "committee_rank"
NUMBERS_FILE = "numbers.json"
SIGMA_FILE = "sigma_f.json"
MIN_COMMITTEE = 2


def committee_forces(
    model_paths: Sequence[str | Path],
    frames: Sequence[Frame],
    *,
    head: str = "Default",
    device: str = "cpu",
    dtype: str = "float64",
    calcs: Sequence[Any] | None = None,
) -> list[list[np.ndarray]]:
    """``forces[model][frame]`` (N, 3) arrays for every committee member and frame."""
    from b20mlip.evaluate.errors import make_calculator, predict  # noqa: PLC0415 - row 9

    if calcs is None:
        if len(model_paths) < MIN_COMMITTEE:
            raise ValueError(
                f"a committee needs at least {MIN_COMMITTEE} models, got {len(model_paths)}"
            )
        for p in model_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"committee model not found: {p}")
        calcs = [make_calculator(p, head, device=device, dtype=dtype) for p in model_paths]
    elif len(calcs) < MIN_COMMITTEE:
        raise ValueError(
            f"a committee needs at least {MIN_COMMITTEE} calculators, got {len(calcs)}"
        )
    out: list[list[np.ndarray]] = []
    for calc in calcs:
        out.append([predict(calc, frame)[1] for frame in frames])
    return out


def sigma_from_forces(forces: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Per-frame ``max_atoms sqrt(Σ_α Var_models F_α)`` from ``forces[model][frame]``."""
    n_models = len(forces)
    if n_models < MIN_COMMITTEE:
        raise ValueError(f"a committee needs at least {MIN_COMMITTEE} models, got {n_models}")
    n_frames = len(forces[0])
    if any(len(f) != n_frames for f in forces):
        raise ValueError("every committee member must predict the same frames")
    sigma = np.empty(n_frames, dtype=float)
    for i in range(n_frames):
        stack = np.stack(
            [np.asarray(forces[m][i], dtype=float) for m in range(n_models)]
        )  # (M, N, 3)
        per_atom = np.sqrt(np.sum(np.var(stack, axis=0, ddof=0), axis=1))  # (N,)
        sigma[i] = float(np.max(per_atom)) if per_atom.size else 0.0
    return sigma


def sigma_f(
    model_paths: Sequence[str | Path],
    frames: Sequence[Frame],
    *,
    head: str = "Default",
    device: str = "cpu",
    dtype: str = "float64",
    calcs: Sequence[Any] | None = None,
) -> np.ndarray:
    """Per-frame committee force uncertainty ``σ_F`` (eV/Å; see module docstring)."""
    if not frames:
        return np.zeros(0, dtype=float)
    return sigma_from_forces(
        committee_forces(model_paths, frames, head=head, device=device, dtype=dtype, calcs=calcs)
    )


def select(
    frames: Sequence[Frame],
    sigma: Sequence[float] | np.ndarray,
    n: int,
    sigma_min: float = 0.0,
    per_group_max: int | None = None,
) -> list[Frame]:
    """Highest-σ selection with a per-group cap and a deterministic tie-break (see module).

    The returned frames carry ``info["committee_sigma_f"]`` and ``info["committee_rank"]``.
    """
    sig = np.asarray(sigma, dtype=float)
    if sig.shape != (len(frames),):
        raise ValueError(f"sigma has shape {sig.shape}, expected ({len(frames)},)")
    if n < 0:
        raise ValueError("n must be >= 0")
    if per_group_max is not None and per_group_max < 1:
        raise ValueError("per_group_max must be >= 1")
    order = sorted(range(len(frames)), key=lambda i: (-sig[i], frames[i].frame_id))
    per_group: dict[str, int] = {}
    chosen: list[Frame] = []
    for i in order:
        if len(chosen) >= n:
            break
        if not np.isfinite(sig[i]) or sig[i] < sigma_min:
            continue
        group = frames[i].group_id
        if per_group_max is not None and per_group.get(group, 0) >= per_group_max:
            continue
        per_group[group] = per_group.get(group, 0) + 1
        chosen.append(
            frames[i].model_copy(
                update={"info": {**frames[i].info, SIGMA_KEY: float(sig[i]), RANK_KEY: len(chosen)}}
            )
        )
    return chosen


def parse_models(text: str | Sequence[str | Path]) -> list[Path]:
    items = (
        [t.strip() for t in text.split(",")] if isinstance(text, str) else [str(t) for t in text]
    )
    return [Path(t) for t in items if t]


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    models: str | Sequence[str | Path],
    frames_path: str | Path,
    n: int = 100,
    out: str | Path = Path("data/frames/candidates_r1.extxyz"),
    sigma_min: float = 0.0,
    per_group_max: int | None = None,
    head: str = "Default",
    calcs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Stage ``active.select``: committee σ_F over ``frames_path`` -> ``out`` candidates."""
    paths = parse_models(models)
    if len(paths) < MIN_COMMITTEE:
        raise ValueError(
            f"--models needs at least {MIN_COMMITTEE} committee members, got {len(paths)}"
        )
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"committee model not found: {p}")
        ctx.add_input(p, "model")
    frames_file = Path(frames_path)
    if not frames_file.is_file():
        raise FileNotFoundError(f"frames file not found: {frames_file}")
    ctx.add_input(frames_file, "frames")
    frames = read_frames(frames_file)
    out_path = Path(out)
    plan = {
        "models": [str(p) for p in paths],
        "model_sha256": [sha256_file(p) for p in paths],
        "head": head,
        "frames": str(frames_file),
        "n_candidates": len(frames),
        "n": int(n),
        "sigma_min": float(sigma_min),
        "per_group_max": per_group_max,
        "out": str(out_path),
        "dry_run": bool(ctx.dry_run),
    }
    (ctx.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / "plan.json", "json")
    if ctx.dry_run:
        ctx.log(n_models=len(paths), n_candidates=len(frames), planned=True)
        return {"n_candidates": len(frames), "n_models": len(paths), "planned": 1}
    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(int(cfg.compute.threads))
    except Exception:  # noqa: BLE001
        pass
    sigma = sigma_f(
        paths, frames, head=head, device=cfg.compute.device, dtype=cfg.compute.dtype, calcs=calcs
    )
    chosen = select(frames, sigma, int(n), float(sigma_min), per_group_max)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_frames(chosen, out_path)
    ctx.add_output(out_path, "frames")
    local = ctx.out_dir / out_path.name
    if local.resolve() != out_path.resolve():
        write_frames(chosen, local)
        ctx.add_output(local, "frames")
    sigma_doc = {
        "unit": "eV/Å",
        "definition": "max over atoms of sqrt(sum_alpha Var_models F_alpha)",
        "n_models": len(paths),
        "sigma_f": {f.frame_id: float(s) for f, s in zip(frames, sigma, strict=True)},
        "selected": [f.frame_id for f in chosen],
    }
    (ctx.out_dir / SIGMA_FILE).write_text(json.dumps(sigma_doc, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / SIGMA_FILE, "json")
    median = float(np.median(sigma)) if sigma.size else 0.0
    maximum = float(np.max(sigma)) if sigma.size else 0.0
    meta = {
        "unit": "eV/Å",
        "n": len(frames),
        "n_models": len(paths),
        "head": head,
        "sigma_min": float(sigma_min),
        "per_group_max": per_group_max,
        "models_sha256": plan["model_sha256"],
    }
    numbers: dict[str, Any] = {
        "active.n_selected": len(chosen),
        "active.n_selected@meta": {**meta, "unit": "count"},
        "active.selected_n": len(chosen),
        "active.selected_n@meta": {**meta, "unit": "count", "alias_of": "active.n_selected"},
        "active.n_candidates": len(frames),
        "active.n_candidates@meta": {**meta, "unit": "count"},
        "active.sigma_f_median": median,
        "active.sigma_f_median@meta": meta,
        "active.sigma_f_max": maximum,
        "active.sigma_f_max@meta": meta,
    }
    (ctx.out_dir / NUMBERS_FILE).write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / NUMBERS_FILE, "json")
    ctx.log(
        n_models=len(paths),
        models_sha256=plan["model_sha256"],
        head=head,
        n_candidates=len(frames),
        n_selected=len(chosen),
        sigma_f_median=median,
        sigma_f_max=maximum,
        sigma_min=float(sigma_min),
        per_group_max=per_group_max,
    )
    return {
        "n_candidates": len(frames),
        "n_selected": len(chosen),
        "n_models": len(paths),
        "sigma_f_median": round(median, 6),
        "sigma_f_max": round(maximum, 6),
        "out": str(out_path),
    }


__all__ = [
    "MIN_COMMITTEE",
    "NUMBERS_FILE",
    "RANK_KEY",
    "SIGMA_FILE",
    "SIGMA_KEY",
    "committee_forces",
    "parse_models",
    "run",
    "select",
    "sigma_f",
    "sigma_from_forces",
]
