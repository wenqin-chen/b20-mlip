"""``data split``: deterministic group-hash 80/10/10 split and the evaluation tiers.

Policy ``group_hash`` (binding): ``u = int(sha256(f"{group_id}|{seed}")[:16], 16) / 2**64``
maps every ``group_id`` to a uniform number in ``[0, 1)``; groups with ``u < f_train`` are
train, ``u < f_train + f_val`` val, the rest test. Frames never leave their group's bucket
except through these exclusions, which move the affected frames to **test**:

* T1 — ``temperature_K > max_train_T`` (frame-level: cooler frames of the same MD group
  may stay in train; the hot frames are the extrapolation test);
* T2 — ``compound in holdout_compounds`` (never trained on, whole groups);
* T3 — ``label_source == "omat24"`` frames not in train (never trained on, whole groups);
* T4a — ``label_source == "mptrj"`` (foundation-training data; forgetting tier, never in
  train here — B2 replay uses MACE's own replay set);
* only ``label_source in train_sources`` (default ``("qe",)``) can be trained on: the SPEC
  forbids mixed-energy-scale training, so unlabelled candidates and other sources are also
  excluded from train.

Tiers: T0 = val + test frames labelled by QE (held-out groups); T1/T2/T3/T4a as above (every
matching frame, wherever it landed); T4b = the WBM sample ids passed in. ``split_id`` is
``sha256(frames_sha256, seed, fractions, policy, holdout, max_train_T, train_sources)[:12]``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from b20mlip.config import Settings
from b20mlip.io import read_frames
from b20mlip.models import Frame, Split, Tier
from b20mlip.provenance import RunContext, sha256_frames

Bucket = Literal["train", "val", "test"]
POLICY = "group_hash"
DEFAULT_FRACTIONS: tuple[float, float, float] = (0.8, 0.1, 0.1)
DEFAULT_HOLDOUT: tuple[str, ...] = ("FeGe", "MnGe")
DEFAULT_TRAIN_SOURCES: tuple[str, ...] = ("qe",)
TIERS: tuple[Tier, ...] = ("T0", "T1", "T2", "T3", "T4a", "T4b")


def group_hash(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{group_id}|{seed}".encode()).hexdigest()
    return int(digest[:16], 16) / float(2**64)


def bucket_of(u: float, fractions: Sequence[float]) -> Bucket:
    f_train, f_val, _ = fractions
    if u < f_train:
        return "train"
    if u < f_train + f_val:
        return "val"
    return "test"


def split_id_for(
    frames_sha256: str,
    seed: int,
    fractions: Sequence[float],
    policy: str = POLICY,
    **extra: Any,
) -> str:
    payload = {
        "frames_sha256": frames_sha256,
        "seed": seed,
        "fractions": [float(f) for f in fractions],
        "policy": policy,
        **extra,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def group_split(
    frames: Sequence[Frame],
    seed: int = 0,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    holdout_compounds: Iterable[str] = DEFAULT_HOLDOUT,
    max_train_T: float = 600.0,
    *,
    wbm_ids: Iterable[str] | None = None,
    train_sources: Iterable[str] = DEFAULT_TRAIN_SOURCES,
) -> Split:
    """Deterministic group split plus tiers (policy in the module docstring)."""
    fr = tuple(float(f) for f in fractions)
    if len(fr) != 3 or abs(sum(fr) - 1.0) > 1e-6 or min(fr) < 0:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1, got {fr}")
    holdout = {c for c in holdout_compounds}
    sources = {s for s in train_sources}
    ids = [f.frame_id for f in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate frame_ids; run `data filter` (dedupe) first")

    groups: dict[str, list[str]] = {}
    for f in frames:
        groups.setdefault(f.group_id, []).append(f.frame_id)
    bucket: dict[str, Bucket] = {gid: bucket_of(group_hash(gid, seed), fr) for gid in groups}

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    tiers: dict[Tier, list[str]] = {t: [] for t in TIERS}
    for f in frames:
        b = bucket[f.group_id]
        hot = f.temperature_K is not None and f.temperature_K > max_train_T
        held_compound = f.compound in holdout
        if hot:
            tiers["T1"].append(f.frame_id)
        if held_compound:
            tiers["T2"].append(f.frame_id)
        trainable = b == "train" and f.label_source in sources and not hot and not held_compound
        if b == "train" and not trainable:
            b = "test"
        # OMat24 / MPtrj frames are tiers only when NOT trained on (the default train_sources =
        # ("qe",) keeps all of them out of train; the bootstrap variant B4 opts OMat24 in, and its
        # held-out groups remain the never-trained VASP tier T3)
        if f.label_source == "omat24" and b != "train":
            tiers["T3"].append(f.frame_id)
        if f.label_source == "mptrj" and b != "train":
            tiers["T4a"].append(f.frame_id)
        if b == "train":
            train.append(f.frame_id)
        elif b == "val":
            val.append(f.frame_id)
        else:
            test.append(f.frame_id)
        if b != "train" and f.label_source == "qe":
            tiers["T0"].append(f.frame_id)
    tiers["T4b"] = [str(i) for i in (wbm_ids or [])]

    frames_sha = sha256_frames(frames)
    split_id = split_id_for(
        frames_sha,
        seed,
        fr,
        POLICY,
        holdout_compounds=sorted(holdout),
        max_train_T=float(max_train_T),
        train_sources=sorted(sources),
    )
    return Split(
        split_id=split_id,
        seed=seed,
        frames_sha256=frames_sha,
        policy="group_hash",
        fractions=(fr[0], fr[1], fr[2]),
        train=train,
        val=val,
        test=test,
        tiers=tiers,
        groups=groups,
        holdout_compounds=sorted(holdout),
        max_train_T=float(max_train_T),
    )


def write_split(split: Split, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(split.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return p


def read_split(path: str | Path) -> Split:
    return Split.model_validate_json(Path(path).read_text(encoding="utf-8"))


def frames_by_bucket(split: Split, frames: Iterable[Frame]) -> dict[str, list[Frame]]:
    """Materialise ``train``/``val``/``test`` frame lists from a split (helper for train/eval)."""
    by_id = {f.frame_id: f for f in frames}
    return {
        name: [by_id[i] for i in getattr(split, name) if i in by_id]
        for name in ("train", "val", "test")
    }


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    frames: str | Path,
    out: str | Path,
    wbm_sample: str | Path | None = None,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    holdout_compounds: Iterable[str] = DEFAULT_HOLDOUT,
    max_train_T: float = 600.0,
    train_sources: Iterable[str] = DEFAULT_TRAIN_SOURCES,
) -> dict[str, Any]:
    """Stage ``data.split``: ``--frames F --out data/splits/`` -> ``<out>/<split_id>.json``."""
    src = Path(frames)
    ctx.add_input(src, "frames")
    wbm_ids: list[str] | None = None
    if wbm_sample is not None:
        from b20mlip.data.wbm import load_sample

        ctx.add_input(Path(wbm_sample), "json")
        wbm_ids = list(load_sample(wbm_sample)["ids"])
    seed = ctx.seed if ctx.seed is not None else 0
    if ctx.dry_run:
        ctx.log(plan={"frames": str(src), "out": str(out), "seed": seed})
        return {"planned": 1}
    data = read_frames(src)
    split = group_split(
        data,
        seed,
        fractions,
        holdout_compounds,
        max_train_T,
        wbm_ids=wbm_ids,
        train_sources=train_sources,
    )
    out_path = Path(out)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / f"{split.split_id}.json"
    write_split(split, out_path)
    ctx.add_output(out_path, "json")
    summary: dict[str, Any] = {
        "split_id": split.split_id,
        "n_frames": len(data),
        "n_groups": len(split.groups),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
    }
    for tier, ids in split.tiers.items():
        summary[f"n_{tier}"] = len(ids)
    ctx.log(split={k: v for k, v in summary.items()})
    return summary


__all__ = [
    "DEFAULT_FRACTIONS",
    "DEFAULT_HOLDOUT",
    "DEFAULT_TRAIN_SOURCES",
    "POLICY",
    "TIERS",
    "bucket_of",
    "frames_by_bucket",
    "group_hash",
    "group_split",
    "read_split",
    "run",
    "split_id_for",
    "write_split",
]
