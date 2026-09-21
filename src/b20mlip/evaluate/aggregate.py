"""Seed aggregation for the bracket tables (stage ``eval.errors`` with ``extras.aggregate``).

Every bracket is trained with several seeds (SPEC.md section 6: "3 seeds, bootstrap CIs"), but
one ``eval errors`` run publishes one model. Publishing whichever seed ran last as *the* bracket
number would hide the seed spread, so ``eval aggregate`` re-publishes every metric of the given
runs under the bracket label with

* ``value`` = the mean over seeds,
* ``ci95`` = ``[min, max]`` over seeds (the seed spread, labelled as such in ``ci95_reason``; the
  per-seed bootstrap CIs are listed in the meta), and
* ``seed`` = the comma-joined seeds, ``n`` = the (identical) frame count.

Only runs of the same tier set, split and frame count can be aggregated; the input run ids are
recorded so gate A2 resolves every source manifest.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from b20mlip.config import Settings
from b20mlip.provenance import RunContext, read_manifest

NUMBERS_FILE = "numbers.json"


def _load_numbers(run_dir: Path) -> dict[str, Any]:
    path = run_dir / NUMBERS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no {NUMBERS_FILE} (not an eval run?)")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(numbers_by_run: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    """Mean over runs per metric key; keys are rewritten to ``<...>.<label>.<metric>``."""
    values: dict[str, list[float]] = defaultdict(list)
    metas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_id, numbers in numbers_by_run.items():
        for key, value in numbers.items():
            if key.endswith("@meta") or not isinstance(value, (int, float)):
                continue
            parts = key.split(".")
            if len(parts) < 4:
                continue
            parts[-2] = label  # <namespace>.<tier>.<label>.<metric>
            new_key = ".".join(parts)
            values[new_key].append(float(value))
            meta = dict(numbers.get(f"{key}@meta", {}))
            meta["source_run_id"] = run_id
            metas[new_key].append(meta)
    out: dict[str, Any] = {}
    for key, vals in values.items():
        n_runs = len(vals)
        mean = sum(vals) / n_runs
        seeds = sorted({str(m.get("seed")) for m in metas[key]})
        first = dict(metas[key][0])
        first.pop("source_run_id", None)
        first.update(
            seed=",".join(seeds),
            n_seeds=n_runs,
            ci95=[min(vals), max(vals)] if n_runs > 1 else first.get("ci95"),
            ci95_reason=(
                f"min-max over {n_runs} training seeds; per-seed bootstrap CIs in per_seed"
                if n_runs > 1
                else first.get("ci95_reason")
            ),
            per_seed=[
                {
                    "run_id": m.get("source_run_id"),
                    "seed": m.get("seed"),
                    "value": v,
                    "ci95": m.get("ci95"),
                }
                for m, v in zip(metas[key], vals, strict=True)
            ],
            model_label=label,
            aggregate="mean_over_seeds",
        )
        out[key] = mean
        out[f"{key}@meta"] = first
    return out


def run(cfg: Settings, ctx: RunContext, *, runs: str, label: str, **_: Any) -> dict[str, Any]:
    """Stage entry: ``--runs id1,id2,id3`` (eval.errors run ids) → aggregated ``numbers.json``."""
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if len(run_ids) < 2:
        raise ValueError("aggregate needs at least two eval.errors run ids")
    numbers_by_run: dict[str, dict[str, Any]] = {}
    n_frames: set[int] = set()
    for rid in run_ids:
        run_dir = Path(cfg.paths.runs_dir) / "eval.errors" / rid
        manifest = read_manifest(run_dir / "manifest.json")
        if manifest.status != "ok":
            raise ValueError(f"run {rid} has status {manifest.status}")
        ctx.add_input(run_dir / NUMBERS_FILE, "json")
        numbers_by_run[rid] = _load_numbers(run_dir)
        n_frames.add(int(manifest.extras.get("n_frames", -1)))
    if len(n_frames) > 1:
        raise ValueError(f"runs evaluate different frame counts {sorted(n_frames)}; not comparable")
    out = aggregate(numbers_by_run, label)
    path = ctx.out_dir / NUMBERS_FILE
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.add_output(path, "json")
    n_keys = sum(1 for k in out if not k.endswith("@meta"))
    ctx.log(aggregate=True, source_runs=run_ids, label=label, n_keys=n_keys)
    return {"label": label, "n_runs": len(run_ids), "n_keys": n_keys}


__all__ = ["aggregate", "run"]
