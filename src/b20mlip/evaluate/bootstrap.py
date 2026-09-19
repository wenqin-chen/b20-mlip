"""Bootstrap confidence intervals over groups (SPEC.md section 6: 2,000 resamples, seeded).

Protocol (binding for every CI published by the evaluate tier):

* the unit of resampling is a **group** (``Frame.group_id`` for error tables, the WBM id for
  discovery metrics), never the individual value, so correlated derivatives of one parent
  (rattled copies, MD snapshots) are drawn together;
* one resample draws ``G`` groups with replacement from the ``G`` observed groups
  (``numpy.random.default_rng(seed).integers(0, G, G)``), pools their values and applies
  ``stat`` to the pool; the interval is the 2.5 / 97.5 percentile of the ``n`` statistics
  (``numpy.percentile``, linear interpolation);
* the same ``seed`` gives the same interval bit for bit; ``stat`` may see a 1-D pool
  (means, RMS) or a 2-D pool whose rows are per-item tuples (paired statistics such as
  F1 differences: see :func:`ci_paired` and ``discovery.paired_delta_f1``).

A single group gives a degenerate interval (the callers publish ``ci95: null`` with a
reason in that case instead of pretending to have an uncertainty).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_N = 2000
DEFAULT_ALPHA = 0.05
Stat = Callable[[np.ndarray], float]


def _arrays(values_by_group: Mapping[str, Any]) -> list[np.ndarray]:
    if not values_by_group:
        raise ValueError("bootstrap needs at least one group")
    arrays: list[np.ndarray] = []
    for key in sorted(values_by_group, key=str):
        arr = np.asarray(values_by_group[key], dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[0] == 0:
            continue
        arrays.append(arr)
    if not arrays:
        raise ValueError("bootstrap needs at least one non-empty group")
    ndims = {a.ndim for a in arrays}
    if len(ndims) != 1:
        raise ValueError(f"groups mix array ranks {sorted(ndims)}")
    return arrays


def resample_indices(n_groups: int, n: int, seed: int) -> np.ndarray:
    """``(n, n_groups)`` group indices drawn with replacement; the seeded protocol above."""
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if n < 1:
        raise ValueError("n (resamples) must be >= 1")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n_groups, size=(int(n), n_groups))


def ci(
    values_by_group: Mapping[str, Any],
    n: int = DEFAULT_N,
    seed: int = 0,
    stat: Stat = np.mean,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """95 % (``alpha=0.05``) percentile bootstrap interval of ``stat`` over pooled groups.

    ``values_by_group`` maps a group id to its values (a scalar, a 1-D array of values or a
    2-D array of per-item rows). Groups are resampled with replacement, their values pooled
    (``numpy.concatenate`` along axis 0) and ``stat(pool)`` evaluated ``n`` times.
    """
    arrays = _arrays(values_by_group)
    indices = resample_indices(len(arrays), n, seed)
    stats = np.empty(int(n), dtype=float)
    for b, idx in enumerate(indices):
        pool = np.concatenate([arrays[i] for i in idx], axis=0)
        stats[b] = float(stat(pool))
    lo, hi = np.percentile(stats, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def ci_paired(
    delta_by_group: Mapping[str, Any],
    n: int = DEFAULT_N,
    seed: int = 0,
    stat: Stat = np.mean,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Interval of a paired statistic: ``delta_by_group`` holds per-group *differences*
    (model B minus model A on identical items), or per-item rows ``[truth, a, b]`` with a
    ``stat`` that computes the difference on the pooled rows (``discovery.paired_delta_f1``).
    Both models see exactly the same resampled groups, which is what makes it paired.
    """
    return ci(delta_by_group, n, seed, stat, alpha=alpha)


def rms(values: np.ndarray) -> float:
    """Root mean square ``sqrt(mean(values²))`` of a pool of raw residuals (an RMSE statistic)."""
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr**2)))


def group_values(items: Sequence[tuple[str, Any]]) -> dict[str, np.ndarray]:
    """``[(group, values), ...]`` -> ``{group: concatenated values}`` (helper for callers)."""
    out: dict[str, list[np.ndarray]] = {}
    for group, values in items:
        arr = np.asarray(values, dtype=float)
        out.setdefault(group, []).append(arr.reshape(-1) if arr.ndim <= 1 else arr)
    return {g: np.concatenate(v, axis=0) for g, v in out.items()}


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_N",
    "ci",
    "ci_paired",
    "group_values",
    "resample_indices",
    "rms",
]
