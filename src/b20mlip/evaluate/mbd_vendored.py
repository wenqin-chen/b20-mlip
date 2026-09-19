"""Discovery metrics vendored VERBATIM from janosh/matbench-discovery (MIT), pinned commit.

Provenance (binding; ``tests/evaluate/test_mbd_vendored.py`` re-hashes the function sources)
==========================================================================================

* Repository: https://github.com/janosh/matbench-discovery
* Commit:     25db9e4c2cbb0d4ed72bad60e433b737b4a217cd  (2026-08-10, "Share runner/eval
              scaffolding and document every module on the API page (#388)")
* File:       matbench_discovery/metrics/discovery.py
* Raw URL:    https://raw.githubusercontent.com/janosh/matbench-discovery/25db9e4c2cbb0d4ed72bad60e433b737b4a217cd/matbench_discovery/metrics/discovery.py
* Retrieved:  2026-09-18 with curl; sha256 of the retrieved file
              88cfb73b80a6abca05020a17751de3b7a6efa6b56257fc7e637e61644a3721f4
* Functions copied byte-for-byte (in this order): ``classify_stable``, ``_safe_div``,
  ``stable_metrics``. The sha256 of their concatenated sources (joined by two newlines) is
  :data:`VERBATIM_SHA256`; the test recomputes it from ``inspect.getsource``.
* Licence: MIT, Copyright (c) 2022 Janosh Riebesell (text below, also in NOTICE).

What is NOT copied: the upstream module prologue. ``matbench_discovery`` is never imported
or installed in this project (it needs Python 3.14 and downloads data at import time;
CONTRACTS.md header), so its three module-level names used by the functions are provided
here instead:

* ``STABILITY_THRESHOLD = 0`` — ``matbench_discovery/__init__.py`` line 57 at the same
  commit (sha256 5d9610b2c197ad739f67df3dcd1cdc4533102deaeed6934fd99b3fa40cd064c7);
* ``r2_score`` — upstream imports ``sklearn.metrics.r2_score``; scikit-learn is not a
  dependency of this project (CONTRACTS.md pins), so :func:`r2_score` below is a NumPy
  re-implementation of sklearn's single-output default (``multioutput="uniform_average"``,
  ``force_finite=True``: a constant ``y_true`` gives 1.0 for a perfect prediction, else 0.0);
* ``pd``, ``np``, ``Sequence`` — the same third-party imports as upstream.

The other upstream functions (``prepare_model_predictions``, ``calc_discovery_metrics``,
``write_all_metrics_to_yaml``, ``wbm_uniq_proto_prevalence``) depend on the full WBM data
set and the leaderboard machinery and are deliberately NOT vendored (SPEC.md section 2: no
leaderboard submission, no DAF/CPS/kappa_SRME claims). ``stable_metrics`` still returns a
``DAF`` entry because the function is verbatim; :mod:`b20mlip.evaluate.discovery` drops it
before anything is published (gate A4 forbids the token in ``numbers.json``).

MIT License

Copyright (c) 2022 Janosh Riebesell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.
"""

# ruff: noqa: E501 - upstream line lengths are kept so the copy stays byte-exact

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

UPSTREAM_REPO = "https://github.com/janosh/matbench-discovery"
UPSTREAM_SHA = "25db9e4c2cbb0d4ed72bad60e433b737b4a217cd"
UPSTREAM_PATH = "matbench_discovery/metrics/discovery.py"
UPSTREAM_URL = (
    f"https://raw.githubusercontent.com/janosh/matbench-discovery/{UPSTREAM_SHA}/{UPSTREAM_PATH}"
)
UPSTREAM_FILE_SHA256 = "88cfb73b80a6abca05020a17751de3b7a6efa6b56257fc7e637e61644a3721f4"
RETRIEVED = "2026-09-18"
LICENSE = "MIT"
COPYRIGHT = "Copyright (c) 2022 Janosh Riebesell"
VERBATIM_FUNCTIONS: tuple[str, ...] = ("classify_stable", "_safe_div", "stable_metrics")
VERBATIM_SHA256 = "aac086e04db2e35b51bf2ef14bba206e12c97295362a19902e97e84529ae969e"

# matbench_discovery/__init__.py at the pinned commit: ``STABILITY_THRESHOLD = 0``
STABILITY_THRESHOLD = 0


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``sklearn.metrics.r2_score`` for one output (``force_finite=True`` semantics).

    ``1 - SS_res / SS_tot``; when ``y_true`` is constant (``SS_tot == 0``) sklearn returns
    1.0 for a perfect prediction and 0.0 otherwise instead of NaN/-inf.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} != {y_pred.shape}")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# --- verbatim from matbench_discovery/metrics/discovery.py @ 25db9e4 (do not edit) -------------


def classify_stable(
    each_true: Sequence[float | None] | pd.Series | np.ndarray,
    each_pred: Sequence[float | None] | pd.Series | np.ndarray,
    *,
    stability_threshold: float = STABILITY_THRESHOLD,
    fillna: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Classify model stability predictions as true/false positive/negatives (usually
    w.r.t DFT-ground truth labels). All energies are assumed to be in eV/atom
    (but shouldn't really matter as long as they're consistent).

    Args:
        each_true (Sequence[float] | pd.Series): Ground truth energy above convex hull
            values.
        each_pred (Sequence[float] | pd.Series): Model-predicted energy above convex
            hull values.
        stability_threshold (float, optional): Maximum energy above convex hull
            for a material to still be considered stable. Usually 0, 0.05 or 0.1.
            Defaults to STABILITY_THRESHOLD, meaning a material has to be directly on
            the hull to be called stable. Negative values mean a material has to pull
            the known hull down by that amount to count as stable. Few materials lie
            below the known hull, so only negative values very close to 0 make sense.
        fillna (bool): Whether to fill NaNs as the model predicting unstable. Defaults
            to True.

    Returns:
        tuple[TP, FN, FP, TN]: Indices as pd.Series for true positives,
            false negatives, false positives and true negatives (in this order).

    Raises:
        ValueError: If sum of positive + negative preds doesn't add up to the total.
    """
    if len(each_true) != len(each_pred):
        raise ValueError(f"{len(each_true)=} != {len(each_pred)=}")

    each_true_arr = pd.to_numeric(pd.Series(each_true), errors="coerce")
    each_pred_arr = pd.to_numeric(pd.Series(each_pred), errors="coerce")

    if stability_threshold is None or np.isnan(stability_threshold):
        raise ValueError("stability_threshold must be a real number")
    actual_pos = each_true_arr <= stability_threshold
    actual_neg = each_true_arr > stability_threshold

    model_pos = each_pred_arr <= stability_threshold
    model_neg = each_pred_arr > stability_threshold

    if fillna:
        nan_mask = each_pred_arr.isna()
        # for in both the model's stable and unstable preds, fill NaNs as unstable
        model_pos[nan_mask] = False
        model_neg[nan_mask] = True

        n_pos, n_neg, total = model_pos.sum(), model_neg.sum(), len(each_pred)
        if n_pos + n_neg != total:
            raise ValueError(
                f"after filling NaNs, the sum of positive ({n_pos}) and negative "
                f"({n_neg}) predictions should add up to {total=}"
            )

    true_pos = actual_pos & model_pos
    false_neg = actual_pos & model_neg
    false_pos = actual_neg & model_pos
    true_neg = actual_neg & model_neg

    return true_pos, false_neg, false_pos, true_neg


def _safe_div(numerator: float, denominator: float) -> float:
    """Ratio of two counts, NaN when the denominator is zero (or NaN)."""
    return numerator / denominator if denominator > 0 else float("nan")


def stable_metrics(
    each_true: Sequence[float | None] | pd.Series | np.ndarray,
    each_pred: Sequence[float | None] | pd.Series | np.ndarray,
    *,
    stability_threshold: float = STABILITY_THRESHOLD,
    fillna: bool = True,
) -> dict[str, float]:
    """Get a dictionary of stability prediction metrics. Mostly binary classification
    metrics, but also MAE, RMSE and R2.

    Args:
        each_true (Sequence[float | None] | pd.Series): true energy above convex hull
        each_pred (Sequence[float | None] | pd.Series): predicted energy above convex
            hull
        stability_threshold (float): Where to place stability threshold relative to
            convex hull in eV/atom, usually 0 or 0.1 eV. Default = STABILITY_THRESHOLD.
        fillna (bool): Whether to fill NaNs as the model predicting unstable. Defaults
            to True.

    Note: Should give equivalent classification metrics to
        sklearn.metrics.classification_report(
            each_true > stability_threshold,
            each_pred > stability_threshold,
            output_dict=True,
        )
        when using the same stability_threshold.

    Returns:
        dict[str, float]: dictionary of classification metrics with keys DAF, Precision,
            Recall, Accuracy, F1, TP, FP, TN, FN, MAE, RMSE, R2.
    """
    n_true_pos, n_false_neg, n_false_pos, n_true_neg = map(
        sum,
        classify_stable(
            each_true, each_pred, stability_threshold=stability_threshold, fillna=fillna
        ),
    )

    n_total_pos = n_true_pos + n_false_neg
    n_total_neg = n_true_neg + n_false_pos
    n_total = n_total_pos + n_total_neg
    # prevalence: dummy discovery rate of stable crystals by selecting randomly from
    # all materials
    prevalence = _safe_div(n_total_pos, n_total)
    precision = _safe_div(n_true_pos, n_true_pos + n_false_pos)
    recall = _safe_div(n_true_pos, n_total_pos)

    # Drop NaNs to calculate regression metrics
    each_true_arr = pd.to_numeric(pd.Series(each_true), errors="coerce")
    each_pred_arr = pd.to_numeric(pd.Series(each_pred), errors="coerce")
    is_nan = each_true_arr.isna() | each_pred_arr.isna()
    each_true = each_true_arr[~is_nan].to_numpy()
    each_pred = each_pred_arr[~is_nan].to_numpy()

    return dict(
        F1=_safe_div(2 * precision * recall, precision + recall),
        DAF=_safe_div(precision, prevalence),
        Precision=precision,
        Recall=recall,
        Accuracy=_safe_div(n_true_pos + n_true_neg, n_total),
        TP=n_true_pos,
        FP=n_false_pos,
        TN=n_true_neg,
        FN=n_false_neg,
        MAE=np.abs(each_true - each_pred).mean(),
        RMSE=((each_true - each_pred) ** 2).mean() ** 0.5,
        R2=r2_score(each_true, each_pred) if len(each_true) > 1 else float("nan"),
    )


# --- end of verbatim block ------------------------------------------------------------------------

__all__ = [
    "COPYRIGHT",
    "LICENSE",
    "RETRIEVED",
    "STABILITY_THRESHOLD",
    "UPSTREAM_FILE_SHA256",
    "UPSTREAM_PATH",
    "UPSTREAM_REPO",
    "UPSTREAM_SHA",
    "UPSTREAM_URL",
    "VERBATIM_FUNCTIONS",
    "VERBATIM_SHA256",
    "classify_stable",
    "r2_score",
    "stable_metrics",
]
