"""Vendored matbench-discovery metrics: byte-exact sources, stored values on wbm_mini, edges."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from b20mlip.evaluate import mbd_vendored as mbd

PRED = [-0.02, 0.01, -0.05, -0.001, -0.01, 0.06, 0.10, None, 0.60, -2.70]
# hand-computed for PRED against the 10 truths of tests/fixtures/wbm_mini.csv.gz (README there)
EXPECTED = {
    "F1": 0.8,
    "Precision": 0.8,
    "Recall": 0.8,
    "Accuracy": 0.8,
    "TP": 4,
    "FP": 1,
    "TN": 4,
    "FN": 1,
    "MAE": 0.024392,
    "RMSE": 0.027954,
    "R2": 0.999032,
    "DAF": 1.6,  # returned by the verbatim function; never published (SPEC section 2, gate A4)
}
TP_IDS = ["wbm-1-16845", "wbm-1-55036", "wbm-3-42176", "wbm-5-6641"]


def test_functions_are_byte_exact_and_matbench_discovery_absent() -> None:
    src = "\n\n".join(inspect.getsource(getattr(mbd, n)) for n in mbd.VERBATIM_FUNCTIONS)
    assert hashlib.sha256(src.encode()).hexdigest() == mbd.VERBATIM_SHA256
    assert mbd.UPSTREAM_SHA == "25db9e4c2cbb0d4ed72bad60e433b737b4a217cd"
    assert mbd.UPSTREAM_SHA in mbd.UPSTREAM_URL and mbd.LICENSE == "MIT"
    assert "Janosh Riebesell" in (mbd.__doc__ or "") and "MIT License" in (mbd.__doc__ or "")
    assert "matbench_discovery" not in sys.modules
    assert importlib.util.find_spec("matbench_discovery") is None
    notice = (Path(__file__).resolve().parents[2] / "NOTICE").read_text(encoding="utf-8")
    assert "matbench-discovery (vendored metrics)" in notice and mbd.UPSTREAM_SHA in notice


def test_stable_metrics_on_wbm_mini_equals_stored_values(wbm_mini: Path) -> None:
    df = pd.read_csv(wbm_mini).set_index("material_id")
    assert len(df) == 10 and df["e_above_hull_mp2020_corrected_ppd_mp"].le(0).sum() == 5
    true = df["e_above_hull_mp2020_corrected_ppd_mp"]
    pred = pd.Series(PRED, index=true.index, dtype=float)
    out = mbd.stable_metrics(true, pred)
    assert set(out) == set(EXPECTED)
    for key, value in EXPECTED.items():
        assert float(out[key]) == pytest.approx(value, abs=1e-6), key
    tp, fn, fp, tn = mbd.classify_stable(true, pred)
    assert list(true.index[tp]) == TP_IDS
    assert list(true.index[fn]) == ["wbm-1-29035"] and list(true.index[fp]) == ["wbm-2-17064"]
    assert int(tn.sum()) == 4 and "wbm-4-18301" in list(true.index[tn])  # NaN -> unstable
    # plain lists and numpy arrays are accepted too
    same = mbd.stable_metrics(true.tolist(), np.array(PRED, dtype=float))
    assert same["F1"] == out["F1"] and same["TP"] == out["TP"]
    # a shifted threshold changes the classification as expected (7 stable, 6 predicted, all TP)
    shifted = mbd.stable_metrics(true, pred, stability_threshold=0.03)
    assert shifted["Precision"] == pytest.approx(1.0) and shifted["Recall"] == pytest.approx(6 / 7)


def test_classify_stable_edge_cases() -> None:
    with pytest.raises(ValueError, match="len"):
        mbd.classify_stable([0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="real number"):
        mbd.classify_stable([0.0], [0.0], stability_threshold=float("nan"))
    tp, fn, fp, tn = mbd.classify_stable([0.0, 0.2], [None, None], fillna=True)
    assert tp.tolist() == [False, False] and fn.tolist() == [True, False]
    assert fp.tolist() == [False, False] and tn.tolist() == [False, True]
    # without fillna a NaN prediction is neither positive nor negative
    tp, fn, fp, tn = mbd.classify_stable([0.0, 0.2], [float("nan"), 0.1], fillna=False)
    assert not any(tp) and not any(fn) and not any(fp) and tn.tolist() == [False, True]
    # a negative threshold demands pulling the hull down
    tp, *_ = mbd.classify_stable([-0.05, -0.01], [-0.05, -0.01], stability_threshold=-0.02)
    assert tp.tolist() == [True, False]
    # degenerate metrics are NaN rather than errors
    out = mbd.stable_metrics([0.1, 0.2], [0.3, 0.4])
    assert np.isnan(out["Precision"]) and np.isnan(out["F1"]) and out["TN"] == 2
    assert np.isnan(mbd.stable_metrics([0.1], [0.3])["R2"])


def test_r2_score_matches_sklearn_semantics() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mbd.r2_score(y, y) == 1.0
    assert mbd.r2_score(y, np.full(3, 2.0)) == pytest.approx(0.0)
    assert mbd.r2_score(y, np.array([1.0, 2.0, 4.0])) == pytest.approx(0.5)
    assert mbd.r2_score(np.ones(3), np.ones(3)) == 1.0  # constant truth, perfect fit
    assert mbd.r2_score(np.ones(3), np.array([1.0, 1.0, 2.0])) == 0.0  # constant truth, error
    with pytest.raises(ValueError, match="shape"):
        mbd.r2_score(y, y[:2])
