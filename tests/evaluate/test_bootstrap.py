"""Group bootstrap: determinism, bracketing, degenerate cases, 2-D pools, paired alias."""

from __future__ import annotations

import numpy as np
import pytest

from b20mlip.evaluate import bootstrap


def _groups(seed: int = 0, n_groups: int = 12, per_group: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"g{i}": rng.normal(1.0 + 0.1 * (i % 12), 0.3, per_group) for i in range(n_groups)}


def test_ci_is_deterministic_and_brackets_the_mean() -> None:
    groups = _groups()
    lo, hi = bootstrap.ci(groups, 500, 3)
    assert (lo, hi) == bootstrap.ci(groups, 500, 3)
    assert bootstrap.ci(groups, 500, 4) != (lo, hi)
    mean = float(np.mean(np.concatenate(list(groups.values()))))
    assert lo < mean < hi and hi - lo < 1.0
    # more groups -> narrower interval
    lo2, hi2 = bootstrap.ci(_groups(n_groups=200), 300, 3)
    assert hi2 - lo2 < hi - lo


def test_ci_degenerate_and_rms_stat() -> None:
    assert bootstrap.ci({"a": [2.0, 2.0], "b": 2.0}, 50, 0) == (2.0, 2.0)
    lo, hi = bootstrap.ci({"a": [1.0, 1.0], "b": [3.0]}, 200, 0, stat=bootstrap.rms)
    assert 1.0 <= lo <= hi <= 3.0
    assert bootstrap.rms(np.array([3.0, -4.0])) == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError, match="at least one group"):
        bootstrap.ci({}, 10, 0)
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap.ci({"a": []}, 10, 0)
    with pytest.raises(ValueError, match="mix array ranks"):
        bootstrap.ci({"a": [1.0], "b": [[1.0, 2.0]]}, 10, 0)
    with pytest.raises(ValueError, match="n_groups"):
        bootstrap.resample_indices(0, 10, 0)
    with pytest.raises(ValueError, match="resamples"):
        bootstrap.resample_indices(3, 0, 0)


def test_resample_protocol_matches_numpy_generator() -> None:
    idx = bootstrap.resample_indices(5, 4, 11)
    expected = np.random.default_rng(11).integers(0, 5, size=(4, 5))
    np.testing.assert_array_equal(idx, expected)


def test_two_dimensional_pools_and_paired_alias() -> None:
    rows = {f"i{k}": np.array([[k, k + 1.0]]) for k in range(10)}

    def delta(pool: np.ndarray) -> float:
        return float(np.mean(pool[:, 1] - pool[:, 0]))

    assert bootstrap.ci(rows, 100, 0, delta) == (1.0, 1.0)
    assert bootstrap.ci_paired(rows, 100, 0, delta) == bootstrap.ci(rows, 100, 0, delta)
    deltas = {f"i{k}": [k - 4.5] for k in range(10)}
    lo, hi = bootstrap.ci_paired(deltas, 300, 1)
    assert lo < 0.0 < hi


def test_group_values_concatenates_per_group() -> None:
    pools = bootstrap.group_values([("a", [1.0, 2.0]), ("b", 3.0), ("a", [4.0])])
    assert pools["a"].tolist() == [1.0, 2.0, 4.0] and pools["b"].tolist() == [3.0]
    rows = bootstrap.group_values([("a", np.ones((2, 3))), ("a", np.zeros((1, 3)))])
    assert rows["a"].shape == (3, 3)
