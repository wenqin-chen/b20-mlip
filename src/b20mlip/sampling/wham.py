"""Free-energy profile from umbrella windows: MBAR (primary) and plain WHAM (cross-check).

``free_energy(windows, k, T, method)`` reads the window series (``umbrella.run_windows``
files, an umbrella run dir, or in-memory dicts), discards each window's equilibration part
(``n_equil`` samples, 20 % by default) and estimates the unbiased profile ``F(cv)`` on a
histogram grid:

* ``method="mbar"``: ``pymbar.MBAR`` on the reduced bias potentials of every sample under every
  window, ``u_kn = k_k (x_n - c_k)^2 / (2 kT)``; the unbiased sample weights
  ``w_n = 1 / sum_k N_k exp(f_k - u_kn)`` give ``F(bin) = -kT ln sum_{n in bin} w_n``
  (identical to ``pymbar.FES(..., fes_type="histogram")`` on populated bins).
* ``method="wham"``: the classic self-consistent histogram iteration
  ``p(b) = sum_k n_k(b) / sum_k N_k exp(f_k - U_k(b)/kT)``, ``f_k = -ln sum_b p(b) exp(-U_k(b)/kT)``
  with the bias evaluated at the bin centres (small, transparent, a cross-check of MBAR).

Basins: the initial basin is the profile minimum on the low-CV half, the final basin the
minimum on the high-CV half; the transition state is the maximum between them. Basin free
energies integrate the profile (``F_A = -kT ln sum_{b in A} exp(-F_b/kT)``), so ``dF_eV =
F(final basin) - F(initial basin)`` is the well-to-well free-energy difference and
``dF_barrier_eV = F(ts) - F(initial basin)``; ``dF_barrier_point_eV = F(ts) - F(initial min)`` is
the point-to-point barrier closest to the 0 K NEB barrier. Block errors split every window's
post-equilibration series into ``n_blocks`` (5) contiguous blocks, re-estimate with the basin
divider fixed and report ``std / sqrt(n_blocks)``; ``ci95`` uses Student's t.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from scipy.special import logsumexp
from scipy.stats import t as student_t

from b20mlip.config import Settings
from b20mlip.models import StageResult
from b20mlip.provenance import RunContext
from b20mlip.sampling import umbrella
from b20mlip.sampling._common import (
    ModelInfo,
    json_safe,
    model_info,
    numbers_meta,
    write_json,
    write_numbers,
)

METHODS = ("mbar", "wham")
DEFAULT_BLOCKS = 5
DEFAULT_MIN_OVERLAP = 0.05
DEFAULT_MIN_BINS = 50
DEFAULT_BINS_PER_WINDOW = 5
WHAM_TOL = 1e-8
WHAM_MAXITER = 20000
PMF_JSON = "pmf.json"
KB_EV = units.kB


@dataclass(frozen=True)
class Window:
    """One umbrella window: bias centre, constant (eV per CV unit²), T (K) and its series."""

    center: float
    k: float
    T: float
    cv: np.ndarray  # post-equilibration samples
    cv_all: np.ndarray
    n_equil: int
    path: str | None = None

    @property
    def n(self) -> int:
        return int(len(self.cv))


# --- input ------------------------------------------------------------------------------------


def _window_from_dict(data: Mapping[str, Any], equil_fraction: float | None, path: str | None):
    series = np.asarray(data["cv"], dtype=float).ravel()
    if equil_fraction is not None:
        n_equil = int(round(float(equil_fraction) * len(series)))
    else:
        n_equil = int(data.get("n_equil", round(umbrella.DEFAULT_EQUIL_FRACTION * len(series))))
    n_equil = min(max(n_equil, 0), max(len(series) - 1, 0))
    return Window(
        center=float(data["center"]),
        k=float(data.get("k", np.nan)),
        T=float(data.get("T", np.nan)),
        cv=series[n_equil:],
        cv_all=series,
        n_equil=n_equil,
        path=path,
    )


def as_windows(
    windows: Any,
    *,
    k: float | None = None,
    T: float | None = None,
    equil_fraction: float | None = None,
) -> list[Window]:
    """Normalise every accepted ``windows`` argument to a list of :class:`Window`.

    Accepted: an umbrella run dir or its ``windows.json``; a sequence of ``window_<i>.npz``
    paths; a sequence of dicts with ``cv``, ``center`` (and optionally ``k``, ``T``,
    ``n_equil``); a sequence of :class:`Window`. ``k``/``T`` override the stored values.
    """
    items: list[Window] = []
    if isinstance(windows, (str, Path)):
        index, _ = umbrella.read_index(windows)
        files = index["files"]
        k = index.get("k", k) if k is None else k
        T = index.get("T", T) if T is None else T
        equil_fraction = index.get("equil_fraction") if equil_fraction is None else equil_fraction
        return as_windows(files, k=k, T=T, equil_fraction=equil_fraction)
    for item in windows:
        if isinstance(item, Window):
            win = (
                item
                if equil_fraction is None
                else _window_from_dict(
                    {"cv": item.cv_all, "center": item.center, "k": item.k, "T": item.T},
                    equil_fraction,
                    item.path,
                )
            )
        elif isinstance(item, (str, Path)):
            win = _window_from_dict(umbrella.load_window(item), equil_fraction, str(item))
        elif isinstance(item, Mapping):
            win = _window_from_dict(item, equil_fraction, None)
        else:
            raise TypeError(f"unsupported window item {type(item).__name__}")
        if k is not None:
            win = replace(win, k=float(k))
        if T is not None:
            win = replace(win, T=float(T))
        items.append(win)
    if not items:
        raise ValueError("no umbrella windows")
    for win in items:
        if not math.isfinite(win.k) or win.k <= 0:
            raise ValueError(
                f"window at {win.center}: bias constant k must be positive, got {win.k}"
            )
        if not math.isfinite(win.T) or win.T <= 0:
            raise ValueError(f"window at {win.center}: temperature must be positive, got {win.T}")
        if win.n < 2:
            raise ValueError(f"window at {win.center}: fewer than 2 post-equilibration samples")
    temperatures = np.array([w.T for w in items])
    if np.ptp(temperatures) > 1e-6 * temperatures.max():
        raise ValueError(f"windows at different temperatures: {sorted(set(temperatures))}")
    return sorted(items, key=lambda w: w.center)


# --- grids and profiles ---------------------------------------------------------------------------


def make_edges(
    windows: Sequence[Window],
    n_bins: int | None = None,
    cv_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Bin edges spanning the samples (or ``cv_range``); ``n_bins`` defaults to ``max(50, 5 K)``."""
    if n_bins is None:
        n_bins = max(DEFAULT_MIN_BINS, DEFAULT_BINS_PER_WINDOW * len(windows))
    if cv_range is None:
        lo = min(float(w.cv.min()) for w in windows)
        hi = max(float(w.cv.max()) for w in windows)
        pad = 1e-9 * max(abs(lo), abs(hi), 1.0)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = float(cv_range[0]), float(cv_range[1])
    if hi <= lo:
        raise ValueError(f"empty CV range [{lo}, {hi}]")
    return np.linspace(lo, hi, int(n_bins) + 1)


def _stack(windows: Sequence[Window]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_n = np.concatenate([w.cv for w in windows])
    N_k = np.array([w.n for w in windows], dtype=int)
    centers = np.array([w.center for w in windows], dtype=float)
    ks = np.array([w.k for w in windows], dtype=float)
    return x_n, N_k, centers, ks


def _bin_profile(x_n: np.ndarray, log_w: np.ndarray, edges: np.ndarray, kT: float) -> np.ndarray:
    """``F(bin) = -kT ln sum_{n in bin} w_n`` (NaN for empty bins), from log-weights."""
    n_bins = len(edges) - 1
    idx = np.digitize(x_n, edges) - 1
    inside = (idx >= 0) & (idx < n_bins)
    order = np.argsort(idx[inside], kind="stable")
    idx_sorted = idx[inside][order]
    lw_sorted = log_w[inside][order]
    F = np.full(n_bins, np.nan)
    if idx_sorted.size:
        bins, starts = np.unique(idx_sorted, return_index=True)
        stops = np.append(starts[1:], idx_sorted.size)
        for b, a, z in zip(bins, starts, stops, strict=True):
            F[b] = -kT * logsumexp(lw_sorted[a:z])
    return F


def mbar_profile(
    windows: Sequence[Window],
    edges: np.ndarray,
    kT: float,
    *,
    initial_f_k: np.ndarray | None = None,
    mbar_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """MBAR reduced free energies ``f_k`` of the biased windows and the unbiased histogram PMF."""
    from pymbar import MBAR

    x_n, N_k, centers, ks = _stack(windows)
    u_kn = 0.5 * ks[:, None] * (x_n[None, :] - centers[:, None]) ** 2 / kT
    kwargs: dict[str, Any] = {"relative_tolerance": 1e-10, "maximum_iterations": 10000}
    kwargs.update(mbar_kwargs or {})
    if initial_f_k is not None and len(initial_f_k) == len(windows):
        kwargs["initial_f_k"] = np.asarray(initial_f_k, dtype=float)
    mbar = MBAR(u_kn, N_k, **kwargs)
    f_k = np.asarray(mbar.f_k, dtype=float)
    log_w = -logsumexp(np.log(N_k)[:, None] + f_k[:, None] - u_kn, axis=0)
    log_w -= logsumexp(log_w)
    F = _bin_profile(x_n, log_w, edges, kT)
    return {"F": F - np.nanmin(F), "f_k": f_k, "n_samples": int(N_k.sum()), "converged": True}


def wham_profile(
    windows: Sequence[Window],
    edges: np.ndarray,
    kT: float,
    *,
    tol: float = WHAM_TOL,
    maxiter: int = WHAM_MAXITER,
    initial_f_k: np.ndarray | None = None,
) -> dict[str, Any]:
    """Classic iterative WHAM on the histogram grid (bias evaluated at bin centres)."""
    _, N_k, centers, ks = _stack(windows)
    cent = 0.5 * (edges[1:] + edges[:-1])
    H = np.array([np.histogram(w.cv, bins=edges)[0] for w in windows], dtype=float)
    n_b = H.sum(axis=0)
    populated = n_b > 0
    u_kb = 0.5 * ks[:, None] * (cent[None, :] - centers[:, None]) ** 2 / kT
    log_N = np.log(N_k.astype(float))
    log_n_b = np.full(len(cent), -np.inf)
    log_n_b[populated] = np.log(n_b[populated])
    f_k = (
        np.asarray(initial_f_k, dtype=float).copy()
        if initial_f_k is not None and len(initial_f_k) == len(windows)
        else np.zeros(len(windows))
    )
    converged = False
    iterations = 0
    while iterations < int(maxiter) and not converged:
        iterations += 1
        log_p = log_n_b - logsumexp(log_N[:, None] + f_k[:, None] - u_kb, axis=0)
        f_new = -logsumexp(log_p[None, :] - u_kb, axis=1)
        f_new -= f_new[0]
        converged = float(np.max(np.abs(f_new - f_k))) < tol
        f_k = f_new
    log_p = log_n_b - logsumexp(log_N[:, None] + f_k[:, None] - u_kb, axis=0)
    log_p -= logsumexp(log_p[populated])
    F = np.full(len(cent), np.nan)
    F[populated] = -kT * log_p[populated]
    return {
        "F": F - np.nanmin(F),
        "f_k": f_k,
        "n_samples": int(N_k.sum()),
        "converged": converged,
        "iterations": iterations,
    }


def profile(
    windows: Sequence[Window], edges: np.ndarray, kT: float, method: str = "mbar", **kw: Any
) -> dict[str, Any]:
    if method == "mbar":
        return mbar_profile(windows, edges, kT, **kw)
    if method == "wham":
        return wham_profile(windows, edges, kT, **kw)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


# --- analysis -------------------------------------------------------------------------------------


def basins(
    grid: np.ndarray,
    F: np.ndarray,
    kT: float,
    lo: float,
    hi: float,
    *,
    split: float | None = None,
) -> dict[str, Any]:
    """Well-to-well ``dF``, the barrier and the basin/transition-state positions of a profile."""
    valid = np.isfinite(F)
    mid = 0.5 * (lo + hi)
    left = valid & (grid <= mid)
    right = valid & (grid >= mid)
    if not left.any() or not right.any():
        raise ValueError("the profile does not cover both basins (no samples on one side)")
    i0 = int(np.flatnonzero(left)[np.argmin(F[left])])
    i1 = int(np.flatnonzero(right)[np.argmin(F[right])])
    barrier_found = True
    if split is None:
        between = valid & (grid >= grid[i0]) & (grid <= grid[i1])
        ts = int(np.flatnonzero(between)[np.argmax(F[between])])
        if ts in (i0, i1):  # monotonic profile: no barrier, divide the range in the middle
            barrier_found = False
            split_cv = mid
            ts = int(np.flatnonzero(valid)[np.argmin(np.abs(grid[valid] - mid))])
        else:
            split_cv = float(grid[ts])
    else:
        split_cv = float(split)
        ts = int(np.flatnonzero(valid)[np.argmin(np.abs(grid[valid] - split_cv))])
    A = valid & (grid < split_cv)
    B = valid & (grid > split_cv)
    if not A.any() or not B.any():
        raise ValueError("the basin divider leaves one basin without samples")
    F_A = float(-kT * logsumexp(-F[A] / kT))
    F_B = float(-kT * logsumexp(-F[B] / kT))
    return {
        "dF_eV": F_B - F_A,
        "F_initial_eV": F_A,
        "F_final_eV": F_B,
        "cv_initial": float(grid[i0]),
        "cv_final": float(grid[i1]),
        "cv_ts": split_cv,
        "F_ts_eV": float(F[ts]),
        "dF_barrier_eV": float(F[ts] - F_A),
        "dF_barrier_reverse_eV": float(F[ts] - F_B),
        "dF_barrier_point_eV": float(F[ts] - F[i0]),
        "dF_min_eV": float(F[i1] - F[i0]),
        "barrier_found": barrier_found,
    }


def adjacent_overlaps(windows: Sequence[Window], edges: np.ndarray) -> list[float]:
    """Histogram overlap ``sum_b min(h_i, h_i+1)`` of neighbouring windows (sorted by centre)."""
    hists = [np.histogram(w.cv, bins=edges)[0] / max(w.n, 1) for w in windows]
    return [float(np.minimum(a, b).sum()) for a, b in zip(hists[:-1], hists[1:], strict=True)]


def _blocks(windows: Sequence[Window], n_blocks: int) -> list[list[Window]]:
    out: list[list[Window]] = []
    for b in range(n_blocks):
        block: list[Window] = []
        for w in windows:
            length = w.n // n_blocks
            if length < 2:
                raise ValueError(
                    f"window at {w.center}: {w.n} samples are too few for {n_blocks} blocks"
                )
            block.append(replace(w, cv=w.cv[b * length : (b + 1) * length]))
        out.append(block)
    return out


def free_energy(
    windows: Any,
    k: float | None = None,
    T: float | None = None,
    method: str = "mbar",
    *,
    n_bins: int | None = None,
    bin_edges: Sequence[float] | None = None,
    cv_range: tuple[float, float] | None = None,
    n_blocks: int = DEFAULT_BLOCKS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    equil_fraction: float | None = None,
    mbar_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Unbiased profile, well-to-well ``dF``, barrier and block error from umbrella windows.

    ``k`` (eV per CV unit²) and ``T`` (K) override what the windows carry. See the module
    docstring for the estimators and the basin definitions.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    t0 = time.perf_counter()
    wins = as_windows(windows, k=k, T=T, equil_fraction=equil_fraction)
    kT = KB_EV * wins[0].T
    edges = (
        np.asarray(bin_edges, dtype=float)
        if bin_edges is not None
        else make_edges(wins, n_bins, cv_range)
    )
    grid = 0.5 * (edges[1:] + edges[:-1])
    centers = [w.center for w in wins]
    lo, hi = min(centers), max(centers)

    full = (
        profile(wins, edges, kT, method, mbar_kwargs=mbar_kwargs)
        if method == "mbar"
        else profile(wins, edges, kT, method)
    )
    F = full["F"]
    analysis = basins(grid, F, kT, lo, hi)

    block_kw: dict[str, Any] = {"initial_f_k": full["f_k"]}
    if method == "mbar":
        block_kw["mbar_kwargs"] = mbar_kwargs
    dF_blocks: list[float] = []
    barrier_blocks: list[float] = []
    for block in _blocks(wins, n_blocks):
        try:
            prof = profile(block, edges, kT, method, **block_kw)
            part = basins(grid, prof["F"], kT, lo, hi, split=analysis["cv_ts"])
        except ValueError:
            continue
        dF_blocks.append(float(part["dF_eV"]))
        barrier_blocks.append(float(part["dF_barrier_eV"]))
    n_valid = len(dF_blocks)
    if n_valid >= 2:
        err = float(np.std(dF_blocks, ddof=1) / math.sqrt(n_valid))
        barrier_err = float(np.std(barrier_blocks, ddof=1) / math.sqrt(n_valid))
        t_factor = float(student_t.ppf(0.975, n_valid - 1))
    else:
        err = barrier_err = float("nan")
        t_factor = float("nan")
    overlaps = adjacent_overlaps(wins, edges)
    overlap_min = min(overlaps) if overlaps else 1.0
    dF = float(analysis["dF_eV"])
    return {
        **analysis,
        "dF_block_err_eV": err,
        "dF_ci95_eV": [dF - t_factor * err, dF + t_factor * err] if n_valid >= 2 else None,
        "dF_blocks_eV": dF_blocks,
        "dF_barrier_block_err_eV": barrier_err,
        "dF_barrier_blocks_eV": barrier_blocks,
        "n_blocks": int(n_blocks),
        "n_blocks_valid": n_valid,
        "t_factor_95": t_factor,
        "cv_grid": grid,
        "F_eV": F,
        "bin_edges": edges,
        "n_bins": int(len(grid)),
        "f_k": full["f_k"],
        "n_windows": len(wins),
        "n_samples": int(full["n_samples"]),
        "n_samples_per_window": [w.n for w in wins],
        "n_equil_per_window": [w.n_equil for w in wins],
        "centers": centers,
        "k": [w.k for w in wins],
        "T": float(wins[0].T),
        "kT_eV": float(kT),
        "method": method,
        "estimator_converged": bool(full["converged"]),
        "iterations": full.get("iterations"),
        "overlaps": overlaps,
        "overlap_min": float(overlap_min),
        "overlap_ok": bool(overlap_min >= min_overlap),
        "min_overlap": float(min_overlap),
        "wall_seconds": time.perf_counter() - t0,
    }


# --- stage ----------------------------------------------------------------------------------------


def _model_from_index(index: Mapping[str, Any]) -> ModelInfo:
    stored = index.get("model_info")
    if isinstance(stored, Mapping):
        return ModelInfo(**{k: stored[k] for k in ModelInfo.__dataclass_fields__ if k in stored})
    model = index.get("model")
    if model and Path(str(model)).is_file():
        return model_info(str(model))
    return ModelInfo(
        path=str(model or ""),
        sha256="",
        label="unknown",
        e0_source="unknown",
        energy_scale="none",
        heads=["Default"],
        variant=None,
        checkpoint=None,
    )


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    run_dir: str | Path,
    method: str = "mbar",
    n_blocks: int = DEFAULT_BLOCKS,
    n_bins: int | None = None,
    T: float | None = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> StageResult:
    """Stage ``sampling.wham``: an umbrella run dir -> ``pmf.json`` + ``numbers.json``."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    index, index_path = umbrella.read_index(run_dir)
    files: list[str] = list(index["files"])
    if not files:
        raise ValueError(f"{index_path}: no completed windows")
    info = _model_from_index(index)
    compound = str(index.get("compound") or "unknown")
    head = str(index.get("head") or "Default")
    seed = ctx.seed if ctx.seed is not None else int(index.get("seed", 0))
    temperature = float(index["T"] if T is None else T)
    other = "wham" if method == "mbar" else "mbar"
    ctx.log(
        model_sha256=info.sha256,
        model_label=info.label,
        head=head,
        compound=compound,
        method=method,
        crosscheck_method=other,
        umbrella_run=str(index_path.parent),
        umbrella_run_id=index.get("run_id"),
        T=temperature,
        n_windows=len(files),
        n_blocks=int(n_blocks),
    )
    if ctx.dry_run:
        return StageResult(
            stage="sampling.wham",
            run_id=ctx.run_id,
            manifest_path=str(ctx.manifest_path),
            status="partial",
            outputs=[],
            summary={"planned": 1, "n_windows": len(files), "method": method, "compound": compound},
        )
    ctx.add_input(index_path, "json")
    for f in files:
        ctx.add_input(f, "other")

    result = free_energy(
        files, k=index.get("k"), T=temperature, method=method, n_bins=n_bins, n_blocks=n_blocks,
        min_overlap=min_overlap, equil_fraction=index.get("equil_fraction"),
    )  # fmt: skip
    cross = free_energy(
        files, k=index.get("k"), T=temperature, method=other, n_bins=n_bins, n_blocks=n_blocks,
        min_overlap=min_overlap, equil_fraction=index.get("equil_fraction"),
    )  # fmt: skip
    crosscheck = {
        "method": other,
        "dF_eV": cross["dF_eV"],
        "dF_block_err_eV": cross["dF_block_err_eV"],
        "dF_barrier_eV": cross["dF_barrier_eV"],
        "F_eV": cross["F_eV"],
        "dF_difference_eV": float(result["dF_eV"] - cross["dF_eV"]),
    }
    document = {
        **result,
        "crosscheck": crosscheck,
        "compound": compound,
        "model_info": info.as_dict(),
        "head": head,
        "seed": seed,
        "umbrella_index": {k: v for k, v in index.items() if k != "files"},
        "umbrella_files": files,
        "k_eVA2": index.get("k_eVA2"),
        "cv_scale_A": index.get("cv_scale_A"),
    }
    write_json(ctx.out_dir / PMF_JSON, document)
    ctx.add_output(ctx.out_dir / PMF_JSON, "json")

    prefix = f"sampling.wham.{compound}.{info.label}"
    n_samples = int(result["n_samples"])
    common = dict(
        head=head,
        n=n_samples,
        seed=seed,
        T=temperature,
        method=method,
        unit="eV",
        compound=compound,
        n_windows=int(result["n_windows"]),
        n_blocks=int(result["n_blocks_valid"]),
        overlap_ok=bool(result["overlap_ok"]),
        overlap_min=float(result["overlap_min"]),
        crosscheck_method=other,
        crosscheck_dF_eV=float(cross["dF_eV"]),
        k_eVA2=index.get("k_eVA2"),
        ps_per_window=index.get("ps"),
    )
    ci_dF = result["dF_ci95_eV"]
    barrier_err = result["dF_barrier_block_err_eV"]
    ci_barrier = (
        [result["dF_barrier_eV"] - result["t_factor_95"] * barrier_err,
         result["dF_barrier_eV"] + result["t_factor_95"] * barrier_err]
        if ci_dF is not None and math.isfinite(barrier_err)
        else None
    )  # fmt: skip
    entries: list[tuple[str, float, dict[str, Any]]] = [
        (
            f"{prefix}.dF_eV",
            result["dF_eV"],
            {"ci95": ci_dF, "ci95_reason": "block error (Student t)"},
        ),
        (
            f"{prefix}.dF_block_err_eV",
            result["dF_block_err_eV"],
            {"ci95": None, "ci95_reason": "is itself the block standard error"},
        ),
        (
            f"{prefix}.n_windows",
            result["n_windows"],
            {"ci95": None, "ci95_reason": "count", "unit": "1"},
        ),
        (
            f"{prefix}.dF_barrier_eV",
            result["dF_barrier_eV"],
            {"ci95": ci_barrier, "ci95_reason": "block error (Student t)"},
        ),
        # README-table aliases (templates/README.md.j2 reads the compound-level keys)
        (f"sampling.umbrella.{compound}.dF_eV", result["dF_eV"], {"ci95": ci_dF}),
        (
            f"sampling.umbrella.{compound}.dF_err_eV",
            result["dF_block_err_eV"],
            {"ci95": None, "ci95_reason": "is itself the block standard error"},
        ),
    ]
    numbers: dict[str, Any] = {}
    for key, value, extra in entries:
        if value is None or not math.isfinite(float(value)):
            continue
        numbers[key] = float(value)
        meta_kw = {**common, **extra}
        ci = meta_kw.pop("ci95")
        reason = meta_kw.pop("ci95_reason", None)
        if ci is not None:
            reason = None
        numbers[f"{key}@meta"] = numbers_meta(info, ci95=ci, ci95_reason=reason, **meta_kw)
    write_numbers(ctx, numbers)
    ctx.log(
        dF_eV=result["dF_eV"],
        dF_block_err_eV=result["dF_block_err_eV"],
        dF_barrier_eV=result["dF_barrier_eV"],
        n_samples=n_samples,
        overlap_ok=bool(result["overlap_ok"]),
        overlap_min=float(result["overlap_min"]),
        crosscheck_dF_eV=float(cross["dF_eV"]),
    )
    summary: dict[str, float | int | str] = {
        "dF_eV": result["dF_eV"],
        "dF_block_err_eV": result["dF_block_err_eV"],
        "dF_barrier_eV": result["dF_barrier_eV"],
        "dF_barrier_point_eV": result["dF_barrier_point_eV"],
        f"dF_{other}_eV": float(cross["dF_eV"]),
        "n_windows": int(result["n_windows"]),
        "n_samples": n_samples,
        "overlap_min": float(result["overlap_min"]),
        "overlap_ok": int(result["overlap_ok"]),
        "method": method,
        "compound": compound,
        "model_label": info.label,
    }
    ok = bool(result["overlap_ok"])
    if not ok:
        summary["note"] = (
            f"neighbouring windows overlap below {min_overlap}: stiffer/more windows needed; "
            "numbers not published (status partial)"
        )
    return StageResult(
        stage="sampling.wham",
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status="ok" if ok else "partial",
        outputs=list(ctx.outputs),
        summary=json_safe(summary),
    )


__all__ = [
    "DEFAULT_BLOCKS",
    "DEFAULT_MIN_OVERLAP",
    "KB_EV",
    "METHODS",
    "PMF_JSON",
    "Window",
    "adjacent_overlaps",
    "as_windows",
    "basins",
    "free_energy",
    "make_edges",
    "mbar_profile",
    "profile",
    "run",
    "wham_profile",
]
