"""Labelled WBM-sample discovery evaluation (tier T4b; CONTRACTS.md row 9, SPEC.md section 6).

Protocol (binding; every choice is recorded in ``discovery.json`` and the numbers meta):

* **Structures and truth** come from the data tier's seeded natural-prevalence sample
  (``b20mlip.data.wbm.sample``; ``ids``, per-id ``truth`` with ``e_form_per_atom`` =
  ``e_form_per_atom_mp2020_corrected`` and ``e_above_hull`` =
  ``e_above_hull_mp2020_corrected_ppd_mp``, both eV/atom) and the WBM initial structures.
  This is a labelled 1,000-structure sample, not the leaderboard (SPEC.md section 2).
* **Relaxation**: ASE FIRE with ``FrechetCellFilter`` (positions + cell), ``fmax`` 0.05 eV/Å,
  500 steps (``cfg.eval``), the Matbench-Discovery protocol; :func:`relax` returns the steps
  taken and the caller records whether ``fmax`` was reached or the cap hit.
* **Formation energy** on the MP scale, as Matbench Discovery does for MP-compatible models::

      e_form = (E_model + ΔE_MP2020 − Σ_e n_e μ_e) / N

  with ``μ_e`` the MP elemental reference energies per atom (``_vendor/
  mp-elemental-ref-entries.json.gz``, vendored from janosh/matbench-discovery at the pinned
  commit ``25db9e4``, path ``data/mp/2023-02-07-mp-elemental-reference-entries.json.gz``, MIT,
  sha256 ``9fd6ef848a91c88840a0d1709b5856d0edb4af22095237e59db52f2acd6bbf79``; 89 elements,
  every entry carries ``correction = 0``) and ``ΔE_MP2020`` pymatgen's
  ``MaterialsProject2020Compatibility`` (``check_potcar=False``) applied to a
  ``ComputedStructureEntry`` of the model-relaxed structure whose ``run_type``/``hubbards``
  follow MP's input set (``+U`` on the listed transition metals when the most electronegative
  element is O or F, exactly what ``MPRelaxSet`` does and what the correction scheme checks).
  Rule R6: corrections are applied to MP-scale energies only. A model on the ``qe`` scale
  (B1, head ``Default``) is first mapped with the per-element offset map from
  ``dft offsets`` (``E_mp ≈ E_qe + Σ_e n_e c_e``) and only if its residual passes
  ``cfg.eval.offset_residual_gate_meV`` (rule R3), else the run refuses; ``omat24``/``none``
  scales have no route to the MP hull and refuse too.
* **Hull distance** by the fixed-hull shortcut of Matbench Discovery
  (``calc_discovery_metrics``: ``each_pred = each_true + e_form_pred − e_form_dft``), i.e.
  the DFT convex hull is kept and only the entry's own energy moves.
* **RMSD** (``rmsd_A``): RMS Cartesian displacement of the internal coordinates between the
  initial and the model-relaxed structure (fractional difference wrapped to the nearest
  image, expressed in the relaxed cell). A pure homogeneous strain contributes zero and is
  reported separately as ``volume_change_pct``. This is *not* the leaderboard's RMSD against
  DFT-relaxed structures (those are not held locally).
* **Metrics** at natural prevalence via the vendored ``stable_metrics`` (F1, precision,
  recall, accuracy, MAE/RMSE of ``e_above_hull``); ``DAF`` is dropped before anything is
  written (SPEC.md section 2, gate A4). 95 % bootstrap CIs resample **ids**. F1 itself is
  never published as a headline number: only the paired ``ΔF1 = F1(model) − F1(B0)`` on the
  identical ids with a paired bootstrap CI (:func:`paired_delta_f1`).
* **Resume**: every finished id is appended to ``results.jsonl`` in the run directory;
  ``resume=True`` (``--resume`` reuses the failed/partial run directory) skips ids already
  there.
"""

from __future__ import annotations

import gzip
import json
import math
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from b20mlip.config import Settings
from b20mlip.evaluate import bootstrap
from b20mlip.evaluate.errors import (
    UNSPECIFIED_E0,
    label_for,
    make_calculator,
    model_energy_scale,
    read_checkpoint,
)
from b20mlip.evaluate.mbd_vendored import STABILITY_THRESHOLD, stable_metrics
from b20mlip.models import Metric
from b20mlip.provenance import RunContext, read_manifest, sha256_file

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
ELEMENTAL_REFS_FILE = VENDOR_DIR / "mp-elemental-ref-entries.json.gz"
ELEMENTAL_REFS_SHA256 = "9fd6ef848a91c88840a0d1709b5856d0edb4af22095237e59db52f2acd6bbf79"
ELEMENTAL_REFS_SOURCE = (
    "janosh/matbench-discovery@25db9e4c2cbb0d4ed72bad60e433b737b4a217cd:"
    "data/mp/2023-02-07-mp-elemental-reference-entries.json.gz (MIT)"
)
RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "discovery.json"
NUMBERS_FILE = "numbers.json"
HULL_COLUMN = "e_above_hull_mp2020_corrected_ppd_mp"
E_FORM_COLUMN = "e_form_per_atom_mp2020_corrected"
DEFAULT_ATOMS_ZIP = Path("raw") / "references" / "wbm-initial-atoms.extxyz.zip"
WBM_REFERENCE: dict[str, Any] = {
    "code": "vasp",
    "functional": "PBE",
    "pseudos": "PAW (Materials Project settings, MP2020-corrected)",
    "e0_source": None,
    "cross_functional": False,
}
DROPPED_METRICS: frozenset[str] = frozenset({"DAF"})  # SPEC section 2 / gate A4
EV_TO_MEV = 1000.0


# --- elemental references and corrections ------------------------------------------------------


def mp_elemental_reference_energies(path: str | Path = ELEMENTAL_REFS_FILE) -> dict[str, float]:
    """``{element: energy per atom (eV)}`` from the vendored MP elemental reference entries
    (``(energy + correction) / n_atoms`` of each ``ComputedEntry`` dict)."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, float] = {}
    for element, entry in data.items():
        n_atoms = float(sum(entry["composition"].values()))
        out[str(element)] = (float(entry["energy"]) + float(entry.get("correction", 0.0))) / n_atoms
    return out


def mp_hubbards(composition: Any) -> dict[str, float]:
    """``MPRelaxSet``'s Hubbard U choice for a composition: when the most electronegative
    element is O or F, every element listed under it in ``LDAUU`` gets that U (else none)."""
    from pymatgen.io.vasp.sets import MPRelaxSet  # noqa: PLC0415

    table = MPRelaxSet.CONFIG["INCAR"]["LDAUU"]
    most_electroneg = max(composition.elements).symbol
    settings = table.get(most_electroneg, {})
    return {
        el.symbol: float(settings[el.symbol])
        for el in composition.elements
        if el.symbol in settings
    }


def mp2020_compat() -> Any:
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility  # noqa: PLC0415

    return MaterialsProject2020Compatibility(check_potcar=False)


def mp2020_correction(atoms: Atoms, energy: float, compat: Any | None = None) -> dict[str, Any]:
    """MP2020 correction (eV, total) of a model energy for ``atoms`` on the MP scale."""
    from pymatgen.entries.computed_entries import ComputedStructureEntry  # noqa: PLC0415
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415

    structure = AseAtomsAdaptor.get_structure(atoms)
    hubbards = mp_hubbards(structure.composition)
    entry = ComputedStructureEntry(
        structure,
        float(energy),
        parameters={
            "run_type": "GGA+U" if hubbards else "GGA",
            "is_hubbard": bool(hubbards),
            "hubbards": hubbards,
        },
        data={},
    )
    compat = compat if compat is not None else mp2020_compat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        processed = compat.process_entries([entry], clean=True, verbose=False)
    if not processed:
        raise ValueError(
            f"MaterialsProject2020Compatibility discarded {structure.composition.reduced_formula}"
        )
    done = processed[0]
    return {
        "correction_eV": float(done.correction),
        "adjustments": [str(adj.name) for adj in done.energy_adjustments],
        "run_type": entry.parameters["run_type"],
        "hubbards": hubbards,
    }


def e_form_per_atom(
    atoms: Atoms,
    energy: float,
    refs: Mapping[str, float],
    *,
    corrections: bool = True,
    compat: Any | None = None,
) -> dict[str, Any]:
    """MP-scale formation energy per atom of ``atoms`` with total model energy ``energy``."""
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(refs))
    if missing:
        raise ValueError(f"no MP elemental reference energy for {missing}")
    e_ref = float(sum(refs[s] for s in symbols))
    corr: dict[str, Any] = {
        "correction_eV": 0.0,
        "adjustments": [],
        "run_type": None,
        "hubbards": {},
    }
    if corrections:
        corr = mp2020_correction(atoms, energy, compat)
    n = len(atoms)
    return {
        "e_form_per_atom": (float(energy) + corr["correction_eV"] - e_ref) / n,
        "e_ref_per_atom": e_ref / n,
        "correction_per_atom": corr["correction_eV"] / n,
        "adjustments": corr["adjustments"],
        "run_type": corr["run_type"],
        "hubbards": corr["hubbards"],
    }


def offset_shift(atoms: Atoms, coefficients: Mapping[str, float]) -> float:
    """``Σ_e n_e c_e`` (eV): the QE -> MP energy shift of ``dft offsets`` for ``atoms``."""
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(coefficients))
    if missing:
        raise ValueError(f"offset map has no coefficient for {missing}")
    return float(sum(coefficients[s] for s in symbols))


def read_offsets(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("coefficients", "residual_meV_atom"):
        if key not in data:
            raise ValueError(f"{path}: not an offsets.json (missing {key!r})")
    return data


# --- relaxation and geometry ----------------------------------------------------------------------


def relax(
    atoms: Atoms, calc: Any, fmax: float = 0.05, steps: int = 500, *, fix_symmetry: bool = False
) -> tuple[Atoms, int]:
    """FIRE + ``FrechetCellFilter`` relaxation of a copy of ``atoms``; ``(relaxed, steps)``.

    ``relaxed`` keeps ``calc`` attached (its final energy/forces/stress are cached).
    ``fix_symmetry`` adds ASE's ``FixSymmetry`` constraint (keeps the space group; used for
    the B20 reference cells of the phonon/elastic stages, never for WBM).
    """
    work = atoms.copy()
    work.calc = calc
    if fix_symmetry:
        from ase.constraints import FixSymmetry  # noqa: PLC0415

        work.set_constraint(FixSymmetry(work))
    opt = FIRE(FrechetCellFilter(work), logfile=None)
    opt.run(fmax=float(fmax), steps=int(steps))
    return work, int(opt.nsteps)


def max_force(atoms: Atoms) -> float:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0


def rmsd_internal(initial: Atoms, relaxed: Atoms) -> float:
    """RMS Cartesian displacement of internal coordinates (Å), strain-invariant (see module)."""
    if len(initial) != len(relaxed):
        raise ValueError("initial and relaxed structures differ in atom count")
    d = relaxed.get_scaled_positions(wrap=False) - initial.get_scaled_positions(wrap=False)
    d -= np.round(d)
    cart = d @ np.asarray(relaxed.cell[:], dtype=float)
    return float(np.sqrt(np.mean(np.sum(cart**2, axis=1))))


# --- classification helpers -----------------------------------------------------------------------


def _f1_of(true: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    """F1 with the vendored classification rules (NaN predictions count as unstable); 0.0
    when precision or recall is undefined (a degenerate model), so bootstrap pools stay finite."""
    actual_pos = true <= threshold
    actual_neg = true > threshold
    model_pos = pred <= threshold  # NaN/inf -> False (unstable), like fillna=True upstream
    tp = float(np.sum(actual_pos & model_pos))
    fn = float(np.sum(actual_pos & ~model_pos))
    fp = float(np.sum(actual_neg & model_pos))
    if tp + fp == 0.0 or tp + fn == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _precision_of(true: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    model_pos = pred <= threshold
    tp = float(np.sum((true <= threshold) & model_pos))
    fp = float(np.sum((true > threshold) & model_pos))
    return 0.0 if tp + fp == 0.0 else tp / (tp + fp)


def _recall_of(true: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    actual_pos = true <= threshold
    tp = float(np.sum(actual_pos & (pred <= threshold)))
    n_pos = float(np.sum(actual_pos))
    return 0.0 if n_pos == 0.0 else tp / n_pos


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nan(value: Any) -> float:
    f = _finite(value)
    return float("nan") if f is None else f


def sample_metrics(
    records: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    n_boot: int,
    seed: int,
    *,
    stability_threshold: float = float(STABILITY_THRESHOLD),
) -> dict[str, Any]:
    """Vendored ``stable_metrics`` at natural prevalence plus id-bootstrap CIs.

    ``records`` are the per-id result dicts (``id``, ``e_above_hull_pred``, ``rmsd_A``,
    ``steps``, ``converged``). Returns ``{"metrics": {name: Metric-dict}, "counts": {...},
    "prevalence": ..., "stability_threshold": ..., "vendored": {...}}``; ``DAF`` is dropped.
    """
    ids = [str(r["id"]) for r in records]
    if not ids:
        raise ValueError("no records to score")
    each_true = np.array([_nan(truth.get(i, {}).get("e_above_hull")) for i in ids], dtype=float)
    each_pred = np.array([_nan(r.get("e_above_hull_pred")) for r in records], dtype=float)
    raw = stable_metrics(each_true, each_pred, stability_threshold=stability_threshold, fillna=True)
    vendored = {
        k: _finite(v) if k not in ("TP", "FP", "TN", "FN") else int(v)
        for k, v in raw.items()
        if k not in DROPPED_METRICS
    }
    thr = float(stability_threshold)
    rows = {i: np.array([[t, p]]) for i, t, p in zip(ids, each_true, each_pred, strict=True)}
    valid = {i: r for i, r in rows.items() if np.isfinite(r[0, 0]) and np.isfinite(r[0, 1])}

    def metric(
        value: float | None,
        stat: Any,
        pools: Mapping[str, np.ndarray],
        unit: str,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        """Metric dict; ``value`` is ``None`` when the vendored metric is undefined (NaN)."""
        ci: list[float] | None = None
        if len(pools) >= 2 and value is not None:
            lo, hi = bootstrap.ci(pools, n_boot, seed, stat)
            ci = [lo * scale, hi * scale]
        return {
            "value": None if value is None else value * scale,
            "ci95": ci,
            "n": len(pools),
            "unit": unit,
        }

    metrics: dict[str, Any] = {
        "f1": metric(vendored["F1"], lambda p: _f1_of(p[:, 0], p[:, 1], thr), rows, "F1"),
        "precision": metric(
            vendored["Precision"], lambda p: _precision_of(p[:, 0], p[:, 1], thr), rows, "fraction"
        ),
        "recall": metric(
            vendored["Recall"], lambda p: _recall_of(p[:, 0], p[:, 1], thr), rows, "fraction"
        ),
        "accuracy": metric(
            vendored["Accuracy"],
            lambda p: float(np.mean((p[:, 0] <= thr) == (p[:, 1] <= thr))),
            rows,
            "fraction",
        ),
        "mae_e_above_hull": metric(
            vendored["MAE"],
            lambda p: float(np.mean(np.abs(p[:, 0] - p[:, 1]))),
            valid,
            "meV/atom",
            EV_TO_MEV,
        ),
        "rmse_e_above_hull": metric(
            vendored["RMSE"],
            lambda p: float(np.sqrt(np.mean((p[:, 0] - p[:, 1]) ** 2))),
            valid,
            "meV/atom",
            EV_TO_MEV,
        ),
    }
    rmsd = {
        str(r["id"]): np.array([float(r["rmsd_A"])])
        for r in records
        if _finite(r.get("rmsd_A")) is not None
    }
    if rmsd:
        metrics["rmsd"] = metric(float(np.mean([v[0] for v in rmsd.values()])), np.mean, rmsd, "Å")
    steps = [int(r.get("steps", 0)) for r in records]
    n_conv = sum(1 for r in records if r.get("converged"))
    return {
        "metrics": metrics,
        "counts": {
            "n_ids": len(ids),
            "n_with_truth": int(np.sum(np.isfinite(each_true))),
            "n_with_prediction": int(np.sum(np.isfinite(each_pred))),
            "n_converged": n_conv,
            "n_capped": len(records) - n_conv,
            "steps_mean": float(np.mean(steps)) if steps else 0.0,
            "runtime_s_mean": float(np.mean([float(r.get("runtime_s", 0.0)) for r in records])),
        },
        "prevalence": float(np.mean(each_true[np.isfinite(each_true)] <= thr))
        if np.any(np.isfinite(each_true))
        else 0.0,
        "stability_threshold": thr,
        "vendored": vendored,
    }


def paired_delta_f1(
    pred_a: Mapping[str, Any],
    pred_b: Mapping[str, Any],
    truth: Mapping[str, Any],
    n_boot: int,
    seed: int,
    *,
    stability_threshold: float = float(STABILITY_THRESHOLD),
) -> Metric:
    """``ΔF1 = F1(b) − F1(a)`` on the identical ids with a paired bootstrap over ids.

    ``pred_*`` map an id to its predicted ``e_above_hull`` (eV/atom; ``None`` = no
    prediction = unstable) and ``truth`` maps ids to the true hull distance (a float or a
    dict with ``e_above_hull``). The id sets of ``pred_a`` and ``pred_b`` must be identical
    and covered by ``truth``; ids whose truth is missing are dropped (counted out of ``n``).
    """
    ids_a, ids_b = set(pred_a), set(pred_b)
    if ids_a != ids_b:
        raise ValueError(
            f"paired ΔF1 needs identical ids: {len(ids_a - ids_b)} only in a, "
            f"{len(ids_b - ids_a)} only in b"
        )
    missing = sorted(i for i in ids_a if i not in truth)
    if missing:
        raise ValueError(f"{len(missing)} ids have no truth (first: {missing[:3]})")
    thr = float(stability_threshold)
    rows: dict[str, np.ndarray] = {}
    for i in sorted(ids_a):
        t = truth[i]
        t_val = _nan(t.get("e_above_hull") if isinstance(t, Mapping) else t)
        if not math.isfinite(t_val):
            continue
        rows[i] = np.array([[t_val, _nan(pred_a[i]), _nan(pred_b[i])]])
    if not rows:
        raise ValueError("no ids with a finite truth")
    pool = np.concatenate(list(rows.values()), axis=0)

    def delta(p: np.ndarray) -> float:
        return _f1_of(p[:, 0], p[:, 2], thr) - _f1_of(p[:, 0], p[:, 1], thr)

    value = delta(pool)
    ci: tuple[float, float] | None = None
    if len(rows) >= 2:
        ci = bootstrap.ci_paired(rows, n_boot, seed, delta)
    return Metric(value=value, ci95=ci, n=len(rows), unit="ΔF1")


# --- the sample run -------------------------------------------------------------------------------


def read_results(path: str | Path) -> dict[str, dict[str, Any]]:
    """``{id: record}`` from a ``results.jsonl`` (last record of an id wins; blank lines ok)."""
    p = Path(path)
    out: dict[str, dict[str, Any]] = {}
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[str(rec["id"])] = rec
    return out


def evaluate_structure(
    wbm_id: str,
    atoms: Atoms,
    calc: Any,
    truth: Mapping[str, Any] | None,
    refs: Mapping[str, float],
    *,
    fmax: float,
    steps: int,
    compat: Any | None = None,
    offsets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Relax one WBM structure and compute its MP-scale ``e_form`` / ``e_above_hull``."""
    t0 = time.perf_counter()
    relaxed, n_steps = relax(atoms, calc, fmax=fmax, steps=steps)
    energy = float(relaxed.get_potential_energy())
    f_max = max_force(relaxed)
    rec: dict[str, Any] = {
        "id": wbm_id,
        "formula": atoms.get_chemical_formula("metal", empirical=False),
        "n_sites": len(atoms),
        "steps": n_steps,
        "fmax_final_eVA": f_max,
        "converged": bool(f_max <= fmax),
        "energy_model_eV": energy,
        "offset_shift_eV": 0.0,
        "rmsd_A": rmsd_internal(atoms, relaxed),
        "volume_change_pct": float(relaxed.get_volume() / atoms.get_volume() - 1.0) * 100.0,
        "e_form_pred": None,
        "e_above_hull_pred": None,
        "e_form_true": None if truth is None else truth.get("e_form_per_atom"),
        "e_above_hull_true": None if truth is None else truth.get("e_above_hull"),
        "error": None,
    }
    try:
        e_total = energy
        if offsets is not None:
            rec["offset_shift_eV"] = offset_shift(relaxed, offsets)
            e_total = energy + rec["offset_shift_eV"]
        form = e_form_per_atom(relaxed, e_total, refs, corrections=True, compat=compat)
        rec["e_form_pred"] = form["e_form_per_atom"]
        rec["correction_per_atom"] = form["correction_per_atom"]
        rec["adjustments"] = form["adjustments"]
        rec["run_type"] = form["run_type"]
        e_form_true = _finite(rec["e_form_true"])
        e_hull_true = _finite(rec["e_above_hull_true"])
        if e_form_true is not None and e_hull_true is not None:
            rec["e_above_hull_pred"] = e_hull_true + (form["e_form_per_atom"] - e_form_true)
    except ValueError as exc:  # no reference energy / correction refused: geometry still counts
        rec["error"] = str(exc)
    rec["runtime_s"] = time.perf_counter() - t0
    return rec


def run_sample(
    model_path: str | Path,
    head: str,
    sample: Mapping[str, Any],
    cfg: Settings,
    ctx: RunContext,
    resume: bool = True,
    *,
    atoms_zip: str | Path | None = None,
    atoms: Mapping[str, Atoms] | None = None,
    energy_scale: str = "mp",
    offsets: Mapping[str, Any] | None = None,
    limit: int | None = None,
    fmax: float | None = None,
    steps: int | None = None,
    seed: int | None = None,
    n_boot: int | None = None,
    calc: Any | None = None,
) -> dict[str, Any]:
    """Relax every id of ``sample`` with the model and score the sample (see module docstring).

    ``atoms`` (``{id: Atoms}``) bypasses the WBM zip (tests, the agent tier); otherwise the
    structures are read from ``atoms_zip`` (default ``<data_dir>/raw/references/
    wbm-initial-atoms.extxyz.zip``). Per-id records go to ``results.jsonl`` in ``ctx.out_dir``
    as they finish; with ``resume=True`` ids already there are skipped. Returns the
    ``discovery.json`` dict (metrics, counts, protocol).
    """
    ids: list[str] = [str(i) for i in sample["ids"]]
    if limit is not None:
        ids = ids[: int(limit)]
    truth: Mapping[str, Mapping[str, Any]] = sample.get("truth", {})
    fmax = float(cfg.eval.fmax if fmax is None else fmax)
    steps = int(cfg.eval.max_steps if steps is None else steps)
    n_boot = int(cfg.eval.bootstrap_n if n_boot is None else n_boot)
    seed = int(
        seed
        if seed is not None
        else (ctx.seed if ctx.seed is not None else cfg.eval.bootstrap_seed)
    )

    coefficients: dict[str, float] | None = None
    offsets_residual: float | None = None
    if energy_scale == "mp":
        pass
    elif energy_scale == "qe":
        if offsets is None:
            raise ValueError(
                "rule R3: a QE-scale model needs the per-element offset map (dft offsets -> "
                "offsets.json) to reach the MP hull; pass offsets="
            )
        offsets_residual = float(offsets["residual_meV_atom"])
        gate = float(cfg.eval.offset_residual_gate_meV)
        if offsets_residual > gate:
            raise ValueError(
                f"rule R3: offset residual {offsets_residual:.2f} meV/atom > gate {gate:g}; WBM "
                "energies are n/a (scale) for this model"
            )
        coefficients = {str(k): float(v) for k, v in offsets["coefficients"].items()}
    else:
        raise ValueError(
            f"energy scale {energy_scale!r} has no route to the MP hull (mp: direct; qe: offset "
            "map; omat24/none: not comparable)"
        )

    results_path = ctx.out_dir / RESULTS_FILE
    done = read_results(results_path) if resume else {}
    if not resume and results_path.exists():
        results_path.unlink()
    todo = [i for i in ids if i not in done]
    if atoms is None and todo:
        from b20mlip.data import wbm  # noqa: PLC0415 - data tier, lazy

        zip_path = (
            Path(atoms_zip)
            if atoms_zip is not None
            else Path(cfg.paths.data_dir) / DEFAULT_ATOMS_ZIP
        )
        if not zip_path.is_file():
            raise FileNotFoundError(f"WBM initial-structures archive not found: {zip_path}")
        structures = wbm.atoms_for_ids(zip_path, todo)
    else:
        structures = {i: atoms[i] for i in todo if i in atoms} if atoms is not None else {}
    missing = [i for i in todo if i not in structures]
    if missing:
        raise KeyError(f"no structure for {len(missing)} ids (first: {missing[:3]})")

    calc = (
        calc
        if calc is not None
        else make_calculator(model_path, head, device=cfg.compute.device, dtype=cfg.compute.dtype)
    )
    refs = mp_elemental_reference_energies()
    compat = mp2020_compat()
    n_new = 0
    with open(results_path, "a", encoding="utf-8") as fh:
        for wbm_id in todo:
            rec = evaluate_structure(
                wbm_id, structures[wbm_id], calc, truth.get(wbm_id), refs,
                fmax=fmax, steps=steps, compat=compat, offsets=coefficients,
            )  # fmt: skip
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[wbm_id] = rec
            n_new += 1
    records = [done[i] for i in ids]
    scored = sample_metrics(records, truth, n_boot, seed)
    return {
        **scored,
        "n_ids": len(ids),
        "n_new": n_new,
        "n_resumed": len(ids) - n_new,
        "sample_seed": int(sample.get("seed", -1)),
        "sample_n": int(sample.get("n", len(sample["ids"]))),
        "sample_prevalence": float(sample.get("prevalence", float("nan"))),
        "limit": limit,
        "head": head,
        "energy_scale": energy_scale,
        "offsets_residual_meV_atom": offsets_residual,
        "fmax": fmax,
        "max_steps": steps,
        "bootstrap_n": n_boot,
        "bootstrap_seed": seed,
        "hull_column": HULL_COLUMN,
        "e_form_column": E_FORM_COLUMN,
        "elemental_refs": ELEMENTAL_REFS_SOURCE,
        "corrections": (
            "MaterialsProject2020Compatibility (pymatgen, check_potcar=False, MPRelaxSet +U rule)"
        ),
        "hull_shortcut": "e_above_hull_pred = e_above_hull_true + (e_form_pred - e_form_true)",
        "rmsd_definition": (
            "RMS internal-coordinate displacement initial -> relaxed (Å); strain-invariant"
        ),
        "results_file": str(results_path),
    }


# --- numbers.json and the stage ------------------------------------------------------------------


def numbers_for_run(
    scored: Mapping[str, Any],
    *,
    label: str,
    head: str,
    e0_source: str,
    energy_scale: str,
    model_sha256: str,
    sample_seed: int,
    delta: Metric | None = None,
    baseline_run_id: str | None = None,
    baseline_label: str = "B0",
) -> dict[str, Any]:
    """Flat entries ``eval.discovery.<label>.<metric>``; F1 only as the paired ``delta_f1``."""
    base_meta: dict[str, Any] = {
        "reference": dict(WBM_REFERENCE, e0_source=e0_source),
        "head": head,
        "seed": int(scored["bootstrap_seed"]),
        "e0_source": e0_source,
        "energy_scale": energy_scale,
        "model_label": label,
        "bracket": label,
        "tier": "T4b",
        "prevalence": "natural",
        "sample_seed": int(sample_seed),
        "sample_n": int(scored["sample_n"]),
        "sample_prevalence": scored.get("sample_prevalence"),
        "model_sha256": model_sha256,
        "hull_column": HULL_COLUMN,
        "fmax": scored["fmax"],
        "max_steps": scored["max_steps"],
        "bootstrap_n": scored["bootstrap_n"],
        "n_converged": scored["counts"]["n_converged"],
    }
    if scored.get("offsets_residual_meV_atom") is not None:
        base_meta["offsets_residual_meV_atom"] = scored["offsets_residual_meV_atom"]
    out: dict[str, Any] = {}
    published = ("precision", "recall", "mae_e_above_hull", "rmse_e_above_hull", "rmsd")
    for name in published:
        m = scored["metrics"].get(name)
        if m is None or _finite(m["value"]) is None:
            continue
        meta = {**base_meta, "n": int(m["n"]), "unit": m["unit"], "ci95": m["ci95"]}
        if m["ci95"] is None:
            meta["ci95_reason"] = "fewer than two ids; no bootstrap"
        out[f"eval.discovery.{label}.{name}"] = float(m["value"])
        out[f"eval.discovery.{label}.{name}@meta"] = meta
    counts = scored["counts"]
    for name, value in (
        ("n_ids", counts["n_ids"]),
        ("n_converged", counts["n_converged"]),
        ("steps_mean", counts["steps_mean"]),
        ("runtime_s_mean", counts["runtime_s_mean"]),
    ):
        out[f"eval.discovery.{label}.{name}"] = float(value)
        out[f"eval.discovery.{label}.{name}@meta"] = {
            **base_meta,
            "n": int(counts["n_ids"]),
            "ci95": None,
            "ci95_reason": "count / mean of a deterministic quantity",
            "unit": "count"
            if name.startswith("n_")
            else ("steps" if name == "steps_mean" else "s"),
        }
    if delta is not None:
        meta = {
            **base_meta,
            "n": int(delta.n),
            "unit": delta.unit,
            "ci95": None if delta.ci95 is None else [delta.ci95[0], delta.ci95[1]],
            "paired_vs": baseline_label,
            "baseline_run_id": baseline_run_id,
        }
        if delta.ci95 is None:
            meta["ci95_reason"] = "fewer than two ids; no paired bootstrap"
        out[f"eval.discovery.{label}.delta_f1"] = float(delta.value)
        out[f"eval.discovery.{label}.delta_f1@meta"] = meta
    if scored.get("offsets_residual_meV_atom") is not None:
        out["offsets.residual_meV_atom"] = float(scored["offsets_residual_meV_atom"])
        out["offsets.residual_meV_atom@meta"] = {
            "source": "offsets.json input of this run",
            "unit": "meV/atom",
            "gate_meV_atom": 20.0,
        }
    return out


def load_baseline(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(predictions {id: e_above_hull_pred}, discovery.json)`` of a finished discovery run."""
    p = Path(run_dir)
    manifest = read_manifest(p)
    if manifest.stage != "eval.discovery" or manifest.status != "ok":
        raise ValueError(
            f"{p}: not an ok eval.discovery run (stage {manifest.stage}, status {manifest.status})"
        )
    summary_path = p / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = read_results(p / RESULTS_FILE)
    preds = {i: r.get("e_above_hull_pred") for i, r in records.items()}
    return preds, summary


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    head: str = "pt_head",
    sample: str | Path,
    atoms_zip: str | Path | None = None,
    baseline_run: str | Path | None = None,
    checkpoint_json: str | Path | None = None,
    energy_scale: str | None = None,
    offsets: str | Path | None = None,
    label: str | None = None,
    e0_source: str | None = None,
    limit: int | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    bootstrap_n: int | None = None,
    atoms: Mapping[str, Atoms] | None = None,
) -> dict[str, Any]:
    """Stage ``eval.discovery``: :func:`run_sample` + ``numbers.json`` (+ paired ΔF1 vs B0).

    ``baseline_run`` is the run directory of the B0 (MPA-0 zero-shot) discovery run on the
    same sample; when given, ``eval.discovery.<label>.delta_f1`` is published with
    ``paired_vs: "B0"``. ``ctx.dry_run`` plans only. ``ctx.resume`` resumes ``results.jsonl``.
    """
    from b20mlip.data import wbm  # noqa: PLC0415 - data tier, lazy

    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if head not in ("Default", "pt_head"):
        raise ValueError(f"head must be Default or pt_head, got {head!r}")
    ctx.add_input(model_path, "model")
    model_sha = sha256_file(model_path)
    info = None
    if checkpoint_json is not None:
        info = read_checkpoint(checkpoint_json)
        ctx.add_input(Path(checkpoint_json), "json")
        if info.sha256 != model_sha:
            raise ValueError("checkpoint.json sha256 does not match the model file")
        if head not in info.heads:
            raise ValueError(f"checkpoint has heads {info.heads}, not {head!r}")
    scale = energy_scale if energy_scale is not None else model_energy_scale(info, head)
    if scale is None:
        raise ValueError("model energy scale unknown: pass --checkpoint-json or --energy-scale")
    if e0_source is None:
        e0_source = (
            "foundation" if head == "pt_head" else (info.e0_source if info else UNSPECIFIED_E0)
        )
    model_label = label or label_for(info, model_path)
    sample_path = Path(sample)
    ctx.add_input(sample_path, "json")
    sample_data = wbm.load_sample(sample_path)
    offsets_data: dict[str, Any] | None = None
    if offsets is not None:
        offsets_data = read_offsets(offsets)
        ctx.add_input(Path(offsets), "json")
    plan = {
        "model": str(model_path),
        "model_sha256": model_sha,
        "head": head,
        "model_label": model_label,
        "energy_scale": scale,
        "e0_source": e0_source,
        "sample": str(sample_path),
        "sample_sha256": sha256_file(sample_path),
        "n_ids": len(sample_data["ids"])
        if limit is None
        else min(int(limit), len(sample_data["ids"])),
        "baseline_run": None if baseline_run is None else str(baseline_run),
        "resume": bool(ctx.resume),
        "dry_run": bool(ctx.dry_run),
    }
    (ctx.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / "plan.json", "json")
    if ctx.dry_run:
        ctx.log(model_sha256=model_sha, head=head, tier="T4b", planned=True)
        return {"model_label": model_label, "head": head, "n_ids": plan["n_ids"], "planned": 1}

    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(int(cfg.compute.threads))
    except Exception:  # noqa: BLE001
        pass
    scored = run_sample(
        model_path, head, sample_data, cfg, ctx, resume=bool(ctx.resume),
        atoms_zip=atoms_zip, atoms=atoms, energy_scale=scale, offsets=offsets_data,
        limit=limit, fmax=fmax, steps=max_steps, seed=seed, n_boot=bootstrap_n,
    )  # fmt: skip
    scored["sample_sha256"] = plan["sample_sha256"]
    scored["model_label"] = model_label
    scored["model_sha256"] = model_sha
    delta: Metric | None = None
    baseline_id: str | None = None
    if baseline_run is not None:
        preds_a, base_summary = load_baseline(baseline_run)
        if base_summary.get("sample_sha256") not in (None, plan["sample_sha256"]):
            raise ValueError("baseline run used a different sample file (sha256 mismatch)")
        records = read_results(ctx.out_dir / RESULTS_FILE)
        ids = [str(i) for i in sample_data["ids"]][: len(records) if limit is None else int(limit)]
        preds_b = {i: records[i].get("e_above_hull_pred") for i in ids if i in records}
        preds_a = {i: preds_a[i] for i in preds_b if i in preds_a}
        delta = paired_delta_f1(
            preds_a,
            preds_b,
            sample_data["truth"],
            int(scored["bootstrap_n"]),
            int(scored["bootstrap_seed"]),
        )
        baseline_id = read_manifest(Path(baseline_run)).run_id
        scored["delta_f1_vs_B0"] = delta.model_dump(mode="json")
        scored["baseline_run_id"] = baseline_id
    summary_path = ctx.out_dir / SUMMARY_FILE
    summary_path.write_text(json.dumps(scored, indent=2, default=str) + "\n", encoding="utf-8")
    ctx.add_output(summary_path, "json")
    ctx.add_output(ctx.out_dir / RESULTS_FILE, "json")
    numbers = numbers_for_run(
        scored, label=model_label, head=head, e0_source=e0_source, energy_scale=scale,
        model_sha256=model_sha, sample_seed=int(sample_data["seed"]), delta=delta,
        baseline_run_id=baseline_id,
    )  # fmt: skip
    numbers_path = ctx.out_dir / NUMBERS_FILE
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(numbers_path, "json")
    ctx.log(
        model_sha256=model_sha, head=head, tier="T4b", n=scored["n_ids"],
        bootstrap_seed=scored["bootstrap_seed"], reference=WBM_REFERENCE,
        energy_scale=scale, e0_source=e0_source, model_label=model_label,
        sample_seed=sample_data["seed"], n_resumed=scored["n_resumed"],
        baseline_run_id=baseline_id,
    )  # fmt: skip

    def rounded(name: str, digits: int) -> float | str:
        value = _finite(scored["metrics"].get(name, {}).get("value"))
        return "undefined" if value is None else round(value, digits)

    out: dict[str, Any] = {
        "model_label": model_label,
        "head": head,
        "n_ids": scored["n_ids"],
        "n_converged": scored["counts"]["n_converged"],
        "n_resumed": scored["n_resumed"],
        "precision": rounded("precision", 4),
        "recall": rounded("recall", 4),
        "mae_e_above_hull_meV": rounded("mae_e_above_hull", 3),
    }
    if "rmsd" in scored["metrics"]:
        out["rmsd_A"] = rounded("rmsd", 4)
    if delta is not None:
        out["delta_f1_vs_B0"] = round(float(delta.value), 4)
    return out


__all__ = [
    "DEFAULT_ATOMS_ZIP",
    "DROPPED_METRICS",
    "ELEMENTAL_REFS_FILE",
    "ELEMENTAL_REFS_SHA256",
    "ELEMENTAL_REFS_SOURCE",
    "E_FORM_COLUMN",
    "HULL_COLUMN",
    "NUMBERS_FILE",
    "RESULTS_FILE",
    "SUMMARY_FILE",
    "WBM_REFERENCE",
    "e_form_per_atom",
    "evaluate_structure",
    "load_baseline",
    "max_force",
    "mp2020_compat",
    "mp2020_correction",
    "mp_elemental_reference_energies",
    "mp_hubbards",
    "numbers_for_run",
    "offset_shift",
    "paired_delta_f1",
    "read_offsets",
    "read_results",
    "relax",
    "rmsd_internal",
    "run",
    "run_sample",
    "sample_metrics",
]
