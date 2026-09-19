"""Energy / force / stress error tables per tier (CONTRACTS.md row 9, SPEC.md section 6).

Metric definitions (binding, all reported in milli-units):

* ``mae_e`` / ``rmse_e``: mean |ΔE| / N and sqrt(mean (ΔE / N)²) over frames, ΔE = E_model −
  E_reference per frame, N its atom count — meV/atom;
* ``mae_f`` / ``rmse_f``: over **all 3N force components** of every frame pooled — meV/Å;
* ``mae_s``: mean |Δσ| over the 6 Voigt components (``xx yy zz yz xz xy``, ASE sign
  convention, eV/Å³ in the frames) of the frames that carry a stress label — meV/Å³; absent
  when no frame has one;
* ``by_config_type``: the same metrics restricted to each ``config_type``;
* every metric carries a 95 % percentile-bootstrap CI over **groups** (``group_id``;
  :mod:`b20mlip.evaluate.bootstrap`, 2,000 resamples, seeded) or ``ci95=None`` when fewer
  than two groups contribute; ``Metric.n`` is the number of contributing frames.

Energy-scale rule (SPEC.md R3): energies are compared only when the model head's energy
scale equals the frames' scale (``mp`` for MPtrj, ``omat24``, ``qe``); otherwise the energy
metrics are **absent** from the table (never fabricated) and the stage summary says why.
A multihead replay checkpoint (B2) is on the ``qe`` scale through head ``Default`` and on the
``mp`` scale through ``pt_head``; :func:`model_energy_scale` encodes that.

Tier T3 (OMat24 VASP forces) must carry ``noise_floor_f``: the force RMSE between the 20
OMat24 frames and their QE re-labels (SPEC.md R5, :func:`noise_floor_from_frames`). Without
it the table is still built but the stage warns and ``numbers.json`` publishes
``noise_floor_f: null`` with a reason, which gate A9 then flags.

The reference of a table is inferred from the frames' ``label_source`` (``mptrj``/``omat24``
-> VASP PBE, ``qe`` -> QE PBE with the project's SSSP family, ``pyscf`` -> PySCF PBE) unless
the caller passes one.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from b20mlip.config import Settings
from b20mlip.evaluate import bootstrap
from b20mlip.io import frame_to_atoms, read_frames
from b20mlip.models import (
    CheckpointInfo,
    EnergyScale,
    ErrorTable,
    Frame,
    Head,
    Metric,
    Reference,
    Split,
    Tier,
)
from b20mlip.provenance import RunContext, sha256_file
from b20mlip.train.finetune import find_frames_for_split, read_split

EV_TO_MEV = 1000.0
NUMBERS_FILE = "numbers.json"
TABLE_FILE = "errors_{tier}.json"
QE_PSEUDOS = "SSSP-efficiency-1.3"
UNITS: dict[str, str] = {
    "mae_e": "meV/atom",
    "rmse_e": "meV/atom",
    "mae_f": "meV/Å",
    "rmse_f": "meV/Å",
    "mae_s": "meV/Å^3",
}
ALL_TIERS: tuple[Tier, ...] = ("T0", "T1", "T2", "T3", "T4a", "T4b")
DEFAULT_TIERS: tuple[Tier, ...] = ("T0", "T1", "T2", "T3", "T4a")
VARIANT_LABELS: dict[str, str] = {
    "naive": "B1",
    "replay": "B2",
    "scratch": "B3",
    "bootstrap": "B4",
}
ZERO_SHOT_LABELS: dict[str, str] = {"medium-mpa-0": "B0", "medium": "B0p", "small": "B0p"}
# label_source -> (code, functional, pseudopotential family)
REFERENCE_BY_SOURCE: dict[str, tuple[str, str, str | None]] = {
    "mptrj": ("vasp", "PBE", "PAW (Materials Project)"),
    "omat24": ("vasp", "PBE", "PAW (OMat24)"),
    "qe": ("qe", "PBE", QE_PSEUDOS),
    "pyscf": ("pyscf", "PBE", "GTH"),
    "mace_zero_shot": ("mace", "PBE", None),
}
UNSPECIFIED_E0 = "unspecified"


# --- model bookkeeping ---------------------------------------------------------------------------


def make_calculator(
    model_path: str | Path, head: str = "Default", *, device: str = "cpu", dtype: str = "float64"
) -> Any:
    """``MACECalculator`` on ``head``; raises when the model has no such head (MACE would
    silently fall back to its last head otherwise)."""
    from mace.calculators import MACECalculator  # noqa: PLC0415 - heavy import

    calc = MACECalculator(
        model_paths=[str(model_path)], device=device, default_dtype=dtype, head=head
    )
    if getattr(calc, "head", head) != head:
        raise ValueError(
            f"model {model_path} has heads {getattr(calc, 'available_heads', ['?'])}, not {head!r}"
        )
    return calc


def read_checkpoint(path: str | Path) -> CheckpointInfo:
    return CheckpointInfo.model_validate_json(Path(path).read_text(encoding="utf-8"))


def label_for(info: CheckpointInfo | None, model_path: str | Path) -> str:
    """Bracket label (README key segment): B0/B0p for zero-shot, B1..B4 by variant, else the
    model file stem."""
    if info is not None:
        if info.variant == "zero_shot":
            return ZERO_SHOT_LABELS.get(info.foundation or "", "B0")
        return VARIANT_LABELS.get(info.variant, info.variant)
    return Path(model_path).stem.replace(".", "_")


def model_energy_scale(info: CheckpointInfo | None, head: str) -> EnergyScale | None:
    """The energy scale of ``head``: ``pt_head`` is always the foundation (MP) scale (R2),
    ``Default`` is the checkpoint's; ``None`` when no checkpoint is known."""
    if head == "pt_head":
        return "mp"
    return None if info is None else info.energy_scale


def frames_energy_scale(frames: Iterable[Frame]) -> str:
    """``"mp"``/``"omat24"``/``"qe"`` when every labelled frame agrees, ``"none"`` when no
    frame carries a scale, ``"mixed"`` otherwise."""
    scales = {f.energy_scale for f in frames} - {"none"}
    if len(scales) > 1:
        return "mixed"
    return next(iter(scales)) if scales else "none"


def energy_metrics_allowed(
    model_scale: str | None, frames: Sequence[Frame]
) -> tuple[bool, str | None]:
    """Rule R3 gate: ``(allowed, reason)``; energies are compared only on one shared scale."""
    if not any(f.energy is not None for f in frames):
        return False, "no energy labels in the frames"
    scale = frames_energy_scale(frames)
    if scale == "mixed":
        return False, "frames mix energy scales"
    if model_scale is None:
        return False, "model energy scale unknown (pass --checkpoint-json or --energy-scale)"
    if model_scale != scale:
        return False, f"model energy scale {model_scale!r} != frames' {scale!r} (rule R3)"
    return True, None


def reference_for(
    frames: Sequence[Frame], *, e0_source: str | None = None, pseudos: str | None = None
) -> Reference:
    """Reference inferred from the frames' ``label_source`` (must be a single source)."""
    sources = {f.label_source for f in frames}
    if len(sources) != 1:
        raise ValueError(f"frames carry several label sources {sorted(sources)}; pass reference=")
    source = next(iter(sources))
    if source not in REFERENCE_BY_SOURCE:
        raise ValueError(f"no reference known for label_source {source!r}; pass reference=")
    code, functional, family = REFERENCE_BY_SOURCE[source]
    return Reference(
        code=cast(Any, code),
        functional=functional,
        pseudos=pseudos if (pseudos is not None and source == "qe") else family,
        e0_source=e0_source,
        cross_functional=False,
    )


# --- predictions and residuals ----------------------------------------------------------------


@dataclass(frozen=True)
class Residual:
    frame_id: str
    group_id: str
    config_type: str
    n_atoms: int
    de_per_atom: float | None  # eV/atom
    df: np.ndarray  # (3N,) eV/Å
    ds: np.ndarray | None  # (6,) eV/Å^3


def predict(calc: Any, frame: Frame) -> tuple[float, np.ndarray, np.ndarray | None]:
    """``(energy eV, forces (N,3) eV/Å, stress Voigt-6 eV/Å^3 or None)`` of one frame."""
    atoms = frame_to_atoms(frame)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress: np.ndarray | None
    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:  # noqa: BLE001 - a model without stress support
        stress = None
    return energy, forces, stress


def residuals(calc: Any, frames: Sequence[Frame], *, with_energy: bool) -> list[Residual]:
    out: list[Residual] = []
    for frame in frames:
        if frame.forces is None:
            raise ValueError(f"frame {frame.frame_id} has no force labels")
        energy, forces, stress = predict(calc, frame)
        n = len(frame.numbers)
        ref_f = np.asarray(frame.forces, dtype=float)
        if ref_f.shape != forces.shape:
            raise ValueError(f"frame {frame.frame_id}: force shape {ref_f.shape} != {forces.shape}")
        de = None
        if with_energy and frame.energy is not None:
            de = (energy - float(frame.energy)) / n
        ds = None
        if frame.stress is not None and stress is not None:
            ds = stress - np.asarray(frame.stress, dtype=float)
        out.append(
            Residual(
                frame_id=frame.frame_id,
                group_id=frame.group_id,
                config_type=str(frame.config_type),
                n_atoms=n,
                de_per_atom=de,
                df=(forces - ref_f).reshape(-1),
                ds=ds,
            )
        )
    return out


def _metric(
    pools: Mapping[str, np.ndarray],
    stat: bootstrap.Stat,
    n_frames: int,
    unit: str,
    bootstrap_n: int,
    seed: int,
) -> Metric:
    pooled = np.concatenate([np.asarray(v, dtype=float) for v in pools.values()])
    value = float(stat(pooled)) * EV_TO_MEV
    ci: tuple[float, float] | None = None
    if len(pools) >= 2:
        lo, hi = bootstrap.ci(pools, bootstrap_n, seed, stat)
        ci = (lo * EV_TO_MEV, hi * EV_TO_MEV)
    return Metric(value=value, ci95=ci, n=n_frames, unit=unit)


def metrics_from_residuals(
    res: Sequence[Residual], bootstrap_n: int, seed: int
) -> dict[str, Metric]:
    """The metric dict of one residual set (energies only when every residual has ``de``)."""
    if not res:
        raise ValueError("no residuals")
    out: dict[str, Metric] = {}
    energies = [r for r in res if r.de_per_atom is not None]
    if energies and len(energies) == len(res):
        abs_e = bootstrap.group_values(
            [(r.group_id, [abs(cast(float, r.de_per_atom))]) for r in energies]
        )
        raw_e = bootstrap.group_values(
            [(r.group_id, [cast(float, r.de_per_atom)]) for r in energies]
        )
        out["mae_e"] = _metric(abs_e, np.mean, len(energies), UNITS["mae_e"], bootstrap_n, seed)
        out["rmse_e"] = _metric(
            raw_e, bootstrap.rms, len(energies), UNITS["rmse_e"], bootstrap_n, seed
        )
    abs_f = bootstrap.group_values([(r.group_id, np.abs(r.df)) for r in res])
    raw_f = bootstrap.group_values([(r.group_id, r.df) for r in res])
    out["mae_f"] = _metric(abs_f, np.mean, len(res), UNITS["mae_f"], bootstrap_n, seed)
    out["rmse_f"] = _metric(raw_f, bootstrap.rms, len(res), UNITS["rmse_f"], bootstrap_n, seed)
    stressed = [r for r in res if r.ds is not None]
    if stressed:
        abs_s = bootstrap.group_values(
            [(r.group_id, np.abs(cast(np.ndarray, r.ds))) for r in stressed]
        )
        out["mae_s"] = _metric(abs_s, np.mean, len(stressed), UNITS["mae_s"], bootstrap_n, seed)
    return out


# --- the error table ------------------------------------------------------------------------------


def evaluate(
    model_path: str | Path,
    head: str,
    frames: Sequence[Frame],
    tier: str,
    reference: Reference | None,
    bootstrap_n: int,
    seed: int,
    *,
    noise_floor_f: float | None = None,
    model_label: str | None = None,
    split_id: str = "",
    run_id: str = "",
    energy_scale: str | None = None,
    e0_source: str | None = None,
    calc: Any | None = None,
    device: str = "cpu",
    dtype: str = "float64",
) -> ErrorTable:
    """Evaluate ``model_path`` (head ``head``) on ``frames`` and build the :class:`ErrorTable`.

    ``energy_scale`` is the model head's scale; energy metrics appear only when it equals the
    frames' scale (:func:`energy_metrics_allowed`). ``reference=None`` infers the reference
    from the frames' ``label_source``. ``calc`` may inject a ready calculator (the stage
    builds one per run; tests share one). ``noise_floor_f`` is stored verbatim (meV/Å).
    """
    if not frames:
        raise ValueError(f"tier {tier}: no frames to evaluate")
    if tier not in ALL_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {ALL_TIERS}")
    if head not in ("Default", "pt_head"):
        raise ValueError(f"head must be Default or pt_head, got {head!r}")
    calc = (
        calc if calc is not None else make_calculator(model_path, head, device=device, dtype=dtype)
    )
    with_energy, _ = energy_metrics_allowed(energy_scale, frames)
    res = residuals(calc, frames, with_energy=with_energy)
    metrics = metrics_from_residuals(res, bootstrap_n, seed)
    by_type: dict[str, dict[str, Metric]] = {}
    for config_type in sorted({r.config_type for r in res}):
        subset = [r for r in res if r.config_type == config_type]
        by_type[config_type] = metrics_from_residuals(subset, bootstrap_n, seed)
    ref = reference if reference is not None else reference_for(frames, e0_source=e0_source)
    return ErrorTable(
        model_sha256=sha256_file(model_path),
        model_label=model_label or label_for(None, model_path),
        head=cast(Head, head),
        tier=tier,
        split_id=split_id,
        reference=ref,
        n_frames=len(res),
        n_atoms=int(sum(r.n_atoms for r in res)),
        metrics=metrics,
        by_config_type=by_type,
        bootstrap_n=int(bootstrap_n),
        bootstrap_seed=int(seed),
        noise_floor_f=None if noise_floor_f is None else float(noise_floor_f),
        run_id=run_id,
    )


# --- noise floor (rule R5) ------------------------------------------------------------------------


def noise_floor_from_frames(
    qe_frames: Sequence[Frame], reference_frames: Sequence[Frame]
) -> dict[str, Any]:
    """Force RMSE (meV/Å) between QE re-labels and the frames they re-label, paired by
    ``frame_id`` (a re-label keeps the geometry hash); also the MAE and the pair count."""
    by_id = {f.frame_id: f for f in reference_frames if f.forces is not None}
    diffs: list[np.ndarray] = []
    for f in qe_frames:
        partner = by_id.get(f.frame_id)
        if f.forces is None or partner is None or partner.numbers != f.numbers:
            continue
        diffs.append(
            (np.asarray(f.forces, dtype=float) - np.asarray(partner.forces, dtype=float)).ravel()
        )
    if not diffs:
        raise ValueError("no frame_id pairs with forces between the QE re-labels and the frames")
    pooled = np.concatenate(diffs)
    return {
        "noise_floor_f": float(np.sqrt(np.mean(pooled**2))) * EV_TO_MEV,
        "mae_f": float(np.mean(np.abs(pooled))) * EV_TO_MEV,
        "n_frames": len(diffs),
        "n_components": int(pooled.size),
        "unit": "meV/Å",
    }


# --- numbers.json ---------------------------------------------------------------------------------


def numbers_for_table(
    table: ErrorTable,
    *,
    e0_source: str,
    energy_scale: str | None,
    noise_floor_reason: str | None = None,
) -> dict[str, Any]:
    """Flat ``numbers.json`` entries for one table (keys ``eval.errors.<tier>.<label>.<m>``)."""
    out: dict[str, Any] = {}
    reference = table.reference.model_dump(mode="json")
    if reference.get("e0_source") is None:
        reference["e0_source"] = e0_source
    for name, metric in table.metrics.items():
        key = f"eval.errors.{table.tier}.{table.model_label}.{name}"
        meta: dict[str, Any] = {
            "reference": reference,
            "head": table.head,
            "n": int(metric.n),
            "seed": int(table.bootstrap_seed),
            "ci95": None if metric.ci95 is None else [float(metric.ci95[0]), float(metric.ci95[1])],
            "e0_source": e0_source,
            "energy_scale": energy_scale,
            "model_label": table.model_label,
            "bracket": table.model_label,
            "tier": table.tier,
            "unit": metric.unit,
            "model_sha256": table.model_sha256,
            "split_id": table.split_id,
            "bootstrap_n": table.bootstrap_n,
            "n_frames": table.n_frames,
            "n_atoms": table.n_atoms,
        }
        if metric.ci95 is None:
            meta["ci95_reason"] = "fewer than two groups in this tier; no group bootstrap"
        if table.tier == "T3":
            meta["noise_floor_f"] = table.noise_floor_f
            if table.noise_floor_f is None:
                meta["noise_floor_reason"] = noise_floor_reason or (
                    "no noise floor given (pass --noise-floor or --noise-floor-frames)"
                )
        out[key] = float(metric.value)
        out[f"{key}@meta"] = meta
    return out


# --- stage ----------------------------------------------------------------------------------------


def parse_tiers(text: str | Sequence[str] | None) -> list[Tier] | None:
    if text is None:
        return None
    items = (
        [t.strip() for t in text.split(",")] if isinstance(text, str) else [str(t) for t in text]
    )
    tiers = [t for t in items if t]
    unknown = [t for t in tiers if t not in ALL_TIERS]
    if unknown:
        raise ValueError(f"unknown tiers {unknown}; expected a subset of {ALL_TIERS}")
    return [cast(Tier, t) for t in tiers]


def tier_frames(
    split: Split, frames: Sequence[Frame], tiers: Sequence[Tier]
) -> dict[Tier, list[Frame]]:
    by_id = {f.frame_id: f for f in frames}
    out: dict[Tier, list[Frame]] = {}
    for tier in tiers:
        ids = split.tiers.get(tier, [])
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(
                f"tier {tier}: {len(missing)} frame ids of split {split.split_id} are not in the "
                f"frames file (first: {missing[:3]})"
            )
        out[tier] = [by_id[i] for i in ids]
    return out


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    head: str = "Default",
    split: str | Path | None = None,
    tiers: str | Sequence[str] | None = None,
    frames_path: str | Path | None = None,
    tier: str | None = None,
    checkpoint_json: str | Path | None = None,
    energy_scale: str | None = None,
    label: str | None = None,
    e0_source: str | None = None,
    noise_floor_f: float | None = None,
    noise_floor_frames: str | Path | None = None,
    bootstrap_n: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Stage ``eval.errors``: one :class:`ErrorTable` per requested tier plus ``numbers.json``.

    Frames come from ``frames_path`` or from the extxyz matching ``split.frames_sha256``
    under ``cfg.paths.data_dir``; tiers from the :class:`Split` (default: every non-empty
    tier except T4b). Without a split the whole frames file is evaluated as ``tier``
    (default ``T0``; smoke runs). ``checkpoint_json`` (the train stage's ``checkpoint.json``)
    supplies the model's energy scale, E0 source and bracket label; ``energy_scale``,
    ``e0_source`` and ``label`` override or replace it. ``ctx.dry_run`` plans only.
    """
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if head not in ("Default", "pt_head"):
        raise ValueError(f"head must be Default or pt_head, got {head!r}")
    ctx.add_input(model_path, "model")
    model_sha = sha256_file(model_path)
    info: CheckpointInfo | None = None
    if checkpoint_json is not None:
        info = read_checkpoint(checkpoint_json)
        ctx.add_input(Path(checkpoint_json), "json")
        if info.sha256 != model_sha:
            raise ValueError(
                f"checkpoint.json sha256 {info.sha256[:12]}... does not match the model file "
                f"{model_sha[:12]}..."
            )
        if head not in info.heads:
            raise ValueError(f"checkpoint has heads {info.heads}, not {head!r}")
    scale: str | None = energy_scale if energy_scale is not None else model_energy_scale(info, head)
    if scale is not None and scale not in ("mp", "omat24", "qe", "none"):
        raise ValueError(f"energy_scale must be mp|omat24|qe|none, got {scale!r}")
    if e0_source is None:
        e0_source = (
            "foundation" if head == "pt_head" else (info.e0_source if info else UNSPECIFIED_E0)
        )
    model_label = label or label_for(info, model_path)
    b_n = int(cfg.eval.bootstrap_n if bootstrap_n is None else bootstrap_n)
    b_seed = int(
        seed
        if seed is not None
        else (ctx.seed if ctx.seed is not None else cfg.eval.bootstrap_seed)
    )

    split_obj: Split | None = None
    if split is not None:
        split_obj = read_split(split)
        ctx.add_input(Path(split), "json")
    if frames_path is None:
        if split_obj is None:
            raise ValueError("eval errors needs --split and/or --frames")
        frames_file = find_frames_for_split(split_obj, cfg.paths.data_dir)
    else:
        frames_file = Path(frames_path)
    if not frames_file.is_file():
        raise FileNotFoundError(f"frames file not found: {frames_file}")
    ctx.add_input(frames_file, "frames")
    all_frames = read_frames(frames_file)

    requested = parse_tiers(tiers)
    if split_obj is not None:
        if requested is None:
            requested = [t for t in DEFAULT_TIERS if split_obj.tiers.get(t)]
        per_tier = tier_frames(split_obj, all_frames, requested)
        split_id = split_obj.split_id
    else:
        one = cast(Tier, tier or (requested[0] if requested else "T0"))
        if one not in ALL_TIERS:
            raise ValueError(f"unknown tier {one!r}")
        per_tier = {one: list(all_frames)}
        split_id = ""
    empty = [t for t, fr in per_tier.items() if not fr]
    if empty:
        raise ValueError(f"tiers without frames: {empty}")

    floor_reason: str | None = None
    floor_info: dict[str, Any] | None = None
    if noise_floor_f is None and noise_floor_frames is not None:
        qe_path = Path(noise_floor_frames)
        ctx.add_input(qe_path, "frames")
        floor_info = noise_floor_from_frames(read_frames(qe_path), all_frames)
        noise_floor_f = float(floor_info["noise_floor_f"])
    if "T3" in per_tier and noise_floor_f is None:
        floor_reason = (
            "tier T3 evaluated without a noise floor (rule R5): pass --noise-floor or "
            "--noise-floor-frames"
        )
        warnings.warn(floor_reason, stacklevel=2)

    plan: dict[str, Any] = {
        "model": str(model_path),
        "model_sha256": model_sha,
        "head": head,
        "model_label": model_label,
        "energy_scale": scale,
        "e0_source": e0_source,
        "frames": str(frames_file),
        "split_id": split_id,
        "tiers": {t: len(fr) for t, fr in per_tier.items()},
        "bootstrap_n": b_n,
        "bootstrap_seed": b_seed,
        "noise_floor_f": noise_floor_f,
        "noise_floor": floor_info,
        "dry_run": bool(ctx.dry_run),
    }
    (ctx.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / "plan.json", "json")
    summary: dict[str, Any] = {
        "model_label": model_label,
        "head": head,
        "tiers": ",".join(per_tier),
        "n_frames": sum(len(fr) for fr in per_tier.values()),
    }
    if ctx.dry_run:
        ctx.log(model_sha256=model_sha, head=head, tier=list(per_tier), planned=True)
        return summary

    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(int(cfg.compute.threads))
    except Exception:  # noqa: BLE001 - threads are a courtesy, never a failure
        pass
    calc = make_calculator(model_path, head, device=cfg.compute.device, dtype=cfg.compute.dtype)
    numbers: dict[str, Any] = {}
    references: dict[str, Any] = {}
    counts: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for tier_name, frames in per_tier.items():
        allowed, reason = energy_metrics_allowed(scale, frames)
        if not allowed and reason:
            skipped[tier_name] = reason
        reference = reference_for(frames, e0_source=e0_source, pseudos=cfg.dft.pseudo_family)
        table = evaluate(
            model_path, head, frames, tier_name, reference, b_n, b_seed,
            noise_floor_f=noise_floor_f if tier_name == "T3" else None,
            model_label=model_label, split_id=split_id, run_id=ctx.run_id,
            energy_scale=scale, e0_source=e0_source, calc=calc,
        )  # fmt: skip
        path = ctx.out_dir / TABLE_FILE.format(tier=tier_name)
        path.write_text(table.model_dump_json(indent=2) + "\n", encoding="utf-8")
        ctx.add_output(path, "json")
        numbers.update(
            numbers_for_table(
                table, e0_source=e0_source, energy_scale=scale, noise_floor_reason=floor_reason
            )
        )
        references[tier_name] = reference.model_dump(mode="json")
        counts[tier_name] = table.n_frames
        summary[f"{tier_name}.mae_f"] = round(table.metrics["mae_f"].value, 3)
        summary[f"{tier_name}.rmse_f"] = round(table.metrics["rmse_f"].value, 3)
        summary[f"{tier_name}.n_frames"] = table.n_frames
        if "mae_e" in table.metrics:
            summary[f"{tier_name}.mae_e"] = round(table.metrics["mae_e"].value, 3)
        summary[f"{tier_name}.energy_metrics"] = "ok" if allowed else f"skipped: {reason}"
    numbers_path = ctx.out_dir / NUMBERS_FILE
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(numbers_path, "json")
    ctx.log(
        model_sha256=model_sha,
        head=head,
        tier=list(per_tier),
        n=counts,
        bootstrap_seed=b_seed,
        bootstrap_n=b_n,
        reference=references,
        energy_scale=scale,
        e0_source=e0_source,
        model_label=model_label,
        energy_metrics_skipped=skipped,
        noise_floor_f=noise_floor_f,
        noise_floor_reason=floor_reason,
    )
    if noise_floor_f is not None:
        summary["noise_floor_f"] = round(float(noise_floor_f), 3)
    elif floor_reason:
        summary["noise_floor_f"] = "missing"
    return summary


__all__ = [
    "ALL_TIERS",
    "DEFAULT_TIERS",
    "EV_TO_MEV",
    "NUMBERS_FILE",
    "QE_PSEUDOS",
    "REFERENCE_BY_SOURCE",
    "UNITS",
    "VARIANT_LABELS",
    "ZERO_SHOT_LABELS",
    "Residual",
    "energy_metrics_allowed",
    "evaluate",
    "frames_energy_scale",
    "label_for",
    "make_calculator",
    "metrics_from_residuals",
    "model_energy_scale",
    "noise_floor_from_frames",
    "numbers_for_table",
    "parse_tiers",
    "predict",
    "read_checkpoint",
    "reference_for",
    "residuals",
    "run",
    "tier_frames",
]
