"""Elastic constants and equation of state of a model (CONTRACTS.md row 9, SPEC.md section 6).

Conventions (binding):

* **Strain**: ``ε`` is the symmetric small-strain tensor; a deformed cell has lattice vectors
  ``a_i' = (I + ε) a_i`` (rows of ``atoms.cell`` → ``cell @ (I + ε).T``), atoms follow
  (``scale_atoms=True``) and the internal coordinates are then relaxed at fixed cell (FIRE,
  ``fmax`` 1e-3 eV/Å) before the stress is read. Voigt order ``xx yy zz yz xz xy`` with
  **engineering** shear strains ``γ_yz = 2 ε_yz`` (so ``σ_yz = C44 γ_yz``).
* **Stress**: ASE's sign convention, ``σ_ij = (1/V) ∂E/∂ε_ij`` (positive = tensile; ASE's
  ``pressure = −tr σ / 3``), Voigt-6 in eV/Å³ from ``atoms.get_stress(voigt=True)``,
  converted to GPa with ``ase.units.GPa`` (1 eV/Å³ = 160.2177 GPa). The residual stress of
  the unstrained (internally relaxed) cell ``σ0`` is subtracted from every strained stress.
* **Constants**: ``C_ij = ∂(σ_i − σ0_i)/∂ε_j`` from a least-squares slope through the origin
  over ``±ε`` for every magnitude in ``strains`` (default 0.5 % and 1 %). For a cubic cell
  the ``ε_xx`` mode gives ``C11 = C_11`` and ``C12 = (C_21 + C_31)/2`` and the ``γ_yz`` mode
  gives ``C44 = C_44``; ``B = (C11 + 2 C12)/3``. ``elastic_tensor`` computes any subset of
  the six Voigt modes (all six for the symmetry check).
* **EOS**: ``npoints`` (default 9) volumes from ``1 − pct/100`` to ``1 + pct/100`` of the
  input volume (isotropic scaling, internal coordinates relaxed), Birch–Murnaghan fit with
  ``ase.eos.EquationOfState(eos="birchmurnaghan")``: ``V0_A3``, ``B0_GPa``, ``B0_prime``
  (``B'``), ``E0_eV``; ``a0_A = a_in · (V0/V_in)^{1/3}`` (the lattice constant for a cubic
  cell); ``residual_meV_atom`` is the RMS fit residual per atom.

``run`` publishes ``eval.elastic.<compound>.<label>.{C11,C12,C44,B,a0,B0}`` (GPa / Å) with
reference code ``mace`` (the model's own prediction; DFT/experiment values are published by
the tiers that compute them).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.eos import EquationOfState
from ase.optimize import FIRE

from b20mlip.config import Settings
from b20mlip.evaluate.discovery import relax
from b20mlip.evaluate.errors import (
    UNSPECIFIED_E0,
    label_for,
    make_calculator,
    model_energy_scale,
    read_checkpoint,
)
from b20mlip.io import frame_to_atoms
from b20mlip.provenance import RunContext, sha256_file

DEFAULT_STRAINS: tuple[float, ...] = (0.005, 0.01)
CUBIC_MODES: tuple[int, ...] = (0, 3)  # ε_xx and γ_yz
ALL_MODES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
INTERNAL_FMAX = 1e-3
INTERNAL_STEPS = 300
EOS_NPOINTS = 9
EV_A3_TO_GPA = 1.0 / units.GPa
NUMBERS_FILE = "numbers.json"
RESULTS_FILE = "elastic.json"
MODEL_REFERENCE: dict[str, Any] = {
    "code": "mace",
    "functional": "PBE",
    "pseudos": None,
    "e0_source": None,
    "cross_functional": False,
}


# --- strain algebra -----------------------------------------------------------------------------


def strain_tensor(voigt: Sequence[float]) -> np.ndarray:
    """Symmetric strain tensor from Voigt components (engineering shears)."""
    v = [float(x) for x in voigt]
    if len(v) != 6:
        raise ValueError("Voigt strain needs 6 components")
    eps = np.zeros((3, 3))
    eps[0, 0], eps[1, 1], eps[2, 2] = v[0], v[1], v[2]
    eps[1, 2] = eps[2, 1] = v[3] / 2.0
    eps[0, 2] = eps[2, 0] = v[4] / 2.0
    eps[0, 1] = eps[1, 0] = v[5] / 2.0
    return eps


def strained(atoms: Atoms, voigt: Sequence[float]) -> Atoms:
    """A copy of ``atoms`` with the cell deformed by ``I + ε`` (atoms scaled along)."""
    out = atoms.copy()
    deform = np.eye(3) + strain_tensor(voigt)
    out.set_cell(np.asarray(atoms.cell[:], dtype=float) @ deform.T, scale_atoms=True)
    return out


def relax_positions(
    atoms: Atoms, calc: Any, fmax: float = INTERNAL_FMAX, steps: int = INTERNAL_STEPS
) -> tuple[Atoms, bool]:
    """Relax internal coordinates at fixed cell; ``(atoms with calc attached, converged)``."""
    work = atoms.copy()
    work.calc = calc
    opt = FIRE(work, logfile=None)
    converged = bool(opt.run(fmax=float(fmax), steps=int(steps)))
    return work, converged


def stress_gpa(
    atoms: Atoms, calc: Any, *, fmax: float = INTERNAL_FMAX, steps: int = INTERNAL_STEPS
) -> tuple[np.ndarray, float, bool]:
    """``(Voigt-6 stress in GPa, energy eV, converged)`` after relaxing the internal coordinates."""
    work, converged = relax_positions(atoms, calc, fmax, steps)
    stress = np.asarray(work.get_stress(voigt=True), dtype=float) * EV_A3_TO_GPA
    return stress, float(work.get_potential_energy()), converged


def _slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(x * y) / np.sum(x * x))


def elastic_tensor(
    atoms: Atoms,
    calc: Any,
    strains: Sequence[float] = DEFAULT_STRAINS,
    modes: Sequence[int] = ALL_MODES,
    *,
    fmax: float = INTERNAL_FMAX,
    steps: int = INTERNAL_STEPS,
) -> dict[str, Any]:
    """``C`` (6x6, GPa; columns of modes not in ``modes`` are NaN) plus details.

    Returns ``{"C_GPa", "sigma0_GPa", "points": [{mode, strain, stress_GPa}], "converged"}``.
    """
    mags = [float(s) for s in strains]
    if not mags or any(m <= 0 for m in mags):
        raise ValueError("strains must be positive magnitudes")
    sigma0, _, conv0 = stress_gpa(atoms, calc, fmax=fmax, steps=steps)
    C = np.full((6, 6), np.nan)
    points: list[dict[str, Any]] = []
    converged = conv0
    for mode in modes:
        xs: list[float] = []
        ys: list[np.ndarray] = []
        for mag in mags:
            for sign in (-1.0, 1.0):
                voigt = [0.0] * 6
                voigt[mode] = sign * mag
                stress, _, conv = stress_gpa(strained(atoms, voigt), calc, fmax=fmax, steps=steps)
                converged = converged and conv
                xs.append(sign * mag)
                ys.append(stress - sigma0)
                points.append(
                    {"mode": int(mode), "strain": sign * mag, "stress_GPa": stress.tolist()}
                )
        x = np.asarray(xs)
        y = np.asarray(ys)  # (n_points, 6)
        for j in range(6):
            C[j, mode] = _slope_through_origin(x, y[:, j])
    return {
        "C_GPa": C,
        "sigma0_GPa": sigma0,
        "points": points,
        "converged": bool(converged),
        "strains": mags,
        "modes": [int(m) for m in modes],
    }


def constants(
    atoms: Atoms,
    calc: Any,
    strains: Sequence[float] = DEFAULT_STRAINS,
    *,
    full: bool = False,
    fmax: float = INTERNAL_FMAX,
    steps: int = INTERNAL_STEPS,
) -> dict[str, float]:
    """Cubic elastic constants ``C11, C12, C44, B`` (GPa) by finite differences of the stress.

    ``full=True`` computes all six Voigt modes and adds ``max_asymmetry_GPa`` (``max |C −
    Cᵀ|``, a consistency check) and the other cubic estimates (``C11_yy``, ``C44_xz`` …).
    """
    modes = ALL_MODES if full else CUBIC_MODES
    tensor = elastic_tensor(atoms, calc, strains, modes, fmax=fmax, steps=steps)
    C = tensor["C_GPa"]
    out: dict[str, float] = {
        "C11": float(C[0, 0]),
        "C12": float((C[1, 0] + C[2, 0]) / 2.0),
        "C44": float(C[3, 3]),
    }
    out["B"] = (out["C11"] + 2.0 * out["C12"]) / 3.0
    out["C12_yy"] = float(C[1, 0])
    out["C12_zz"] = float(C[2, 0])
    out["sigma0_max_abs_GPa"] = float(np.max(np.abs(tensor["sigma0_GPa"])))
    out["converged"] = 1.0 if tensor["converged"] else 0.0
    out["n_points"] = float(len(tensor["points"]))
    if full:
        out["C11_yy"] = float(C[1, 1])
        out["C11_zz"] = float(C[2, 2])
        out["C44_xz"] = float(C[4, 4])
        out["C44_xy"] = float(C[5, 5])
        out["max_asymmetry_GPa"] = float(np.max(np.abs(C - C.T)))
        for i in range(6):
            for j in range(6):
                out[f"C{i + 1}{j + 1}_GPa"] = float(C[i, j])
    return out


# --- equation of state ----------------------------------------------------------------------------


def eos(
    atoms: Atoms,
    calc: Any,
    pct: float = 8.0,
    *,
    npoints: int = EOS_NPOINTS,
    fmax: float = INTERNAL_FMAX,
    steps: int = INTERNAL_STEPS,
) -> dict[str, Any]:
    """Birch–Murnaghan EOS over ``±pct`` % volume: ``a0_A, V0_A3, B0_GPa, B0_prime, residual``."""
    if pct <= 0 or npoints < 4:
        raise ValueError("eos needs pct > 0 and at least 4 points")
    factors = np.linspace(1.0 - pct / 100.0, 1.0 + pct / 100.0, int(npoints))
    cell0 = np.asarray(atoms.cell[:], dtype=float)
    v_in = float(atoms.get_volume())
    volumes: list[float] = []
    energies: list[float] = []
    converged = True
    for f in factors:
        scaled = atoms.copy()
        scaled.set_cell(cell0 * f ** (1.0 / 3.0), scale_atoms=True)
        work, conv = relax_positions(scaled, calc, fmax, steps)
        converged = converged and conv
        volumes.append(float(work.get_volume()))
        energies.append(float(work.get_potential_energy()))
    fit = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, b = fit.fit(warn=False)
    params = np.asarray(fit.eos_parameters, dtype=float)  # E0, B0, BP, V0
    model = fit.func(np.asarray(volumes), *params)
    residual = float(np.sqrt(np.mean((np.asarray(energies) - model) ** 2))) / len(atoms) * 1e3
    a_in = float(np.cbrt(v_in)) if _is_cubic(cell0) else float(np.linalg.norm(cell0[0]))
    return {
        "a0_A": a_in * (float(v0) / v_in) ** (1.0 / 3.0),
        "V0_A3": float(v0),
        "B0_GPa": float(b) * EV_A3_TO_GPA,
        "B0_prime": float(params[2]),
        "E0_eV": float(e0),
        "residual_meV_atom": residual,
        "v0_in_range": bool(min(volumes) < v0 < max(volumes)),
        "converged": bool(converged),
        "n_points": int(npoints),
        "pct": float(pct),
        "volumes_A3": volumes,
        "energies_eV": energies,
        "a_in_A": a_in,
        "cubic": _is_cubic(cell0),
    }


def _is_cubic(cell: np.ndarray, tol: float = 1e-6) -> bool:
    lengths = np.linalg.norm(cell, axis=1)
    if not np.allclose(lengths, lengths[0], atol=tol):
        return False
    return bool(np.allclose(cell @ cell.T, np.eye(3) * lengths[0] ** 2, atol=tol))


# --- stage ----------------------------------------------------------------------------------------


def reference_atoms(cfg: Settings, compound: str, structure: str | Path | None) -> Atoms:
    """The compound's reference cell (``dft.structures.reference_frame``), labels stripped."""
    from b20mlip.dft.structures import reference_frame  # noqa: PLC0415 - row 6, allowed

    frame = reference_frame(cfg, compound, structure)
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    atoms.info = {}
    return atoms


def numbers_for_run(
    results: dict[str, Any],
    *,
    compound: str,
    label: str,
    head: str,
    e0_source: str,
    energy_scale: str | None,
    model_sha256: str,
    cell_source: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "reference": dict(MODEL_REFERENCE, e0_source=e0_source),
        "head": head,
        "seed": 0,
        "ci95": None,
        "ci95_reason": "deterministic finite-difference quantity; no resampling",
        "e0_source": e0_source,
        "energy_scale": energy_scale,
        "model_label": label,
        "bracket": label,
        "compound": compound,
        "cell_source": cell_source,
        "model_sha256": model_sha256,
        "tier": "elastic",
    }
    out: dict[str, Any] = {}
    consts = results.get("constants")
    if consts:
        n = int(consts.get("n_points", 0)) or 1
        for name in ("C11", "C12", "C44", "B"):
            value = consts.get(name)
            if value is None or not math.isfinite(float(value)):
                continue
            key = f"eval.elastic.{compound}.{label}.{name}"
            out[key] = float(value)
            out[f"{key}@meta"] = {**base, "n": n, "unit": "GPa", "strains": results.get("strains")}
    fit = results.get("eos")
    if fit:
        for name, field, unit in (("a0", "a0_A", "Å"), ("B0", "B0_GPa", "GPa")):
            value = fit.get(field)
            if value is None or not math.isfinite(float(value)):
                continue
            key = f"eval.elastic.{compound}.{label}.{name}"
            out[key] = float(value)
            out[f"{key}@meta"] = {
                **base,
                "n": int(fit["n_points"]),
                "unit": unit,
                "eos": "birchmurnaghan",
                "pct": fit["pct"],
            }
    return out


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    compound: str,
    structure: str | Path | None = None,
    cell: str = "dft",
    strains: Sequence[float] = DEFAULT_STRAINS,
    pct: float = 8.0,
    npoints: int = EOS_NPOINTS,
    head: str = "Default",
    label: str | None = None,
    checkpoint_json: str | Path | None = None,
    e0_source: str | None = None,
    do_constants: bool = True,
    do_eos: bool = True,
    full: bool = False,
    calc: Any | None = None,
) -> dict[str, Any]:
    """Stage ``eval.elastic``: ``C11/C12/C44/B`` and the EOS of ``compound`` with the model.

    ``cell="dft"`` uses the reference cell as given; ``cell="model"`` relaxes it with the model
    first (FixSymmetry + FrechetCellFilter, ``cfg.eval`` fmax/steps). A failed EOS fit is
    recorded (``eos_error``) instead of failing the stage; its numbers are then absent.
    """
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if cell not in ("dft", "model"):
        raise ValueError(f"cell must be dft or model, got {cell!r}")
    ctx.add_input(model_path, "model")
    model_sha = sha256_file(model_path)
    info = None
    if checkpoint_json is not None:
        info = read_checkpoint(checkpoint_json)
        ctx.add_input(Path(checkpoint_json), "json")
        if info.sha256 != model_sha:
            raise ValueError("checkpoint.json sha256 does not match the model file")
    if e0_source is None:
        e0_source = (
            "foundation" if head == "pt_head" else (info.e0_source if info else UNSPECIFIED_E0)
        )
    model_label = label or label_for(info, model_path)
    scale = model_energy_scale(info, head)
    if structure is not None:
        ctx.add_input(Path(structure), "frames")
    atoms = reference_atoms(cfg, compound, structure)
    results: dict[str, Any] = {
        "compound": compound,
        "model": str(model_path),
        "model_sha256": model_sha,
        "model_label": model_label,
        "head": head,
        "cell_source": "dft" if cell == "dft" else "model_relaxed",
        "a_in_A": float(np.cbrt(atoms.get_volume())),
        "strains": [float(s) for s in strains],
        "pct": float(pct),
        "npoints": int(npoints),
        "dry_run": bool(ctx.dry_run),
    }
    if ctx.dry_run:
        (ctx.out_dir / RESULTS_FILE).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        ctx.log(model_sha256=model_sha, head=head, compound=compound, planned=True)
        return {"compound": compound, "model_label": model_label, "planned": 1}
    calc = (
        calc
        if calc is not None
        else make_calculator(model_path, head, device=cfg.compute.device, dtype=cfg.compute.dtype)
    )
    if cell == "model":
        relaxed, n_steps = relax(
            atoms, calc, fmax=cfg.eval.fmax, steps=cfg.eval.max_steps, fix_symmetry=True
        )
        relaxed.calc = None
        relaxed.set_constraint()
        results["relax_steps"] = n_steps
        results["a_model_A"] = float(np.cbrt(relaxed.get_volume()))
        atoms = relaxed
    if do_constants:
        results["constants"] = constants(atoms, calc, strains, full=full)
    if do_eos:
        try:
            results["eos"] = eos(atoms, calc, pct, npoints=npoints)
        except (RuntimeError, ValueError, TypeError) as exc:  # curve_fit failed to converge
            results["eos"] = None
            results["eos_error"] = f"{type(exc).__name__}: {exc}"
    path = ctx.out_dir / RESULTS_FILE
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(path, "json")
    numbers = numbers_for_run(
        results, compound=compound, label=model_label, head=head, e0_source=e0_source,
        energy_scale=scale, model_sha256=model_sha, cell_source=results["cell_source"],
    )  # fmt: skip
    numbers_path = ctx.out_dir / NUMBERS_FILE
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(numbers_path, "json")
    ctx.log(
        model_sha256=model_sha, head=head, tier="elastic", compound=compound,
        n=len(results.get("constants", {}) or {}) and int(results["constants"]["n_points"]),
        bootstrap_seed=None, reference=MODEL_REFERENCE, e0_source=e0_source,
        energy_scale=scale, model_label=model_label, cell_source=results["cell_source"],
        eos_error=results.get("eos_error"),
    )  # fmt: skip
    summary: dict[str, Any] = {
        "compound": compound,
        "model_label": model_label,
        "cell_source": results["cell_source"],
    }
    if results.get("constants"):
        for name in ("C11", "C12", "C44", "B"):
            summary[f"{name}_GPa"] = round(float(results["constants"][name]), 3)
    if results.get("eos"):
        summary["a0_A"] = round(float(results["eos"]["a0_A"]), 5)
        summary["B0_GPa"] = round(float(results["eos"]["B0_GPa"]), 3)
    elif results.get("eos_error"):
        summary["eos"] = f"failed: {results['eos_error']}"
    return summary


__all__ = [
    "ALL_MODES",
    "CUBIC_MODES",
    "DEFAULT_STRAINS",
    "EOS_NPOINTS",
    "EV_A3_TO_GPA",
    "INTERNAL_FMAX",
    "INTERNAL_STEPS",
    "MODEL_REFERENCE",
    "NUMBERS_FILE",
    "RESULTS_FILE",
    "constants",
    "elastic_tensor",
    "eos",
    "numbers_for_run",
    "reference_atoms",
    "relax_positions",
    "run",
    "strain_tensor",
    "strained",
    "stress_gpa",
]
