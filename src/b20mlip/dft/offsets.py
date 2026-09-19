"""``dft offsets``: per-element linear map between the QE and the MP (MPtrj) energy scales.

Pairs frames by ``frame_id`` (the QE re-labels of the 60 MPtrj B20 frames keep the geometry
hash), then fits per atom::

    (E_mp - E_qe) / N  ≈  Σ_e x_e c_e ,   x_e = n_e / N  (element fractions)

by least squares. ``residual_meV_atom`` is the RMS residual; gate R3 (SPEC.md section 4) is
``residual ≤ cfg.eval.offset_residual_gate_meV``. With only stoichiometric AB compounds the design
matrix is rank-deficient (FeSi, MnSi, CoSi, FeGe span 4 directions for 5 elements): the residual
and the *predicted* offsets of those compositions are unique, the individual coefficients are the
minimum-norm solution and ``unique=false`` says so.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from b20mlip.config import Settings
from b20mlip.dft.stages import stage_result
from b20mlip.io import read_frames
from b20mlip.models import Frame, StageResult
from b20mlip.provenance import RunContext

SCHEMA = "b20mlip.offsets.v1"


def pair_frames(qe: Sequence[Frame], mp: Sequence[Frame]) -> list[tuple[Frame, Frame]]:
    """``(qe, mp)`` pairs with the same ``frame_id`` and an energy on both sides (qe order)."""
    by_id: dict[str, Frame] = {}
    for f in mp:
        if f.energy is not None:
            by_id.setdefault(f.frame_id, f)
    pairs: list[tuple[Frame, Frame]] = []
    seen: set[str] = set()
    for f in qe:
        if f.energy is None or f.frame_id in seen:
            continue
        partner = by_id.get(f.frame_id)
        if partner is None or partner.numbers != f.numbers:
            continue
        pairs.append((f, partner))
        seen.add(f.frame_id)
    return pairs


def fit(qe: Sequence[Frame], mp: Sequence[Frame]) -> dict[str, Any]:
    """Least-squares per-element offsets (eV/atom of that element) and the residual in meV/atom."""
    pairs = pair_frames(qe, mp)
    if not pairs:
        raise ValueError("no frame_id pairs with energies between the QE and MP frame sets")
    numbers = sorted({int(z) for f, _ in pairs for z in f.numbers})
    symbols = [chemical_symbols[z] for z in numbers]
    n_atoms = np.array([len(f.numbers) for f, _ in pairs], dtype=np.float64)
    e_qe = np.array([float(f.energy or 0.0) for f, _ in pairs], dtype=np.float64)
    e_mp = np.array([float(m.energy or 0.0) for _, m in pairs], dtype=np.float64)
    design = np.array(
        [[f.numbers.count(z) / len(f.numbers) for z in numbers] for f, _ in pairs],
        dtype=np.float64,
    )
    target = (e_mp - e_qe) / n_atoms
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    rms = float(np.sqrt(np.mean(residual**2))) * 1e3
    return {
        "schema": SCHEMA,
        "model": "E_mp ≈ E_qe + Σ_e n_e·c_e (least squares on energies per atom)",
        "coefficients": {s: float(c) for s, c in zip(symbols, coef, strict=True)},
        "residual_meV_atom": rms,
        "max_abs_residual_meV_atom": float(np.max(np.abs(residual))) * 1e3,
        "n": len(pairs),
        "rank": int(rank),
        "n_elements": len(symbols),
        "unique": bool(rank == len(symbols)),
        "pairs": [
            {
                "frame_id": f.frame_id,
                "compound": f.compound,
                "n_atoms": int(n),
                "e_qe_eV": float(eq),
                "e_mp_eV": float(em),
                "residual_meV_atom": float(r) * 1e3,
            }
            for (f, _), n, eq, em, r in zip(pairs, n_atoms, e_qe, e_mp, residual, strict=True)
        ],
        "units": {"coefficients": "eV per atom of the element", "residual": "meV/atom"},
    }


def run(
    cfg: Settings, ctx: RunContext, *, qe: str | Path, mp: str | Path, out: str | Path
) -> StageResult:
    qe_path, mp_path, out_path = Path(qe), Path(mp), Path(out)
    ctx.add_input(qe_path, "frames")
    ctx.add_input(mp_path, "frames")
    result = fit(read_frames(qe_path), read_frames(mp_path))
    gate = float(cfg.eval.offset_residual_gate_meV)
    result["gate_meV_atom"] = gate
    result["passed"] = bool(result["residual_meV_atom"] <= gate)
    result["source_run_id"] = ctx.run_id
    local = ctx.out_dir / "offsets.json"
    local.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(local, "json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, out_path)
    ctx.add_output(out_path, "json")
    ctx.log(
        n=result["n"],
        residual_meV_atom=result["residual_meV_atom"],
        gate_meV_atom=gate,
        passed=result["passed"],
        coefficients=result["coefficients"],
        unique=result["unique"],
    )
    return stage_result(
        ctx,
        "ok",
        {
            "n": result["n"],
            "residual_meV_atom": result["residual_meV_atom"],
            "gate_meV_atom": gate,
            "passed": result["passed"],
            "unique": result["unique"],
            "out": str(out_path),
        },
    )


__all__ = ["SCHEMA", "fit", "pair_frames", "run"]
