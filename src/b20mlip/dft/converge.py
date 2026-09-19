"""``dft converge``: cutoff x k-spacing scan on the 8-atom cell and the cheapest converged point.

Scan: ``cfg.dft.converge_ecuts_ry`` x ``cfg.dft.converge_k_spacings`` (default 40..90 Ry x
0.35..0.20 1/Å = 24 units), ``ecutrho = dual x ecutwfc`` with the SSSP dual of the compound
(12 for Fe/Mn USPP/PAW, 8 otherwise). Deltas are taken against the densest point (highest
cutoff, smallest spacing): ``ΔE`` = |E/atom - E_ref/atom| in meV/atom, ``ΔF`` = max over atoms
and components of |F - F_ref| in meV/Å. The selected setting is the cheapest point
(cost ∝ ecut^1.5 x number of k-points) meeting ``cfg.dft.thresholds``; the JSON written to
``configs/dft/qe_<compound>_converged.json`` records every point so the choice is auditable.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from b20mlip.config import Settings
from b20mlip.dft import qe
from b20mlip.dft.stages import stage_result, submit_units, write_counts
from b20mlip.dft.structures import reference_frame
from b20mlip.models import Frame, StageResult, Status
from b20mlip.provenance import RunContext

SCHEMA = "b20mlip.qe_converged.v1"
SCAN_JSON = "scan.json"


def unit_id(compound: str, ecut_ry: float, k_spacing: float) -> str:
    return f"{compound}_ec{int(round(ecut_ry))}_k{k_spacing:g}"


def scan_points(
    cfg: Settings, symbols: Sequence[str], cell: Sequence[Sequence[float]]
) -> list[dict[str, Any]]:
    """Every (ecut, k-spacing) point of the scan with its ``ecut_rho`` (SSSP dual) and k-mesh."""
    _, _, dual = qe.sssp_cutoffs(symbols, cfg)
    points: list[dict[str, Any]] = []
    for ecut in cfg.dft.converge_ecuts_ry:
        for spacing in cfg.dft.converge_k_spacings:
            points.append(
                {
                    "ecut_ry": float(ecut),
                    "ecut_rho": float(round(dual * float(ecut), 6)),
                    "k_spacing_inv_A": float(spacing),
                    "kmesh": list(qe.kmesh(cell, float(spacing))),
                }
            )
    return points


def plan(cfg: Settings, compound: str, root: str | Path, frame: Frame) -> list[dict[str, Any]]:
    """Plan one unit per scan point (same geometry, different parameters); writes ``scan.json``."""
    points = scan_points(cfg, qe.species_order(frame.numbers), frame.cell)
    ids = [unit_id(compound, p["ecut_ry"], p["k_spacing_inv_A"]) for p in points]
    overrides = [
        {
            "ecut_ry": p["ecut_ry"],
            "ecut_rho": p["ecut_rho"],
            "k_spacing_inv_A": p["k_spacing_inv_A"],
        }
        for p in points
    ]
    qe.plan_units([frame] * len(points), root, cfg, unit_ids=ids, overrides=overrides)
    for uid, point in zip(ids, points, strict=True):
        point["unit_id"] = uid
    Path(root, SCAN_JSON).write_text(
        json.dumps({"compound": compound, "frame_id": frame.frame_id, "points": points}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return points


def cost(ecut_ry: float, kmesh: Sequence[int]) -> float:
    """Relative plane-wave SCF cost: number of plane waves ∝ ecut^1.5 times the k-point count."""
    return float(ecut_ry) ** 1.5 * float(np.prod([int(k) for k in kmesh]))


def analyse(
    points: Sequence[Mapping[str, Any]],
    thresholds: tuple[float, float],
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    """Deltas against the reference point and the cheapest setting within ``thresholds``.

    ``points``: ``unit_id, ecut_ry, ecut_rho, k_spacing_inv_A, kmesh, converged, energy_atom_eV,
    forces`` (eV/Å, ``None`` when missing). ``thresholds`` = ``(E_meV_atom, F_meV_A)``.
    """
    e_thr, f_thr = float(thresholds[0]), float(thresholds[1])
    usable = [p for p in points if p.get("converged") and p.get("energy_atom_eV") is not None]
    if not usable:
        raise ValueError("no converged scan point with an energy: nothing to analyse")
    if reference is not None:
        ref = next((p for p in usable if p["unit_id"] == reference), None)
        if ref is None:
            raise ValueError(f"reference unit {reference!r} is missing or unconverged")
    else:
        ref = max(usable, key=lambda p: (float(p["ecut_ry"]), -float(p["k_spacing_inv_A"])))
    ref_forces = None if ref.get("forces") is None else np.asarray(ref["forces"], dtype=float)
    rows: list[dict[str, Any]] = []
    for p in points:
        row = {
            "unit_id": p["unit_id"],
            "ecut_ry": float(p["ecut_ry"]),
            "ecut_rho": float(p["ecut_rho"]),
            "k_spacing_inv_A": float(p["k_spacing_inv_A"]),
            "kmesh": [int(k) for k in p["kmesh"]],
            "converged": bool(p.get("converged")),
            "cost": cost(float(p["ecut_ry"]), p["kmesh"]),
            "energy_atom_eV": p.get("energy_atom_eV"),
            "max_force_eVA": None
            if p.get("forces") is None
            else float(np.max(np.abs(np.asarray(p["forces"], dtype=float)))),
            "dE_meV_atom": None,
            "dF_meV_A": None,
            "meets": False,
        }
        if row["converged"] and p.get("energy_atom_eV") is not None:
            row["dE_meV_atom"] = (
                abs(float(p["energy_atom_eV"]) - float(ref["energy_atom_eV"])) * 1e3
            )
            if ref_forces is not None and p.get("forces") is not None:
                diff = np.asarray(p["forces"], dtype=float) - ref_forces
                if diff.shape == ref_forces.shape:
                    row["dF_meV_A"] = float(np.max(np.abs(diff))) * 1e3
            row["meets"] = (
                row["dE_meV_atom"] <= e_thr
                and row["dF_meV_A"] is not None
                and row["dF_meV_A"] <= f_thr
            )
        rows.append(row)
    candidates = [r for r in rows if r["meets"]]
    selected = (
        min(candidates, key=lambda r: (r["cost"], r["ecut_ry"], -r["k_spacing_inv_A"]))
        if candidates
        else None
    )
    return {
        "reference_unit": ref["unit_id"],
        "thresholds": {"E_meV_atom": e_thr, "F_meV_A": f_thr},
        "points": rows,
        "selected": selected,
    }


def collect_points(
    root: str | Path, points: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Attach parsed energies/forces to the scan points from the unit directories."""
    frames, counts = qe.collect(root)
    by_id = {f.unit_id: f for f in frames}
    out: list[dict[str, Any]] = []
    for p in points:
        frame = by_id.get(p["unit_id"])
        row = dict(p)
        row["converged"] = bool(frame is not None and frame.converged)
        row["energy_atom_eV"] = (
            None
            if frame is None or frame.energy is None
            else float(frame.energy) / len(frame.numbers)
        )
        row["forces"] = None if frame is None else frame.forces
        out.append(row)
    return out, counts


def converged_settings(
    compound: str, analysis: Mapping[str, Any], *, run_id: str, frame_id: str
) -> dict[str, Any]:
    sel = analysis["selected"]
    return {
        "schema": SCHEMA,
        "compound": compound,
        "frame_id": frame_id,
        "ecut_ry": None if sel is None else sel["ecut_ry"],
        "ecut_rho": None if sel is None else sel["ecut_rho"],
        "k_spacing_inv_A": None if sel is None else sel["k_spacing_inv_A"],
        "kmesh": None if sel is None else sel["kmesh"],
        "selected_unit": None if sel is None else sel["unit_id"],
        "reference_unit": analysis["reference_unit"],
        "thresholds": analysis["thresholds"],
        "points": analysis["points"],
        "units": {"energy": "meV/atom", "force": "meV/A", "ecut": "Ry", "k_spacing": "1/A"},
        "source_run_id": run_id,
    }


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    compound: str,
    structure: str | Path | None = None,
    units: str | Path | None = None,
    out: str | Path | None = None,
    wait: bool = True,
    collect_only: bool = False,
    template: str = "qe_array",
    resources: Mapping[str, Any] | None = None,
) -> StageResult:
    """Plan (and run) the scan, then select the cheapest converged setting."""
    root = Path(units) if units is not None else Path(cfg.paths.dft_dir) / f"converge_{compound}"
    out_path = (
        Path(out) if out is not None else Path("configs/dft") / f"qe_{compound}_converged.json"
    )
    frame = reference_frame(cfg, compound, structure)
    if structure is not None:
        ctx.add_input(Path(structure), "frames")
    if collect_only and (root / SCAN_JSON).is_file():
        points = json.loads((root / SCAN_JSON).read_text(encoding="utf-8"))["points"]
    else:
        points = plan(cfg, compound, root, frame)
    summary: dict[str, Any] = {
        "compound": compound,
        "units_root": str(root),
        "n_points": len(points),
        "frame_id": frame.frame_id,
    }
    if ctx.dry_run:
        ctx.log(n_units=len(points), dry_run=True, units_root=str(root))
        return stage_result(ctx, "partial", summary)
    pending = qe.pending_units(root, [p["unit_id"] for p in points])
    if pending and not collect_only:
        submit_units(cfg, ctx, root, pending, template=template, resources=resources, wait=wait)
    rows, counts = collect_points(root, points)
    write_counts(ctx, counts)
    ctx.log(
        n_units=len(points),
        n_failed=counts["failed"] + counts["unconverged"],
        counts=dict(counts),
        units_root=str(root),
        pseudo_md5s={
            s: cfg.dft.pseudo_md5s[s]
            for s in qe.species_order(frame.numbers)
            if s in cfg.dft.pseudo_md5s
        },
    )
    if counts["done"] == 0:
        return stage_result(
            ctx, "partial" if counts["missing"] else "failed", {**summary, **counts}
        )
    thresholds = (cfg.dft.thresholds.E_meV_atom, cfg.dft.thresholds.F_meV_A)
    analysis = analyse(rows, thresholds)
    settings = converged_settings(compound, analysis, run_id=ctx.run_id, frame_id=frame.frame_id)
    local = ctx.out_dir / "converge.json"
    local.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(local, "json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, out_path)
    ctx.add_output(out_path, "json")
    sel = analysis["selected"]
    ctx.log(
        selected=None if sel is None else sel["unit_id"], reference_unit=analysis["reference_unit"]
    )
    status: Status = "ok" if sel is not None and counts["missing"] == 0 else "partial"
    return stage_result(
        ctx,
        status,
        {
            **summary,
            **counts,
            "reference_unit": analysis["reference_unit"],
            "selected_unit": None if sel is None else sel["unit_id"],
            "ecut_ry": None if sel is None else sel["ecut_ry"],
            "ecut_rho": None if sel is None else sel["ecut_rho"],
            "k_spacing_inv_A": None if sel is None else sel["k_spacing_inv_A"],
            "dE_meV_atom": None if sel is None else sel["dE_meV_atom"],
            "dF_meV_A": None if sel is None else sel["dF_meV_A"],
            "out": str(out_path),
        },
    )


__all__ = [
    "SCHEMA",
    "analyse",
    "collect_points",
    "converged_settings",
    "cost",
    "plan",
    "run",
    "scan_points",
    "unit_id",
]
