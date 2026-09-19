"""``dft e0s``: isolated-atom QE energies in the MACE ``--E0s`` dict format (``E0s_qe.json``).

One spin-polarised unit per element: a single atom centred in a cubic box of
``cfg.dft.isolated_atom_box_A`` (12 Å), Γ point only (``K_POINTS gamma``), ``nspin=2`` with
``starting_magnetization=1.0`` (fully polarised start, so Mn/Fe/Co relax into their high-spin
atomic ground states), the same cutoffs, smearing and pseudopotentials as the solids so the
energies live on the QE scale of the training frames (SPEC.md R1). Output::

    {"14": E_Si, "25": E_Mn, "26": E_Fe, "27": E_Co, "32": E_Ge}   # eV, atomic-number keys

exactly what ``mace_run_train --E0s configs/dft/E0s_qe.json`` loads; a sidecar
``E0s_qe.meta.json`` records the parameters and the units the numbers came from.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ase.data import atomic_numbers

from b20mlip.config import Settings
from b20mlip.dft import qe
from b20mlip.dft.stages import stage_result, submit_units, write_counts
from b20mlip.dft.structures import elements_of
from b20mlip.io import frame_id_for
from b20mlip.models import Frame, StageResult, Status
from b20mlip.provenance import RunContext

DEFAULT_OUT = Path("configs/dft/E0s_qe.json")
STARTING_MAGNETIZATION = 1.0


def unit_id(element: str) -> str:
    return f"E0_{element}"


def isolated_atom_frame(element: str, box_A: float) -> Frame:
    """One atom of ``element`` at the centre of a ``box_A`` cubic cell."""
    if element not in atomic_numbers:
        raise ValueError(f"unknown element {element!r}")
    numbers = [int(atomic_numbers[element])]
    positions = [[box_A / 2.0] * 3]
    cell = [[box_A, 0.0, 0.0], [0.0, box_A, 0.0], [0.0, 0.0, box_A]]
    return Frame(
        frame_id=frame_id_for(numbers, positions, cell),
        group_id=f"{element}/offset/{unit_id(element)}",
        compound=element,
        config_type="offset",  # ConfigType has no "isolated_atom" (contract note); flagged in info
        parent_id=unit_id(element),
        numbers=numbers,
        positions=positions,
        cell=cell,
        info={"isolated_atom": True, "box_A": float(box_A)},
    )


def overrides_for(element: str) -> dict[str, Any]:
    return {
        "nspin": 2,
        "starting_magnetization": {element: STARTING_MAGNETIZATION},
        "kpoints": "gamma",
    }


def plan(cfg: Settings, elements: Sequence[str], root: str | Path) -> list[str]:
    frames = [isolated_atom_frame(el, cfg.dft.isolated_atom_box_A) for el in elements]
    ids = [unit_id(el) for el in elements]
    return qe.plan_units(
        frames, root, cfg, unit_ids=ids, overrides=[overrides_for(el) for el in elements]
    )


def collect(root: str | Path, elements: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    """``{"<Z>": E_eV}`` for every converged element plus per-element details."""
    frames, counts = qe.collect(root)
    by_id = {f.unit_id: f for f in frames}
    energies: dict[str, float] = {}
    details: dict[str, Any] = {}
    for el in elements:
        f = by_id.get(unit_id(el))
        if f is None:
            details[el] = {"status": "missing"}
            continue
        details[el] = {
            "status": "done" if f.converged else "unconverged",
            "energy_eV": f.energy,
            "total_magnetization_muB": f.total_magnetization,
            "scf_steps": f.scf_steps,
            "wall_seconds": f.wall_seconds,
        }
        if f.converged and f.energy is not None:
            energies[str(atomic_numbers[el])] = float(f.energy)
    return dict(sorted(energies.items(), key=lambda kv: int(kv[0]))), {**details, "counts": counts}


def write_e0s(
    energies: dict[str, float], out: str | Path, meta: dict[str, Any] | None = None
) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(energies, indent=2) + "\n", encoding="utf-8")
    if meta is not None:
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return p


def isolated_atoms(
    cfg: Settings,
    elements: Sequence[str],
    *,
    root: str | Path | None = None,
    executor: Any | None = None,
    ctx: RunContext | None = None,
    out: str | Path | None = None,
    wait: bool = True,
) -> dict[str, float]:
    """Plan the isolated-atom units, run them when an executor context is given, collect energies.

    Without ``ctx`` (no executor) this plans and collects whatever outputs already exist, so the
    units can be run by any means and re-collected. Writes ``out`` (MACE ``--E0s`` format) when
    every element converged.
    """
    root = Path(root) if root is not None else Path(cfg.paths.dft_dir) / "e0s"
    ids = plan(cfg, elements, root)
    if ctx is not None and (executor is not None or ctx.executor is not None):
        if executor is not None and ctx.executor is None:
            ctx.executor = executor
        pending = qe.pending_units(root, ids)
        if pending:
            submit_units(cfg, ctx, root, pending, template="qe_array", wait=wait)
    energies, details = collect(root, elements)
    if out is not None and len(energies) == len(elements):
        write_e0s(
            energies,
            out,
            {
                "units": "eV",
                "keys": "atomic numbers (MACE --E0s format)",
                "elements": {el: details[el] for el in elements},
                "settings": {
                    "box_A": cfg.dft.isolated_atom_box_A,
                    "nspin": 2,
                    "starting_magnetization": STARTING_MAGNETIZATION,
                    "kpoints": "gamma",
                    "ecut_ry": cfg.dft.ecut_ry,
                    "ecut_rho": cfg.dft.ecut_rho,
                    "smearing": cfg.dft.smearing,
                    "degauss_ry": cfg.dft.degauss_ry,
                    "pseudo_family": cfg.dft.pseudo_family,
                    "pseudo_md5s": {el: cfg.dft.pseudo_md5s.get(el) for el in elements},
                },
                "source_run_id": None if ctx is None else ctx.run_id,
            },
        )
    return energies


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    out: str | Path = DEFAULT_OUT,
    elements: Sequence[str] | None = None,
    units: str | Path | None = None,
    wait: bool = True,
) -> StageResult:
    els = list(elements) if elements else elements_of(cfg.data.compounds)
    root = Path(units) if units is not None else Path(cfg.paths.dft_dir) / "e0s"
    summary: dict[str, Any] = {"elements": ",".join(els), "units_root": str(root)}
    if ctx.dry_run:
        ids = plan(cfg, els, root)
        ctx.log(n_units=len(ids), dry_run=True, units_root=str(root))
        return stage_result(ctx, "partial", {**summary, "n_units": len(ids)})
    energies = isolated_atoms(cfg, els, root=root, ctx=ctx, out=None, wait=wait)
    _, details = collect(root, els)
    counts = details.pop("counts")
    write_counts(ctx, counts)
    ctx.log(
        n_units=len(els),
        n_failed=counts["failed"] + counts["unconverged"],
        counts=dict(counts),
        elements=details,
        pseudo_md5s={el: cfg.dft.pseudo_md5s[el] for el in els if el in cfg.dft.pseudo_md5s},
    )
    if len(energies) != len(els):
        missing = [el for el in els if str(atomic_numbers[el]) not in energies]
        status: Status = "partial" if counts["missing"] else "failed"
        return stage_result(ctx, status, {**summary, **counts, "missing": ",".join(missing)})
    local = write_e0s(energies, ctx.out_dir / "E0s_qe.json")
    ctx.add_output(local, "json")
    isolated_atoms(cfg, els, root=root, ctx=None, out=out, wait=False)  # final file + sidecar
    ctx.add_output(Path(out), "json")
    ctx.add_output(Path(out).with_suffix(".meta.json"), "json")
    return stage_result(
        ctx,
        "ok",
        {
            **summary,
            **counts,
            "out": str(out),
            **{f"E0_{el}": energies[str(atomic_numbers[el])] for el in els},
        },
    )


__all__ = [
    "DEFAULT_OUT",
    "STARTING_MAGNETIZATION",
    "collect",
    "isolated_atom_frame",
    "isolated_atoms",
    "overrides_for",
    "plan",
    "run",
    "unit_id",
    "write_e0s",
]
