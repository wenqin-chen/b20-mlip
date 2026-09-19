"""Climbing-image NEB vacancy-hop barrier (CONTRACTS.md row 11, ``sampling neb``).

``barrier`` relaxes both end states (FIRE, ``fmax`` 0.05 eV/Å, at most 500 steps), builds an
``ase.mep.NEB`` band of ``images`` images (spring 0.1 eV/Å², climbing image, IDPP interpolation
with the minimum-image convention) and optimises it with FIRE (``fmax`` 0.05 eV/Å, at most 300
steps). The forward barrier is ``max(E) - E[0]``, the reverse one ``max(E) - E[-1]`` and
``dE = E[-1] - E[0]``; all in eV on the model's own energy scale (a model-vs-model number).

``run`` is the stage function: the end states come from ``--initial/--final`` extxyz files or
from an automatic vacancy-hop setup (``vacancy.vacancy_hop_endpoints`` on the compound's
reference cell). It writes ``neb_path.traj`` (the converged band with energies and forces),
``neb.json`` and ``numbers.json`` (``sampling.neb.<compound>.<model_label>.E_a_eV`` plus the
README-table alias ``sampling.neb.<compound>.Ea_eV``). A band that did not converge within the
step budget is reported with ``status="partial"`` so that its number is never published.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.mep import NEB
from ase.optimize import FIRE

from b20mlip.config import Settings
from b20mlip.models import StageResult
from b20mlip.provenance import RunContext
from b20mlip.sampling import vacancy
from b20mlip.sampling._common import (
    head_label,
    json_safe,
    make_calculator,
    model_info,
    numbers_meta,
    reference_cell,
    write_json,
    write_numbers,
)

DEFAULT_IMAGES = 7
DEFAULT_FMAX = 0.05
DEFAULT_RELAX_STEPS = 500
DEFAULT_NEB_STEPS = 300
DEFAULT_SPRING = 0.1
NEB_METHOD = "improvedtangent"  # ASE's recommended tangent (the old default warns)
PATH_TRAJ = "neb_path.traj"
RESULT_JSON = "neb.json"
ENDPOINTS_EXTXYZ = ("initial.extxyz", "final.extxyz")
METHOD = "neb-ci-idpp-fire"
CalcFactory = Callable[[], Calculator]


def relax(
    atoms: Atoms, fmax: float = DEFAULT_FMAX, steps: int = DEFAULT_RELAX_STEPS
) -> dict[str, Any]:
    """FIRE relaxation in place (fixed cell); ``atoms.calc`` must be set."""
    opt = FIRE(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return {
        "converged": converged,
        "steps": int(opt.nsteps),
        "energy_eV": float(atoms.get_potential_energy()),
        "fmax_eVA": float(np.abs(atoms.get_forces()).max()),
    }


def _labelled_copy(atoms: Atoms) -> Atoms:
    """A calculator-free copy carrying energy/forces on a ``SinglePointCalculator``."""
    out = atoms.copy()
    out.calc = SinglePointCalculator(
        out, energy=float(atoms.get_potential_energy()), forces=np.asarray(atoms.get_forces())
    )
    return out


def barrier(
    model_path: str | Path | None,
    initial: Atoms,
    final: Atoms,
    images: int = DEFAULT_IMAGES,
    *,
    calc_factory: CalcFactory | None = None,
    fmax: float = DEFAULT_FMAX,
    relax_steps: int = DEFAULT_RELAX_STEPS,
    neb_steps: int = DEFAULT_NEB_STEPS,
    k: float = DEFAULT_SPRING,
    climb: bool = True,
    mic: bool = True,
    path_traj: str | Path | None = None,
    cfg: Settings | None = None,
    head: str | None = None,
) -> dict[str, Any]:
    """The NEB barrier between ``initial`` and ``final`` (see the module docstring).

    ``calc_factory`` returns a fresh calculator per image (tests pass ``LennardJones``); by
    default every image gets its own ``MACECalculator`` of ``model_path``.
    """
    if images < 3:
        raise ValueError(f"a band needs at least 3 images, got {images}")
    if len(initial) != len(final) or not np.array_equal(initial.numbers, final.numbers):
        raise ValueError("initial and final states must contain the same atoms in the same order")
    if calc_factory is None:
        if model_path is None:
            raise ValueError("barrier needs a model_path or a calc_factory")

        def calc_factory() -> Calculator:
            return make_calculator(model_path, cfg, head)

    t0 = time.perf_counter()
    first = initial.copy()
    last = final.copy()
    first.calc = calc_factory()
    last.calc = calc_factory()
    relax_info = {
        "initial": relax(first, fmax, relax_steps),
        "final": relax(last, fmax, relax_steps),
    }

    band = [first] + [first.copy() for _ in range(images - 2)] + [last]
    for image in band[1:-1]:
        image.calc = calc_factory()
    neb = NEB(band, k=k, climb=climb, method=NEB_METHOD)
    neb.interpolate(method="idpp", mic=mic)
    opt = FIRE(neb, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=neb_steps))
    energies = np.array([float(image.get_potential_energy()) for image in band])
    band_fmax = float(np.abs(neb.get_forces()).max())
    top = int(np.argmax(energies))

    labelled = [_labelled_copy(image) for image in band]
    traj_path: str | None = None
    if path_traj is not None:
        Path(path_traj).parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(path_traj), labelled)
        traj_path = str(path_traj)
    return {
        "E_a_eV": float(energies[top] - energies[0]),
        "E_a_reverse_eV": float(energies[top] - energies[-1]),
        "dE_eV": float(energies[-1] - energies[0]),
        "images": [float(e) for e in energies],
        "images_relative_eV": [float(e - energies[0]) for e in energies],
        "n_images": int(images),
        "top_image": top,
        "converged": converged
        and relax_info["initial"]["converged"]
        and relax_info["final"]["converged"],
        "neb_converged": converged,
        "steps": int(opt.nsteps),
        "fmax_eVA": band_fmax,
        "fmax_target": float(fmax),
        "spring_eVA2": float(k),
        "climb": bool(climb),
        "neb_method": NEB_METHOD,
        "relax": relax_info,
        "path_traj": traj_path,
        "band": labelled,
        "wall_seconds": time.perf_counter() - t0,
        "method": METHOD,
    }


def read_endpoints(initial: str | Path, final: str | Path) -> tuple[Atoms, Atoms]:
    """Read two single-configuration structure files (extxyz or anything ASE reads).

    Labels and metadata are dropped except ``info["compound"]`` (written by the stages), so
    a defect cell keeps the compound of its perfect parent instead of ``Fe4Si3``.
    """
    first = ase_read(str(initial), index=-1)
    last = ase_read(str(final), index=-1)
    if not isinstance(first, Atoms) or not isinstance(last, Atoms):
        raise ValueError("end-state files must each hold a single configuration")
    for atoms in (first, last):
        atoms.calc = None
        compound = atoms.info.get("compound")
        atoms.info = {"compound": str(compound)} if compound else {}
    return first, last


def write_endpoints(out_dir: Path, first: Atoms, last: Atoms, compound: str) -> list[Path]:
    """``initial.extxyz``/``final.extxyz`` with ``info["compound"]`` for later re-use."""
    paths: list[Path] = []
    for name, atoms in zip(ENDPOINTS_EXTXYZ, (first, last), strict=True):
        copy = atoms.copy()
        copy.calc = None
        copy.info = {"compound": compound}
        path = out_dir / name
        ase_write(str(path), copy, format="extxyz")
        paths.append(path)
    return paths


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    compound: str | None = None,
    initial: str | Path | None = None,
    final: str | Path | None = None,
    images: int = DEFAULT_IMAGES,
    species: str = "Si",
    supercell: Sequence[int] = vacancy.DEFAULT_SUPERCELL,
    structure: str | Path | None = None,
    label: str | None = None,
    head: str | None = None,
    fmax: float | None = None,
    relax_steps: int | None = None,
    neb_steps: int | None = None,
) -> StageResult:
    """Stage ``sampling.neb``: end states -> NEB barrier -> ``neb.json`` + ``numbers.json``."""
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if (initial is None) != (final is None):
        raise ValueError("--initial and --final must be given together")
    if initial is None and compound is None:
        raise ValueError("pass --compound (automatic vacancy hop) or --initial/--final")
    fmax = cfg.eval.fmax if fmax is None else float(fmax)
    relax_steps = cfg.eval.max_steps if relax_steps is None else int(relax_steps)
    neb_steps = DEFAULT_NEB_STEPS if neb_steps is None else int(neb_steps)
    seed = ctx.seed if ctx.seed is not None else 0
    info = model_info(model_path, label)
    head_name = head_label(head)

    if initial is not None and final is not None:
        first, last = read_endpoints(initial, final)
        compound = compound or first.info.get("compound") or vacancy.compound_of(first)
        first.info = {}
        last.info = {}
        hop = vacancy.hop_info_from_endpoints(first, last, base={"compound": compound})
        source = f"{initial}, {final}"
    else:
        assert compound is not None
        cell, source = reference_cell(cfg, compound, structure)
        first, last, hop = vacancy.vacancy_hop_endpoints(cell, tuple(supercell), species)
    plan: dict[str, Any] = {
        "model": str(model_path),
        "model_label": info.label,
        "compound": compound,
        "n_atoms": len(first),
        "n_images": int(images),
        "hop_distance_A": hop["hop_distance"],
        "species": hop["species"],
        "structure_source": source,
        "fmax": fmax,
        "relax_steps": relax_steps,
        "neb_steps": neb_steps,
    }
    ctx.log(
        model_sha256=info.sha256,
        model_label=info.label,
        head=head_name,
        compound=compound,
        n_images=int(images),
        method=METHOD,
        hop=json_safe(hop),
        structure_source=source,
    )
    if ctx.dry_run:
        return StageResult(
            stage="sampling.neb",
            run_id=ctx.run_id,
            manifest_path=str(ctx.manifest_path),
            status="partial",
            outputs=[],
            summary={"planned": 1, **{k: v for k, v in plan.items() if not isinstance(v, list)}},
        )

    ctx.add_input(model_path, "model")
    if initial is not None and final is not None:
        ctx.add_input(initial)
        ctx.add_input(final)
    for path in write_endpoints(ctx.out_dir, first, last, str(compound)):
        ctx.add_output(path, "frames")

    result = barrier(
        model_path,
        first,
        last,
        images,
        fmax=fmax,
        relax_steps=relax_steps,
        neb_steps=neb_steps,
        path_traj=ctx.out_dir / PATH_TRAJ,
        cfg=cfg,
        head=head,
    )
    ctx.add_output(ctx.out_dir / PATH_TRAJ, "traj")
    band = result.pop("band")
    cv = vacancy.HopCV(hop)
    result["cv_along_path"] = [cv.value(image) for image in band]
    document = {**plan, **result, "hop": hop, "model_info": info.as_dict(), "head": head_name}
    write_json(ctx.out_dir / RESULT_JSON, document)
    ctx.add_output(ctx.out_dir / RESULT_JSON, "json")

    prefix = f"sampling.neb.{compound}.{info.label}"
    common: dict[str, Any] = dict(
        head=head_name,
        n=int(images),
        seed=seed,
        T=0.0,
        method=METHOD,
        ci95=None,
        ci95_reason="deterministic NEB barrier (no sampling)",
        unit="eV",
        compound=compound,
        n_atoms=len(first),
        converged=bool(result["converged"]),
    )
    numbers: dict[str, Any] = {}
    for key, value in (
        (f"{prefix}.E_a_eV", result["E_a_eV"]),
        (f"{prefix}.E_a_reverse_eV", result["E_a_reverse_eV"]),
        (f"{prefix}.dE_eV", result["dE_eV"]),
        (f"sampling.neb.{compound}.Ea_eV", result["E_a_eV"]),  # README table alias
    ):
        numbers[key] = value
        numbers[f"{key}@meta"] = numbers_meta(info, **common)
    write_numbers(ctx, numbers)
    ctx.log(converged=bool(result["converged"]), neb_steps=result["steps"])

    summary: dict[str, float | int | str] = {
        "E_a_eV": result["E_a_eV"],
        "E_a_reverse_eV": result["E_a_reverse_eV"],
        "dE_eV": result["dE_eV"],
        "converged": int(result["converged"]),
        "steps": result["steps"],
        "n_images": int(images),
        "compound": str(compound),
        "model_label": info.label,
    }
    if not result["converged"]:
        summary["note"] = "band or end states not converged: raise --neb-steps/--relax-steps"
    return StageResult(
        stage="sampling.neb",
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status="ok" if result["converged"] else "partial",
        outputs=list(ctx.outputs),
        summary=summary,
    )


__all__ = [
    "DEFAULT_FMAX",
    "DEFAULT_IMAGES",
    "DEFAULT_NEB_STEPS",
    "DEFAULT_RELAX_STEPS",
    "DEFAULT_SPRING",
    "METHOD",
    "NEB_METHOD",
    "PATH_TRAJ",
    "RESULT_JSON",
    "barrier",
    "read_endpoints",
    "relax",
    "run",
    "write_endpoints",
]
