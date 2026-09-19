"""``data sample``: round-0 candidate configurations derived from relaxed B20 parents.

For every parent structure (one relaxed 8-atom P2_1 3 cell per compound, e.g. the most
relaxed MPtrj frame or an OPTIMADE structure) the generator produces unlabelled frames
(``label_source="none"``, no energies/forces) of these ``config_type`` values:

* ``strain``      isotropic and uniaxial (x, y, z) strains ``cfg.data.strains``;
* ``shear``       simple shears ``cfg.data.shears`` on the xy, xz and yz planes;
* ``rattle``      Gaussian displacements with sigma in ``{1/3, 2/3, 1} x cfg.data.rattle_A``,
                  ``rattle_seeds`` draws each;
* ``eos``         isotropic volume scan +-``cfg.data.eos_pct`` % (``n_eos`` points);
* ``vacancy``     one TM and one Si/Ge vacancy in a 2x2x2 supercell (63 atoms);
* ``phonon_disp`` the phonopy displacement set (2x2x2, 0.03 Å) plus the undisplaced
                  supercell (``phonopy_disp_number=-1``) for residual-force subtraction;
* ``md``          NVT (Langevin) snapshots at ``cfg.data.temperatures_K`` with an injected
                  ASE calculator (the CLI loads MACE-MPA-0 only with ``--md``).

Lineage: ``group_id="<compound>/<config_type>/<parent frame_id>"`` and
``parent_id=<parent frame_id>``, so every derivative of one parent shares a group and the
group-hash split never separates them. Generator parameters are recorded in ``info``
(``strain``, ``strain_axis``, ``shear``, ``shear_plane``, ``rattle_sigma_A``, ``eos_volume_scale``,
``vacancy_species``, ``phonopy_disp_number``/``_atom``/``_vector``, ``md_step``...).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator

from b20mlip.config import Settings
from b20mlip.data._common import (
    DEFAULT_SYMPREC,
    TM_ELEMENTS,
    X_ELEMENTS,
    clean_info,
    is_b20_cell,
)
from b20mlip.io import frame_from_atoms, frame_to_atoms, read_frames, write_frames
from b20mlip.models import ConfigType, Frame
from b20mlip.provenance import RunContext

ALL_CONFIG_TYPES: tuple[str, ...] = (
    "strain",
    "shear",
    "rattle",
    "eos",
    "vacancy",
    "phonon_disp",
    "md",
)
SHEAR_PLANES: tuple[tuple[int, int, str], ...] = ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz"))
AXES = ("x", "y", "z")
Log = Callable[..., None] | None


@dataclass
class SampleOptions:
    """Knobs of :func:`generate` (defaults give ~110 8-atom frames per compound)."""

    config_types: tuple[str, ...] = ALL_CONFIG_TYPES
    rattle_levels: int = 3
    rattle_seeds: int = 10
    n_eos: int = 11
    supercell: tuple[int, int, int] = (2, 2, 2)
    phonon_distance: float = 0.03
    md_supercell: tuple[int, int, int] = (1, 1, 1)
    md_equil_steps: int = 500
    md_snapshots: int = 10
    md_stride: int = 100
    md_timestep_fs: float | None = None  # default cfg.md.timestep_fs
    md_friction: float | None = None  # default cfg.md.friction (1/fs)
    extra: dict[str, Any] = field(default_factory=dict)


# --- parents --------------------------------------------------------------------------------


def max_abs_force(frame: Frame) -> float:
    if frame.forces is None:
        return float("inf")
    arr = np.asarray(frame.forces, dtype=float)
    return float(np.abs(arr).max()) if arr.size else 0.0


def select_parents(
    frames: Iterable[Frame],
    compounds: Sequence[str],
    *,
    symprec: float = DEFAULT_SYMPREC,
    stats: dict[str, Any] | None = None,
) -> list[Frame]:
    """One relaxed B20 parent per compound: the spg-198 8-atom frame with the smallest max|F|.

    Ties (or frames without forces) fall back to the last frame in file order. Compounds
    without a B20 frame are reported in ``stats["missing"]`` and skipped.
    """
    by_compound: dict[str, list[Frame]] = {}
    for f in frames:
        if f.compound in compounds and is_b20_cell(frame_to_atoms(f), symprec):
            by_compound.setdefault(f.compound, []).append(f)
    parents: list[Frame] = []
    missing: list[str] = []
    for compound in compounds:
        cands = by_compound.get(compound)
        if not cands:
            missing.append(compound)
            continue
        best = min(cands, key=lambda f: (max_abs_force(f), -cands.index(f)))
        parents.append(best)
    if stats is not None:
        stats.update(
            parents={p.compound: p.frame_id for p in parents},
            parent_ids={p.compound: p.parent_id for p in parents},
            missing=missing,
        )
    return parents


def parent_atoms(frame: Frame) -> Atoms:
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    atoms.info = {}
    return atoms


# --- generators (each: parent -> frames) -------------------------------------------------------


def derived_frame(
    atoms: Atoms,
    parent: Frame,
    config_type: ConfigType,
    info: dict[str, Any],
    *,
    temperature_K: float | None = None,
) -> Frame:
    a = atoms.copy()
    a.calc = None
    a.info = {}
    frame = frame_from_atoms(
        a,
        group_id=f"{parent.compound}/{config_type}/{parent.frame_id}",
        compound=parent.compound,
        config_type=config_type,
        parent_id=parent.frame_id,
    )
    return frame.model_copy(update={"info": clean_info(info), "temperature_K": temperature_K})


def deform(atoms: Atoms, F: np.ndarray) -> Atoms:
    """Apply the deformation gradient ``F`` to the cell (lattice vectors are rows)."""
    out = atoms.copy()
    out.set_cell(np.asarray(atoms.cell[:]) @ np.asarray(F).T, scale_atoms=True)
    return out


def strain_frames(atoms: Atoms, parent: Frame, strains: Iterable[float]) -> list[Frame]:
    frames: list[Frame] = []
    for s in strains:
        iso = np.eye(3) * (1.0 + s)
        frames.append(
            derived_frame(deform(atoms, iso), parent, "strain", {"strain": s, "strain_axis": "iso"})
        )
        for axis, label in enumerate(AXES):
            F = np.eye(3)
            F[axis, axis] += s
            frames.append(
                derived_frame(
                    deform(atoms, F), parent, "strain", {"strain": s, "strain_axis": label}
                )
            )
    return frames


def shear_frames(atoms: Atoms, parent: Frame, shears: Iterable[float]) -> list[Frame]:
    frames: list[Frame] = []
    for g in shears:
        for i, j, plane in SHEAR_PLANES:
            F = np.eye(3)
            F[i, j] = g
            frames.append(
                derived_frame(deform(atoms, F), parent, "shear", {"shear": g, "shear_plane": plane})
            )
    return frames


def rattle_frames(
    atoms: Atoms,
    parent: Frame,
    rattle_A: float,
    rng: np.random.Generator,
    *,
    levels: int = 3,
    seeds: int = 10,
) -> list[Frame]:
    frames: list[Frame] = []
    sigmas = [rattle_A * (k + 1) / levels for k in range(levels)]
    for sigma in sigmas:
        for k in range(seeds):
            out = atoms.copy()
            out.positions = out.positions + rng.normal(0.0, sigma, out.positions.shape)
            frames.append(
                derived_frame(out, parent, "rattle", {"rattle_sigma_A": sigma, "rattle_draw": k})
            )
    return frames


def eos_frames(atoms: Atoms, parent: Frame, eos_pct: float, n: int = 11) -> list[Frame]:
    frames: list[Frame] = []
    for v in np.linspace(1.0 - eos_pct / 100.0, 1.0 + eos_pct / 100.0, n):
        F = np.eye(3) * float(v) ** (1.0 / 3.0)
        frames.append(
            derived_frame(deform(atoms, F), parent, "eos", {"eos_volume_scale": float(v)})
        )
    return frames


def vacancy_frames(
    atoms: Atoms, parent: Frame, supercell: tuple[int, int, int] = (2, 2, 2)
) -> list[Frame]:
    sc = atoms.repeat(supercell)
    symbols = sc.get_chemical_symbols()
    frames: list[Frame] = []
    for pool, label in ((TM_ELEMENTS, "TM"), (X_ELEMENTS, "X")):
        idx = next((i for i, s in enumerate(symbols) if s in pool), None)
        if idx is None:
            continue
        out = sc.copy()
        species = symbols[idx]
        del out[idx]
        frames.append(
            derived_frame(
                out,
                parent,
                "vacancy",
                {
                    "vacancy_species": species,
                    "vacancy_kind": label,
                    "vacancy_site_index": idx,
                    "supercell": "x".join(str(k) for k in supercell),
                },
            )
        )
    return frames


def phonon_disp_frames(
    atoms: Atoms,
    parent: Frame,
    supercell: tuple[int, int, int] = (2, 2, 2),
    distance: float = 0.03,
) -> list[Frame]:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell[:]),
        scaled_positions=atoms.get_scaled_positions(),
    )
    ph = Phonopy(unitcell, supercell_matrix=np.diag(supercell), primitive_matrix=None)
    ph.generate_displacements(distance=distance)
    sc_label = "x".join(str(k) for k in supercell)
    base = {"phonopy_distance": distance, "supercell": sc_label}

    def to_atoms(cell: Any) -> Atoms:
        return Atoms(
            symbols=cell.symbols,
            cell=np.asarray(cell.cell),
            scaled_positions=np.asarray(cell.scaled_positions),
            pbc=True,
        )

    frames = [
        derived_frame(
            to_atoms(ph.supercell), parent, "phonon_disp", {**base, "phonopy_disp_number": -1}
        )
    ]
    displacements = ph.dataset["first_atoms"]
    cells = list(ph.supercells_with_displacements)
    for i, (cell, disp) in enumerate(zip(cells, displacements, strict=True)):
        info = {
            **base,
            "phonopy_disp_number": i,
            "phonopy_disp_atom": int(disp["number"]),
            "phonopy_disp_vector": [float(x) for x in disp["displacement"]],
        }
        frames.append(derived_frame(to_atoms(cell), parent, "phonon_disp", info))
    return frames


def md_frames(
    atoms: Atoms,
    parent: Frame,
    calc: Calculator,
    temperatures_K: Iterable[float],
    rng: np.random.Generator,
    *,
    timestep_fs: float = 2.0,
    friction: float = 0.01,
    equil_steps: int = 500,
    snapshots: int = 10,
    stride: int = 100,
    supercell: tuple[int, int, int] = (1, 1, 1),
    log: Log = None,
) -> list[Frame]:
    """NVT Langevin snapshots; the calculator's labels are *not* kept (candidates only)."""
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import Stationary

    try:  # ASE >= 3.29
        from ase.md.velocitydistribution import thermalize_momenta
    except ImportError:  # pragma: no cover - older ASE
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        def thermalize_momenta(atoms: Atoms, temperature_K: float, *, rng: Any = None) -> None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)

    frames: list[Frame] = []
    sc_label = "x".join(str(k) for k in supercell)
    for T in temperatures_K:
        md_atoms = atoms.repeat(supercell) if supercell != (1, 1, 1) else atoms.copy()
        md_atoms.calc = calc
        thermalize_momenta(md_atoms, temperature_K=float(T), rng=rng)
        Stationary(md_atoms)
        md_atoms.set_constraint(FixCom())  # keep the centre of mass fixed (ASE >= 3.28 way)
        seed = int(rng.integers(0, 2**31 - 1))
        dyn = Langevin(
            md_atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=float(T),
            friction=friction / units.fs,
            fixcm=False,
            rng=np.random.default_rng(seed),
        )
        dyn.run(equil_steps)
        step = equil_steps
        for k in range(snapshots):
            dyn.run(stride)
            step += stride
            info = {
                "md_temperature_K": float(T),
                "md_step": step,
                "md_time_ps": step * timestep_fs / 1000.0,
                "md_snapshot": k,
                "md_engine": "ase.md.langevin.Langevin",
                "md_timestep_fs": timestep_fs,
                "md_friction_per_fs": friction,
                "md_calculator": type(calc).__name__,
                "md_seed": seed,
                "md_kinetic_temperature_K": float(md_atoms.get_temperature()),
                "supercell": sc_label,
            }
            frames.append(derived_frame(md_atoms, parent, "md", info, temperature_K=float(T)))
            if log is not None:
                log(md=f"{parent.compound} {T:g} K step {step}")
    return frames


# --- stage -----------------------------------------------------------------------------------


def plan(
    cfg: Settings, structures: Sequence[Frame], options: SampleOptions, md: bool
) -> dict[str, int]:
    """Expected frame counts per config type (what a dry run reports)."""
    d = cfg.data
    per_parent = {
        "strain": len(d.strains) * (1 + len(AXES)),
        "shear": len(d.shears) * len(SHEAR_PLANES),
        "rattle": options.rattle_levels * options.rattle_seeds,
        "eos": options.n_eos,
        "vacancy": 2,
        "phonon_disp": None,  # depends on symmetry (phonopy); 5 for an ideal B20 cell
        "md": len(d.temperatures_K) * options.md_snapshots if md else 0,
    }
    out: dict[str, int] = {}
    for ct in options.config_types:
        n = per_parent.get(ct)
        if n is None:
            n = 5
        out[ct] = n * len(structures)
    out["total_planned"] = sum(out.values())
    out["parents"] = len(structures)
    return out


def generate(
    cfg: Settings,
    ctx: RunContext | None,
    structures: Sequence[Frame],
    rng: np.random.Generator,
    *,
    calc: Calculator | None = None,
    md: bool = False,
    options: SampleOptions | None = None,
) -> list[Frame]:
    """Derived candidate frames for every parent in ``structures`` (see the module doc).

    ``md=True`` requires ``calc``. Under ``ctx.dry_run`` only the plan is logged and no frame
    is produced.
    """
    opts = options or SampleOptions()
    if md and calc is None:
        raise ValueError("md=True needs an ASE calculator (calc=...)")
    log = ctx.log if ctx is not None else None
    if ctx is not None and ctx.dry_run:
        ctx.log(plan=plan(cfg, structures, opts, md))
        return []
    d = cfg.data
    frames: list[Frame] = []
    counts: dict[str, int] = {}
    for parent in structures:
        atoms = parent_atoms(parent)
        produced: dict[str, list[Frame]] = {}
        if "strain" in opts.config_types:
            produced["strain"] = strain_frames(atoms, parent, d.strains)
        if "shear" in opts.config_types:
            produced["shear"] = shear_frames(atoms, parent, d.shears)
        if "rattle" in opts.config_types:
            produced["rattle"] = rattle_frames(
                atoms, parent, d.rattle_A, rng, levels=opts.rattle_levels, seeds=opts.rattle_seeds
            )
        if "eos" in opts.config_types:
            produced["eos"] = eos_frames(atoms, parent, d.eos_pct, opts.n_eos)
        if "vacancy" in opts.config_types:
            produced["vacancy"] = vacancy_frames(atoms, parent, opts.supercell)
        if "phonon_disp" in opts.config_types:
            produced["phonon_disp"] = phonon_disp_frames(
                atoms, parent, opts.supercell, opts.phonon_distance
            )
        if md and "md" in opts.config_types and calc is not None:
            produced["md"] = md_frames(
                atoms,
                parent,
                calc,
                d.temperatures_K,
                rng,
                timestep_fs=opts.md_timestep_fs or cfg.md.timestep_fs,
                friction=opts.md_friction or cfg.md.friction,
                equil_steps=opts.md_equil_steps,
                snapshots=opts.md_snapshots,
                stride=opts.md_stride,
                supercell=opts.md_supercell,
                log=log,
            )
        for ct, lst in produced.items():
            counts[ct] = counts.get(ct, 0) + len(lst)
            frames.extend(lst)
        if log is not None:
            log(**{f"sampled_{parent.compound}": {ct: len(v) for ct, v in produced.items()}})
    if ctx is not None:
        ctx.log(sample_counts=counts, parents=[p.frame_id for p in structures])
    return frames


def load_mace_calculator(cfg: Settings, model_path: str | Path | None = None) -> Calculator:
    """MACE-MPA-0 medium as an ASE calculator (CPU, float64); imported lazily."""
    path = (
        Path(model_path)
        if model_path is not None
        else Path(cfg.paths.models_dir) / "foundation" / "mace-mpa-0-medium.model"
    )
    if not path.is_file():
        raise FileNotFoundError(f"foundation model not found: {path}")
    from mace.calculators import MACECalculator  # slow import, only when the file exists

    return MACECalculator(
        model_paths=str(path), device=cfg.compute.device, default_dtype=cfg.compute.dtype
    )


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    out: str | Path,
    parents: str | Path | None = None,
    md: bool = False,
    compounds: Sequence[str] | None = None,
    options: SampleOptions | None = None,
    calc: Calculator | None = None,
) -> dict[str, Any]:
    """Stage ``data.sample``: parents file -> candidate frames at ``out``."""
    parents_path = (
        Path(parents)
        if parents is not None
        else Path(cfg.paths.data_dir) / "frames" / "mptrj_b20.extxyz"
    )
    if not parents_path.is_file():
        raise FileNotFoundError(
            f"parents file not found: {parents_path} (run `data pull --sources mptrj --extract`)"
        )
    ctx.add_input(parents_path, "frames")
    wanted = list(compounds or cfg.data.compounds)
    stats: dict[str, Any] = {}
    structures = select_parents(read_frames(parents_path), wanted, stats=stats)
    ctx.log(parent_selection=stats)
    if not structures:
        raise ValueError(f"no B20 parent found for {wanted} in {parents_path}")
    rng = np.random.default_rng(ctx.seed if ctx.seed is not None else 0)
    if md and calc is None and not ctx.dry_run:
        calc = load_mace_calculator(cfg)
    frames = generate(cfg, ctx, structures, rng, calc=calc, md=md, options=options)
    summary: dict[str, Any] = {"parents": len(structures), "frames": len(frames)}
    if ctx.dry_run:
        summary.update(ctx.extras.get("plan", {}))
        return summary
    out_path = Path(out)
    write_frames(frames, out_path)
    ctx.add_output(out_path, "frames")
    for ct, n in ctx.extras.get("sample_counts", {}).items():
        summary[f"n_{ct}"] = n
    return summary


__all__ = [
    "ALL_CONFIG_TYPES",
    "SampleOptions",
    "deform",
    "derived_frame",
    "eos_frames",
    "generate",
    "load_mace_calculator",
    "md_frames",
    "phonon_disp_frames",
    "plan",
    "rattle_frames",
    "run",
    "select_parents",
    "shear_frames",
    "strain_frames",
    "vacancy_frames",
]
