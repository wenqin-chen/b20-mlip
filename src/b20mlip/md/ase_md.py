"""ASE MD (CONTRACTS.md row 10): ``run(atoms, calc, ensemble, T, ps, timestep_fs, cfg, ctx)``,
``thermal_expansion``, ``vdos``, ``rdf`` and the ``md ase`` stage.

Ensembles (``cfg.md`` holds every constant; SPEC.md section 5/6)
----------------------------------------------------------------
* ``nve`` — velocity Verlet. ``drift_meV_atom_ps`` is the slope of a linear fit of the total
  energy per atom (meV) against time (ps) over the production window.
* ``nvt`` — Langevin with friction ``cfg.md.friction`` (1/fs), seeded.
* ``npt`` — ASE's Nosé–Hoover / Parrinello–Rahman integrator (``ase.md.melchionna.MelchionnaNPT``,
  Melchionna, Ciccotti and Holian 1993; ``ase.md.npt.NPT`` on older ASE), external pressure
  0 GPa, thermostat time ``cfg.md.taut`` fs and ``pfactor = (cfg.md.taup fs)^2 x
  cfg.md.bulk_modulus_GPa`` (ASE's recipe for a barostat time ``taup``), with
  ``set_fraction_traceless(0)`` so the box only scales isotropically: the B20 cell stays cubic
  and a(T) is one number. ASE needs a triangular cell; the cubic cell is diagonal already, any
  other cell is rotated to standard form first.

Every run starts from a seeded Maxwell–Boltzmann distribution at T (centre-of-mass momentum
removed), excludes ``cfg.md.equil_ps`` from every average (when the run is shorter than that, the
whole run is used and the manifest says so), writes ``md_<ens>_<T>K.traj`` (every
``cfg.md.dump_every_steps`` steps), ``thermo_<ens>_<T>K.csv`` (every ``thermo_every_steps``:
step, time_ps, T_K, E_pot_eV, E_kin_eV, E_tot_eV, a_A, volume_A3, pressure_GPa), the VDOS
``vdos_<ens>_<T>K.json`` (velocity autocorrelation via FFT, velocities sampled every
``vdos_every_steps`` steps, unit area) and the RDF ``rdf_<ens>_<T>K.json`` (ASE ``get_rdf`` over
the production trajectory frames, ``rmax`` clipped to half the smallest cell height). Progress
goes to ``ctx.log(md_progress=...)`` and the ``b20mlip.md`` logger about every picosecond.

``a`` is the cubic lattice parameter of the *input* cell (``(V / reps^3)^(1/3)``, see
:mod:`b20mlip.md.common`); ``thermal_expansion(results)`` fits ``a(T)`` linearly over >= 3 NPT
temperatures and returns ``alpha = (da/dT) / a(T_ref=300 K)``.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from numpy.typing import ArrayLike

from b20mlip.config import Settings
from b20mlip.md.common import (
    EXPERIMENTAL_A_A,
    block_ci95,
    compound_of,
    fmt_T,
    lattice_parameter,
    linear_fit,
    load_structure,
    make_supercell,
    md_numbers,
    model_provenance,
    relpath,
    stage_result,
    write_numbers,
    write_result,
)
from b20mlip.models import Head, MDResult, StageResult
from b20mlip.provenance import RunContext, sha256_file

log = logging.getLogger("b20mlip.md")

ENSEMBLES: tuple[str, ...] = ("nve", "nvt", "npt")
MEV_PER_THZ = 4.135667696  # h in meV ps: E[meV] = 4.1357 * f[THz]
THERMO_COLUMNS: tuple[str, ...] = (
    "step",
    "time_ps",
    "T_K",
    "E_pot_eV",
    "E_kin_eV",
    "E_tot_eV",
    "a_A",
    "volume_A3",
    "pressure_GPa",
)
MIN_VDOS_FRAMES = 4
PLAN_JSON = "plan.json"


def _npt_class() -> Any:
    try:
        from ase.md.melchionna import MelchionnaNPT

        return MelchionnaNPT
    except ImportError:  # pragma: no cover - ASE < 3.27
        from ase.md.npt import NPT

        return NPT


def check_ensemble(ensemble: str) -> str:
    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}, got {ensemble!r}")
    return ensemble


def n_steps(ps: float, timestep_fs: float) -> int:
    if timestep_fs <= 0 or ps <= 0:
        raise ValueError("ps and timestep_fs must be positive")
    return max(1, int(round(ps * 1000.0 / timestep_fs)))


def _triangular(atoms: Atoms) -> Atoms:
    """ASE's NPT wants a triangular cell: rotate to standard form when it is not."""
    cell = np.asarray(atoms.cell[:], dtype=float)
    upper = cell[1, 0] == cell[2, 0] == cell[2, 1] == 0.0
    lower = cell[0, 1] == cell[0, 2] == cell[1, 2] == 0.0
    if upper or lower:
        return atoms
    rcell, rotation = atoms.cell.standard_form()
    out = atoms.copy()
    out.set_cell(rcell, scale_atoms=False)
    out.set_positions(atoms.get_positions() @ rotation.T)
    if atoms.has("momenta"):
        out.set_momenta(atoms.get_momenta() @ rotation.T)
    return out


def make_dynamics(atoms: Atoms, ensemble: str, T: float, timestep_fs: float, cfg: Settings, rng):
    """The ASE integrator for ``ensemble`` (see the module docstring for the constants)."""
    dt = timestep_fs * units.fs
    if ensemble == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if ensemble == "nvt":
        return Langevin(
            atoms, timestep=dt, temperature_K=T, friction=cfg.md.friction / units.fs, rng=rng
        )
    bulk_modulus = cfg.md.bulk_modulus_GPa * units.GPa
    dyn = _npt_class()(
        atoms,
        timestep=dt,
        temperature_K=T,
        externalstress=0.0,
        ttime=cfg.md.taut * units.fs,
        pfactor=(cfg.md.taup * units.fs) ** 2 * bulk_modulus,
    )
    dyn.set_fraction_traceless(0.0)  # isotropic box scaling only
    return dyn


def thermostat_label(ensemble: str, cfg: Settings) -> str:
    return {
        "nve": "none (velocity Verlet)",
        "nvt": f"Langevin friction {cfg.md.friction} 1/fs",
        "npt": (
            f"Nose-Hoover/Parrinello-Rahman (Melchionna) taut {cfg.md.taut} fs, taup "
            f"{cfg.md.taup} fs, B {cfg.md.bulk_modulus_GPa} GPa, isotropic, 0 GPa"
        ),
    }[ensemble]


# --- analysis -------------------------------------------------------------------------------------


def _velocities(traj: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """``(velocities (n, N, 3) in ASE units, masses or None)`` from an array, images or a file."""
    if isinstance(traj, (str, Path)):
        traj = ase_read(str(traj), index=":")
    if isinstance(traj, np.ndarray):
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("velocities must have shape (nframes, natoms, 3)")
        return arr, None
    images = list(traj)
    if not images:
        raise ValueError("empty trajectory")
    arr = np.stack([np.asarray(img.get_velocities(), dtype=float) for img in images])
    return arr, np.asarray(images[0].get_masses(), dtype=float)


def vdos(
    traj: Any, dt_fs: float, *, masses: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Vibrational density of states from the velocity autocorrelation function.

    ``traj`` is an array ``(nframes, natoms, 3)``, a sequence of ``Atoms`` with velocities or a
    trajectory file; ``dt_fs`` is the sampling interval. By the Wiener–Khinchin theorem the
    Fourier transform of the (mass-weighted) VACF is the summed power spectrum of the velocity
    components, so the DOS is ``sum_i m_i sum_a |FFT(v_ia)|^2`` on ``rfftfreq`` frequencies,
    converted to meV (``E = h f``) and normalised to unit area. Returns ``(freq_meV, dos)``.
    """
    vel, traj_masses = _velocities(traj)
    n = vel.shape[0]
    if n < 2:
        raise ValueError("the VDOS needs at least two velocity frames")
    if masses is None:
        masses = traj_masses
    weights = np.ones(vel.shape[1]) if masses is None else np.asarray(masses, dtype=float)
    spectrum = np.abs(np.fft.rfft(vel - vel.mean(axis=0, keepdims=True), axis=0)) ** 2
    dos = np.einsum("kia,i->k", spectrum, weights)
    freq_thz = np.fft.rfftfreq(n, d=dt_fs) * 1000.0  # 1/fs -> THz
    freq_mev = freq_thz * MEV_PER_THZ
    area = float(np.trapezoid(dos, freq_mev)) if freq_mev.size > 1 else 0.0
    if area > 0:
        dos = dos / area
    return freq_mev, dos


def rdf(images: Sequence[Atoms], rmax: float, nbins: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Total RDF averaged over ``images`` (ASE ``get_rdf``); ``rmax`` is clipped to half the
    smallest cell height of every frame. Returns ``(r_A, g, rmax_used)``."""
    from ase.geometry.rdf import get_rdf, get_recommended_r_max

    if not images:
        raise ValueError("the RDF needs at least one frame")
    limit = min(get_recommended_r_max(img.cell, img.pbc) for img in images)
    rmax_used = min(float(rmax), 0.999 * float(limit))
    g, r = get_rdf(list(images), rmax_used, int(nbins))
    return np.asarray(r, dtype=float), np.asarray(g, dtype=float), rmax_used


def energy_drift(time_ps: ArrayLike, e_tot_per_atom_eV: ArrayLike) -> tuple[float, float | None]:
    """``(slope in meV/atom/ps, slope standard error or None)`` of E_tot/N against time."""
    slope, _, se = linear_fit(time_ps, np.asarray(e_tot_per_atom_eV, dtype=float) * 1000.0)
    return slope, se


def thermal_expansion_fit(results: Sequence[MDResult], T_ref: float = 300.0) -> dict[str, Any]:
    """Linear fit of ``a(T)`` over the NPT results (>= 3 distinct temperatures).

    ``alpha_per_K = slope / a_fit(T_ref)`` with ``a_fit(T) = intercept + slope T``. Returns the
    fit details (``temperatures_K``, ``a_A``, ``slope_A_per_K``, ``intercept_A``, ``a_ref_A``,
    ``T_ref_K``, ``n``, ``alpha_ci95`` from the slope error when >= 4 points).
    """
    points: dict[float, float] = {}
    for res in results:
        if res.ensemble == "npt" and res.a_mean_A is not None:
            points[float(res.temperature_K)] = float(res.a_mean_A)
    if len(points) < 3:
        raise ValueError(
            f"thermal expansion needs a(T) from NPT runs at >= 3 temperatures, got {sorted(points)}"
        )
    temps = sorted(points)
    a_vals = [points[t] for t in temps]
    slope, intercept, se = linear_fit(temps, a_vals)
    a_ref = intercept + slope * T_ref
    alpha = slope / a_ref
    ci = None
    if se is not None and len(temps) >= 4:
        ci = ((slope - 1.96 * se) / a_ref, (slope + 1.96 * se) / a_ref)
    return {
        "alpha_per_K": float(alpha),
        "slope_A_per_K": float(slope),
        "intercept_A": float(intercept),
        "a_ref_A": float(a_ref),
        "T_ref_K": float(T_ref),
        "temperatures_K": temps,
        "a_A": a_vals,
        "n": len(temps),
        "slope_se_A_per_K": se,
        "alpha_ci95": ci,
    }


def thermal_expansion(results: Sequence[MDResult], T_ref: float = 300.0) -> float:
    """Linear thermal-expansion coefficient (1/K) from ``a(T)`` (:func:`thermal_expansion_fit`)."""
    return thermal_expansion_fit(results, T_ref)["alpha_per_K"]


# --- the run --------------------------------------------------------------------------------------


def _production_mask(time_ps: np.ndarray, equil_ps: float) -> tuple[np.ndarray, bool]:
    mask = time_ps >= equil_ps - 1e-12
    if mask.sum() >= 2:
        return mask, True
    return np.ones_like(mask, dtype=bool), False


def run(
    atoms: Atoms,
    calc: Any,
    ensemble: str,
    T: float,
    ps: float,
    timestep_fs: float,
    cfg: Settings,
    ctx: RunContext,
    *,
    natoms: int | None = None,
    seed: int | None = None,
    head: Head = "Default",
    compound: str | None = None,
    model_sha256: str | None = None,
    stats: dict[str, Any] | None = None,
) -> MDResult:
    """Run one MD trajectory and return its :class:`MDResult` (files under ``ctx.out_dir``).

    ``atoms`` is the input cell (repeated to at least ``natoms``, default ``cfg.md.natoms``);
    ``calc`` any ASE calculator (a ``MACECalculator`` in production). ``stats``, when given, is
    filled with the production-window statistics the numbers file needs (``n_production``,
    ``T_mean_K``, ``a_ci95``, ``drift_ci95``, ``production_window``).
    """
    ensemble = check_ensemble(ensemble)
    steps = n_steps(ps, timestep_fs)
    seed = int(seed if seed is not None else (ctx.seed if ctx.seed is not None else 0))
    rng = np.random.default_rng(seed)
    cell, reps = make_supercell(atoms, natoms if natoms is not None else cfg.md.natoms)
    if ensemble == "npt":
        cell = _triangular(cell)
    cell.calc = calc
    MaxwellBoltzmannDistribution(cell, temperature_K=T, rng=rng, force_temp=True)
    Stationary(cell)
    compound = compound or compound_of(atoms)
    dyn = make_dynamics(cell, ensemble, T, timestep_fs, cfg, rng)

    tag = f"{ensemble}_{fmt_T(T)}K"
    traj_path = ctx.out_dir / f"md_{tag}.traj"
    thermo_path = ctx.out_dir / f"thermo_{tag}.csv"
    vdos_path = ctx.out_dir / f"vdos_{tag}.json"
    rdf_path = ctx.out_dir / f"rdf_{tag}.json"
    thermo_every = max(1, min(int(cfg.md.thermo_every_steps), steps))
    dump_every = max(1, min(int(cfg.md.dump_every_steps), steps))
    vdos_every = max(1, min(int(cfg.md.vdos_every_steps), steps))
    progress_every = max(1, int(round(1000.0 / timestep_fs)))
    n_atoms = len(cell)
    dt_ps = timestep_fs / 1000.0

    rows: list[dict[str, float]] = []
    velocities: list[np.ndarray] = []
    vel_steps: list[int] = []
    frames: list[Atoms] = []
    frame_steps: list[int] = []
    t0 = time.perf_counter()
    traj = Trajectory(str(traj_path), "w", cell)

    def record() -> None:
        step = dyn.nsteps
        results = getattr(cell.calc, "results", {}) or {}
        stress = results.get("stress")
        pressure = float("nan")
        if stress is not None:
            s = np.asarray(stress, dtype=float)
            pressure = float(-np.mean(s[:3]) / units.GPa)
        e_pot = float(cell.get_potential_energy())
        e_kin = float(cell.get_kinetic_energy())
        vol = float(cell.get_volume())
        rows.append(
            {
                "step": step,
                "time_ps": step * dt_ps,
                "T_K": float(cell.get_temperature()),
                "E_pot_eV": e_pot,
                "E_kin_eV": e_kin,
                "E_tot_eV": e_pot + e_kin,
                "a_A": lattice_parameter(vol, reps),
                "volume_A3": vol,
                "pressure_GPa": pressure,
            }
        )

    def sample_velocities() -> None:
        velocities.append(np.asarray(cell.get_velocities(), dtype=float).copy())
        vel_steps.append(dyn.nsteps)

    def dump() -> None:
        traj.write()
        frames.append(cell.copy())
        frame_steps.append(dyn.nsteps)

    def progress() -> None:
        step = dyn.nsteps
        wall = time.perf_counter() - t0
        payload = {
            "step": step,
            "time_ps": round(step * dt_ps, 4),
            "T_K": round(float(cell.get_temperature()), 2),
            "wall_s": round(wall, 2),
            "s_per_step": round(wall / step, 4) if step else None,
        }
        ctx.log(md_progress=payload)
        log.info("md %s %s: step %d / %d (%.2f ps) T=%.1f K wall %.1f s", ensemble, compound,
                 step, steps, step * dt_ps, payload["T_K"], wall)  # fmt: skip

    dyn.attach(record, interval=thermo_every)
    dyn.attach(sample_velocities, interval=vdos_every)
    dyn.attach(dump, interval=dump_every)
    dyn.attach(progress, interval=progress_every)
    try:
        dyn.run(steps)
    finally:
        traj.close()
    if rows[-1]["step"] != steps:  # the final step when steps is not a multiple of the interval
        record()
    wall = time.perf_counter() - t0

    with open(thermo_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(THERMO_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    time_ps = np.array([r["time_ps"] for r in rows])
    mask, equil_excluded = _production_mask(time_ps, cfg.md.equil_ps)
    prod = [r for r, m in zip(rows, mask, strict=True) if m]
    e_tot_atom = np.array([r["E_tot_eV"] for r in prod]) / n_atoms
    t_prod = np.array([r["time_ps"] for r in prod])
    a_series = np.array([r["a_A"] for r in prod])
    drift: float | None = None
    drift_se: float | None = None
    if ensemble == "nve" and len(prod) >= 2 and np.ptp(t_prod) > 0:
        drift, drift_se = energy_drift(t_prod, e_tot_atom)
    a_mean = float(a_series.mean()) if ensemble == "npt" else None
    a_std = (
        float(a_series.std(ddof=1))
        if ensemble == "npt" and len(prod) > 1
        else (0.0 if ensemble == "npt" else None)
    )

    # VDOS over the production window (>= MIN_VDOS_FRAMES sampled velocity frames)
    vdos_out: str | None = None
    vel_prod = [
        v
        for v, s in zip(velocities, vel_steps, strict=True)
        if mask_step(s, dt_ps, cfg.md.equil_ps, equil_excluded)
    ]
    if len(vel_prod) >= MIN_VDOS_FRAMES:
        freq, dos = vdos(np.stack(vel_prod), vdos_every * timestep_fs, masses=cell.get_masses())
        vdos_path.write_text(
            json.dumps(
                {
                    "freq_meV": freq.tolist(),
                    "dos": dos.tolist(),
                    "dt_fs": vdos_every * timestep_fs,
                    "n_frames": len(vel_prod),
                    "natoms": n_atoms,
                    "mass_weighted": True,
                    "window": "production",
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        vdos_out = str(vdos_path)
    # RDF over the production trajectory frames (the last frame when none is in the window)
    rdf_out: str | None = None
    frames_prod = [
        f
        for f, s in zip(frames, frame_steps, strict=True)
        if mask_step(s, dt_ps, cfg.md.equil_ps, equil_excluded)
    ]
    frames_prod = frames_prod or frames[-1:]
    if frames_prod:
        r, g, rmax_used = rdf(frames_prod, cfg.md.rdf_rmax_A, cfg.md.rdf_nbins)
        rdf_path.write_text(
            json.dumps(
                {"r_A": r.tolist(), "g": g.tolist(), "rmax_A": rmax_used, "nbins": cfg.md.rdf_nbins,
                 "n_frames": len(frames_prod), "window": "production"},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )  # fmt: skip
        rdf_out = str(rdf_path)

    for path in (traj_path, thermo_path):
        ctx.add_output(path)
    if vdos_out:
        ctx.add_output(vdos_path, "json")
    if rdf_out:
        ctx.add_output(rdf_path, "json")
    if stats is not None:
        stats.update(
            n_production=len(prod),
            n_thermo=len(rows),
            production_window="production (equil_ps excluded)" if equil_excluded else
            "all samples (run shorter than cfg.md.equil_ps)",
            T_mean_K=float(np.mean([r["T_K"] for r in prod])),
            E_pot_mean_eV_atom=float(np.mean([r["E_pot_eV"] for r in prod]) / n_atoms),
            a_ci95=block_ci95(a_series) if ensemble == "npt" else None,
            drift_ci95=(
                (drift - 1.96 * drift_se, drift + 1.96 * drift_se)
                if drift is not None and drift_se is not None
                else None
            ),
            wall_s=wall,
            s_per_step=wall / steps,
            reps=reps,
            seed=seed,
        )  # fmt: skip
    return MDResult(
        engine="ase",
        model_sha256=model_sha256 or "",
        head=head,
        compound=compound,
        natoms=n_atoms,
        ensemble=cast(Any, ensemble),
        temperature_K=float(T),
        pressure_GPa=0.0 if ensemble == "npt" else None,
        timestep_fs=float(timestep_fs),
        steps=steps,
        drift_meV_atom_ps=drift,
        a_mean_A=a_mean,
        a_std_A=a_std,
        alpha_per_K=None,
        traj_sha256=sha256_file(traj_path),
        vdos_path=vdos_out,
        rdf_path=rdf_out,
        run_id=ctx.run_id,
    )


def mask_step(step: int, dt_ps: float, equil_ps: float, equil_excluded: bool) -> bool:
    return (step * dt_ps >= equil_ps - 1e-12) if equil_excluded else True


# --- the stage ------------------------------------------------------------------------------------


def make_calculator(model: str | Path, cfg: Settings, head: str) -> Any:
    """A ``MACECalculator`` for ``model`` on ``cfg.compute``; the head must exist in the model."""
    from mace.calculators import MACECalculator

    try:
        import torch

        torch.set_num_threads(int(cfg.compute.threads))
    except Exception:  # pragma: no cover - torch always present with mace
        pass
    calc = MACECalculator(
        model_paths=str(model), device=cfg.compute.device, default_dtype=cfg.compute.dtype,
        head=head,
    )  # fmt: skip
    if getattr(calc, "head", head) != head:
        raise ValueError(f"model {model} has no head {head!r} (available: {calc.available_heads})")
    return calc


def stage(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    compound: str,
    ensemble: str,
    T: float | Sequence[float],
    ps: float,
    natoms: int | None = None,
    head: str = "Default",
    structure: str | Path | None = None,
    label: str | None = None,
    a_exp_A: float | None = None,
    calc: Any | None = None,
) -> StageResult:
    """Stage ``md.ase`` (``b20mlip md ase``): one trajectory per temperature in ``T``.

    Writes per temperature the files :func:`run` describes plus ``md_result_<ens>_<T>K.json``,
    and ``numbers.json`` (keys ``md.ase.<compound>.<label>.<ensemble>.<T>.{drift_meV_atom_ps,
    a_mean_A,a_std_A,alpha_per_K}`` plus the README aliases ``md.ase.<compound>.{drift_meV_atom_ps,
    a_300K_A,a_exp_A,a_dev_pct,alpha_per_K}``). With >= 3 NPT temperatures ``alpha_per_K`` is
    filled on every result (``thermal_expansion``). ``ctx.dry_run`` writes ``plan.json`` only.
    """
    ensemble = check_ensemble(ensemble)
    temps = sorted({float(t) for t in ([T] if isinstance(T, (int, float)) else T)})
    if not temps:
        raise ValueError("at least one temperature is required")
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    ctx.add_input(model_path, "model")
    if structure is not None:
        ctx.add_input(Path(structure), "frames")
    prov = model_provenance(model_path)
    label = label or prov["label"]
    seed = ctx.seed if ctx.seed is not None else 0
    atoms = load_structure(cfg, compound, structure)
    target = natoms if natoms is not None else cfg.md.natoms
    _, reps = make_supercell(atoms, target)
    steps = n_steps(ps, cfg.md.timestep_fs)
    plan = {
        "model": str(model_path),
        "model_sha256": prov["sha256"],
        "label": label,
        "head": head,
        "compound": compound,
        "ensemble": ensemble,
        "temperatures_K": temps,
        "ps": ps,
        "steps": steps,
        "timestep_fs": cfg.md.timestep_fs,
        "natoms": len(atoms) * reps**3,
        "reps": reps,
        "equil_ps": cfg.md.equil_ps,
        "thermostat": thermostat_label(ensemble, cfg),
        "seed": seed,
        "dry_run": bool(ctx.dry_run),
    }
    (ctx.out_dir / PLAN_JSON).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ctx.out_dir / PLAN_JSON, "json")
    ctx.log(
        engine="ase", timestep_fs=cfg.md.timestep_fs, thermostat=plan["thermostat"],
        ensemble=ensemble, temperatures_K=temps, model_sha256=prov["sha256"], head=head,
        model_label=label, e0_source=prov["e0_source"], energy_scale=prov["energy_scale"],
        natoms=plan["natoms"], reps=reps, steps=steps, equil_ps=cfg.md.equil_ps,
    )  # fmt: skip
    if ctx.dry_run:
        return stage_result(ctx, "partial", {"planned": len(temps), "steps": steps})

    calc = calc if calc is not None else make_calculator(model_path, cfg, head)
    results: list[MDResult] = []
    stats: dict[float, dict[str, Any]] = {}
    for temp in temps:
        st: dict[str, Any] = {}
        res = run(
            atoms, calc, ensemble, temp, ps, cfg.md.timestep_fs, cfg, ctx,
            natoms=target, seed=seed, head=cast(Head, head), compound=compound,
            model_sha256=prov["sha256"], stats=st,
        )  # fmt: skip
        results.append(res)
        stats[temp] = st
    fit: dict[str, Any] | None = None
    if ensemble == "npt" and len(results) >= 3:
        fit = thermal_expansion_fit(results)
        results = [r.model_copy(update={"alpha_per_K": fit["alpha_per_K"]}) for r in results]
    for res in results:
        write_result(ctx, res, f"md_result_{res.ensemble}_{fmt_T(res.temperature_K)}K.json")
    numbers = md_numbers(
        results, stats, provenance=prov, label=label, seed=seed, fit=fit, a_exp_A=a_exp_A
    )
    write_numbers(ctx, numbers)
    ctx.log(
        results={fmt_T(r.temperature_K): r.model_dump(mode="json") for r in results},
        stats={fmt_T(t): s for t, s in stats.items()},
        thermal_expansion=fit,
        numbers_keys=sorted(k for k in numbers if not k.endswith("@meta")),
    )
    summary: dict[str, Any] = {
        "engine": "ase",
        "compound": compound,
        "ensemble": ensemble,
        "n_runs": len(results),
        "steps": steps,
        "natoms": results[0].natoms,
        "model_label": label,
    }
    for res in results:
        key = fmt_T(res.temperature_K)
        if res.drift_meV_atom_ps is not None:
            summary[f"drift_meV_atom_ps_{key}K"] = res.drift_meV_atom_ps
        if res.a_mean_A is not None:
            summary[f"a_mean_A_{key}K"] = res.a_mean_A
        summary[f"T_mean_K_{key}K"] = stats[res.temperature_K]["T_mean_K"]
        summary[f"traj_{key}K"] = relpath(ctx.out_dir / f"md_{res.ensemble}_{key}K.traj", ctx)
    if fit is not None:
        summary["alpha_per_K"] = fit["alpha_per_K"]
    if ensemble == "npt" and compound in EXPERIMENTAL_A_A:
        summary["a_exp_A"] = a_exp_A if a_exp_A is not None else EXPERIMENTAL_A_A[compound]
    return stage_result(ctx, "ok", summary)


__all__ = [
    "ENSEMBLES",
    "MEV_PER_THZ",
    "MIN_VDOS_FRAMES",
    "THERMO_COLUMNS",
    "check_ensemble",
    "energy_drift",
    "make_calculator",
    "make_dynamics",
    "n_steps",
    "rdf",
    "run",
    "stage",
    "thermal_expansion",
    "thermal_expansion_fit",
    "thermostat_label",
    "vdos",
]
