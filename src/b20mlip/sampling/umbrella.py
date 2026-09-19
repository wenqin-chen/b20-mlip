"""Umbrella sampling of the vacancy hop (CONTRACTS.md row 11, ``sampling umbrella``).

Pure-Python umbrella sampling (SPEC.md section 2: PLUMED is optional, this is primary):

* :class:`HarmonicBias` wraps any ASE calculator and adds ``1/2 k (cv - center)^2`` to the
  energy and ``-k (cv - center) dcv/dr`` to the forces, where ``cv(atoms) -> (value, gradient)``
  (``vacancy.HopCV`` for the hop coordinate: only the hopping atom moves it).
* :func:`run_windows` runs one Langevin window per bias centre (ASE ``Langevin``, friction
  0.01/fs, 2 fs) and stores the CV time series of every window in ``window_<i>.npz`` plus a
  ``windows.json`` index. Windows are resumable (complete files are skipped) and start from
  the previous window's last frame (``init="sequential"``) or from the linear interpolation
  between the end states at the window centre (``init="interp"``). All samples are stored; the
  first ``equil_fraction`` (20 %) is discarded by ``wham.free_energy``.

Units: the config's ``sampling.k_eVA2`` is a spring constant on the physical hop coordinate
(eV/Å²); the hop CV is dimensionless (0 = initial, 1 = final), so the bias constant in CV units
is ``k_cv = k_eVA2 * hop_distance^2`` (eV). ``windows.json`` records both. ``free_energy`` reads
``k`` in CV units, matching the stored series.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary

from b20mlip.config import Settings
from b20mlip.models import StageResult
from b20mlip.provenance import RunContext
from b20mlip.sampling import vacancy
from b20mlip.sampling._common import (
    head_label,
    json_safe,
    make_calculator,
    model_info,
    reference_cell,
    write_json,
)

WINDOW_FMT = "window_{:02d}.npz"
INDEX_NAME = "windows.json"
INDEX_SCHEMA = "b20mlip.umbrella.v1"
DEFAULT_TIMESTEP_FS = 2.0
DEFAULT_FRICTION_PER_FS = 0.01
DEFAULT_EQUIL_FRACTION = 0.2
DEFAULT_INIT = "sequential"
INITS = ("sequential", "interp")
METHOD = "umbrella-langevin"
CVFunction = Callable[[Atoms], tuple[float, np.ndarray]]
CalcFactory = Callable[[], Calculator]
LogFn = Callable[[str], None]


class HarmonicBias(Calculator):
    """``calc`` plus a harmonic bias ``1/2 k (cv(atoms) - center)^2`` on a collective variable.

    ``cv`` is a callable returning ``(value, gradient)`` with ``gradient`` of shape ``(N, 3)``
    (``dcv/dr``). Energies and forces of the wrapped calculator are kept in ``unbiased_energy``
    / ``unbiased_forces``; ``cv_value`` and ``bias_energy`` are those of the last evaluation.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calc: Calculator, cv: CVFunction, k: float, center: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.calc = calc
        self.cv = cv
        self.k = float(k)
        self.center = float(center)
        self.cv_value: float | None = None
        self.bias_energy: float | None = None
        self.unbiased_energy: float | None = None
        self.unbiased_forces: np.ndarray | None = None

    def bias(self, atoms: Atoms) -> tuple[float, float, np.ndarray]:
        """``(cv, bias energy, bias forces)`` at ``atoms`` without touching the wrapped calc."""
        value, grad = self.cv(atoms)
        dev = float(value) - self.center
        return float(value), 0.5 * self.k * dev * dev, -self.k * dev * np.asarray(grad)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        current = self.atoms
        assert current is not None
        energy = float(self.calc.get_potential_energy(current))
        forces = np.array(self.calc.get_forces(current), dtype=float)
        value, bias_energy, bias_forces = self.bias(current)
        self.cv_value = value
        self.bias_energy = bias_energy
        self.unbiased_energy = energy
        self.unbiased_forces = forces
        total = energy + bias_energy
        self.results = {"energy": total, "free_energy": total, "forces": forces + bias_forces}


def window_path(out_dir: str | Path, index: int) -> Path:
    return Path(out_dir) / WINDOW_FMT.format(index)


def load_window(path: str | Path) -> dict[str, Any]:
    """The arrays of one ``window_<i>.npz`` as a dict (scalars unwrapped)."""
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, Any] = {}
        for key in data.files:
            arr = data[key]
            out[key] = arr.item() if arr.ndim == 0 else arr
    return out


def window_complete(path: Path, center: float, n_steps: int) -> bool:
    if not path.is_file():
        return False
    try:
        data = load_window(path)
    except (OSError, ValueError):
        return False
    return (
        bool(data.get("complete", False))
        and int(data.get("n_steps", -1)) == int(n_steps)
        and abs(float(data.get("center", np.nan)) - center) < 1e-9
    )


def _steps_for(ps: float, timestep_fs: float) -> int:
    steps = int(round(float(ps) * 1000.0 / float(timestep_fs)))
    if steps < 1:
        raise ValueError(f"{ps} ps at {timestep_fs} fs is less than one step")
    return steps


class _Recorder:
    """MD observer: the CV, unbiased potential energy, temperature and time of every sample."""

    def __init__(self, atoms: Atoms, bias: HarmonicBias, dyn: Langevin, cv: CVFunction) -> None:
        self.atoms = atoms
        self.bias = bias
        self.dyn = dyn
        self.cv_fn = cv
        self.cv: list[float] = []
        self.epot: list[float] = []
        self.temperature: list[float] = []
        self.time_fs: list[float] = []

    def __call__(self) -> None:
        value = self.bias.cv_value
        if value is None:  # first call before any force evaluation
            value = float(self.cv_fn(self.atoms)[0])
        energy = self.bias.unbiased_energy
        self.cv.append(float(value))
        self.epot.append(float(energy) if energy is not None else float("nan"))
        self.temperature.append(float(self.atoms.get_temperature()))
        self.time_fs.append(float(self.dyn.get_time() / units.fs))


def _thermalise(atoms: Atoms, T: float, rng: np.random.Generator) -> None:
    """Maxwell-Boltzmann momenta (``thermalize_momenta`` on ASE >= 3.29, else the old name)."""
    try:
        from ase.md.velocitydistribution import thermalize_momenta
    except ImportError:  # pragma: no cover - ASE < 3.29
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    else:
        thermalize_momenta(atoms, T, rng=rng)


def _restart_from(data: dict[str, Any], template: Atoms) -> Atoms:
    atoms = template.copy()
    atoms.calc = None
    atoms.set_cell(np.asarray(data["cell"], dtype=float))
    atoms.set_positions(np.asarray(data["positions"], dtype=float))
    return atoms


def run_windows(
    model_path: str | Path | None,
    atoms: Atoms,
    cv: CVFunction,
    centers: Sequence[float],
    k: float,
    ps: float,
    T: float,
    *,
    timestep_fs: float = DEFAULT_TIMESTEP_FS,
    friction_per_fs: float = DEFAULT_FRICTION_PER_FS,
    seed: int = 0,
    out_dir: str | Path,
    calc_factory: CalcFactory | None = None,
    init: str = DEFAULT_INIT,
    endpoints: tuple[Atoms, Atoms] | None = None,
    sample_every: int = 1,
    equil_fraction: float = DEFAULT_EQUIL_FRACTION,
    log_every: int = 10,
    log: LogFn | None = None,
    cfg: Settings | None = None,
    head: str | None = None,
    index_extra: dict[str, Any] | None = None,
) -> list[str]:
    """Run (or resume) one Langevin window per centre; returns the window files in order.

    ``k`` is in eV per CV unit squared; ``T`` in K; ``ps`` picoseconds per window. Window ``i``
    uses ``numpy.random.default_rng(seed + i)`` for its velocities and thermostat noise.
    ``init="interp"`` needs ``endpoints=(initial, final)``; the window then starts from the
    minimum-image linear interpolation at fraction ``center`` (the hop CV runs from 0 to 1).
    """
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    if init == "interp" and endpoints is None:
        raise ValueError("init='interp' needs endpoints=(initial, final)")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers_arr = [float(c) for c in centers]
    n_steps = _steps_for(ps, timestep_fs)
    if calc_factory is None:
        if model_path is None:
            raise ValueError("run_windows needs a model_path or a calc_factory")

        def calc_factory() -> Calculator:
            return make_calculator(model_path, cfg, head)

    say: LogFn = log or (lambda _msg: None)
    index: dict[str, Any] = {
        "schema": INDEX_SCHEMA,
        "method": METHOD,
        "model": None if model_path is None else str(model_path),
        "k": float(k),
        "T": float(T),
        "ps": float(ps),
        "timestep_fs": float(timestep_fs),
        "friction_per_fs": float(friction_per_fs),
        "n_steps": n_steps,
        "sample_every": int(sample_every),
        "equil_fraction": float(equil_fraction),
        "seed": int(seed),
        "init": init,
        "n_atoms": len(atoms),
        "centers": centers_arr,
        "files": [],
        **(index_extra or {}),
    }
    base_calc: Calculator | None = None
    previous: Atoms | None = None
    files: list[str] = []
    for i, center in enumerate(centers_arr):
        path = window_path(out, i)
        if window_complete(path, center, n_steps):
            say(f"window {i} ({center:.3f}) complete, skipping")
            files.append(str(path))
            previous = _restart_from(load_window(path), atoms)
            index["files"] = list(files)
            write_json(out / INDEX_NAME, index)
            continue
        if init == "interp":
            assert endpoints is not None
            start = vacancy.interpolate_endpoints(endpoints[0], endpoints[1], center)
        elif previous is not None:
            start = previous.copy()
        else:
            start = atoms.copy()
        start.calc = None
        if base_calc is None:
            base_calc = calc_factory()
        bias = HarmonicBias(base_calc, cv, k, center)
        start.calc = bias
        rng = np.random.default_rng(int(seed) + i)
        _thermalise(start, T, rng)
        Stationary(start)
        dyn = Langevin(
            start,
            timestep_fs * units.fs,
            temperature_K=T,
            friction=friction_per_fs / units.fs,
            rng=rng,
            logfile=None,
        )
        recorder = _Recorder(start, bias, dyn, cv)
        t0 = time.perf_counter()
        start.get_forces()  # evaluate the initial state so the first record is consistent
        dyn.attach(recorder, interval=sample_every)
        dyn.run(n_steps)
        wall = time.perf_counter() - t0
        n_samples = len(recorder.cv)
        payload: dict[str, Any] = {
            "cv": np.asarray(recorder.cv, dtype=float),
            "epot_eV": np.asarray(recorder.epot, dtype=float),
            "temperature_K": np.asarray(recorder.temperature, dtype=float),
            "time_fs": np.asarray(recorder.time_fs, dtype=float),
            "center": float(center),
            "k": float(k),
            "T": float(T),
            "dt_fs": float(timestep_fs),
            "friction_per_fs": float(friction_per_fs),
            "n_steps": int(n_steps),
            "n_samples": int(n_samples),
            "n_equil": int(round(equil_fraction * n_samples)),
            "sample_every": int(sample_every),
            "seed": int(seed) + i,
            "window": int(i),
            "positions": np.asarray(start.get_positions(), dtype=float),
            "cell": np.asarray(start.cell[:], dtype=float),
            "numbers": np.asarray(start.numbers, dtype=int),
            "wall_seconds": float(wall),
            "complete": True,
        }
        tmp = path.with_suffix(".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(fh, **payload)
        tmp.replace(path)
        files.append(str(path))
        index["files"] = list(files)
        write_json(out / INDEX_NAME, index)
        series = np.asarray(recorder.cv)
        say(
            f"window {i} ({center:.3f}): {n_steps} steps in {wall:.1f} s, "
            f"<cv> = {series.mean():.3f} +- {series.std():.3f}, "
            f"<T> = {np.mean(recorder.temperature):.0f} K"
        )
        if log_every and (i + 1) % log_every == 0:
            say(f"{i + 1}/{len(centers_arr)} windows done")
        previous = start.copy()
        previous.calc = None
    return files


def read_index(run_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """``windows.json`` of an umbrella run dir (or the index file itself) and its path."""
    p = Path(run_dir)
    index_path = p if p.is_file() else p / INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"no {INDEX_NAME} in {p}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if data.get("schema") != INDEX_SCHEMA:
        raise ValueError(
            f"{index_path}: expected schema {INDEX_SCHEMA!r}, got {data.get('schema')!r}"
        )
    files = [str(index_path.parent / Path(f).name) for f in data.get("files", [])]
    data["files"] = files
    return data, index_path


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    model: str | Path,
    compound: str,
    windows: int | None = None,
    ps: float | None = None,
    k: float | None = None,
    T: float | None = None,
    species: str = "Si",
    supercell: Sequence[int] = vacancy.DEFAULT_SUPERCELL,
    structure: str | Path | None = None,
    init: str = DEFAULT_INIT,
    timestep_fs: float | None = None,
    label: str | None = None,
    head: str | None = None,
    cv_range: tuple[float, float] = (0.0, 1.0),
) -> StageResult:
    """Stage ``sampling.umbrella``: vacancy-hop system -> ``windows`` Langevin windows.

    ``k`` is ``sampling.k_eVA2`` (eV/Å² on the hop coordinate) and is converted to CV units
    with the hop distance; the stage writes no ``numbers.json`` (``sampling wham`` does).
    """
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    n_windows = cfg.sampling.windows if windows is None else int(windows)
    ps_value = cfg.sampling.ps_per_window if ps is None else float(ps)
    k_eVA2 = cfg.sampling.k_eVA2 if k is None else float(k)
    T_value = cfg.sampling.T if T is None else float(T)
    dt_fs = cfg.md.timestep_fs if timestep_fs is None else float(timestep_fs)
    friction = cfg.md.friction
    if n_windows < 2:
        raise ValueError("need at least two umbrella windows")
    seed = ctx.seed if ctx.seed is not None else 0
    info = model_info(model_path, label)
    head_name = head_label(head)

    cell, source = reference_cell(cfg, compound, structure)
    initial, final, hop = vacancy.vacancy_hop_endpoints(cell, tuple(supercell), species)
    cv = vacancy.HopCV(hop)
    k_cv = k_eVA2 * cv.scale_A**2
    centers = np.linspace(float(cv_range[0]), float(cv_range[1]), n_windows).tolist()
    n_steps = _steps_for(ps_value, dt_fs)
    plan: dict[str, float | int | str] = {
        "model_label": info.label,
        "compound": compound,
        "n_atoms": len(initial),
        "n_windows": n_windows,
        "ps_per_window": ps_value,
        "steps_per_window": n_steps,
        "T": T_value,
        "k_eVA2": k_eVA2,
        "k_cv_eV": float(k_cv),
        "hop_distance_A": float(hop["hop_distance"]),
        "species": str(hop["species"]),
        "structure_source": source,
        "init": init,
    }
    ctx.log(
        model_sha256=info.sha256,
        model_label=info.label,
        head=head_name,
        compound=compound,
        engine="ase",
        thermostat="langevin",
        timestep_fs=dt_fs,
        friction_per_fs=friction,
        T=T_value,
        n_windows=n_windows,
        ps_per_window=ps_value,
        k_eVA2=k_eVA2,
        k_cv_eV=float(k_cv),
        method=METHOD,
        hop=json_safe(hop),
        structure_source=source,
    )
    if ctx.dry_run:
        return StageResult(
            stage="sampling.umbrella",
            run_id=ctx.run_id,
            manifest_path=str(ctx.manifest_path),
            status="partial",
            outputs=[],
            summary={"planned": 1, **plan},
        )

    ctx.add_input(model_path, "model")
    from b20mlip.sampling.neb import write_endpoints

    for path in write_endpoints(ctx.out_dir, initial, final, compound):
        ctx.add_output(path, "frames")
    t0 = time.perf_counter()
    progress: list[str] = []

    def log(msg: str) -> None:
        progress.append(msg)
        ctx.log(progress=list(progress))

    files = run_windows(
        model_path,
        initial,
        cv,
        centers,
        k_cv,
        ps_value,
        T_value,
        timestep_fs=dt_fs,
        friction_per_fs=friction,
        seed=seed,
        out_dir=ctx.out_dir,
        init=init,
        endpoints=(initial, final),
        log=log,
        cfg=cfg,
        head=head,
        index_extra={
            "compound": compound,
            "species": hop["species"],
            "supercell": list(supercell),
            "hop": json_safe(hop),
            "k_eVA2": k_eVA2,
            "cv_scale_A": cv.scale_A,
            "model_info": info.as_dict(),
            "head": head_name,
            "structure_source": source,
            "run_id": ctx.run_id,
        },
    )
    for f in files:
        ctx.add_output(f, "other")
    ctx.add_output(ctx.out_dir / INDEX_NAME, "json")
    summary: dict[str, float | int | str] = {
        **plan,
        "n_files": len(files),
        "wall_seconds": time.perf_counter() - t0,
        "run_dir": str(ctx.out_dir),
    }
    return StageResult(
        stage="sampling.umbrella",
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status="ok" if len(files) == n_windows else "partial",
        outputs=list(ctx.outputs),
        summary=summary,
    )


__all__ = [
    "DEFAULT_EQUIL_FRACTION",
    "DEFAULT_FRICTION_PER_FS",
    "DEFAULT_INIT",
    "DEFAULT_TIMESTEP_FS",
    "INDEX_NAME",
    "INDEX_SCHEMA",
    "INITS",
    "METHOD",
    "WINDOW_FMT",
    "HarmonicBias",
    "load_window",
    "read_index",
    "run",
    "run_windows",
    "window_complete",
    "window_path",
]
