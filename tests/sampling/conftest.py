"""Sampling-tier fixtures: analytic double well (Brownian dynamics), LJ vacancy hop, tiny MACE.

The double well ``U(x) = a (x^2 - 1)^2 + b x`` (eV, x dimensionless) is sampled with overdamped
Langevin (Euler-Maruyama, D = 1) in 12 harmonic windows, once per session; its well-to-well
free-energy difference and barrier are known by numerical integration of ``exp(-U/kT)`` over
each basin (CONTRACTS.md section 7: "WHAM on an analytic double well ... known dF +- 0.02 eV").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.spacegroup import crystal

from b20mlip.sampling import umbrella, vacancy

DW_A = 0.25  # eV
DW_B = 0.05  # eV (asymmetry)
DW_KT = 0.05  # eV
DW_K = 3.0  # eV per CV unit^2 (bias)
DW_WINDOWS = 12
DW_CENTERS = np.linspace(-1.5, 1.5, DW_WINDOWS)
DW_STEPS = 200_000
DW_DT = 1e-3
DW_EVERY = 4
DW_SEED = 0


def dw_energy(x: np.ndarray) -> np.ndarray:
    return DW_A * (x * x - 1.0) ** 2 + DW_B * x


def dw_force(x: np.ndarray) -> np.ndarray:
    return -(4.0 * DW_A * x * (x * x - 1.0) + DW_B)


@dataclass(frozen=True)
class DoubleWell:
    windows: list[dict[str, Any]]  # free_energy input: cv series, centre, k, T
    T: float
    kT: float
    k: float
    dF_true: float  # F(right basin) - F(left basin), basin-integrated
    barrier_true: float  # U(ts) - U(left minimum)
    ts: float
    minima: tuple[float, float]


def double_well_reference() -> tuple[float, float, float, tuple[float, float]]:
    """``(dF, barrier, x_ts, (x_left, x_right))`` by quadrature of ``exp(-U/kT)``."""
    x = np.linspace(-2.5, 2.5, 200_001)
    u = dw_energy(x)
    d = np.diff(u)
    minima = x[1:-1][(d[:-1] < 0) & (d[1:] > 0)]
    between = (x > minima[0]) & (x < minima[-1])
    ts = float(x[between][np.argmax(u[between])])
    w = np.exp(-u / DW_KT)
    dx = x[1] - x[0]
    f_left = -DW_KT * math.log(w[x < ts].sum() * dx)
    f_right = -DW_KT * math.log(w[x > ts].sum() * dx)
    barrier = float(dw_energy(np.array([ts]))[0] - dw_energy(np.array([minima[0]]))[0])
    return f_right - f_left, barrier, ts, (float(minima[0]), float(minima[-1]))


def brownian_windows(
    centers: np.ndarray = DW_CENTERS,
    k: float = DW_K,
    kT: float = DW_KT,
    steps: int = DW_STEPS,
    dt: float = DW_DT,
    every: int = DW_EVERY,
    seed: int = DW_SEED,
) -> list[dict[str, Any]]:
    """Overdamped Langevin in every window at once (vectorised Euler-Maruyama, D = 1)."""
    rng = np.random.default_rng(seed)
    x = np.array(centers, dtype=float)
    sq = math.sqrt(2.0 * dt)
    series: list[np.ndarray] = []
    for step in range(steps):
        force = dw_force(x) - k * (x - centers)
        x = x + force / kT * dt + sq * rng.standard_normal(len(x))
        if step % every == 0:
            series.append(x.copy())
    arr = np.array(series).T
    T = kT / units.kB
    return [
        {"cv": arr[i], "center": float(centers[i]), "k": float(k), "T": float(T)}
        for i in range(len(centers))
    ]


@pytest.fixture(scope="session")
def double_well() -> DoubleWell:
    dF, barrier, ts, minima = double_well_reference()
    return DoubleWell(
        windows=brownian_windows(),
        T=DW_KT / units.kB,
        kT=DW_KT,
        k=DW_K,
        dF_true=dF,
        barrier_true=barrier,
        ts=ts,
        minima=minima,
    )


def write_synthetic_run(root: Path, dw: DoubleWell, *, compound: str = "FeSi") -> Path:
    """An umbrella run dir (window files + index) built from the double-well series."""
    root.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for i, w in enumerate(dw.windows):
        path = umbrella.window_path(root, i)
        n = len(w["cv"])
        np.savez(
            path,
            cv=np.asarray(w["cv"]),
            center=w["center"],
            k=w["k"],
            T=w["T"],
            dt_fs=1.0,
            n_steps=n - 1,
            n_samples=n,
            n_equil=int(round(0.2 * n)),
            positions=np.zeros((1, 3)),
            cell=np.eye(3),
            numbers=np.array([14]),
            complete=True,
        )
        files.append(str(path))
    index = {
        "schema": umbrella.INDEX_SCHEMA,
        "method": umbrella.METHOD,
        "model": "synthetic.model",
        "k": dw.k,
        "k_eVA2": dw.k,
        "cv_scale_A": 1.0,
        "T": dw.T,
        "ps": 1.0,
        "timestep_fs": 1.0,
        "equil_fraction": 0.2,
        "seed": 0,
        "centers": [w["center"] for w in dw.windows],
        "files": files,
        "compound": compound,
        "head": "Default",
        "run_id": "synthetic",
        "model_info": {
            "path": "synthetic.model",
            "sha256": "0" * 64,
            "label": "B1",
            "e0_source": "E0s_qe.json",
            "energy_scale": "qe",
            "heads": ["Default"],
            "variant": "naive",
            "checkpoint": None,
        },
    }
    (root / umbrella.INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return root


@pytest.fixture
def synthetic_run(tmp_path: Path, double_well: DoubleWell) -> Path:
    return write_synthetic_run(tmp_path / "umbrella_run", double_well)


# --- Lennard-Jones vacancy hop (fcc 2x2x2, one vacancy) ------------------------------------------


def lj_calc() -> LennardJones:
    return LennardJones(epsilon=1.0, sigma=1.0, rc=3.0, smooth=True)


@dataclass(frozen=True)
class LJHop:
    initial: Atoms
    final: Atoms
    info: dict[str, Any]


@pytest.fixture(scope="session")
def lj_hop() -> LJHop:
    prim = bulk("Ar", "fcc", a=1.55, cubic=True)
    initial, final, info = vacancy.vacancy_hop_endpoints(prim, (2, 2, 2), "Ar")
    return LJHop(initial=initial, final=final, info=info)


# --- B20 cells and the tiny MACE calculator ------------------------------------------------------


@pytest.fixture
def fesi_atoms() -> Atoms:
    """FeSi B20 (P2_1 3, 8 atoms), the same parameters as the root conftest's fixture frames."""
    atoms = crystal(
        ["Fe", "Si"],
        basis=[(0.137, 0.137, 0.137), (0.842, 0.842, 0.842)],
        spacegroup=198,
        cellpar=[4.48, 4.48, 4.48, 90, 90, 90],
        primitive_cell=False,
    )
    atoms.info.clear()
    return atoms


@pytest.fixture(scope="session")
def tiny_calc(tiny_mace: Any) -> Any:
    return tiny_mace.calculator()
