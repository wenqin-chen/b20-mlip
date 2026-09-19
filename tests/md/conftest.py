"""md-tier fixtures: fast MD settings, B20 / argon cells, the fake ``lmp``, golden paths."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.spacegroup import crystal

from b20mlip.config import Settings
from b20mlip.io import frame_from_atoms, write_frames
from b20mlip.models import Frame

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
GOLDEN_LOG = GOLDEN_DIR / "lammps_thermo.log"
GOLDEN_PARITY = GOLDEN_DIR / "parity_forces.json"
FAKE_LMP = Path(__file__).resolve().parent / "fake_lmp.py"
B20: dict[str, tuple[float, float, float, str, str]] = {
    "FeSi": (4.48, 0.137, 0.842, "Fe", "Si"),
    "CoSi": (4.44, 0.144, 0.843, "Co", "Si"),
    "MnSi": (4.56, 0.138, 0.846, "Mn", "Si"),
    "FeGe": (4.70, 0.135, 0.842, "Fe", "Ge"),
}
FAST_MD: dict[str, object] = {
    "equil_ps": 0.0,
    "thermo_every_steps": 2,
    "dump_every_steps": 5,
    "vdos_every_steps": 1,
    "natoms": 8,
    "rdf_rmax_A": 6.0,
    "rdf_nbins": 50,
}


@pytest.fixture(scope="session")
def golden_log() -> Path:
    return GOLDEN_LOG


@pytest.fixture(scope="session")
def fake_lmp_cmd() -> str:
    return f"{sys.executable} {FAKE_LMP}"


@pytest.fixture
def md_settings(settings: Settings) -> Settings:
    """Isolated settings with a fast md section (no equilibration, dense logging)."""
    return settings.model_copy(update={"md": settings.md.model_copy(update=dict(FAST_MD))})


@pytest.fixture
def lammps_settings(md_settings: Settings, fake_lmp_cmd: str) -> Settings:
    return md_settings.model_copy(
        update={"cluster": md_settings.cluster.model_copy(update={"lammps_cmd": fake_lmp_cmd})}
    )


@pytest.fixture
def slurm_settings(lammps_settings: Settings, tmp_path: Path) -> Settings:
    return lammps_settings.model_copy(
        update={
            "cluster": lammps_settings.cluster.model_copy(
                update={
                    "account": "acct",
                    "partition_cpu": "cpu-part",
                    "partition_gpu": "gpu-h200",
                    "scratch": "/gscratch/b20",
                    "modules": ["gcc", "cuda/12.6"],
                    "control_path": str(tmp_path / "cm.sock"),
                    "lammps_cmd": "/gscratch/b20/b20-mlip/bin/lmp",
                }
            )
        }
    )


@pytest.fixture
def b20_atoms() -> Callable[..., Atoms]:
    """Factory: ``b20_atoms("MnSi", rattle=0.0, seed=0)`` -> 8-atom P2_1 3 cell."""

    def make(compound: str, *, rattle: float = 0.0, seed: int = 0) -> Atoms:
        a, u, v, tm, x = B20[compound]
        atoms = crystal(
            [tm, x],
            basis=[(u, u, u), (v, v, v)],
            spacegroup=198,
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=False,
        )
        atoms.info.clear()
        if rattle:
            rng = np.random.default_rng(seed)
            atoms.positions = atoms.positions + rng.normal(0.0, rattle, atoms.positions.shape)
        return atoms

    return make


@pytest.fixture
def b20_frames(b20_atoms: Callable[..., Atoms]) -> Callable[..., list[Frame]]:
    """Factory: ``b20_frames("MnSi", n=3)`` -> n rattled frames with distinct ids."""

    def make(compound: str, n: int = 3) -> list[Frame]:
        frames = []
        for i in range(n):
            atoms = b20_atoms(compound, rattle=0.03 if i else 0.0, seed=i)
            frames.append(
                frame_from_atoms(
                    atoms,
                    group_id=f"{compound}/rattle/{compound}-p0",
                    compound=compound,
                    config_type="rattle" if i else "relax",
                    parent_id=f"{compound}-p0",
                )
            )
        return frames

    return make


@pytest.fixture
def structure_file(tmp_path: Path, b20_frames: Callable[..., list[Frame]]) -> Path:
    """An extxyz with one relaxed MnSi frame (the ``--structure`` of the stages)."""
    path = tmp_path / "mnsi_structure.extxyz"
    write_frames(b20_frames("MnSi", n=1), path)
    return path


@pytest.fixture
def argon_cell() -> Atoms:
    """The 4-atom fcc argon cell whose 2x2x2 supercell produced the golden log."""
    return bulk("Ar", "fcc", a=5.26, cubic=True)


@pytest.fixture
def argon_file(tmp_path: Path, argon_cell: Atoms) -> Path:
    path = tmp_path / "argon.extxyz"
    frame = frame_from_atoms(
        argon_cell, group_id="Ar/relax/ar", compound="Ar", config_type="relax", parent_id="ar"
    )
    write_frames([frame], path)
    return path


@pytest.fixture
def lj_calc() -> LennardJones:
    return LennardJones(sigma=3.4, epsilon=0.0104, rc=8.5)
