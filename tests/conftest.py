"""Shared fixtures (CONTRACTS.md section 7): offline session, tiny B20 frames, isolated settings.

``tiny_b20.extxyz`` is generated in-test: 12 synthetic 8-atom P2_1 3 (B20, space group 198)
frames — 4 compounds x 3 config types with random energies/forces/stress and group ids,
``label_source="none"`` — plus 3 frames tagged ``energy_scale="qe"`` with
``config_energy_weight=0`` to exercise the weight round trip (15 frames in total).
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup import crystal

from b20mlip.config import Settings
from b20mlip.io import frame_from_atoms, write_frames
from b20mlip.models import Frame

# compound -> (a [A], u_TM, u_X, TM, X, magnetic); approximate experimental B20 parameters
B20_PARAMS: dict[str, tuple[float, float, float, str, str, bool]] = {
    "FeSi": (4.48, 0.137, 0.842, "Fe", "Si", False),
    "CoSi": (4.44, 0.144, 0.843, "Co", "Si", False),
    "MnSi": (4.56, 0.138, 0.846, "Mn", "Si", True),
    "FeGe": (4.70, 0.135, 0.842, "Fe", "Ge", True),
}
CONFIG_TYPES = ("relax", "strain", "rattle")
QE_TAGGED = ("FeSi", "CoSi", "MnSi")


def _blocked(*args: object, **kwargs: object) -> None:
    raise RuntimeError("network access is disabled for the whole test session")


@pytest.fixture(scope="session", autouse=True)
def block_network() -> Iterator[None]:
    """No test may open a socket (CONTRACTS.md section 7)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(socket, "create_connection", _blocked)
    mp.setattr(socket.socket, "connect", _blocked)
    mp.setattr(socket.socket, "connect_ex", _blocked)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def clean_b20_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any B20_* variables from the developer's shell so config tests are deterministic."""
    for key in list(os.environ):
        if key.upper().startswith("B20_"):
            monkeypatch.delenv(key, raising=False)


def b20_atoms(compound: str, config_type: str, rng: np.random.Generator) -> Atoms:
    a, u, v, tm, x, _ = B20_PARAMS[compound]
    atoms = crystal(
        [tm, x],
        basis=[(u, u, u), (v, v, v)],
        spacegroup=198,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=False,
    )
    atoms.info.clear()  # drop the Spacegroup object / unit_cell tag that crystal() attaches
    if config_type == "strain":
        atoms.set_cell(atoms.cell[:] * 1.03, scale_atoms=True)
    elif config_type == "rattle":
        atoms.positions = atoms.positions + rng.normal(0.0, 0.05, atoms.positions.shape)
    return atoms


def _label(atoms: Atoms, rng: np.random.Generator, magnetic: bool) -> None:
    n = len(atoms)
    results = {
        "energy": float(-6.0 * n + rng.normal(0.0, 0.1)),
        "forces": rng.normal(0.0, 0.5, (n, 3)),
        "stress": rng.normal(0.0, 0.01, 6),  # Voigt 6, eV/A^3
    }
    if magnetic:
        results["magmoms"] = rng.normal(1.0, 0.1, n)
        atoms.info["total_magnetization"] = float(np.sum(results["magmoms"]))
    atoms.calc = SinglePointCalculator(atoms, **results)


def make_tiny_frames(seed: int = 0) -> list[Frame]:
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    for compound, (_, _, _, _, _, magnetic) in B20_PARAMS.items():
        parent = f"{compound}-p0"
        for config_type in CONFIG_TYPES:
            atoms = b20_atoms(compound, config_type, rng)
            if config_type == "rattle":
                atoms.info["temperature_K"] = 300.0
            atoms.info["lineage"] = parent
            _label(atoms, rng, magnetic)
            frames.append(
                frame_from_atoms(
                    atoms,
                    group_id=f"{compound}/{config_type}/{parent}",
                    compound=compound,
                    config_type=config_type,  # type: ignore[arg-type]
                    parent_id=parent,
                )
            )
    for compound in QE_TAGGED:
        atoms = b20_atoms(compound, "rattle", rng)  # distinct geometry -> distinct frame_id
        atoms.info["qe_unit"] = f"u-{compound}"
        _label(atoms, rng, B20_PARAMS[compound][5])
        frame = frame_from_atoms(
            atoms,
            group_id=f"{compound}/noise_floor/{compound}-p0",
            compound=compound,
            config_type="noise_floor",
            parent_id=f"{compound}-p0",
            label_source="qe",
            energy_scale="qe",
        )
        frames.append(frame.model_copy(update={"weights": {"energy": 0.0}}))
    return frames


@pytest.fixture(scope="session")
def tiny_frames() -> list[Frame]:
    return make_tiny_frames()


@pytest.fixture(scope="session")
def tiny_b20_path(tmp_path_factory: pytest.TempPathFactory, tiny_frames: list[Frame]) -> Path:
    path = tmp_path_factory.mktemp("fixtures") / "tiny_b20.extxyz"
    write_frames(tiny_frames, path)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with every writable path inside ``tmp_path`` (never the repo's runs/)."""
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "runs_dir": str(tmp_path / "runs"),
                "models_dir": str(tmp_path / "models"),
                "dft_dir": str(tmp_path / "dft"),
                "reports_dir": str(tmp_path / "reports"),
            }
        }
    )


@pytest.fixture
def repo() -> Path:
    return Path(__file__).resolve().parents[1]
