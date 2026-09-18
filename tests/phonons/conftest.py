"""Phonon-tier fixtures: the 8-atom FeSi B20 cell, fcc Al with EMT, the tiny MACE calculator."""

from __future__ import annotations

from typing import Any

import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.spacegroup import crystal


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


@pytest.fixture
def al_atoms() -> Atoms:
    return bulk("Al", "fcc", a=4.05, cubic=True)


@pytest.fixture
def emt() -> EMT:
    return EMT()


@pytest.fixture(scope="session")
def tiny_calc(tiny_mace: Any) -> Any:
    return tiny_mace.calculator()
