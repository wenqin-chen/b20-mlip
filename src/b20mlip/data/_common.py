"""Helpers shared by the data modules: element sets, compound names, spglib symmetry.

Naming convention (binding for every frame the data tier writes): ``Frame.compound`` is ASE's
empirical formula in ``"metal"`` mode (metals first, e.g. ``FeSi``, ``MnGe``, ``Co2FeSi``),
which is also the fallback :func:`b20mlip.io.read_frames` uses, so a compound name survives an
extxyz round trip without metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

FAMILY_ELEMENTS: frozenset[str] = frozenset({"Mn", "Fe", "Co", "Si", "Ge"})
TM_ELEMENTS: frozenset[str] = frozenset({"Mn", "Fe", "Co"})
X_ELEMENTS: frozenset[str] = frozenset({"Si", "Ge"})
B20_SPACEGROUP = 198  # P2_1 3
B20_NATOMS = 8
DEFAULT_SYMPREC = 0.1


def normalize_elements(elements: Iterable[str | int]) -> frozenset[str]:
    """Accept symbols or atomic numbers (``"Fe"``, ``26``) and return a set of symbols."""
    out: set[str] = set()
    for el in elements:
        if isinstance(el, str):
            sym = el.strip()
            if sym not in atomic_numbers:
                raise ValueError(f"unknown element symbol {el!r}")
            out.add(sym)
        else:
            out.add(chemical_symbols[int(el)])
    return frozenset(out)


def element_set(numbers: Iterable[int]) -> frozenset[str]:
    return frozenset(chemical_symbols[int(z)] for z in numbers)


def element_key(elements: Iterable[str]) -> str:
    """Canonical ``"Co-Fe-Si"`` key for per-element-set counting (alphabetical)."""
    return "-".join(sorted(set(elements)))


def compound_name(atoms_or_numbers: Atoms | Iterable[int]) -> str:
    """Empirical formula, metals first (``Fe4Si4`` -> ``FeSi``, ``Fe1Co2Si1`` -> ``Co2FeSi``)."""
    atoms = (
        atoms_or_numbers
        if isinstance(atoms_or_numbers, Atoms)
        else Atoms(numbers=[int(z) for z in atoms_or_numbers])
    )
    return atoms.get_chemical_formula("metal", empirical=True)


def spglib_cell(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(atoms.cell[:], dtype=float),
        np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float),
        np.asarray(atoms.numbers, dtype=int),
    )


def spacegroup_number(atoms: Atoms, symprec: float = DEFAULT_SYMPREC) -> int:
    """International space-group number from spglib (``0`` when symmetry search fails)."""
    import spglib  # local import: keeps module import cheap for the CLI

    dataset: Any = spglib.get_symmetry_dataset(spglib_cell(atoms), symprec=symprec)
    if dataset is None:
        return 0
    number = getattr(dataset, "number", None)
    if number is None and isinstance(dataset, dict):  # spglib < 2.5 returned a dict
        number = dataset.get("number")
    return int(number or 0)


def is_b20_cell(atoms: Atoms, symprec: float = DEFAULT_SYMPREC) -> bool:
    """8-atom TM-X (TM in {Mn,Fe,Co}, X in {Si,Ge}) cell with space group 198 (P2_1 3)."""
    if len(atoms) != B20_NATOMS:
        return False
    els = element_set(atoms.numbers)
    if len(els) != 2 or not (els & TM_ELEMENTS) or not (els & X_ELEMENTS):
        return False
    return spacegroup_number(atoms, symprec) == B20_SPACEGROUP


def n_transition_metals(numbers: Iterable[int]) -> int:
    return sum(1 for z in numbers if chemical_symbols[int(z)] in TM_ELEMENTS)


def clean_info(info: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None``/empty-string values (extxyz cannot represent them unambiguously)."""
    out: dict[str, Any] = {}
    for key, value in info.items():
        if value is None or (isinstance(value, str) and value == ""):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        out[str(key)] = value
    return out


__all__ = [
    "B20_NATOMS",
    "B20_SPACEGROUP",
    "DEFAULT_SYMPREC",
    "FAMILY_ELEMENTS",
    "TM_ELEMENTS",
    "X_ELEMENTS",
    "clean_info",
    "compound_name",
    "element_key",
    "element_set",
    "is_b20_cell",
    "n_transition_metals",
    "normalize_elements",
    "spacegroup_number",
    "spglib_cell",
]
