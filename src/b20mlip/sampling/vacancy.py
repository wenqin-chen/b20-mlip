"""Vacancy-hop end states and the hop collective variable (sampling tier helper).

``vacancy_hop_endpoints`` builds an ``n1 x n2 x n3`` supercell of a B20 cell (8 -> 63 atoms for
2 x 2 x 2), removes one atom of the chosen sublattice (``"Si"``/``"Ge"``/an element symbol, ``"X"``
for the p-block site, ``"TM"`` for the transition-metal site) and moves that vacancy's nearest
same-species neighbour into the empty site for the final state. Everything is deterministic:
without ``rng`` the vacancy is the sublattice atom closest to the cell centre, with ``rng`` it is
drawn from the sublattice; the hopping atom is the closest same-species neighbour (minimum image,
lowest index on ties).

The hop CV is the projection of the hopping atom's displacement from its initial site onto the
hop vector (initial site -> vacancy site) divided by the squared hop length: 0 in the initial
state, 1 in the final state, dimensionless. Only the hopping atom moves it, so its gradient is
``hop_vector / |hop_vector|^2`` on that atom and zero elsewhere; minimum-image displacements make
it safe when the atom crosses a cell boundary. ``HopCV.scale_A`` (the hop length) converts a bias
constant in eV/A^2 on the physical coordinate into eV per CV unit^2 (``umbrella.run_windows``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.spacegroup import crystal

# compound -> (a [A], u_TM, u_X, TM, X): approximate experimental B20 cells (P2_1 3, 8 atoms).
# Used only when no reference cell (``--structure`` or the MPtrj extract) is available.
B20_PARAMS: dict[str, tuple[float, float, float, str, str]] = {
    "FeSi": (4.48, 0.137, 0.842, "Fe", "Si"),
    "CoSi": (4.44, 0.144, 0.843, "Co", "Si"),
    "MnSi": (4.56, 0.138, 0.846, "Mn", "Si"),
    "FeGe": (4.70, 0.135, 0.842, "Fe", "Ge"),
}
P_BLOCK: frozenset[str] = frozenset({"Si", "Ge"})
SPECIES_ALIASES: tuple[str, ...] = ("TM", "X")
DEFAULT_SUPERCELL: tuple[int, int, int] = (2, 2, 2)
CV_NAME = "hop"
_TIE_DECIMALS = 6


def b20_cell(compound: str) -> Atoms:
    """The tabulated 8-atom B20 cell of ``compound`` (fallback when no reference cell exists)."""
    if compound not in B20_PARAMS:
        raise ValueError(
            f"no tabulated B20 cell for {compound!r} (known: {sorted(B20_PARAMS)}); "
            "pass --structure"
        )
    a, u, v, tm, x = B20_PARAMS[compound]
    atoms = crystal(
        [tm, x],
        basis=[(u, u, u), (v, v, v)],
        spacegroup=198,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=False,
    )
    atoms.info.clear()
    return atoms


def compound_of(atoms: Atoms) -> str:
    """Empirical formula in metal-first order (``"FeSi"``)."""
    return str(atoms.get_chemical_formula("metal", empirical=True))


def resolve_species(atoms: Atoms, species: str) -> str:
    """Map ``"TM"``/``"X"`` (or an element symbol present in ``atoms``) to an element symbol."""
    symbols = sorted(set(atoms.get_chemical_symbols()))
    if species == "X":
        candidates = [s for s in symbols if s in P_BLOCK]
    elif species == "TM":
        candidates = [s for s in symbols if s not in P_BLOCK]
    else:
        candidates = [s for s in symbols if s == species]
    if len(candidates) != 1:
        raise ValueError(
            f"species {species!r} does not select exactly one element of {symbols} "
            f"(use an element symbol, 'TM' or 'X'); candidates: {candidates}"
        )
    return candidates[0]


def supercell_tuple(supercell: int | Sequence[int]) -> tuple[int, int, int]:
    reps = (supercell,) * 3 if isinstance(supercell, int) else tuple(int(n) for n in supercell)
    if len(reps) != 3 or any(n < 1 for n in reps):
        raise ValueError(f"supercell must be three positive integers, got {supercell!r}")
    return reps[0], reps[1], reps[2]


def mic_vectors(vectors: np.ndarray, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-image ``vectors`` (shape ``(n, 3)``) and their lengths for ``atoms``' cell/pbc."""
    arr = np.asarray(vectors, dtype=float).reshape(-1, 3)
    vecs, lengths = find_mic(arr, atoms.cell, atoms.pbc)
    return np.asarray(vecs), np.asarray(lengths)


def vacancy_hop_endpoints(
    atoms_prim: Atoms,
    supercell: int | Sequence[int] = DEFAULT_SUPERCELL,
    species: str = "Si",
    rng: np.random.Generator | None = None,
) -> tuple[Atoms, Atoms, dict[str, Any]]:
    """Initial state (one vacancy), final state (neighbour hopped into it) and the hop info."""
    reps = supercell_tuple(supercell)
    sc = atoms_prim.repeat(reps)
    sc.info = {}
    sc.calc = None
    symbol = resolve_species(sc, species)
    symbols = np.asarray(sc.get_chemical_symbols())
    members = np.flatnonzero(symbols == symbol)
    if len(members) < 2:
        raise ValueError(f"need at least two {symbol} atoms for a vacancy hop, got {len(members)}")
    if rng is None:
        centre = 0.5 * np.asarray(sc.cell[:]).sum(axis=0)
        to_centre = np.linalg.norm(sc.positions[members] - centre, axis=1)
        order = np.lexsort((members, np.round(to_centre, _TIE_DECIMALS)))
        vacancy = int(members[order[0]])
    else:
        vacancy = int(rng.choice(members))
    others = members[members != vacancy]
    vecs, lengths = mic_vectors(sc.positions[others] - sc.positions[vacancy], sc)
    order = np.lexsort((others, np.round(lengths, _TIE_DECIMALS)))
    hop = int(others[order[0]])
    hop_vector = -vecs[order[0]]  # from the hopping atom to the vacancy site
    distance = float(lengths[order[0]])
    shortest = float(np.min(np.linalg.norm(sc.cell[:], axis=1)))
    if distance >= shortest:
        raise ValueError(
            f"hop of {distance:.2f} A is not shorter than the cell ({shortest:.2f} A): "
            "the CV would be ambiguous; use a larger supercell"
        )

    initial = sc.copy()
    del initial[vacancy]
    hop_index = hop - (1 if hop > vacancy else 0)
    final = initial.copy()
    final.positions[hop_index] = final.positions[hop_index] + hop_vector
    site = initial.positions[hop_index]
    info: dict[str, Any] = {
        "cv": CV_NAME,
        "compound": compound_of(atoms_prim),
        "species": symbol,
        "supercell": list(reps),
        "n_atoms": len(initial),
        "vacancy_index_supercell": vacancy,
        "hop_index_supercell": hop,
        "hop_index": int(hop_index),
        "initial_site": [float(x) for x in site],
        "vacancy_site": [float(x) for x in site + hop_vector],
        "hop_vector": [float(x) for x in hop_vector],
        "hop_distance": distance,
    }
    return initial, final, info


def hop_info_from_endpoints(
    initial: Atoms, final: Atoms, *, base: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hop info for two arbitrary end states: the hopping atom is the one that moves the most."""
    if len(initial) != len(final) or not np.array_equal(initial.numbers, final.numbers):
        raise ValueError("initial and final states must contain the same atoms in the same order")
    disp, lengths = mic_vectors(final.positions - initial.positions, initial)
    hop = int(np.argmax(lengths))
    distance = float(lengths[hop])
    if distance < 1e-6:
        raise ValueError("initial and final states coincide: no hop to sample")
    others = np.delete(lengths, hop)
    site = initial.positions[hop]
    info: dict[str, Any] = dict(base or {})
    info.update(
        {
            "cv": CV_NAME,
            "species": initial.get_chemical_symbols()[hop],
            "n_atoms": len(initial),
            "hop_index": hop,
            "initial_site": [float(x) for x in site],
            "vacancy_site": [float(x) for x in site + disp[hop]],
            "hop_vector": [float(x) for x in disp[hop]],
            "hop_distance": distance,
            "max_other_displacement_A": float(others.max()) if len(others) else 0.0,
        }
    )
    info.setdefault("compound", compound_of(initial))
    return info


def interpolate_endpoints(initial: Atoms, final: Atoms, fraction: float) -> Atoms:
    """Linear (minimum-image) interpolation ``initial + fraction * (final - initial)``."""
    disp, _ = mic_vectors(final.positions - initial.positions, initial)
    out = initial.copy()
    out.calc = None
    out.positions = initial.positions + float(fraction) * disp
    return out


class HopCV:
    """Callable ``cv(atoms) -> (value, gradient)`` for the hop CV described by ``info``.

    The displacement of the hopping atom is taken as the minimum image relative to the hop
    midpoint, so the CV is unambiguous as long as the hop is shorter than the cell (the atom
    is never more than half a hop from the midpoint at either end state).
    """

    def __init__(self, info: dict[str, Any]) -> None:
        self.info = dict(info)
        self.index = int(info["hop_index"])
        self.site = np.asarray(info["initial_site"], dtype=float)
        self.vector = np.asarray(info["hop_vector"], dtype=float)
        self.scale_A = float(np.linalg.norm(self.vector))
        if self.scale_A <= 0.0:
            raise ValueError("hop vector must be non-zero")
        self.midpoint = self.site + 0.5 * self.vector
        self._grad_row = self.vector / self.scale_A**2

    def value(self, atoms: Atoms) -> float:
        if self.index >= len(atoms):
            raise IndexError(f"hop atom {self.index} not in a {len(atoms)}-atom structure")
        from_mid, _ = mic_vectors(atoms.positions[self.index] - self.midpoint, atoms)
        return float((from_mid[0] + 0.5 * self.vector) @ self._grad_row)

    def gradient(self, atoms: Atoms) -> np.ndarray:
        grad = np.zeros((len(atoms), 3))
        grad[self.index] = self._grad_row
        return grad

    def __call__(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        return self.value(atoms), self.gradient(atoms)


def hop_cv(atoms: Atoms, info: dict[str, Any]) -> float:
    """The hop CV of ``atoms`` (0 = initial state, 1 = final state)."""
    return HopCV(info).value(atoms)


__all__ = [
    "B20_PARAMS",
    "CV_NAME",
    "DEFAULT_SUPERCELL",
    "HopCV",
    "P_BLOCK",
    "SPECIES_ALIASES",
    "b20_cell",
    "compound_of",
    "hop_cv",
    "hop_info_from_endpoints",
    "interpolate_endpoints",
    "mic_vectors",
    "resolve_species",
    "supercell_tuple",
    "vacancy_hop_endpoints",
]
