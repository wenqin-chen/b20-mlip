"""Enhanced sampling (CONTRACTS.md row 11): NEB barrier, umbrella windows, MBAR/WHAM free energy.

Modules: ``vacancy`` (vacancy-hop end states and the hop collective variable), ``neb``
(climbing-image NEB barrier), ``umbrella`` (harmonic bias + Langevin windows), ``wham``
(MBAR/WHAM free-energy profile with block errors) and ``cli`` (the ``sampling`` command group).
Heavy imports (torch, mace, pymbar) happen inside the stage functions only.
"""

from __future__ import annotations

__all__ = ["neb", "umbrella", "vacancy", "wham"]
