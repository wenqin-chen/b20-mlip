"""DFT tier (CONTRACTS.md row 6): Quantum ESPRESSO inputs/outputs, E0s, offsets, PySCF baseline.

Modules
-------
* :mod:`b20mlip.dft.qe` — ``render_pw_input``, ``plan_units``, ``parse_pw_output``, ``collect``,
  the per-unit bash script and the unit conventions (energies eV, forces eV/Å, stress eV/Å³ in
  the ASE sign, k-mesh rule).
* :mod:`b20mlip.dft.stages` — the ``dft prep`` / ``dft run`` / ``dft collect`` stage functions.
* :mod:`b20mlip.dft.converge` — cutoff x k-spacing scan and the cheapest-converged selection.
* :mod:`b20mlip.dft.phonons` — phonopy displacement sets and ``force_sets.json``.
* :mod:`b20mlip.dft.e0s` — isolated-atom energies in the MACE ``--E0s`` format.
* :mod:`b20mlip.dft.offsets` — per-element QE -> MP energy-scale map and its residual gate.
* :mod:`b20mlip.dft.pyscf_pbc` — optional lower-fidelity PySCF PBC single point (guarded import).
* :mod:`b20mlip.dft.cli` — typer commands plugged into the root ``b20mlip dft`` group.

Nothing here needs ``pw.x`` on the machine that plans or parses: inputs are rendered, units are
planned and outputs are parsed from files, so the whole tier is tested offline against fixtures.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "converge",
    "e0s",
    "offsets",
    "phonons",
    "pyscf_pbc",
    "qe",
    "stages",
    "structures",
]
