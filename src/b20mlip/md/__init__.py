"""MD tier (CONTRACTS.md row 10): ``ase_md`` (ASE NVE/NVT/NPT + VDOS/RDF), ``lammps`` (input
rendering, SLURM/local submission, thermo-log parsing) and ``parity`` (the 20-frame ASE-vs-LAMMPS
gate A6). Stage functions are ``ase_md.stage``, ``lammps.run`` and ``parity.run``; the physics
entry points keep the CONTRACTS.md section 6 signatures (``ase_md.run(atoms, calc, ...)``,
``lammps.render_input``, ``lammps.parse_thermo``, ``parity.check``)."""

from b20mlip.md import ase_md, lammps, parity

__all__ = ["ase_md", "lammps", "parity"]
