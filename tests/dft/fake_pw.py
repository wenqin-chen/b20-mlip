"""Test double for ``pw.x``: ``python fake_pw.py -in pw.in > pw.out``.

Modes (environment variables, read by the ``dft run`` unit script's child process):

* ``FAKE_PW_MODE=golden`` (default): print ``tests/fixtures/golden/pw.out`` verbatim, whatever the
  input (the 8-atom MnSi case of the parser tests).
* ``FAKE_PW_MODE=model``: parse the real ``pw.in`` (cell, positions, species, cutoffs, k-mesh,
  nspin), compute Lennard-Jones forces/stress with ASE and an energy that converges with cutoff
  and k-mesh, and print a QE 7.x-formatted output for that geometry (converge, phonons and E0s
  tests). Energy per atom: ``E_LJ/N + 0.050 exp(-(ecut-40)/10) + 0.200/n_k`` eV.
* ``FAKE_PW_FAIL_UNITS=a,b``: units (cwd basename) that print ``pw_unconverged.out`` instead.
* ``FAKE_PW_CALLS=<path>``: append the unit id to this file on every call (resume tests).

Importable too: ``format_pw_out`` builds the synthetic output used in ``model`` mode.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parents[0] / "fixtures" / "golden"
RY = 13.605693012183622  # ase.units.Ry (CODATA 2014), the constant b20mlip.dft.qe uses
BOHR = 0.5291772105638411  # ase.units.Bohr
KBAR_PER_RY_BOHR3 = RY / BOHR**3 / 6.241509125883258e-4


def parse_pw_in(text: str) -> dict:
    """The parts of a ``pw.in`` the fake needs (ibrav=0, angstrom cards, automatic/gamma k)."""
    lines = text.splitlines()

    def namelist_value(key: str, default: str) -> str:
        m = re.search(rf"^\s*{key}\s*=\s*([^\s,]+)", text, re.M)
        return m.group(1).strip("'") if m else default

    def card(name: str) -> list[str]:
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(name))
        out = []
        for ln in lines[start + 1 :]:
            if not ln.strip() or re.match(r"^[A-Z_]+( |$)", ln.strip()):
                break
            out.append(ln.split())
        return out

    cell = np.array([[float(x) for x in row] for row in card("CELL_PARAMETERS")])
    atoms = card("ATOMIC_POSITIONS")
    symbols = [row[0] for row in atoms]
    positions = np.array([[float(x) for x in row[1:4]] for row in atoms])
    species = [row[0] for row in card("ATOMIC_SPECIES")]
    kline = next(
        (lines[i + 1].split() for i, ln in enumerate(lines) if ln.startswith("K_POINTS automatic")),
        None,
    )
    kmesh = [int(x) for x in kline[:3]] if kline else [1, 1, 1]
    return {
        "cell": cell,
        "symbols": symbols,
        "positions": positions,
        "species": species,
        "kmesh": kmesh,
        "ecutwfc": float(namelist_value("ecutwfc", "90.0")),
        "ecutrho": float(namelist_value("ecutrho", "1080.0")),
        "nspin": int(namelist_value("nspin", "1")),
        "prefix": namelist_value("prefix", "pwscf"),
    }


def model_labels(spec: dict) -> dict:
    """LJ energy/forces/stress (eV, eV/Å, eV/Å³ ASE sign) with a cutoff/k-mesh dependence."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    atoms = Atoms(symbols=spec["symbols"], positions=spec["positions"], cell=spec["cell"], pbc=True)
    atoms.calc = LennardJones(sigma=2.0, epsilon=0.5, rc=5.0, smooth=True)
    n = len(atoms)
    nk = int(np.prod(spec["kmesh"]))
    ecut = spec["ecutwfc"]
    e_atom_extra = 0.050 * math.exp(-(ecut - 40.0) / 10.0) + 0.200 / nk
    energy = float(atoms.get_potential_energy()) + n * e_atom_extra
    forces = np.asarray(atoms.get_forces()) * (1.0 + 0.001 * math.exp(-(ecut - 40.0) / 10.0))
    stress = np.asarray(atoms.get_stress(voigt=False)) if n > 1 else np.zeros((3, 3))
    tm = {"Mn", "Fe", "Co"}
    magmoms = None
    if spec["nspin"] == 2:
        magmoms = [1.0 if s in tm else -0.04 for s in spec["symbols"]]
    return {"energy_eV": energy, "forces_eV_A": forces, "stress_eV_A3": stress, "magmoms": magmoms}


def format_pw_out(
    spec: dict,
    labels: dict,
    *,
    converged: bool = True,
    n_iter: int = 7,
    fermi_eV: float = 12.3456,
    wall_s: float = 42.5,
    version: str = "7.3",
) -> str:
    """A QE 7.x-shaped ``pw.out`` (same blocks the golden fixture has) for ``spec``/``labels``."""
    n = len(spec["symbols"])
    types = {s: i + 1 for i, s in enumerate(spec["species"])}
    e_ry = labels["energy_eV"] / RY
    f_ry = np.asarray(labels["forces_eV_A"]) * BOHR / RY
    s_qe = -np.asarray(labels["stress_eV_A3"]) * BOHR**3 / RY  # QE sign: compressive positive
    magmoms = labels.get("magmoms")
    out = [
        "",
        f"     Program PWSCF v.{version} starts on 18Sep2026 at 10:00:00 ",
        "",
        "     Reading input from pw.in",
        "",
        f"     number of atoms/cell      =        {n:5d}",
        f"     number of atomic types    =        {len(spec['species']):5d}",
        f"     kinetic-energy cutoff     =    {spec['ecutwfc']:10.4f}  Ry",
        f"     charge density cutoff     =    {spec['ecutrho']:10.4f}  Ry",
        "     Exchange-correlation= PBE",
        "",
        "     Self-consistent Calculation",
    ]
    for it in range(1, n_iter + 1):
        out += [
            "",
            f"     iteration #{it:3d}     ecut=    {spec['ecutwfc']:5.2f} Ry     beta= 0.40",
            "     Davidson diagonalization with overlap",
            "",
            f"     total energy              =   {e_ry + 0.01 / it:15.8f} Ry",
            f"     estimated scf accuracy    <   {0.1 / it**3:15.8f} Ry",
        ]
        if magmoms is not None:
            out += [
                "",
                f"     total magnetization       = {sum(magmoms):8.2f} Bohr mag/cell",
                "     absolute magnetization    = "
                f"{sum(abs(m) for m in magmoms):8.2f} Bohr mag/cell",
            ]
    out += ["", "     End of self-consistent calculation", ""]
    if not converged:
        out += [
            f"     convergence NOT achieved after {n_iter:3d} iterations: stopping",
            "",
            f"     PWSCF        : {wall_s * 0.9:9.2f}s CPU {wall_s:9.2f}s WALL",
            "",
            "   JOB DONE.",
            "",
        ]
        return "\n".join(out)
    out += [
        f"     the Fermi energy is {fermi_eV:10.4f} ev",
        "",
        f"!    total energy              =   {e_ry:15.8f} Ry",
        "     estimated scf accuracy    <          1.2E-09 Ry",
        "",
    ]
    if magmoms is not None:
        out += [
            f"     total magnetization       = {sum(magmoms):8.2f} Bohr mag/cell",
            f"     absolute magnetization    = {sum(abs(m) for m in magmoms):8.2f} Bohr mag/cell",
            "",
            "     Magnetic moment per site  (integrated on atomic sphere of radius R)",
        ]
        out += [
            f"     atom {i + 1:3d} (R=0.357)  charge= {12.0:7.4f}  magn= {m:7.4f}"
            for i, m in enumerate(magmoms)
        ]
        out.append("")
    out += [f"     convergence has been achieved in {n_iter:3d} iterations", ""]
    out += ["     Forces acting on atoms (cartesian axes, Ry/au):", ""]
    for i, (sym, f) in enumerate(zip(spec["symbols"], f_ry, strict=True)):
        out.append(
            f"     atom {i + 1:4d} type {types[sym]:2d}   force = "
            f"{f[0]:14.8f}{f[1]:14.8f}{f[2]:14.8f}"
        )
    out += ["     The non-local contrib.  to forces"]
    for i, sym in enumerate(spec["symbols"]):
        out.append(
            f"     atom {i + 1:4d} type {types[sym]:2d}   force = {0.1:14.8f}{0.1:14.8f}{0.1:14.8f}"
        )
    total = float(np.sqrt(np.sum(f_ry**2)))
    out += ["", f"     Total force = {total:12.6f}     Total SCF correction =     0.000001", ""]
    p_kbar = float(np.trace(s_qe) / 3.0 * KBAR_PER_RY_BOHR3)
    out += [
        "",
        "     Computing stress (Cartesian axis) and pressure",
        "",
        f"          total   stress  (Ry/bohr**3)                   (kbar)     P= {p_kbar:11.2f}",
    ]
    for row in range(3):
        ry = "".join(f"{s_qe[row, c]:13.8f}" for c in range(3))
        kb = "".join(f"{s_qe[row, c] * KBAR_PER_RY_BOHR3:12.2f}" for c in range(3))
        out.append(f"{ry}    {kb}")
    out += [
        "",
        "",
        f"     PWSCF        : {wall_s * 0.9:9.2f}s CPU {wall_s:9.2f}s WALL",
        "",
        "   JOB DONE.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    unit = Path.cwd().name
    calls = os.environ.get("FAKE_PW_CALLS")
    if calls:
        with open(calls, "a", encoding="utf-8") as fh:
            fh.write(unit + "\n")
    fail_units = {u for u in os.environ.get("FAKE_PW_FAIL_UNITS", "").split(",") if u}
    if unit in fail_units:
        sys.stdout.write((GOLDEN / "pw_unconverged.out").read_text(encoding="utf-8"))
        return 0
    mode = os.environ.get("FAKE_PW_MODE", "golden")
    if mode == "golden":
        sys.stdout.write((GOLDEN / "pw.out").read_text(encoding="utf-8"))
        return 0
    in_path = Path(argv[argv.index("-in") + 1]) if "-in" in argv else Path("pw.in")
    spec = parse_pw_in(in_path.read_text(encoding="utf-8"))
    sys.stdout.write(format_pw_out(spec, model_labels(spec)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
