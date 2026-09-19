"""A stand-in for ``lmp`` used by the md tests (no LAMMPS with ``pair_style mace`` exists here).

Invoked as ``python fake_lmp.py [kokkos flags] -in <input> -log <log>`` in the job directory:

* an ``in.mace`` (header ``# b20mlip-md``) copies the real golden thermo log
  ``tests/fixtures/golden/lammps_thermo.log`` to ``<log>``, writes a format-faithful
  ``dump.lammpstrj`` (six frames of the atoms in ``data.lmp`` with random velocities, timesteps
  0..50) and ``final.data``;
* an ``in.parity`` (header ``# b20mlip-parity frame_id=...``) writes ``log.lammps`` with a
  ``Step PotEng`` thermo block and ``forces.dump`` from the JSON at ``$FAKE_LMP_PARITY_JSON``
  (``{frame_id: {energy, forces}}``); a frame missing from the JSON writes nothing.

``FAKE_LMP_CALLS`` (a file path) records every call; ``FAKE_LMP_FAIL=1`` exits 3 without output.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

GOLDEN_LOG = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "lammps_thermo.log"


def _args(argv: list[str]) -> tuple[str, str]:
    input_name, log_name = "in.lammps", "log.lammps"
    i = 0
    while i < len(argv):
        if argv[i] == "-in" and i + 1 < len(argv):
            input_name = argv[i + 1]
            i += 2
        elif argv[i] == "-log" and i + 1 < len(argv):
            log_name = argv[i + 1]
            i += 2
        else:
            i += 1
    return input_name, log_name


def _header(text: str, tag: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(tag):
            return dict(tok.split("=", 1) for tok in line[len(tag) :].split() if "=" in tok)
    return {}


def _read_data(path: Path) -> tuple[int, float, np.ndarray, list[int]]:
    """``(natoms, box edge, positions, types)`` of an orthogonal ``data.lmp``."""
    lines = path.read_text().splitlines()
    natoms = int(next(ln.split()[0] for ln in lines if ln.strip().endswith("atoms")))
    hi = float(next(ln.split()[1] for ln in lines if "xlo xhi" in ln))
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Atoms")) + 2
    rows = [ln.split() for ln in lines[start : start + natoms] if ln.strip()]
    types = [int(r[1]) for r in rows]
    pos = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
    return natoms, hi, pos, types


def _write_dump(path: Path, natoms: int, hi: float, pos: np.ndarray, types: list[int]) -> None:
    rng = np.random.default_rng(7)
    with open(path, "w", encoding="utf-8") as fh:
        for k in range(6):
            vel = rng.normal(0.0, 5.0, (natoms, 3))  # A/ps, metal units
            fh.write(f"ITEM: TIMESTEP\n{10 * k}\nITEM: NUMBER OF ATOMS\n{natoms}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n" + f"0 {hi}\n" * 3)
            fh.write("ITEM: ATOMS id type x y z vx vy vz\n")
            for i in range(natoms):
                p = pos[i] + 0.01 * k
                fh.write(
                    f"{i + 1} {types[i]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{vel[i, 0]:.6f} {vel[i, 1]:.6f} {vel[i, 2]:.6f}\n"
                )


def _md(input_text: str, log: Path) -> None:
    shutil.copyfile(GOLDEN_LOG, log)
    data = Path("data.lmp")
    if data.is_file():
        natoms, hi, pos, types = _read_data(data)
        _write_dump(Path("dump.lammpstrj"), natoms, hi, pos, types)
        shutil.copyfile(data, "final.data")


def _parity(input_text: str, log: Path) -> int:
    header = _header(input_text, "# b20mlip-parity")
    fid = header.get("frame_id", "")
    source = os.environ.get("FAKE_LMP_PARITY_JSON")
    if not source:
        sys.stderr.write("FAKE_LMP_PARITY_JSON is not set\n")
        return 2
    table = json.loads(Path(source).read_text())
    if fid not in table:
        sys.stderr.write(f"frame {fid} not in {source}\n")
        return 0  # a unit that leaves no output
    energy = float(table[fid]["energy"])
    forces = np.asarray(table[fid]["forces"], dtype=float)
    natoms = forces.shape[0]
    log.write_text(
        "LAMMPS (22 Jul 2025 - Update 6)\n"
        + input_text
        + "Per MPI rank memory allocation (min/avg/max) = 3.0 | 3.0 | 3.0 Mbytes\n"
        "   Step         PotEng     \n"
        f"         0 {energy:.15g}\n"
        f"Loop time of 1e-06 on 1 procs for 0 steps with {natoms} atoms\n\n"
        "Total wall time: 0:00:00\n"
    )
    with open("forces.dump", "w", encoding="utf-8") as fh:
        fh.write(f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{natoms}\n")
        fh.write("ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n")
        fh.write("ITEM: ATOMS id type fx fy fz\n")
        for i in range(natoms):
            fh.write(f"{i + 1} 1 {forces[i, 0]:.15g} {forces[i, 1]:.15g} {forces[i, 2]:.15g}\n")
    return 0


def main(argv: list[str]) -> int:
    input_name, log_name = _args(argv)
    calls = os.environ.get("FAKE_LMP_CALLS")
    if calls:
        with open(calls, "a", encoding="utf-8") as fh:
            fh.write(f"{Path.cwd()} {input_name}\n")
    if os.environ.get("FAKE_LMP_FAIL"):
        sys.stderr.write("fake lmp: forced failure\n")
        return 3
    text = Path(input_name).read_text(encoding="utf-8")
    log = Path(log_name)
    if "# b20mlip-parity" in text:
        return _parity(text, log)
    _md(text, log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
