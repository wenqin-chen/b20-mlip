"""Optional lower-fidelity PBC baseline with PySCF (SPEC.md: never a headline reference).

``single_point(frame, cfg)`` runs KRKS (``nspin=1``) or KUKS (``nspin=2``) PBE with GTH
pseudopotentials and Gaussian density fitting and returns a ``DFTFrame`` with ``code="pyscf"``,
``functional="PBE"`` and ``info["lower_fidelity"]=True`` (audit gate A5 flags PySCF rows). The
import is guarded: without the ``[pyscf]`` extra the module imports fine and ``single_point``
raises ``ImportError`` with the install hint. A Mac probe showed an 8-atom FeSi cell does not
finish in 15 min, so this stays minimal (energy, forces when the gradient is available, no stress).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from b20mlip.config import Settings
from b20mlip.dft import qe
from b20mlip.models import DFTFrame, Frame

DEFAULT_BASIS = "gth-dzvp"
DEFAULT_PSEUDO = "gth-pbe"
INSTALL_HINT = "pyscf is not installed; install the optional extra: `uv sync --extra pyscf`"


def pyscf_available() -> bool:
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return False
    return True


def _require_pyscf() -> Any:
    try:
        import pyscf.pbc.gto  # noqa: F401
        import pyscf.pbc.scf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(INSTALL_HINT) from exc
    return __import__("pyscf")


def cell_spec(
    frame: Frame,
    cfg: Settings,
    *,
    basis: str = DEFAULT_BASIS,
    pseudo: str = DEFAULT_PSEUDO,
    kmesh: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Everything ``pyscf.pbc.gto.Cell`` needs (pure Python, testable without pyscf)."""
    atoms = [
        (chemical_symbols[int(z)], [float(x) for x in pos])
        for z, pos in zip(frame.numbers, frame.positions, strict=True)
    ]
    mesh = list(kmesh) if kmesh is not None else list(qe.kmesh(frame.cell, cfg.dft.k_spacing_inv_A))
    nspin = qe.nspin_for(frame.compound, cfg, frame.numbers)
    return {
        "atom": atoms,
        "a": np.asarray(frame.cell, dtype=float).tolist(),
        "basis": basis,
        "pseudo": pseudo,
        "unit": "Angstrom",
        "kmesh": mesh,
        "nspin": nspin,
        "xc": "pbe",
        "smearing_ha": cfg.dft.degauss_ry / 2.0,  # Ry -> Hartree
    }


def _compute(spec: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - needs pyscf + hours
    pyscf = _require_pyscf()
    from pyscf.pbc import gto, scf
    from pyscf.pbc.scf import addons

    cell = gto.Cell()
    cell.atom = spec["atom"]
    cell.a = spec["a"]
    cell.basis = spec["basis"]
    cell.pseudo = spec["pseudo"]
    cell.unit = spec["unit"]
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(spec["kmesh"])
    mf = scf.KUKS(cell, kpts) if spec["nspin"] == 2 else scf.KRKS(cell, kpts)
    mf.xc = spec["xc"]
    mf = mf.density_fit()
    mf = addons.smearing_(mf, sigma=spec["smearing_ha"], method="fermi")
    energy_ha = mf.kernel()
    forces = None
    try:
        grad = mf.nuc_grad_method().kernel()
        forces = (-np.asarray(grad) * pyscf.data.nist.HARTREE2EV / pyscf.data.nist.BOHR).tolist()
    except Exception:  # noqa: BLE001 - gradients are optional for the baseline
        forces = None
    magnetization = None
    if spec["nspin"] == 2:
        dm = mf.make_rdm1()
        magnetization = float(
            np.real(
                np.sum(
                    [np.trace(mf.get_ovlp()[k] @ (dm[0][k] - dm[1][k])) for k in range(len(kpts))]
                )
            )
            / len(kpts)
        )
    return {
        "energy_eV": float(energy_ha) * pyscf.data.nist.HARTREE2EV,
        "forces": forces,
        "converged": bool(mf.converged),
        "scf_steps": int(getattr(mf, "cycles", 0) or 0),
        "total_magnetization": magnetization,
        "pyscf_version": pyscf.__version__,
    }


def as_dft_frame(
    frame: Frame, spec: dict[str, Any], result: dict[str, Any], cfg: Settings
) -> DFTFrame:
    """Assemble the ``DFTFrame`` (kept separate from the compute so it is unit-tested)."""
    base = {name: getattr(frame, name) for name in Frame.model_fields}
    info = {
        **frame.info,
        "lower_fidelity": True,
        "basis": spec["basis"],
        "pseudo": spec["pseudo"],
        "density_fitting": True,
        "kmesh": list(spec["kmesh"]),
    }
    if result.get("pyscf_version"):
        info["pyscf_version"] = result["pyscf_version"]
    base.update(
        energy=result.get("energy_eV"),
        forces=result.get("forces"),
        stress=None,
        magmoms=None,
        total_magnetization=result.get("total_magnetization"),
        label_source="pyscf",
        energy_scale="none",
        info=info,
    )
    return DFTFrame(
        **base,
        code="pyscf",
        functional="PBE",
        pseudo_md5s={},
        ecut_ry=0.0,
        k_spacing=cfg.dft.k_spacing_inv_A,
        nspin=int(spec["nspin"]),
        smearing="fermi",
        degauss_ry=float(spec["smearing_ha"]) * 2.0,
        converged=bool(result.get("converged")),
        scf_steps=int(result.get("scf_steps", 0)),
        abs_magnetization=None,
        fermi_eV=None,
        branch_ok=None,
        wall_seconds=float(result.get("wall_seconds", 0.0)),
        unit_id=f"pyscf_{frame.frame_id}",
    )


def single_point(
    frame: Frame,
    cfg: Settings,
    *,
    basis: str = DEFAULT_BASIS,
    pseudo: str = DEFAULT_PSEUDO,
    kmesh: Sequence[int] | None = None,
) -> DFTFrame:
    """KRKS/KUKS PBE single point (GTH pseudos, density fitting); ``ImportError`` without pyscf."""
    if not pyscf_available():
        raise ImportError(INSTALL_HINT)
    import time

    spec = cell_spec(frame, cfg, basis=basis, pseudo=pseudo, kmesh=kmesh)
    t0 = time.perf_counter()
    result = _compute(spec)
    result["wall_seconds"] = time.perf_counter() - t0
    return as_dft_frame(frame, spec, result, cfg)


__all__ = [
    "DEFAULT_BASIS",
    "DEFAULT_PSEUDO",
    "INSTALL_HINT",
    "as_dft_frame",
    "cell_spec",
    "pyscf_available",
    "single_point",
]
