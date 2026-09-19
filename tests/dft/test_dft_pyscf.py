"""pyscf_pbc: the guarded import, the pure-Python cell spec / frame assembly, optional real run."""

from __future__ import annotations

import pytest

from b20mlip.config import Settings
from b20mlip.dft import pyscf_pbc
from b20mlip.models import DFTFrame, Frame


def test_cell_spec_and_frame_assembly(golden_frame: Frame, golden_cfg: Settings) -> None:
    spec = pyscf_pbc.cell_spec(golden_frame, golden_cfg)
    assert spec["atom"][0][0] == "Mn" and spec["atom"][4][0] == "Si" and len(spec["atom"]) == 8
    assert spec["a"] == golden_frame.cell and spec["kmesh"] == [6, 6, 6] and spec["nspin"] == 2
    assert spec["basis"] == "gth-dzvp" and spec["pseudo"] == "gth-pbe" and spec["xc"] == "pbe"
    assert spec["smearing_ha"] == pytest.approx(0.005)
    assert pyscf_pbc.cell_spec(golden_frame, golden_cfg, kmesh=[1, 1, 1], basis="gth-szv")[
        "kmesh"
    ] == [1, 1, 1]
    result = {
        "energy_eV": -1.5,
        "forces": [[0.0] * 3] * 8,
        "converged": True,
        "scf_steps": 12,
        "total_magnetization": 3.9,
        "pyscf_version": "2.14.0",
        "wall_seconds": 1.5,
    }
    frame = pyscf_pbc.as_dft_frame(golden_frame, spec, result, golden_cfg)
    assert isinstance(frame, DFTFrame) and frame.code == "pyscf" and frame.functional == "PBE"
    assert frame.info["lower_fidelity"] is True and frame.info["density_fitting"] is True
    assert frame.info["pyscf_version"] == "2.14.0" and frame.info["basis"] == "gth-dzvp"
    assert (
        frame.label_source == "pyscf"
        and frame.energy == -1.5
        and frame.converged
        and frame.scf_steps == 12
    )
    assert (
        frame.nspin == 2 and frame.degauss_ry == pytest.approx(0.01) and frame.smearing == "fermi"
    )
    assert frame.total_magnetization == 3.9 and frame.stress is None and frame.pseudo_md5s == {}
    assert frame.unit_id == f"pyscf_{golden_frame.frame_id}" and frame.wall_seconds == 1.5


def test_single_point_guard_without_pyscf(golden_frame: Frame, golden_cfg: Settings) -> None:
    if pyscf_pbc.pyscf_available():
        pytest.skip("pyscf is installed; the guard cannot be exercised")
    with pytest.raises(ImportError, match="uv sync --extra pyscf"):
        pyscf_pbc.single_point(golden_frame, golden_cfg)


@pytest.mark.slow
def test_single_point_runs_with_pyscf(golden_cfg: Settings) -> None:
    pytest.importorskip("pyscf")
    from b20mlip.dft.e0s import isolated_atom_frame

    frame = isolated_atom_frame("Si", 6.0)
    out = pyscf_pbc.single_point(frame, golden_cfg, basis="gth-szv", kmesh=[1, 1, 1])
    assert out.code == "pyscf" and out.energy is not None and out.info["lower_fidelity"] is True
