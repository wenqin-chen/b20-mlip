"""Fixtures of the dft tier: golden MnSi frame, deterministic settings, the fake ``pw.x``."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase.spacegroup import crystal

from b20mlip.config import Settings
from b20mlip.io import frame_from_atoms
from b20mlip.models import Frame

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
FAKE_PW = Path(__file__).resolve().parent / "fake_pw.py"
PSEUDO_DIR = "/gscratch/b20/pseudos/sssp_efficiency_1.3_pbe"
B20: dict[str, tuple[float, float, float, str, str]] = {
    "FeSi": (4.48, 0.137, 0.842, "Fe", "Si"),
    "CoSi": (4.44, 0.144, 0.843, "Co", "Si"),
    "MnSi": (4.56, 0.138, 0.846, "Mn", "Si"),
    "FeGe": (4.70, 0.135, 0.842, "Fe", "Ge"),
}


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR


@pytest.fixture
def golden_frame() -> Frame:
    return Frame.model_validate(json.loads((GOLDEN_DIR / "mnsi_8atom.frame.json").read_text()))


@pytest.fixture
def golden_cfg() -> Settings:
    """The configuration the golden pw.in was rendered with (absolute pseudo_dir, defaults)."""
    return Settings.model_validate({"dft": {"pseudo_dir": PSEUDO_DIR}})


@pytest.fixture(scope="session")
def fake_qe_cmd() -> str:
    return f"{sys.executable} {FAKE_PW}"


@pytest.fixture(scope="session")
def fake_pw() -> ModuleType:
    """``tests/dft/fake_pw.py`` as a module (its ``format_pw_out`` / ``parse_pw_in``)."""
    spec = importlib.util.spec_from_file_location("fake_pw", FAKE_PW)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dft_settings(settings: Settings, tmp_path: Path, fake_qe_cmd: str) -> Settings:
    """Isolated settings whose ``pw.x`` is the fake and whose dft/data dirs live in tmp_path."""
    (tmp_path / "pseudos").mkdir(exist_ok=True)
    return settings.model_copy(
        update={
            "cluster": settings.cluster.model_copy(update={"qe_cmd": fake_qe_cmd}),
            "dft": settings.dft.model_copy(update={"pseudo_dir": str(tmp_path / "pseudos")}),
        }
    )


@pytest.fixture
def slurm_settings(dft_settings: Settings, tmp_path: Path) -> Settings:
    return dft_settings.model_copy(
        update={
            "cluster": dft_settings.cluster.model_copy(
                update={
                    "account": "acct",
                    "partition_cpu": "cpu-part",
                    "partition_gpu": "gpu-part",
                    "scratch": "/gscratch/b20",
                    "modules": ["gcc", "openmpi", "quantum-espresso/7.3"],
                    "control_path": str(tmp_path / "cm.sock"),
                    "qe_cmd": "mpirun -np 8 pw.x -nk 2",
                }
            )
        }
    )


@pytest.fixture
def cli_args() -> Callable[[Path, str], list[str]]:
    """Global ``--set`` options isolating a CLI invocation in tmp_path with the fake pw.x."""

    def make(tmp_path: Path, fake_qe_cmd: str) -> list[str]:
        return [
            "--set", f"paths.runs_dir={tmp_path / 'runs'}",
            "--set", f"paths.dft_dir={tmp_path / 'dft'}",
            "--set", f"cluster.qe_cmd={fake_qe_cmd}",
            "--set", f"dft.pseudo_dir={tmp_path / 'pseudos'}",
        ]  # fmt: skip

    return make


@pytest.fixture
def b20_frame() -> Callable[..., Frame]:
    """Factory: ``b20_frame("MnSi", config_type="rattle", scale=1.0, rattle=0.0, seed=0)``."""

    def make(
        compound: str,
        config_type: str = "relax",
        *,
        scale: float = 1.0,
        rattle: float = 0.0,
        seed: int = 0,
    ) -> Frame:
        a, u, v, tm, x = B20[compound]
        atoms = crystal(
            [tm, x],
            basis=[(u, u, u), (v, v, v)],
            spacegroup=198,
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=False,
        )
        atoms.info.clear()
        if scale != 1.0:
            atoms.set_cell(atoms.cell[:] * scale, scale_atoms=True)
        if rattle:
            rng = np.random.default_rng(seed)
            atoms.positions = atoms.positions + rng.normal(0.0, rattle, atoms.positions.shape)
        parent = f"{compound}-p0"
        return frame_from_atoms(
            atoms,
            group_id=f"{compound}/{config_type}/{parent}",
            compound=compound,
            config_type=config_type,  # type: ignore[arg-type]
            parent_id=parent,
        )

    return make
