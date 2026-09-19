"""Evaluate-tier fixtures: the tiny MACE calculator, QE-tagged fixture frames, the FeSi cell,
the WBM mini summary, settings under tmp_path and a config overlay for CLI runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from ase import Atoms

from b20mlip.config import Settings
from b20mlip.io import frame_to_atoms, write_frames
from b20mlip.models import Frame

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WBM_MINI = FIXTURES / "wbm_mini.csv.gz"
BOOT_N = 60  # resamples in tests (the protocol default 2,000 is exercised by the config only)


def retag(frames: list[Frame], scale: str = "qe", source: str = "qe") -> list[Frame]:
    return [f.model_copy(update={"energy_scale": scale, "label_source": source}) for f in frames]


@pytest.fixture(scope="session")
def tiny_calc(tiny_mace: Any) -> Any:
    return tiny_mace.calculator()


@pytest.fixture(scope="session")
def qe_frames(tiny_frames: list[Frame]) -> list[Frame]:
    """Every fixture frame tagged as QE-labelled (energies comparable to a QE-scale model)."""
    return retag(tiny_frames)


@pytest.fixture(scope="session")
def qe_frames_path(tmp_path_factory: pytest.TempPathFactory, qe_frames: list[Frame]) -> Path:
    path = tmp_path_factory.mktemp("eval") / "frames_qe.extxyz"
    write_frames(qe_frames, path)
    return path


@pytest.fixture(scope="session")
def fesi_frame(tiny_frames: list[Frame]) -> Frame:
    return next(f for f in tiny_frames if f.compound == "FeSi" and f.config_type == "relax")


@pytest.fixture
def fesi_atoms(fesi_frame: Frame) -> Atoms:
    atoms = frame_to_atoms(fesi_frame)
    atoms.calc = None
    atoms.info = {}
    return atoms


@pytest.fixture(scope="session")
def fesi_structure_path(tmp_path_factory: pytest.TempPathFactory, fesi_frame: Frame) -> Path:
    """An extxyz holding only the FeSi reference cell (``--structure`` for phonons/elastic)."""
    path = tmp_path_factory.mktemp("struct") / "fesi.extxyz"
    write_frames([fesi_frame], path)
    return path


@pytest.fixture
def wbm_mini() -> Path:
    assert WBM_MINI.is_file()
    return WBM_MINI


@pytest.fixture
def eval_settings(settings: Settings) -> Settings:
    """tmp_path settings with a small bootstrap (tests never need 2,000 resamples)."""
    return settings.model_copy(
        update={"eval": settings.eval.model_copy(update={"bootstrap_n": BOOT_N, "max_steps": 30})}
    )


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    """A ``--config`` overlay putting every path under tmp_path with a small bootstrap."""
    data = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "runs_dir": str(tmp_path / "runs"),
            "models_dir": str(tmp_path / "models"),
            "dft_dir": str(tmp_path / "dft"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "eval": {"bootstrap_n": BOOT_N, "max_steps": 30},
    }
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
