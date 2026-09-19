"""Active-tier fixtures: the tiny calculator, a committee of the same model twice plus a
re-exported copy, and the fixture frames on disk."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from b20mlip.io import write_frames
from b20mlip.models import Frame


@pytest.fixture(scope="session")
def tiny_calc(tiny_mace: Any) -> Any:
    return tiny_mace.calculator()


@pytest.fixture(scope="session")
def committee_paths(tiny_mace: Any, tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Three committee members with identical weights: the file twice and a byte copy."""
    copy = tmp_path_factory.mktemp("committee") / "tiny_copy.model"
    shutil.copyfile(tiny_mace.model_path, copy)
    return [Path(tiny_mace.model_path), Path(tiny_mace.model_path), copy]


@pytest.fixture(scope="session")
def frames_path(tmp_path_factory: pytest.TempPathFactory, tiny_frames: list[Frame]) -> Path:
    path = tmp_path_factory.mktemp("active") / "candidates.extxyz"
    write_frames(tiny_frames, path)
    return path


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    data = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "runs_dir": str(tmp_path / "runs"),
            "models_dir": str(tmp_path / "models"),
            "dft_dir": str(tmp_path / "dft"),
            "reports_dir": str(tmp_path / "reports"),
        }
    }
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
