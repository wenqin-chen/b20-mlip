"""Train-tier fixtures: tiny settings, re-tagged frame files, a fake foundation file, E0s."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from b20mlip.config import Settings
from b20mlip.io import write_frames
from b20mlip.models import EnergyScale, Frame, LabelSource, Split
from b20mlip.provenance import sha256_frames

# Same tiny architecture as the root conftest's ``tiny_mace`` fixture (kept local on purpose:
# test packages never import each other's conftest modules).
TINY_TRAIN: dict[str, object] = {
    "epochs": 1,
    "batch_size": 4,
    "scratch": {"hidden_irreps": "8x0e", "r_max": 4.0},
    "extra_args": [
        "--num_interactions", "1", "--max_ell", "1", "--correlation", "2",
        "--num_radial_basis", "4",
    ],
}  # fmt: skip
FOUNDATION_BASENAME = "mace-mpa-0-medium.model"
E0S = {"14": -7.2, "25": -4.8, "26": -4.8, "27": -4.8, "32": -7.2}


def retag(frames: Sequence[Frame], scale: EnergyScale, source: LabelSource) -> list[Frame]:
    return [f.model_copy(update={"energy_scale": scale, "label_source": source}) for f in frames]


def make_split(frames: Sequence[Frame], seed: int = 0) -> Split:
    """A ``Split`` over ``frames``: strain frames -> val, rattle -> test, the rest -> train."""
    groups: dict[str, list[str]] = {}
    for f in frames:
        groups.setdefault(f.group_id, []).append(f.frame_id)
    return Split(
        split_id=f"test-split-s{seed}",
        seed=seed,
        frames_sha256=sha256_frames(frames),
        train=[f.frame_id for f in frames if f.config_type not in ("strain", "rattle")],
        val=[f.frame_id for f in frames if f.config_type == "strain"],
        test=[f.frame_id for f in frames if f.config_type == "rattle"],
        tiers={},
        groups=groups,
    )


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Settings:
    """Tiny scratch architecture, every path under tmp_path."""
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "runs_dir": str(tmp_path / "runs"),
                "models_dir": str(tmp_path / "models"),
                "dft_dir": str(tmp_path / "dft"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "train": TINY_TRAIN,
        }
    )


@pytest.fixture
def fake_foundation(tiny_cfg: Settings) -> Path:
    """A placeholder file where ``resolve_foundation`` looks first (never loaded by torch)."""
    path = Path(tiny_cfg.paths.models_dir) / "foundation" / FOUNDATION_BASENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real MACE model\n")
    return path


@pytest.fixture
def e0s_path(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "dft" / "E0s_qe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(E0S) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def qe_cfg(tiny_cfg: Settings, fake_foundation: Path, e0s_path: Path) -> Settings:
    """Tiny settings with a (fake) foundation model and an E0s_qe.json in place."""
    return tiny_cfg.model_copy(
        update={"train": tiny_cfg.train.model_copy(update={"e0s_file": str(e0s_path)})}
    )


@pytest.fixture
def qe_frames(tiny_frames: list[Frame]) -> list[Frame]:
    return retag(tiny_frames, "qe", "qe")


@pytest.fixture
def qe_frames_path(tmp_path: Path, qe_frames: list[Frame]) -> Path:
    path = tmp_path / "frames_qe.extxyz"
    write_frames(qe_frames, path)
    return path


@pytest.fixture
def omat_frames_path(tmp_path: Path, tiny_frames: list[Frame]) -> Path:
    path = tmp_path / "frames_omat24.extxyz"
    write_frames(retag(tiny_frames, "omat24", "omat24"), path)
    return path


@pytest.fixture
def split_paths(tmp_path: Path) -> dict[str, Path]:
    """Placeholder extxyz paths for pure argv tests (never read)."""
    return {part: tmp_path / "data" / f"{part}.extxyz" for part in ("train", "valid", "test")}


@pytest.fixture
def cluster_cfg(qe_cfg: Settings, tmp_path: Path) -> Settings:
    return qe_cfg.model_copy(
        update={
            "cluster": qe_cfg.cluster.model_copy(
                update={
                    "account": "acct",
                    "partition_cpu": "cpu-part",
                    "partition_gpu": "gpu-part",
                    "scratch": "/gscratch/b20",
                    "modules": ["gcc", "cuda"],
                    "control_path": str(tmp_path / "cm.sock"),
                }
            )
        }
    )
