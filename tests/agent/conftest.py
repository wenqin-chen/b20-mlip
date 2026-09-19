"""Agent-tier fixtures: tmp settings with 1x1x1 phonons and tiny MD, the fixture datasets under
``data_dir/frames`` (reference cells, QE-tagged labelled frames, a candidate pool), a two-member
committee (the tiny model and a byte copy), synthetic phonon references from the tiny model,
the task file and a ready :class:`ToolContext`."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from b20mlip.agent import tasks as tasks_mod
from b20mlip.agent.tools import ToolContext, write_phonon_reference
from b20mlip.config import Settings
from b20mlip.io import write_frames
from b20mlip.models import Frame

REPO = Path(__file__).resolve().parents[2]
TASKS_PATH = REPO / "evals" / "agent_tasks.jsonl"
TRACES_DIR = REPO / "tests" / "fixtures" / "traces"
BOOT_N = 30
FAST_MD: dict[str, object] = {
    "equil_ps": 0.0,
    "thermo_every_steps": 2,
    "dump_every_steps": 5,
    "vdos_every_steps": 1,
    "natoms": 8,
}


def agent_settings_for(root: Path) -> Settings:
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(root / "data"),
                "runs_dir": str(root / "runs"),
                "models_dir": str(root / "models"),
                "dft_dir": str(root / "dft"),
                "reports_dir": str(root / "reports"),
            },
            "agent": {
                "phonon_supercell": [1, 1, 1],
                "refs_dir": str(root / "refs"),
                "trace_dir": str(root / "traces"),
            },
            "eval": {"bootstrap_n": BOOT_N, "max_steps": 20},
            "md": dict(FAST_MD),
        }
    )


def populate(root: Path, tiny_frames: list[Frame], model_path: Path) -> None:
    """The datasets and committee copy every agent test needs, under ``root``."""
    frames_dir = root / "data" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    write_frames(
        [f for f in tiny_frames if f.config_type == "relax"], frames_dir / "mptrj_b20.extxyz"
    )
    qe = [f.model_copy(update={"energy_scale": "qe", "label_source": "qe"}) for f in tiny_frames]
    write_frames(qe, frames_dir / "labelled_r0.extxyz")
    write_frames(tiny_frames, frames_dir / "candidates_r0_filtered.extxyz")
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if not (models_dir / "tiny_copy.model").is_file():
        shutil.copyfile(model_path, models_dir / "tiny_copy.model")


@pytest.fixture(scope="session")
def tiny_calc(tiny_mace: Any) -> Any:
    return tiny_mace.calculator()


@pytest.fixture(scope="session")
def refs_dir(
    tmp_path_factory: pytest.TempPathFactory,
    tiny_mace: Any,
    tiny_frames: list[Frame],
    tiny_calc: Any,
) -> Path:
    """Synthetic ``phonons_<compound>_qe.json`` references computed with the tiny model (1x1x1)."""
    root = tmp_path_factory.mktemp("agent_refs")
    cfg = agent_settings_for(root)
    populate(root, tiny_frames, Path(tiny_mace.model_path))
    out = root / "refs"
    out.mkdir()
    for compound in ("FeSi", "CoSi", "MnSi"):
        write_phonon_reference(
            cfg, compound, tiny_calc, out, supercell=[1, 1, 1], source_label="synthetic-tiny"
        )
    return out


@pytest.fixture
def agent_settings(tmp_path: Path, tiny_mace: Any, tiny_frames: list[Frame]) -> Settings:
    populate(tmp_path, tiny_frames, Path(tiny_mace.model_path))
    return agent_settings_for(tmp_path)


@pytest.fixture
def tool_context(
    agent_settings: Settings, tiny_mace: Any, tiny_calc: Any, refs_dir: Path
) -> ToolContext:
    return ToolContext(
        agent_settings,
        model=Path(tiny_mace.model_path),
        committee=[
            Path(tiny_mace.model_path),
            Path(agent_settings.paths.models_dir) / "tiny_copy.model",
        ],
        refs_dir=refs_dir,
        calc=tiny_calc,
    )


@pytest.fixture
def tool_kwargs(tiny_mace: Any, tiny_calc: Any, refs_dir: Path) -> dict[str, Any]:
    """Keyword arguments for ``runner.run`` / ``run_task_stage`` (ready calculator, references)."""
    return {"model": Path(tiny_mace.model_path), "calc": tiny_calc, "refs_dir": refs_dir}


@pytest.fixture(scope="session")
def task_specs() -> list[tasks_mod.TaskSpec]:
    return tasks_mod.load_tasks(TASKS_PATH)


@pytest.fixture(scope="session")
def traces_dir() -> Path:
    assert TRACES_DIR.is_dir()
    return TRACES_DIR
