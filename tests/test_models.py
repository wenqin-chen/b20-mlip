"""CONTRACTS.md section 2 models: extra='forbid' everywhere, literal enums, defaults."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from b20mlip import models
from b20mlip.models import (
    AgentTask,
    Artifact,
    B20Model,
    Budget,
    DFTFrame,
    ErrorTable,
    Frame,
    Manifest,
    Metric,
    Reference,
    Split,
    StageResult,
)

ALL_MODELS = [
    cls
    for _, cls in inspect.getmembers(models, inspect.isclass)
    if issubclass(cls, BaseModel) and cls.__module__ == models.__name__
]


def minimal_frame(**overrides: object) -> Frame:
    data: dict[str, object] = {
        "frame_id": "0123456789abcdef",
        "group_id": "FeSi/relax/p0",
        "compound": "FeSi",
        "config_type": "relax",
        "parent_id": "p0",
        "numbers": [26, 14],
        "positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "cell": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
    }
    data.update(overrides)
    return Frame.model_validate(data)


CONTRACT_MODELS = {
    "Frame", "Split", "Artifact", "SlurmInfo", "Manifest", "DFTFrame", "CheckpointInfo",
    "Reference", "Metric", "ErrorTable", "PhononResult", "MDResult", "Budget", "AgentTask",
    "NumberRef", "AgentReport", "StageResult",
}  # fmt: skip


def test_every_model_forbids_extra_fields() -> None:
    assert {cls.__name__ for cls in ALL_MODELS} == CONTRACT_MODELS | {"B20Model"}
    for cls in ALL_MODELS:
        assert cls.model_config.get("extra") == "forbid", cls.__name__
        assert issubclass(cls, B20Model), cls.__name__


def test_frame_rejects_unknown_field_and_bad_enum() -> None:
    frame = minimal_frame()
    assert frame.pbc == (True, True, True)
    assert frame.label_source == "none" and frame.energy_scale == "none"
    with pytest.raises(ValidationError):
        minimal_frame(bogus=1)
    with pytest.raises(ValidationError):
        minimal_frame(config_type="wobble")
    with pytest.raises(ValidationError):
        minimal_frame(label_source="vasp")
    with pytest.raises(ValidationError):
        minimal_frame(energy_scale="pbesol")


def test_mutable_defaults_are_not_shared() -> None:
    a, b = minimal_frame(), minimal_frame()
    a.weights["energy"] = 0.0
    a.info["x"] = 1
    assert b.weights == {} and b.info == {}


def test_dftframe_extends_frame() -> None:
    base = minimal_frame().model_dump()
    dft = DFTFrame.model_validate(
        {
            **base,
            "code": "qe",
            "functional": "PBE",
            "pseudo_md5s": {"Fe": "abc"},
            "ecut_ry": 90.0,
            "k_spacing": 0.25,
            "nspin": 1,
            "smearing": "mv",
            "degauss_ry": 0.01,
            "converged": True,
            "scf_steps": 12,
            "abs_magnetization": None,
            "fermi_eV": 10.2,
            "branch_ok": None,
            "wall_seconds": 180.0,
            "unit_id": "u0001",
        }
    )
    assert isinstance(dft, Frame)
    with pytest.raises(ValidationError):
        DFTFrame.model_validate({**dft.model_dump(), "functional": "LDA"})


def test_split_tier_keys_are_literal() -> None:
    kwargs = dict(
        split_id="v1", seed=0, frames_sha256="0" * 64, train=[], val=[], test=[], groups={}
    )
    split = Split.model_validate({**kwargs, "tiers": {"T0": [], "T4b": ["x"]}})
    assert split.fractions == (0.8, 0.1, 0.1) and split.holdout_compounds == ["FeGe", "MnGe"]
    assert split.max_train_T == 600.0 and split.policy == "group_hash"
    with pytest.raises(ValidationError):
        Split.model_validate({**kwargs, "tiers": {"T9": []}})


def test_budget_defaults_match_spec() -> None:
    b = Budget()
    assert (b.max_calls, b.max_wall_s, b.max_dft_frames) == (25, 1800, 200)
    assert (b.max_node_hours, b.max_atoms, b.max_md_ps, b.approve_cluster) == (10, 512, 20, False)
    task = AgentTask(task_id="t1", prompt="p", tools_allowed=["list_data"])
    assert task.budget == Budget() and task.injected_failure is None
    with pytest.raises(ValidationError):
        AgentTask(task_id="t1", prompt="p", tools_allowed=[], injected_failure="network")


def test_manifest_literals() -> None:
    base = dict(
        stage="train",
        run_id="20260917T000000-abcdef-0",
        created_at=datetime.now(UTC),
        host="h",
        user="u",
        git_sha="deadbeef",
        git_dirty=False,
        uv_lock_sha256="",
        python="3.11.15",
        packages={},
        config_sha256="0" * 64,
        config={},
        seed=0,
        inputs=[],
        outputs=[],
        wall_seconds=1.0,
        status="ok",
    )
    m = Manifest.model_validate(base)
    assert m.schema_version == 1 and m.slurm is None and m.extras == {}
    with pytest.raises(ValidationError):
        Manifest.model_validate({**base, "schema_version": 2})
    with pytest.raises(ValidationError):
        Manifest.model_validate({**base, "stage": "train.naive"})
    with pytest.raises(ValidationError):
        Manifest.model_validate({**base, "status": "running"})
    with pytest.raises(ValidationError):
        Artifact(path="p", sha256="s", bytes=1, kind="checkpoint")


def test_metric_and_error_table() -> None:
    ref = Reference(code="qe", functional="PBE", pseudos="SSSP-eff-1.3", e0_source="E0s_qe.json")
    assert ref.cross_functional is False
    metric = Metric(value=1.0, ci95=None, n=10, unit="meV/A")
    table = ErrorTable(
        model_sha256="m",
        model_label="B1",
        head="Default",
        tier="T0",
        split_id="v1",
        reference=ref,
        n_frames=10,
        n_atoms=80,
        metrics={"mae_f": metric},
        bootstrap_n=2000,
        bootstrap_seed=0,
        run_id="r",
    )
    assert table.by_config_type == {} and table.noise_floor_f is None
    with pytest.raises(ValidationError):
        ErrorTable.model_validate({**table.model_dump(), "head": "pt"})
    with pytest.raises(ValidationError):
        ErrorTable.model_validate({**table.model_dump(), "tier": "T5"})


def test_stage_result_summary_types() -> None:
    result = StageResult(
        stage="bench", run_id="r", manifest_path="m", status="ok", outputs=[], summary={"n": 1}
    )
    assert result.summary == {"n": 1}
    with pytest.raises(ValidationError):
        StageResult(
            stage="bench",
            run_id="r",
            manifest_path="m",
            status="ok",
            outputs=[],
            summary={"x": [1]},
        )
