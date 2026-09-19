"""Model metadata, reference cells and the numbers meta shared by the sampling stages."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from b20mlip.config import Settings
from b20mlip.models import CheckpointInfo, Frame
from b20mlip.provenance import sha256_file
from b20mlip.sampling import _common as c


def _checkpoint(model: Path, variant: str = "naive") -> CheckpointInfo:
    return CheckpointInfo(
        model_path=str(model), sha256=sha256_file(model), lammps_path=None, variant=variant,
        foundation="medium-mpa-0", foundation_sha256=None, heads=["Default"],
        e0_source="E0s_qe.json", energy_scale="qe", seed=0, epochs=1, lr=1e-4, batch_size=4,
        split_id=None, replay_samples=None, train_run_id=None,
    )  # type: ignore[arg-type]  # fmt: skip


def test_model_info_reads_checkpoint_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "train" / "r1"
    model = run_dir / "models" / "b20_naive.model"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    plain = c.model_info(model)
    assert plain.label == "b20_naive" and plain.e0_source == "unknown"
    assert plain.energy_scale == "none" and plain.checkpoint is None and plain.variant is None
    assert plain.sha256 == sha256_file(model) and plain.heads == ["Default"]
    (run_dir / "checkpoint.json").write_text(_checkpoint(model).model_dump_json())
    info = c.model_info(model)
    assert info.label == "B1" and info.e0_source == "E0s_qe.json" and info.energy_scale == "qe"
    assert info.variant == "naive" and info.checkpoint == str(run_dir / "checkpoint.json")
    assert info.reference() == {
        "code": "mace", "functional": "PBE", "pseudos": None, "e0_source": "E0s_qe.json",
        "cross_functional": False,
    }  # fmt: skip
    assert c.model_info(model, label="my label!").label == "my_label"
    # a checkpoint for another model file is ignored
    other = run_dir / "models" / "other.model"
    other.write_bytes(b"x")
    assert c.model_info(other).label == "other"
    (run_dir / "models" / "checkpoint.json").write_text("{not json")
    assert c.find_checkpoint(model) == run_dir / "checkpoint.json"
    foundation = tmp_path / "mace-mpa-0-medium.model"
    foundation.write_bytes(b"f")
    zero_shot = c.model_info(foundation)
    assert zero_shot.label == "B0" and zero_shot.e0_source == "foundation"
    assert zero_shot.energy_scale == "mp" and zero_shot.variant == "zero_shot"
    assert info.as_dict()["label"] == "B1"


def test_labels_heads_and_json_helpers(tmp_path: Path) -> None:
    assert c.sanitise_label(" B0′ ") == "B0" and c.sanitise_label("a.b/c") == "a_b_c"
    assert c.sanitise_label("--") == "model" and c.sanitise_label("ok-1_x") == "ok-1_x"
    assert c.head_label(None) == "Default" and c.head_label("pt_head") == "pt_head"
    assert c.head_label("weird") == "Default"
    payload = {
        "a": np.float64(1.5), "b": np.array([1, 2]), "c": float("nan"), "d": Path("x"),
        "e": (np.int64(3), True, None), "f": {"g": math.inf}, "h": object(),
    }  # fmt: skip
    safe = c.json_safe(payload)
    assert safe["a"] == 1.5 and safe["b"] == [1, 2] and safe["c"] is None and safe["d"] == "x"
    assert safe["e"] == [3, True, None] and safe["f"] == {"g": None} and isinstance(safe["h"], str)
    path = c.write_json(tmp_path / "sub" / "x.json", payload)
    assert json.loads(path.read_text())["c"] is None
    info = c.ModelInfo(
        path="m", sha256="s", label="B1", e0_source="E0s_qe.json", energy_scale="qe",
        heads=["Default"], variant="naive", checkpoint=None,
    )  # fmt: skip
    meta = c.numbers_meta(info, head="Default", n=12, seed=0, T=300.0, method="mbar",
                          ci95=(0.1, 0.3), unit="eV")  # fmt: skip
    assert meta["ci95"] == [0.1, 0.3] and "ci95_reason" not in meta and meta["unit"] == "eV"
    assert meta["model_label"] == "B1" and meta["model_sha256"] == "s" and meta["n"] == 12
    none = c.numbers_meta(info, head="Default", n=1, seed=0, T=0.0, method="neb", ci95=None)
    assert none["ci95"] is None and none["ci95_reason"] == "not a sampled quantity"


def test_reference_cell_sources(
    settings: Settings, tiny_b20_path: Path, tiny_frames: list[Frame]
) -> None:
    cell, source = c.reference_cell(settings, "FeSi")
    assert source == "tabulated" and len(cell) == 8 and cell.calc is None and cell.info == {}
    assert cell.get_chemical_formula("metal", empirical=True) == "FeSi"
    cell, source = c.reference_cell(settings, "MnSi", tiny_b20_path)
    assert source == str(tiny_b20_path) and len(cell) == 8 and cell.calc is None
    energies = [f.energy for f in tiny_frames if f.compound == "MnSi" and f.energy is not None]
    best = min(
        (f for f in tiny_frames if f.compound == "MnSi" and f.energy is not None),
        key=lambda f: f.energy,  # type: ignore[arg-type, return-value]
    )
    assert energies and np.allclose(cell.positions, best.positions)
    with pytest.raises(ValueError, match="no frame of compound"):
        c.reference_cell(settings, "NiSi", tiny_b20_path)
    with pytest.raises(ValueError, match="no tabulated"):
        c.reference_cell(settings, "NiSi")
