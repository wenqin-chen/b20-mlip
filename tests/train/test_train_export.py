"""export.to_lammps round trip (torch.jit.load), the export stage, and MACE's view of our files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from b20mlip.config import Settings
from b20mlip.provenance import read_manifest, run_stage, sha256_file
from b20mlip.train import export


def test_lammps_export_round_trip(tiny_mace: object, tmp_path: Path) -> None:
    import torch

    model_path = Path(getattr(tiny_mace, "model_path"))  # noqa: B009
    lammps_path = Path(getattr(tiny_mace, "lammps_path"))  # noqa: B009
    assert lammps_path == export.lammps_path_for(model_path)
    assert lammps_path.name.endswith("-lammps.pt") and lammps_path.stat().st_size > 0
    scripted = torch.jit.load(str(lammps_path), map_location="cpu")
    assert int(scripted.r_max.item()) == 4 and int(scripted.num_interactions.item()) == 1
    assert [int(z) for z in scripted.atomic_numbers.tolist()] == [14, 25, 26, 27, 32]
    assert int(scripted.head.item()) == 0

    cmd = export.export_command(model_path, "Default")
    assert cmd[:3] == [sys.executable, "-m", "mace.cli.create_lammps_model"]
    assert cmd[3:] == [str(model_path), "--head", "Default", "--dtype", "float64",
                       "--format", "libtorch"]  # fmt: skip
    assert export.choose_head(model_path) == "Default"
    with pytest.raises(ValueError, match="heads"):
        export.choose_head(model_path, "pt_head")
    with pytest.raises(FileNotFoundError):
        export.to_lammps(tmp_path / "missing.model")


def test_export_stage_writes_manifest_with_both_hashes(
    tiny_mace: object, tiny_cfg: Settings, tmp_path: Path
) -> None:
    src = Path(getattr(tiny_mace, "model_path"))  # noqa: B009
    model = tmp_path / "copy" / "tiny.model"
    model.parent.mkdir()
    shutil.copy2(src, model)

    dry = run_stage("export", tiny_cfg, export.run, dry_run=True, model=model)
    assert dry.status == "partial" and not export.lammps_path_for(model).exists()
    record = json.loads((Path(dry.manifest_path).parent / "export.json").read_text())
    assert record["lammps_sha256"] is None and record["dry_run"] is True

    result = run_stage("export", tiny_cfg, export.run, model=model, head="Default")
    assert result.status == "ok", result.summary
    lammps = export.lammps_path_for(model)
    assert lammps.is_file()
    manifest = read_manifest(result.manifest_path)
    record = json.loads((Path(result.manifest_path).parent / "export.json").read_text())
    assert record["model_sha256"] == sha256_file(model) == manifest.extras["model_sha256"]
    assert record["lammps_sha256"] == sha256_file(lammps) == manifest.extras["lammps_sha256"]
    assert record["head"] == "Default" and manifest.extras["head"] == "Default"
    assert {Path(a.path).name for a in manifest.outputs} == {"tiny.model-lammps.pt", "export.json"}
    assert [a.kind for a in manifest.inputs] == ["model"]
    assert result.summary["lammps_path"] == str(lammps)

    missing = run_stage("export", tiny_cfg, export.run, model=tmp_path / "nope.model")
    assert missing.status == "failed"


def test_mace_loader_sees_our_keys_and_per_config_weights(tiny_b20_path: Path) -> None:
    """mace-torch 0.3.16 reads energy/forces/stress from the calculator and config_*_weight."""
    from mace.data.utils import KeySpecification, load_from_xyz

    keyspec = KeySpecification.from_defaults()  # REF_* keys, head, dipole, ... (MACE's own)
    keyspec.update(
        info_keys={"energy": "energy", "stress": "stress"}, arrays_keys={"forces": "forces"}
    )
    _, configs = load_from_xyz(
        str(tiny_b20_path), key_specification=keyspec, head_name="Default",
        config_type_weights={"Default": 1.0},
    )  # fmt: skip
    assert len(configs) == 15
    zero = [c for c in configs if c.property_weights["energy"] == 0.0]
    assert len(zero) == 3 and all(c.config_type == "noise_floor" for c in zero)
    for c in configs:
        assert c.weight == 1.0 and c.head == "Default"
        assert c.property_weights["forces"] == 1.0 and c.property_weights["stress"] == 1.0
        assert isinstance(c.properties["energy"], float)
        assert np.asarray(c.properties["forces"]).shape == (8, 3)
        assert np.asarray(c.properties["stress"]).size in (6, 9)
