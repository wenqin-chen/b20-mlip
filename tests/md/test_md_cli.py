"""`b20mlip md ase|lammps|parity` through the root CLI (plugin hook), CliRunner smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from b20mlip.cli import REGISTERED, app
from b20mlip.md.cli import COMMANDS
from b20mlip.provenance import read_manifest

from .conftest import FAST_MD

runner = CliRunner()


def _payload(output: str) -> dict:
    """The StageResult JSON: MACE prints a cuequivariance notice before it and CliRunner may
    append torch warnings from stderr after it, so decode exactly one object."""
    return json.JSONDecoder().raw_decode(output[output.index("{") :])[0]


def _overlay(tmp_path: Path, **cluster: object) -> Path:
    data = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "runs_dir": str(tmp_path / "runs"),
            "models_dir": str(tmp_path / "models"),
        },
        "md": dict(FAST_MD),
        "cluster": cluster,
    }
    path = tmp_path / "md.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_md_commands_are_registered_and_documented() -> None:
    assert REGISTERED["md"] == COMMANDS == {"ase", "lammps", "parity"}
    result = runner.invoke(app, ["md", "--help"])
    assert result.exit_code == 0 and "[stub]" not in result.output
    for name in ("ase", "lammps", "parity"):
        assert name in result.output
        sub = runner.invoke(app, ["md", name, "--help"])
        assert sub.exit_code == 0 and "--model" in sub.output and "[stub]" not in sub.output
    assert "--ensemble" in runner.invoke(app, ["md", "ase", "--help"]).output
    assert "--lammps-json" in runner.invoke(app, ["md", "parity", "--help"]).output
    assert runner.invoke(app, ["md", "ase"]).exit_code == 2  # missing --model: usage error


def test_md_ase_cli(tmp_path: Path, tiny_mace, structure_file: Path) -> None:  # type: ignore[no-untyped-def]
    overlay = _overlay(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(overlay), "--seed", "4", "md", "ase", "--model", str(tiny_mace.model_path),
         "--compound", "MnSi", "--ensemble", "nve", "--T", "100", "--ps", "0.02", "--natoms", "8",
         "--structure", str(structure_file), "--label", "B1"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "md.ase" and payload["status"] == "ok"
    assert payload["summary"]["ensemble"] == "nve" and payload["summary"]["steps"] == 10
    assert (
        "drift_meV_atom_ps_100K" in payload["summary"] and payload["summary"]["model_label"] == "B1"
    )
    assert any(p.endswith("md_nve_100K.traj") for p in payload["outputs"])
    manifest = read_manifest(payload["manifest"])
    assert manifest.seed == 4 and manifest.extras["engine"] == "ase"
    numbers = json.loads((Path(payload["manifest"]).parent / "numbers.json").read_text())
    assert numbers["md.ase.MnSi.B1.nve.100.drift_meV_atom_ps@meta"]["seed"] == 4
    dry = runner.invoke(
        app,
        ["--config", str(overlay), "--dry-run", "md", "ase", "--model", str(tiny_mace.model_path),
         "--compound", "MnSi", "--ensemble", "npt", "--T", "100", "--T", "300", "--T", "500",
         "--ps", "1", "--structure", str(structure_file)],
    )  # fmt: skip
    assert dry.exit_code == 1 and _payload(dry.output)["status"] == "partial"
    assert _payload(dry.output)["summary"] == {"planned": 3, "steps": 500}
    bad = runner.invoke(
        app,
        ["--config", str(overlay), "md", "ase", "--model", str(tiny_mace.model_path),
         "--compound", "MnSi", "--ensemble", "nph", "--structure", str(structure_file)],
    )  # fmt: skip
    assert bad.exit_code == 1 and "ensemble" in _payload(bad.output)["summary"]["error"]


def test_md_lammps_cli(tmp_path: Path, tiny_mace, argon_file: Path, fake_lmp_cmd: str) -> None:  # type: ignore[no-untyped-def]
    overlay = _overlay(tmp_path, lammps_cmd=fake_lmp_cmd)
    result = runner.invoke(
        app,
        ["--config", str(overlay), "md", "lammps", "--model", str(tiny_mace.model_path),
         "--compound", "Ar", "--T", "100", "--ps", "0.04", "--natoms", "32", "--structure",
         str(argon_file), "--cpu"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "md.lammps" and payload["status"] == "ok"
    assert payload["summary"]["natoms"] == 32 and payload["summary"]["a_mean_A"] > 5.2
    assert read_manifest(payload["manifest"]).extras["gpu"] is False
    no_lammps = _overlay(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(no_lammps), "md", "lammps", "--model", str(tiny_mace.model_path),
         "--compound", "Ar", "--structure", str(argon_file)],
    )  # fmt: skip
    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["status"] == "failed" and "lammps_cmd not found" in payload["summary"]["error"]


def test_md_parity_cli(
    tmp_path: Path, tiny_mace, b20_frames, fake_lmp_cmd: str, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from b20mlip.io import write_frames
    from b20mlip.md import parity

    frames = b20_frames("FeSi", n=2)
    frames_file = tmp_path / "gate.extxyz"
    write_frames(frames, frames_file)
    reference = parity.ase_reference(None, frames, calc=tiny_mace.calculator())
    side = tmp_path / "lammps_side.json"
    side.write_text(json.dumps(reference))
    overlay = _overlay(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(overlay), "md", "parity", "--model", str(tiny_mace.model_path),
         "--frames", str(frames_file), "--lammps-json", str(side)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "md.parity" and payload["summary"]["parity_passed"] == 1
    assert payload["summary"]["n_frames"] == 2 and payload["summary"]["enough_frames"] == 0
    staged = runner.invoke(
        app,
        ["--config", str(overlay), "md", "parity", "--model", str(tiny_mace.model_path),
         "--frames", str(frames_file)],
    )  # fmt: skip
    assert staged.exit_code == 1 and _payload(staged.output)["status"] == "partial"
    assert "lammps_cmd not found" in _payload(staged.output)["summary"]["note"]
    monkeypatch.setenv("FAKE_LMP_PARITY_JSON", str(side))
    with_fake = _overlay(tmp_path, lammps_cmd=fake_lmp_cmd)
    ran = runner.invoke(
        app,
        ["--config", str(with_fake), "md", "parity", "--model", str(tiny_mace.model_path),
         "--frames", str(frames_file), "--tol-f", "0.01"],
    )  # fmt: skip
    assert ran.exit_code == 0, ran.output
    assert _payload(ran.output)["summary"]["parity_passed"] == 1
