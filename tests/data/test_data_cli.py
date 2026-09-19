"""`b20mlip data {pull,sample,filter,split}` wired through the root CLI plugin hook."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from helpers_data import b20_frame
from typer.testing import CliRunner

from b20mlip.cli import REGISTERED, app
from b20mlip.data import pull
from b20mlip.data.cli import COMMANDS, register
from b20mlip.io import read_frames, write_frames

runner = CliRunner()


def _base(tmp_path: Path) -> list[str]:
    return [
        "--set", f"paths.data_dir={tmp_path / 'data'}",
        "--set", f"paths.runs_dir={tmp_path / 'runs'}",
    ]  # fmt: skip


def test_registration() -> None:
    assert REGISTERED["data"] == COMMANDS == {"pull", "sample", "filter", "split"}
    import typer

    assert register(typer.Typer()) == COMMANDS
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for name in ("pull", "sample", "filter", "split"):
        assert name in result.output
    assert "[stub]" not in result.output
    assert "--extract" in runner.invoke(app, ["data", "pull", "--help"]).output


def test_pull_dry_run_is_partial(tmp_path: Path) -> None:
    result = runner.invoke(
        app, [*_base(tmp_path), "--dry-run", "data", "pull", "--sources", "phonondb"]
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "partial" and payload["stage"] == "data.pull"
    assert payload["summary"]["phonondb_pbe_103_structures.status"] == "planned_download"
    assert (tmp_path / "runs" / "data.pull").is_dir()


def test_pull_verifies_local_file_and_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phonondb_file: Path
) -> None:
    src = pull.Source("phonondb", "phonondb_pbe_103_structures", "https://f.test/ph",
                      "raw/references/phononDB-PBE-103-structures.extxyz",
                      phonondb_file.stat().st_size, "CC-BY-4.0")  # fmt: skip
    monkeypatch.setattr(pull, "SOURCES", {"phonondb": [src]})
    dest = tmp_path / "data" / src.relpath
    dest.parent.mkdir(parents=True)
    shutil.copy(phonondb_file, dest)
    result = runner.invoke(
        app, [*_base(tmp_path), "data", "pull", "--sources", "phonondb", "--extract"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok" and payload["summary"]["phonondb_structures"] == 2
    assert payload["summary"]["phonondb_pbe_103_structures.status"] == "verified"
    assert any(p.endswith("sources.json") for p in payload["outputs"])


def test_pull_unknown_source_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, [*_base(tmp_path), "data", "pull", "--sources", "bogus"])
    assert result.exit_code == 1 and "unknown source" in result.output


def test_sample_filter_split_pipeline(tmp_path: Path) -> None:
    parents = tmp_path / "parents.extxyz"
    write_frames([b20_frame("FeSi", seed=1), b20_frame("CoSi", seed=2)], parents)
    cand = tmp_path / "cand.extxyz"
    result = runner.invoke(
        app,
        [*_base(tmp_path), "--seed", "0", "data", "sample", "--parents", str(parents), "--out",
         str(cand), "--config-types", "strain,eos", "--compounds", "FeSi,CoSi"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)["summary"]
    assert summary["frames"] == 2 * (24 + 11) and summary["n_eos"] == 22
    assert len(read_frames(cand)) == 70
    filtered = tmp_path / "filtered.extxyz"
    result = runner.invoke(
        app, [*_base(tmp_path), "data", "filter", "--frames", str(cand), "--out", str(filtered)]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)["summary"]
    assert summary["n_in"] == 70 and summary["n_out"] == summary["n_after_dedupe"]
    # per cubic parent: 24 strain -> 6 iso + 6 uniaxial (x/y/z are rotated copies) = 12;
    # 11 eos -> 9 (the 0.936 and 1.064 volume points sit within 0.2 % of the +-2 % iso strains)
    assert summary["n_out"] == 42
    result = runner.invoke(
        app,
        [*_base(tmp_path), "--seed", "1", "data", "split", "--frames", str(filtered), "--out",
         str(tmp_path / "splits"), "--fractions", "0.8,0.1,0.1", "--holdout", "FeGe,MnGe"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    split_id = payload["summary"]["split_id"]
    assert (tmp_path / "splits" / f"{split_id}.json").is_file()
    assert payload["summary"]["n_train"] == 0  # unlabelled candidates never train


def test_sample_rejects_unknown_config_type(tmp_path: Path) -> None:
    result = runner.invoke(
        app, [*_base(tmp_path), "data", "sample", "--config-types", "bogus", "--out", "x.extxyz"]
    )
    assert result.exit_code == 2 and "unknown config types" in result.output


def test_filter_and_split_need_frames_option(tmp_path: Path) -> None:
    assert runner.invoke(app, [*_base(tmp_path), "data", "filter", "--out", "o"]).exit_code == 2
    assert runner.invoke(app, [*_base(tmp_path), "data", "split"]).exit_code == 2
