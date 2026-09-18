"""collect(): only ok manifests, newest wins, stale detection, file format, loaders, validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from b20mlip.config import Settings
from b20mlip.models import NumberRef
from b20mlip.provenance import sha256_file
from b20mlip.report import numbers as nums

from .conftest import FixtureRuns, RunMaker


def test_collect_only_ok_newest_wins_and_stale(cfg: Settings, fixture_runs: FixtureRuns) -> None:
    out = Path(cfg.report.numbers_path)
    refs = nums.collect(cfg.paths.runs_dir, out=out)
    assert set(refs) == {
        "eval.errors.T0.B0.mae_f",
        "eval.errors.T0.B1.mae_f",
        "eval.errors.T3.B0.mae_f",
        "parity.passed",
    }
    newer = fixture_runs.newer
    assert refs["eval.errors.T0.B1.mae_f"] == NumberRef(
        key="eval.errors.T0.B1.mae_f",
        value=30.5,
        run_id=newer.run_id,
        manifest_sha256=sha256_file(newer.manifest_path),
    )
    assert refs["eval.errors.T0.B0.mae_f"].run_id == fixture_runs.ok.run_id
    assert refs["parity.passed"].value == 0.0  # bool -> 0/1

    data = json.loads(out.read_text())
    assert list(data)[0] == "@stale"
    stale_path = Path(fixture_runs.stale.outputs[0].path)
    assert data["@stale"] == [
        {
            "key": "md.ase.MnSi.a_300K_A",
            "run_id": fixture_runs.stale.run_id,
            "stage": "md.ase",
            "path": str(stale_path),
            "expected_sha256": fixture_runs.stale.outputs[0].sha256,
            "actual_sha256": sha256_file(stale_path),
        }
    ]
    keys = [k for k in data if k != "@stale"]
    assert keys == sorted(keys)
    entry = data["eval.errors.T0.B1.mae_f"]
    assert entry["stage"] == "eval.errors" and entry["value"] == 30.5
    assert entry["meta"]["reference"]["code"] == "qe"  # file-level @meta merged in
    assert entry["meta"]["ci95"] == [28.9, 32.0] and entry["meta"]["bracket"] == "B1"
    assert entry["meta"]["tier"] == "T0"

    assert nums.load(out) == refs
    entries, stale = nums.load_entries(out)
    assert set(entries) == set(refs) and len(stale) == 1
    assert nums.load_entries(out.with_name("missing.json")) == ({}, [])
    harvest = nums.harvest(cfg.paths.runs_dir)
    assert harvest.n_manifests == 5 and harvest.n_ok == 3
    assert harvest.refs() == refs


def test_collect_skips_own_stage_unregistered_and_absent_files(
    cfg: Settings, make_run: RunMaker
) -> None:
    make_run("report", {"x.y": 1.0})  # our own output is never harvested
    make_run("bench", {"bench.s_per_frame": 0.12}, register=False)  # not in manifest outputs
    make_run("bench", None)  # no numbers.json at all
    assert nums.collect(cfg.paths.runs_dir, out=None) == {}


def test_missing_numbers_file_is_stale_star(cfg: Settings, make_run: RunMaker) -> None:
    result = make_run("bench", {"bench.x": 1.0})
    Path(result.outputs[0].path).unlink()
    harvest = nums.harvest(cfg.paths.runs_dir)
    assert harvest.entries == {}
    assert harvest.stale[0]["key"] == "*" and harvest.stale[0]["actual_sha256"] is None


def test_collect_default_out(
    cfg: Settings, make_run: RunMaker, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    make_run("bench", {"bench.x": 2})
    refs = nums.collect(cfg.paths.runs_dir)
    assert (tmp_path / "reports" / "numbers.json").is_file() and refs["bench.x"].value == 2.0


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([1], "top level"),
        ({"bad key": 1}, "invalid key"),
        ({"a.b": "12"}, "finite number"),
        ({"a.b": float("nan")}, "finite number"),
        ({"a.b@meta": {"n": 1}}, "meta without a number"),
        ({"a.b": 1, "a.b@meta": 3}, "must be an object"),
        ({"@meta": 3, "a.b": 1}, "must be an object"),
    ],
)
def test_parse_numbers_file_rejects(data: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        nums.parse_numbers_file(data)


def test_parse_numbers_file_coerces_and_merges() -> None:
    parsed = nums.parse_numbers_file(
        {
            "@meta": {"head": "Default", "seed": 0},
            "a.ok": True,
            "a.n": 3,
            "a.n@meta": {"seed": 1, "n": 3},
        }
    )
    assert parsed == {
        "a.ok": (1.0, {"head": "Default", "seed": 0}),
        "a.n": (3.0, {"head": "Default", "seed": 1, "n": 3}),
    }


def test_harvest_is_loud_on_a_bad_stage_file(cfg: Settings, make_run: RunMaker) -> None:
    make_run("bench", {"bench.x": "oops"})
    with pytest.raises(ValueError, match="bench/"):
        nums.harvest(cfg.paths.runs_dir)


def test_load_rejects_malformed_files(tmp_path: Path) -> None:
    path = tmp_path / "numbers.json"
    path.write_text(json.dumps({"a.b": 1}))
    with pytest.raises(ValueError, match="must be an object"):
        nums.load(path)
    path.write_text("[1]")
    with pytest.raises(ValueError, match="top level"):
        nums.load_entries(path)
