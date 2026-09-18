"""CONTRACTS.md section 5: run_id format, manifest round trip vs schema, run_stage failure path."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from b20mlip.config import Settings
from b20mlip.models import Artifact, Frame, Manifest, SlurmInfo, StageResult
from b20mlip.provenance import (
    RUN_ID_RE,
    RunContext,
    artifact_for,
    find_runs,
    git_info,
    infer_kind,
    make_run_id,
    manifest_json_schema,
    package_versions,
    read_manifest,
    run_stage,
    sha256_bytes,
    sha256_file,
    sha256_frames,
    write_manifest,
)

SHA_HELLO = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_hashes(tmp_path: Path, tiny_frames: list[Frame]) -> None:
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello")
    assert sha256_file(f) == SHA_HELLO == sha256_bytes(b"hello")
    art = artifact_for(f)
    assert art == Artifact(path=str(f), sha256=SHA_HELLO, bytes=5, kind="log")
    assert infer_kind("x.extxyz") == "frames" and infer_kind("m.model") == "model"
    assert infer_kind("a.json") == "json" and infer_kind("c.yaml") == "yaml"
    assert infer_kind("t.traj") == "traj" and infer_kind("weird.bin") == "other"
    with pytest.raises(FileNotFoundError):
        artifact_for(tmp_path / "missing")

    h = sha256_frames(tiny_frames)
    assert h == sha256_frames(list(tiny_frames)) and len(h) == 64
    assert sha256_frames(tiny_frames[::-1]) != h  # order is part of dataset identity
    assert sha256_frames([]) != h


def test_run_id_format() -> None:
    stamp = datetime(2026, 9, 17, 12, 34, 56, tzinfo=UTC)
    assert make_run_id("abcdef0123456789", 3, stamp) == "20260917T123456-abcdef-3"
    assert make_run_id("abcdef0123456789", None, stamp) == "20260917T123456-abcdef-none"
    assert RUN_ID_RE.match(make_run_id("0" * 64, 0))
    assert RUN_ID_RE.match("20260917T123456-abcdef-none")
    assert not RUN_ID_RE.match("2026-09-17-abcdef-1")


def test_git_info_and_packages(tmp_path: Path) -> None:
    sha, dirty = git_info(tmp_path)  # not a repository
    assert sha == "unknown" and dirty is False
    versions = package_versions()
    assert "pydantic" in versions and "b20-mlip" in versions


def _stage_ok(cfg: Settings, ctx: RunContext, *, n: int) -> dict[str, Any]:
    src = ctx.out_dir / "input.txt"
    src.write_text("in")
    ctx.add_input(src)
    out = ctx.out_dir / "result.json"
    out.write_text(json.dumps({"n": n}))
    ctx.add_output(out, "json")
    ctx.add_output(out, "json")  # re-adding the same path is idempotent
    ctx.log(model_sha256="m", head="Default", tier="T0")
    ctx.slurm = SlurmInfo(
        job_ids=["1"],
        account="a",
        partition="p",
        nodes=1,
        wall="01:00:00",
        units_done=n,
        units_failed=0,
    )
    return {"n": n}


def test_run_stage_ok_round_trip_and_schema(settings: Settings, repo: Path) -> None:
    result = run_stage("eval.errors", settings, _stage_ok, seed=7, n=3)
    assert result.status == "ok" and result.summary == {"n": 3}
    assert result.stage == "eval.errors" and RUN_ID_RE.match(result.run_id)
    assert result.run_id.endswith(f"-{settings.sha256()[:6]}-7")
    assert len(result.outputs) == 1 and result.outputs[0].kind == "json"

    manifest_path = Path(result.manifest_path)
    assert (
        manifest_path == settings.paths.runs_dir / "eval.errors" / result.run_id / "manifest.json"
    )
    manifest = read_manifest(manifest_path)
    assert read_manifest(manifest_path.parent) == manifest  # directory form
    assert manifest.status == "ok" and manifest.seed == 7 and manifest.wall_seconds >= 0
    assert manifest.config_sha256 == settings.sha256()
    assert manifest.config == settings.model_dump(mode="json")
    assert manifest.inputs[0].kind == "log" and manifest.outputs == result.outputs
    assert manifest.extras["tier"] == "T0" and manifest.slurm is not None
    assert manifest.python and manifest.host and manifest.user
    assert "pydantic" in manifest.packages
    assert manifest.created_at.tzinfo is not None

    raw = json.loads(manifest_path.read_text())
    assert Manifest.model_validate(raw) == manifest
    assert raw["schema_version"] == 1
    schema = json.loads((repo / "docs" / "manifest.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(raw)
    assert schema == manifest_json_schema(), "docs/manifest.schema.json drifted: run `make schema`"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    required = set(schema["required"])
    assert {
        "schema_version", "stage", "run_id", "created_at", "host", "user", "git_sha",
        "git_dirty", "uv_lock_sha256", "python", "packages", "config_sha256", "config",
        "inputs", "outputs", "wall_seconds", "status",
    } <= required | {"schema_version"}  # fmt: skip
    assert set(schema["$defs"]["Artifact"]["required"]) == {"path", "sha256", "bytes", "kind"}


def test_run_stage_failure_writes_manifest(settings: Settings) -> None:
    def boom(cfg: Settings, ctx: RunContext) -> None:
        ctx.add_output(_touch(ctx.out_dir / "partial.log"))
        raise RuntimeError("pw.x died")

    result = run_stage("dft.run", settings, boom, seed=0)
    assert result.status == "failed"
    assert result.summary == {"error": "RuntimeError('pw.x died')"}
    manifest = read_manifest(result.manifest_path)
    assert manifest.status == "failed"
    assert manifest.extras["error"] == "RuntimeError('pw.x died')"
    assert "pw.x died" in manifest.extras["traceback"]
    assert [a.path for a in manifest.outputs] == [
        str(settings.paths.runs_dir / "dft.run" / result.run_id / "partial.log")
    ]

    with pytest.raises(RuntimeError, match="pw.x died"):
        run_stage("dft.run", settings, boom, seed=1, reraise=True)
    assert [m.status for m in find_runs("dft.run", settings.paths.runs_dir)] == ["failed", "failed"]


def _touch(path: Path) -> Path:
    path.write_text("x")
    return path


def test_run_stage_dry_run_is_partial_without_outputs(settings: Settings) -> None:
    def plan(cfg: Settings, ctx: RunContext) -> dict[str, int]:
        assert ctx.dry_run
        ctx.add_output(_touch(ctx.out_dir / "would_be_output.json"))
        return {"planned_units": 5}

    result = run_stage("dft.prep", settings, plan, dry_run=True)
    assert result.status == "partial" and result.outputs == []
    assert result.summary == {"planned_units": 5}
    assert read_manifest(result.manifest_path).outputs == []
    assert result.run_id.endswith("-none")


def test_run_stage_accepts_stage_result(settings: Settings) -> None:
    def fn(cfg: Settings, ctx: RunContext) -> StageResult:
        art = artifact_for(_touch(ctx.out_dir / "a.yaml"))
        return StageResult(
            stage="report", run_id=ctx.run_id, manifest_path=str(ctx.manifest_path),
            status="partial", outputs=[art], summary={"note": "half"},
        )  # fmt: skip

    result = run_stage("report", settings, fn)
    assert result.status == "partial" and result.summary == {"note": "half"}
    assert [a.kind for a in result.outputs] == ["yaml"]
    assert read_manifest(result.manifest_path).status == "partial"


def test_find_runs_sorted_and_tolerant(settings: Settings) -> None:
    runs_dir = settings.paths.runs_dir
    assert find_runs("bench", runs_dir) == []
    first = run_stage("bench", settings, lambda cfg, ctx: {"i": 1}, seed=0)
    second = run_stage("bench", settings, lambda cfg, ctx: {"i": 2}, seed=1)
    assert first.run_id != second.run_id
    (runs_dir / "bench" / "junk").mkdir()
    (runs_dir / "bench" / "junk" / "manifest.json").write_text("{not json")
    (runs_dir / "bench" / "empty").mkdir()
    ids = [m.run_id for m in find_runs("bench", runs_dir)]
    assert ids == [first.run_id, second.run_id]


def test_resume_reuses_failed_run_only(settings: Settings) -> None:
    failed = run_stage("train", settings, lambda cfg, ctx: 1 / 0, seed=0)
    assert failed.status == "failed"
    ctx = RunContext("train", settings, 0, True, None)
    assert ctx.run_id == failed.run_id and ctx.resumed_from == failed.run_id
    assert ctx.out_dir == settings.paths.runs_dir / "train" / failed.run_id
    # a different seed or config never resumes someone else's run
    other = RunContext("train", settings, 1, True, None)
    assert other.run_id != failed.run_id and other.resumed_from is None
    ok = run_stage("train", settings, lambda cfg, ctx: {}, seed=0, resume=True)
    assert ok.run_id == failed.run_id and ok.status == "ok"
    fresh = RunContext("train", settings, 0, True, None)
    assert fresh.run_id != failed.run_id  # finished runs are never reused
    # a crashed run without a manifest is resumable too
    crashed = settings.paths.runs_dir / "train" / f"20200101T000000-{settings.sha256()[:6]}-0"
    crashed.mkdir(parents=True)
    assert RunContext("train", settings, 0, True, None).run_id == fresh.run_id  # newest first


def test_same_second_run_ids_are_unique(settings: Settings) -> None:
    stamp = datetime(2026, 9, 17, 1, 2, 3, tzinfo=UTC)
    a = RunContext("bench", settings, 0, False, None, now=stamp)
    b = RunContext("bench", settings, 0, False, None, now=stamp)
    assert a.run_id != b.run_id and a.out_dir.is_dir() and b.out_dir.is_dir()
    assert RUN_ID_RE.match(a.run_id) and RUN_ID_RE.match(b.run_id)
    assert a.run_id == "20260917T010203-" + settings.sha256()[:6] + "-0"
    assert re.match(r"20260917T010204-", b.run_id)


def test_write_manifest_atomic(tmp_path: Path, settings: Settings) -> None:
    ctx = RunContext("bench", settings, None, False, None)
    manifest = ctx.build_manifest("ok", 0.5)
    path = write_manifest(manifest, tmp_path / "nested" / "manifest.json")
    assert path.is_file() and not path.with_suffix(".json.tmp").exists()
    assert read_manifest(path) == manifest
