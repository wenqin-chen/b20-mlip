"""cluster sync: sacct + rsync; units are done only when their marker came back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from b20mlip.cluster.sync import elapsed_seconds, job_ids_of, pull, units_of
from b20mlip.config import Settings
from b20mlip.executors import ClusterUnreachable, JobHandle
from b20mlip.provenance import RunContext, read_manifest, run_stage

from .conftest import SACCT_DONE, SACCT_RUNNING, SCRATCH, FakeRunner, Scenario


def _remote_job(remote_root: Path, name: str = "qe_r0") -> Path:
    job = remote_root / "jobs" / name
    (job / "units").mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "units.txt").write_text("u1\nu2\nu3\n")
    (job / "units" / "u1.done").write_text('{"unit": "u1", "state": "done", "returncode": 0}\n')
    (job / "units" / "u3.failed").write_text('{"unit": "u3", "state": "failed", "returncode": 1}\n')
    (job / "logs" / "u1.log").write_text("JOB DONE\n")
    (job / "u1").mkdir()
    (job / "u1" / "pw.out").write_text("!    total energy = -1.0 Ry\n")
    return job


def _local_handle(cfg: Settings, name: str = "qe_r0", ids: list[str] | None = None) -> Path:
    dest = Path(cfg.paths.runs_dir) / "slurm" / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "handle.json").write_text(
        JobHandle(
            job_ids=ids or ["4243"], workdir=f"{SCRATCH}/b20-mlip/jobs/{name}"
        ).model_dump_json()
    )
    return dest


def test_pull_marks_units_from_markers_only(discovered_cfg: Settings, remote_root: Path) -> None:
    job = _remote_job(remote_root)
    dest = _local_handle(discovered_cfg)
    fake = FakeRunner(Scenario(sacct=[SACCT_DONE]), remote_root=remote_root)
    result = run_stage("cluster.sync", discovered_cfg, pull, runner=fake)
    assert result.status == "partial", result.summary  # u2 has no marker: still pending
    assert result.summary == {
        "jobs": 1,
        "units_done": 1,
        "units_failed": 1,
        "units_pending": 1,
        "dest": str(Path(discovered_cfg.paths.runs_dir) / "slurm"),
    }
    manifest = read_manifest(result.manifest_path)
    report = manifest.extras["jobs"]["qe_r0"]
    assert report["job_ids"] == ["4243"] and report["all_terminal"] is True
    assert report["pending"] == ["u2"] and report["crashed_without_marker"] is True
    assert [r["state"] for r in report["sacct"]] == ["COMPLETED", "COMPLETED", "TIMEOUT"]
    assert manifest.slurm is not None
    assert manifest.slurm.job_ids == ["4243"] and manifest.slurm.units_done == 1
    assert manifest.slurm.units_failed == 1 and manifest.slurm.nodes == 2
    assert manifest.slurm.wall == "02:00:01" and manifest.slurm.partition == "compute"
    assert "sacct states never mark a unit done" in manifest.extras["rule"]
    assert (dest / "units" / "u1.done").is_file() and (dest / "logs" / "u1.log").is_file()
    assert (dest / "u1" / "pw.out").read_text().startswith("!")
    assert not (dest / "units" / "u2.done").exists()
    sync_json = json.loads((Path(result.manifest_path).parent / "sync.json").read_text())
    assert sync_json["qe_r0"]["units_done"] == 1
    assert [Path(a.path).name for a in manifest.outputs] == ["sync.json"]
    remote = fake.remote_commands()
    assert remote[0] == f"ls -1 {SCRATCH}/b20-mlip/jobs"
    assert remote[1] == "sacct -j 4243 -X -P -o JobID,State,ExitCode,Elapsed,NodeList"
    rsync = fake.rsyncs()[0]
    assert rsync[:3] == ["rsync", "-az", "--prune-empty-dirs"]
    assert (
        "--exclude=*.save" in rsync and "--exclude=build" in rsync and "--exclude=*.wfc*" in rsync
    )
    assert rsync[-2:] == [f"tillicum:{SCRATCH}/b20-mlip/jobs/qe_r0/", f"{dest}/"]

    # the job finishes on the cluster: the marker appears, a second sync is idempotent and ok
    (job / "units" / "u2.done").write_text('{"unit": "u2", "state": "done", "returncode": 0}\n')
    again = run_stage(
        "cluster.sync",
        discovered_cfg,
        pull,
        runner=FakeRunner(Scenario(sacct=[SACCT_DONE]), remote_root=remote_root),
        jobs=["qe_r0"],
    )
    assert again.status == "ok" and again.summary["units_pending"] == 0
    assert again.summary["units_done"] == 2 and again.summary["units_failed"] == 1


def test_explicit_jobs_and_missing_handle(discovered_cfg: Settings, remote_root: Path) -> None:
    _remote_job(remote_root, "phonons")
    fake = FakeRunner(Scenario(sacct=[SACCT_RUNNING]), remote_root=remote_root)
    result = run_stage("cluster.sync", discovered_cfg, pull, runner=fake, jobs=["phonons"])
    assert result.status == "partial"
    report = read_manifest(result.manifest_path).extras["jobs"]["phonons"]
    assert report["job_ids"] == [] and report["sacct"] == [] and report["all_terminal"] is False
    assert report["units_done"] == 1 and report["units_pending"] == 1
    assert not any(c.startswith("sacct") or c.startswith("ls -1") for c in fake.remote_commands())
    assert read_manifest(result.manifest_path).slurm is None


def test_sacct_failure_is_recorded_not_fatal(discovered_cfg: Settings, remote_root: Path) -> None:
    _remote_job(remote_root)
    _local_handle(discovered_cfg)

    class SacctFails(FakeRunner):
        def _respond(self, cmd, tup):  # type: ignore[no-untyped-def]
            if cmd.startswith("sacct"):
                from b20mlip.executors import CommandResult

                return CommandResult(tup, 1, "", "sacct: error: slurmdbd down")
            return super()._respond(cmd, tup)

    result = run_stage(
        "cluster.sync", discovered_cfg, pull, runner=SacctFails(Scenario(), remote_root=remote_root)
    )
    report = read_manifest(result.manifest_path).extras["jobs"]["qe_r0"]
    assert report["sacct_error"] == "sacct: error: slurmdbd down" and report["units_done"] == 1


def test_socket_death_is_never_job_completion(discovered_cfg: Settings, remote_root: Path) -> None:
    _remote_job(remote_root)
    _local_handle(discovered_cfg)
    ctx = RunContext("cluster.sync", discovered_cfg)
    with pytest.raises(ClusterUnreachable, match="ssh -fN tillicum"):
        pull(
            discovered_cfg,
            ctx,
            runner=FakeRunner(Scenario(socket_ok=False), remote_root=remote_root),
        )
    mid = FakeRunner(Scenario(die_after_ssh=1, sacct=[SACCT_DONE]), remote_root=remote_root)
    result = run_stage("cluster.sync", discovered_cfg, pull, runner=mid)
    assert result.status == "failed" and "ClusterUnreachable" in result.summary["error"]
    manifest = read_manifest(result.manifest_path)
    assert manifest.slurm is None and "jobs" not in manifest.extras
    assert not (Path(result.manifest_path).parent / "sync.json").exists()
    assert mid.rsyncs() == []  # died at sacct, before any pull


def test_sync_needs_scratch_and_dry_run(cluster_cfg: Settings, discovered_cfg: Settings) -> None:
    result = run_stage("cluster.sync", cluster_cfg, pull, runner=FakeRunner())
    assert result.status == "failed" and "cluster bootstrap" in result.summary["error"]
    fake = FakeRunner()
    result = run_stage("cluster.sync", discovered_cfg, pull, runner=fake, dry_run=True, jobs=["a"])
    assert result.status == "partial" and fake.calls == [] and result.summary == {"jobs": 1}


def test_helpers(tmp_path: Path) -> None:
    assert elapsed_seconds("02:00:01") == 7201 and elapsed_seconds("1-00:00:00") == 86400
    assert elapsed_seconds("03:10") == 190 and elapsed_seconds("junk") == 0
    assert job_ids_of(tmp_path) == []
    (tmp_path / "handle.json").write_text("{not json")
    assert job_ids_of(tmp_path) == []
    assert units_of(tmp_path) == []
    (tmp_path / "units").mkdir()
    (tmp_path / "units" / "b.failed").write_text("{}")
    (tmp_path / "units" / "a.done").write_text("{}")
    (tmp_path / "units" / "note.txt").write_text("")
    assert units_of(tmp_path) == ["a", "b"]
    (tmp_path / "units.txt").write_text("x\n\ny\n")
    assert units_of(tmp_path) == ["x", "y"]
