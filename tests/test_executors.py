"""LocalExecutor resume markers; SlurmExecutor rendering/submission/polling with a fake runner."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from b20mlip.config import Settings
from b20mlip.executors import (
    ClusterUnreachable,
    CommandResult,
    JobHandle,
    JobSpec,
    JobSubmitError,
    LocalExecutor,
    SlurmExecutor,
    get_executor,
    mark_unit,
    parse_sacct,
    unit_slug,
    unit_state,
    unit_states,
    write_job_files,
)

COUNTING_SCRIPT = """
echo "$B20_UNIT" >> "$B20_WORKDIR/calls.log"
mkdir -p "$B20_WORKDIR/out"
printf 'unit=%s index=%s omp=%s job=%s extra=%s\\n' "$B20_UNIT" "$B20_UNIT_INDEX" \\
  "$OMP_NUM_THREADS" "$B20_JOB_NAME" "${EXTRA:-}" > "$B20_WORKDIR/out/$B20_UNIT.txt"
if [ "$B20_UNIT" = "u2" ] && [ "${FIX:-0}" != "1" ]; then exit 3; fi
"""

GENERIC_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --account={{ account }}
#SBATCH --partition={{ partition }}
#SBATCH --time={{ resources.time | default('01:00:00') }}
#SBATCH --array=1-{{ n_units }}%{{ resources.max_parallel | default(8) }}
{% for m in cluster.modules %}module load {{ m }}
{% endfor %}
export OMP_NUM_THREADS={{ threads }}
{% for k, v in env.items() %}export {{ k }}={{ v }}
{% endfor %}
export B20_WORKDIR={{ workdir }} B20_JOB_NAME={{ job_name }}
export B20_UNIT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$B20_WORKDIR/{{ units_name }}")
bash "$B20_WORKDIR/{{ script_name }}"
"""


@pytest.fixture
def cluster_settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(
        update={
            "cluster": settings.cluster.model_copy(
                update={
                    "account": "acct",
                    "partition_cpu": "cpu-part",
                    "partition_gpu": "gpu-part",
                    "scratch": "/gscratch/b20",
                    "modules": ["gcc", "quantum-espresso"],
                    "control_path": str(tmp_path / "cm.sock"),
                }
            )
        }
    )


@pytest.fixture
def template_dir(tmp_path: Path) -> Path:
    d = tmp_path / "templates"
    d.mkdir()
    (d / "generic.sbatch.j2").write_text(GENERIC_TEMPLATE)
    return d


# --- markers -----------------------------------------------------------------------------------


def test_unit_slug_and_markers(tmp_path: Path) -> None:
    assert unit_slug("MnSi/strain/0001") == "MnSi_strain_0001"
    assert unit_slug("a b") == "a_b" and unit_slug("///") == "unit"
    assert unit_state(tmp_path, "u1") == "pending"
    p = mark_unit(tmp_path, "u1", "failed", {"returncode": 3})
    assert p.name == "u1.failed" and unit_state(tmp_path, "u1") == "failed"
    mark_unit(tmp_path, "u1", "done", {"returncode": 0})
    assert unit_state(tmp_path, "u1") == "done" and not p.exists()
    assert unit_states(tmp_path, ["u1", "u2"]) == {"u1": "done", "u2": "pending"}
    spec = JobSpec(name="j", script="true", units=["a", "b"])
    script, units = write_job_files(tmp_path, spec)
    assert script.read_text().startswith("#!/usr/bin/env bash") and units.read_text() == "a\nb\n"


# --- local -------------------------------------------------------------------------------------


def test_local_executor_runs_units_and_resumes(settings: Settings, tmp_path: Path) -> None:
    ex = LocalExecutor(settings)
    spec = JobSpec(
        name="qe_r0", script=COUNTING_SCRIPT, units=["u1", "u2", "u3"], env={"EXTRA": "x"}
    )
    handle = ex.submit(spec)
    workdir = Path(handle.workdir)
    assert workdir == settings.paths.runs_dir / "local" / "qe_r0" and handle.job_ids[0].startswith(
        "local:qe_r0:"
    )
    assert unit_states(workdir, spec.units) == {"u1": "done", "u2": "failed", "u3": "done"}
    assert ex.summary(handle) == {"units_done": 2, "units_failed": 1}
    assert ex.wait(handle) is None
    assert (workdir / "calls.log").read_text().split() == ["u1", "u2", "u3"]
    assert (workdir / "out" / "u3.txt").read_text() == "unit=u3 index=3 omp=6 job=qe_r0 extra=x\n"
    assert (workdir / "logs" / "u2.log").is_file()

    # resubmit: done units are skipped, the failed one is retried (and now succeeds)
    fixed = spec.model_copy(update={"env": {"EXTRA": "x", "FIX": "1"}})
    ex.submit(fixed)
    assert (workdir / "calls.log").read_text().split() == ["u1", "u2", "u3", "u2"]
    assert ex.summary(handle) == {"units_done": 3, "units_failed": 0}
    ex.submit(fixed)  # nothing left to do
    assert (workdir / "calls.log").read_text().split() == ["u1", "u2", "u3", "u2"]

    dest = tmp_path / "fetched"
    arts = ex.fetch(handle, dest)
    assert {Path(a.path).name for a in arts} >= {
        "calls.log",
        "u1.txt",
        "u1.done",
        "units.txt",
        "script.sh",
    }
    assert all(Path(a.path).is_relative_to(dest) for a in arts)
    assert ex.fetch(handle, workdir)  # same dir: no copy, still listed


def test_local_executor_parallel_and_errors(settings: Settings) -> None:
    ex = LocalExecutor(settings, workdir_root=settings.paths.runs_dir / "custom")
    spec = JobSpec(
        name="par", script=COUNTING_SCRIPT, units=[f"p{i}" for i in range(6)],
        resources={"parallel": 3}, env={"FIX": "1"},
    )  # fmt: skip
    handle = ex.submit(spec)
    assert ex.summary(handle) == {"units_done": 6, "units_failed": 0}
    assert sorted((Path(handle.workdir) / "calls.log").read_text().split()) == sorted(spec.units)
    with pytest.raises(JobSubmitError, match="no units"):
        ex.submit(JobSpec(name="empty", script="true", units=[]))
    assert isinstance(get_executor(settings, "local"), LocalExecutor)
    assert isinstance(get_executor(settings, "slurm"), SlurmExecutor)
    with pytest.raises(ValueError, match="unknown executor"):
        get_executor(settings, "cloud")


# --- slurm --------------------------------------------------------------------------------------


class FakeRunner:
    """Records every argv; answers ssh/rsync/sacct without touching a socket."""

    def __init__(self, sacct: list[str] | None = None, socket_ok: bool = True, ssh_rc: int = 0):
        self.calls: list[list[str]] = []
        self.sacct: Iterator[str] = iter(sacct or [])
        self.socket_ok = socket_ok
        self.ssh_rc = ssh_rc

    def __call__(self, argv: list[str]) -> CommandResult:
        self.calls.append(list(argv))
        tup = tuple(argv)
        if argv[0] == "ssh" and "-O" in argv:
            return CommandResult(
                tup, 0 if self.socket_ok else 255, "", "" if self.socket_ok else "no socket"
            )
        if argv[0] == "rsync":
            return CommandResult(tup, self.ssh_rc, "", "")
        if argv[0] == "ssh":
            if self.ssh_rc:
                return CommandResult(tup, self.ssh_rc, "", "Connection closed")
            remote = argv[-1]
            if "sbatch" in remote:
                return CommandResult(tup, 0, "12345;cluster\n", "")
            if "sacct" in remote:
                return CommandResult(tup, 0, next(self.sacct, ""), "")
            return CommandResult(tup, 0, "", "")
        raise AssertionError(f"unexpected command {argv}")

    def remote_commands(self) -> list[str]:
        return [c[-1] for c in self.calls if c[0] == "ssh" and "-O" not in c]


def _spec() -> JobSpec:
    return JobSpec(
        name="qe_r0", script="pw.x -in $B20_UNIT/pw.in > $B20_UNIT/pw.out",
        units=["MnSi/strain/0001", "MnSi/strain/0002", "FeSi/eos/0001"],
        resources={"time": "02:00:00", "max_parallel": 2}, env={"ESPRESSO_TMPDIR": "/tmp"},
    )  # fmt: skip


def test_slurm_render(cluster_settings: Settings, template_dir: Path) -> None:
    ex = SlurmExecutor(cluster_settings, runner=FakeRunner(), template_dir=template_dir)
    text = ex.render(_spec())
    assert "#SBATCH --job-name=qe_r0" in text
    assert "#SBATCH --account=acct" in text and "#SBATCH --partition=cpu-part" in text
    assert "#SBATCH --time=02:00:00" in text and "#SBATCH --array=1-3%2" in text
    assert "module load gcc\nmodule load quantum-espresso\n" in text
    assert "export OMP_NUM_THREADS=6" in text and "export ESPRESSO_TMPDIR=/tmp" in text
    assert "B20_WORKDIR=/gscratch/b20/b20-mlip/jobs/qe_r0" in text
    assert 'sed -n "${SLURM_ARRAY_TASK_ID}p" "$B20_WORKDIR/units.txt"' in text
    gpu = _spec().model_copy(update={"resources": {"partition": "gpu-part"}})
    assert "#SBATCH --partition=gpu-part" in ex.render(gpu)
    with pytest.raises(FileNotFoundError, match="no SLURM template"):
        ex.render(_spec().model_copy(update={"resources": {"template": "missing"}}))


def test_slurm_submit_wait_fetch(
    cluster_settings: Settings, template_dir: Path, tmp_path: Path
) -> None:
    sacct = [
        "12345_1|RUNNING|0:0|acct|cpu-part|1|00:01:00\n12345_[2-3]|PENDING|0:0|acct|cpu-part|1|00:00:00\n",
        "12345_1|COMPLETED|0:0|acct|cpu-part|1|00:03:10\n"
        "12345_2|COMPLETED|0:0|acct|cpu-part|1|00:02:59\n"
        "12345_3|TIMEOUT|0:1|acct|cpu-part|1|02:00:01\n"
        "12345_3.batch|TIMEOUT|0:1|acct||1|02:00:01\n",
    ]
    runner = FakeRunner(sacct=sacct)
    sleeps: list[float] = []
    ex = SlurmExecutor(
        cluster_settings, runner=runner, template_dir=template_dir,
        staging_root=tmp_path / "staging", sleep=sleeps.append,
    )  # fmt: skip
    handle = ex.submit(_spec())
    assert handle == JobHandle(job_ids=["12345"], workdir="/gscratch/b20/b20-mlip/jobs/qe_r0")
    staging = tmp_path / "staging" / "qe_r0"
    assert {p.name for p in staging.iterdir()} >= {
        "job.sbatch",
        "units.txt",
        "script.sh",
        "spec.json",
        "handle.json",
        "units",
        "logs",
    }
    assert (staging / "units.txt").read_text().splitlines() == _spec().units

    cp = str(tmp_path / "cm.sock")
    assert runner.calls[0] == ["ssh", "-S", cp, "-O", "check", "tillicum"]
    assert runner.remote_commands()[0] == "mkdir -p /gscratch/b20/b20-mlip/jobs/qe_r0"
    rsync = next(c for c in runner.calls if c[0] == "rsync")
    assert rsync == [
        "rsync",
        "-az",
        "-e",
        f"ssh -S {cp} -o BatchMode=yes",
        f"{staging}/",
        "tillicum:/gscratch/b20/b20-mlip/jobs/qe_r0/",
    ]
    assert (
        runner.remote_commands()[1]
        == "cd /gscratch/b20/b20-mlip/jobs/qe_r0 && sbatch --parsable job.sbatch"
    )
    assert all(
        "-o" in c and "BatchMode=yes" in c for c in runner.calls if c[0] == "ssh" and "-O" not in c
    )

    info = ex.wait(handle, poll_s=5)
    assert sleeps == [5]
    assert info.job_ids == ["12345"] and info.account == "acct" and info.partition == "cpu-part"
    assert (info.units_done, info.units_failed, info.nodes, info.wall) == (2, 1, 1, "02:00:01")
    assert "sacct --jobs 12345 --allocations --parsable2 --noheader" in runner.remote_commands()[2]

    dest = tmp_path / "results"
    dest.mkdir()
    (dest / "pw.out").write_text("JOB DONE")  # what rsync would have pulled
    arts = ex.fetch(handle, dest)
    assert runner.calls[-1] == [
        "rsync",
        "-az",
        "-e",
        f"ssh -S {cp} -o BatchMode=yes",
        "tillicum:/gscratch/b20/b20-mlip/jobs/qe_r0/",
        f"{dest}/",
    ]
    assert [Path(a.path).name for a in arts] == ["pw.out"] and arts[0].kind == "log"


def test_slurm_refuses_without_socket_or_scratch(
    cluster_settings: Settings, template_dir: Path, tmp_path: Path
) -> None:
    dead = FakeRunner(socket_ok=False)
    ex = SlurmExecutor(
        cluster_settings, runner=dead, template_dir=template_dir, staging_root=tmp_path / "s"
    )
    with pytest.raises(ClusterUnreachable, match="ssh -fN tillicum"):
        ex.submit(_spec())
    assert [c[0] for c in dead.calls] == ["ssh"]  # nothing rsynced or submitted
    assert ex.check_socket() is False

    no_scratch = cluster_settings.model_copy(
        update={"cluster": cluster_settings.cluster.model_copy(update={"scratch": None})}
    )
    ex2 = SlurmExecutor(no_scratch, runner=FakeRunner(), template_dir=template_dir)
    with pytest.raises(JobSubmitError, match="cluster bootstrap"):
        ex2.submit(_spec())
    with pytest.raises(JobSubmitError, match="no units"):
        ex.submit(_spec().model_copy(update={"units": []}))


def test_slurm_socket_death_is_not_job_completion(
    cluster_settings: Settings, template_dir: Path
) -> None:
    runner = FakeRunner(ssh_rc=255)
    ex = SlurmExecutor(
        cluster_settings, runner=runner, template_dir=template_dir, sleep=lambda s: None
    )
    handle = JobHandle(job_ids=["777"], workdir="/gscratch/b20/b20-mlip/jobs/x")
    with pytest.raises(ClusterUnreachable, match="socket"):
        ex.wait(handle, poll_s=1)
    with pytest.raises(ClusterUnreachable):
        ex.fetch(handle, template_dir)


def test_slurm_wait_timeout_and_sacct_error(cluster_settings: Settings, template_dir: Path) -> None:
    runner = FakeRunner(sacct=["777|RUNNING|0:0|acct|cpu-part|1|00:00:01\n"] * 5)
    ex = SlurmExecutor(
        cluster_settings, runner=runner, template_dir=template_dir, sleep=lambda s: None
    )
    handle = JobHandle(job_ids=["777"], workdir="/x")
    with pytest.raises(TimeoutError):
        ex.wait(handle, poll_s=0, timeout_s=0)

    class SacctFails(FakeRunner):
        def __call__(self, argv: list[str]) -> CommandResult:
            res = super().__call__(argv)
            return CommandResult(res.argv, 1, "", "sacct: error") if "sacct" in argv[-1] else res

    ex2 = SlurmExecutor(cluster_settings, runner=SacctFails(), template_dir=template_dir)
    with pytest.raises(RuntimeError, match="sacct failed"):
        ex2.job_rows(handle)


def test_parse_sacct() -> None:
    rows = parse_sacct(
        "1_1|COMPLETED|0:0|a|p|2|00:10:00\n"
        "1_1.batch|COMPLETED|0:0|a||2|00:10:00\n"
        "1_2|CANCELLED by 42|0:0|a|p|2|00:00:05\n"
        "1_[3-4]|PENDING|0:0|a|p|1|00:00:00\n"
        "garbage\n"
        "\n"
        "1_5|RUNNING|0:0|a|p|notanumber|00:00:01\n"
    )
    assert [r.job_id for r in rows] == ["1_1", "1_2", "1_[3-4]", "1_5"]
    assert [r.state for r in rows] == ["COMPLETED", "CANCELLED", "PENDING", "RUNNING"]
    assert [r.terminal for r in rows] == [True, True, False, False]
    assert rows[0].nnodes == 2 and rows[3].nnodes == 0
    assert parse_sacct("") == []


def test_slurm_push_and_pull_dir_go_through_the_runner(tmp_path: Path) -> None:
    """push_dir makes the remote parent and rsyncs local -> alias:remote; pull_dir excludes QE
    scratch (tmp/, *.wfc*, *.save/) and mirrors alias:remote -> local."""
    from b20mlip.config import Settings
    from b20mlip.executors import SlurmExecutor

    calls: list[list[str]] = []

    def runner(argv):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        from b20mlip.executors import CommandResult

        return CommandResult(tuple(argv), 0, "", "")

    cfg = Settings().model_copy(
        update={
            "cluster": Settings().cluster.model_copy(
                update={"scratch": "/scratch/u", "control_path": str(tmp_path / "cm.sock")}
            )
        }
    )
    ex = SlurmExecutor(cfg, runner=runner)
    local = tmp_path / "units"
    local.mkdir()
    ex.push_dir(local, "/scratch/u/b20-mlip/dft/r0")
    ex.pull_dir("/scratch/u/b20-mlip/dft/r0", local)
    ssh_calls = [c for c in calls if c[0] == "ssh"]
    rsync_calls = [c for c in calls if c[0] == "rsync"]
    assert (
        ssh_calls
        and "mkdir -p /scratch/u/b20-mlip/dft/r0 /scratch/u/b20-mlip/dft" in ssh_calls[0][-1]
    )
    assert rsync_calls[0][-2:] == [f"{local.resolve()}/", "tillicum:/scratch/u/b20-mlip/dft/r0/"]
    pull = rsync_calls[1]
    assert pull[-2:] == ["tillicum:/scratch/u/b20-mlip/dft/r0/", f"{local}/"]
    assert "tmp/" in pull and "*.wfc*" in pull and "*.save/" in pull


def test_render_context_applies_cluster_resource_defaults(tmp_path: Path) -> None:
    """cluster.resources[<template>] (the site overlay) fills in what a JobSpec leaves unset."""
    from b20mlip.config import Settings
    from b20mlip.executors import JobSpec, SlurmExecutor

    cfg = Settings().model_copy(
        update={
            "cluster": Settings().cluster.model_copy(
                update={
                    "scratch": "/scratch/u",
                    "control_path": str(tmp_path / "cm.sock"),
                    "resources": {"train_replay": {"time": "01:30:00", "mem": "48G", "gpus": 1}},
                }
            )
        }
    )
    ex = SlurmExecutor(cfg, runner=lambda argv: None)  # type: ignore[arg-type]
    spec = JobSpec(
        name="j", script="x", units=["u"], resources={"template": "train_replay", "mem": "96G"}
    )
    res = ex.render_context(spec)["resources"]
    assert res["time"] == "01:30:00" and res["gpus"] == 1 and res["mem"] == "96G"
    plain = JobSpec(name="j", script="x", units=["u"], resources={"template": "qe_array"})
    assert "time" not in ex.render_context(plain)["resources"]
