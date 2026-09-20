"""Executors (CONTRACTS.md section 6): ``LocalExecutor`` and ``SlurmExecutor``.

Unit protocol (shared by both executors and by every ``templates/slurm/*.sbatch.j2``):

* a job is a ``JobSpec`` whose ``script`` is a bash snippet run once per unit with the
  environment ``B20_UNIT`` (the unit id), ``B20_UNIT_INDEX`` (1-based), ``B20_WORKDIR``,
  ``B20_JOB_NAME`` and ``OMP_NUM_THREADS`` (pinned BLAS threads) plus ``JobSpec.env``;
* the job directory holds ``units.txt`` (one unit per line), ``script.sh``, ``logs/<unit>.log``
  and the resume markers ``units/<unit>.done`` / ``units/<unit>.failed`` (JSON payloads);
* a unit with a ``.done`` marker is skipped on resubmission; a ``.failed`` marker is cleared
  and the unit retried. No ``srun`` is used for single-task units.

The SLURM executor never calls ``ssh``/``sacct``/``rsync`` directly: every command goes through
an injectable ``runner`` so the offline test suite exercises rendering, submission parsing and
polling with a fake. A dead ControlMaster socket raises :class:`ClusterUnreachable`; it is
never mistaken for a finished job.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import jinja2

from b20mlip.config import Settings, repo_root
from b20mlip.models import Artifact, B20Model, SlurmInfo
from b20mlip.provenance import artifact_for

UnitState = Literal["done", "failed", "pending"]
SCRIPT_NAME = "script.sh"
UNITS_NAME = "units.txt"
SBATCH_NAME = "job.sbatch"
DEFAULT_TEMPLATE = "generic"
TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
    }
)
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SCRIPT_HEADER = (
    "#!/usr/bin/env bash\n"
    "# Rendered by b20mlip.executors. Reads B20_UNIT, B20_UNIT_INDEX, B20_WORKDIR, B20_JOB_NAME.\n"
    "set -euo pipefail\n"
)


# --- specs ------------------------------------------------------------------------------------


class JobSpec(B20Model):
    name: str
    script: str
    units: list[str]
    resources: dict[str, Any] = {}
    env: dict[str, str] = {}


class JobHandle(B20Model):
    job_ids: list[str]
    workdir: str


class Executor(Protocol):
    def submit(self, spec: JobSpec) -> JobHandle: ...

    def wait(self, handle: JobHandle, poll_s: float = 60) -> SlurmInfo | None: ...

    def fetch(self, handle: JobHandle, dest: Path) -> list[Artifact]: ...


class ClusterUnreachable(RuntimeError):
    """The ssh transport failed (ControlMaster socket dead, MFA expired). Not a job outcome."""


class JobSubmitError(RuntimeError):
    """sbatch (or the local runner) refused the job."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


Runner = Callable[[list[str]], CommandResult]


def subprocess_runner(argv: list[str]) -> CommandResult:
    """Default runner: run ``argv`` locally, capture output, never raise on non-zero exit."""
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    return CommandResult(tuple(argv), proc.returncode, proc.stdout, proc.stderr)


# --- unit markers -----------------------------------------------------------------------------


def unit_slug(unit: str) -> str:
    """Filesystem-safe marker name for a unit id."""
    return _SLUG_RE.sub("_", unit).strip("_") or "unit"


def marker_path(workdir: str | Path, unit: str, state: UnitState) -> Path:
    return Path(workdir) / "units" / f"{unit_slug(unit)}.{state}"


def unit_state(workdir: str | Path, unit: str) -> UnitState:
    if marker_path(workdir, unit, "done").is_file():
        return "done"
    if marker_path(workdir, unit, "failed").is_file():
        return "failed"
    return "pending"


def unit_states(workdir: str | Path, units: list[str]) -> dict[str, UnitState]:
    return {unit: unit_state(workdir, unit) for unit in units}


def mark_unit(
    workdir: str | Path, unit: str, state: Literal["done", "failed"], payload: dict[str, Any]
) -> Path:
    """Write ``units/<unit>.<state>`` (JSON) and remove the opposite marker."""
    other: Literal["done", "failed"] = "failed" if state == "done" else "done"
    marker_path(workdir, unit, other).unlink(missing_ok=True)
    path = marker_path(workdir, unit, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"unit": unit, "state": state, "at": datetime.now(UTC).isoformat(), **payload}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_job_files(workdir: str | Path, spec: JobSpec) -> tuple[Path, Path]:
    """Write ``script.sh`` and ``units.txt`` into ``workdir``; returns their paths."""
    wd = Path(workdir)
    (wd / "units").mkdir(parents=True, exist_ok=True)
    (wd / "logs").mkdir(parents=True, exist_ok=True)
    script = wd / SCRIPT_NAME
    script.write_text(_SCRIPT_HEADER + spec.script.rstrip("\n") + "\n", encoding="utf-8")
    script.chmod(0o755)
    units = wd / UNITS_NAME
    units.write_text("".join(f"{u}\n" for u in spec.units), encoding="utf-8")
    return script, units


def artifacts_under(root: str | Path) -> list[Artifact]:
    """Hash every regular file below ``root`` (sorted, relative order stable)."""
    base = Path(root)
    return [artifact_for(p) for p in sorted(base.rglob("*")) if p.is_file()]


# --- local -------------------------------------------------------------------------------------


class LocalExecutor:
    """Run every unit as a bash subprocess in ``<runs_dir>/local/<job name>/``.

    ``resources["parallel"]`` (default 1) units run concurrently; ``OMP_NUM_THREADS`` defaults to
    ``cfg.compute.threads``. ``submit`` blocks until the units finish, so ``wait`` returns
    ``None`` immediately (local runs carry no SLURM accounting).
    """

    def __init__(self, cfg: Settings, *, workdir_root: str | Path | None = None) -> None:
        self.cfg = cfg
        self.root = (
            Path(workdir_root) if workdir_root is not None else Path(cfg.paths.runs_dir) / "local"
        )

    def workdir(self, name: str) -> Path:
        return self.root / name

    def submit(self, spec: JobSpec) -> JobHandle:
        if not spec.units:
            raise JobSubmitError(f"job {spec.name!r} has no units")
        workdir = self.workdir(spec.name)
        script, _ = write_job_files(workdir, spec)
        todo = [
            (i, u) for i, u in enumerate(spec.units, start=1) if unit_state(workdir, u) != "done"
        ]
        parallel = max(1, int(spec.resources.get("parallel", 1)))

        def run_unit(item: tuple[int, str]) -> None:
            index, unit = item
            marker_path(workdir, unit, "failed").unlink(missing_ok=True)
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", str(self.cfg.compute.threads))
            env.update(spec.env)
            env.update(
                B20_UNIT=unit,
                B20_UNIT_INDEX=str(index),
                B20_WORKDIR=str(workdir),
                B20_JOB_NAME=spec.name,
            )
            log = workdir / "logs" / f"{unit_slug(unit)}.log"
            t0 = time.perf_counter()
            with open(log, "a", encoding="utf-8") as fh:
                proc = subprocess.run(
                    ["bash", str(script)], cwd=workdir, env=env, stdout=fh, stderr=subprocess.STDOUT
                )
            payload = {
                "returncode": proc.returncode,
                "wall_seconds": round(time.perf_counter() - t0, 3),
                "log": str(log),
            }
            mark_unit(workdir, unit, "done" if proc.returncode == 0 else "failed", payload)

        if parallel == 1:
            for item in todo:
                run_unit(item)
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                list(pool.map(run_unit, todo))
        return JobHandle(job_ids=[f"local:{spec.name}:{os.getpid()}"], workdir=str(workdir))

    def wait(self, handle: JobHandle, poll_s: float = 60) -> SlurmInfo | None:
        return None

    def summary(self, handle: JobHandle) -> dict[str, int]:
        units_dir = Path(handle.workdir) / "units"
        done = len(list(units_dir.glob("*.done"))) if units_dir.is_dir() else 0
        failed = len(list(units_dir.glob("*.failed"))) if units_dir.is_dir() else 0
        return {"units_done": done, "units_failed": failed}

    def fetch(self, handle: JobHandle, dest: Path) -> list[Artifact]:
        src = Path(handle.workdir)
        dest = Path(dest)
        if src.resolve() != dest.resolve():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        return artifacts_under(dest)


# --- slurm -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SacctRow:
    job_id: str
    state: str
    exit_code: str
    account: str
    partition: str
    nnodes: int
    elapsed: str

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def parse_sacct(text: str) -> list[SacctRow]:
    """Parse ``sacct --parsable2 --noheader --allocations`` rows.

    Fields: ``JobID|State|ExitCode|Account|Partition|NNodes|Elapsed``. Job steps (``123.batch``)
    are ignored; ``"CANCELLED by 42"`` normalises to ``CANCELLED``.
    """
    rows: list[SacctRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("|")
        if len(parts) < 3 or "." in parts[0]:
            continue
        parts += [""] * (7 - len(parts))
        state = parts[1].strip().split()[0].upper() if parts[1].strip() else "UNKNOWN"
        try:
            nnodes = int(parts[5]) if parts[5].strip() else 0
        except ValueError:
            nnodes = 0
        rows.append(
            SacctRow(parts[0].strip(), state, parts[2], parts[3], parts[4], nnodes, parts[6])
        )
    return rows


class SlurmExecutor:
    """Render ``templates/slurm/<template>.sbatch.j2``, stage it, submit over ssh, poll sacct.

    All remote access goes through the ControlMaster socket ``cfg.cluster.control_path`` for
    the alias ``cfg.cluster.alias`` (the user opens it with an interactive MFA login). The
    template is chosen by ``spec.resources["template"]`` (default ``"generic"``).
    """

    def __init__(
        self,
        cfg: Settings,
        *,
        runner: Runner | None = None,
        template_dir: str | Path | None = None,
        staging_root: str | Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.cluster = cfg.cluster
        self.runner: Runner = runner or subprocess_runner
        self.template_dir = (
            Path(template_dir) if template_dir is not None else repo_root() / "templates" / "slurm"
        )
        self.staging_root = (
            Path(staging_root) if staging_root is not None else Path(cfg.paths.runs_dir) / "slurm"
        )
        self._sleep = sleep
        self.jinja = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,  # shell scripts, not HTML
        )

    # -- transport ---------------------------------------------------------------------------

    @property
    def control_path(self) -> str:
        return os.path.expanduser(self.cluster.control_path)

    @property
    def ssh_transport(self) -> str:
        return f"ssh -S {shlex.quote(self.control_path)} -o BatchMode=yes"

    def ssh_argv(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            "-S",
            self.control_path,
            "-o",
            "BatchMode=yes",
            self.cluster.alias,
            remote_command,
        ]

    def check_socket(self) -> bool:
        res = self.runner(["ssh", "-S", self.control_path, "-O", "check", self.cluster.alias])
        return res.returncode == 0

    def _ssh(self, remote_command: str) -> CommandResult:
        res = self.runner(self.ssh_argv(remote_command))
        if res.returncode == 255:
            raise ClusterUnreachable(
                f"ssh to {self.cluster.alias!r} failed (socket {self.control_path} dead?): "
                f"{res.stderr.strip()}"
            )
        return res

    def _rsync(self, src: str, dst: str) -> CommandResult:
        res = self.runner(["rsync", "-az", "-e", self.ssh_transport, src, dst])
        if res.returncode == 255:
            raise ClusterUnreachable(f"rsync transport to {self.cluster.alias!r} failed")
        if not res.ok:
            raise JobSubmitError(f"rsync {src} -> {dst} failed: {res.stderr.strip()}")
        return res

    # -- rendering ---------------------------------------------------------------------------

    def remote_workdir(self, name: str) -> str:
        if not self.cluster.scratch:
            raise JobSubmitError(
                "cluster.scratch is not set; run `b20mlip cluster bootstrap` "
                "(writes configs/cluster/tillicum.yaml) first"
            )
        return f"{self.cluster.scratch.rstrip('/')}/b20-mlip/jobs/{name}"

    def template_name(self, spec: JobSpec) -> str:
        return f"{spec.resources.get('template', DEFAULT_TEMPLATE)}.sbatch.j2"

    def effective_resources(self, spec: JobSpec) -> dict[str, Any]:
        """``cluster.resources[<template>]`` site defaults (the overlay) under the spec's own."""
        template = str(spec.resources.get("template", DEFAULT_TEMPLATE))
        site = dict(self.cluster.resources.get(template, {}))
        return {**site, **dict(spec.resources)}

    def render_context(self, spec: JobSpec) -> dict[str, Any]:
        resources = self.effective_resources(spec)
        return {
            "job_name": spec.name,
            "script": spec.script,
            "units": list(spec.units),
            "n_units": len(spec.units),
            "resources": resources,
            "env": dict(spec.env),
            "cluster": self.cluster.model_dump(mode="json"),
            "account": resources.get("account", self.cluster.account),
            "partition": resources.get("partition", self.cluster.partition_cpu),
            "threads": self.cfg.compute.threads,
            "workdir": self.remote_workdir(spec.name),
            "script_name": SCRIPT_NAME,
            "units_name": UNITS_NAME,
        }

    def render(self, spec: JobSpec) -> str:
        try:
            template = self.jinja.get_template(self.template_name(spec))
        except jinja2.TemplateNotFound as exc:
            raise FileNotFoundError(
                f"no SLURM template {self.template_name(spec)!r} in {self.template_dir}"
            ) from exc
        return template.render(**self.render_context(spec))

    # -- executor protocol -------------------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobHandle:
        if not spec.units:
            raise JobSubmitError(f"job {spec.name!r} has no units")
        remote = self.remote_workdir(spec.name)
        staging = self.staging_root / spec.name
        staging.mkdir(parents=True, exist_ok=True)
        (staging / SBATCH_NAME).write_text(self.render(spec), encoding="utf-8")
        write_job_files(staging, spec)
        (staging / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")

        if not self.check_socket():
            raise ClusterUnreachable(
                f"no live ControlMaster socket at {self.control_path}; "
                f"run `ssh -fN {self.cluster.alias}` (MFA) first"
            )
        self._ssh(f"mkdir -p {shlex.quote(remote)}")
        self._rsync(f"{staging}/", f"{self.cluster.alias}:{remote}/")
        res = self._ssh(f"cd {shlex.quote(remote)} && sbatch --parsable {SBATCH_NAME}")
        if not res.ok or not res.stdout.strip():
            raise JobSubmitError(f"sbatch failed for {spec.name!r}: {res.stderr.strip()}")
        job_id = res.stdout.strip().splitlines()[-1].split(";")[0].strip()
        handle = JobHandle(job_ids=[job_id], workdir=remote)
        (staging / "handle.json").write_text(
            handle.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return handle

    def job_rows(self, handle: JobHandle) -> list[SacctRow]:
        ids = ",".join(handle.job_ids)
        res = self._ssh(
            f"sacct --jobs {shlex.quote(ids)} --allocations --parsable2 --noheader "
            "--format=JobID,State,ExitCode,Account,Partition,NNodes,Elapsed"
        )
        if not res.ok:
            raise RuntimeError(f"sacct failed: {res.stderr.strip()}")
        return parse_sacct(res.stdout)

    def wait(
        self, handle: JobHandle, poll_s: float = 60, *, timeout_s: float | None = None
    ) -> SlurmInfo:
        started = time.monotonic()
        while True:
            rows = self.job_rows(handle)
            if rows and all(row.terminal for row in rows):
                break
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                raise TimeoutError(f"jobs {handle.job_ids} still running after {timeout_s} s")
            self._sleep(poll_s)
        done = sum(row.state == "COMPLETED" for row in rows)
        return SlurmInfo(
            job_ids=list(handle.job_ids),
            account=next((r.account for r in rows if r.account), self.cluster.account or ""),
            partition=next((r.partition for r in rows if r.partition), ""),
            nodes=max((r.nnodes for r in rows), default=0),
            wall=max((r.elapsed for r in rows), default=""),
            units_done=done,
            units_failed=len(rows) - done,
        )

    def fetch(self, handle: JobHandle, dest: Path) -> list[Artifact]:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._rsync(f"{self.cluster.alias}:{handle.workdir}/", f"{dest}/")
        return artifacts_under(dest)

    # -- directory mirroring (unit roots live outside the job dir) ---------------------------

    def push_dir(self, local_dir: str | Path, remote_dir: str) -> None:
        """Mirror ``local_dir`` to ``alias:remote_dir`` (parents created; extra remote files
        kept, so markers/outputs already on the cluster survive a re-push)."""
        parent = posixpath.dirname(remote_dir.rstrip("/"))
        self._ssh(f"mkdir -p {shlex.quote(remote_dir)} {shlex.quote(parent)}")
        self._rsync(
            f"{Path(local_dir).resolve()}/", f"{self.cluster.alias}:{remote_dir.rstrip('/')}/"
        )

    def pull_dir(self, remote_dir: str, local_dir: str | Path) -> None:
        """Mirror ``alias:remote_dir`` back into ``local_dir`` (QE scratch ``tmp/`` excluded)."""
        dest = Path(local_dir)
        dest.mkdir(parents=True, exist_ok=True)
        res = self.runner(
            [
                "rsync",
                "-az",
                "--exclude",
                "tmp/",
                "--exclude",
                "*.wfc*",
                "--exclude",
                "*.save/",
                "-e",
                self.ssh_transport,
                f"{self.cluster.alias}:{remote_dir.rstrip('/')}/",
                f"{dest}/",
            ]
        )
        if res.returncode == 255:
            raise ClusterUnreachable(f"rsync transport to {self.cluster.alias!r} failed")
        if not res.ok:
            raise JobSubmitError(f"rsync {remote_dir} -> {dest} failed: {res.stderr.strip()}")


def get_executor(cfg: Settings, kind: str = "local") -> Executor:
    if kind == "local":
        return LocalExecutor(cfg)
    if kind == "slurm":
        return SlurmExecutor(cfg)
    raise ValueError(f"unknown executor {kind!r} (expected 'local' or 'slurm')")


__all__ = [
    "ClusterUnreachable",
    "CommandResult",
    "DEFAULT_TEMPLATE",
    "Executor",
    "JobHandle",
    "JobSpec",
    "JobSubmitError",
    "LocalExecutor",
    "Runner",
    "SacctRow",
    "SlurmExecutor",
    "TERMINAL_STATES",
    "UnitState",
    "artifacts_under",
    "get_executor",
    "mark_unit",
    "marker_path",
    "parse_sacct",
    "subprocess_runner",
    "unit_slug",
    "unit_state",
    "unit_states",
    "write_job_files",
]
