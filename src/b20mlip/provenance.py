"""Provenance (CONTRACTS.md section 5): hashing, ``Manifest`` I/O, ``RunContext``, ``run_stage``.

Every stage writes ``runs/<stage>/<run_id>/manifest.json`` with
``run_id = f"{utc:%Y%m%dT%H%M%S}-{config_sha256[:6]}-{seed}"`` (``seed`` is ``none`` when the
stage takes no seed). The manifest is written even when the stage fails (``status="failed"``)
and on ``--dry-run`` (``status="partial"``, no outputs). ``docs/manifest.schema.json`` is the
JSON Schema (draft 2020-12) of :class:`~b20mlip.models.Manifest`.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from b20mlip.config import Settings, repo_root, sha256_of_json
from b20mlip.models import (
    Artifact,
    ArtifactKind,
    Frame,
    Manifest,
    SlurmInfo,
    StageName,
    StageResult,
    Status,
)

if TYPE_CHECKING:  # executors.py is a later build row; typing-only import avoids a cycle
    from b20mlip.executors import Executor

MANIFEST_NAME = "manifest.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}-[0-9a-f]{6}-(-?\d+|none)$")
TRACKED_PACKAGES: tuple[str, ...] = (
    "b20-mlip",
    "mace-torch",
    "torch",
    "e3nn",
    "ase",
    "phonopy",
    "pymatgen",
    "spglib",
    "seekpath",
    "pymbar",
    "numpy",
    "pydantic",
    "pydantic-settings",
    "anthropic",
    "pyscf",
)
_KIND_BY_SUFFIX: dict[str, ArtifactKind] = {
    ".extxyz": "frames",
    ".xyz": "frames",
    ".model": "model",
    ".pt": "model",
    ".pth": "model",
    ".json": "json",
    ".jsonl": "json",
    ".log": "log",
    ".out": "log",
    ".txt": "log",
    ".traj": "traj",
    ".yaml": "yaml",
    ".yml": "yaml",
}

# --- hashing ---------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frames(frames: Iterable[Frame]) -> str:
    """Order-dependent SHA-256 over the canonical JSON of each frame (dataset identity)."""
    h = hashlib.sha256()
    for frame in frames:
        payload = json.dumps(
            frame.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), default=str
        )
        h.update(payload.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def infer_kind(path: str | Path) -> ArtifactKind:
    return _KIND_BY_SUFFIX.get(Path(path).suffix.lower(), "other")


def artifact_for(path: str | Path, kind: ArtifactKind | None = None) -> Artifact:
    """Hash a file on disk into an :class:`Artifact` (kind inferred from the suffix if omitted)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"artifact does not exist or is not a file: {p}")
    return Artifact(
        path=str(p), sha256=sha256_file(p), bytes=p.stat().st_size, kind=kind or infer_kind(p)
    )


# --- environment facts -----------------------------------------------------------------------


def make_run_id(config_sha256: str, seed: int | None, now: datetime | None = None) -> str:
    """``YYYYmmddTHHMMSS-<config sha prefix>-<seed|none>`` in UTC."""
    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    return f"{stamp:%Y%m%dT%H%M%S}-{config_sha256[:6]}-{seed if seed is not None else 'none'}"


def git_info(root: Path | None = None) -> tuple[str, bool]:
    """``(git sha, dirty)``; ``("unknown", False)`` outside a repository or without commits."""
    cwd = root or repo_root()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    git_sha = sha.stdout.strip() if sha.returncode == 0 else "unknown"
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    return git_sha, dirty


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = dist_version(name)
        except PackageNotFoundError:
            continue
    return out


def uv_lock_sha256(root: Path | None = None) -> str:
    lock = (root or repo_root()) / "uv.lock"
    return sha256_file(lock) if lock.is_file() else ""


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover
        return "unknown"


def _username() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # pragma: no cover
        return os.environ.get("USER", "unknown")


# --- manifest I/O ----------------------------------------------------------------------------


def write_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Atomically write ``manifest`` as indented JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_manifest(path: str | Path) -> Manifest:
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    return Manifest.model_validate_json(p.read_text(encoding="utf-8"))


def find_runs(stage: StageName, runs_dir: str | Path = "runs") -> list[Manifest]:
    """All readable manifests of ``stage`` under ``runs_dir``, oldest first."""
    stage_dir = Path(runs_dir) / stage
    if not stage_dir.is_dir():
        return []
    manifests: list[Manifest] = []
    for run_dir in sorted(stage_dir.iterdir()):
        path = run_dir / MANIFEST_NAME
        if not path.is_file():
            continue
        try:
            manifests.append(read_manifest(path))
        except ValueError:  # unreadable or foreign manifest: skip, never crash a listing
            continue
    manifests.sort(key=lambda m: (m.created_at, m.run_id))
    return manifests


def manifest_json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of :class:`Manifest`; committed at docs/manifest.schema.json."""
    schema = Manifest.model_json_schema()
    return {"$schema": SCHEMA_DIALECT, **schema}


def write_manifest_schema(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest_json_schema(), indent=2) + "\n", encoding="utf-8")
    return p


# --- run context -----------------------------------------------------------------------------


class RunContext:
    """Per-run state handed to every stage function (``fn(cfg, ctx, **kw)``).

    * ``ctx.out_dir`` — ``runs/<stage>/<run_id>/``; put every output there.
    * ``ctx.add_input(path, kind)`` / ``ctx.add_output(path, kind)`` — hash artifacts.
    * ``ctx.log(**extras)`` — free-form manifest extras (stage contracts in CONTRACTS.md section 5).
    * ``ctx.dry_run`` — stages must plan only, never compute, when this is set.
    * ``ctx.resume`` — with ``resume=True`` the most recent failed/partial run of the same stage,
      configuration and seed is reused (same ``out_dir``), so per-unit markers are found again.
    """

    def __init__(
        self,
        stage: StageName,
        cfg: Settings,
        seed: int | None = None,
        resume: bool = False,
        executor: Executor | None = None,
        *,
        dry_run: bool = False,
        runs_dir: str | Path | None = None,
        now: datetime | None = None,
    ) -> None:
        self.stage: StageName = stage
        self.cfg = cfg
        self.seed = seed
        self.resume = resume
        self.executor = executor
        self.dry_run = dry_run
        self.config_sha256 = cfg.sha256()
        self.created_at = (now or datetime.now(UTC)).astimezone(UTC)
        self.runs_dir = Path(runs_dir) if runs_dir is not None else Path(cfg.paths.runs_dir)
        self.inputs: list[Artifact] = []
        self.outputs: list[Artifact] = []
        self.extras: dict[str, Any] = {}
        self.slurm: SlurmInfo | None = None
        self.resumed_from: str | None = None

        stage_dir = self.runs_dir / stage
        run_id = self._resumable_run_id(stage_dir) if resume else None
        if run_id is None:
            run_id = self._fresh_run_id(stage_dir)
        else:
            self.resumed_from = run_id
        self.run_id = run_id
        self.out_dir = stage_dir / run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- run ids ---------------------------------------------------------------------------

    def _fresh_run_id(self, stage_dir: Path) -> str:
        stamp = self.created_at
        for _ in range(3600):  # same second, same config, same seed: nudge the timestamp
            run_id = make_run_id(self.config_sha256, self.seed, stamp)
            if not (stage_dir / run_id).exists():
                return run_id
            stamp = stamp + timedelta(seconds=1)
        raise RuntimeError(f"could not allocate a unique run_id under {stage_dir}")

    def _resumable_run_id(self, stage_dir: Path) -> str | None:
        if not stage_dir.is_dir():
            return None
        suffix = f"-{self.config_sha256[:6]}-{self.seed if self.seed is not None else 'none'}"
        for run_dir in sorted(stage_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
                continue
            manifest_path = run_dir / MANIFEST_NAME
            if not manifest_path.is_file():
                return run_dir.name  # crashed before its manifest was written: resumable
            try:
                manifest = read_manifest(manifest_path)
            except ValueError:
                continue
            if manifest.status in ("failed", "partial"):
                return run_dir.name
        return None

    # -- artifacts and extras -----------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_NAME

    def add_input(self, path: str | Path, kind: ArtifactKind | None = None) -> Artifact:
        art = artifact_for(path, kind)
        self.inputs.append(art)
        return art

    def add_output(self, path: str | Path, kind: ArtifactKind | None = None) -> Artifact:
        art = artifact_for(path, kind)
        self.outputs = [a for a in self.outputs if a.path != art.path] + [art]
        return art

    def log(self, **extras: Any) -> None:
        self.extras.update(extras)

    # -- manifest ------------------------------------------------------------------------------

    def build_manifest(self, status: Status, wall_seconds: float) -> Manifest:
        root = repo_root()
        git_sha, git_dirty = git_info(root)
        return Manifest(
            stage=self.stage,
            run_id=self.run_id,
            created_at=self.created_at,
            host=_hostname(),
            user=_username(),
            git_sha=git_sha,
            git_dirty=git_dirty,
            uv_lock_sha256=uv_lock_sha256(root),
            python=platform.python_version(),
            packages=package_versions(),
            config_sha256=self.config_sha256,
            config=self.cfg.model_dump(mode="json"),
            seed=self.seed,
            inputs=list(self.inputs),
            outputs=[] if self.dry_run else list(self.outputs),
            wall_seconds=float(wall_seconds),
            status=status,
            slurm=self.slurm,
            extras=dict(self.extras),
        )

    def write_manifest(self, status: Status, wall_seconds: float) -> Path:
        return write_manifest(self.build_manifest(status, wall_seconds), self.manifest_path)


StageFn = Callable[..., Any]


def run_stage(
    stage: StageName,
    cfg: Settings,
    fn: StageFn,
    *,
    seed: int | None = None,
    resume: bool = False,
    executor: Executor | None = None,
    dry_run: bool = False,
    runs_dir: str | Path | None = None,
    reraise: bool = False,
    **kw: Any,
) -> StageResult:
    """Run ``fn(cfg, ctx, **kw)`` under a :class:`RunContext`, always writing the manifest.

    ``fn`` may return a :class:`StageResult`, a summary ``dict`` or ``None``. An exception
    yields ``status="failed"`` (error and traceback in ``extras``) and is swallowed unless
    ``reraise=True``. ``dry_run=True`` yields ``status="partial"`` with no outputs.
    """
    ctx = RunContext(
        stage, cfg, seed, resume, executor, dry_run=dry_run, runs_dir=runs_dir, now=None
    )
    status: Status = "ok"
    summary: dict[str, float | int | str] = {}
    t0 = time.perf_counter()
    try:
        result = fn(cfg, ctx, **kw)
        if isinstance(result, StageResult):
            status = result.status
            summary = dict(result.summary)
            for art in result.outputs:
                if all(a.path != art.path for a in ctx.outputs):
                    ctx.outputs.append(art)
        elif isinstance(result, dict):
            summary = dict(result)
        if dry_run:
            status = "partial"
    except Exception as exc:
        status = "failed"
        summary = {"error": repr(exc)}
        ctx.log(error=repr(exc), traceback=traceback.format_exc())
        if reraise:
            ctx.write_manifest(status, time.perf_counter() - t0)
            raise
    wall = time.perf_counter() - t0
    manifest_path = ctx.write_manifest(status, wall)
    return StageResult(
        stage=stage,
        run_id=ctx.run_id,
        manifest_path=str(manifest_path),
        status=status,
        outputs=[] if dry_run else list(ctx.outputs),
        summary=summary,
    )


__all__ = [
    "MANIFEST_NAME",
    "RUN_ID_RE",
    "RunContext",
    "SCHEMA_DIALECT",
    "TRACKED_PACKAGES",
    "artifact_for",
    "find_runs",
    "git_info",
    "infer_kind",
    "make_run_id",
    "manifest_json_schema",
    "package_versions",
    "read_manifest",
    "run_stage",
    "sha256_bytes",
    "sha256_file",
    "sha256_frames",
    "sha256_of_json",
    "uv_lock_sha256",
    "write_manifest",
    "write_manifest_schema",
]
