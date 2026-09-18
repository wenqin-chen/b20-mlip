"""``b20mlip cluster sync``: pull job results back (SPEC.md section 8, "socket death != job done").

For every job directory under ``<scratch>/b20-mlip/jobs/`` (or the ``jobs`` given):

1. ``sacct -j <ids> -X -P -o JobID,State,ExitCode,Elapsed,NodeList`` for the ids recorded in the
   local staging ``runs/slurm/<job>/handle.json`` (written by ``SlurmExecutor.submit``);
2. ``rsync`` of ``units/*.done|*.failed``, ``logs/`` and the result files into
   ``runs/slurm/<job>/`` (QE wavefunctions and build trees excluded);
3. unit states from the pulled markers only — a unit without ``units/<unit>.done`` stays
   ``pending`` whatever sacct says, and a dead ControlMaster socket raises
   :class:`ClusterUnreachable` instead of pretending anything finished.

The manifest lists ``units_done`` / ``units_failed`` / ``units_pending`` per job; the stage is
``ok`` when no unit is pending, else ``partial`` (run it again later).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from b20mlip.cluster.remote import (
    RESULT_EXCLUDES,
    Transport,
    nodelist_count,
    parse_sacct_jobs,
    q,
)
from b20mlip.config import Settings
from b20mlip.executors import JobHandle, Runner, unit_states
from b20mlip.models import SlurmInfo, StageResult
from b20mlip.provenance import RunContext

SYNC_NAME = "sync.json"
MARKER_RULE = (
    "a unit is done only when units/<unit>.done exists (failed: units/<unit>.failed); "
    "sacct states never mark a unit done; a dead socket aborts the sync"
)


def elapsed_seconds(text: str) -> int:
    """``D-HH:MM:SS`` / ``HH:MM:SS`` / ``MM:SS`` -> seconds (unparsable -> 0)."""
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", text.strip())
    if not m:
        return 0
    days, hours, minutes, seconds = (int(x) if x else 0 for x in m.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def job_ids_of(dest: Path) -> list[str]:
    handle = dest / "handle.json"
    if not handle.is_file():
        return []
    try:
        return list(JobHandle.model_validate_json(handle.read_text(encoding="utf-8")).job_ids)
    except ValueError:
        return []


def units_of(dest: Path) -> list[str]:
    units_txt = dest / "units.txt"
    if units_txt.is_file():
        return [
            ln.strip() for ln in units_txt.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    units_dir = dest / "units"
    if not units_dir.is_dir():
        return []
    return sorted({p.stem for p in units_dir.iterdir() if p.suffix in (".done", ".failed")})


def pull(
    cfg: Settings,
    ctx: RunContext,
    *,
    runner: Runner | None = None,
    jobs: list[str] | None = None,
    template_dir: str | Path | None = None,
) -> StageResult:
    """Stage entry point (``run_stage("cluster.sync", cfg, pull, runner=..., jobs=...)``)."""
    staging_root = Path(cfg.paths.runs_dir) / "slurm"
    t = Transport(cfg, runner, template_dir=template_dir, staging_root=staging_root)
    if ctx.dry_run:
        ctx.log(plan=["socket_check", "list_jobs", "sacct", "rsync_pull"], rule=MARKER_RULE)
        return StageResult(
            stage="cluster.sync",
            run_id=ctx.run_id,
            manifest_path=str(ctx.manifest_path),
            status="partial",
            outputs=[],
            summary={"jobs": len(jobs or [])},
        )
    scratch = cfg.cluster.scratch
    if not scratch:
        raise RuntimeError("cluster.scratch is not set; run `b20mlip cluster bootstrap` first")
    t.check_socket()
    jobs_dir = f"{scratch.rstrip('/')}/b20-mlip/jobs"
    if jobs is None:
        listing = t.run("list_jobs", jobs_dir=q(jobs_dir))
        jobs = (
            [ln.strip() for ln in listing.stdout.splitlines() if ln.strip()] if listing.ok else []
        )

    report: dict[str, dict[str, Any]] = {}
    all_ids: list[str] = []
    total_done = total_failed = total_pending = 0
    max_nodes = 0
    wall = ""
    for job in jobs:
        workdir = f"{jobs_dir}/{job}"
        dest = staging_root / job
        dest.mkdir(parents=True, exist_ok=True)
        ids = job_ids_of(dest)
        rows = []
        sacct_error: str | None = None
        if ids:
            res = t.run("sacct", ids=q(",".join(ids)))
            if res.ok:
                rows = parse_sacct_jobs(res.stdout)
            else:
                sacct_error = res.stderr.strip() or f"rc {res.returncode}"
        t.rsync("rsync_pull", f"{t.alias}:{workdir}/", f"{dest}/", excludes=RESULT_EXCLUDES)
        units = units_of(dest)
        states = unit_states(dest, units)
        n_done = sum(s == "done" for s in states.values())
        n_failed = sum(s == "failed" for s in states.values())
        pending = [u for u, s in states.items() if s == "pending"]
        all_terminal = bool(rows) and all(r.terminal for r in rows)
        report[job] = {
            "job_ids": ids,
            "sacct": [asdict(r) for r in rows],
            "sacct_error": sacct_error,
            "all_terminal": all_terminal,
            "units": len(units),
            "units_done": n_done,
            "units_failed": n_failed,
            "units_pending": len(pending),
            "pending": pending[:50],
            "crashed_without_marker": all_terminal and bool(pending),
            "dest": str(dest),
        }
        all_ids += ids
        total_done += n_done
        total_failed += n_failed
        total_pending += len(pending)
        for r in rows:
            max_nodes = max(max_nodes, nodelist_count(r.nodelist))
            if elapsed_seconds(r.elapsed) >= elapsed_seconds(wall):
                wall = r.elapsed

    sync_path = ctx.out_dir / SYNC_NAME
    sync_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ctx.add_output(sync_path, "json")
    if all_ids:
        ctx.slurm = SlurmInfo(
            job_ids=all_ids,
            account=cfg.cluster.account or "",
            partition=cfg.cluster.partition_cpu or cfg.cluster.partition_gpu or "",
            nodes=max_nodes,
            wall=wall,
            units_done=total_done,
            units_failed=total_failed,
        )
    ctx.log(jobs=report, commands=t.records_json(), rule=MARKER_RULE, jobs_dir=jobs_dir)
    status: Any = "ok" if total_pending == 0 else "partial"
    return StageResult(
        stage="cluster.sync",
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status=status,
        outputs=list(ctx.outputs),
        summary={
            "jobs": len(jobs),
            "units_done": total_done,
            "units_failed": total_failed,
            "units_pending": total_pending,
            "dest": str(staging_root),
        },
    )


__all__ = ["MARKER_RULE", "SYNC_NAME", "elapsed_seconds", "job_ids_of", "pull", "units_of"]
