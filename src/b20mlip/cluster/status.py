"""``b20mlip cluster status``: ``squeue`` for the user plus local unit-marker counts.

No manifest: this is a read-only view. A dead socket raises :class:`ClusterUnreachable`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from b20mlip.cluster.remote import Transport, parse_squeue
from b20mlip.config import Settings
from b20mlip.executors import Runner

QUEUE_COLUMNS = ("JOBID", "NAME", "STATE", "TIME", "PARTITION", "NODELIST(REASON)")
LOCAL_COLUMNS = ("JOB", "UNITS", "DONE", "FAILED", "PENDING")


def local_marker_counts(staging_root: str | Path) -> dict[str, dict[str, int]]:
    """Per job dir under ``runs/slurm/``: ``units.txt`` count and ``.done``/``.failed`` markers."""
    root = Path(staging_root)
    out: dict[str, dict[str, int]] = {}
    if not root.is_dir():
        return out
    for job_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        units_txt = job_dir / "units.txt"
        units_dir = job_dir / "units"
        if not units_txt.is_file() and not units_dir.is_dir():
            continue
        total = (
            len([ln for ln in units_txt.read_text(encoding="utf-8").splitlines() if ln.strip()])
            if units_txt.is_file()
            else 0
        )
        done = len(list(units_dir.glob("*.done"))) if units_dir.is_dir() else 0
        failed = len(list(units_dir.glob("*.failed"))) if units_dir.is_dir() else 0
        total = max(total, done + failed)
        out[job_dir.name] = {
            "units": total,
            "done": done,
            "failed": failed,
            "pending": total - done - failed,
        }
    return out


def show(cfg: Settings, *, runner: Runner | None = None) -> dict[str, Any]:
    """``squeue -u $USER`` rows plus local marker counts (raises when the socket is dead)."""
    t = Transport(cfg, runner)
    t.check_socket()
    res = t.run("squeue")
    rows = parse_squeue(res.stdout) if res.ok else []
    return {
        "alias": cfg.cluster.alias,
        "queue": [asdict(r) for r in rows],
        "squeue_error": None if res.ok else (res.stderr.strip() or f"rc {res.returncode}"),
        "local": local_marker_counts(Path(cfg.paths.runs_dir) / "slurm"),
        "commands": t.records_json(),
    }


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers).rstrip()]
    lines += [fmt.format(*row).rstrip() for row in rows]
    return "\n".join(lines)


def format_table(report: dict[str, Any]) -> str:
    """Plain-text rendering used by the CLI."""
    out = [f"queue on {report.get('alias', '?')} (squeue -u $USER):"]
    queue = report.get("queue", [])
    if queue:
        out.append(
            _table(
                QUEUE_COLUMNS,
                [
                    [r["job_id"], r["name"], r["state"], r["time"], r["partition"], r["reason"]]
                    for r in queue
                ],
            )
        )
    else:
        out.append(
            "  (no jobs)"
            if not report.get("squeue_error")
            else f"  squeue failed: {report['squeue_error']}"
        )
    out.append("")
    out.append("local unit markers (runs/slurm/<job>/units):")
    local = report.get("local", {})
    if local:
        out.append(
            _table(
                LOCAL_COLUMNS,
                [
                    [job, str(c["units"]), str(c["done"]), str(c["failed"]), str(c["pending"])]
                    for job, c in local.items()
                ],
            )
        )
    else:
        out.append("  (no synced jobs)")
    return "\n".join(out)


__all__ = ["LOCAL_COLUMNS", "QUEUE_COLUMNS", "format_table", "local_marker_counts", "show"]
