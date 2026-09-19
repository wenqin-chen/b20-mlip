"""``dft prep`` / ``dft run`` / ``dft collect`` stage functions (``fn(cfg, ctx, **kw)``).

``run`` submits the pending units of a planned root through ``ctx.executor`` (local: runs them
now; SLURM: array job over the cluster mirror of the root) and, after ``wait``, re-collects the
unit directories. The units root is the single source of truth: markers ``.done``/``.failed`` and
``pw.out`` live inside each unit directory, so ``dft collect`` (and ``--resume``) never need the
job's own bookkeeping. For SLURM the root is mirrored at ``<cluster.scratch>/b20-mlip/<root
relative to the repo>`` (``cluster bootstrap`` rsyncs the repo there; ``cluster sync`` pulls the
outputs back before ``dft collect``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from b20mlip.config import Settings, repo_root
from b20mlip.dft import qe
from b20mlip.executors import JobHandle, SlurmExecutor
from b20mlip.io import read_frames, write_frames
from b20mlip.models import DFTFrame, SlurmInfo, StageResult, Status
from b20mlip.provenance import RunContext

HANDLE_JSON = "handle.json"
COUNTS_JSON = "counts.json"


def stage_result(
    ctx: RunContext, status: Status, summary: Mapping[str, Any] | None = None
) -> StageResult:
    """A ``StageResult`` for the current context (``run_stage`` fills the manifest path)."""
    clean: dict[str, float | int | str] = {}
    for key, value in (summary or {}).items():
        if isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, (int, float, str)):
            clean[key] = value
        elif value is not None:
            clean[key] = json.dumps(value, sort_keys=True, default=str)
    return StageResult(
        stage=ctx.stage,
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status=status,
        outputs=list(ctx.outputs),
        summary=clean,
    )


def units_root_for(executor: object, root: str | Path, cfg: Settings) -> str:
    """The units root as the *executing* machine sees it.

    Local executors get the absolute local path. A ``SlurmExecutor`` gets the cluster mirror
    ``<cluster.scratch>/b20-mlip/<root relative to the repository>`` (falls back to the root's
    basename when the root lies outside the checkout).
    """
    r = Path(root).resolve()
    if isinstance(executor, SlurmExecutor):
        scratch = cfg.cluster.scratch
        if not scratch:
            raise ValueError("cluster.scratch is unset; run `b20mlip cluster bootstrap` first")
        try:
            rel = r.relative_to(repo_root())
        except ValueError:
            rel = Path(r.name)
        return f"{scratch.rstrip('/')}/b20-mlip/{rel.as_posix()}"
    return str(r)


def pseudo_md5s_for(frames: list[DFTFrame], cfg: Settings) -> dict[str, str]:
    symbols: list[str] = []
    for frame in frames:
        for s in qe.species_order(frame.numbers):
            if s not in symbols:
                symbols.append(s)
    return {s: cfg.dft.pseudo_md5s[s] for s in symbols if s in cfg.dft.pseudo_md5s}


def pw_version_of(frames: list[DFTFrame]) -> str | None:
    for frame in frames:
        version = frame.info.get("pw_version")
        if version:
            return str(version)
    return None


def submit_units(
    cfg: Settings,
    ctx: RunContext,
    root: str | Path,
    unit_ids: list[str],
    *,
    template: str = "qe_array",
    resources: Mapping[str, Any] | None = None,
    wait: bool = True,
    name: str | None = None,
) -> tuple[JobHandle, SlurmInfo | None]:
    """Submit ``unit_ids`` of ``root`` through ``ctx.executor``; ``wait`` polls to completion.

    The job name carries the run id, so every submission gets a fresh executor workdir: the
    executor's own job-level markers never shadow the unit-directory markers (the ones
    ``pending_units`` consults), which are the single source of truth for resume.
    """
    if ctx.executor is None:
        raise ValueError("no executor on the run context (pass --executor local|slurm)")
    if not cfg.cluster.qe_cmd:
        raise ValueError(
            "cluster.qe_cmd is unset: set it to the pw.x launcher, e.g. "
            "`--set cluster.qe_cmd='mpirun -np 8 pw.x -nk 2'` or configs/cluster/tillicum.yaml"
        )
    spec = qe.job_spec(
        root,
        unit_ids,
        cfg,
        name=name or f"{qe.job_name(root)}-{ctx.run_id}",
        template=template,
        resources=resources,
        units_root=units_root_for(ctx.executor, root, cfg),
    )
    (ctx.out_dir / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    handle = ctx.executor.submit(spec)
    (ctx.out_dir / HANDLE_JSON).write_text(
        handle.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    ctx.add_output(ctx.out_dir / HANDLE_JSON, "json")
    info = ctx.executor.wait(handle) if wait else None
    if info is not None:
        ctx.slurm = info
    ctx.log(job_ids=list(handle.job_ids), job_workdir=handle.workdir)
    return handle, info


def write_counts(ctx: RunContext, counts: Mapping[str, Any]) -> Path:
    path = ctx.out_dir / COUNTS_JSON
    path.write_text(json.dumps(dict(counts), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.add_output(path, "json")
    return path


# --- prep --------------------------------------------------------------------------------------


def prep(cfg: Settings, ctx: RunContext, *, frames: str | Path, out: str | Path) -> dict[str, Any]:
    """One unit directory (``pw.in`` + ``unit.json``) per frame of ``frames`` under ``out``."""
    frames_path = Path(frames)
    ctx.add_input(frames_path, "frames")
    loaded = read_frames(frames_path)
    if not loaded:
        raise ValueError(f"no frames in {frames_path}")
    root = Path(out)
    ids = qe.plan_units(loaded, root, cfg)
    index = ctx.out_dir / "units.json"
    index.write_text(
        json.dumps({"units_root": str(root), "units": ids}, indent=2) + "\n", encoding="utf-8"
    )
    ctx.add_output(index, "json")
    ctx.log(
        n_units=len(ids),
        units_root=str(root),
        pseudo_md5s=dict(cfg.dft.pseudo_md5s),
        ecut_ry=cfg.dft.ecut_ry,
        ecut_rho=cfg.dft.ecut_rho,
        k_spacing_inv_A=cfg.dft.k_spacing_inv_A,
    )
    return {"n_units": len(ids), "units_root": str(root)}


# --- run ---------------------------------------------------------------------------------------


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    units: str | Path,
    limit: int | None = None,
    wait: bool = True,
    template: str = "qe_array",
    parallel: int = 1,
    resources: Mapping[str, Any] | None = None,
) -> StageResult:
    """Run the pending units under ``units`` (per-unit resume: ``.done`` units are skipped)."""
    root = Path(units)
    all_ids = qe.list_units(root)
    if not all_ids:
        raise FileNotFoundError(f"no planned units under {root} (run `dft prep` first)")
    pending = qe.pending_units(root, all_ids)
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    res: dict[str, Any] = {"parallel": max(1, int(parallel)), **dict(resources or {})}
    summary: dict[str, Any] = {
        "units_root": str(root),
        "n_planned": len(all_ids),
        "n_submitted": len(pending),
    }
    if not pending:
        frames, counts = qe.collect(root)
        write_counts(ctx, counts)
        ctx.log(n_units=0, n_failed=0, n_branch_rejected=0, counts=dict(counts))
        return stage_result(ctx, "ok", {**summary, **counts, "note": "nothing pending"})
    if ctx.dry_run:
        spec = qe.job_spec(root, pending, cfg, template=template, resources=res)
        (ctx.out_dir / "spec.json").write_text(
            spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        ctx.log(n_units=len(pending), n_failed=0, n_branch_rejected=0, dry_run=True)
        return stage_result(ctx, "partial", {**summary, "dry_run": 1})

    _handle, info = submit_units(
        cfg, ctx, root, pending, template=template, resources=res, wait=wait
    )
    frames, counts = qe.collect(root)
    write_counts(ctx, counts)
    if info is not None:  # SLURM accounting: outputs arrive with `cluster sync`
        n_done, n_failed = info.units_done, info.units_failed
    else:
        states = {u: qe.unit_status(root / u) for u in pending}
        n_done = sum(s == "done" for s in states.values())
        n_failed = sum(s == "failed" for s in states.values())
    n_branch_rejected = sum(f.branch_ok is False for f in frames)
    ctx.log(
        pw_version=pw_version_of(frames),
        pseudo_md5s=pseudo_md5s_for(frames, cfg) or dict(cfg.dft.pseudo_md5s),
        n_units=len(pending),
        n_failed=n_failed,
        n_branch_rejected=n_branch_rejected,
        counts=dict(counts),
        template=template,
    )
    if not wait:
        status: Status = "partial"
    elif n_done == len(pending) and n_failed == 0:
        status = "ok"
    elif n_done == 0:
        status = "failed"
    else:
        status = "partial"
    return stage_result(
        ctx,
        status,
        {**summary, **counts, "n_done": n_done, "n_failed": n_failed, "waited": wait},
    )


# --- collect -----------------------------------------------------------------------------------


def collect(cfg: Settings, ctx: RunContext, *, units: str | Path, out: str | Path) -> StageResult:
    """Parse every unit under ``units``; write the converged frames (QE labels) to ``out``."""
    root = Path(units)
    frames, counts = qe.collect(root)
    if counts["planned"] == 0:
        raise FileNotFoundError(f"no planned units under {root}")
    converged = [qe.to_plain_frame(f) for f in frames if f.converged]
    out_path = Path(out)
    if converged:
        ctx.add_output(write_frames(converged, out_path).path, "frames")
    write_counts(ctx, counts)
    n_branch_rejected = sum(f.branch_ok is False for f in frames)
    ctx.log(
        pw_version=pw_version_of(frames),
        pseudo_md5s=pseudo_md5s_for(frames, cfg) or dict(cfg.dft.pseudo_md5s),
        n_units=counts["planned"],
        n_failed=counts["failed"] + counts["unconverged"],
        n_branch_rejected=n_branch_rejected,
        counts=dict(counts),
        units_root=str(root),
    )
    status: Status = "ok" if converged else "failed"
    return stage_result(
        ctx,
        status,
        {
            **counts,
            "n_frames": len(converged),
            "n_branch_rejected": n_branch_rejected,
            "out": str(out_path),
        },
    )


__all__ = [
    "collect",
    "prep",
    "pseudo_md5s_for",
    "pw_version_of",
    "run",
    "stage_result",
    "submit_units",
    "units_root_for",
    "write_counts",
]
