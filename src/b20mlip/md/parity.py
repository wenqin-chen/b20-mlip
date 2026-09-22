"""The ASE-vs-LAMMPS parity gate (SPEC.md section 6, CONTRACTS.md A6): ``check``, the LAMMPS
single-point inputs, their collection and the ``md parity`` stage.

``check(model_path, frames, lammps_forces_json, tol_f=1e-3, tol_e=1e-4)`` evaluates the ASE
``MACECalculator`` on every frame and compares with the LAMMPS numbers of ``{frame_id:
{"energy": eV, "forces": [[eV/Å] * 3] * natoms}}`` (keys starting with ``@`` are ignored). It
returns ``{"passed", "max_dF_eVA", "max_dE_eV_atom", "n_frames", "tol_f", "tol_e", "frames":
[...per-frame table...], "missing": [...]}``; ``passed`` requires every frame present, max|ΔF| <
tol_f over all force components and |ΔE|/N < tol_e on every frame. The gate needs 20 frames
(SPEC.md); fewer are compared but flagged (``n_frames``) — ``parity.<label>.passed`` is published
only by the stage, which uses whatever ``--frames`` holds.

LAMMPS side: ``parity_lammps_inputs(frames, model_lammps, out_dir)`` writes one unit directory
per frame (``<out_dir>/<frame_id>/{data.lmp, in.parity}``: ``run 0`` with ``thermo_style custom
step pe`` and a ``dump ... id type fx fy fz`` at full precision); ``collect_parity(job_dir)``
parses ``log.lammps`` + ``forces.dump`` of every unit into the JSON above. The stage submits all
frames as ONE unit (a bash loop, ``templates/slurm/lammps.sbatch.j2`` without ``resources.input``)
because twenty 8-atom single points take seconds.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from b20mlip.config import Settings
from b20mlip.executors import JobSpec, LocalExecutor
from b20mlip.io import frame_to_atoms, read_frames
from b20mlip.md import ase_md, lammps
from b20mlip.md.common import (
    elements_of_atoms,
    model_provenance,
    number_meta,
    relpath,
    sanitize_key_segment,
    stage_result,
    write_numbers,
)
from b20mlip.models import Frame, StageResult
from b20mlip.provenance import RunContext

log = logging.getLogger("b20mlip.md")

PARITY_INPUT = "in.parity"
FORCES_DUMP = "forces.dump"
FRAMES_SUBDIR = "frames"
PARITY_JSON = "parity_lammps.json"
RESULT_JSON = "parity.json"
GATE_FRAMES = 20
DEFAULT_TOL_F = 1e-3
DEFAULT_TOL_E = 1e-4
PARITY_HEADER = "# b20mlip-parity"

_PARITY_TEMPLATE = """{header} frame_id={frame_id} natoms={natoms} compound={compound}
# single point for the ASE-vs-LAMMPS parity gate (b20mlip.md.parity); ML-MACE syntax per
# https://mace-docs.readthedocs.io/en/latest/guide/lammps.html
units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p
read_data       data.lmp

pair_style      mace no_domain_decomposition
pair_coeff      * * {model} {elements}

thermo_style    custom step pe
thermo_modify   format float %.15g
thermo          1
dump            1 all custom 1 {dump} id type fx fy fz
dump_modify     1 sort id format float %.15g
run             0
"""


# --- the comparison -------------------------------------------------------------------------------


def _as_atoms(frames: Any) -> list[tuple[str, Atoms]]:
    """``[(frame_id, Atoms)]`` from Frames, Atoms (``info["frame_id"]``) or an extxyz path."""
    if isinstance(frames, (str, Path)):
        frames = read_frames(frames)
    out: list[tuple[str, Atoms]] = []
    for i, item in enumerate(frames):
        if isinstance(item, Frame):
            atoms = frame_to_atoms(item)
            atoms.calc = None
            out.append((item.frame_id, atoms))
        elif isinstance(item, Atoms):
            fid = str(item.info.get("frame_id", f"frame{i:03d}"))
            out.append((fid, item.copy()))
        else:
            raise TypeError(f"frames must be Frame or Atoms, got {type(item).__name__}")
    return out


def load_lammps_json(source: str | Path | Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    data = (
        json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(data, Mapping):
        raise ValueError("the LAMMPS parity JSON must be an object {frame_id: {energy, forces}}")
    return {k: v for k, v in data.items() if not k.startswith("@")}


def ase_reference(
    model_path: str | Path | None,
    frames: Any,
    *,
    head: str = "Default",
    calc: Any | None = None,
    cfg: Settings | None = None,
) -> dict[str, dict[str, Any]]:
    """ASE energies/forces of the frames with the MACE calculator, in the parity JSON layout."""
    if calc is None:
        if model_path is None:
            raise ValueError("either model_path or calc is required")
        calc = ase_md.make_calculator(model_path, cfg or Settings.model_validate({}), head)
    out: dict[str, dict[str, Any]] = {}
    for fid, atoms in _as_atoms(frames):
        atoms.calc = calc
        out[fid] = {
            "energy": float(atoms.get_potential_energy()),
            "forces": np.asarray(atoms.get_forces(), dtype=float).tolist(),
            "natoms": len(atoms),
        }
    return out


def compare(
    reference: Mapping[str, Mapping[str, Any]],
    lammps_values: Mapping[str, Mapping[str, Any]],
    tol_f: float = DEFAULT_TOL_F,
    tol_e: float = DEFAULT_TOL_E,
) -> dict[str, Any]:
    """The parity verdict from two ``{frame_id: {energy, forces}}`` maps (ASE = reference)."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for fid, ref in reference.items():
        other = lammps_values.get(fid)
        if other is None:
            missing.append(fid)
            continue
        f_ref = np.asarray(ref["forces"], dtype=float)
        f_lmp = np.asarray(other["forces"], dtype=float)
        if f_ref.shape != f_lmp.shape:
            raise ValueError(f"frame {fid}: force shapes differ {f_ref.shape} vs {f_lmp.shape}")
        natoms = f_ref.shape[0]
        d_f = float(np.max(np.abs(f_ref - f_lmp))) if natoms else 0.0
        d_e = abs(float(ref["energy"]) - float(other["energy"])) / max(1, natoms)
        rows.append(
            {
                "frame_id": fid,
                "natoms": int(natoms),
                "E_ase_eV": float(ref["energy"]),
                "E_lammps_eV": float(other["energy"]),
                "dE_eV_atom": d_e,
                "max_dF_eVA": d_f,
                "rms_dF_eVA": float(np.sqrt(np.mean((f_ref - f_lmp) ** 2))) if natoms else 0.0,
                "passed": bool(d_f < tol_f and d_e < tol_e),
            }
        )
    max_df = max((r["max_dF_eVA"] for r in rows), default=float("nan"))
    max_de = max((r["dE_eV_atom"] for r in rows), default=float("nan"))
    passed = bool(rows) and not missing and all(r["passed"] for r in rows)
    return {
        "passed": passed,
        "max_dF_eVA": max_df,
        "max_dE_eV_atom": max_de,
        "n_frames": len(rows),
        "n_missing": len(missing),
        "missing": missing,
        "tol_f": float(tol_f),
        "tol_e": float(tol_e),
        "gate_frames": GATE_FRAMES,
        "enough_frames": len(rows) >= GATE_FRAMES,
        "frames": rows,
    }


def check(
    model_path: str | Path | None,
    frames: Any,
    lammps_forces_json: str | Path | Mapping[str, Any],
    tol_f: float = DEFAULT_TOL_F,
    tol_e: float = DEFAULT_TOL_E,
    *,
    head: str = "Default",
    calc: Any | None = None,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    """ASE ``MACECalculator`` forces/energies on ``frames`` vs the LAMMPS JSON (module docstring).

    ``calc`` (an existing calculator) saves reloading the model; ``cfg`` sets device/dtype/threads.
    """
    reference = ase_reference(model_path, frames, head=head, calc=calc, cfg=cfg)
    verdict = compare(reference, load_lammps_json(lammps_forces_json), tol_f, tol_e)
    verdict["head"] = head
    verdict["model_path"] = None if model_path is None else str(model_path)
    return verdict


# --- LAMMPS single points -------------------------------------------------------------------------


def parity_lammps_inputs(
    frames: Any,
    model_lammps: str | Path,
    out_dir: str | Path,
    *,
    elements: Sequence[str] | None = None,
) -> list[str]:
    """One LAMMPS ``run 0`` unit directory per frame under ``out_dir``; returns the frame ids.

    ``model_lammps`` is written into ``pair_coeff`` as given (the job stages the file next to
    the frame directories, so pass ``../<name>-lammps.pt`` or an absolute path). ``elements``
    fixes the LAMMPS type order (default: every element of the frames by atomic number).
    """
    items = _as_atoms(frames)
    if not items:
        raise ValueError("no frames for the parity inputs")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if elements is None:
        numbers = {int(z) for _, atoms in items for z in atoms.numbers}
        elements = elements_of_atoms(numbers)
    ids: list[str] = []
    for fid, atoms in items:
        unit = root / fid
        unit.mkdir(parents=True, exist_ok=True)
        lammps.write_data(atoms, unit / lammps.DATA_NAME, elements)
        (unit / PARITY_INPUT).write_text(
            _PARITY_TEMPLATE.format(
                header=PARITY_HEADER,
                frame_id=fid,
                natoms=len(atoms),
                compound=atoms.get_chemical_formula("metal", empirical=True),
                model=str(model_lammps),
                elements=" ".join(elements),
                dump=FORCES_DUMP,
            ),
            encoding="utf-8",
        )
        ids.append(fid)
    (root / "units.json").write_text(
        json.dumps({"units": ids, "elements": list(elements)}, indent=2) + "\n", encoding="utf-8"
    )
    return ids


def parse_forces_dump(path: str | Path) -> np.ndarray:
    """``(natoms, 3)`` forces of a ``dump ... id type fx fy fz`` file, sorted by id."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[tuple[int, list[float]]] = []
    columns: list[str] | None = None
    in_atoms = False
    for line in lines:
        if line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]
            in_atoms = True
            continue
        if line.startswith("ITEM:"):
            in_atoms = False
            continue
        if in_atoms and columns is not None and line.strip():
            parts = line.split()
            rec = dict(zip(columns, parts, strict=False))
            rows.append((int(rec["id"]), [float(rec["fx"]), float(rec["fy"]), float(rec["fz"])]))
    if not rows:
        raise ValueError(f"no atoms in {path}")
    rows.sort(key=lambda r: r[0])
    return np.array([r[1] for r in rows], dtype=float)


def parse_single_point_energy(log_path: str | Path) -> float:
    """``pe`` of the (only) thermo row of a ``run 0`` log."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    blocks = lammps.parse_thermo_blocks(text)
    if not blocks or blocks[-1]["rows"].size == 0:
        raise ValueError(f"no thermo row in {log_path}")
    columns = list(blocks[-1]["columns"])
    col = next((c for c in ("pe", "poteng") if c in columns), None)
    if col is None:
        raise ValueError(f"no pe column in {log_path} (thermo_style custom step pe expected)")
    return float(blocks[-1]["rows"][-1, columns.index(col)])


def collect_parity(job_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """``{frame_id: {energy, forces, natoms}}`` from every finished unit under ``job_dir``
    (``<job_dir>/<frame_id>/`` or ``<job_dir>/frames/<frame_id>/``); written to ``out`` when
    given. Units without a log are listed under ``"@missing"``."""
    root = Path(job_dir)
    if (root / FRAMES_SUBDIR).is_dir():
        root = root / FRAMES_SUBDIR
    units = (
        json.loads((root / "units.json").read_text(encoding="utf-8"))["units"]
        if (root / "units.json").is_file()
        else sorted(p.name for p in root.iterdir() if p.is_dir())
    )
    result: dict[str, Any] = {}
    missing: list[str] = []
    for fid in units:
        unit = root / fid
        log_path, dump_path = unit / lammps.LOG_NAME, unit / FORCES_DUMP
        if not (log_path.is_file() and dump_path.is_file()):
            missing.append(fid)
            continue
        forces = parse_forces_dump(dump_path)
        result[fid] = {
            "energy": parse_single_point_energy(log_path),
            "forces": forces.tolist(),
            "natoms": int(forces.shape[0]),
        }
    if missing:
        result["@missing"] = missing
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def parity_unit_script(cmd: str) -> str:
    """Bash: run every frame directory's single point in sequence (one unit)."""
    return (
        'cd "$B20_WORKDIR"\n'
        f"for d in {FRAMES_SUBDIR}/*/; do\n"
        f'  ( cd "$d" && {cmd} -in {PARITY_INPUT} -log {lammps.LOG_NAME} )\n'
        "done\n"
    )


# --- the stage ------------------------------------------------------------------------------------


def _numbers(
    verdict: Mapping[str, Any], prov: Mapping[str, Any], head: str, seed: int | None, label: str
) -> dict[str, Any]:
    n = max(1, int(verdict["n_frames"]))
    meta = number_meta(
        provenance=prov, head=head, n=n, seed=seed, engine="lammps-vs-ase", ensemble=None, T=None,
        ci95=None, ci95_reason="deterministic maximum over frames", label=label,
        tol_f_eVA=verdict["tol_f"], tol_e_eV_atom=verdict["tol_e"], n_missing=verdict["n_missing"],
        enough_frames=bool(verdict["enough_frames"]), gate_frames=GATE_FRAMES,
    )  # fmt: skip
    # keys carry the model label: every exported model passes its own gate, and a later model's
    # run must not replace an earlier model's verdict (the report reads the unlabelled keys of
    # runs made before 2026-09-22 as B0's)
    tag = sanitize_key_segment(label)
    out: dict[str, Any] = {
        f"parity.{tag}.passed": 1 if verdict["passed"] else 0,
        f"parity.{tag}.passed@meta": {**meta, "unit": "bool"},
        f"parity.{tag}.n_frames": int(verdict["n_frames"]),
        f"parity.{tag}.n_frames@meta": {**meta, "unit": "frames"},
    }
    if verdict["n_frames"]:
        for key, unit in (("max_dF_eVA", "eV/A"), ("max_dE_eV_atom", "eV/atom")):
            out[f"parity.{tag}.{key}"] = float(verdict[key])
            out[f"parity.{tag}.{key}@meta"] = {**meta, "unit": unit}
            out[f"md.parity.{tag}.{key}"] = float(verdict[key])
            out[f"md.parity.{tag}.{key}@meta"] = {**meta, "unit": unit}
    return out


def run(
    cfg: Settings,
    ctx: RunContext,
    model: str | Path,
    frames: str | Path,
    lammps_json: str | Path | None = None,
    executor: Any | None = None,
    *,
    head: str = "Default",
    tol_f: float = DEFAULT_TOL_F,
    tol_e: float = DEFAULT_TOL_E,
    wait: bool = True,
    gpu: bool | None = None,
    label: str | None = None,
    calc: Any | None = None,
) -> StageResult:
    """Stage ``md.parity`` (``b20mlip md parity``).

    With ``lammps_json`` the check runs now. Without it the LAMMPS single points are staged as
    one job: when ``cluster.lammps_cmd`` is usable by the executor the job runs (local) or is
    submitted and waited for (SLURM), the results are collected and checked; otherwise the inputs
    are left under ``<run dir>/parity_inputs/`` and the stage ends ``partial`` ("run them, then
    pass --lammps-json"). ``--resume`` with a fetched ``job/`` collects and checks. Outputs:
    ``parity.json`` (verdict + per-frame table), ``parity_lammps.json`` (collected LAMMPS side)
    and ``numbers.json`` (``parity.<label>.{passed, max_dF_eVA, max_dE_eV_atom, n_frames}`` plus
    ``md.parity.<label>.*`` aliases).
    """
    executor = executor if executor is not None else ctx.executor
    frames_path = Path(frames)
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames file not found: {frames_path}")
    ctx.add_input(frames_path, "frames")
    frame_list = read_frames(frames_path)
    if not frame_list:
        raise ValueError(f"no frames in {frames_path}")
    model_path = Path(model)
    prov = model_provenance(model_path)
    label = label or prov["label"]
    seed = ctx.seed if ctx.seed is not None else 0
    base = Path(prov["model_path"])
    if base.is_file():
        ctx.add_input(base, "model")
    ctx.log(
        engine="lammps-vs-ase", n_frames=len(frame_list), head=head, model_sha256=prov["sha256"],
        model_label=label, e0_source=prov["e0_source"], energy_scale=prov["energy_scale"],
        tol_f_eVA=tol_f, tol_e_eV_atom=tol_e, gate_frames=GATE_FRAMES,
        timestep_fs=None, thermostat=None,
    )  # fmt: skip

    lammps_values: dict[str, Any] | None = None
    source = ""
    fetched = ctx.out_dir / lammps.JOB_SUBDIR
    if lammps_json is not None:
        ctx.add_input(Path(lammps_json), "json")
        lammps_values = load_lammps_json(lammps_json)
        source = str(lammps_json)
    elif ctx.resume and (fetched / FRAMES_SUBDIR).is_dir():
        lammps_values = load_lammps_json(collect_parity(fetched, ctx.out_dir / PARITY_JSON))
        ctx.add_output(ctx.out_dir / PARITY_JSON, "json")
        source = str(fetched)
        ctx.log(resumed=True)
    else:
        lammps_model, _ = lammps.resolve_model(model_path, ctx)
        if executor is None:
            raise ValueError("no executor on the run context (pass --executor local|slurm)")
        gpu_flag = lammps.use_gpu(executor, cfg, gpu)
        cmd: str | None
        try:
            cmd = lammps.lammps_command(cfg, gpu=gpu_flag)
            if isinstance(executor, LocalExecutor):
                lammps.check_local_command(cmd)
        except FileNotFoundError as exc:
            cmd = None
            reason = str(exc)
        job_name = f"md-parity-{ctx.run_id}"
        job_dir = (
            lammps.job_dir_for(executor, job_name, ctx) if cmd else ctx.out_dir / "parity_inputs"
        )
        ids = parity_lammps_inputs(
            frame_list, f"../../{lammps_model.name}", job_dir / FRAMES_SUBDIR
        )
        job_dir.mkdir(parents=True, exist_ok=True)
        if not (job_dir / lammps_model.name).is_file():
            shutil.copy2(lammps_model, job_dir / lammps_model.name)
        ctx.log(job_dir=str(job_dir), units=ids, gpu=gpu_flag, lammps_command=cmd)
        if ctx.dry_run:
            return stage_result(ctx, "partial", {"planned": len(ids), "job_dir": str(job_dir)})
        if cmd is None:
            return stage_result(
                ctx, "partial",
                {"staged": len(ids), "job_dir": str(job_dir),
                 "note": f"{reason}; run the inputs and pass --lammps-json <parity_lammps.json>"},
            )  # fmt: skip
        spec = JobSpec(
            name=job_name,
            script=parity_unit_script(cmd),
            units=["parity"],
            resources={
                "template": lammps.SLURM_TEMPLATE,
                "gpu": gpu_flag,
                "gpus": 1 if gpu_flag else 0,
                "partition": cfg.cluster.partition_gpu if gpu_flag else cfg.cluster.partition_cpu,
                "lammps_cmd": cfg.cluster.lammps_cmd,
                "time": "00:30:00",
            },
            env={"LAMMPS_CMD": cmd},
        )
        (ctx.out_dir / "spec.json").write_text(
            spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        handle = executor.submit(spec)
        (ctx.out_dir / lammps.HANDLE_JSON).write_text(
            handle.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        ctx.add_output(ctx.out_dir / lammps.HANDLE_JSON, "json")
        ctx.log(job_ids=list(handle.job_ids), job_workdir=handle.workdir)
        if not wait:
            return stage_result(
                ctx, "partial",
                {"submitted": len(ids), "job_ids": list(handle.job_ids), "waited": 0,
                 "note": "rerun with --resume after `b20mlip cluster sync`"},
            )  # fmt: skip
        info = executor.wait(handle)
        if info is not None:
            ctx.slurm = info
        executor.fetch(handle, fetched)
        collected = collect_parity(fetched, ctx.out_dir / PARITY_JSON)
        ctx.add_output(ctx.out_dir / PARITY_JSON, "json")
        lammps_values = load_lammps_json(collected)
        source = str(fetched)
        if "@missing" in collected:
            ctx.log(lammps_missing=collected["@missing"])

    verdict = check(
        model_path if base.is_file() else None, frame_list, lammps_values, tol_f, tol_e,
        head=head, calc=calc, cfg=cfg,
    )  # fmt: skip
    verdict["lammps_source"] = source
    verdict["model_sha256"] = prov["sha256"]
    verdict["run_id"] = ctx.run_id
    result_path = ctx.out_dir / RESULT_JSON
    result_path.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    ctx.add_output(result_path, "json")
    numbers = _numbers(verdict, prov, head, seed, label)
    write_numbers(ctx, numbers)
    ctx.log(
        parity_passed=verdict["passed"], max_dF_eVA=verdict["max_dF_eVA"],
        max_dE_eV_atom=verdict["max_dE_eV_atom"], n_compared=verdict["n_frames"],
        n_missing=verdict["n_missing"], enough_frames=verdict["enough_frames"],
        numbers_keys=sorted(k for k in numbers if not k.endswith("@meta")),
    )  # fmt: skip
    return stage_result(
        ctx, "ok",
        {"parity_passed": verdict["passed"], "max_dF_eVA": verdict["max_dF_eVA"],
         "max_dE_eV_atom": verdict["max_dE_eV_atom"], "n_frames": verdict["n_frames"],
         "n_missing": verdict["n_missing"], "enough_frames": verdict["enough_frames"],
         "model_label": label, "result": relpath(result_path, ctx)},
    )  # fmt: skip


__all__ = [
    "DEFAULT_TOL_E",
    "DEFAULT_TOL_F",
    "FORCES_DUMP",
    "FRAMES_SUBDIR",
    "GATE_FRAMES",
    "PARITY_INPUT",
    "PARITY_JSON",
    "RESULT_JSON",
    "ase_reference",
    "check",
    "collect_parity",
    "compare",
    "load_lammps_json",
    "parity_lammps_inputs",
    "parity_unit_script",
    "parse_forces_dump",
    "parse_single_point_energy",
    "run",
]
