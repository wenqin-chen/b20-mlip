"""LAMMPS MD with ``pair_style mace`` (CONTRACTS.md row 10): input rendering, job submission
through the executors, thermo-log parsing and the ``md lammps`` stage.

No LAMMPS runs on the development machine (the ML-MACE package needs the ACEsuit/lammps ``mace``
branch built by ``cluster bootstrap``): everything here is deterministic plumbing that the test
suite drives with a fake ``lammps_cmd`` and the real thermo log ``tests/fixtures/golden/
lammps_thermo.log`` (brew LAMMPS 22 Jul 2025, the rendered template with ``lj/cut`` in place of
``pair_style mace``).

Input (``templates/lammps/in.mace.j2``, ML-MACE syntax verified against
https://mace-docs.readthedocs.io/en/latest/guide/lammps.html on 2026-09-18)::

    units metal / atom_style atomic / atom_modify map yes / newton on / boundary p p p
    read_data data.lmp                     # ase.io.write(format="lammps-data", specorder=elements)
    pair_style mace no_domain_decomposition
    pair_coeff * * <model>-lammps.pt <elements in type order>
    velocity all create T seed ; fix 1 all npt temp T T $(100*dt) iso 0 0 $(1000*dt)  (nvt|nve)
    thermo_style custom step temp pe ke etotal press vol lx ; thermo N ; run equil ; dump ; run

The first line of the input is a ``# b20mlip-md key=value ...`` header (compound, ensemble, T,
timestep_fs, equil_steps, steps, reps, natoms_per_cell, natoms, head, model_label, model_sha256,
seed) that LAMMPS echoes into ``log.lammps``, so :func:`parse_thermo` can rebuild the
:class:`MDResult` from the log alone; explicit keyword arguments override the header.

Job layout (both executors; the sampling / agent tiers reuse it): the job directory
(``LocalExecutor.workdir(name)`` = ``<runs_dir>/local/<name>``, or ``SlurmExecutor.staging_root /
name`` rsynced to ``<scratch>/b20-mlip/jobs/<name>``) holds ``data.lmp``, ``in.mace``, the
``-lammps.pt`` copy and ``md_job.json``; the unit script runs ``<lammps_cmd> [kokkos flags] -in
in.mace -log log.lammps`` there (``templates/slurm/lammps.sbatch.j2`` on the cluster, GPU
partition, one unit); ``log.lammps``, ``dump.lammpstrj`` and ``final.data`` are fetched into
``<run dir>/job/``. Kokkos GPU flags: ``-k on g 1 -sf kk -pk kokkos newton on neigh half``.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import jinja2
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from b20mlip.config import Settings, repo_root
from b20mlip.executors import JobHandle, JobSpec, LocalExecutor, SlurmExecutor
from b20mlip.md import ase_md
from b20mlip.md.common import (
    EXPERIMENTAL_A_A,
    block_ci95,
    elements_of_atoms,
    fmt_T,
    lammps_path_for,
    lattice_parameter,
    load_structure,
    make_supercell,
    md_numbers,
    model_provenance,
    relpath,
    stage_result,
    write_numbers,
    write_result,
)
from b20mlip.models import Head, MDResult, StageResult
from b20mlip.provenance import RunContext, sha256_file

log = logging.getLogger("b20mlip.md")

TEMPLATE_DIR = "lammps"
TEMPLATE_NAME = "in.mace.j2"
SLURM_TEMPLATE = "lammps"
INPUT_NAME = "in.mace"
DATA_NAME = "data.lmp"
LOG_NAME = "log.lammps"
DUMP_NAME = "dump.lammpstrj"
JOB_JSON = "md_job.json"
HANDLE_JSON = "handle.json"
JOB_SUBDIR = "job"
KOKKOS_FLAGS = "-k on g 1 -sf kk -pk kokkos newton on neigh half"
HEADER_TAG = "# b20mlip-md"
HEADER_STRING_KEYS: frozenset[str] = frozenset(
    {"engine", "compound", "ensemble", "head", "model_label", "model_sha256"}
)
NUMBER_RE = re.compile(r"^[-+]?(\d+\.?\d*(e[-+]?\d+)?|\.\d+(e[-+]?\d+)?|inf|nan)$", re.I)
LOOP_RE = re.compile(
    r"^Loop time of\s+(?P<seconds>[\d.eE+-]+)\s+on\s+(?P<procs>\d+)\s+procs\s+for\s+"
    r"(?P<steps>\d+)\s+steps\s+with\s+(?P<natoms>\d+)\s+atoms"
)
TIMESTEP_RE = re.compile(r"^\s*timestep\s+([\d.eE+-]+)")
# thermo header names LAMMPS prints for the keywords of `thermo_style custom`
THERMO_ALIASES: dict[str, str] = {
    "poteng": "pe",
    "kineng": "ke",
    "toteng": "etotal",
    "volume": "vol",
}
MIN_VDOS_FRAMES = ase_md.MIN_VDOS_FRAMES


# --- rendering ------------------------------------------------------------------------------------


def jinja_env(template_dir: str | Path | None = None) -> jinja2.Environment:
    root = (
        Path(template_dir) if template_dir is not None else repo_root() / "templates" / TEMPLATE_DIR
    )
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(root)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # a LAMMPS input, not HTML
    )


def steps_for(cfg: Settings, ps: float) -> int:
    return ase_md.n_steps(ps, cfg.md.timestep_fs)


def render_input(
    cfg: Settings,
    model_lammps: str | Path,
    elements: Sequence[str],
    T: float,
    ps: float,
    natoms: int,
    *,
    ensemble: str = "npt",
    natoms_per_cell: int = 8,
    seed: int = 0,
    compound: str = "",
    head: str = "Default",
    model_label: str = "",
    model_sha256: str = "",
    equil_ps: float | None = None,
    template_dir: str | Path | None = None,
) -> str:
    """The ``in.mace`` text (see the module docstring). ``natoms`` is the supercell size; the
    header records ``reps`` from ``natoms_per_cell`` so the parser can report the lattice
    parameter of the input cell. ``model_lammps`` is written as given (a bare file name for a
    staged job)."""
    ensemble = ase_md.check_ensemble(ensemble)
    if not elements:
        raise ValueError("elements must list the LAMMPS types in order")
    reps = max(1, int(round((max(1, int(natoms)) / max(1, int(natoms_per_cell))) ** (1.0 / 3.0))))
    equil = cfg.md.equil_ps if equil_ps is None else float(equil_ps)
    context = {
        "compound": compound,
        "ensemble": ensemble,
        "T": float(T),
        "timestep_fs": float(cfg.md.timestep_fs),
        "timestep_ps": float(cfg.md.timestep_fs) / 1000.0,
        "equil_steps": int(round(equil * 1000.0 / cfg.md.timestep_fs)) if equil > 0 else 0,
        "steps": steps_for(cfg, ps),
        "reps": reps,
        "natoms_per_cell": int(natoms_per_cell),
        "natoms": int(natoms),
        "head": head,
        "model_label": model_label,
        "model_sha256": model_sha256,
        "seed": int(seed) if int(seed) > 0 else 12345,  # LAMMPS wants a positive seed
        "model_lammps": str(model_lammps),
        "elements": list(elements),
        "thermo_every": int(cfg.md.thermo_every_steps),
        "dump_every": int(cfg.md.dump_every_steps),
    }
    return jinja_env(template_dir).get_template(TEMPLATE_NAME).render(**context)


def write_data(atoms: Atoms, path: str | Path, elements: Sequence[str]) -> Path:
    """``data.lmp`` in metal units, ``atom_style atomic``, types in ``elements`` order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ase_write(
        str(p), atoms, format="lammps-data", specorder=list(elements), masses=True,
        atom_style="atomic", units="metal",
    )  # fmt: skip
    return p


def lammps_command(cfg: Settings, *, gpu: bool) -> str:
    """``cluster.lammps_cmd`` plus the Kokkos GPU flags; raises when it is unset."""
    cmd = cfg.cluster.lammps_cmd
    if not cmd or not str(cmd).strip():
        raise FileNotFoundError(
            "lammps_cmd not found: cluster.lammps_cmd is unset (run `b20mlip cluster bootstrap`, "
            "or --set cluster.lammps_cmd=/path/to/lmp)"
        )
    return f"{cmd} {KOKKOS_FLAGS}" if gpu else str(cmd)


def check_local_command(cmd: str) -> str:
    """The executable of ``cmd`` must exist on PATH for a local run."""
    exe = shlex.split(cmd)[0]
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise FileNotFoundError(f"lammps_cmd not found on this machine: {exe!r} (from {cmd!r})")
    return exe


def unit_script(cmd: str, input_name: str = INPUT_NAME) -> str:
    """The per-unit bash snippet: run LAMMPS in the job directory."""
    return f'cd "$B20_WORKDIR"\n{cmd} -in {input_name} -log {LOG_NAME}\n'


def job_dir_for(executor: Any, job_name: str, ctx: RunContext) -> Path:
    """Where the executor expects the job's input files (see the module docstring)."""
    if isinstance(executor, SlurmExecutor):
        return Path(executor.staging_root) / job_name
    if isinstance(executor, LocalExecutor):
        return executor.workdir(job_name)
    return ctx.out_dir / JOB_SUBDIR


def use_gpu(executor: Any, cfg: Settings, gpu: bool | None) -> bool:
    if gpu is not None:
        return bool(gpu)
    return isinstance(executor, SlurmExecutor) and bool(cfg.cluster.partition_gpu)


# --- parsing --------------------------------------------------------------------------------------


def parse_header(text: str) -> dict[str, Any]:
    """``key=value`` tokens of the ``# b20mlip-md`` header (numbers converted)."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith(HEADER_TAG):
            continue
        for token in line[len(HEADER_TAG) :].split():
            key, sep, value = token.partition("=")
            if not sep:
                continue
            if key in HEADER_STRING_KEYS:
                out[key] = value
                continue
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
        break
    return out


def parse_thermo_blocks(text: str) -> list[dict[str, Any]]:
    """Every thermo block of a LAMMPS log: ``{"columns", "rows" (ndarray), "loop"}``.

    A block starts at the ``Step ...`` header line and ends at ``Loop time of ...`` (whose
    ``steps``, ``natoms``, ``procs`` and ``seconds`` go into ``loop``); lines that are not pure
    numbers (warnings) are skipped. An unfinished block (crashed run) has ``loop=None``.
    """
    blocks: list[dict[str, Any]] = []
    columns: list[str] | None = None
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if columns is None:
            if line.startswith("Step ") or line == "Step":
                columns = [c.lower() for c in line.split()]
                rows = []
            continue
        m = LOOP_RE.match(line)
        if m:
            blocks.append(
                {
                    "columns": columns,
                    "rows": np.array(rows, dtype=float).reshape(len(rows), len(columns)),
                    "loop": {
                        "seconds": float(m.group("seconds")),
                        "procs": int(m.group("procs")),
                        "steps": int(m.group("steps")),
                        "natoms": int(m.group("natoms")),
                    },
                }
            )
            columns = None
            continue
        parts = line.split()
        if len(parts) == len(columns) and all(NUMBER_RE.match(p) for p in parts):
            rows.append([float(p) for p in parts])
    if columns is not None:
        blocks.append(
            {
                "columns": columns,
                "rows": np.array(rows, dtype=float).reshape(len(rows), len(columns)),
                "loop": None,
            }
        )
    return blocks


def thermo_table(blocks: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Concatenate the blocks (same columns) into ``{column: values}``; a continued run repeats
    the last step of the previous block, which is dropped. LAMMPS header names (``PotEng``,
    ``KinEng``, ``TotEng``, ``Volume``) are also available under their keywords (``pe``, ``ke``,
    ``etotal``, ``vol``)."""
    if not blocks:
        raise ValueError("no thermo block (Step ... Loop time) in the log")
    columns = list(blocks[0]["columns"])
    if "step" not in columns:
        raise ValueError("thermo output has no Step column")
    parts: list[np.ndarray] = []
    last_step: float | None = None
    for block in blocks:
        if list(block["columns"]) != columns:
            raise ValueError(
                f"thermo columns changed between runs: {columns} -> {block['columns']}"
            )
        rows = np.asarray(block["rows"], dtype=float)
        if rows.size == 0:
            continue
        step_col = columns.index("step")
        if last_step is not None and rows[0, step_col] == last_step:
            rows = rows[1:]
        if rows.size:
            parts.append(rows)
            last_step = rows[-1, step_col]
    if not parts:
        raise ValueError("thermo blocks hold no rows")
    table = np.vstack(parts)
    out = {col: table[:, i] for i, col in enumerate(columns)}
    for name, alias in THERMO_ALIASES.items():  # both spellings are available
        if name in out and alias not in out:
            out[alias] = out[name]
        elif alias in out and name not in out:
            out[name] = out[alias]
    return out


def parse_timestep_ps(text: str) -> float | None:
    for line in text.splitlines():
        m = TIMESTEP_RE.match(line)
        if m:
            return float(m.group(1))
    return None


def analyze_thermo(text: str, **meta: Any) -> dict[str, Any]:
    """Everything :func:`parse_thermo` derives from a log, as a dict (``MDResult`` fields plus
    ``n_rows``, ``n_production``, ``T_mean_K``, ``a_ci95``, ``drift_ci95``, ``natoms_log``).

    Keyword overrides: ``ensemble``, ``T``, ``timestep_fs``, ``equil_ps``, ``reps``,
    ``natoms_per_cell``, ``compound``, ``head``, ``model_sha256``, ``run_id``, ``traj_sha256``,
    ``vdos_path``, ``rdf_path``, ``model_label``. Header values fill what is not given; the
    ``timestep`` command echoed by LAMMPS is the last resort for the time step.
    """
    header = parse_header(text)
    blocks = parse_thermo_blocks(text)
    table = thermo_table(blocks)
    loops = [b["loop"] for b in blocks if b["loop"]]
    natoms_log = int(loops[-1]["natoms"]) if loops else None
    natoms = int(meta.get("natoms") or natoms_log or header.get("natoms") or 0)
    if natoms <= 0:
        raise ValueError("cannot determine the atom count (no `Loop time ... N atoms` line)")
    ensemble = meta.get("ensemble") or header.get("ensemble")
    if ensemble not in ase_md.ENSEMBLES:
        raise ValueError(f"unknown ensemble {ensemble!r}: pass ensemble= or keep the input header")
    timestep_fs = meta.get("timestep_fs") or header.get("timestep_fs")
    if timestep_fs is None:
        ts_ps = parse_timestep_ps(text)
        timestep_fs = ts_ps * 1000.0 if ts_ps is not None else None
    if timestep_fs is None:
        raise ValueError("cannot determine the time step: pass timestep_fs=")
    timestep_fs = float(timestep_fs)
    reps = int(meta.get("reps") or header.get("reps") or 1)
    if meta.get("equil_ps") is not None:
        equil_ps = float(meta["equil_ps"])
    else:
        equil_ps = float(header.get("equil_steps", 0)) * timestep_fs / 1000.0
    T_raw = meta.get("T")
    T = float(T_raw if T_raw is not None else header.get("T", 0.0))
    compound = str(meta.get("compound") or header.get("compound") or "")
    head = str(meta.get("head") or header.get("head") or "Default")
    model_sha = str(meta.get("model_sha256") or header.get("model_sha256") or "")

    steps = table["step"]
    time_ps = steps * timestep_fs / 1000.0
    mask = time_ps >= equil_ps - 1e-12
    if mask.sum() < 2:
        mask = np.ones_like(mask, dtype=bool)
        window = "all samples (run shorter than equil_ps)"
    else:
        window = "production (equil_ps excluded)"
    drift = drift_se = None
    if ensemble == "nve" and "toteng" in table and mask.sum() >= 2 and np.ptp(time_ps[mask]) > 0:
        drift, drift_se = ase_md.energy_drift(time_ps[mask], table["toteng"][mask] / natoms)
    a_mean = a_std = None
    a_ci = None
    if ensemble == "npt":
        if "lx" in table:
            a_series = table["lx"][mask] / reps
        elif "volume" in table:
            a_series = np.array([lattice_parameter(v, reps) for v in table["volume"][mask]])
        else:
            raise ValueError("NPT log without lx or volume columns")
        a_mean = float(a_series.mean())
        a_std = float(a_series.std(ddof=1)) if a_series.size > 1 else 0.0
        a_ci = block_ci95(a_series)
    total_steps = int(steps[-1] - steps[0])
    return {
        "engine": "lammps",
        "model_sha256": model_sha,
        "head": head,
        "compound": compound,
        "natoms": natoms,
        "natoms_log": natoms_log,
        "ensemble": ensemble,
        "temperature_K": T,
        "pressure_GPa": 0.0 if ensemble == "npt" else None,
        "timestep_fs": timestep_fs,
        "steps": total_steps,
        "drift_meV_atom_ps": drift,
        "drift_ci95": (
            (drift - 1.96 * drift_se, drift + 1.96 * drift_se)
            if drift is not None and drift_se is not None
            else None
        ),
        "a_mean_A": a_mean,
        "a_std_A": a_std,
        "a_ci95": a_ci,
        "alpha_per_K": None,
        "n_rows": int(steps.size),
        "n_production": int(mask.sum()),
        "production_window": window,
        "equil_ps": equil_ps,
        "reps": reps,
        "T_mean_K": float(table["temp"][mask].mean()) if "temp" in table else None,
        "E_pot_mean_eV_atom": float(table["pe"][mask].mean()) / natoms if "pe" in table else None,
        "n_runs": len(blocks),
        "loop_seconds": float(sum(lp["seconds"] for lp in loops)) if loops else None,
        "header": header,
        "model_label": meta.get("model_label") or header.get("model_label"),
    }


def parse_thermo(log: str | Path, **meta: Any) -> MDResult:
    """Parse ``log.lammps`` into an :class:`MDResult` (see :func:`analyze_thermo` for the keyword
    overrides). ``run_id``, ``traj_sha256``, ``vdos_path`` and ``rdf_path`` come from ``meta``
    (defaults: ``""``/``None``); ``model_sha256`` falls back to the header, then ``""``."""
    p = Path(log)
    text = p.read_text(encoding="utf-8", errors="replace")
    info = analyze_thermo(text, **meta)
    return MDResult(
        engine="lammps",
        model_sha256=info["model_sha256"],
        head=cast(Head, info["head"]),
        compound=info["compound"],
        natoms=info["natoms"],
        ensemble=cast(Any, info["ensemble"]),
        temperature_K=info["temperature_K"],
        pressure_GPa=info["pressure_GPa"],
        timestep_fs=info["timestep_fs"],
        steps=info["steps"],
        drift_meV_atom_ps=info["drift_meV_atom_ps"],
        a_mean_A=info["a_mean_A"],
        a_std_A=info["a_std_A"],
        alpha_per_K=None,
        traj_sha256=str(meta.get("traj_sha256") or ""),
        vdos_path=meta.get("vdos_path"),
        rdf_path=meta.get("rdf_path"),
        run_id=str(meta.get("run_id") or ""),
    )


def read_dump(path: str | Path, elements: Sequence[str] | None = None) -> list[Atoms]:
    """Frames of a ``dump ... id type x y z vx vy vz`` file (metal units -> ASE units)."""
    images = ase_read(
        str(path), format="lammps-dump-text", index=":", specorder=list(elements or []) or None
    )
    return list(images) if isinstance(images, list) else [images]


def dump_analysis(
    dump: Path, elements: Sequence[str], cfg: Settings, ctx: RunContext, tag: str, equil_steps: int
) -> tuple[str | None, str | None, int]:
    """VDOS + RDF JSON files from a dump (production frames: timestep >= equil_steps)."""
    images = read_dump(dump, elements)
    prod = [img for img in images if int(img.info.get("timestep", 0)) >= equil_steps] or images
    vdos_out = rdf_out = None
    if len(prod) >= MIN_VDOS_FRAMES:
        ts = [int(img.info.get("timestep", i)) for i, img in enumerate(prod)]
        dt_steps = max(1, int(np.median(np.diff(ts)))) if len(ts) > 1 else 1
        freq, dos = ase_md.vdos(prod, dt_steps * cfg.md.timestep_fs)
        path = ctx.out_dir / f"vdos_{tag}.json"
        payload = {
            "freq_meV": freq.tolist(),
            "dos": dos.tolist(),
            "dt_fs": dt_steps * cfg.md.timestep_fs,
            "n_frames": len(prod),
            "natoms": len(prod[0]),
            "mass_weighted": True,
            "window": "production",
            "source": dump.name,
        }
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        ctx.add_output(path, "json")
        vdos_out = str(path)
    if prod:
        r, g, rmax_used = ase_md.rdf(prod, cfg.md.rdf_rmax_A, cfg.md.rdf_nbins)
        path = ctx.out_dir / f"rdf_{tag}.json"
        path.write_text(
            json.dumps(
                {"r_A": r.tolist(), "g": g.tolist(), "rmax_A": rmax_used, "nbins": cfg.md.rdf_nbins,
                 "n_frames": len(prod), "window": "production", "source": dump.name},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )  # fmt: skip
        ctx.add_output(path, "json")
        rdf_out = str(path)
    return vdos_out, rdf_out, len(images)


# --- the stage ------------------------------------------------------------------------------------


def resolve_model(model: str | Path, ctx: RunContext) -> tuple[Path, dict[str, Any]]:
    """``(-lammps.pt path, provenance)`` for a ``.model`` (needs its export) or a ``-lammps.pt``."""
    given = Path(model)
    lammps = lammps_path_for(given)
    if not lammps.is_file():
        raise FileNotFoundError(
            f"no LAMMPS export {lammps} for {given}; run `b20mlip export --model {given}` first"
        )
    prov = model_provenance(given)
    if prov["sha256"] is None:  # only the -lammps.pt exists: identify the model by it
        prov["sha256"] = sha256_file(lammps)
    base = Path(prov["model_path"])
    if base.is_file():
        ctx.add_input(base, "model")
    ctx.add_input(lammps, "model")
    return lammps, prov


def _tail(path: Path, n: int = 20) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def run(
    cfg: Settings,
    ctx: RunContext,
    model: str | Path,
    compound: str,
    T: float,
    ps: float,
    natoms: int | None = None,
    executor: Any | None = None,
    *,
    ensemble: str = "npt",
    structure: str | Path | None = None,
    head: str = "Default",
    label: str | None = None,
    wait: bool = True,
    gpu: bool | None = None,
    a_exp_A: float | None = None,
) -> StageResult:
    """Stage ``md.lammps`` (``b20mlip md lammps``): stage the job, submit, fetch, parse.

    Fails cleanly (``lammps_cmd not found``) when ``cluster.lammps_cmd`` is unset or, with the
    local executor, not on PATH. ``ctx.dry_run`` stages the inputs only (``status=partial``);
    ``wait=False`` submits and returns ``partial`` (``handle.json`` in the run directory; rerun
    with ``--resume`` once ``cluster sync`` brought ``job/log.lammps`` back). Outputs:
    ``job/{in.mace,data.lmp,log.lammps,dump.lammpstrj}``, ``md_result_<ens>_<T>K.json``,
    ``vdos_/rdf_<ens>_<T>K.json`` (from the dump) and ``numbers.json`` (``md.lammps.…`` keys as
    in :mod:`b20mlip.md.ase_md`).
    """
    ensemble = ase_md.check_ensemble(ensemble)
    executor = executor if executor is not None else ctx.executor
    if executor is None:
        raise ValueError("no executor on the run context (pass --executor local|slurm)")
    lammps_model, prov = resolve_model(model, ctx)
    label = label or prov["label"]
    seed = ctx.seed if ctx.seed is not None else 0
    if structure is not None:
        ctx.add_input(Path(structure), "frames")
    atoms = load_structure(cfg, compound, structure)
    target = natoms if natoms is not None else cfg.md.natoms
    cell, reps = make_supercell(atoms, target)
    elements = elements_of_atoms(cell)
    steps = steps_for(cfg, ps)
    equil_steps = int(round(cfg.md.equil_ps * 1000.0 / cfg.md.timestep_fs))
    tag = f"{ensemble}_{fmt_T(T)}K"
    gpu_flag = use_gpu(executor, cfg, gpu)
    job_name = f"md-lammps-{compound}-{ctx.run_id}"
    fetched = ctx.out_dir / JOB_SUBDIR
    log_path = fetched / LOG_NAME

    if not (ctx.resume and log_path.is_file()):
        cmd = lammps_command(cfg, gpu=gpu_flag)
        if isinstance(executor, LocalExecutor):
            check_local_command(cmd)
        job_dir = job_dir_for(executor, job_name, ctx)
        job_dir.mkdir(parents=True, exist_ok=True)
        write_data(cell, job_dir / DATA_NAME, elements)
        text = render_input(
            cfg, lammps_model.name, elements, T, ps, len(cell), ensemble=ensemble,
            natoms_per_cell=len(atoms), seed=seed, compound=compound, head=head,
            model_label=label, model_sha256=prov["sha256"] or "",
        )  # fmt: skip
        (job_dir / INPUT_NAME).write_text(text, encoding="utf-8")
        shutil.copy2(lammps_model, job_dir / lammps_model.name)
        job_meta = {
            "job_name": job_name,
            "job_dir": str(job_dir),
            "command": cmd,
            "gpu": gpu_flag,
            "compound": compound,
            "ensemble": ensemble,
            "T": float(T),
            "ps": ps,
            "steps": steps,
            "equil_steps": equil_steps,
            "timestep_fs": cfg.md.timestep_fs,
            "natoms": len(cell),
            "reps": reps,
            "natoms_per_cell": len(atoms),
            "elements": elements,
            "model": str(prov["model_path"]),
            "model_sha256": prov["sha256"],
            "lammps_model": lammps_model.name,
            "lammps_sha256": prov["lammps_sha256"] or sha256_file(lammps_model),
            "head": head,
            "model_label": label,
            "seed": seed,
            "dry_run": bool(ctx.dry_run),
        }
        (job_dir / JOB_JSON).write_text(json.dumps(job_meta, indent=2) + "\n", encoding="utf-8")
        ctx.log(
            engine="lammps", timestep_fs=cfg.md.timestep_fs, ensemble=ensemble,
            thermostat=f"fix {ensemble} Tdamp 100*dt Pdamp 1000*dt (Nose-Hoover, iso 0 bar)",
            lammps_command=cmd, gpu=gpu_flag, job_dir=str(job_dir), natoms=len(cell), reps=reps,
            steps=steps, equil_steps=equil_steps, model_sha256=prov["sha256"], head=head,
            model_label=label, e0_source=prov["e0_source"], energy_scale=prov["energy_scale"],
            lammps_sha256=job_meta["lammps_sha256"], pair_style="mace no_domain_decomposition",
        )  # fmt: skip
        if ctx.dry_run:
            return stage_result(
                ctx, "partial", {"planned": 1, "steps": steps, "job_dir": str(job_dir)}
            )
        spec = JobSpec(
            name=job_name,
            script=unit_script(cmd),
            units=[ctx.run_id],
            resources={
                "template": SLURM_TEMPLATE,
                "input": INPUT_NAME,
                "gpu": gpu_flag,
                "gpus": 1 if gpu_flag else 0,
                "partition": cfg.cluster.partition_gpu if gpu_flag else cfg.cluster.partition_cpu,
                "lammps_cmd": cfg.cluster.lammps_cmd,
            },
            env={"LAMMPS_CMD": cmd},
        )
        (ctx.out_dir / "spec.json").write_text(
            spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        handle = executor.submit(spec)
        (ctx.out_dir / HANDLE_JSON).write_text(
            handle.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        ctx.add_output(ctx.out_dir / HANDLE_JSON, "json")
        ctx.log(job_ids=list(handle.job_ids), job_workdir=handle.workdir)
        if not wait:
            return stage_result(
                ctx, "partial",
                {"submitted": 1, "job_ids": list(handle.job_ids), "waited": 0,
                 "note": "rerun with --resume after `b20mlip cluster sync`"},
            )  # fmt: skip
        info = executor.wait(handle)
        if info is not None:
            ctx.slurm = info
        executor.fetch(handle, fetched)
    else:
        handle_path = ctx.out_dir / HANDLE_JSON
        handle = (
            JobHandle.model_validate_json(handle_path.read_text(encoding="utf-8"))
            if handle_path.is_file()
            else None
        )
        ctx.log(resumed=True, job_ids=list(handle.job_ids) if handle else [])

    failed = sorted((fetched / "units").glob("*.failed")) if (fetched / "units").is_dir() else []
    if failed or not log_path.is_file():
        unit_logs = sorted((fetched / "logs").glob("*.log")) if (fetched / "logs").is_dir() else []
        detail = _tail(unit_logs[-1]) if unit_logs else _tail(log_path)
        raise RuntimeError(
            f"LAMMPS job {job_name} {'failed' if failed else 'left no ' + LOG_NAME}:\n{detail}"
        )
    for name in (INPUT_NAME, DATA_NAME):
        if (fetched / name).is_file():
            ctx.add_output(fetched / name, "other")
    ctx.add_output(log_path, "log")
    dump = fetched / DUMP_NAME
    traj_sha = sha256_file(dump) if dump.is_file() else sha256_file(log_path)
    vdos_out = rdf_out = None
    n_dump = 0
    if dump.is_file():
        ctx.add_output(dump, "traj")
        try:
            vdos_out, rdf_out, n_dump = dump_analysis(dump, elements, cfg, ctx, tag, equil_steps)
        except Exception as exc:  # noqa: BLE001 - analysis is best effort, the log is the result
            ctx.log(dump_analysis_error=repr(exc))
            log.warning("dump analysis failed for %s: %r", dump, exc)
    info_d = analyze_thermo(
        log_path.read_text(encoding="utf-8", errors="replace"),
        ensemble=ensemble, T=T, timestep_fs=cfg.md.timestep_fs, equil_ps=cfg.md.equil_ps,
        reps=reps, natoms_per_cell=len(atoms), compound=compound, head=head,
        model_sha256=prov["sha256"], model_label=label,
    )  # fmt: skip
    result = MDResult(
        engine="lammps",
        model_sha256=prov["sha256"] or "",
        head=cast(Head, head),
        compound=compound,
        natoms=info_d["natoms"],
        ensemble=cast(Any, ensemble),
        temperature_K=float(T),
        pressure_GPa=info_d["pressure_GPa"],
        timestep_fs=cfg.md.timestep_fs,
        steps=info_d["steps"],
        drift_meV_atom_ps=info_d["drift_meV_atom_ps"],
        a_mean_A=info_d["a_mean_A"],
        a_std_A=info_d["a_std_A"],
        alpha_per_K=None,
        traj_sha256=traj_sha,
        vdos_path=vdos_out,
        rdf_path=rdf_out,
        run_id=ctx.run_id,
    )
    if info_d["natoms_log"] is not None and info_d["natoms_log"] != len(cell):
        ctx.log(natoms_mismatch={"input": len(cell), "log": info_d["natoms_log"]})
        log.warning(
            "LAMMPS log reports %d atoms, the staged cell has %d", info_d["natoms_log"], len(cell)
        )
    write_result(ctx, result, f"md_result_{tag}.json")
    stat_keys = ("n_production", "a_ci95", "drift_ci95", "T_mean_K")
    stats = {float(T): {k: info_d[k] for k in stat_keys}}
    numbers = md_numbers([result], stats, provenance=prov, label=label, seed=seed, a_exp_A=a_exp_A)
    write_numbers(ctx, numbers)
    ctx.log(
        result=result.model_dump(mode="json"),
        stats={**stats[float(T)], "n_rows": info_d["n_rows"], "n_runs": info_d["n_runs"],
               "production_window": info_d["production_window"], "n_dump_frames": n_dump,
               "loop_seconds": info_d["loop_seconds"]},
        numbers_keys=sorted(k for k in numbers if not k.endswith("@meta")),
    )  # fmt: skip
    summary: dict[str, Any] = {
        "engine": "lammps",
        "compound": compound,
        "ensemble": ensemble,
        "T": float(T),
        "steps": result.steps,
        "natoms": result.natoms,
        "model_label": label,
        "T_mean_K": info_d["T_mean_K"],
        "n_production": info_d["n_production"],
        "log": relpath(log_path, ctx),
    }
    if result.drift_meV_atom_ps is not None:
        summary["drift_meV_atom_ps"] = result.drift_meV_atom_ps
    if result.a_mean_A is not None:
        summary["a_mean_A"] = result.a_mean_A
        if compound in EXPERIMENTAL_A_A or a_exp_A is not None:
            summary["a_exp_A"] = a_exp_A if a_exp_A is not None else EXPERIMENTAL_A_A[compound]
    return stage_result(ctx, "ok", summary)


__all__ = [
    "DATA_NAME",
    "DUMP_NAME",
    "HEADER_STRING_KEYS",
    "HEADER_TAG",
    "INPUT_NAME",
    "JOB_JSON",
    "KOKKOS_FLAGS",
    "LOG_NAME",
    "SLURM_TEMPLATE",
    "THERMO_ALIASES",
    "analyze_thermo",
    "check_local_command",
    "dump_analysis",
    "job_dir_for",
    "lammps_command",
    "parse_header",
    "parse_thermo",
    "parse_thermo_blocks",
    "parse_timestep_ps",
    "read_dump",
    "render_input",
    "resolve_model",
    "run",
    "thermo_table",
    "unit_script",
    "use_gpu",
    "write_data",
]
