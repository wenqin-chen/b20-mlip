"""Day-1 timing task (SPEC.md section 15; CONTRACTS.md row 15): ``b20mlip bench --out runs/bench/``.

Five measurements, one MACE model (default: the MPA-0 medium foundation file), CPU float64,
``cfg.compute.threads`` threads. Each is a named sub-task whose result is cached in
``<out>/bench.json`` the moment it finishes, so a re-run skips what is done (``bench.force``
re-measures) and a crash loses at most one sub-task:

* ``a`` **finetune** — one naive fine-tuning epoch per size class (``mace_run_train``,
  ``finetune.build_args(variant="naive", energy_scale="mp")``, ``--E0s foundation``): 80
  8-atom frames (the MPtrj B20 frames with their DFT labels; cycled and rattled copies beyond
  the source, and frames without ``mp``-scale labels, are labelled zero-shot by the same model)
  and 20 64-atom frames (2x2x2 supercells with zero-shot labels, ``label_source
  "mace_zero_shot"``, ``energy_scale "mp"``). The two classes run as two separate short
  trainings (``bench.finetune_epochs`` epochs, the last epoch is the measurement because epoch 0
  carries the warm-up); the epoch wall time is read from the timestamps of MACE's validation
  lines (``Initial:`` then ``Epoch k:``) so start-up is excluded, and the pure optimiser-step
  time from MACE's ``results/*_train.txt``. ``s_per_frame = epoch wall / n_train`` (the same
  convention as the 0.12 s/frame figure in docs/design/facts_2026-09-18.md).
* ``b`` **md** — NVT Langevin (``cfg.md`` constants) at 64 and 512 atoms, ``bench.md_steps``
  steps, the first ``bench.md_warmup_steps`` excluded -> s/step.
* ``c`` **relax** — FIRE + ``FrechetCellFilter`` (``evaluate.discovery.relax``, ``cfg.eval.fmax``,
  ``cfg.eval.max_steps``) on the first ``bench.n_relax`` ids of the WBM sample -> mean/median
  s and steps per structure, capped count.
* ``d`` **phonons** — ``phonons.harmonic.compute`` on the FeSi reference cell, 2x2x2, 0.03 A ->
  wall s and the number of displacements.
* ``e`` **fwbw** — one training-style batch (4 x 64-atom cells) forward + backward with forces
  in a *fresh* Python process -> s/batch and the child's peak RSS
  (``resource.getrusage(RUSAGE_SELF).ru_maxrss``, bytes on macOS, kB on Linux).

A sub-task that fails records ``{"error": ...}`` (never a number) and the stage ends with
``status="partial"``. ``derive_schedule`` re-derives the SPEC.md section 5 runtimes from the
measurements (formulas as strings); ``bench_report`` renders the markdown tables of
docs/BENCH.md. ``numbers.json`` (report-tier format, keys ``bench.<measurement>.<metric>``) is
written into the run directory; bench numbers are never README claims.

The root CLI takes only ``--out``; every other knob is ``--set bench.<key>=<value>``
(``bench.model``, ``bench.tiers=b,c``, ``bench.quick=true``, ``bench.force=true``, sizes).
``bench.quick`` scales everything down for the tiny CI model (8 + 5 frames, one epoch, 20 MD
steps, 2 relaxations, 1x1x1 phonons, one fwbw repeat).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import resource
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from b20mlip.config import Settings, repo_root
from b20mlip.data import wbm
from b20mlip.dft.structures import reference_frame
from b20mlip.evaluate import discovery
from b20mlip.io import frame_from_atoms, frame_to_atoms, read_frames
from b20mlip.md.ase_md import make_dynamics, thermostat_label
from b20mlip.md.common import make_supercell, model_provenance, stage_result, write_numbers
from b20mlip.models import Frame, StageResult, Status
from b20mlip.phonons import harmonic
from b20mlip.provenance import RunContext, sha256_file
from b20mlip.train import finetune

log = logging.getLogger("b20mlip.bench")

SCHEMA = "b20mlip.bench.v1"
BENCH_JSON = "bench.json"
PLAN_JSON = "plan.json"
FWBW_MARKER = "B20MLIP_FWBW_RESULT "
TIER_NAMES: dict[str, str] = {
    "a": "finetune",
    "b": "md",
    "c": "relax",
    "d": "phonons",
    "e": "fwbw",
}
TIER_LETTERS: dict[str, str] = {name: letter for letter, name in TIER_NAMES.items()}
ALL_TIERS = ",".join(TIER_NAMES)
SIZE_CLASSES: tuple[str, ...] = ("8atom", "64atom")
SUPERCELL_64 = 2  # 8-atom B20 cell x 2x2x2 = 64 atoms
RATTLE_8ATOM_A = 0.05  # rattle of the cycled 8-atom copies (beyond the source frames)
RATTLE_64ATOM_A = 0.02  # rattle of the supercells (so no two frames are identical)
FWBW_TIMEOUT_S = 900
# SPEC.md section 5 plan constants re-derived by derive_schedule (the plan, not configuration)
ROUND0_N8 = 600
ROUND0_N64 = 60
N_SEEDS = 3
MD_PS = 40.0
NPT512_PS = 100.0
PHONONDB_N = 103
# bench.quick: what the tiny CI model gets
QUICK_PARAMS: dict[str, Any] = {
    "n_frames_8atom": 8,
    "n_frames_64atom": 5,
    "finetune_epochs": 1,
    "finetune_timeout_s": 900,
    "md_steps": 20,
    "md_warmup_steps": 2,
    "n_relax": 2,
    "phonon_supercell": 1,
    "fwbw_repeats": 1,
}
# which bench parameters each measurement depends on (a change invalidates the cached entry)
PARAMS_OF: dict[str, tuple[str, ...]] = {
    "finetune": ("n_frames_8atom", "n_frames_64atom", "finetune_epochs"),
    "md": ("compound", "md_natoms", "md_steps", "md_warmup_steps", "md_T"),
    "relax": ("n_relax",),
    "phonons": ("compound", "phonon_supercell", "phonon_distance"),
    "fwbw": ("compound", "fwbw_batch", "fwbw_natoms", "fwbw_repeats"),
}
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) INFO: "
    r"(?P<what>Initial|Epoch (?P<epoch>\d+)): head: (?P<head>\S+),"
)


# --- small helpers ------------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ru_maxrss_bytes() -> int:
    """Peak resident set size of this process in bytes (macOS reports bytes, Linux kB)."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def host_info() -> dict[str, Any]:
    ram: float | None = None
    try:
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):  # pragma: no cover - exotic platforms
        ram = None
    try:
        hostname = socket.gethostname()
    except OSError:  # pragma: no cover
        hostname = "unknown"
    versions: dict[str, str] = {}
    for name in ("mace-torch", "torch", "ase", "phonopy"):
        try:
            versions[name] = dist_version(name)
        except PackageNotFoundError:  # pragma: no cover
            continue
    return {
        "hostname": hostname,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gib": None if ram is None else round(ram, 2),
        "versions": versions,
    }


def parse_tiers(spec: str | Iterable[str] | None) -> list[str]:
    """``"b,c"`` / ``"md,relax"`` / ``["a", "phonons"]`` -> ``["b", "c"]`` (canonical order)."""
    items: Iterable[str] = ALL_TIERS if spec is None else spec
    if isinstance(items, str):
        items = items.split(",")
    letters: set[str] = set()
    for raw in items:
        key = str(raw).strip().lower()
        if not key:
            continue
        letter = key if key in TIER_NAMES else TIER_LETTERS.get(key)
        if letter is None:
            raise ValueError(
                f"unknown bench tier {raw!r}; expected letters {ALL_TIERS} or names "
                f"{', '.join(TIER_NAMES.values())}"
            )
        letters.add(letter)
    return [letter for letter in TIER_NAMES if letter in letters]


def bench_params(cfg: Settings, quick: bool) -> dict[str, Any]:
    """``cfg.bench`` as a dict with the quick-mode sizes applied."""
    params = cfg.bench.model_dump(mode="json")
    if quick:
        params.update(QUICK_PARAMS)
    params["quick"] = bool(quick)
    return params


def _params_subset(params: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {key: params[key] for key in PARAMS_OF[name]}


def _thread_env(cfg: Settings) -> dict[str, str]:
    env = dict(os.environ)
    threads = str(int(cfg.compute.threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = threads
    src = str(repo_root() / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _tail(path: Path, n: int = 25) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _progress(message: str) -> None:
    log.info("%s", message)
    print(f"bench: {message}", file=sys.stderr, flush=True)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def set_threads(cfg: Settings) -> None:
    import torch  # noqa: PLC0415 - heavy, only when something is measured

    torch.set_num_threads(int(cfg.compute.threads))


def make_bench_calculator(model_path: str | Path, cfg: Settings) -> Any:
    """A ``MACECalculator`` on the model's own default head and ``cfg.compute``.

    The foundation files name their single head ``default`` while fine-tuned checkpoints use
    ``Default`` (the project's ``Head`` literal), so no head is requested: MACE takes the only
    head of a single-head model and the case-insensitive ``default`` head of a multihead one.
    ``calc.head`` records which one was used.
    """
    from mace.calculators import MACECalculator  # noqa: PLC0415 - heavy

    set_threads(cfg)
    return MACECalculator(
        model_paths=str(model_path), device=cfg.compute.device, default_dtype=cfg.compute.dtype
    )


# --- structures ---------------------------------------------------------------------------------


def reference_cell(cfg: Settings, compound: str) -> Atoms:
    """The relaxed reference cell of ``compound`` (dft tier choice) as bare ``Atoms``."""
    frame = reference_frame(cfg, compound)
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    atoms.info = {}
    return atoms


def source_frames(cfg: Settings) -> tuple[list[Frame], Path]:
    """The MPtrj B20 frames (data tier extract, else the dft tier's by-mp-id extract)."""
    data_dir = Path(cfg.paths.data_dir)
    for rel in (Path("frames") / "mptrj_b20.extxyz", Path("raw") / "mptrj" / "b20_mptrj.extxyz"):
        path = data_dir / rel
        if path.is_file():
            frames = read_frames(path)
            if frames:
                return frames, path
    raise FileNotFoundError(
        f"no MPtrj B20 frames under {data_dir} (data/frames/mptrj_b20.extxyz); run "
        "`b20mlip data pull --sources mptrj` first"
    )


def wbm_structures(cfg: Settings, n: int) -> tuple[dict[str, Atoms], dict[str, Any]]:
    """The first ``n`` ids of the WBM sample and their initial structures (zip read per member)."""
    data_dir = Path(cfg.paths.data_dir)
    sample_path = data_dir / "wbm" / f"sample_{cfg.data.wbm_sample_n}_s{cfg.data.wbm_seed}.json"
    zip_path = data_dir / discovery.DEFAULT_ATOMS_ZIP
    if not sample_path.is_file():
        raise FileNotFoundError(f"WBM sample not found: {sample_path} (run `b20mlip data sample`)")
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"WBM initial structures not found: {zip_path} (run `b20mlip data pull --sources wbm`)"
        )
    sample = wbm.load_sample(sample_path)
    ids = [str(i) for i in sample["ids"][: int(n)]]
    structures = wbm.atoms_for_ids(zip_path, ids)
    return structures, {"sample": str(sample_path), "atoms_zip": str(zip_path), "ids": ids}


# --- (a) fine-tune epoch ------------------------------------------------------------------------


def zero_shot_label(atoms: Atoms, calc: Any) -> Atoms:
    """A copy of ``atoms`` with the calculator's energy/forces(/stress) as single-point labels."""
    work = atoms.copy()
    work.info = {}
    work.calc = calc
    results: dict[str, Any] = {
        "energy": float(work.get_potential_energy()),
        "forces": np.asarray(work.get_forces(), dtype=float),
    }
    try:
        results["stress"] = np.asarray(work.get_stress(), dtype=float)
    except (PropertyNotImplementedError, NotImplementedError):  # pragma: no cover - calc dependent
        pass
    work.calc = SinglePointCalculator(work, **results)
    return work


def _has_mp_labels(frame: Frame) -> bool:
    return frame.energy is not None and frame.forces is not None and frame.energy_scale == "mp"


def finetune_frames(
    source: Sequence[Frame],
    calc: Any,
    n_8atom: int,
    n_64atom: int,
    seed: int,
    *,
    supercell: int = SUPERCELL_64,
) -> dict[str, list[Frame]]:
    """The two size classes of the fine-tune benchmark (see the module docstring).

    ``8atom``: the source frames in order (their ``mp``-scale DFT labels kept), then rattled
    copies labelled zero-shot until ``n_8atom``; a source frame without ``mp`` labels is
    relabelled zero-shot too. ``64atom``: ``supercell``^3 repetitions of the source cells,
    rattled by ``RATTLE_64ATOM_A`` and labelled zero-shot. Deterministic in ``seed``.
    """
    if not source:
        raise ValueError("the fine-tune benchmark needs at least one source frame")
    rng = np.random.default_rng(int(seed))
    eight: list[Frame] = []
    for i in range(int(n_8atom)):
        frame = source[i % len(source)]
        if i < len(source) and _has_mp_labels(frame):
            eight.append(frame)
            continue
        atoms = frame_to_atoms(frame)
        atoms.calc = None
        atoms.info = {}
        if i >= len(source):
            noise = rng.normal(0.0, RATTLE_8ATOM_A, atoms.positions.shape)
            atoms.positions = atoms.positions + noise
        labelled = zero_shot_label(atoms, calc)
        eight.append(
            frame_from_atoms(
                labelled,
                group_id=f"{frame.compound}/bench8/{frame.frame_id[:8]}",
                compound=frame.compound,
                config_type="rattle",
                parent_id=frame.frame_id,
                label_source="mace_zero_shot",
                energy_scale="mp",
            )
        )
    big: list[Frame] = []
    reps = (int(supercell),) * 3
    for i in range(int(n_64atom)):
        frame = source[i % len(source)]
        atoms = frame_to_atoms(frame)
        atoms.calc = None
        atoms.info = {}
        cell = atoms.repeat(reps)
        cell.positions = cell.positions + rng.normal(0.0, RATTLE_64ATOM_A, cell.positions.shape)
        labelled = zero_shot_label(cell, calc)
        big.append(
            frame_from_atoms(
                labelled,
                group_id=f"{frame.compound}/bench64/{frame.frame_id[:8]}",
                compound=frame.compound,
                config_type="rattle",
                parent_id=frame.frame_id,
                label_source="mace_zero_shot",
                energy_scale="mp",
            )
        )
    return {"8atom": eight, "64atom": big}


def split_frames(frames: Sequence[Frame], batch_size: int = 4) -> tuple[list[Frame], list[Frame]]:
    """Train / valid by position: about a tenth validates, and the training part is cut to a
    multiple of ``batch_size`` (MACE's training loader drops the last incomplete batch, so every
    training frame is then really processed once per epoch); the rest validates."""
    n = len(frames)
    if n < int(batch_size) + 1:
        raise ValueError(
            f"a size class needs at least batch_size + 1 = {int(batch_size) + 1} frames, got {n}"
        )
    n_valid = max(1, n // 10)
    n_train = (n - n_valid) // int(batch_size) * int(batch_size)
    return list(frames[:n_train]), list(frames[n_train:])


def parse_epoch_walls(log_text: str) -> dict[str, Any]:
    """Epoch wall seconds from MACE's validation lines.

    MACE logs ``Initial: head: ...`` after the pre-training validation and ``Epoch k: head:
    ...`` after each training epoch plus its validation, so consecutive timestamps bracket one
    full epoch (training + validation, no start-up). Returns ``{"initial_ts": unix seconds or
    None, "epoch_wall_s": {epoch: seconds}}``.
    """
    stamps: list[tuple[int | None, float]] = []
    for line in log_text.splitlines():
        match = LOG_LINE_RE.match(line.strip())
        if not match:
            continue
        stamp = datetime.strptime(match["ts"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        epoch = None if match["epoch"] is None else int(match["epoch"])
        if stamps and stamps[-1][0] == epoch:  # one line per head: keep the first
            continue
        stamps.append((epoch, stamp))
    walls: dict[int, float] = {}
    for (_, prev_stamp), (epoch, stamp) in zip(stamps, stamps[1:], strict=False):
        if epoch is not None:
            walls[epoch] = stamp - prev_stamp
    initial = stamps[0][1] if stamps and stamps[0][0] is None else None
    return {"initial_ts": initial, "epoch_wall_s": walls}


def parse_step_times(results_txt: str | Path) -> dict[int, dict[str, float]]:
    """Per-epoch sum of MACE's optimiser-step times (``mode == "opt"`` records) and batch counts."""
    per_epoch: dict[int, dict[str, float]] = {}
    path = Path(results_txt)
    if not path.is_file():
        return per_epoch
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") != "opt" or not isinstance(rec.get("time"), (int, float)):
            continue
        epoch = int(rec.get("epoch") or 0)
        entry = per_epoch.setdefault(epoch, {"steps_s": 0.0, "n_batches": 0})
        entry["steps_s"] += float(rec["time"])
        entry["n_batches"] += 1
    return per_epoch


def time_finetune_epochs(
    cfg: Settings,
    model_path: str | Path,
    frames: Sequence[Frame],
    work_dir: str | Path,
    *,
    name: str,
    epochs: int,
    seed: int,
    timeout_s: float,
    cleanup: bool = True,
) -> dict[str, Any]:
    """One naive ``mace_run_train`` on ``frames`` (split 90/10) for ``epochs`` epochs; timing of
    the last epoch. Raises on a non-zero exit or a timeout (the caller records the error)."""
    work = Path(work_dir).resolve()  # the child runs with cwd=work: every argv path is absolute
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    train, valid = split_frames(frames, int(cfg.train.batch_size))
    files = finetune.write_split_files({"train": train, "valid": valid}, work / finetune.DATA_DIR)
    argv = finetune.build_args(
        cfg, "naive", files, int(seed), work,
        energy_scale="mp", name=name, epochs=int(epochs), foundation=Path(model_path).resolve(),
    )  # fmt: skip
    cmd = finetune.train_command(argv)
    log_path = work / finetune.STDOUT_LOG
    t_launch = time.time()
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ " + shlex.join(cmd) + "\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=work, env=_thread_env(cfg), stdout=fh, stderr=subprocess.STDOUT,
                timeout=float(timeout_s), check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"mace_run_train ({name}) exceeded {timeout_s} s; log: {log_path}"
            ) from None
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"mace_run_train ({name}) exited with {proc.returncode}; last lines of {log_path}:\n"
            + _tail(log_path)
        )
    parsed = parse_epoch_walls(log_path.read_text(encoding="utf-8", errors="replace"))
    if not parsed["epoch_wall_s"]:
        raise RuntimeError(f"no 'Epoch k:' validation line in {log_path}; cannot time an epoch")
    results = sorted((work / finetune.RESULTS_DIR).glob(f"{name}_run-{seed}_train.txt"))
    steps = parse_step_times(results[-1]) if results else {}
    last = max(parsed["epoch_wall_s"])
    epoch_wall = float(parsed["epoch_wall_s"][last])
    n_train = len(train)  # a multiple of the batch size: every training frame is processed
    step = steps.get(last, {})
    steps_s = step.get("steps_s")
    n_batches = int(step.get("n_batches", 0))
    out: dict[str, Any] = {
        "name": name,
        "natoms": len(frames[0].numbers),
        "n_frames": len(frames),
        "n_train": n_train,
        "n_valid": len(valid),
        "batch_size": int(cfg.train.batch_size),
        "epochs": int(epochs),
        "measured_epoch": int(last),
        "s_per_epoch_wall": epoch_wall,
        "s_per_frame": epoch_wall / n_train,
        "s_per_epoch_steps": None if steps_s is None else float(steps_s),
        "n_batches": n_batches,
        "s_per_batch": None if steps_s is None or not n_batches else float(steps_s) / n_batches,
        "s_per_frame_steps": None if steps_s is None else float(steps_s) / n_train,
        "epoch_wall_s": {str(k): float(v) for k, v in sorted(parsed["epoch_wall_s"].items())},
        "startup_s": None if parsed["initial_ts"] is None else parsed["initial_ts"] - t_launch,
        "subprocess_wall_s": wall,
        "returncode": int(proc.returncode),
        "label_sources": dict(Counter(f.label_source for f in frames)),
        "argv": list(argv),
        "log": str(log_path),
        "results": str(results[-1]) if results else None,
        "train_file": str(files["train"]),
        "valid_file": str(files["valid"]),
    }
    if cleanup:  # the trained weights are not a result; keep logs, results and frames
        for sub in (finetune.CHECKPOINTS_DIR, finetune.MODEL_DIR):
            shutil.rmtree(work / sub, ignore_errors=True)
        out["cleanup"] = f"{finetune.CHECKPOINTS_DIR}/ and {finetune.MODEL_DIR}/ removed"
    return out


def bench_finetune(
    cfg: Settings,
    *,
    model_path: str | Path,
    calc: Any,
    params: Mapping[str, Any],
    work_dir: str | Path,
    seed: int,
    source: Sequence[Frame] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measurement (a): s/frame/epoch per size class. ``previous`` (the cached entry) lets a
    re-run keep a size class that already succeeded; a failed class is recorded under
    ``classes[<class>]["error"]`` and makes the whole measurement an error entry."""
    source_path: Path | None = None
    frames = list(source) if source is not None else None
    if frames is None:
        frames, source_path = source_frames(cfg)
    n8, n64 = int(params["n_frames_8atom"]), int(params["n_frames_64atom"])
    cached = dict((previous or {}).get("classes") or {})
    done = {k: v for k, v in cached.items() if isinstance(v, dict) and "error" not in v}
    classes: dict[str, list[Frame]] = {}
    if len(done) < len(SIZE_CLASSES):
        _progress(f"finetune: labelling {n8} x 8-atom + {n64} x 64-atom frames")
        classes = finetune_frames(frames, calc, n8, n64, seed)
    result: dict[str, Any] = {
        "source": None if source_path is None else str(source_path),
        "n_source": len(frames),
        "variant": "naive",
        "energy_scale": "mp",
        "e0s": "foundation",
        "classes": {},
    }
    errors: list[str] = []
    for cls in SIZE_CLASSES:
        if cls in done:
            result["classes"][cls] = dict(done[cls])
            _progress(f"finetune {cls}: cached")
            continue
        _progress(
            f"finetune {cls}: {len(classes[cls])} frames, {params['finetune_epochs']} epoch(s)"
        )
        try:
            result["classes"][cls] = time_finetune_epochs(
                cfg, model_path, classes[cls], Path(work_dir) / cls,
                name=f"bench_{cls}", epochs=int(params["finetune_epochs"]), seed=seed,
                timeout_s=float(params["finetune_timeout_s"]),
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - recorded, never faked
            log.exception("finetune %s failed", cls)
            result["classes"][cls] = {"error": repr(exc), "traceback": traceback.format_exc()}
            errors.append(f"{cls}: {exc!r}")
    for cls in SIZE_CLASSES:
        entry = result["classes"][cls]
        if "error" not in entry:
            result[f"s_per_frame_{cls}"] = entry["s_per_frame"]
            result[f"s_per_epoch_{cls}"] = entry["s_per_epoch_wall"]
    startups = [
        c["startup_s"] for c in result["classes"].values() if c.get("startup_s") is not None
    ]
    result["startup_s"] = _mean(startups) if startups else None
    if errors:
        result["error"] = "; ".join(errors)
    return result


# --- (b) MD -----------------------------------------------------------------------------------


def bench_md(
    cfg: Settings, *, calc: Any, atoms: Atoms, params: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    """Measurement (b): s/step of NVT Langevin MD at each ``bench.md_natoms`` size."""
    steps, warmup = int(params["md_steps"]), int(params["md_warmup_steps"])
    if steps <= warmup:
        raise ValueError("bench.md_steps must exceed bench.md_warmup_steps")
    temperature = float(params["md_T"])
    dt = float(cfg.md.timestep_fs)
    result: dict[str, Any] = {
        "ensemble": "nvt",
        "thermostat": thermostat_label("nvt", cfg),
        "timestep_fs": dt,
        "T_K": temperature,
        "steps": steps,
        "warmup_steps": warmup,
        "sizes": {},
    }
    for target in params["md_natoms"]:
        cell, reps = make_supercell(atoms, int(target))
        cell.calc = calc
        rng = np.random.default_rng(int(seed))
        MaxwellBoltzmannDistribution(cell, temperature_K=temperature, rng=rng, force_temp=True)
        Stationary(cell)
        dyn = make_dynamics(cell, "nvt", temperature, dt, cfg, rng)
        _progress(f"md: {len(cell)} atoms, {steps} steps ({warmup} warm-up)")
        dyn.run(warmup)
        t0 = time.perf_counter()
        dyn.run(steps - warmup)
        wall = time.perf_counter() - t0
        timed = steps - warmup
        row = {
            "natoms": len(cell),
            "reps": reps,
            "steps_timed": timed,
            "wall_s": wall,
            "s_per_step": wall / timed,
            "atom_steps_per_s": len(cell) * timed / wall,
            "T_final_K": float(cell.get_temperature()),
            "rss_peak_gb": ru_maxrss_bytes() / 2**30,  # this process so far (GiB)
        }
        result["sizes"][str(len(cell))] = row
        result[f"s_per_step_{len(cell)}"] = row["s_per_step"]
    return result


# --- (c) WBM relaxations ----------------------------------------------------------------------


def bench_relax(
    cfg: Settings,
    *,
    calc: Any,
    structures: Mapping[str, Atoms],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measurement (c): FIRE + FrechetCellFilter on each structure; s and steps per structure."""
    if not structures:
        raise ValueError("no structures to relax")
    fmax, max_steps = float(cfg.eval.fmax), int(cfg.eval.max_steps)
    rows: list[dict[str, Any]] = []
    for wid, atoms in structures.items():
        t0 = time.perf_counter()
        _, steps = discovery.relax(atoms, calc, fmax=fmax, steps=max_steps)
        wall = time.perf_counter() - t0
        rows.append(
            {
                "id": str(wid),
                "natoms": len(atoms),
                "steps": int(steps),
                "wall_s": wall,
                "capped": int(steps) >= max_steps,
            }
        )
        _progress(f"relax {wid}: {len(atoms)} atoms, {steps} steps, {wall:.1f} s")
    walls = [r["wall_s"] for r in rows]
    counts = [float(r["steps"]) for r in rows]
    total_steps = sum(r["steps"] for r in rows)
    return {
        "n": len(rows),
        "fmax": fmax,
        "max_steps": max_steps,
        "optimizer": "FIRE + FrechetCellFilter",
        "s_per_structure": _mean(walls),
        "s_per_structure_median": _median(walls),
        "steps_per_structure": _mean(counts),
        "steps_per_structure_median": _median(counts),
        "n_capped": sum(1 for r in rows if r["capped"]),
        "s_per_relax_step": sum(walls) / total_steps if total_steps else None,
        "natoms_mean": _mean([float(r["natoms"]) for r in rows]),
        "total_wall_s": sum(walls),
        "source": dict(source or {}),
        "structures": rows,
    }


# --- (d) phonons ------------------------------------------------------------------------------


def res_label(atoms: Atoms, size: int) -> str:
    return f"{harmonic.compound_of(atoms)} {size}x{size}x{size}"


def bench_phonons(
    cfg: Settings, *, calc: Any, atoms: Atoms, params: Mapping[str, Any], run_id: str = ""
) -> dict[str, Any]:
    """Measurement (d): ``harmonic.compute`` wall time and the number of displacements."""
    size = int(params["phonon_supercell"])
    distance = float(params["phonon_distance"])
    supercell = (size, size, size)
    phonon = harmonic.new_phonopy(atoms, supercell)
    phonon.generate_displacements(distance=distance)
    n_disp = len(phonon.supercells_with_displacements or [])
    _progress(f"phonons: {res_label(atoms, size)}, {n_disp} displacements")
    t0 = time.perf_counter()
    res = harmonic.compute(atoms, calc, supercell, distance, cell_source="dft", run_id=run_id)
    wall = time.perf_counter() - t0
    return {
        "compound": res.compound,
        "supercell": list(supercell),
        "natoms_cell": len(atoms),
        "natoms_supercell": len(atoms) * size**3,
        "distance_A": distance,
        "n_displacements": n_disp,
        "s": wall,
        "s_per_displacement": wall / n_disp if n_disp else None,
        "n_branches": len(res.frequencies_meV[0]) if res.frequencies_meV else 0,
        "n_qpoints": len(res.qpoints),
        "imaginary_count": int(res.imaginary_count),
    }


# --- (e) forward + backward -------------------------------------------------------------------


def fwbw_measure(
    model_path: str | Path,
    atoms_list: Sequence[Atoms],
    *,
    threads: int,
    dtype: str = "float64",
    batch_size: int = 4,
    repeats: int = 3,
) -> dict[str, Any]:
    """One training-style batch through a MACE model: forward with forces, loss, backward.

    Times ``repeats`` passes after one warm-up pass and samples the process peak RSS before the
    model is loaded, after the batch is built and after the passes (``ru_maxrss``: bytes on
    macOS, kB on Linux, both converted to GiB).
    """
    import torch  # noqa: PLC0415 - heavy
    from mace import data as mace_data  # noqa: PLC0415
    from mace.tools import torch_geometric, torch_tools  # noqa: PLC0415
    from mace.tools import utils as mace_utils  # noqa: PLC0415

    torch.set_num_threads(int(threads))
    torch_tools.set_default_dtype(dtype)
    rss_before = ru_maxrss_bytes()
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model = model.to(getattr(torch, dtype))
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    z_table = mace_utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)
    heads = [str(h) for h in (getattr(model, "heads", None) or ["Default"])]
    keyspec = mace_data.KeySpecification()
    dataset = [
        mace_data.AtomicData.from_config(
            mace_data.config_from_atoms(a, key_specification=keyspec, head_name=heads[0]),
            z_table=z_table,
            cutoff=r_max,
            heads=heads,
        )
        for a in atoms_list
    ]
    loader = torch_geometric.dataloader.DataLoader(  # a list is what MACE's calculator passes
        dataset=dataset,  # type: ignore[arg-type]
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch_dict = batch.to_dict()
    rss_loaded = ru_maxrss_bytes()
    times: list[float] = []
    for _ in range(int(repeats) + 1):
        t0 = time.perf_counter()
        out = model(
            batch_dict, training=True, compute_force=True, compute_virials=False,
            compute_stress=False,
        )  # fmt: skip
        loss = out["energy"].sum() + out["forces"].square().sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t0)
    rss_after = ru_maxrss_bytes()
    gib = float(2**30)
    return {
        "batch_size": min(int(batch_size), len(dataset)),
        "natoms_per_structure": [len(a) for a in atoms_list],
        "n_atoms": int(batch_dict["positions"].shape[0]),
        "n_edges": int(batch_dict["edge_index"].shape[1]),
        "repeats": int(repeats),
        "s_per_batch": _mean(times[1:]),
        "s_per_batch_warmup": times[0],
        "s_per_batch_all": times,
        "rss_before_gb": rss_before / gib,
        "rss_loaded_gb": rss_loaded / gib,
        "peak_rss_gb": rss_after / gib,
        "rss_unit": "GiB",
        "dtype": dtype,
        "threads": int(threads),
        "model_r_max": r_max,
        "heads": heads,
    }


def _fwbw_child_main() -> None:
    """Entry point of the fresh process ``bench_fwbw`` spawns (JSON spec on stdin, one
    ``FWBW_MARKER`` line on stdout)."""
    spec = json.load(sys.stdin)
    atoms_list = ase_read(spec["structures"], index=":", format="extxyz")
    result = fwbw_measure(
        spec["model"], list(atoms_list), threads=int(spec["threads"]), dtype=spec["dtype"],
        batch_size=int(spec["batch"]), repeats=int(spec["repeats"]),
    )  # fmt: skip
    sys.stdout.write("\n" + FWBW_MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()


def bench_fwbw(
    cfg: Settings,
    *,
    model_path: str | Path,
    atoms: Atoms,
    params: Mapping[str, Any],
    work_dir: str | Path,
    seed: int,
    timeout_s: float = FWBW_TIMEOUT_S,
) -> dict[str, Any]:
    """Measurement (e): ``fwbw_measure`` in a fresh interpreter so the peak RSS is that of a
    process that loads the model and runs one batch (not of this bench process)."""
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cell, reps = make_supercell(atoms, int(params["fwbw_natoms"]))
    rng = np.random.default_rng(int(seed))
    batch: list[Atoms] = []
    for _ in range(int(params["fwbw_batch"])):
        item = cell.copy()
        item.calc = None
        item.info = {}
        item.positions = item.positions + rng.normal(0.0, RATTLE_64ATOM_A, item.positions.shape)
        batch.append(item)
    structures = work / "batch.extxyz"
    ase_write(structures, batch, format="extxyz")
    spec = {
        "model": str(Path(model_path).resolve()),
        "structures": str(structures),
        "threads": int(cfg.compute.threads),
        "dtype": cfg.compute.dtype,
        "batch": int(params["fwbw_batch"]),
        "repeats": int(params["fwbw_repeats"]),
    }
    cmd = [sys.executable, "-c", "from b20mlip.bench import _fwbw_child_main as main; main()"]
    _progress(f"fwbw: batch {len(batch)} x {len(cell)} atoms in a fresh process")
    log_path = work / "fwbw_stdout.log"
    try:
        proc = subprocess.run(
            cmd, input=json.dumps(spec), capture_output=True, text=True, env=_thread_env(cfg),
            timeout=float(timeout_s), check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"fwbw child exceeded {timeout_s} s") from None
    log_path.write_text(
        "$ " + shlex.join(cmd) + "\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fwbw child exited with {proc.returncode}; last lines of {log_path}:\n"
            + _tail(log_path)
        )
    line = next(
        (ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(FWBW_MARKER)), None
    )
    if line is None:
        raise RuntimeError(f"fwbw child printed no result line; see {log_path}")
    result = json.loads(line[len(FWBW_MARKER) :])
    result.update({"process": "child", "reps": reps, "structures": str(structures),
                   "log": str(log_path)})  # fmt: skip
    return result


# --- schedule and report ------------------------------------------------------------------------


def _ok(measurements: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    entry = measurements.get(name)
    return entry if isinstance(entry, dict) and "error" not in entry else None


def derive_schedule(measurements: Mapping[str, Any], cfg: Settings) -> dict[str, Any]:
    """Re-derive the SPEC.md section 5 runtime table from the measurements.

    Every entry carries its ``formula`` as a string, the ``inputs`` it used and the value (in the
    unit named by the key); an entry whose inputs are missing carries ``error`` instead.
    """
    out: dict[str, Any] = {}

    def put(key: str, value: float | None, formula: str, inputs: dict[str, Any], why: str) -> None:
        entry: dict[str, Any] = {"formula": formula, "inputs": inputs, "spec": why}
        missing = [k for k, v in inputs.items() if v is None]
        if value is None or missing:
            entry["error"] = f"missing measurement(s): {', '.join(missing) or 'value'}"
        else:
            entry["value"] = float(value)
        out[key] = entry

    epochs = int(cfg.train.epochs)
    dt = float(cfg.md.timestep_fs)
    n_temps = len(cfg.data.temperatures_K)
    ft = _ok(measurements, "finetune")
    s8 = None if ft is None else ft.get("s_per_frame_8atom")
    s64 = None if ft is None else ft.get("s_per_frame_64atom")
    startup = None if ft is None else ft.get("startup_s")
    epoch_s = None if s8 is None or s64 is None else ROUND0_N8 * s8 + ROUND0_N64 * s64
    put(
        "finetune_round0_epoch_s", epoch_s,
        f"{ROUND0_N8} * s_per_frame_8atom + {ROUND0_N64} * s_per_frame_64atom",
        {"s_per_frame_8atom": s8, "s_per_frame_64atom": s64},
        f"train naive round-0 ({ROUND0_N8} x 8 + {ROUND0_N64} x 64 frames), s/epoch",
    )  # fmt: skip
    per_seed = None if epoch_s is None else (epochs * epoch_s + (startup or 0.0)) / 60.0
    put(
        "finetune_round0_per_seed_min", per_seed,
        f"({epochs} epochs * finetune_round0_epoch_s + startup_s) / 60",
        {"finetune_round0_epoch_s": epoch_s, "startup_s": startup, "epochs": epochs},
        f"train naive: {epochs} epochs per seed",
    )  # fmt: skip
    put(
        "finetune_round0_3seeds_h", None if per_seed is None else N_SEEDS * per_seed / 60.0,
        f"{N_SEEDS} seeds * finetune_round0_per_seed_min / 60",
        {"finetune_round0_per_seed_min": per_seed},
        f"train naive: {N_SEEDS} seeds overnight",
    )  # fmt: skip

    md = _ok(measurements, "md")
    s_step_64 = None if md is None else md.get("s_per_step_64")
    s_step_512 = None if md is None else md.get("s_per_step_512")
    steps_40ps = MD_PS * 1000.0 / dt
    per_t = None if s_step_64 is None else steps_40ps * s_step_64 / 3600.0
    put(
        "md_nvt_64_40ps_per_T_h", per_t,
        f"({MD_PS} ps / {dt} fs) steps * s_per_step_64 / 3600",
        {"s_per_step_64": s_step_64, "steps": steps_40ps},
        f"md ase: 64 atoms, {MD_PS:g} ps per temperature",
    )  # fmt: skip
    put(
        "md_nvt_64_40ps_3T_h", None if per_t is None else n_temps * per_t,
        f"{n_temps} temperatures * md_nvt_64_40ps_per_T_h",
        {"md_nvt_64_40ps_per_T_h": per_t, "n_temperatures": n_temps},
        f"md ase: {MD_PS:g} ps x {n_temps} temperatures overnight",
    )  # fmt: skip
    steps_100ps = NPT512_PS * 1000.0 / dt
    put(
        "md_512_100ps_ase_h",
        None if s_step_512 is None else steps_100ps * s_step_512 / 3600.0,
        f"({NPT512_PS} ps / {dt} fs) steps * s_per_step_512 / 3600",
        {"s_per_step_512": s_step_512, "steps": steps_100ps},
        f"md: {NPT512_PS:g} ps at 512 atoms on the Mac with ASE (the LAMMPS/GPU job's local cost)",
    )  # fmt: skip
    windows, ps_window = int(cfg.sampling.windows), float(cfg.sampling.ps_per_window)
    steps_umbrella = windows * ps_window * 1000.0 / dt
    put(
        "umbrella_windows_64_h",
        None if s_step_64 is None else steps_umbrella * s_step_64 / 3600.0,
        f"{windows} windows * ({ps_window} ps / {dt} fs) steps * s_per_step_64 / 3600",
        {"s_per_step_64": s_step_64, "steps": steps_umbrella},
        f"sampling umbrella: {windows} windows x {ps_window:g} ps at 64 atoms",
    )  # fmt: skip

    relax = _ok(measurements, "relax")
    s_struct = None if relax is None else relax.get("s_per_structure")
    s_relax_step = None if relax is None else relax.get("s_per_relax_step")
    n_wbm = int(cfg.data.wbm_sample_n)
    max_steps = int(cfg.eval.max_steps)
    put(
        "wbm_relax_h", None if s_struct is None else n_wbm * s_struct / 3600.0,
        f"{n_wbm} structures * s_per_structure / 3600",
        {"s_per_structure": s_struct},
        f"eval discovery: WBM-{n_wbm} per model (mean over the timed structures)",
    )  # fmt: skip
    put(
        "wbm_relax_cap_h",
        None if s_relax_step is None else n_wbm * max_steps * s_relax_step / 3600.0,
        f"{n_wbm} structures * {max_steps} steps * s_per_relax_step / 3600",
        {"s_per_relax_step": s_relax_step},
        f"eval discovery: worst case, every structure at the {max_steps}-step cap",
    )  # fmt: skip

    ph = _ok(measurements, "phonons")
    s_ph = None if ph is None else ph.get("s")
    put(
        "phonondb103_h", None if s_ph is None else PHONONDB_N * s_ph / 3600.0,
        f"{PHONONDB_N} compounds * phonons_s / 3600",
        {"phonons_s": s_ph},
        f"eval phonons: phononDB-{PHONONDB_N} per model (2x2x2 FeSi as the unit cost)",
    )  # fmt: skip

    fw = _ok(measurements, "fwbw")
    peak = None if fw is None else fw.get("peak_rss_gb")
    ram = host_info()["ram_gib"]
    put(
        "fwbw_ram_headroom_gib", None if peak is None or ram is None else ram - peak,
        "host RAM (GiB) - fwbw peak_rss_gb",
        {"ram_gib": ram, "peak_rss_gb": peak},
        "train: 64-atom batches fit the Mac (positive headroom)",
    )  # fmt: skip
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def bench_report(bench_json: Mapping[str, Any] | str | Path) -> str:
    """Markdown tables (measurements, then the derived schedule) of a ``bench.json``."""
    state: Mapping[str, Any]
    if isinstance(bench_json, (str, Path)):
        state = json.loads(Path(bench_json).read_text(encoding="utf-8"))
    else:
        state = bench_json
    model = state.get("model") or {}
    host = state.get("host") or {}
    lines: list[str] = [
        f"Model `{Path(str(model.get('path', '?'))).name}` (sha256 "
        f"`{str(model.get('sha256') or '?')[:12]}`), host `{host.get('hostname', '?')}` "
        f"({host.get('machine', '?')}), {state.get('threads', '?')} threads, "
        f"{state.get('dtype', '?')}, quick={state.get('quick', False)}, "
        f"status **{state.get('status', '?')}**, updated {state.get('updated_at', '?')}.",
        "",
        "| Measurement | Metric | Value | Unit | n | Notes |",
        "|---|---|---|---|---|---|",
    ]
    m = state.get("measurements") or {}

    def row(name: str, metric: str, value: Any, unit: str, n: Any, notes: str = "") -> None:
        lines.append(f"| {name} | {metric} | {_fmt(value)} | {unit} | {_fmt(n)} | {notes} |")

    for letter, name in TIER_NAMES.items():
        entry = m.get(name)
        label = f"({letter}) {name}"
        if entry is None:
            row(label, "-", None, "", None, "not measured")
            continue
        if "error" in entry:
            row(label, "error", None, "", None, str(entry["error"]).splitlines()[0][:120])
            continue
        if name == "finetune":
            for cls in SIZE_CLASSES:
                c = entry.get("classes", {}).get(cls, {})
                if "error" in c or not c:
                    row(label, f"s_per_frame_{cls}", None, "", None, c.get("error", "missing"))
                    continue
                row(
                    label, f"s_per_frame_{cls}", c["s_per_frame"], "s/frame/epoch", c["n_train"],
                    f"epoch wall {_fmt(c['s_per_epoch_wall'])} s, {c['n_batches']} batches of "
                    f"{c['batch_size']}, steps-only {_fmt(c.get('s_per_frame_steps'))} s/frame, "
                    f"epoch {c['measured_epoch']} of {c['epochs']}",
                )  # fmt: skip
            row(label, "startup_s", entry.get("startup_s"), "s", len(entry.get("classes", {})),
                "mace_run_train start-up (model load, data) before validation 0")  # fmt: skip
        elif name == "md":
            for size, r in entry.get("sizes", {}).items():
                row(
                    label, f"s_per_step_{size}", r["s_per_step"], "s/step", r["steps_timed"],
                    f"{entry['ensemble']} {entry['T_K']:g} K, {entry['timestep_fs']:g} fs, "
                    f"{_fmt(r['atom_steps_per_s'])} atom-steps/s"
                    + (f", process peak RSS {_fmt(r['rss_peak_gb'])} GiB"
                       if r.get("rss_peak_gb") is not None else ""),
                )  # fmt: skip
        elif name == "relax":
            row(label, "s_per_structure", entry["s_per_structure"], "s", entry["n"],
                f"median {_fmt(entry['s_per_structure_median'])} s; fmax {entry['fmax']}, "
                f"cap {entry['max_steps']}, capped {entry['n_capped']}")  # fmt: skip
            row(label, "steps_per_structure", entry["steps_per_structure"], "steps", entry["n"],
                f"median {_fmt(entry['steps_per_structure_median'])}; "
                f"{_fmt(entry.get('s_per_relax_step'))} s per FIRE step")  # fmt: skip
        elif name == "phonons":
            sc = "x".join(str(s) for s in entry["supercell"])
            row(label, f"{entry['compound']} {sc} s", entry["s"], "s", entry["n_displacements"],
                f"{entry['n_displacements']} displacements of {entry['natoms_supercell']} atoms, "
                f"{entry['distance_A']} A; {entry['imaginary_count']} imaginary")  # fmt: skip
        elif name == "fwbw":
            row(label, "peak_rss_gb", entry["peak_rss_gb"], "GiB", entry["repeats"],
                f"fresh process; before load {_fmt(entry['rss_before_gb'])}, batch built "
                f"{_fmt(entry['rss_loaded_gb'])} GiB")  # fmt: skip
            row(label, "s_per_batch", entry["s_per_batch"], "s", entry["repeats"],
                f"{entry['batch_size']} x {entry['natoms_per_structure'][0]} atoms, "
                f"{entry['n_edges']} edges, forward+backward with forces")  # fmt: skip
    schedule = state.get("schedule") or {}
    if schedule:
        lines += ["", "| Schedule entry | Value | Formula | SPEC row |", "|---|---|---|---|"]
        for key, entry in schedule.items():
            value = entry.get("value")
            shown = _fmt(value) if value is not None else f"n/a ({entry.get('error', '')})"
            lines.append(f"| {key} | {shown} | `{entry['formula']}` | {entry.get('spec', '')} |")
    return "\n".join(lines) + "\n"


# --- bench.json state and numbers.json --------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
                   encoding="utf-8")  # fmt: skip
    os.replace(tmp, path)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data


def new_state(
    cfg: Settings, model_path: Path, model_sha: str, label: str, quick: bool
) -> dict[str, Any]:
    stamp = utc_now()
    return {
        "schema": SCHEMA,
        "created_at": stamp,
        "updated_at": stamp,
        "status": "partial",
        "model": {"path": str(model_path), "sha256": model_sha, "label": label},
        "threads": int(cfg.compute.threads),
        "dtype": cfg.compute.dtype,
        "device": cfg.compute.device,
        "quick": bool(quick),
        "host": host_info(),
        "measurements": {},
        "schedule": {},
        "runs": [],
    }


def measurement_cached(state: Mapping[str, Any], name: str, params: Mapping[str, Any]) -> bool:
    """A cached entry counts when it has no error and was measured with the same parameters."""
    entry = state.get("measurements", {}).get(name)
    if not isinstance(entry, dict) or "error" in entry:
        return False
    return entry.get("params") == _params_subset(params, name)


def bench_numbers(state: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """The report-tier ``numbers.json`` payload (keys ``bench.<measurement>.<metric>``)."""
    model = state.get("model") or {}
    host = state.get("host") or {}
    base: dict[str, Any] = {
        "reference": {
            "code": "mace",
            "functional": "PBE",
            "pseudos": None,
            "e0_source": "foundation",
        },  # fmt: skip
        "e0_source": "foundation",
        "head": "Default",
        "seed": int(seed),
        "ci95": None,
        "ci95_reason": "single timing run",
        "host": host.get("hostname"),
        "threads": state.get("threads"),
        "dtype": state.get("dtype"),
        "model_sha256": model.get("sha256"),
        "model_label": model.get("label"),
        "quick": bool(state.get("quick", False)),
    }
    numbers: dict[str, Any] = {}

    def put(key: str, value: Any, n: int, **extra: Any) -> None:
        if value is None or not np.isfinite(float(value)):
            return
        numbers[key] = float(value)
        numbers[f"{key}@meta"] = {**base, "n": max(1, int(n)), **extra}

    m = state.get("measurements") or {}
    ft = _ok(m, "finetune")
    if ft is not None:
        for cls in SIZE_CLASSES:
            c = ft.get("classes", {}).get(cls) or {}
            if "error" in c or "s_per_frame" not in c:
                continue
            put(f"bench.finetune.s_per_frame_{cls}", c["s_per_frame"], c["n_train"],
                unit="s/frame/epoch", batch_size=c["batch_size"], natoms=c["natoms"])  # fmt: skip
            put(f"bench.finetune.s_per_epoch_{cls}", c["s_per_epoch_wall"], c["n_train"],
                unit="s", n_valid=c["n_valid"])  # fmt: skip
        put("bench.finetune.startup_s", ft.get("startup_s"), len(ft.get("classes", {})), unit="s")
    md = _ok(m, "md")
    if md is not None:
        for size, r in md.get("sizes", {}).items():
            put(f"bench.md.s_per_step_{size}", r["s_per_step"], r["steps_timed"], unit="s/step",
                natoms=r["natoms"], ensemble=md["ensemble"], T=md["T_K"])  # fmt: skip
    relax = _ok(m, "relax")
    if relax is not None:
        put("bench.relax.s_per_structure", relax["s_per_structure"], relax["n"], unit="s")
        put("bench.relax.steps_per_structure", relax["steps_per_structure"], relax["n"],
            unit="steps")  # fmt: skip
        put("bench.relax.s_per_relax_step", relax.get("s_per_relax_step"), relax["n"], unit="s")
    ph = _ok(m, "phonons")
    if ph is not None:
        sc = "".join(str(s) for s in ph["supercell"])
        put(f"bench.phonons_{str(ph['compound']).lower()}_{sc}.s", ph["s"], ph["n_displacements"],
            unit="s", n_displacements=ph["n_displacements"])  # fmt: skip
    fw = _ok(m, "fwbw")
    if fw is not None:
        put("bench.fwbw.peak_rss_gb", fw["peak_rss_gb"], fw["repeats"], unit="GiB",
            batch_size=fw["batch_size"], n_atoms=fw["n_atoms"])  # fmt: skip
        put("bench.fwbw.s_per_batch", fw["s_per_batch"], fw["repeats"], unit="s")
    for key, entry in (state.get("schedule") or {}).items():
        if "value" in entry:
            put(f"bench.schedule.{key}", entry["value"], 1, formula=entry["formula"],
                ci95_reason="derived from a single timing run")  # fmt: skip
    return numbers


# --- the stage ----------------------------------------------------------------------------------


def resolve_model(cfg: Settings, model: str | Path | None) -> Path:
    """``model`` argument -> ``cfg.bench.model`` -> the foundation of ``cfg.train.foundation``."""
    chosen = model if model is not None else cfg.bench.model
    if chosen is None:
        return finetune.resolve_foundation(cfg)
    path = Path(chosen)
    if not path.is_file():
        raise FileNotFoundError(f"bench model not found: {path}")
    return path.resolve()


def run(
    cfg: Settings,
    ctx: RunContext,
    out: str | Path,
    *,
    model: str | Path | None = None,
    tiers: str | Iterable[str] | None = None,
    quick: bool | None = None,
    force: bool | None = None,
    source: Sequence[Frame] | None = None,
) -> StageResult:
    """Stage ``bench`` (``b20mlip bench --out <out>``): the five measurements into
    ``<out>/bench.json``, ``numbers.json`` + manifest into ``ctx.out_dir``.

    ``model``/``tiers``/``quick``/``force`` default to ``cfg.bench`` (the root CLI passes only
    ``out``; use ``--set bench.<key>=...``). ``source`` overrides the 8-atom source frames of
    measurement (a) (tests). ``ctx.dry_run`` writes ``plan.json`` only and returns
    ``status="partial"``.
    """
    bcfg = cfg.bench
    quick = bool(bcfg.quick if quick is None else quick)
    force = bool(bcfg.force if force is None else force)
    letters = parse_tiers(bcfg.tiers if tiers is None else tiers)
    if not letters:
        raise ValueError("no bench tiers selected")
    params = bench_params(cfg, quick)
    seed = int(ctx.seed if ctx.seed is not None else 0)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / BENCH_JSON
    model_path = resolve_model(cfg, model)
    prov = model_provenance(model_path)
    model_sha = str(prov["sha256"])
    ctx.add_input(model_path, "model")

    plan = {
        "out": str(out_dir),
        "bench_json": str(bench_path),
        "model": str(model_path),
        "model_sha256": model_sha,
        "model_label": prov["label"],
        "tiers": letters,
        "measurements": [TIER_NAMES[t] for t in letters],
        "quick": quick,
        "force": force,
        "seed": seed,
        "threads": int(cfg.compute.threads),
        "dtype": cfg.compute.dtype,
        "params": params,
        "dry_run": bool(ctx.dry_run),
    }
    plan_path = ctx.out_dir / PLAN_JSON
    _write_json(plan_path, plan)
    ctx.add_output(plan_path, "json")
    ctx.log(
        model_sha256=model_sha, model_label=prov["label"], head="Default", tiers=letters,
        quick=quick, force=force, threads=int(cfg.compute.threads), dtype=cfg.compute.dtype,
        bench_json=str(bench_path),
    )  # fmt: skip
    if ctx.dry_run:
        return stage_result(ctx, "partial", {"planned": len(letters), "tiers": ",".join(letters)})

    state = load_state(bench_path)
    reset_reason: str | None = None
    if state is not None and (
        state.get("model", {}).get("sha256") != model_sha or bool(state.get("quick")) != quick
    ):
        reset_reason = "model or quick flag differs from the cached bench.json"
        stale = bench_path.with_name(f"bench.stale-{datetime.now(UTC):%Y%m%dT%H%M%S}.json")
        shutil.move(bench_path, stale)
        state = None
    if state is None:
        state = new_state(cfg, model_path, model_sha, str(prov["label"]), quick)
        if reset_reason:
            state["notes"] = [reset_reason]
    state["threads"], state["dtype"] = int(cfg.compute.threads), cfg.compute.dtype

    set_threads(cfg)
    calc: Any = None
    cell: Atoms | None = None

    def get_calc() -> Any:
        nonlocal calc
        if calc is None:
            calc = make_bench_calculator(model_path, cfg)
            state["model"]["head"] = str(calc.head)
            ctx.log(model_head=str(calc.head))
        return calc

    def get_cell() -> Atoms:
        nonlocal cell
        if cell is None:
            cell = reference_cell(cfg, params["compound"])
        return cell

    done: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    t_stage = time.perf_counter()
    for letter in letters:
        name = TIER_NAMES[letter]
        if not force and measurement_cached(state, name, params):
            skipped.append(name)
            _progress(f"{name}: cached in {bench_path} (use bench.force=true to re-measure)")
            continue
        t0 = time.perf_counter()
        entry: dict[str, Any]
        try:
            if name == "finetune":
                cached = state["measurements"].get(name)
                reusable = (
                    not force
                    and isinstance(cached, dict)
                    and cached.get("params") == _params_subset(params, name)
                )
                entry = bench_finetune(
                    cfg, model_path=model_path, calc=get_calc(), params=params,
                    work_dir=out_dir / "finetune", seed=seed, source=source,
                    previous=cached if reusable else None,
                )  # fmt: skip
            elif name == "md":
                entry = bench_md(cfg, calc=get_calc(), atoms=get_cell(), params=params, seed=seed)
            elif name == "relax":
                structures, src = wbm_structures(cfg, int(params["n_relax"]))
                entry = bench_relax(cfg, calc=get_calc(), structures=structures, source=src)
            elif name == "phonons":
                entry = bench_phonons(
                    cfg, calc=get_calc(), atoms=get_cell(), params=params, run_id=ctx.run_id
                )
            else:
                entry = bench_fwbw(
                    cfg, model_path=model_path, atoms=get_cell(), params=params,
                    work_dir=out_dir / "fwbw", seed=seed,
                )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - recorded in bench.json, never a number
            log.exception("bench %s failed", name)
            entry = {"error": repr(exc), "traceback": traceback.format_exc()}
        entry["params"] = _params_subset(params, name)
        entry["wall_s"] = time.perf_counter() - t0
        entry["measured_at"] = utc_now()
        entry["run_id"] = ctx.run_id
        state["measurements"][name] = entry
        (failed if "error" in entry else done).append(name)
        state["updated_at"] = utc_now()
        state["status"] = "partial"
        _write_json(bench_path, state)
        _progress(f"{name}: {'FAILED' if 'error' in entry else 'done'} in {entry['wall_s']:.1f} s")

    state["schedule"] = derive_schedule(state["measurements"], cfg)
    requested = [TIER_NAMES[t] for t in letters]
    # this run is ok when every requested sub-task has a number; the bench file is ok only when
    # all five measurements are present without an error
    run_status: Status = (
        "ok" if all(_ok(state["measurements"], n) is not None for n in requested) else "partial"
    )
    state["status"] = (
        "ok"
        if all(_ok(state["measurements"], n) is not None for n in TIER_NAMES.values())
        else "partial"
    )
    state["runs"].append(
        {
            "run_id": ctx.run_id,
            "manifest": str(ctx.manifest_path),
            "tiers": requested,
            "done": done,
            "failed": failed,
            "skipped": skipped,
            "status": run_status,
            "wall_s": time.perf_counter() - t_stage,
        }
    )
    state["updated_at"] = utc_now()
    _write_json(bench_path, state)
    ctx.add_output(bench_path, "json")
    numbers = bench_numbers(state, seed)
    write_numbers(ctx, numbers)
    report_path = ctx.out_dir / "bench_report.md"
    report_path.write_text(bench_report(state), encoding="utf-8")
    ctx.add_output(report_path, "other")
    ctx.log(
        tiers_done=done, tiers_failed=failed, tiers_skipped=skipped,
        bench_sha256=sha256_file(bench_path), bench_status=state["status"], run_status=run_status,
        numbers_keys=sorted(k for k in numbers if not k.endswith("@meta")),
        schedule={k: v.get("value") for k, v in state["schedule"].items()},
    )  # fmt: skip
    summary: dict[str, Any] = {
        "status": run_status,
        "bench_status": state["status"],
        "tiers": ",".join(letters),
        "done": ",".join(done),
        "failed": ",".join(failed),
        "skipped": ",".join(skipped),
        "bench_json": str(bench_path),
        "model_label": prov["label"],
    }
    for key in (
        "bench.finetune.s_per_frame_8atom", "bench.finetune.s_per_frame_64atom",
        "bench.md.s_per_step_64", "bench.md.s_per_step_512", "bench.relax.s_per_structure",
        "bench.relax.steps_per_structure", "bench.fwbw.peak_rss_gb",
    ):  # fmt: skip
        if key in numbers:
            summary[key] = numbers[key]
    for key in numbers:
        if key.startswith("bench.phonons_") and not key.endswith("@meta"):
            summary[key] = numbers[key]
    return stage_result(ctx, run_status, summary)


__all__ = [
    "ALL_TIERS",
    "BENCH_JSON",
    "QUICK_PARAMS",
    "SIZE_CLASSES",
    "TIER_NAMES",
    "bench_finetune",
    "bench_fwbw",
    "bench_md",
    "bench_numbers",
    "bench_params",
    "bench_phonons",
    "bench_relax",
    "bench_report",
    "derive_schedule",
    "finetune_frames",
    "fwbw_measure",
    "load_state",
    "make_bench_calculator",
    "measurement_cached",
    "parse_epoch_walls",
    "parse_step_times",
    "parse_tiers",
    "reference_cell",
    "resolve_model",
    "ru_maxrss_bytes",
    "run",
    "source_frames",
    "split_frames",
    "time_finetune_epochs",
    "wbm_structures",
    "zero_shot_label",
]
