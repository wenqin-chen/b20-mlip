"""MACE fine-tuning (CONTRACTS.md row 7): ``build_args`` (the ``mace_run_train`` argv) and ``run``.

Brackets (SPEC.md section 6) map to ``variant``:

* ``naive`` (B1): ``--foundation_model <MPA-0> --multiheads_finetuning False``. E0s follow the
  frames' energy scale (rule R1): QE frames use ``cfg.train.e0s_file`` (``E0s_qe.json``), never
  ``foundation``; MP-scale frames (a smoke test on MPtrj) use ``--E0s foundation``.
* ``replay`` (B2): multihead fine-tuning, QE data in head ``Default`` and an MPtrj replay subset
  in ``pt_head`` (``--pt_train_file mp``, fetched by MACE over the network; never run in tests).
* ``scratch`` (B3): no foundation model, the ``cfg.train.scratch`` architecture and
  ``--E0s average`` (least-squares E0s from the training set; ``e0_source="estimated"``).
* ``bootstrap`` (B4): naive on OMat24 frames with forces + stress only (rule R4): every frame
  is written with ``config_energy_weight=0`` and the global ``--energy_weight`` is ``0``.

Key conventions verified against mace-torch 0.3.16 (``mace/data/utils.py``): frames written by
:func:`b20mlip.io.write_frames` carry ``energy``/``forces``/``stress`` on a
``SinglePointCalculator``; with ``--energy_key energy --forces_key forces --stress_key stress``
MACE's loader pulls them through ``atoms.get_potential_energy()`` etc. into ``REF_*`` keys (with
a benign warning), and per-configuration weights are read from
``atoms.info["config_<property>_weight"]`` — exactly what ``Frame.weights`` becomes. MACE reads
``atoms.info["config_type"]`` too; every one of our config types gets weight 1 via
``--config_type_weights``.

Energy-scale bookkeeping: a training set may carry one labelled scale (``qe``, ``mp`` or
``omat24``); frames with ``energy_scale="none"`` (unlabelled synthetic fixtures) are tolerated
alongside it. Mixing two labelled scales raises (SPEC.md section 2: no mixed-scale training).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from b20mlip.config import Settings, repo_root
from b20mlip.executors import JobSpec, SlurmExecutor
from b20mlip.io import read_frames, write_frames
from b20mlip.models import CheckpointInfo, EnergyScale, Frame, Head, Split
from b20mlip.provenance import RunContext, sha256_file, sha256_frames

Variant = Literal["naive", "replay", "scratch", "bootstrap"]
E0Source = Literal["foundation", "estimated", "E0s_qe.json", "E0s_scratch.json"]
VARIANTS: tuple[Variant, ...] = ("naive", "replay", "scratch", "bootstrap")

# What b20mlip.io.write_frames emits (ASE SinglePointCalculator results), and what MACE 0.3.16
# rewrites to REF_* internally. Do not change without re-checking mace/data/utils.py.
ENERGY_KEY = "energy"
FORCES_KEY = "forces"
STRESS_KEY = "stress"
CONFIG_TYPE_WEIGHTS = '{"Default":1.0}'

MACE_TRAIN_SCRIPT = "mace_run_train"
MACE_TRAIN_MODULE = "mace.cli.run_train"
DATA_DIR = "data"
LOG_DIR = "logs"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
MODEL_DIR = "models"
DOWNLOADS_DIR = "downloads"
STDOUT_LOG = "mace_stdout.log"
PLAN_NAME = "plan.json"
CHECKPOINT_JSON = "checkpoint.json"
SLURM_TEMPLATE = "train_replay"
SLURM_JOB_DIR = "job"

# MACE foundation checkpoints this project may start from (MIT; SPEC.md section 6 brackets).
# Mirrors mace.calculators.foundations_models.mace_mp_urls for the names we use, so resolving a
# local copy never imports torch. The cache file name is MACE's: the URL basename with every
# character that is not alphanumeric or "_" removed.
FOUNDATION_URLS: dict[str, str] = {
    "medium-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
}
FOUNDATION_SHA256: dict[str, str] = {
    "medium-mpa-0": "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
}
ZERO_SHOT_NAMES: dict[str, str] = {"mpa-0": "medium-mpa-0", "mp-0": "medium"}
FOUNDATION_SUBDIR = "foundation"


# --- foundation checkpoints --------------------------------------------------------------------


def mace_cache_dir() -> Path:
    """Where ``mace_mp`` caches downloads (``$XDG_CACHE_HOME/mace`` or ``~/.cache/mace``)."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mace"


def cache_name(url: str) -> str:
    return "".join(c for c in os.path.basename(url) if c.isalnum() or c == "_")


def foundation_candidates(name: str, cfg: Settings) -> list[Path]:
    """Local places a foundation checkpoint may live, most authoritative first."""
    if name not in FOUNDATION_URLS:
        return [Path(name)]
    url = FOUNDATION_URLS[name]
    base = os.path.basename(url)
    models_dir = Path(cfg.paths.models_dir)
    return [
        models_dir / FOUNDATION_SUBDIR / base,
        repo_root() / models_dir / FOUNDATION_SUBDIR / base,
        mace_cache_dir() / cache_name(url),
        mace_cache_dir() / base,
    ]


def resolve_foundation(cfg: Settings, name: str | None = None) -> Path:
    """Return the local file for ``cfg.train.foundation`` (a mace_mp name or a path).

    Raises ``FileNotFoundError`` naming the download URL; stages never download weights
    themselves (the user runs ``data pull`` or ``mace_mp`` once, then the file is cached).
    """
    name = name or cfg.train.foundation
    for cand in foundation_candidates(name, cfg):
        if cand.is_file():
            return cand.resolve()
    url = FOUNDATION_URLS.get(name)
    hint = f" (download {url} into {cfg.paths.models_dir}/{FOUNDATION_SUBDIR}/)" if url else ""
    raise FileNotFoundError(f"foundation model {name!r} not found locally{hint}")


def e0_policy(variant: Variant, energy_scale: EnergyScale, cfg: Settings) -> tuple[str, E0Source]:
    """``(--E0s value, CheckpointInfo.e0_source)`` for a variant and the frames' energy scale.

    Rule R1 (SPEC.md section 4): QE-scale frames train with ``E0s_qe.json``; ``foundation`` E0s
    are only ever used on the foundation's own scale (``mp``) or when energies are not trained
    at all (``bootstrap``).
    """
    if variant == "scratch":
        return "average", "estimated"
    if variant == "bootstrap":
        return "foundation", "foundation"
    if energy_scale == "qe":
        return cfg.train.e0s_file, "E0s_qe.json"
    if energy_scale == "mp":
        return "foundation", "foundation"
    raise ValueError(
        f"{variant} fine-tuning needs frames on the 'qe' or 'mp' energy scale, got "
        f"{energy_scale!r} (OMat24 energies are on a third scale: use variant='bootstrap', "
        "which trains forces and stress only)"
    )


def check_variant(variant: str) -> Variant:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    return variant


# --- argv ----------------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return repr(float(value))


def build_args(
    cfg: Settings,
    variant: str,
    split_paths: Mapping[str, str | Path],
    seed: int,
    out_dir: str | Path,
    *,
    energy_scale: EnergyScale = "qe",
    name: str | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    resume: bool = False,
    foundation: str | Path | None = None,
    e0s_file: str | Path | None = None,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """The exact ``mace_run_train`` argument list (without the program name).

    ``split_paths`` maps ``"train"``/``"valid"`` (required) and ``"test"`` (optional) to extxyz
    files. All MACE output directories live under ``out_dir`` (``logs/``, ``results/``,
    ``checkpoints/``, ``models/``); pass ``out_dir="."`` for a job that ``cd``s into its work
    directory first (the SLURM template). ``foundation`` overrides the resolved foundation path
    (a remote job references its staged copy); ``e0s_file`` overrides ``cfg.train.e0s_file``.
    ``extra_args`` are appended after ``cfg.train.extra_args`` (argparse keeps the last value).
    """
    v = check_variant(variant)
    if "train" not in split_paths or "valid" not in split_paths:
        raise ValueError("split_paths needs 'train' and 'valid' (and optionally 'test')")
    out = Path(out_dir)
    train = cfg.train
    e0s_value, _ = e0_policy(v, energy_scale, cfg)
    if v in ("naive", "replay") and e0s_value == train.e0s_file and e0s_file is not None:
        e0s_value = str(e0s_file)
    lr = train.lr if lr is None else float(lr)
    epochs = train.epochs if epochs is None else int(epochs)
    energy_weight = 0.0 if v == "bootstrap" else train.energy_weight
    loss = "universal" if v == "replay" else train.loss  # MACE forces universal for multihead

    args: list[str] = [
        "--name", name or f"b20_{v}",
        "--seed", str(int(seed)),
        "--work_dir", str(out),
        "--log_dir", str(out / LOG_DIR),
        "--results_dir", str(out / RESULTS_DIR),
        "--checkpoints_dir", str(out / CHECKPOINTS_DIR),
        "--model_dir", str(out / MODEL_DIR),
        "--downloads_dir", str(out / DOWNLOADS_DIR),
        "--train_file", str(split_paths["train"]),
        "--valid_file", str(split_paths["valid"]),
    ]  # fmt: skip
    if split_paths.get("test") is not None:
        args += ["--test_file", str(split_paths["test"])]
    args += [
        "--energy_key", ENERGY_KEY,
        "--forces_key", FORCES_KEY,
        "--stress_key", STRESS_KEY,
        "--config_type_weights", CONFIG_TYPE_WEIGHTS,
        "--loss", loss,
        "--energy_weight", _fmt(energy_weight),
        "--forces_weight", _fmt(train.forces_weight),
        "--stress_weight", _fmt(train.stress_weight),
        "--default_dtype", cfg.compute.dtype,
        "--device", cfg.compute.device,
        "--batch_size", str(train.batch_size),
        "--valid_batch_size", str(train.batch_size),
        "--max_num_epochs", str(epochs),
        "--lr", _fmt(lr),
        "--ema",
        "--ema_decay", _fmt(train.ema_decay),
        "--eval_interval", "1",
        "--save_cpu",
        "--plot", "False",
    ]  # fmt: skip

    if v == "scratch":
        args += [
            "--hidden_irreps", train.scratch.hidden_irreps,
            "--r_max", _fmt(train.scratch.r_max),
            "--E0s", e0s_value,
        ]  # fmt: skip
    else:
        model = str(foundation) if foundation is not None else str(resolve_foundation(cfg))
        args += ["--foundation_model", model]
        if v == "replay":
            args += [
                "--multiheads_finetuning", "True",
                "--pt_train_file", train.replay.pt_train_file,
                "--num_samples_pt", str(train.replay.num_samples_pt),
                "--subselect_pt", train.replay.subselect_pt,
                "--force_mh_ft_lr", "True",  # keep our --lr/--ema_decay instead of MACE's 1e-4
            ]  # fmt: skip
        else:
            args += ["--multiheads_finetuning", "False"]
        args += ["--E0s", e0s_value]
        if e0s_value == "foundation":
            # Energies stay on the foundation scale: keep every foundation element so the
            # checkpoint can still be evaluated on out-of-family structures.
            args += ["--foundation_model_elements", "True"]
    if resume:
        args += ["--restart_latest"]
    args += list(train.extra_args) + list(extra_args or [])
    return args


def train_command(args: Sequence[str]) -> list[str]:
    """Full local command: the venv's Python running ``mace.cli.run_train`` (no PATH lookup)."""
    return [sys.executable, "-m", MACE_TRAIN_MODULE, *args]


def join_argv(args: Sequence[str]) -> str:
    """Shell-quoted argument string for the SLURM template."""
    return shlex.join(args)


# --- frames and splits ---------------------------------------------------------------------------


def frames_energy_scale(frames: Iterable[Frame]) -> EnergyScale:
    """The one labelled energy scale of a frame set (``"none"`` if every frame is unlabelled)."""
    scales = {f.energy_scale for f in frames} - {"none"}
    if len(scales) > 1:
        raise ValueError(
            f"mixed energy scales {sorted(scales)} in one training set (SPEC.md section 2 "
            "forbids mixed-scale training)"
        )
    return next(iter(scales)) if scales else "none"


def _group_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{group_id}|{seed}".encode()).hexdigest()


def group_split(
    frames: Sequence[Frame], seed: int, valid_fraction: float = 0.1
) -> tuple[list[Frame], list[Frame]]:
    """Deterministic train/valid split by ``group_id`` (``sha256(group_id|seed)`` order).

    Used when no ``Split`` exists yet (``--frames`` given directly). Every derivative of one
    parent shares a group, so a rattled copy never leaks into validation. At least one group
    goes to each side when there are two or more groups.
    """
    groups: dict[str, list[Frame]] = {}
    for f in frames:
        groups.setdefault(f.group_id, []).append(f)
    order = sorted(groups, key=lambda g: _group_key(g, seed))
    n_valid = int(round(valid_fraction * len(order)))
    if len(order) >= 2:
        n_valid = min(max(n_valid, 1), len(order) - 1)
    valid_groups = set(order[:n_valid])
    train = [f for g in order if g not in valid_groups for f in groups[g]]
    valid = [f for g in order if g in valid_groups for f in groups[g]]
    return train, valid


def read_split(path: str | Path) -> Split:
    return Split.model_validate_json(Path(path).read_text(encoding="utf-8"))


def find_frames_for_split(split: Split, data_dir: str | Path) -> Path:
    """Locate the extxyz whose dataset hash equals ``split.frames_sha256`` under ``data_dir``."""
    root = Path(data_dir)
    candidates = sorted(root.rglob("*.extxyz")) if root.is_dir() else []
    for cand in candidates:
        if sha256_file(cand) == split.frames_sha256:
            return cand
    for cand in candidates:
        try:
            if sha256_frames(read_frames(cand)) == split.frames_sha256:
                return cand
        except Exception:  # noqa: BLE001 - a foreign extxyz is simply not the one
            continue
    raise FileNotFoundError(
        f"no extxyz under {root} matches split {split.split_id} (frames_sha256 "
        f"{split.frames_sha256[:12]}...); pass --frames explicitly"
    )


def select_split_frames(split: Split, frames: Sequence[Frame]) -> dict[str, list[Frame]]:
    by_id = {f.frame_id: f for f in frames}
    out: dict[str, list[Frame]] = {}
    for part, ids in (("train", split.train), ("valid", split.val), ("test", split.test)):
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(
                f"split {split.split_id}: {len(missing)} {part} frame ids not in the frames "
                f"file (first: {missing[:3]})"
            )
        out[part] = [by_id[i] for i in ids]
    if not out["train"] or not out["valid"]:
        raise ValueError(f"split {split.split_id} has an empty train or valid part")
    return out


def with_energy_weight_zero(frames: Iterable[Frame]) -> list[Frame]:
    """Rule R4: forces + stress only (``config_energy_weight=0`` on every frame)."""
    return [f.model_copy(update={"weights": {**f.weights, "energy": 0.0}}) for f in frames]


def write_split_files(parts: Mapping[str, Sequence[Frame]], data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for part in ("train", "valid", "test"):
        frames = parts.get(part)
        if frames:
            path = data_dir / f"{part}.extxyz"
            write_frames(frames, path)
            paths[part] = path
    return paths


# --- results -------------------------------------------------------------------------------------

_METRIC_KEYS: dict[str, tuple[str, float]] = {
    # MACE results key -> (our key, factor to milli-units); eV -> meV, eV/A -> meV/A
    "rmse_e_per_atom": ("rmse_e_per_atom_meV", 1e3),
    "mae_e_per_atom": ("mae_e_per_atom_meV", 1e3),
    "rmse_f": ("rmse_f_meV_A", 1e3),
    "mae_f": ("mae_f_meV_A", 1e3),
    "rmse_stress": ("rmse_stress_meV_A3", 1e3),
    "mae_stress": ("mae_stress_meV_A3", 1e3),
    "loss": ("loss", 1.0),
}


_RE_LOADED_CKPT = re.compile(r"Loading checkpoint: .*_epoch-(\d+)\.pt")


def exported_epoch(log_path: str | Path | None) -> int | None:
    """The epoch of the checkpoint MACE reloaded before exporting the model (its best validation
    loss), from the training log's last ``Loading checkpoint: ..._epoch-N.pt`` line."""
    if log_path is None or not Path(log_path).is_file():
        return None
    found = _RE_LOADED_CKPT.findall(Path(log_path).read_text(encoding="utf-8", errors="replace"))
    return int(found[-1]) if found else None


def parse_val_metrics(
    results_txt: str | Path, head: str = "Default", *, epoch: int | None = None
) -> dict[str, float]:
    """Validation metrics from MACE's ``results/<tag>_train.txt`` (JSON lines).

    With ``epoch`` given, the ``mode == "eval"`` record of that epoch is used (the checkpoint
    MACE exported: its best validation loss, see :func:`exported_epoch`); otherwise the last
    record wins. Energies and forces are converted to meV / meV/Å; ``epoch`` is included when
    MACE reports one.
    """
    path = Path(results_txt)
    if not path.is_file():
        return {}
    last: dict[str, Any] | None = None
    chosen: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") != "eval":
            continue
        if rec.get("head") not in (None, head):
            continue
        last = rec
        if epoch is not None and rec.get("epoch") == epoch:
            chosen = rec
    last = chosen if chosen is not None else last
    if last is None:
        return {}
    out: dict[str, float] = {}
    for key, (name, factor) in _METRIC_KEYS.items():
        value = last.get(key)
        if isinstance(value, (int, float)):
            out[name] = float(value) * factor
    if isinstance(last.get("epoch"), int):
        out["epoch"] = float(last["epoch"])
    return out


def locate_model(out_dir: str | Path, name: str) -> Path:
    """The final model MACE wrote.

    ``models/<name>_stagetwo.model`` if stage two ran, else ``models/<name>.model``; falls back
    to the checkpoint copy ``checkpoints/<name>_run-*.model``.
    """
    out = Path(out_dir)
    for cand in (out / MODEL_DIR / f"{name}_stagetwo.model", out / MODEL_DIR / f"{name}.model"):
        if cand.is_file():
            return cand
    ckpts = sorted((out / CHECKPOINTS_DIR).glob(f"{name}_run-*.model")) if out.is_dir() else []
    ckpts = [c for c in ckpts if "stagetwo" not in c.name] or ckpts
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"mace_run_train left no model named {name!r} under {out}")


def inspect_model(path: str | Path) -> dict[str, Any]:
    """Heads, cutoff, layers and elements of a saved MACE model (lazy torch import)."""
    import torch  # noqa: PLC0415 - heavy, only when a model is actually inspected

    model = torch.load(str(path), map_location="cpu", weights_only=False)
    heads = getattr(model, "heads", None) or ["Default"]
    return {
        "heads": [str(h) for h in heads],
        "r_max": float(model.r_max.item()),
        "num_interactions": int(model.num_interactions.item()),
        "atomic_numbers": [int(z) for z in model.atomic_numbers.tolist()],
    }


def _tail(path: Path, n: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _env(cfg: Settings) -> dict[str, str]:
    env = dict(os.environ)
    threads = str(cfg.compute.threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = threads
    return env


# --- stage ---------------------------------------------------------------------------------------


def _resolve_inputs(
    cfg: Settings, ctx: RunContext, split: str | Path | None, frames: str | Path | None
) -> tuple[Split | None, Path]:
    split_obj: Split | None = None
    if split is not None:
        split_path = Path(split)
        split_obj = read_split(split_path)
        ctx.add_input(split_path, "json")
    if frames is None:
        if split_obj is None:
            raise ValueError("train needs a split (--split) or an extxyz of frames (--frames)")
        frames_path = find_frames_for_split(split_obj, cfg.paths.data_dir)
    else:
        frames_path = Path(frames)
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames file not found: {frames_path}")
    ctx.add_input(frames_path, "frames")
    return split_obj, frames_path


def _run_local(cmd: list[str], cfg: Settings, out_dir: Path) -> tuple[int, float]:
    log = out_dir / LOG_DIR / STDOUT_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + shlex.join(cmd) + "\n")
        fh.flush()
        # no cwd change: every path in the argv is relative to the caller's working directory
        # (a relative cfg.paths.runs_dir such as "runs/" broke under cwd=out_dir, 2026-09-19)
        proc = subprocess.run(cmd, env=_env(cfg), stdout=fh, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, time.perf_counter() - t0


def _run_slurm(
    cfg: Settings,
    ctx: RunContext,
    executor: SlurmExecutor,
    *,
    variant: Variant,
    name: str,
    split_files: Mapping[str, Path],
    seed: int,
    energy_scale: EnergyScale,
    foundation: Path | None,
    e0s_file: Path | None,
    lr: float | None,
    epochs: int | None,
    extra_args: Sequence[str] | None,
) -> tuple[Path, list[str]]:
    """Stage data + weights next to the sbatch, submit ``train_replay``, wait, fetch results.

    The remote argv is relative to the job's work directory (the template ``cd``s there);
    ``--restart_latest`` makes a resubmission after a timeout resume from MACE's last checkpoint.
    """
    job_name = f"train_{variant}_{ctx.run_id}"
    staging = executor.staging_root / job_name
    (staging / DATA_DIR).mkdir(parents=True, exist_ok=True)
    remote_paths: dict[str, str] = {}
    for part, path in split_files.items():
        shutil.copy2(path, staging / DATA_DIR / path.name)
        remote_paths[part] = f"{DATA_DIR}/{path.name}"
    remote_foundation: str | None = None
    if foundation is not None:
        remote_foundation = f"{FOUNDATION_SUBDIR}.model"
        shutil.copy2(foundation, staging / remote_foundation)
    remote_e0s: str | None = None
    if e0s_file is not None:
        remote_e0s = e0s_file.name
        shutil.copy2(e0s_file, staging / remote_e0s)
    argv = build_args(
        cfg, variant, remote_paths, seed, ".",
        energy_scale=energy_scale, name=name, lr=lr, epochs=epochs, resume=True,
        foundation=remote_foundation, e0s_file=remote_e0s, extra_args=extra_args,
    )  # fmt: skip
    spec = JobSpec(
        name=job_name,
        script=f"uv run --no-sync {MACE_TRAIN_SCRIPT} {join_argv(argv)}",
        units=[ctx.run_id],
        resources={
            "template": SLURM_TEMPLATE,
            "argv": join_argv(argv),
            "partition": cfg.cluster.partition_gpu,
            "gpus": 1,
        },
    )
    handle = executor.submit(spec)
    ctx.log(slurm_job_ids=list(handle.job_ids), slurm_workdir=handle.workdir)
    ctx.slurm = executor.wait(handle)
    job_dir = ctx.out_dir / SLURM_JOB_DIR
    executor.fetch(handle, job_dir)
    return job_dir, argv


def run(
    cfg: Settings,
    ctx: RunContext,
    variant: str,
    split: str | Path | None = None,
    seed: int | None = None,
    *,
    frames: str | Path | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    name: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> CheckpointInfo | None:
    """Stage ``train`` (CONTRACTS.md row 7): frames -> extxyz -> ``mace_run_train`` -> checkpoint.

    Frames come from ``frames`` (extxyz) and are partitioned by the ``Split`` at ``split``
    (train/val/test ids) or, without a split, by an in-function 90/10 group split (no test
    set). With a split but no ``frames``, the extxyz whose hash matches ``split.frames_sha256``
    is looked up under ``cfg.paths.data_dir``. Outputs under ``ctx.out_dir``: ``data/*.extxyz``,
    ``plan.json`` (argv + files), MACE's ``logs/``, ``results/``, ``checkpoints/``, ``models/``
    and ``checkpoint.json`` (the returned ``CheckpointInfo``). ``ctx.dry_run`` writes the plan
    and the extxyz files only and returns ``None``. With a ``SlurmExecutor`` the job runs on the
    cluster (``templates/slurm/train_replay.sbatch.j2``) and results are fetched back.
    """
    v = check_variant(variant)
    seed = int(seed if seed is not None else (ctx.seed if ctx.seed is not None else 0))
    name = name or f"b20_{v}"
    out_dir = ctx.out_dir

    split_obj, frames_path = _resolve_inputs(cfg, ctx, split, frames)
    all_frames = read_frames(frames_path)
    if split_obj is not None:
        parts = select_split_frames(split_obj, all_frames)
        sha_match = split_obj.frames_sha256 in (sha256_frames(all_frames), sha256_file(frames_path))
        ctx.log(split_frames_sha256_match=sha_match)
    else:
        train_frames, valid_frames = group_split(all_frames, seed)
        parts = {"train": train_frames, "valid": valid_frames}
    if not parts["train"] or not parts.get("valid"):
        raise ValueError("training needs at least one train and one validation frame")

    used = [f for part in parts.values() for f in part]
    energy_scale = frames_energy_scale(used)
    if v == "bootstrap":
        parts = {k: with_energy_weight_zero(fr) for k, fr in parts.items()}
        # rule R4: energies carry weight 0, so the model's energy scale stays the foundation's
        # ("mp"); the frames' scale (OMat24) is recorded in the manifest, not on the checkpoint
        ctx.log(frames_energy_scale=energy_scale)
        energy_scale = "mp"
    split_files = write_split_files(parts, out_dir / DATA_DIR)

    foundation: Path | None = None
    foundation_sha: str | None = None
    if v != "scratch":
        foundation = resolve_foundation(cfg)
        foundation_sha = sha256_file(foundation)
        ctx.add_input(foundation, "model")
    e0s_value, e0_source = e0_policy(v, energy_scale, cfg)
    e0s_path: Path | None = None
    if e0_source == "E0s_qe.json":
        e0s_path = Path(e0s_value)
        if not e0s_path.is_file() and (repo_root() / e0s_path).is_file():
            e0s_path = repo_root() / e0s_path
        if not e0s_path.is_file():
            raise FileNotFoundError(
                f"rule R1: QE-scale frames need isolated-atom E0s at {e0s_value} "
                "(run `b20mlip dft e0s`); 'foundation' E0s are never used on the QE scale"
            )
        e0s_path = e0s_path.resolve()
        ctx.add_input(e0s_path, "json")
    resume = bool(ctx.resume) and any((out_dir / CHECKPOINTS_DIR).glob("*.pt"))
    args = build_args(
        cfg, v, split_files, seed, out_dir,
        energy_scale=energy_scale, name=name, lr=lr, epochs=epochs, resume=resume,
        foundation=foundation, e0s_file=e0s_path, extra_args=extra_args,
    )  # fmt: skip
    plan = {
        "variant": v,
        "name": name,
        "seed": seed,
        "energy_scale": energy_scale,
        "e0_source": e0_source,
        "foundation": None if foundation is None else str(foundation),
        "foundation_sha256": foundation_sha,
        "files": {k: str(p) for k, p in split_files.items()},
        "counts": {k: len(fr) for k, fr in parts.items()},
        "command": train_command(args),
        "argv": args,
        "dry_run": bool(ctx.dry_run),
    }
    plan_path = out_dir / PLAN_NAME
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.add_output(plan_path, "json")
    for path in split_files.values():
        ctx.add_output(path, "frames")
    ctx.log(
        variant=v,
        foundation=cfg.train.foundation if foundation is not None else None,
        foundation_sha256=foundation_sha,
        e0_source=e0_source,
        energy_scale=energy_scale,
        n_train=len(parts["train"]),
        n_valid=len(parts["valid"]),
        n_test=len(parts.get("test", [])),
        split_id=None if split_obj is None else split_obj.split_id,
        mace_argv=args,
    )
    if ctx.dry_run:
        return None

    model_root = out_dir
    if isinstance(ctx.executor, SlurmExecutor):
        model_root, remote_argv = _run_slurm(
            cfg, ctx, ctx.executor,
            variant=v, name=name, split_files=split_files, seed=seed,
            energy_scale=energy_scale, foundation=foundation, e0s_file=e0s_path,
            lr=lr, epochs=epochs, extra_args=extra_args,
        )  # fmt: skip
        ctx.log(mace_argv_remote=remote_argv)
    else:
        returncode, wall = _run_local(train_command(args), cfg, out_dir)
        ctx.log(mace_returncode=returncode, mace_wall_seconds=round(wall, 2))
        log_path = out_dir / LOG_DIR / STDOUT_LOG
        if log_path.is_file():
            ctx.add_output(log_path, "log")
        if returncode != 0:
            raise RuntimeError(
                f"mace_run_train exited with {returncode}; last lines of {log_path}:\n"
                + _tail(log_path)
            )

    model_path = locate_model(model_root, name)
    ctx.add_output(model_path, "model")
    compiled = model_path.with_name(model_path.name.replace(".model", "_compiled.model"))
    if compiled.is_file() and compiled != model_path:
        ctx.add_output(compiled, "model")
    for log in sorted((model_root / LOG_DIR).glob("*.log")):
        ctx.add_output(log, "log")
    results = sorted((model_root / RESULTS_DIR).glob(f"{name}_run-{seed}_train.txt"))
    logs = sorted((model_root / LOG_DIR).glob("*.log"))
    best_epoch = exported_epoch(logs[-1]) if logs else None
    val_metrics = parse_val_metrics(results[-1], epoch=best_epoch) if results else {}
    if best_epoch is not None:
        val_metrics["exported_epoch"] = float(best_epoch)
    for res in results:
        ctx.add_output(res, "log")
    details = inspect_model(model_path)
    heads = [cast(Head, h) for h in details["heads"] if h in ("Default", "pt_head")]
    info = CheckpointInfo(
        model_path=str(model_path),
        sha256=sha256_file(model_path),
        lammps_path=None,
        variant=v,
        foundation=cfg.train.foundation if foundation is not None else None,
        foundation_sha256=foundation_sha,
        heads=heads or ["Default"],
        e0_source=e0_source,
        energy_scale=energy_scale,
        seed=seed,
        epochs=cfg.train.epochs if epochs is None else int(epochs),
        lr=cfg.train.lr if lr is None else float(lr),
        batch_size=cfg.train.batch_size,
        split_id=None if split_obj is None else split_obj.split_id,
        replay_samples=cfg.train.replay.num_samples_pt if v == "replay" else None,
        train_run_id=ctx.run_id,
        val_metrics=val_metrics,
    )
    ckpt_path = out_dir / CHECKPOINT_JSON
    ckpt_path.write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ctx.add_output(ckpt_path, "json")
    ctx.log(
        heads=info.heads,
        model_sha256=info.sha256,
        val_metrics=val_metrics,
        model_r_max=details["r_max"],
        model_num_interactions=details["num_interactions"],
        model_elements=details["atomic_numbers"],
        replay_samples=info.replay_samples,
        epochs=info.epochs,
        lr=info.lr,
        batch_size=info.batch_size,
    )
    return info


def zero_shot_checkpoint(cfg: Settings, which: str = "mpa-0") -> CheckpointInfo:
    """Brackets B0 (``"mpa-0"``) and B0' (``"mp-0"``): the foundation model as a checkpoint.

    Nothing is trained or downloaded; the file must already be local (see
    :func:`resolve_foundation`). Energies are on the MP scale with foundation E0s.
    """
    if which not in ZERO_SHOT_NAMES:
        raise ValueError(f"which must be one of {sorted(ZERO_SHOT_NAMES)}, got {which!r}")
    name = ZERO_SHOT_NAMES[which]
    path = resolve_foundation(cfg, name)
    sha = sha256_file(path)
    return CheckpointInfo(
        model_path=str(path),
        sha256=sha,
        lammps_path=None,
        variant="zero_shot",
        foundation=name,
        foundation_sha256=sha,
        heads=["Default"],
        e0_source="foundation",
        energy_scale="mp",
        seed=0,
        epochs=0,
        lr=0.0,
        batch_size=0,
        split_id=None,
        replay_samples=None,
        train_run_id=None,
        val_metrics={},
    )


__all__ = [
    "CONFIG_TYPE_WEIGHTS",
    "ENERGY_KEY",
    "FORCES_KEY",
    "FOUNDATION_SHA256",
    "FOUNDATION_URLS",
    "MACE_TRAIN_MODULE",
    "MACE_TRAIN_SCRIPT",
    "STRESS_KEY",
    "VARIANTS",
    "Variant",
    "build_args",
    "check_variant",
    "e0_policy",
    "find_frames_for_split",
    "foundation_candidates",
    "frames_energy_scale",
    "group_split",
    "inspect_model",
    "join_argv",
    "locate_model",
    "mace_cache_dir",
    "parse_val_metrics",
    "read_split",
    "resolve_foundation",
    "run",
    "select_split_frames",
    "train_command",
    "with_energy_weight_zero",
    "write_split_files",
    "zero_shot_checkpoint",
]
