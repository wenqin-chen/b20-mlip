"""Settings and ``load_config`` (CONTRACTS.md section 4).

Merge order, lowest to highest priority::

    configs/default.yaml -> --config YAMLs (in order) -> environment (B20_*) -> --set key=value

Every section is a pydantic model with ``extra="forbid"``; a misspelt key anywhere is an error.
``Settings.sha256()`` hashes the fully resolved configuration and is the manifest's
``config_sha256`` (CONTRACTS.md section 5).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

from b20mlip.models import B20Model, Budget

ENV_PREFIX = "B20_"
ENV_NESTED_DELIMITER = "__"
DEFAULT_CONFIG_RELPATH = Path("configs") / "default.yaml"


def repo_root(start: Path | None = None) -> Path:
    """Return the checkout root (the directory holding ``pyproject.toml`` and ``configs/``).

    Searches upward from ``start`` (default: this file, which works for an editable install),
    then from the current working directory; falls back to the current working directory.
    """
    candidates = [Path(__file__).resolve()] if start is None else [Path(start).resolve()]
    candidates.append(Path.cwd().resolve())
    for origin in candidates:
        for cand in [origin, *origin.parents]:
            if (cand / "pyproject.toml").is_file() and (cand / "configs").is_dir():
                return cand
    return Path.cwd().resolve()


def default_config_path() -> Path:
    """Location of ``configs/default.yaml`` for this checkout."""
    return repo_root() / DEFAULT_CONFIG_RELPATH


# --- sections (CONTRACTS.md section 4; values from SPEC.md) ------------------------------------


class PathsConfig(B20Model):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    models_dir: Path = Path("models")
    dft_dir: Path = Path("dft")
    reports_dir: Path = Path("reports")


class ComputeConfig(B20Model):
    threads: int = 6
    dtype: Literal["float32", "float64"] = "float64"
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class ClusterConfig(B20Model):
    alias: str = "tillicum"
    control_path: str = "~/.ssh/tillicum-cm"
    account: str | None = None
    partition_cpu: str | None = None
    partition_gpu: str | None = None
    scratch: str | None = None
    modules: list[str] = []
    qe_cmd: str | None = None
    lammps_cmd: str | None = None
    micromamba_env: str | None = None
    # cluster tier (SPEC.md section 8). `qos` is discovered (the association's DefaultQOS);
    # the rest are build knobs for `b20mlip cluster bootstrap`, never discovered.
    qos: str | None = None
    install_qe: bool = False  # run the micromamba QE fallback (else it is only planned)
    lammps_sha: str | None = "4d222cb3ee2a6b14083c778968497bf9e0efc4b4"  # ACEsuit/lammps mace
    libtorch_cpu_url: str = (
        "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.14.0%2Bcpu.zip"
    )
    libtorch_cuda_url: str = (
        "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.14.0%2Bcu126.zip"
    )
    # per-template SLURM resource defaults (ntasks, cpus_per_task, gpus, mem, time, max_parallel,
    # partition, ...), e.g. {"qe_array": {"ntasks": 2, "gpus": 1, "mem": "30G"}}; a stage's explicit
    # resources override them. Site-specific: set in the cluster overlay, never here.
    resources: dict[str, dict[str, Any]] = {}


class DataConfig(B20Model):
    compounds: list[str] = ["FeSi", "CoSi", "MnSi", "FeGe"]
    strains: list[float] = [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06]
    shears: list[float] = [-0.04, -0.02, 0.02, 0.04]
    rattle_A: float = 0.15
    eos_pct: float = 8.0
    temperatures_K: list[float] = [300.0, 600.0, 900.0]
    force_cap_eVA: float = 15.0
    omat24_url: str = "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/251210/omat24_1M_251210.tar.gz"
    omat24_sha256: str | None = None
    wbm_sample_n: int = 1000
    wbm_seed: int = 0


class DFTThresholds(B20Model):
    E_meV_atom: float = 1.0
    F_meV_A: float = 5.0


class DFTConfig(B20Model):
    """QE PBE settings. Cutoffs, pseudopotential file names and md5s come from the SSSP efficiency
    1.3.0 PBE metadata (``configs/dft/sssp_efficiency_1.3_pbe.json``, Materials Cloud record
    rcyfm-68h65, retrieved 2026-09-17): ``ecut_ry``/``ecut_rho`` are the maxima over
    Fe/Mn/Co/Si/Ge (Fe: 90/1080 Ry, dual 12) until ``dft converge`` says otherwise."""

    ecut_ry: float = 90.0
    ecut_rho: float = 1080.0
    k_spacing_inv_A: float = 0.25
    smearing: str = "mv"
    degauss_ry: float = 0.01
    nspin: dict[str, int] = {"FeSi": 1, "CoSi": 1, "MnSi": 2, "FeGe": 2, "MnGe": 2}
    starting_magnetization: dict[str, float] = {"Mn": 0.5, "Fe": 0.5, "Co": 0.0}
    pseudo_dir: str = "pseudos/sssp_efficiency"
    pseudo_family: str = "SSSP-efficiency-1.3"
    pseudos: dict[str, str] = {
        "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
        "Mn": "mn_pbe_v1.5.uspp.F.UPF",
        "Co": "Co_pbe_v1.2.uspp.F.UPF",
        "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
        "Ge": "ge_pbe_v1.4.uspp.F.UPF",
    }
    pseudo_md5s: dict[str, str] = {
        "Fe": "e86618425769142926afa95317d90200",
        "Mn": "82ef2b46521d7a7d9e736dc3972e4928",
        "Co": "5f91765df6ddd3222702df6e7b74a16d",
        "Si": "0b0bb1205258b0d07b9f9672cf965d36",
        "Ge": "9c9eaa91e581c3f09632fb3098b2c6b2",
    }
    sssp_json: str = "configs/dft/sssp_efficiency_1.3_pbe.json"
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.4
    electron_maxstep: int = 100
    branch_tol_muB: float = 0.3
    m_ref_muB: dict[str, float] = {}
    noise_floor_n: int = 20
    isolated_atom_box_A: float = 12.0
    converge_ecuts_ry: list[float] = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    converge_k_spacings: list[float] = [0.35, 0.3, 0.25, 0.2]
    thresholds: DFTThresholds = DFTThresholds()


class ReplayConfig(B20Model):
    pt_train_file: str = "mp"
    num_samples_pt: int = 10000
    subselect_pt: Literal["fps", "random"] = "fps"


class ScratchConfig(B20Model):
    hidden_irreps: str = "64x0e+64x1o"
    r_max: float = 5.0


class TrainConfig(B20Model):
    foundation: str = "medium-mpa-0"
    lr: float = 1.0e-4
    epochs: int = 30
    batch_size: int = 4
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    stress_weight: float = 1.0
    ema_decay: float = 0.99
    replay: ReplayConfig = ReplayConfig()
    e0s_file: str = "configs/dft/E0s_qe.json"
    scratch: ScratchConfig = ScratchConfig()
    # mace_run_train --loss; "universal" (Huber on E/F/stress) is what MACE forces for multihead
    # replay, so every bracket trains with the same loss unless overridden.
    loss: Literal["weighted", "stress", "huber", "universal", "forces_only", "ef"] = "universal"
    # Extra mace_run_train flags appended verbatim (e.g. the tiny CI architecture:
    # ["--num_interactions", "1", "--max_ell", "1", "--correlation", "2"]).
    extra_args: list[str] = []


class EvalConfig(B20Model):
    bootstrap_n: int = 2000
    bootstrap_seed: int = 0
    fmax: float = 0.05
    max_steps: int = 500
    offset_residual_gate_meV: float = 20.0


class MDConfig(B20Model):
    """MD tier (SPEC.md section 5/6). ``friction`` is the ASE Langevin friction in 1/fs; ``taut``
    / ``taup`` (fs) are the ASE NPT thermostat / barostat time constants (``pfactor = taup**2 *
    bulk_modulus``); LAMMPS uses ``100*dt`` / ``1000*dt`` (in.mace.j2). ``equil_ps`` is excluded
    from every average; ``natoms`` is the default supercell size; ``vdos_every_steps`` samples
    velocities for the VDOS (10 fs at 2 fs steps resolves 60 meV phonons)."""

    timestep_fs: float = 2.0
    friction: float = 0.01
    taut: float = 100.0
    taup: float = 1000.0
    equil_ps: float = 10.0
    natoms: int = 64
    bulk_modulus_GPa: float = 150.0
    thermo_every_steps: int = 10
    dump_every_steps: int = 100
    vdos_every_steps: int = 5
    rdf_rmax_A: float = 6.0
    rdf_nbins: int = 200


class SamplingConfig(B20Model):
    k_eVA2: float = 5.0
    windows: int = 12
    ps_per_window: float = 15.0
    T: float = 300.0


class AgentConfig(B20Model):
    """Agent tier (SPEC.md section 7). ``trace_dir`` holds recorded traces (``<task_id>.jsonl``,
    what ``agent run --record`` writes and the mock backend replays); ``tasks_path`` the 12-task
    eval; ``refs_dir`` the reference ``phonons_<compound>_<label>.json`` files ``compare_phonons``
    reads; ``phonon_supercell``/``phonon_distance`` the defaults the ``phonons`` tool uses when
    the caller passes ``[]``/``0``; ``max_tokens`` the per-turn output cap of the live backend."""

    backend: Literal["anthropic", "mock", "scripted"] = "mock"
    model_id: str = "claude-opus-5"
    budget: Budget = Budget()
    trace_dir: Path = Path("runs/agent/traces")
    tasks_path: Path = Path("evals/agent_tasks.jsonl")
    refs_dir: Path = Path("evals/refs")
    phonon_supercell: list[int] = [2, 2, 2]
    phonon_distance: float = 0.03
    max_tokens: int = 16000


class ReportConfig(B20Model):
    numbers_path: Path = Path("reports/numbers.json")


class BenchConfig(B20Model):
    """Day-1 timing task (SPEC.md section 15; ``b20mlip bench``). The root CLI takes only
    ``--out``, so every knob here is set with ``--set bench.<key>=<value>``: ``model`` is the MACE
    file to time (``null`` = the foundation of ``train.foundation``), ``tiers`` the comma list of
    measurements ``a`` fine-tune epoch, ``b`` MD, ``c`` WBM relaxations, ``d`` phonons, ``e``
    forward+backward RSS (letters or names), ``quick`` scales every size down for the tiny CI
    model, ``force`` re-measures sub-tasks already cached in ``<out>/bench.json``."""

    model: str | None = None
    tiers: str = "a,b,c,d,e"
    quick: bool = False
    force: bool = False
    compound: str = "FeSi"
    # (a) one naive fine-tuning epoch per size class, batch train.batch_size
    n_frames_8atom: int = 80
    n_frames_64atom: int = 20
    finetune_epochs: int = 2  # the last epoch is the measurement (epoch 0 carries warm-up)
    finetune_timeout_s: int = 3600
    # (b) NVT Langevin steps at md.timestep_fs; the warm-up steps are excluded
    md_natoms: list[int] = [64, 512]
    md_steps: int = 200
    md_warmup_steps: int = 10
    md_T: float = 300.0
    # (c) FIRE + FrechetCellFilter on the first n WBM-sample structures (eval.fmax / max_steps)
    n_relax: int = 20
    # (d) phonopy finite displacements
    phonon_supercell: int = 2
    phonon_distance: float = 0.03
    # (e) forward + backward (forces) on one training batch, in a fresh process
    fwbw_batch: int = 4
    fwbw_natoms: int = 64
    fwbw_repeats: int = 3


class Settings(BaseSettings):
    """Fully resolved project configuration.

    ``Settings()`` reads only the environment (``B20_SECTION__KEY``); use :func:`load_config`
    for the full merge order. ``model_validate(dict)`` bypasses the environment entirely.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        case_sensitive=False,
        extra="forbid",
        nested_model_default_partial_update=True,
    )

    paths: PathsConfig = PathsConfig()
    compute: ComputeConfig = ComputeConfig()
    cluster: ClusterConfig = ClusterConfig()
    data: DataConfig = DataConfig()
    dft: DFTConfig = DFTConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    md: MDConfig = MDConfig()
    sampling: SamplingConfig = SamplingConfig()
    agent: AgentConfig = AgentConfig()
    report: ReportConfig = ReportConfig()
    bench: BenchConfig = BenchConfig()

    def sha256(self) -> str:
        """Hash of the resolved configuration (canonical JSON, sorted keys)."""
        return sha256_of_json(self.model_dump(mode="json"))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


# --- helpers ---------------------------------------------------------------------------------


def sha256_of_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins, lists and scalars are replaced, not merged."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


_CLOCK_RE = re.compile(r"^\d{1,3}(:\d{2}){1,2}$|^\d+-\d{1,2}(:\d{2}){0,2}$")


def _override_value(raw: str) -> Any:
    """YAML-scalar parse of a ``--set`` value, except that SLURM clock strings (``12:00:00``,
    ``1-00:00:00``) stay strings: YAML 1.1 reads ``12:00:00`` as the sexagesimal integer 43200,
    which sbatch takes as minutes (a 720-hour reservation, Tillicum 2026-09-21)."""
    text = raw.strip()
    if text == "":
        return ""
    if _CLOCK_RE.match(text):
        return text
    return yaml.safe_load(raw)


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=[x,y]"]`` into ``{"a": {"b": 1}, "c": ["x", "y"]}``.

    Values are parsed as YAML scalars (``1`` -> int, ``true`` -> bool, ``null`` -> None,
    ``'1'`` -> str). An empty value is the empty string.
    """
    out: dict[str, Any] = {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--set expects key=value (dotted key), got {item!r}")
        value: Any = _override_value(raw)
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return out


def env_layer() -> dict[str, Any]:
    """The ``B20_*`` environment as a nested dict (prefix and ``__`` delimiter applied)."""
    return dict(EnvSettingsSource(Settings)())


def load_config(
    paths: list[Path] | None = None,
    overrides: list[str] | None = None,
    *,
    default_path: Path | None = None,
    use_env: bool = True,
) -> Settings:
    """Merge ``configs/default.yaml`` -> ``paths`` (in order) -> environment -> ``overrides``.

    A missing ``configs/default.yaml`` falls back to the model defaults (they are identical by
    construction; ``tests/test_config.py`` enforces it).
    """
    data: dict[str, Any] = {}
    default_file = default_path if default_path is not None else default_config_path()
    if default_file.is_file():
        data = read_yaml(default_file)
    for path in paths or []:
        data = deep_merge(data, read_yaml(Path(path)))
    if use_env:
        data = deep_merge(data, env_layer())
    data = deep_merge(data, parse_overrides(list(overrides or [])))
    return Settings.model_validate(data)


__all__ = [
    "AgentConfig",
    "BenchConfig",
    "ClusterConfig",
    "ComputeConfig",
    "DFTConfig",
    "DFTThresholds",
    "DataConfig",
    "ENV_NESTED_DELIMITER",
    "ENV_PREFIX",
    "EvalConfig",
    "MDConfig",
    "PathsConfig",
    "ReplayConfig",
    "ReportConfig",
    "SamplingConfig",
    "ScratchConfig",
    "Settings",
    "TrainConfig",
    "deep_merge",
    "default_config_path",
    "env_layer",
    "load_config",
    "parse_overrides",
    "read_yaml",
    "repo_root",
    "sha256_of_json",
]
