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
    ecut_ry: float = 90.0
    ecut_rho: float = 1080.0
    k_spacing_inv_A: float = 0.25
    smearing: str = "mv"
    degauss_ry: float = 0.01
    nspin: dict[str, int] = {"FeSi": 1, "CoSi": 1, "MnSi": 2, "FeGe": 2, "MnGe": 2}
    starting_magnetization: dict[str, float] = {"Mn": 0.5, "Fe": 0.5, "Co": 0.0}
    pseudo_dir: str = "pseudos/sssp_efficiency"
    pseudo_md5s: dict[str, str] = {}
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.4
    electron_maxstep: int = 100
    branch_tol_muB: float = 0.3
    noise_floor_n: int = 20
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


class EvalConfig(B20Model):
    bootstrap_n: int = 2000
    bootstrap_seed: int = 0
    fmax: float = 0.05
    max_steps: int = 500
    offset_residual_gate_meV: float = 20.0


class MDConfig(B20Model):
    timestep_fs: float = 2.0
    friction: float = 0.01
    taut: float = 100.0
    taup: float = 1000.0
    equil_ps: float = 10.0


class SamplingConfig(B20Model):
    k_eVA2: float = 5.0
    windows: int = 12
    ps_per_window: float = 15.0
    T: float = 300.0


class AgentConfig(B20Model):
    backend: Literal["anthropic", "mock", "scripted"] = "mock"
    model_id: str = "claude-opus-5"
    budget: Budget = Budget()
    trace_dir: Path = Path("runs/agent/traces")


class ReportConfig(B20Model):
    numbers_path: Path = Path("reports/numbers.json")


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
        value: Any = yaml.safe_load(raw) if raw.strip() != "" else ""
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
