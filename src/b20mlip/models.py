"""All pydantic models and literal enums (CONTRACTS.md section 2, verbatim).

Every model derives from :class:`B20Model`, which forbids unknown fields so that a typo in a
manifest, config or frame never passes silently. Field names, types and defaults are the
contract shared by all tiers; do not rename without updating CONTRACTS.md.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# --- literal enums -------------------------------------------------------------------------

Tier = Literal["T0", "T1", "T2", "T3", "T4a", "T4b"]
ConfigType = Literal[
    "relax",
    "strain",
    "shear",
    "rattle",
    "eos",
    "md",
    "vacancy",
    "phonon_disp",
    "noise_floor",
    "offset",
    "wbm",
    "omat24",
]
LabelSource = Literal["mptrj", "omat24", "qe", "pyscf", "mace_zero_shot", "none"]
EnergyScale = Literal["mp", "omat24", "qe", "none"]
Head = Literal["Default", "pt_head"]
StageName = Literal[
    "data.pull",
    "data.sample",
    "data.filter",
    "data.split",
    "dft.converge",
    "dft.prep",
    "dft.run",
    "dft.collect",
    "dft.phonons",
    "dft.e0s",
    "dft.offsets",
    "train",
    "export",
    "eval.errors",
    "eval.discovery",
    "eval.phonons",
    "eval.elastic",
    "md.ase",
    "md.lammps",
    "md.parity",
    "sampling.neb",
    "sampling.umbrella",
    "sampling.wham",
    "active.select",
    "agent.run",
    "agent.eval",
    "report",
    "bench",
    "cluster.bootstrap",
    "cluster.sync",
]
Status = Literal["ok", "failed", "partial"]
ArtifactKind = Literal["frames", "model", "json", "log", "traj", "yaml", "other"]


class B20Model(BaseModel):
    """Shared base: unknown fields are an error everywhere."""

    model_config = ConfigDict(extra="forbid")


# --- frames and splits -----------------------------------------------------------------------


class Frame(B20Model):
    frame_id: str  # sha256(numbers,positions,cell)[:16]
    group_id: str  # "<compound>/<config_type>/<lineage>"
    compound: str
    config_type: ConfigType
    parent_id: str
    numbers: list[int]
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: tuple[bool, bool, bool] = (True, True, True)
    energy: float | None = None
    forces: list[list[float]] | None = None
    stress: list[float] | None = None  # Voigt 6, eV/Å^3
    magmoms: list[float] | None = None
    total_magnetization: float | None = None
    label_source: LabelSource = "none"
    energy_scale: EnergyScale = "none"
    temperature_K: float | None = None
    weights: dict[str, float] = {}  # -> info["config_*_weight"]
    info: dict[str, Any] = {}


class Split(B20Model):
    split_id: str
    seed: int
    frames_sha256: str
    policy: Literal["group_hash", "group_hash_v2"] = "group_hash"
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    train: list[str]  # frame_ids
    val: list[str]
    test: list[str]
    tiers: dict[Tier, list[str]]
    groups: dict[str, list[str]]
    holdout_compounds: list[str] = ["FeGe", "MnGe"]
    max_train_T: float = 600.0


# --- provenance ------------------------------------------------------------------------------


class Artifact(B20Model):
    path: str
    sha256: str
    bytes: int
    kind: ArtifactKind


class SlurmInfo(B20Model):
    job_ids: list[str]
    account: str
    partition: str
    nodes: int
    wall: str
    units_done: int
    units_failed: int


class Manifest(B20Model):
    schema_version: Literal[1] = 1
    stage: StageName
    run_id: str
    created_at: datetime
    host: str
    user: str
    git_sha: str
    git_dirty: bool
    uv_lock_sha256: str
    python: str
    packages: dict[str, str]
    config_sha256: str
    config: dict[str, Any]
    seed: int | None
    inputs: list[Artifact]
    outputs: list[Artifact]
    wall_seconds: float
    status: Status
    slurm: SlurmInfo | None = None
    extras: dict[str, Any] = {}


# --- DFT -------------------------------------------------------------------------------------


class DFTFrame(Frame):
    code: Literal["qe", "pyscf", "vasp"]
    functional: Literal["PBE", "PBEsol"]
    pseudo_md5s: dict[str, str]
    ecut_ry: float
    k_spacing: float
    nspin: int
    smearing: str
    degauss_ry: float
    converged: bool
    scf_steps: int
    abs_magnetization: float | None
    fermi_eV: float | None
    branch_ok: bool | None
    wall_seconds: float
    unit_id: str


# --- training --------------------------------------------------------------------------------


class CheckpointInfo(B20Model):
    model_path: str
    sha256: str
    lammps_path: str | None
    variant: Literal["naive", "replay", "scratch", "bootstrap", "zero_shot"]
    foundation: str | None
    foundation_sha256: str | None
    heads: list[Head]
    e0_source: Literal["foundation", "estimated", "E0s_qe.json", "E0s_scratch.json"]
    energy_scale: EnergyScale
    seed: int
    epochs: int
    lr: float
    batch_size: int
    split_id: str | None
    replay_samples: int | None
    train_run_id: str | None
    val_metrics: dict[str, float] = {}


# --- evaluation ------------------------------------------------------------------------------


class Reference(B20Model):
    code: Literal["qe", "vasp", "abinit", "pyscf", "experiment", "mace"]
    functional: str | None
    pseudos: str | None
    e0_source: str | None
    cross_functional: bool = False


class Metric(B20Model):
    value: float
    ci95: tuple[float, float] | None
    n: int
    unit: str


class ErrorTable(B20Model):
    model_sha256: str
    model_label: str
    head: Head
    tier: Tier
    split_id: str
    reference: Reference
    n_frames: int
    n_atoms: int
    metrics: dict[str, Metric]  # mae_e, rmse_e, mae_f, rmse_f, mae_s
    by_config_type: dict[str, dict[str, Metric]] = {}
    bootstrap_n: int
    bootstrap_seed: int
    noise_floor_f: float | None = None
    run_id: str


class PhononResult(B20Model):
    compound: str
    source_label: str
    reference: Reference
    supercell: list[int]
    displacement: float
    cell_source: Literal["dft", "model_relaxed"]
    qpath_labels: list[str]
    qpoints: list[list[float]]
    frequencies_meV: list[list[float]]
    dos_meV: list[float] | None
    dos: list[float] | None
    imaginary_count: int
    softening_index: float | None
    omega_mae_meV: float | None
    run_id: str


class MDResult(B20Model):
    engine: Literal["ase", "lammps"]
    model_sha256: str
    head: Head
    compound: str
    natoms: int
    ensemble: Literal["nve", "nvt", "npt"]
    temperature_K: float
    pressure_GPa: float | None
    timestep_fs: float
    steps: int
    drift_meV_atom_ps: float | None
    a_mean_A: float | None
    a_std_A: float | None
    alpha_per_K: float | None
    traj_sha256: str
    vdos_path: str | None
    rdf_path: str | None
    run_id: str


# --- agent -----------------------------------------------------------------------------------


class Budget(B20Model):
    max_calls: int = 25
    max_wall_s: int = 1800
    max_dft_frames: int = 200
    max_node_hours: float = 10
    max_atoms: int = 512
    max_md_ps: float = 20
    approve_cluster: bool = False


class AgentTask(B20Model):
    task_id: str
    prompt: str
    tools_allowed: list[str]
    budget: Budget = Budget()
    gold: dict[str, Any] | None = None
    tolerance: dict[str, float] = {}
    injected_failure: (
        Literal["tool_error", "bad_structure", "budget_exhausted", "sum_rule"] | None
    ) = None


class NumberRef(B20Model):
    key: str
    value: float
    run_id: str
    manifest_sha256: str


class AgentReport(B20Model):
    task_id: str
    backend: Literal["anthropic", "mock", "scripted"]
    model_id: str | None
    answer: dict[str, Any]
    numbers: list[NumberRef]
    tool_calls: int
    invalid_calls: int
    dag_valid: bool
    grounded: bool
    violations: list[str]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    wall_s: float
    trace_path: str


# --- stage results ---------------------------------------------------------------------------


class StageResult(B20Model):
    stage: StageName
    run_id: str
    manifest_path: str
    status: Status
    outputs: list[Artifact]
    summary: dict[str, float | int | str] = {}


__all__ = [
    "AgentReport",
    "AgentTask",
    "Artifact",
    "ArtifactKind",
    "B20Model",
    "Budget",
    "CheckpointInfo",
    "ConfigType",
    "DFTFrame",
    "EnergyScale",
    "ErrorTable",
    "Frame",
    "Head",
    "LabelSource",
    "MDResult",
    "Manifest",
    "Metric",
    "NumberRef",
    "PhononResult",
    "Reference",
    "SlurmInfo",
    "Split",
    "StageName",
    "StageResult",
    "Status",
    "Tier",
]
