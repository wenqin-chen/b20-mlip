# b20-mlip — CONTRACTS v1.0 (shared contracts for parallel coding agents)

Python 3.11, `uv`, src layout, package `b20mlip`, pydantic v2 (`extra="forbid"`), typer CLI `b20mlip`. Pinned: mace-torch==0.3.16, ase, phonopy, pymatgen, spglib, seekpath, pymbar, pydantic, pydantic-settings, typer, pyyaml, jinja2, numpy, pandas, httpx, anthropic; extras `[omat]` fairchem-core (OMat24 ASE-LMDB), `[pyscf]`. `matbench_discovery` is **never** a dependency (needs Python 3.14; downloads at import).

## 1 Module map, build order, dependencies
| # | Module | Depends on | Owner deliverable |
|---|---|---|---|
| 0 | `models.py` (all pydantic models, enums) | pydantic, numpy | this file §2 |
| 1 | `config.py` (`Settings`, `load_config`) | 0 | §4 |
| 2 | `provenance.py` (hashing, `Manifest` I/O, `RunContext`, `run_stage`) | 0,1 | §5 |
| 3 | `io.py` (`Frame`↔`ase.Atoms`, extxyz) | 0 | — |
| 4 | `executors.py` (`LocalExecutor`, `SlurmExecutor`, jinja templates) | 1,2 | — |
| 5 | `data/{pull,mptrj,omat24,optimade,wbm,sample,filters,split}.py` | 2,3 | frames, splits |
| 6 | `dft/{qe,e0s,offsets,pyscf_pbc}.py` | 3,4 | `DFTFrame`s |
| 7 | `train/{finetune,export}.py` | 1,2,3 | `CheckpointInfo` |
| 8 | `phonons/harmonic.py` | 3 | `PhononResult` |
| 9 | `evaluate/{errors,bootstrap,mbd_vendored,discovery,phonon_compare,elastic}.py` | 3,8 | `ErrorTable` |
| 10 | `md/{ase_md,lammps,parity}.py` | 3,4,7 | `MDResult` |
| 11 | `sampling/{neb,umbrella,wham}.py` | 10 | ΔF, E_a |
| 12 | `active/committee.py` | 9 | selection |
| 13 | `agent/{tools,guard,trace,backends,runner,eval}.py` | 5–12 | `AgentReport` |
| 14 | `report/{numbers,build,audit}.py` | 2 | `numbers.json`, README |
| 15 | `cli.py`, `bench.py` | all | — |
Build order = table order; a module may import only rows above it. Cross-cutting rule: every public stage function has the signature `run(cfg: Settings, ctx: RunContext, **kw) -> StageResult` and never prints numbers to README directly.

## 2 Models (`b20mlip/models.py`)
```python
Tier = Literal["T0","T1","T2","T3","T4a","T4b"]
ConfigType = Literal["relax","strain","shear","rattle","eos","md","vacancy","phonon_disp","noise_floor","offset","wbm","omat24"]
LabelSource = Literal["mptrj","omat24","qe","pyscf","mace_zero_shot","none"]
EnergyScale = Literal["mp","omat24","qe","none"]
Head = Literal["Default","pt_head"]
StageName = Literal["data.pull","data.sample","data.filter","data.split","dft.converge","dft.prep","dft.run",
  "dft.collect","dft.phonons","dft.e0s","dft.offsets","train","export","eval.errors","eval.discovery",
  "eval.phonons","eval.elastic","md.ase","md.lammps","md.parity","sampling.neb","sampling.umbrella",
  "sampling.wham","active.select","agent.run","agent.eval","report","bench","cluster.bootstrap","cluster.sync"]

class Frame(BaseModel):
    frame_id: str                 # sha256(numbers,positions,cell)[:16]
    group_id: str                 # "<compound>/<config_type>/<lineage>"
    compound: str; config_type: ConfigType; parent_id: str
    numbers: list[int]; positions: list[list[float]]; cell: list[list[float]]
    pbc: tuple[bool,bool,bool] = (True,True,True)
    energy: float|None = None; forces: list[list[float]]|None = None
    stress: list[float]|None = None          # Voigt 6, eV/Å^3
    magmoms: list[float]|None = None; total_magnetization: float|None = None
    label_source: LabelSource = "none"; energy_scale: EnergyScale = "none"
    temperature_K: float|None = None; weights: dict[str,float] = {}   # -> info["config_*_weight"]
    info: dict[str, Any] = {}

class Split(BaseModel):
    split_id: str; seed: int; frames_sha256: str; policy: Literal["group_hash"] = "group_hash"
    fractions: tuple[float,float,float] = (0.8,0.1,0.1)
    train: list[str]; val: list[str]; test: list[str]      # frame_ids
    tiers: dict[Tier, list[str]]; groups: dict[str, list[str]]
    holdout_compounds: list[str] = ["FeGe","MnGe"]; max_train_T: float = 600.0

class Artifact(BaseModel):
    path: str; sha256: str; bytes: int; kind: Literal["frames","model","json","log","traj","yaml","other"]

class SlurmInfo(BaseModel):
    job_ids: list[str]; account: str; partition: str; nodes: int; wall: str; units_done: int; units_failed: int

class Manifest(BaseModel):
    schema_version: Literal[1] = 1
    stage: StageName; run_id: str; created_at: datetime; host: str; user: str
    git_sha: str; git_dirty: bool; uv_lock_sha256: str; python: str; packages: dict[str,str]
    config_sha256: str; config: dict[str, Any]; seed: int|None
    inputs: list[Artifact]; outputs: list[Artifact]
    wall_seconds: float; status: Literal["ok","failed","partial"]
    slurm: SlurmInfo|None = None; extras: dict[str, Any] = {}

class DFTFrame(Frame):
    code: Literal["qe","pyscf","vasp"]; functional: Literal["PBE","PBEsol"]; pseudo_md5s: dict[str,str]
    ecut_ry: float; k_spacing: float; nspin: int; smearing: str; degauss_ry: float
    converged: bool; scf_steps: int; abs_magnetization: float|None; fermi_eV: float|None
    branch_ok: bool|None; wall_seconds: float; unit_id: str

class CheckpointInfo(BaseModel):
    model_path: str; sha256: str; lammps_path: str|None; variant: Literal["naive","replay","scratch","bootstrap","zero_shot"]
    foundation: str|None; foundation_sha256: str|None; heads: list[Head]
    e0_source: Literal["foundation","estimated","E0s_qe.json","E0s_scratch.json"]; energy_scale: EnergyScale
    seed: int; epochs: int; lr: float; batch_size: int; split_id: str|None; replay_samples: int|None
    train_run_id: str|None; val_metrics: dict[str,float] = {}

class Reference(BaseModel):
    code: Literal["qe","vasp","abinit","pyscf","experiment","mace"]; functional: str|None; pseudos: str|None
    e0_source: str|None; cross_functional: bool = False

class Metric(BaseModel):
    value: float; ci95: tuple[float,float]|None; n: int; unit: str

class ErrorTable(BaseModel):
    model_sha256: str; model_label: str; head: Head; tier: Tier; split_id: str; reference: Reference
    n_frames: int; n_atoms: int; metrics: dict[str, Metric]       # mae_e, rmse_e, mae_f, rmse_f, mae_s
    by_config_type: dict[str, dict[str, Metric]] = {}; bootstrap_n: int; bootstrap_seed: int
    noise_floor_f: float|None = None; run_id: str

class PhononResult(BaseModel):
    compound: str; source_label: str; reference: Reference; supercell: list[int]; displacement: float
    cell_source: Literal["dft","model_relaxed"]; qpath_labels: list[str]; qpoints: list[list[float]]
    frequencies_meV: list[list[float]]; dos_meV: list[float]|None; dos: list[float]|None
    imaginary_count: int; softening_index: float|None; omega_mae_meV: float|None; run_id: str

class MDResult(BaseModel):
    engine: Literal["ase","lammps"]; model_sha256: str; head: Head; compound: str; natoms: int
    ensemble: Literal["nve","nvt","npt"]; temperature_K: float; pressure_GPa: float|None; timestep_fs: float; steps: int
    drift_meV_atom_ps: float|None; a_mean_A: float|None; a_std_A: float|None; alpha_per_K: float|None
    traj_sha256: str; vdos_path: str|None; rdf_path: str|None; run_id: str

class Budget(BaseModel):
    max_calls: int = 25; max_wall_s: int = 1800; max_dft_frames: int = 200; max_node_hours: float = 10
    max_atoms: int = 512; max_md_ps: float = 20; approve_cluster: bool = False

class AgentTask(BaseModel):
    task_id: str; prompt: str; tools_allowed: list[str]; budget: Budget = Budget()
    gold: dict[str, Any]|None = None; tolerance: dict[str,float] = {}
    injected_failure: Literal["tool_error","bad_structure","budget_exhausted","sum_rule"]|None = None

class NumberRef(BaseModel): key: str; value: float; run_id: str; manifest_sha256: str
class AgentReport(BaseModel):
    task_id: str; backend: Literal["anthropic","mock","scripted"]; model_id: str|None; answer: dict[str, Any]
    numbers: list[NumberRef]; tool_calls: int; invalid_calls: int; dag_valid: bool; grounded: bool
    violations: list[str]; tokens_in: int; tokens_out: int; cost_usd: float; wall_s: float; trace_path: str

class StageResult(BaseModel):
    stage: StageName; run_id: str; manifest_path: str; status: Literal["ok","failed","partial"]
    outputs: list[Artifact]; summary: dict[str, float|int|str] = {}
```

## 3 CLI (`b20mlip/cli.py`, typer; global `--config PATH` repeatable, `--set key=value`, `--seed INT`, `--resume`, `--executor local|slurm`, `--dry-run`)
```
data pull      --sources mptrj,omat24,wbm,phonondb
data sample    --out data/frames/candidates_r0.extxyz
data filter    --frames F --out F2            # dedupe, force cap, magnetic branch
data split     --frames F --out data/splits/v1.json
dft converge   --compound MnSi
dft prep       --frames F --out dft/r0/       # one unit dir per frame: pw.in, unit.json
dft run        --units dft/r0/ [--limit N]    # per-unit resume; SLURM array or local
dft collect    --units dft/r0/ --out data/frames/labelled_r0.extxyz
dft phonons    --compound FeSi --supercell 2 2 2 --distance 0.03
dft e0s        --out configs/dft/E0s_qe.json
dft offsets    --qe F_qe --mp F_mp --out configs/dft/offsets.json
train          --variant naive|replay|scratch|bootstrap --split S --seed N [--lr] [--epochs]
export         --model M                      # mace_create_lammps_model
eval errors    --model M --head Default|pt_head --split S --tiers T0,T1,T2,T3,T4a
eval discovery --model M --head pt_head --sample data/wbm/sample_1000_s0.json
eval phonons   --model M --compound C --reference qe|phonondb103|pbesol [--cell dft|model]
eval elastic   --model M --compound C
md ase         --model M --compound C --ensemble nvt|npt|nve --T 300 --ps 40 --natoms 64
md lammps      --model M --compound C --T 300 --ps 100 --natoms 512
md parity      --model M --frames F           # 20 frames ASE vs LAMMPS
sampling neb   --model M --images 7 ; sampling umbrella --model M --windows 12 --ps 15 ; sampling wham --run R
active select  --models M1,M2,M3 --frames F --n 100 --out data/frames/candidates_r1.extxyz
screen         --compounds FeSi,MnSi --model M   # deterministic gold script
agent run      (--task TASKS.jsonl:ID | --prompt TEXT) --backend anthropic|mock|scripted --budget B [--approve-cluster]
agent eval     --tasks evals/agent_tasks.jsonl --backend mock|anthropic
cluster bootstrap | sync | status
report build [--readme] ; report audit [--strict]
bench          --out runs/bench/
```
Every command exits 0 only if the `StageResult.status == "ok"`; `--dry-run` writes the manifest with `status="partial"` and no outputs.

## 4 Config (`b20mlip/config.py`, pydantic-settings, env prefix `B20_`, `env_nested_delimiter="__"`)
`load_config(paths: list[Path], overrides: list[str]) -> Settings` merges `configs/default.yaml` → given YAMLs → env → `--set`. Sections (each a BaseModel): `paths{data_dir,runs_dir,models_dir,dft_dir,reports_dir}`, `compute{threads:int=6,dtype:"float64",device:"cpu"}`, `cluster{alias:"tillicum",control_path:"~/.ssh/tillicum-cm",account,partition_cpu,partition_gpu,scratch,modules:list,qe_cmd,lammps_cmd,micromamba_env}`, `data{compounds:list,strains,shears,rattle_A,eos_pct,temperatures_K,force_cap_eVA:15,omat24_url,omat24_sha256,wbm_sample_n:1000,wbm_seed:0}`, `dft{ecut_ry,ecut_rho,k_spacing_inv_A:0.25,smearing:"mv",degauss_ry:0.01,nspin:dict[str,int],starting_magnetization:dict,pseudo_dir,pseudo_md5s:dict,conv_thr,mixing_beta,electron_maxstep,branch_tol_muB:0.3,noise_floor_n:20,thresholds{E_meV_atom:1,F_meV_A:5}}`, `train{foundation:"medium-mpa-0",lr,epochs,batch_size:4,energy_weight,forces_weight,stress_weight,ema_decay,replay{pt_train_file:"mp",num_samples_pt:10000,subselect_pt:"fps"},e0s_file,scratch{hidden_irreps:"64x0e+64x1o",r_max:5.0}}`, `eval{bootstrap_n:2000,bootstrap_seed:0,fmax:0.05,max_steps:500,offset_residual_gate_meV:20}`, `md{timestep_fs:2,friction,taut,taup,equil_ps:10}`, `sampling{k_eVA2:5,windows:12,ps_per_window:15,T:300}`, `agent{backend:"mock",model_id:"claude-opus-5",budget:Budget,trace_dir}`, `report{numbers_path:"reports/numbers.json"}`. `Settings.sha256()` hashes the resolved model dump; it is the manifest's `config_sha256`.

## 5 Provenance (`b20mlip/provenance.py`)
`runs/<stage>/<run_id>/manifest.json` where `run_id = f"{utc:%Y%m%dT%H%M%S}-{config_sha256[:6]}-{seed}"`. JSON Schema (draft 2020-12) = `Manifest.model_json_schema()` committed at `docs/manifest.schema.json`; required: `schema_version, stage, run_id, created_at, host, user, git_sha, git_dirty, uv_lock_sha256, python, packages, config_sha256, config, inputs, outputs, wall_seconds, status`; `inputs[]/outputs[]` require `path, sha256, bytes, kind`; `extras` is free-form but stage contracts add: train → `foundation_sha256, e0_source, heads, energy_scale`; dft.run → `pw_version, pseudo_md5s, n_units, n_failed, n_branch_rejected`; eval.* → `model_sha256, head, tier, n, bootstrap_seed, reference`; md.* → `engine, timestep_fs, thermostat`; agent.* → `backend, model_id, tokens_in, tokens_out, cost_usd`. API: `sha256_file(path) -> str`; `sha256_frames(frames) -> str`; `class RunContext(stage, cfg, seed, resume, executor)` with `ctx.out_dir`, `ctx.add_input(path, kind)`, `ctx.add_output(path, kind)`, `ctx.log(**extras)`; `run_stage(stage, cfg, fn, **kw) -> StageResult` wraps `fn(cfg, ctx, **kw)`, measures wall time, writes the manifest even on failure (`status="failed"`), and returns the result; `read_manifest(path) -> Manifest`; `find_runs(stage) -> list[Manifest]`.

## 6 Function signatures
```python
# io.py
frame_from_atoms(atoms, *, group_id, compound, config_type, parent_id, label_source, energy_scale) -> Frame
frame_to_atoms(frame) -> ase.Atoms      # forces via SinglePointCalculator; weights -> info["config_energy_weight"] etc.
read_frames(path) -> list[Frame]; write_frames(frames, path) -> Artifact
# executors.py
class JobSpec(BaseModel): name; script: str; units: list[str]; resources: dict; env: dict
class JobHandle(BaseModel): job_ids: list[str]; workdir: str
class Executor(Protocol): submit(spec) -> JobHandle; wait(handle, poll_s=60) -> SlurmInfo|None; fetch(handle, dest) -> list[Artifact]
LocalExecutor(cfg); SlurmExecutor(cfg)   # renders templates/slurm/*.sbatch.j2; sacct polling; ssh over cfg.cluster.control_path
# data/
pull.fetch(cfg, ctx, sources) -> StageResult
mptrj.extract_family(zip_path, elements, out) -> list[Frame]; mptrj.b20_frames(zip_path) -> list[Frame]
omat24.stream_filter(tarball, elements, out, force_cap) -> tuple[list[Frame], dict]   # dict: counts, n_b20_spg198
optimade.search(elements, *, nelements=2, spacegroup=198, providers=("mp","alexandria")) -> list[Frame]   # keyless; httpx
wbm.sample(summary_csv, atoms_zip, n, seed, family_elements) -> dict   # ids, prevalence, in_family_ids
sample.generate(cfg, ctx, structures, rng) -> list[Frame]          # strain/shear/rattle/eos/md/vacancy/phonon_disp
filters.dedupe(frames) -> list[Frame]; filters.force_cap(frames, cap) -> list[Frame]
filters.magnetic_branch(frames: list[DFTFrame], m_ref: dict[str,float], tol) -> tuple[list[DFTFrame], dict]
split.group_split(frames, seed, fractions, holdout_compounds, max_train_T) -> Split
# dft/
qe.render_pw_input(frame, cfg) -> str; qe.parse_pw_output(path, frame) -> DFTFrame; qe.unit_dir(frame, root) -> Path
qe.plan_units(frames, root) -> list[str]; qe.collect(root) -> tuple[list[DFTFrame], dict]
e0s.isolated_atoms(cfg, elements) -> dict[str,float]       # writes E0s_qe.json
offsets.fit(qe: list[DFTFrame], mp: list[Frame]) -> dict    # per-element linear map + residual_meV_atom
pyscf_pbc.single_point(frame, cfg) -> DFTFrame
# train/
finetune.build_args(cfg, variant, split_paths, seed, out_dir) -> list[str]   # exact mace_run_train argv
finetune.run(cfg, ctx, variant, split, seed) -> CheckpointInfo
export.to_lammps(model_path) -> str
# phonons/
harmonic.compute(atoms, calc, supercell, distance, *, cell_source, qpath="seekpath") -> PhononResult
harmonic.from_force_sets(atoms, force_sets_json, supercell) -> PhononResult    # QE reference
# evaluate/
errors.evaluate(model_path, head, frames, tier, reference, bootstrap_n, seed) -> ErrorTable
bootstrap.ci(values_by_group, n, seed, stat=np.mean) -> tuple[float,float]
mbd_vendored.stable_metrics(each_true, each_pred, *, stability_threshold=0.0, fillna=True) -> dict[str,float]
mbd_vendored.classify_stable(e_true, e_pred, *, stability_threshold=0.0) -> tuple[...]   # verbatim from janosh/matbench-discovery @ pinned SHA, MIT, attributed
discovery.relax(atoms, calc, fmax=0.05, steps=500) -> tuple[ase.Atoms, int]
discovery.run_sample(model_path, head, sample, cfg, ctx, resume=True) -> dict   # per-id e_form, e_above_hull, rmsd
discovery.paired_delta_f1(pred_a, pred_b, truth, n_boot, seed) -> Metric
phonon_compare.omega_mae(model: PhononResult, ref: PhononResult) -> float
phonon_compare.softening_index(model, ref) -> float
elastic.constants(atoms, calc, strains=(0.005,0.01)) -> dict[str,float]; elastic.eos(atoms, calc, pct=8) -> dict
# md/
ase_md.run(atoms, calc, ensemble, T, ps, timestep_fs, cfg, ctx) -> MDResult
lammps.render_input(cfg, model_lammps, elements, T, ps, natoms) -> str; lammps.parse_thermo(log) -> MDResult
parity.check(model_path, frames, lammps_forces_json, tol_f=1e-3, tol_e=1e-4) -> dict   # {"passed": bool, ...}
# sampling/
neb.barrier(model_path, initial, final, images=7) -> dict
umbrella.HarmonicBias(calc, cv, k, center)   # ase Calculator wrapper
umbrella.run_windows(model_path, atoms, cv, centers, k, ps, T) -> list[str]
wham.free_energy(windows, k, T, method="mbar") -> dict   # ΔF, block error
# active/
committee.sigma_f(model_paths, frames) -> np.ndarray; committee.select(frames, sigma, n, sigma_min, per_group_max) -> list[Frame]
# agent/
tools.TOOLS: list[Callable]           # @beta_tool, strict schemas, return JSON str with run_id
guard.Guard(budget, allowlist).check(call) -> None | raises GuardViolation
trace.Trace(path).record(call, result_hash, manifest_sha)
backends.AnthropicBackend(model_id).run(task, tools) ; MockBackend(trace_path) ; ScriptedBackend()
runner.run(task: AgentTask, backend, cfg, ctx) -> AgentReport
eval.score(report, task) -> dict
# report/
numbers.collect(runs_dir) -> dict[str, NumberRef]     # writes reports/numbers.json
build.render(numbers, template="README.md.j2") -> str
audit.run(readme, numbers, runs_dir, strict) -> list[str]   # violations; [] means pass
```

## 7 Offline test strategy (`pytest -m "not network"`, CI has no keys, no cluster, no downloads)
- `tests/fixtures/tiny_b20.extxyz`: 12 synthetic 8-atom P2₁3 frames (4 compounds × 3 config types, random forces/energies, group ids, `label_source="none"`), plus 3 tagged `energy_scale="qe"` with `config_energy_weight=0` to test weight round-trip.
- `tiny_mace` session fixture: built in-test with `mace_run_train --hidden_irreps 8x0e --r_max 4.0 --num_interactions 1 --max_ell 1 --correlation 2 --num_radial_basis 4 --max_num_epochs 1 --batch_size 4 --default_dtype float64 --device cpu` on the fixture (< 30 s); exported once to `-lammps.pt` to test `export`.
- `golden/pw.out` (real 8-atom MnSi QE output, 2 SCF steps, magnetization lines) → `parse_pw_output` exactness; `golden/pw.in` → `render_pw_input` snapshot.
- `golden/lammps_thermo.log` → `parse_thermo`; parity test uses ASE forces vs a stored JSON.
- `fixtures/wbm_mini.csv.gz` (10 rows, checksummed) → vendored `stable_metrics` equals stored values; `paired_delta_f1` CI covers the exact difference.
- WHAM on an analytic double well (Brownian dynamics, known ΔF ± 0.02 eV); NEB on a Lennard-Jones hop.
- Phonons: `harmonic.compute` on fixture Si with the tiny model returns 3N branches, acoustic sum rule |ω(Γ)| < 0.1 meV after enforcement.
- Agent: `MockBackend` replays `fixtures/traces/*.jsonl` (recorded live runs, redacted); guard tests for every `Budget` field; `ScriptedBackend` gold answers on 3 tasks; eval scoring tests. No `anthropic` client constructed in CI (`ANTHROPIC_API_KEY` unset → `MockBackend` enforced).
- Provenance: manifest round-trip validates against `docs/manifest.schema.json`; `run_stage` writes `status="failed"` on exception.
- Audit: README fixture with a stray number fails; generated README passes.
- `tests/test_no_network.py`: `socket.create_connection` monkeypatched to raise for the whole session; `import matbench_discovery` asserted absent from `sys.modules`.

## 8 Honesty gates (`b20mlip report audit`; CI runs `--strict`)
- A1 Every numeral in README outside `<!-- gen:start -->…<!-- gen:end -->` blocks must be wrapped `<!-- num:<key> -->`; each key exists in `numbers.json` with `value, run_id, manifest_sha256`.
- A2 Every `run_id` resolves to a manifest with `status="ok"` whose output sha matches the artifact on disk.
- A3 Every metric entry carries `reference.code`, `reference.functional`, `e0_source`, `head`, `n`, `seed`, `ci95` (or an explicit `ci95: null` with reason).
- A4 F1 entries require `prevalence: "natural"`, `paired_vs: "B0"`, `sample_seed`, `n=1000`; keys `daf`, `cps`, `kappa_srme`, `leaderboard` are forbidden anywhere.
- A5 A table row may cite one `reference.code`+`functional`; PBEsol values only in columns flagged `cross_functional=true`; PySCF rows flagged `lower_fidelity=true`.
- A6 The string "LAMMPS" in README claim sections or the bullet requires `numbers["parity.passed"] == 1`.
- A7 Forbidden strings in claims: "first", "state-of-the-art", "SOTA", "leaderboard", "outperforms all".
- A8 WBM energy metrics for a model with `energy_scale != "mp"` and `head != "pt_head"` require `numbers["offsets.residual_meV_atom"] ≤ 20`.
- A9 Any `ErrorTable` for tier T3 must report `noise_floor_f`; a claimed improvement below the floor is flagged.
- A10 Agent numbers require `model_id` and `trace_path` in the manifest; mock-backend numbers may not appear in README.
- A11 `MODEL_CARD.md`, `DATA_CARD.md`, `NOTICE` exist and list licences (MIT weights; MPtrj MIT; sAlex/OMat24/WBM CC-BY-4.0; SSSP cited, not redistributed).
Exit 1 with a JSON list of violations; `--strict` also fails on `status="partial"` manifests cited by README.
