# B20-MLIP — production design (P3)

## 1 Project name + one-liner
**b20-mlip** (`github.com/wenqin-chen/b20-mlip`, MIT, package `b20mlip`): MACE-MPA-0 fine-tuned on in-house Quantum ESPRESSO data for the B20 skyrmion-host family (MnSi, FeSi, CoSi, FeGe, MnGe), deployed in ASE and LAMMPS MD, validated against DFT phonons, scored with a Matbench-Discovery-style harness, and driven by a budgeted tool-calling agent — every artifact with a provenance manifest.

## 2 Who benefits and why
- **Chiral-magnet groups** (MnSi/FeGe/MnGe skyrmion lattices): DFT-faithful lattice dynamics and thermal expansion at MD cost; no open fine-tuned B20 potential is known.
- **MLIP practitioners**: a measured answer to "does small-data fine-tuning forget general stability?" (naive-vs-replay ablation, WBM subset).
- **The author / platform engineers**: a reference pipeline (manifests, resumable SLURM stages, offline CI) that backs every phrase in p3_targets.md.

## 3 Material family + scientific question
B20 structure (P2₁3, 8-atom cell; Mn/Fe/Co/Si/Ge): magnetic MnSi, FeGe, MnGe (collinear FM in DFT) and non-magnetic FeSi, CoSi. Question: *how much does fine-tuning MACE-MPA-0 on ~1k spin-polarised PBE frames improve forces, phonons and thermal expansion of B20 skyrmion hosts, and does it degrade general stability?* Limit: the potential learns the FM PES, no explicit spins.
Data: structures from Materials Project (`mp-api`, formula + spacegroup 198; ids in the manifest) with Alexandria (CC BY 4.0, keyless) fallback; labels generated in-house with QE 7.5 (PBE, SSSP-efficiency pseudopotentials; no licence entanglement).

## 4 Pipeline stages
Every stage writes `manifest.json` (git SHA, `uv.lock` hash, package versions, input hashes, config, seed, host, SLURM job id); `--resume` skips finished units.

| CLI stage | Inputs → tool | Where, runtime | Outputs |
|---|---|---|---|
| `data pull` | family spec → mp-api/Alexandria | Mac, 1 min | `data/raw/*.extxyz` |
| `data sample` | strains ±6%/±4% shear, rattles ≤0.15 Å, MPA-0 Langevin snapshots 300/600/900 K, 20 Fe-vacancy FeSi frames | Mac, 1–2 h | `data/candidates.extxyz` (~1,000, `config_type` tag) |
| `dft converge` | MnSi ecut/k scan (E ≤1 meV/atom, F ≤5 meV/Å) + isolated-atom E0s | Tillicum, 1 h | `configs/dft/qe_b20.yaml`, `E0s.json` |
| `dft run` | pw.x SCF, spin-polarised PBE, MV smearing, k-spacing ≤0.25 Å⁻¹, forces + stress | Tillicum array, per-frame resume; ~3 min/frame, ≈50 node-h | `dft/<hash>/{pw.out,frame.extxyz,manifest.json}`, failure rate |
| `dft phonons` | 2×2×2 phonopy displacements (0.03 Å) → pw.x forces; FeSi, CoSi, MnSi | Tillicum, 4–8 × 64-atom SCF, ~1 h each | `phonons_dft/<compound>/force_sets.json` |
| `train` | `mace_run_train` from `medium-mpa-0`, `--E0s E0s.json`, float64, AdamW+EMA; `naive` (no replay, lr 1e-3) vs `replay` (`--pt_train_file mp`, 5k FPS-selected frames, lr 1e-4); 20–30 epochs, 3 seeds | Mac CPU, 0.5–1.5 h/run | `models/<variant>-s<seed>/{.model,-lammps.pt}` |
| `eval errors` / `eval discovery` / `phonons` | tiers (§5); WBM stratified 1k; phonopy 2×2×2 with DFT's displacement set | Mac; min / 2–6 h per model, resumable / min | `eval/*.json`; bands, DOS, ω-MAE |
| `md ase` / `md lammps` | 64 atoms, NPT, 1 fs, 50 ps × {100,300,500} K; LAMMPS adds 20 ps NVE via `pair_style mace no_domain_decomposition` | Mac 2–3 h / Tillicum ~1 h | a(T), α, RDF, VDOS; ASE–LAMMPS agreement |
| `md umbrella` | FeSi Fe-vacancy migration: NEB (7 images); 12 windows, k=5 eV/Å², 20 ps each; WHAM (`pymbar`) | Mac, 3–4 h | barrier, ΔF(300 K) |
| `report` | all manifests | Mac, s | `reports/numbers.json`, figures → README |

Cluster hand-off: `b20mlip cluster bootstrap` verifies `ssh -O check tillicum` (after the user's MFA login), rsyncs the repo, runs `uv sync`, writes account/partition/modules to `configs/cluster/tillicum.yaml`, installs QE 7.5, submits `build_lammps_mace.sbatch` (ACEsuit/lammps `mace` branch + libtorch; SHA recorded). `SlurmExecutor` renders `templates/slurm/*.sbatch.j2` and polls `sacct`; `cluster sync` pulls results.

## 5 Evaluation plan
**Metrics.** E-MAE (meV/atom), F-MAE/RMSE (meV/Å, all atoms and components), stress MAE (meV/Å³), bootstrap 95% CIs per `config_type` and tier; phonon ω-MAE (cm⁻¹, sorted bands on a shared 100-point seekpath path) plus imaginary-mode flag (ω < −3 cm⁻¹); α=(1/a)da/dT from NPT a(T); |a_ASE−a_LAMMPS| and NVE drift (meV/atom/ps); umbrella ΔF vs NEB barrier.
**Matbench-Discovery-style subset.** Seeded stratified 1,000 WBM structures over (e_above_hull bin × n_elements) from `wbm_summary` (figshare files/64706751) + `wbm_initial_atoms` (files/48169597); MbD protocol (ASE FIRE + FrechetCellFilter, fmax 0.05 eV/Å, ≤500 steps); MP2020-corrected e_form; stable ⇔ e_above_hull ≤ 0; F1/precision/recall/DAF/RMSD via `matbench_discovery.metrics`. Labelled "stratified 1k subset, non-compliant (MPA-0 derivative), no κ_SRME/CPS, not the leaderboard".
**Brackets.** MACE-MP-0 small and MPA-0 zero-shot; naive and replay fine-tunes (3 seeds); DFT noise floor from 20 frames at tighter settings.
**Held-out.** StructureMatcher dedupe, then split by `sha256(frame_id+seed)`: (i) in-distribution 10%; (ii) OOD-temperature: all 900 K snapshots (train ≤600 K); (iii) OOD-composition: MnGe never trained; (iv) WBM subset (forgetting).
**Honesty.** README numbers only via `reports/numbers.json`; `b20mlip report --audit` fails CI on any number without a manifest source; seeds, hashes, DFT failure counts, subset labels mandatory in captions.

## 6 Agentic/high-throughput workflow
`b20mlip agent run --task tasks/active_learning.yaml --budget configs/agent/budget.yaml --backend anthropic|scripted`. One live campaign: fine-tune → MD → score frames by 3-seed committee force disagreement → select ≤50 → DFT on Tillicum → retrain → report. Tools (pydantic schemas, 1:1 with CLI stages): `sample_candidates`, `select_uncertain`, `submit_dft`, `collect_dft`, `finetune`, `evaluate`, `run_md`, `write_report`. Guardrails: caps (≤200 DFT frames, ≤10 node-hours, ≤12 h wall, ≤40 calls), tool allowlist, no shell, `submit_dft` dry-runs unless `--approve-cluster`, every call logged to `agent/trace.jsonl` with argument/manifest hashes, report rejected unless every number traces to a manifest. Backend: Anthropic SDK tool runner (`client.beta.messages.tool_runner`, `@beta_tool`, `claude-opus-5`) plus a deterministic `scripted` planner. Evaluation: 8 scenarios (e.g. "hold out CoSi, report F-MAE") scored on DAG-valid tool order, budget violations, report grounding, completion; live traces replayed offline in CI.

## 7 Repo skeleton
```
b20-mlip/
  pyproject.toml            uv, pinned deps, [project.scripts] b20mlip=b20mlip.cli:app
  Makefile                  setup|lint|test|data|dft|train|eval|md|phonons|report
  configs/{data,dft,train,eval,md,phonons,agent}/*.yaml, configs/cluster/tillicum.yaml
  src/b20mlip/config.py     pydantic-settings; load_config(path, overrides) -> Settings
  src/b20mlip/provenance.py write_manifest(stage, out_dir, inputs, params, extra) -> Manifest
  src/b20mlip/executors/    Executor protocol submit(JobSpec)->JobHandle, wait, fetch; LocalExecutor, SlurmExecutor
  src/b20mlip/data/         pull.fetch_family(spec); sample.generate(seeds,cfg,rng); split.split_frames(frames,cfg,seed)->Split
  src/b20mlip/dft/          qe.render_pw_input(atoms,cfg)->str; qe.parse_pw_output(path)->DFTFrame; e0s.py; pyscf_pbc.py
  src/b20mlip/train/        finetune.build_mace_args(cfg)->list[str]; run_finetune(cfg,data)->CheckpointInfo; export.to_lammps
  src/b20mlip/evaluate/     errors.errors_by_tier(model,split)->ErrorTable; discovery.run_wbm_subset(model,cfg); phonon_compare.band_mae
  src/b20mlip/md/           ase_md.run_npt(atoms,calc,cfg)->MDResult; lammps.render_input/parse_thermo; bias.HarmonicBias; wham.py
  src/b20mlip/phonons/      harmonic.compute_phonons(atoms,calc,supercell,distance)->PhononResult
  src/b20mlip/agent/        tools.py, runner.run_agent(task,budget,backend)->AgentReport, guard.py, trace.py
  src/b20mlip/report/       build.py (numbers.json, figures, README templating, --audit)
  src/b20mlip/cli.py        typer; every stage: run(cfg: StageConfig, ctx: RunContext) -> StageResult
  templates/                slurm/{qe_array,lammps,build_lammps}.sbatch.j2, qe/pw.in.j2, lammps/npt.in.j2
  scripts/tillicum/         bootstrap.sh, build_lammps_mace.sh, install_qe.sh
  tests/                    fixtures: tiny_b20.extxyz (12 synthetic frames), tiny_mace.model (4 channels, L=0), golden pw.out, replays
  .github/workflows/ci.yml  ruff, mypy --strict, pytest (offline), report --audit
```

## 8 Day-by-day plan (★ = Tillicum MFA login)
1. ★ Scaffold, config, provenance, CLI, fixtures, CI; `data pull`; `cluster bootstrap`.
2. ★ `data sample`; QE templates + parser; submit `dft converge`.
3. ★ Freeze DFT settings; submit production array; split, metrics, train wrapper.
4. Zero-shot baselines; phonons module; ASE MD smoke runs.
5. ★ `cluster sync`; dataset v1; naive fine-tune; tier errors.
6. ★ Replay fine-tune; ablation; LAMMPS export; submit `dft phonons`.
7. Discovery harness; WBM subset overnight.
8. ★ LAMMPS NPT/NVE vs ASE.
9. ★ Collect DFT phonons; ω-MAE tables; VDOS vs harmonic DOS.
10. NEB, umbrella sampling, WHAM.
11. ★ Agent tools, guardrails, planners; live active-learning round.
12. ★ Collect, retrain, agent eval; buffer.
13. README with real numbers, MODEL_CARD/DATA_CARD, `report --audit`.
14. Honesty audit, tag v0.1.0, fill the bullet.

## 9 Risks and fallbacks
- **Cluster/queue delays** → PySCF-PBC labels for FeSi/CoSi small cells (slow, labelled); bullet numbers follow reality.
- **LAMMPS build fails** (libtorch ABI) → CPU/OpenMP build first; if unfixable, ASE-only and "LAMMPS" leaves the bullet.
- **Spin-polarised SCF non-convergence** → `mixing_beta 0.3`, `electron_maxstep 200`; failures counted; MnSi PBE-moment overestimate disclosed.
- **Forgetting / costly magnetic phonon DFT** → replay variant is the deliverable, both reported; FeSi/CoSi phonons only, MnSi foundation-vs-fine-tuned and phonondb if indexed.
- **Mac wall time / API skew / agent cost** → WBM subset 500 for extra seeds; vendored `matbench_discovery` metrics with attribution; scripted planner default.

## 10 Claimable milestone + bullet
Milestone (v0.1.0): ≥600 QE frames; fine-tune beats MPA-0 zero-shot F-MAE (ID tier, CIs); LAMMPS agrees with ASE; phonons of ≥2 compounds vs own DFT; WBM subset for ≥3 models; one agent-driven active-learning round; CI green.
Bullet: "Developed and benchmarked a fine-tuned MACE-MPA-0 equivariant-GNN interatomic potential for B20 skyrmion hosts (MnSi/FeSi/CoSi/FeGe/MnGe) on __ in-house Quantum ESPRESSO structures (force MAE __ meV/Å vs __ zero-shot; held-out MnGe __ meV/Å), deployed it in LAMMPS and ASE molecular dynamics, validated phonon spectra against DFT (ω-MAE __ cm⁻¹), ran umbrella sampling, and open-sourced it with a Matbench-Discovery-style stability harness (F1 __ on a stratified 1k WBM subset) and a provenance-checked tool-calling agent."

## 11 Keyword coverage
| Posting phrase | Backed by |
|---|---|
| DFT tools / first-principles on realistic systems | QE SCF/phonon stages, convergence protocol, E0s |
| LAMMPS use / source modification | `pair_style mace` build + runs; modification *not* claimed |
| MLIP development; GNNs; AI/ML on simulation data | MPA-0 fine-tuning ablation, committee uncertainty |
| Enhanced sampling / statistical mechanics; vibrational properties | umbrella sampling + WHAM vs NEB; phonopy vs DFT ω-MAE, VDOS, thermal expansion |
| Agentic AI, tool-calling, closed-loop, active learning; orchestrate coding agents | budgeted agent, live active-learning round, trace audit; built with AI coding agents under CI |
| High-throughput/reproducible HPC workflows; materials databases | MP/Alexandria pull, SLURM arrays, manifests, resume |
| Magnetism / quantum materials | B20 skyrmion hosts, FM DFT labels |
