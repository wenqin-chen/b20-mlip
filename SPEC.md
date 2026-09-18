# b20-mlip — SPEC v1.0 (hybrid synthesis, 2026-09-17)

**One-liner.** `b20-mlip` (package `b20mlip`, MIT, github.com/wenqin-chen/b20-mlip): fine-tune MACE-MPA-0 (equivariant GNN) on in-house spin-polarised Quantum ESPRESSO (QE) frames of B20 skyrmion hosts; measure what it fixes (force/phonon softening) and costs (forgetting); deploy in ASE and LAMMPS MD; umbrella-sample a vacancy hop; drive it with a provenance-checked tool-calling agent. Architecture from production (manifests, `--resume`, executors, `report --audit`; scope cut), schedule from mvp (v0.1 by Day 4, no cluster), science from impact.

## 1 Goals
G1 400–600 QE PBE 8-atom frames + ~60 63/64-atom phonon/vacancy frames: one code, functional and magnetic branch. G2 Brackets B0/B0′/B1/B2/B3 on tiers T0–T4, 3 seeds, bootstrap CIs. G3 Same-code phonon/elastic references (FeSi, CoSi, MnSi); softening index. G4 ASE NPT thermal expansion; LAMMPS only after a 20-frame parity gate; umbrella-sampled vacancy-hop ΔF vs NEB. G5 One closed loop: committee σ_F → ≤100 QE frames → retrain. G6 Guardrailed agent, mock backend, 12-task eval. G7 README numbers regenerated from manifests; `report --audit` gates CI; negative results ship.

## 2 Non-goals
No spin degrees of freedom, DMI or helimagnetism; no LAMMPS source patch; no leaderboard submission (MPA-0 derivatives are non-compliant); no κ_SRME, CPS, DAF or full WBM; no "first" claims; no mixed-energy-scale training; no MP API key dependency; PySCF never a headline reference; PLUMED only if `brew install plumed` works (pure-Python umbrella + pymbar primary).

## 3 Material family and questions
B20, P2₁3, 8 atoms/cell: FeSi mp-871 (non-magnetic, `nspin=1`), CoSi mp-7577 (NM), MnSi mp-1431 (FM, `nspin=2`), FeGe mp-21255 (FM; non-cubic mp-22510 excluded), MnGe stretch (OPTIMADE search; else Mn-substituted FeGe, labelled). MPtrj holds 8–14 near-equilibrium relaxation frames per compound (measured): strained, rattled, finite-T and vacancy frames are new to MPA-0 *per frame*; relaxed parents were seen (disclosed).
Q1 Softening: does MPA-0 underpredict B20 PES curvature (ω, C_ij, B0), and does naive/replay fine-tuning fix it? Q2 Transfer: trained on FeSi+MnSi+CoSi, does never-trained FeGe (MnGe) improve? Q3 Forgetting: naive vs replay on held-out MPtrj forces and WBM-1000 paired ΔF1. Q4 (stretch) MnSi FM vs NM PBE phonons: which does zero-shot MPA-0 resemble? Every table: collinear-FM PBE PES, no spin DOF; PBE overestimates the MnSi moment (≈1.0 vs 0.4 μB).

## 4 Data plan
| Source | Licence, size | Role and filters |
|---|---|---|
| MPtrj extxyz (local) | MIT, 1.52 GB | 60 B20 frames: keyless structures + offset set; ~1,450 in-family frames: forgetting tier T4a (seen by MPA-0, labelled); no stress; never a zero-shot-vs-FT test |
| OMat24 1M subsplit | CC-BY-4.0, 2.27 GB, deleted after filtering | in-chemsys {Mn,Fe,Co}×{Si,Ge} frames (few hundred expected; spglib-198 count reported): Day-1 bootstrap set (forces+stress only) and never-trained VASP tier T3; max‖F‖ ≤ 15 eV/Å, dedupe, grouped by parent |
| WBM atoms + summary (local) | CC-BY-4.0, 111 MB | seeded natural-prevalence 1,000 sample (16.7 % stable) + in-family stratum N=312 (1.3 % stable); vendored metrics, checksummed refs |
| phononDB-PBE-103 | CC-BY-4.0, 76 KB | general phonon harness (VASP PBE: other code, same functional, labelled) |
| MP / Alexandria OPTIMADE | keyless | metadata, MnGe search; `MPRester` only if `MP_API_KEY` set |
| QE in-house | SSSP-efficiency pseudopotentials: cite + md5, never redistribute | round-0 ≈ 400–600 × 8-atom (strain ±6 %, shear ±4 %, rattle ≤ 0.15 Å, EOS ±8 %, NVT snapshots 300/600/900 K), ≈ 20 × 63-atom FeSi vacancy, 2×2×2 phonopy displacement sets (FeSi, CoSi, MnSi), 20 noise-floor frames; round-1 ≤ 100 |
| PySCF KRKS (local) | Apache-2.0 | labelled lower-fidelity baseline (Si 16-atom, FeSi Γ), optional |

**Filters.** StructureMatcher dedupe; magnetic-branch filter: total/absolute magnetization logged per QE frame, reject |m per TM atom − m_ref(compound)| > 0.3 μB or unconverged SCF; counts in the manifest.
**Splits.** `group_id = compound/config_type/lineage` (all derivatives of one parent share a group); `sha256(group_id|seed)` → 80/10/10 of groups. Tiers: T0 held-out groups; T1 900 K frames (train ≤ 600 K); T2 FeGe (+MnGe) never trained; T3 OMat24 VASP forces/stress vs noise floor; T4a forces on held-out MPtrj in-family frames (forgetting); T4b WBM-1000 paired ΔF1.
**Energy scales (binding).** Three scales: MP/foundation (MPtrj, sAlex), OMat24-VASP (different PAW set), QE/SSSP. R1 QE frames train with isolated-atom `E0s_qe.json` (QE, same pseudopotentials/cutoffs), never `--E0s foundation`. R2 B2 replay: QE data in head `Default`, MPtrj replay in `pt_head` (foundation E0s); B20 properties via `head="Default"`, WBM via `MACECalculator(..., head="pt_head")`, stated in captions. R3 B1 naive is on the QE scale: WBM hull metrics only if the per-element offset map (QE single points on the 60 MPtrj B20 frames → `offsets.json`) has residual ≤ 20 meV/atom, else "n/a (scale)" and forces-only forgetting. R4 OMat24 frames, if trained on (bootstrap, ablation B4), carry `config_energy_weight=0`. R5 Cross-code force noise floor: 20 OMat24 frames re-labelled by QE → F-RMSE floor; differences below it are not claims. R6 MP2020 corrections only on MP-scale energies.

## 5 Pipeline stages
Mac = M2 Pro CPU float64, 6 threads; Tillicum = SLURM (CPU nodes for QE, H200 GPU for replay/LAMMPS). MPA-0 medium; **measured** (`data/timing`): 6.7 s/epoch on 54 × 8-atom frames, batch 4 ⇒ 0.12 s/frame/epoch; 64-atom ≈ 8×.

| Stage (CLI) | Where | Runtime |
|---|---|---|
| `data pull` / `data sample` | Mac | 5 min (+40 min OMat24 download) / 2–3 h (zero-shot MD snapshots) |
| `dft converge` / `dft run` / `dft phonons` (arrays, per-unit resume) | ★Tillicum | converge 1 h (E ≤ 1 meV/atom, F ≤ 5 meV/Å → `E0s_qe.json`); 8-atom ≈ 3 min (k-spacing ≤ 0.25 Å⁻¹, MV smearing 0.01 Ry); 64-atom ≈ 1 h; 45–60 node-h total |
| `train naive` (B1, B3) | Mac | round-0 (600 × 8 + 60 × 64): ≈ 2.2 min/epoch → 30 epochs ≈ 66 min/seed; 3 seeds overnight |
| `train replay` (B2) | ★Tillicum GPU | 10k MPtrj replay ≈ 75 min/epoch on Mac → GPU (estimate ≤ 1 h/20 epochs); Mac fallback 2k samples overnight |
| `eval errors` / `phonons` / `elastic` | Mac | minutes per tier; seconds per compound; phononDB-103 ≈ 1 h/model |
| `eval discovery` | Mac or GPU | 3–6 h/model (11 h at the 500-step cap); resumable; one model per night |
| `md ase` / `md lammps` | Mac / ★Tillicum GPU | 64 atoms ≈ 0.25 s/step, 2 fs, 40 ps × {100,300,500} K ≈ 5 h overnight / parity 20 frames + 100 ps NPT/512 atoms ≈ 1 h |
| `sampling neb` / `umbrella` / `active select` | Mac | 10 min; 12 windows × 15 ps @ 2 fs ≈ 6 h overnight; minutes |

Manifests (`runs/<stage>/<run_id>/manifest.json`, CONTRACTS §5): git sha, `uv.lock` hash, versions, config hash, input/output sha256, seed, host, wall time; dft adds pw.x version, pseudo md5s, SLURM ids, magnetization, failure counts; train: foundation sha, E0 source, heads, lr, epochs; eval: model sha, head, tier N, bootstrap seed, reference; md: engine, timestep, thermostat, drift; agent: backend, model id, tokens.

## 6 Evaluation protocol
**Metrics.** E-MAE = mean|ΔE|/N (meV/atom); F-MAE/RMSE over all 3N components (meV/Å); σ-MAE (meV/Å³); 95 % bootstrap CIs (2,000 resamples over groups). Phonons: ω-MAE (meV) over sorted branches at 100 seekpath q-points, model at the DFT cell (model-relaxed variant separate); softening index s = median ω_model/ω_ref; imaginary count (ω < −0.4 meV). Elastic C11/C12/C44/B (±0.5/1 % strains); EOS a0, B0 (Birch–Murnaghan). MD: NVE drift (meV/atom/ps); a(300 K), α vs experiment (MnSi a = 4.558 Å); VDOS vs harmonic DOS; |a_ASE − a_LAMMPS|. Umbrella ΔF(300 K) ± block error vs NEB E_a. Discovery: vendored `stable_metrics` (F1, precision, recall, e_above_hull MAE/RMSE, RMSD) at natural prevalence; **F1 only as paired ΔF1 vs B0 on the identical seeded sample with bootstrap CI**.
**Brackets.** B0 MPA-0 medium zero-shot · B0′ MP-0 medium zero-shot · B1 naive (`--E0s E0s_qe.json`; lr from a {1e-4, 3e-4, 1e-3} sweep on seed 0) · B2 multihead replay (lr 1e-4, 10k FPS MPtrj, 20 epochs) · B3 scratch MACE small on identical frames · B4 optional OMat24 forces-only ablation · REF QE · EXP experiment. Data curve 300/600/all for B1, seeds stated.
**Honesty rules** (CONTRACTS §8): reference code, functional, pseudopotentials, E0 source, head, N, seed and CI on every metric; one code+functional per row, PBEsol only in a cross-functional column; no DAF/CPS/κ_SRME or leaderboard numbers ("labelled 1,000-structure WBM sample, not the leaderboard"); "unseen" per frame, seen parents disclosed; "absent from phonondb, MP-DFPT and MbD-103 as of 2026-09-18", never "first"; "LAMMPS" only after parity (20 frames, max|ΔF| < 1e-3 eV/Å, |ΔE| < 1e-4 eV/atom); the D1 timing-run validation swing (6 frames) is noise, not a result.

## 7 Agent design
Backends: `anthropic` (Python SDK `client.beta.messages.tool_runner`, `@beta_tool` functions, `strict: true`, `claude-opus-5`, `fallbacks="default"` with beta `server-side-fallback-2026-07-01`); `mock` (replays recorded traces; CI); `scripted` (deterministic DAG planner = gold answers). Tools, 1:1 with stage functions, returning numbers + run ids only: `list_data`, `get_structure`, `relax`, `phonons`, `compare_phonons`, `evaluate_errors`, `run_md` (≤ 20 ps, ≤ 512 atoms), `select_frames`, `submit_dft` (dry-run unless `--approve-cluster`), `write_report`. Guardrails: element allowlist {Fe,Mn,Co,Si,Ge}; ≤ 25 calls, ≤ 30 min, ≤ 200 DFT frames, ≤ 10 node-h; no shell; every call logged with argument and manifest hashes; reports rejected unless every number resolves to a run id. Eval: 12 tasks (`evals/agent_tasks.jsonl`, gold from `b20mlip screen`) scored on accuracy, invalid-call rate, DAG-valid order, provenance completeness, recovery from 4 injected failures, tokens and $/task.

## 8 Cluster hand-off
Alias `tillicum` (ControlPath `~/.ssh/tillicum-cm`, ControlPersist 4 h). The user runs `ssh -fN tillicum` (MFA); `b20mlip cluster bootstrap` then runs `ssh -O check`, discovers account/partitions/scratch/modules into `configs/cluster/tillicum.yaml`, rsyncs the repo, `uv sync`, checks QE (`module load`/`which pw.x`, else `micromamba create -p $SCRATCH/qe -c conda-forge qe=7.5`) and submits `build_lammps_mace.sbatch` (ACEsuit/lammps `mace` branch + libtorch, `-D PKG_ML-MACE=ON`, Kokkos+CUDA; SHA and log recorded). SLURM templates `templates/slurm/{qe_array,qe_phonons,train_replay,lammps,build_lammps}.sbatch.j2`: per-unit resume, pinned BLAS threads, no srun for single-task jobs. `cluster sync` pulls results; socket death ≠ job done.

## 9 Repo layout and CI
`src/b20mlip/{models,config,provenance,io,executors,data,dft,train,phonons,evaluate,md,sampling,active,agent,report,cli,bench}` (CONTRACTS §1), `configs/`, `templates/`, `scripts/tillicum/`, `evals/`, `tests/` (offline), `docs/`, `MODEL_CARD.md`, `DATA_CARD.md`, `NOTICE`. CI: `uv sync --frozen`, `ruff`, `mypy`, offline `pytest` with the tiny in-test MACE model, `b20mlip report audit --strict`, import guard against `matbench_discovery` (downloads at import).

## 10 14-day plan (★ = Tillicum MFA login; a slipped login never blocks Mac work)
1. ★#1 Scaffold (config, provenance, CLI, fixtures, CI green); `data pull`; **Day-1 timing task**; `cluster bootstrap` with LAMMPS build and converge scan submitted; OMat24 download.
2. OMat24 filter → bootstrap set + T3; zero-shot B0/B0′ on T3 and MPtrj-B20; `data sample` round-0; QE inputs.
3. ★#2 Freeze `qe_b20.yaml`, `E0s_qe.json`; submit production array (round-0, phonon sets, noise-floor, offset frames); bootstrap fine-tune (OMat24, forces-only); `eval errors`.
4. Phonons module (zero-shot + bootstrap B20, phononDB-103 smoke); ASE NVT 20 ps + NVE drift; `report audit`; **tag v0.1** (data → fine-tune → eval → MD → phonons, no cluster).
5. WBM-1000 B0 overnight; WHAM tests; agent tool skeleton.
6. ★#3 `cluster sync`; magnetic filter, failure counts, noise floor, `offsets.json`; dataset v1 + splits; B1 × 3 seeds (≈ 3.3 h); submit B2 replay (GPU).
7. Eval v1 (B0, B0′, B1, B3) on T0–T4a; QE phonons → ω-MAE, s; B3 scratch; WBM-1000 B1 overnight if the offset gate passes.
8. ★#4 Collect B2; export; submit LAMMPS parity + 100 ps NPT; committee σ_F → ≤ 100 frames → submit round-1 QE; ASE NPT a(T) overnight.
9. Umbrella windows overnight; NEB; VDOS; elastic/EOS.
10. WHAM ΔF vs NEB; agent tools, guardrails, scripted planner, mock replays; WBM-1000 B2 (`pt_head`) overnight.
11. Agent live runs (`claude-opus-5`), traces recorded; 12-task eval.
12. ★#5 Collect LAMMPS + round-1 labels; parity gate; retrain B2-r1; agent `submit_dft` dry-run demo; round-1 eval.
13. README from `numbers.json`; MODEL_CARD, DATA_CARD, NOTICE; `report audit`; design notes.
14. Buffer; tag v1.0; fill the bullet.

## 11 Risks and fallbacks
Login #1/#2 slip → converge/array on the next login, v0.1 unaffected; login #3 past Day 8 → ship OMat24 bootstrap + PySCF FeSi Γ, bullet variant B. No QE module → micromamba `qe`. FM SCF trouble → `mixing_beta 0.3`, `electron_maxstep 200`, fixed `starting_magnetization`; NM MnSi fallback stated. LAMMPS build fails → CPU/OpenMP build, else ASE-only and "LAMMPS" leaves the bullet. Replay slow → 5k (GPU) or 2k (Mac) samples, stated. Offset residual > 20 meV/atom → B1 forces-only forgetting; B2 via `pt_head` stands. Softening unfixed or forgetting large → reported. Overrun → drop round-1, MnGe, data curve. 16 GB RAM → batch 4, ≤ 64-atom cells.

## 12 Claimable milestone
v0.1: fine-tuned model, held-out metrics, MnSi MD, B20 phonons vs labelled references, CI green. v1.0: ≥ 400 QE frames; B1/B2 beat B0 on T0 with CIs; ≥ 2 compounds' phonons vs own QE; WBM-1000 for B0, B2 (+B1 if the scale gate passes); ≥ 100 ps ASE MD; parity decided; one AL round; agent eval; audit green.

## 13 Bullet (placeholders filled only from `numbers.json`)
Variant A: "Fine-tuned MACE-MPA-0 (equivariant GNN) on __ in-house spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi/MnSi/CoSi): held-out force MAE __→__ meV/Å, phonon ω-MAE vs same-code DFT __→__ meV, never-trained FeGe __ meV/Å; forgetting quantified as paired ΔF1 = __ [CI] on a labelled 1,000-structure WBM sample (Matbench-Discovery protocol, not the leaderboard); deployed in ASE[/LAMMPS] MD (thermal expansion within __ % of experiment), umbrella-sampled a vacancy-hop free energy (ΔF = __ eV vs NEB __ eV), one committee-uncertainty active-learning round; open-sourced (MIT) with a provenance-checked tool-calling agent."
Variant B (no cluster): QE clause → "__ OMat24 DFT frames of the Mn–Fe–Co–Si–Ge space (forces+stress)"; drop "/LAMMPS"; add "QE round pending".

## 14 Keyword coverage
| Posting phrase | Backed by |
|---|---|
| "DFT/Post-HF computational tools"; "DFT on realistic systems"; "vibrational properties calculations" | QE stages, E0s, noise floor; phonopy vs own QE, phononDB-103, VDOS; PySCF baseline |
| "MLIP development"; "graph neural networks, applied to materials" | MACE brackets B0–B3, data curve, transfer, forgetting |
| "modifying the source code of LAMMPS or OpenMM" | not claimed; LAMMPS `pair_style mace` after parity |
| "statistical mechanics and enhanced sampling algorithms" | umbrella sampling + WHAM/MBAR vs NEB |
| "agents"; "simulation-aware agents"; "tool-calling"; "orchestrate coding agents in development workflows" | budgeted agent, 12-task eval, trace audit; repo built with Claude Code (CONTRIBUTING.md) |
| "closed-loop optimization"; "active-learning loops" | committee σ_F → QE → retrain |
| "high-throughput computational workflows or materials databases"; "reproducible HPC or cloud workflows"; Matbench-Discovery-style harness | OPTIMADE/MPtrj/WBM pipeline, SLURM arrays, manifests, `--resume`; vendored metrics, labelled WBM-1000 sample |
| "magnetism in quantum materials"; "theoretical models to real materials and experimental observables" | FM PES of MnSi/FeGe; a(T), C_ij, phonons vs experiment; Q4 |

## 15 Day-1 timing task (blocks every schedule number)
`b20mlip bench --out runs/bench/` (Mac, MPA-0 medium, float64, 6 threads) measures: (a) one naive fine-tuning epoch on 100 frames (80 × 8-atom + 20 × 64-atom), batch 4 → s/frame per size class; (b) 200 MD steps at 64 and 512 atoms → s/step; (c) 20 WBM-sample FIRE+FrechetCellFilter relaxations → s and steps per structure; (d) phonopy 2×2×2 FeSi → s; (e) forward+backward at batch 4 × 64 atoms → peak RSS. Writes `bench.json` + manifest; §5 runtimes and §10 overnight slots are re-derived from it before Day 2 (today only the 8-atom 0.12 s/frame figure is measured).
