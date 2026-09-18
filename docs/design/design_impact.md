# P3 design (impact-first): `b20-mlip`

## 1. Project name + one-liner
**b20-mlip** (package `b20mlip`, MIT, github.com/wenqin-chen/b20-mlip): does fine-tuning MACE-MPA-0 on a few hundred new spin-polarised Quantum ESPRESSO frames fix its phonon/elastic softening for B20 skyrmion hosts (FeSi, MnSi, FeGe, CoSi), and what does it cost in forgetting?

## 2. Who benefits and why
- **MLIP developers**: first public PBE phonon + elastic references for B20 FeSi/MnSi/FeGe (none of mp-871/1431/7577/21255 is in the 10,034-entry phonondb index, grepped 2026-09-18; MP DFPT covers non-magnetic insulators only), plus a laptop-runnable fine-tune→forget harness (`b20mlip eval forget`).
- **Skyrmion/magnetism community**: an FM-state potential for MD/phonons of MnSi/FeGe with stated limits (no spin DOF, DM interaction or helimagnetic order).
- **Hiring managers** (ByteDance Seed, Lila, Periodic): one repo spanning DFT → MLIP → LAMMPS/ASE MD → phonons → enhanced sampling → agentic screening.

## 3. Material family + scientific question
B20 cubic P2₁3, 8 atoms/cell (MP OPTIMADE): FeSi mp-871 (NM, a=4.402 Å), MnSi mp-1431 (FM, 4.484), CoSi mp-7577 (NM, 4.383), FeGe mp-21255 (FM, 4.716; mp-22510 is the non-cubic polymorph, excluded). MPtrj holds only relaxation frames, ≈10–40 per mp-id (counts grepped Day 1): near-equilibrium collinear-FM data, so finite-T, strained and defect frames are new to the foundation model.
**Q1 softening**: uMLIPs underpredict PES curvature (Deng et al. 2024, arXiv:2405.07105; Loew et al. 2024, arXiv:2412.16551). Does MPA-0 soften B20 phonons/C_ij, and does naive or replay fine-tuning fix it? **Q2 transfer**: trained on FeSi+MnSi only, does FeGe/CoSi improve? **Q3 forgetting**: cost on a stratified WBM subset. **Q4 (stretch)**: MnSi FM vs NM DFT phonons — which does zero-shot MPA-0 resemble?

## 4. Pipeline stages
- **S0 fetch** — mp-ids → `b20mlip fetch` (OPTIMADE/MPRester); Mac, minutes → `data/structures/mp-871.cif`…
- **S1 frames** — strains ±4 %, rattles 0.02–0.15 Å, EOS ±8 %, zero-shot-MPA-0 NVT snapshots at 300/600/900 K, phonopy 2×2×2 displacements → `b20mlip frames`; Mac, ~3 h → `data/frames/round0/*.extxyz` (~630: ≈580 8-atom, ≈50 64-atom).
- **S2 DFT labels** — QE pw.x PBE, SSSP-efficiency pseudos/cutoffs, MV smearing 0.01 Ry, k-spacing 0.15 Å⁻¹, nspin=2 FM for MnSi/FeGe, forces+stress → `b20mlip qe-prep` + `cluster/qe_array.sbatch` (hpc-job-runner); **cluster (MFA)**, ≈5 min/8-atom, ≈45 min/64-atom, ≈90 node-h → `labelled.extxyz`.
- **S3 fine-tune** — `mace_run_train` naive (`--multiheads_finetuning=False --E0s="estimated" --lr=1e-3`), replay (`--multiheads_finetuning=True --pt_train_file=mp --num_samples_pt=10000 --subselect_pt=fps --E0s="foundation" --lr=1e-4`), scratch small MACE; 3 seeds each; Mac CPU float64, ≈10 min/100 epochs naive, ≈35 min/20 epochs replay → `models/B{1,2,3}_s{0,1,2}.model`.
- **S4 evaluate** — `b20mlip eval` (E/F/σ, phonopy 2×2×2, elastic, EOS, WBM subset); Mac, WBM-1000 ≈3 h/model → `results/*.json`.
- **S5 MD** — ASE Langevin/NPT, 64–512 atoms (`b20mlip md`), Mac, 100 ps ≈2–3 h; LAMMPS `pair_style mace` (`mace_create_lammps_model`, `cluster/lammps_mace.sbatch`), **cluster**, 200 ps NPT/512 atoms → a(T), VDOS.
- **S6 enhanced sampling** — ASE NEB + umbrella sampling (harmonic-bias calculator + WHAM), Fe-vacancy hop in 63-atom FeSi; Mac overnight (12 windows × 15 ps ≈ 6 h); 3 QE single points (**cluster**) → ΔF(300 K), E_a.
- **S7 active learning** — committee σ_F (3 seeds) on 600 K MD → `select_frames()` 100 frames → QE (**cluster**) → re-fine-tune → `data/frames/round1/`, `models/B2r1`.

Provenance: each stage writes `runs/<stage>/<id>/manifest.json` (git sha, input/output sha256, versions, seeds; S2 adds pw.x version, pseudopotential md5s, SLURM ids).

## 5. Evaluation plan
Metrics: E-MAE = mean|ΔE|/N (meV/atom); F-MAE over all 3N components (meV/Å) + RMSE; σ-MAE (GPa). Phonons: ω-MAE (meV) along Γ–X–M–Γ–R vs QE; softening index s = median_{q,ν} ω_model/ω_DFT; imaginary-mode count. Elastic: C11, C12, C44, B from ±0.5/1 % strains; a0, B0 errors (Birch–Murnaghan). MD: NVE drift (meV/atom/ps); a(300 K) vs experiment; VDOS vs harmonic DOS. WBM-1000: stratified WBM subset (figshare files/64706751, files/48169597), 250 each of {stable, unstable} × {contains Fe/Mn/Co/Si/Ge, not}; official protocol (FIRE + FrechetCellFilter, fmax 0.05, 500 steps) and `matbench_discovery` metric code → F1, MAE(e_above_hull), RMSD; forgetting Δ = FT − zero-shot.
**Brackets**: B0 MPA-0 zero-shot; B0′ MP-0 medium zero-shot; B1 naive FT; B2 replay FT; B3 scratch MACE on identical frames; REF = QE; EXP = experiment.
**Held-out**: H1 random 10 % of FeSi/MnSi frames (`data/splits.json`); H2 900 K frames, training on ≤600 K; H3 FeGe + CoSi never trained; H4 WBM-1000.
**Honesty**: `results/*.json` are written only by `b20mlip eval`; `scripts/render_readme.py` renders README tables and CI fails if README numbers ≠ JSON (Receipts pattern, sec-filing-qa); every number carries run id + git sha + model sha; negative results are reported. "Matbench-Discovery-style" = official metric code and relaxation protocol on a labelled stratified subset with explicit N, never a leaderboard claim (MPA-0 derivatives are non-compliant).

## 6. Agentic / high-throughput workflow
`b20mlip agent "<task>"`: Anthropic Python SDK Tool Runner (`client.beta.messages.tool_runner`, `@beta_tool`, `strict: true` schemas), harness from sec-filing-qa. Deterministic tools, each writing a run manifest: `get_structure`, `relax`, `phonons`, `elastic`, `md` (≤20 ps), `compare`, `select_frames`, `write_report`. Guardrails (`agent/guardrails.py`): element allowlist {Fe,Mn,Co,Si,Ge}; ≤512 atoms; ≤25 tool calls, 30 min; JSON-schema validation; no shell, no cluster submission; outputs are numbers + run ids, never free text. Job: screen N compounds for dynamical stability and softening with model X, re-running with a larger supercell when Γ acoustic modes break the sum rule, then a triage report. Evaluation (`evals/agent_tasks.jsonl`, 12 tasks, gold answers from the plain script `b20mlip screen`): answer accuracy, invalid-tool-call rate, provenance completeness (every number resolves to a run id), recovery on 4 injected failures, tokens/task. Its value is triage and reporting, not new numbers.

## 7. Repo skeleton
```
b20-mlip/
  pyproject.toml           mace-torch, ase, phonopy, pymatgen, matbench-discovery (git), anthropic
  src/b20mlip/
    cli.py                 fetch|frames|qe-prep|train|eval|md|neb|umbrella|screen|agent
    structures.py          fetch_mp(mp_id); canonical B20 cells
    frames.py              make_strain_rattle(), sample_md_frames(), phonopy_displacements()
    qe.py                  write_pw_input(), parse_pw_output(); SSSP md5 manifest
    train.py               finetune(cfg): naive / replay / scratch over mace_run_train
    evaluate/              ef.py, phonons.py (band_mae, softening_index), elastic.py, wbm.py
    md/                    ase_md.py (langevin, npt, vdos), lammps.py (export, parity_check)
    sampling/              neb.py, umbrella.py (HarmonicBias), wham.py
    active.py              committee_sigma(), select_frames()
    agent/                 tools.py, loop.py, guardrails.py
  cluster/                 qe_array.sbatch, lammps_mace.sbatch, build_lammps_mace.sh
  data/                    structures, splits.json; frames -> Zenodo DOI
  evals/agent_tasks.jsonl  12 gold-answer agent tasks
  scripts/render_readme.py README tables from JSON (CI-checked)
  tests/                   metrics, WHAM on analytic double well, tool schemas, guardrails
  docs/design_decisions.md one paragraph each
```

## 8. Day-by-day (cluster logins in bold)
D1 env, `fetch`, MPtrj counts, round-0 `frames`. D2 QE convergence inputs, `qe-prep`; **CLUSTER #1**: convergence + round-0 array, LAMMPS build. D3 zero-shot B0/B0′ phonons, elastic, EOS; WBM-1000 builder. D4 parse DFT, splits, E0s; B1 naive FT. D5 B2 replay, B3 scratch. D6 eval v1 on H1–H3; phonons vs QE. D7 **CLUSTER #2**: LAMMPS parity (20 frames, |ΔF| < 1e-3 eV/Å), 200 ps NPT; QE NEB single points. D8 ASE MD a(T), VDOS; NEB + umbrella overnight. D9 WHAM ΔF; committee σ → 100 frames; **CLUSTER #3**: round-1 labels. D10 agent tools, guardrails, tests. D11 agent eval; round-1 FT + eval. D12 WBM-1000 for B0/B1/B2. D13 render README, CI, Zenodo, `docs/design_decisions.md`. D14 buffer.

## 9. Risks and fallbacks
- No cluster login → PySCF KRKS (GTH-PBE, 3×3×3 k) on 8-atom FeSi, ~150 frames, labelled lower-fidelity.
- FM SCF non-convergence → adjust starting_magnetization/smearing; fall back to NM MnSi and say so.
- LAMMPS+MACE build fails → ASE MD is the result; bullet drops "LAMMPS"; README shows the build log.
- Fine-tuning does not fix softening, or forgetting is large → report it; Q1/Q3 are questions, not promises.
- Time overrun → drop round-1 AL and CoSi; keep FeGe transfer and WBM-1000.
- 16 GB RAM, MPS unusable → 2×2×2 supercells, batch_size 2, CPU float64.

## 10. Claimable milestone + bullet
Claim only when: (a) `results/eval_v1.json` holds B0/B1/B2/REF on H1–H3; (b) FeSi + MnSi phonons vs own QE; (c) ≥100 ps ASE MD and LAMMPS parity passed (else drop "LAMMPS"); (d) WBM-1000 for B0 and best FT; (e) public repo, CI green, README rendered from JSON.
Bullet: "Fine-tuned MACE-MPA-0 (equivariant GNN) on __ new spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi/MnSi): force MAE __→__ meV/Å, phonon-frequency MAE vs DFT __→__ meV, forgetting quantified on a 1,000-structure stratified WBM subset under the Matbench-Discovery protocol; deployed in LAMMPS/ASE MD (__ ps), umbrella-sampled a vacancy-hop free energy, open-sourced with a tool-calling screening agent (MIT)."

## 11. Keyword coverage
| Posting phrase (verbatim) | Backed by |
|---|---|
| "developing or using DFT/Post-HF computational tools" / "vibrational properties calculations" | QE pw.x baseline, phonopy vs QE, PySCF fallback |
| "modifying the source code of LAMMPS or OpenMM" | NOT claimed; LAMMPS used via ACEsuit `pair_style mace` |
| "materials simulation and MLIP development" / "graph neural networks, applied to materials" | MACE fine-tuning brackets B0–B3, softening study |
| "statistical mechanics and enhanced sampling algorithms" | umbrella sampling + WHAM, NEB |
| "orchestrate coding agents in development workflows" | repo built with Claude Code (CONTRIBUTING.md) |
| "agents" / "agentic AI systems, autonomous scientific workflows, or simulation-aware agents" | screening agent + 12-task eval |
| "closed-loop optimization" / "active-learning loops" | committee-σ selection → DFT → retrain (S7) |
| "high-throughput computational workflows or materials databases" / "reproducible HPC or cloud workflows" | MP/OPTIMADE fetch, WBM-1000, SLURM arrays + manifests |
| "modeling superconductivity and/or magnetism in quantum materials" / "relating theoretical models to real materials and experimental observables" | FM-state PES of MnSi/FeGe skyrmion hosts (Q4); a(T), C_ij, phonons vs experiment |
| "Matbench-Discovery-style evaluation harness" (resume) | labelled WBM-1000 subset, official protocol |
