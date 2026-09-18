# P3 design (MVP, ship-in-14-days): `b20-mlip`

## 1 Project name + one-liner
**b20-mlip** (package `b20mlip`; MIT code; weights from MIT MACE-MPA-0 on CC-BY-4.0 data). Fine-tune MACE-MPA-0 on OMat24 non-equilibrium DFT frames for the Mn–Fe–Co–Si–Ge space (B20 chiral magnets MnSi/FeGe/MnGe, FeSi, CoSi), validate by ASE (+LAMMPS) MD and phonons against DFT references and an own PySCF/QE baseline, and ship a Matbench-Discovery-style harness plus a provenance-logging pipeline agent.

## 2 Who benefits and why
- MLIP practitioners: a reproducible recipe for adapting a foundation model to one chemistry for MD (data curve, naive vs replay, measured forgetting); every number regenerates from `b20mlip run`.
- Skyrmion/chiral-magnet groups: an open PES for B20 lattice dynamics where no phonondb/MP reference exists.
- The author: every posting keyword backed by a real run on tools he already uses.

## 3 Material family + scientific question
Family: elements and binaries of {Mn,Fe,Co}×{Si,Ge}; anchors MnSi/FeGe/MnGe (skyrmion hosts, the author's field), FeSi, CoSi, β-FeSi2, Si, Ge.
Q1 data efficiency: frames needed (300/1k/3k) for DFT-quality forces and phonons; Q2 transfer: leave-Mn–Ge-out (LOCO); Q3 forgetting on out-of-chemsys frames and a WBM subset.
Data: OMat24 (CC-BY-4.0, PBE(+U), rattled+AIMD, unseen by MPA-0): val tarballs 0.07–0.26 GB each, 1M subsplit 2.27 GB (`dl.fbaipublicfiles.com/opencatalystproject/data/omat/251210/omat24_1M_251210.tar.gz`) → ~2M frames filtered locally; if <2k in-chemsys, add the 6.8 GB `train/rattled-300-subsampled.tar.gz`. Phonon references: NIMS phonondb-PBEsol has Si (mp-149, `mdr.nims.go.jp/download_all/m039k924j.zip`) and β-FeSi2 (mp-1714) but **no** B20 or elemental Fe/Mn/Co/Ge, so B20 references are own DFT (PySCF FeSi Γ locally; QE MnSi on cluster, optional) and experiment (MnSi a=4.558 Å, INS phonons; FeSi Raman). General harness: the 103-compound PBE phonondb set (simple binaries, no family overlap, labelled).
README caveat: MACE has no spin variables; the PES is the collinear DFT ground state; helimagnetism/DMI out of scope.

## 4 Pipeline stages
Every stage writes `runs/<run_id>/provenance.json`: git SHA, exact CLI, config hash, input/output sha256s, package versions, seed, host, wall time.

| # | Stage: inputs → tool | Where, runtime | Outputs |
|---|---|---|---|
| 1 | OMat24 val+1M tarballs → `fairchem.core.datasets.AseDBDataset`, keep in-chemsys frames; group split by parent id (fallback formula+spglib); LOCO split | Mac, 30–60 min (download-bound) | `data/b20_{train,val,test,loco}.extxyz`, `manifest.json` |
| 2 | `mace_run_train --foundation_model mace-mpa-0-medium.model --multiheads_finetuning False --E0s foundation --lr 1e-3 --max_num_epochs 30 --batch_size 4 --seed N` (+p3_facts loss/EMA flags); replay: `--multiheads_finetuning True --pt_train_file mp --num_samples_pt 10000 --lr 1e-4` | Mac ≈0.1 s/frame fwd+bwd → 30 epochs×3k frames ≈2.5 h small / ≈6 h medium (overnight); replay: Tillicum GPU ~30 min or Mac overnight with 2k samples | `models/b20-{naive,replay}-s{N}.model`, `-lammps.pt` |
| 3 | EFS eval (zero-shot MPA-0/MP-0, fine-tuned) on test, LOCO, 500 out-of-chemsys frames (forgetting) | Mac, 2 min | `results/efs_<model>.json` |
| 4 | phonopy 2×2×2 supercell, 0.01 Å, `auto_band_structure`; references: phonondb `phonopy_params.yaml`, MP DFPT (`mpr.get_phonon_bandstructure_by_material_id`) | Mac, seconds/material; 103-set ≈1 h/model | `results/phonons/*.json`, PNG |
| 5 | PySCF KRKS PBE, GTH/gth-dzvp, GDF: Si 2×2×2 (16 atoms, one ± displacement) ≈1 h; FeSi Γ-point (8 atoms, ~14 SCFs) overnight, stretch. Cluster-optional QE pw.x MnSi 2×2×2 | Mac / Tillicum | own `phonopy_params.yaml`, SCF logs |
| 6 | ASE Langevin 1 fs, 64-atom MnSi: NVT 300 K 20 ps (≈35 min, ~100 ms/step); NVE drift; NPT 100–500 K×30 ps (≈5 h overnight) → a(T), α. Cluster-optional: `mace_create_lammps_model`, `pair_style mace no_domain_decomposition`, same cell 100 ps, cross-engine check | Mac / Tillicum | trajectories, `results/md_*.json` |
| 7 | Si vacancy hop (63 atoms): CI-NEB (≈3 min); umbrella 8 windows×20k steps at 600 K, harmonic-bias calculator + pymbar (≈4.5 h overnight); PLUMED metadynamics (`ase.calculators.plumed`) if `brew install plumed` works | Mac | ΔF(T)±block error vs NEB |
| 8 | 1,000 WBM structures stratified by batch×stability + all in-chemsys (figshare files/48169597, /64706751); FIRE+FrechetCellFilter fmax 0.05, 500 steps; MP2020 corrections | Mac, ≈2 h/model | `results/wbm_subset_<model>.json` |
| 9 | Agent/high-throughput runner (§6); `scripts/make_readme_tables.py` rebuilds README from `results/*.json` | Mac / Tillicum | `runs/`, README |

## 5 Evaluation plan
- Energy MAE (meV/atom) = mean|E_pred/N − E_DFT/N|; force MAE (meV/Å) over all atoms×components; stress MAE (meV/Å³); on test (group split), LOCO, out-of-chemsys.
- Brackets: MP-0 small/MPA-0 zero-shot, naive, replay; forgetting = Δ out-of-chemsys MAE, Δ WBM F1 vs MPA-0.
- Data curve 300/1k/3k frames, 30 epochs each; 3 seeds at 300 frames; 3k single-seed, stated.
- Phonons: frequency MAE (THz) over reference q-path and branches, max-frequency error, imaginary-mode count (<−0.1 THz), DOS overlap. PBEsol references labelled (few-% expected offset); PBE-consistent references: own PySCF/QE, 103 PBE set; FeSi Γ modes vs Raman literature.
- MD: NVE drift (meV/atom/ps), RDF/MSD stability, a(300 K), α vs experiment. Sampling: NEB barrier vs ΔF(600 K) ± block error.
- "Matbench-Discovery-style" = MBD's protocol and metrics (relaxation, MP2020-corrected e_form, hull distance, F1/DAF/MAE/RMSE/RMSD) on a labelled 1,000-structure stratified subset (0.4% of WBM) with MPA-0 rerun identically; leaderboard numbers only as context; MPA-0 derivatives non-compliant.
- Honesty: numbers only from provenance-logged `results/*.json`; README tables generated, never hand-edited; CI `verify_readme.py` fails on any README number without a results source; subsets and functional mismatches labelled; `MODEL_CARD.md` lists data, licences, failure modes.

## 6 Agentic/high-throughput workflow
Anthropic Python SDK tool runner (`client.beta.messages.tool_runner`, `@beta_tool`), model `claude-opus-5`. Eight thin tools over stage functions — `list_data`, `finetune`, `evaluate_efs`, `run_md`, `compute_phonons`, `compare_phonons`, `wbm_subset`, `read_results` — each provenance-logged, returning paths + metrics only. Final answers pass a receipts check (every number must appear in a tool output, as in sec-filing-qa). `--llm mock` replays recorded tool sequences (CI needs no key). Eval: 8 tasks ("fine-tune 300 frames seed 1, evaluate LOCO, report force MAE") with programmatic checkers (stage order, parameters, provenance, numbers cited); ≈cents/task. High-throughput: the same stage functions loop over material lists with content-hash caching; `slurm/phonons_array.sbatch` per hpc-job-runner conventions.

## 7 Repo skeleton
```
b20-mlip/ pyproject.toml (uv, pinned) · README.md (generated tables) · LICENSE · MODEL_CARD.md
configs/ dataset.yaml finetune_{naive,replay}.yaml phonons.yaml md_{nvt,npt}.yaml umbrella.yaml wbm_subset.yaml
src/b20mlip/ cli.py (`b20mlip <stage>`) · provenance.py
  data/{omat24.py,splits.py,references.py}   # download+filter, splits, reference fetch
  train/finetune.py                          # config → mace_run_train CLI
  eval/{efs.py,phonons.py,wbm.py,kappa.py}   # metrics; kappa = phono3py stretch
  md/{ase_md.py,lammps.py,sampling.py}       # MD runners, LAMMPS export, NEB/umbrella/PLUMED
  dft/{pyscf_phonons.py,qe.py}               # local baseline, QE input generator
  agent/{tools.py,run.py,eval.py}            # beta_tools, runner+mock, task suite
scripts/{make_readme_tables.py,verify_readme.py} · slurm/*.sbatch · tests/ · .github/workflows/ci.yml · results/
```

## 8 Day-by-day (cluster login only Days 10/13)
1. Env; models; OMat24 val+1M → filter, splits, manifest; mp-ids via `mpr.materials.summary.search(formula='MnSi', spacegroup_symbol='P2_13')`.
2. Zero-shot eval; naive fine-tune 1k frames (MP-0 small dev, MPA-0 medium overnight).
3. Phonon pipeline + metrics (Si/FeSi2, B20 set); PySCF Si phonons.
4. ASE MD MnSi 20 ps + NVE drift; README v0; **tag v0.1 = vertical slice**.
5. Data curve + forgetting eval. 6. WBM subset, both models. 7. 103-set phonon harness; buffer.
8. NEB + umbrella/PLUMED overnight. 9. Agent tools, mock mode, 8-task eval.
10. Cluster day (MFA login): LAMMPS mace-branch build + 100 ps run; GPU replay fine-tune; QE MnSi phonons submitted. Else NPT sweep locally.
11. Thermal expansion; LOCO fine-tune/eval; FeSi Γ PySCF (stretch).
12. Tests, CI, docs, model card. 13. Buffer: cluster outputs; κ_SRME on ≤10 small-cell materials (stretch).
14. Release v1.0, HF weights, measured bullet numbers.
Cut: rMD17/molecules, LAMMPS source patches, full WBM, full κ_SRME, spin-aware models.

## 9 Risks and fallbacks
- Too few in-chemsys frames → rattled-300-subsampled train tarball; last resort MP tasks-API trajectories, seen-by-MPA-0 confound documented.
- MPA-0 medium too slow on Mac → curve with MP-0 small, medium only at 3k; or cluster GPU.
- OOD degradation → reported; replay variant.
- No cluster login → LAMMPS/QE/GPU stages dropped, marked planned; bullet says "ASE MD", stays valid.
- PLUMED fails → pure-Python umbrella + pymbar (primary anyway). PySCF FeSi SCF trouble → Si-only baseline; FeSi via QE optional.
- matbench-discovery package drift → protocol reimplemented with pymatgen; reference files vendored with checksums.
- 16 GB RAM → batch 4, float64, ≤64-atom cells.

## 10 Claimable milestone + bullet
Milestone (Day 4): fine-tuned model, held-out metrics, MnSi MD, Si/FeSi2 phonons vs DFT, all provenance-logged, tagged v0.1.
Bullet: "Fine-tuned an equivariant MACE-MPA-0 interatomic potential on __ OMat24 DFT structures of B20 chiral magnets (held-out force MAE __ meV/Å vs __ zero-shot), deployed it in ASE[/LAMMPS] MD (MnSi thermal expansion within __% of experiment), validated phonon spectra against DFT (phonondb, own PySCF[/QE]; MAE __ THz), and open-sourced a Matbench-Discovery-style stability/phonon harness (F1 __ on a labelled 1,000-structure WBM subset) driven by a provenance-logging tool-calling agent."

## 11 Keyword coverage
| Posting phrase | Backed by |
|---|---|
| DFT tools; first-principles on realistic systems | PySCF PBC phonons (Si, FeSi Γ); QE MnSi (optional) |
| Vibrational properties | phonopy vs phonondb/MP DFPT/own DFT; 103-set harness |
| MLIP development; GNNs | MACE fine-tuning, data curve, LOCO, forgetting study |
| MD engine (LAMMPS/OpenMM) | ASE MD + LAMMPS `pair_style mace` run (no source patch claimed) |
| Enhanced sampling / statistical mechanics | umbrella+MBAR or PLUMED metadynamics vs NEB |
| Agents; tool-calling; orchestrating coding agents | tool-runner agent with provenance + task eval |
| High-throughput workflows / materials databases; reproducible HPC | OMat24→MP/phonondb/WBM pipeline, SLURM arrays, uv-pinned env, CI, receipts check, Matbench-Discovery-style subset (F1, DAF, RMSD) |
| Magnetism/superconductivity in quantum materials | B20 skyrmion hosts; stated spin caveat |
