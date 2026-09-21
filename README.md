# b20-mlip

<!-- gen:start:oneliner -->
`b20-mlip` (package `b20mlip`, MIT): fine-tune MACE-MPA-0 (equivariant GNN) on in-house
spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi, CoSi, MnSi, FeGe); measure
what it fixes (force/phonon softening) and costs (forgetting); deploy in ASE MD;
umbrella-sample a vacancy hop; drive it with a provenance-checked tool-calling agent.
<!-- gen:end -->

**Status: under construction.** <!-- gen:start:status -->153 numbers from 26 runs are published in `reports/numbers.json` (0 stale).<!-- gen:end -->
Every number on this page is regenerated from `reports/numbers.json`, which is itself built from
run manifests and gated by `b20mlip report audit --strict` (honesty gates A1–A11 in
`CONTRACTS.md`). A value that reads `pending` has not been produced by any run; nothing here is
typed by hand.

## What it does

- **Data.** In-house Quantum ESPRESSO PBE frames of the B20 hosts: eight-atom cells under strain,
  shear, rattle, equation-of-state scans and finite-temperature snapshots, plus vacancy and phonon
  supercells; OMat24 in-chemsys frames (forces and stress); MPtrj in-family frames for replay and
  forgetting; a labelled, seeded WBM sample. Every frame carries its energy scale (MP, OMat24-VASP
  or QE/SSSP) and the three are never mixed in one loss.
- **Models (brackets).** B0 MACE-MPA-0 medium zero-shot; B0′ MACE-MP-0 medium zero-shot; B1 naive
  fine-tune with QE isolated-atom E0s; B2 multihead replay (QE data in head `Default`, MPtrj replay
  in `pt_head`); B3 scratch MACE on identical frames. Three seeds each, bootstrap CIs over groups.
- **Tiers.** T0 held-out groups; T1 snapshots hotter than any training frame; T2 FeGe, never
  trained on; T3 OMat24 VASP forces against a cross-code noise floor; T4a forces on held-out MPtrj
  in-family frames (forgetting); T4b paired ΔF1 on the WBM sample.
- **Physics checks.** Same-code phonon and elastic references, softening index, ASE NPT thermal
  expansion against experiment, an umbrella-sampled vacancy hop against NEB, one
  committee-uncertainty active-learning round.
- **Agent.** A budgeted tool-calling agent (element allowlist, call, wall-time, DFT-frame and
  node-hour caps, no shell) whose reports are rejected unless every number resolves to a run id;
  a mock backend replays recorded traces in CI.

## Results

Each value below is wrapped in a `num:` marker naming its key in `reports/numbers.json`; the
provenance list under every table gives reference code and functional, E0 source, head, sample
size, seed, confidence interval and run id (gate A3). A cell reads `pending` until its stage has
run; a table without any number says so explicitly.

### Force errors by tier

<!-- gen:start:forces -->
| Tier | B0 MPA-0 zero-shot | B0′ MP-0 zero-shot | B1 naive fine-tune | B2 multihead replay | B3 scratch |
| --- | --- | --- | --- | --- | --- |
| T0 held-out groups | <!-- num:eval.errors.T0.B0.mae_f -->65.59<!-- /num --> [27.72, 91.66] | <!-- num:eval.errors.T0.B0p.mae_f -->165.8<!-- /num --> [50.97, 248.4] | <!-- num:eval.errors.T0.B1.mae_f -->65.51<!-- /num --> [63.06, 66.92] | <!-- num:eval.errors.T0.B2.mae_f -->59.07<!-- /num --> [28.95, 79.75] | <!-- num:eval.errors.T0.B3.mae_f -->pending<!-- /num --> |
| T1 hot snapshots | <!-- num:eval.errors.T1.B0.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T1.B0p.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T1.B1.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T1.B2.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T1.B3.mae_f -->pending<!-- /num --> |
| T2 FeGe (never trained) | <!-- num:eval.errors.T2.B0.mae_f -->77.09<!-- /num --> [31.03, 101.7] | <!-- num:eval.errors.T2.B0p.mae_f -->190.8<!-- /num --> [39.4, 275.8] | <!-- num:eval.errors.T2.B1.mae_f -->81.64<!-- /num --> [78.23, 83.66] | <!-- num:eval.errors.T2.B2.mae_f -->72.67<!-- /num --> [43.6, 88.87] | <!-- num:eval.errors.T2.B3.mae_f -->pending<!-- /num --> |
| T3 OMat24 VASP | <!-- num:eval.errors.T3.B0.mae_f -->156.9<!-- /num --> [136.5, 183.5] | <!-- num:eval.errors.T3.B0p.mae_f -->273.4<!-- /num --> [238.3, 311.8] | <!-- num:eval.errors.T3.B1.mae_f -->157.2<!-- /num --> [156.7, 157.8] | <!-- num:eval.errors.T3.B2.mae_f -->157.6<!-- /num --> [140, 180.8] | <!-- num:eval.errors.T3.B3.mae_f -->pending<!-- /num --> |
| T4a MPtrj forgetting | <!-- num:eval.errors.T4a.B0.mae_f -->36.47<!-- /num --> [23.37, 49.53] | <!-- num:eval.errors.T4a.B0p.mae_f -->63.72<!-- /num --> [45.6, 83.65] | <!-- num:eval.errors.T4a.B1.mae_f -->32.9<!-- /num --> [31.83, 33.44] | <!-- num:eval.errors.T4a.B2.mae_f -->25.72<!-- /num --> [13.96, 39.04] | <!-- num:eval.errors.T4a.B3.mae_f -->pending<!-- /num --> |

_Force MAE in meV/Å over all force components with 95 % bootstrap CIs (2,000 resamples over groups). Each tier row cites one reference code + functional (T0–T2 the project's QE PBE, T3 OMat24 VASP PBE, T4a MPtrj VASP PBE; the provenance list says which). T3 cells carry the cross-code noise floor._

<details><summary>Provenance (one line per cell)</summary>

- **T0 held-out groups / B0 MPA-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 147; seed 0; CI95 [27.72, 91.66]; run `20260920T230245-fb89d0-none`
- **T0 held-out groups / B0′ MP-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 147; seed 0; CI95 [50.97, 248.4]; run `20260920T230302-fb89d0-none`
- **T0 held-out groups / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 147; seed 0; CI95 [63.06, 66.92]; run `20260921T012141-fb89d0-none`
- **T0 held-out groups / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 147; seed 0; CI95 [28.95, 79.75]; run `20260921T024110-fb89d0-none`
- **T2 FeGe (never trained) / B0 MPA-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 115; seed 0; CI95 [31.03, 101.7]; run `20260920T230245-fb89d0-none`
- **T2 FeGe (never trained) / B0′ MP-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 115; seed 0; CI95 [39.4, 275.8]; run `20260920T230302-fb89d0-none`
- **T2 FeGe (never trained) / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 115; seed 0; CI95 [78.23, 83.66]; run `20260921T012141-fb89d0-none`
- **T2 FeGe (never trained) / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 115; seed 0; CI95 [43.6, 88.87]; run `20260921T024110-fb89d0-none`
- **T3 OMat24 VASP / B0 MPA-0 zero-shot** — reference vasp/PBE (PAW (OMat24)); E0 foundation; head Default; n = 91; seed 0; CI95 [136.5, 183.5]; noise floor 21.53 meV/Å; run `20260920T225306-fb89d0-none`
- **T3 OMat24 VASP / B0′ MP-0 zero-shot** — reference vasp/PBE (PAW (OMat24)); E0 foundation; head Default; n = 91; seed 0; CI95 [238.3, 311.8]; noise floor 21.53 meV/Å; run `20260920T225328-fb89d0-none`
- **T3 OMat24 VASP / B1 naive fine-tune** — reference vasp/PBE (PAW (OMat24)); E0 E0s_qe.json; head Default; n = 91; seed 0; CI95 [156.7, 157.8]; noise floor 21.53 meV/Å; run `20260921T024249-fb89d0-none`
- **T3 OMat24 VASP / B2 multihead replay** — reference vasp/PBE (PAW (OMat24)); E0 E0s_qe.json; head Default; n = 91; seed 0; CI95 [140, 180.8]; noise floor 21.53 meV/Å; run `20260921T024155-fb89d0-none`
- **T4a MPtrj forgetting / B0 MPA-0 zero-shot** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head Default; n = 65; seed 0; CI95 [23.37, 49.53]; run `20260919T075717-fb89d0-none`
- **T4a MPtrj forgetting / B0′ MP-0 zero-shot** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head Default; n = 65; seed 0; CI95 [45.6, 83.65]; run `20260919T075806-fb89d0-none`
- **T4a MPtrj forgetting / B1 naive fine-tune** — reference vasp/PBE (PAW (Materials Project)); E0 E0s_qe.json; head Default; n = 65; seed 0; CI95 [31.83, 33.44]; run `20260921T012142-fb89d0-none`
- **T4a MPtrj forgetting / B2 multihead replay** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head pt_head; n = 65; seed 0; CI95 [13.96, 39.04]; run `20260921T024132-fb89d0-none`

</details>
<!-- gen:end -->


### Energy errors by tier

<!-- gen:start:energies -->
| Tier | B0 MPA-0 zero-shot | B0′ MP-0 zero-shot | B1 naive fine-tune | B2 multihead replay | B3 scratch |
| --- | --- | --- | --- | --- | --- |
| T0 held-out groups | <!-- num:eval.errors.T0.B0.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T0.B0p.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T0.B1.mae_e -->51.39<!-- /num --> [47.4, 53.76] | <!-- num:eval.errors.T0.B2.mae_e -->39.75<!-- /num --> [30.8, 48.41] | <!-- num:eval.errors.T0.B3.mae_e -->pending<!-- /num --> |
| T1 hot snapshots | <!-- num:eval.errors.T1.B0.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T1.B0p.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T1.B1.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T1.B2.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T1.B3.mae_e -->pending<!-- /num --> |
| T2 FeGe (never trained) | <!-- num:eval.errors.T2.B0.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T2.B0p.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T2.B1.mae_e -->61.44<!-- /num --> [57.33, 64.13] | <!-- num:eval.errors.T2.B2.mae_e -->40.51<!-- /num --> [30.65, 51.54] | <!-- num:eval.errors.T2.B3.mae_e -->pending<!-- /num --> |
| T3 OMat24 VASP | <!-- num:eval.errors.T3.B0.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T3.B0p.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T3.B1.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T3.B2.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T3.B3.mae_e -->pending<!-- /num --> |
| T4a MPtrj forgetting | <!-- num:eval.errors.T4a.B0.mae_e -->10.76<!-- /num --> [8.737, 12.75] | <!-- num:eval.errors.T4a.B0p.mae_e -->24.78<!-- /num --> [16.96, 36.4] | <!-- num:eval.errors.T4a.B1.mae_e -->pending<!-- /num --> | <!-- num:eval.errors.T4a.B2.mae_e -->44.02<!-- /num --> [34.26, 51.73] | <!-- num:eval.errors.T4a.B3.mae_e -->pending<!-- /num --> |

_Energy MAE in meV/atom on the reference's own energy scale; B1 energies are on the QE scale (E0s from isolated-atom QE), B0/B0′/B2 `pt_head` numbers on the MP scale._

<details><summary>Provenance (one line per cell)</summary>

- **T0 held-out groups / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 147; seed 0; CI95 [47.4, 53.76]; run `20260921T012141-fb89d0-none`
- **T0 held-out groups / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 147; seed 0; CI95 [30.8, 48.41]; run `20260921T024110-fb89d0-none`
- **T2 FeGe (never trained) / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 115; seed 0; CI95 [57.33, 64.13]; run `20260921T012141-fb89d0-none`
- **T2 FeGe (never trained) / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 115; seed 0; CI95 [30.65, 51.54]; run `20260921T024110-fb89d0-none`
- **T4a MPtrj forgetting / B0 MPA-0 zero-shot** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head Default; n = 65; seed 0; CI95 [8.737, 12.75]; run `20260919T075717-fb89d0-none`
- **T4a MPtrj forgetting / B0′ MP-0 zero-shot** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head Default; n = 65; seed 0; CI95 [16.96, 36.4]; run `20260919T075806-fb89d0-none`
- **T4a MPtrj forgetting / B2 multihead replay** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head pt_head; n = 65; seed 0; CI95 [34.26, 51.73]; run `20260921T024132-fb89d0-none`

</details>
<!-- gen:end -->


### Forgetting on the WBM sample

<!-- gen:start:discovery -->
_Not yet run: no `eval.discovery` numbers are published in `reports/numbers.json`._

_Labelled, seeded 1,000-structure WBM sample at natural prevalence (16.7 % stable), vendored Matbench-Discovery metrics; F1 only as a paired difference vs B0 on the identical sample with a bootstrap CI. No public ranking is claimed or comparable._
<!-- gen:end -->


### Phonons

<!-- gen:start:phonons -->
_Not yet run: no `eval.phonons` numbers are published in `reports/numbers.json`._

_ω-MAE over sorted branches at 100 seekpath q-points, model at the DFT cell; s = median ω_model/ω_ref; imaginary = ω < −0.4 meV. B20 rows cite the project's own QE PBE; phononDB103 rows cite VASP PBE (other code, same functional). The last column is the only place a PBEsol reference appears._
<!-- gen:end -->


### Molecular dynamics and thermal expansion

<!-- gen:start:parity -->
Parity gate (twenty frames, ASE against the second MD engine): **not yet run**;
max|ΔF| <!-- num:md.parity.max_dF_eVA -->pending<!-- /num --> eV/Å, max|ΔE| <!-- num:md.parity.max_dE_eV_atom -->pending<!-- /num --> eV/atom.
<!-- gen:end -->

<!-- gen:start:thermal -->
| Compound / engine | a(300 K) (Å) | a experiment (Å) | deviation (%) | α (1/K) |
| --- | --- | --- | --- | --- |
| FeSi / ASE | <!-- num:md.ase.FeSi.a_300K_A -->4.435<!-- /num --> [4.43, 4.439] | <!-- num:md.ase.FeSi.a_exp_A -->4.489<!-- /num --> (no CI: tabulated experimental value) | <!-- num:md.ase.FeSi.a_dev_pct -->-1.213<!-- /num --> (no CI: derived from a_300K_A and a_exp_A) | <!-- num:md.ase.FeSi.alpha_per_K -->pending<!-- /num --> |
| CoSi / ASE | <!-- num:md.ase.CoSi.a_300K_A -->4.428<!-- /num --> [4.427, 4.429] | <!-- num:md.ase.CoSi.a_exp_A -->4.444<!-- /num --> (no CI: tabulated experimental value) | <!-- num:md.ase.CoSi.a_dev_pct -->-0.3629<!-- /num --> (no CI: derived from a_300K_A and a_exp_A) | <!-- num:md.ase.CoSi.alpha_per_K -->pending<!-- /num --> |
| MnSi / ASE | <!-- num:md.ase.MnSi.a_300K_A -->4.557<!-- /num --> [4.554, 4.56] | <!-- num:md.ase.MnSi.a_exp_A -->4.558<!-- /num --> (no CI: tabulated experimental value) | <!-- num:md.ase.MnSi.a_dev_pct -->-0.01557<!-- /num --> (no CI: derived from a_300K_A and a_exp_A) | <!-- num:md.ase.MnSi.alpha_per_K -->0.000006563<!-- /num --> (no CI: fewer than 4 temperatures: no slope error) |
| FeGe / ASE | <!-- num:md.ase.FeGe.a_300K_A -->4.701<!-- /num --> [4.696, 4.706] | <!-- num:md.ase.FeGe.a_exp_A -->4.7<!-- /num --> (no CI: tabulated experimental value) | <!-- num:md.ase.FeGe.a_dev_pct -->0.01874<!-- /num --> (no CI: derived from a_300K_A and a_exp_A) | <!-- num:md.ase.FeGe.alpha_per_K -->pending<!-- /num --> |

_NPT thermal expansion at 300 K, 64-atom cells, 2 fs steps, against experiment: every cell of a row cites reference code `experiment` (the literature lattice constant is itself a published number). Rows of a second engine appear only after the parity gate passes._

<details><summary>Provenance (one line per cell)</summary>

- **FeSi / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 [4.43, 4.439]; run `20260920T224837-fd5fe0-0`
- **FeSi / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: tabulated experimental value); run `20260920T224837-fd5fe0-0`
- **FeSi / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); run `20260920T224837-fd5fe0-0`
- **CoSi / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 [4.427, 4.429]; run `20260920T225220-fd5fe0-0`
- **CoSi / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: tabulated experimental value); run `20260920T225220-fd5fe0-0`
- **CoSi / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); run `20260920T225220-fd5fe0-0`
- **MnSi / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 1501; seed 0; CI95 [4.554, 4.56]; run `20260920T230308-fd5fe0-0`
- **MnSi / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 1501; seed 0; CI95 (no CI: tabulated experimental value); run `20260920T230308-fd5fe0-0`
- **MnSi / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 1501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); run `20260920T230308-fd5fe0-0`
- **MnSi / ASE / α (1/K)** — reference experiment/None; E0 foundation; head Default; n = 3; seed 0; CI95 (no CI: fewer than 4 temperatures: no slope error); run `20260920T230308-fd5fe0-0`
- **FeGe / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 [4.696, 4.706]; run `20260920T225557-fd5fe0-0`
- **FeGe / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: tabulated experimental value); run `20260920T225557-fd5fe0-0`
- **FeGe / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); run `20260920T225557-fd5fe0-0`

</details>
<!-- gen:end -->


<!-- gen:start:stability -->
| Compound / engine | NVE drift (meV/atom/ps) |
| --- | --- |
| MnSi / ASE | <!-- num:md.ase.MnSi.drift_meV_atom_ps -->0.0001657<!-- /num --> [0.00002303, 0.0003084] |

_Pending rows (no number published yet): FeSi / ASE, CoSi / ASE, FeGe / ASE._

_NVE energy drift of the same trajectories; self-consistency numbers cite reference code `mace` with the training functional._

<details><summary>Provenance (one line per cell)</summary>

- **MnSi / ASE / NVE drift (meV/atom/ps)** — reference mace/PBE; E0 foundation; head Default; n = 501; seed 0; CI95 [0.00002303, 0.0003084]; run `20260920T225932-fd5fe0-0`

</details>
<!-- gen:end -->


### Vacancy hop: umbrella sampling versus NEB

<!-- gen:start:sampling -->
| Compound | umbrella ΔF (eV) | block error (eV) | NEB E_a (eV) |
| --- | --- | --- | --- |
| FeSi | <!-- num:sampling.umbrella.FeSi.dF_eV -->-0.001228<!-- /num --> [-0.0199, 0.01744] | <!-- num:sampling.umbrella.FeSi.dF_err_eV -->0.006724<!-- /num --> (no CI: is itself the block standard error) | <!-- num:sampling.neb.FeSi.Ea_eV -->0.6923<!-- /num --> (no CI: deterministic NEB barrier (no sampling)) |

_Pending rows (no number published yet): CoSi, MnSi, FeGe._

_Vacancy hop at 300 K: 12 umbrella windows × 15 ps, MBAR/WHAM free energy with block error, against the NEB barrier on the same model (reference code `mace`)._

<details><summary>Provenance (one line per cell)</summary>

- **FeSi / umbrella ΔF (eV)** — reference mace/PBE; E0 foundation; head Default; n = 144024; seed 0; CI95 [-0.0199, 0.01744]; run `20260920T225238-fb89d0-none`
- **FeSi / block error (eV)** — reference mace/PBE; E0 foundation; head Default; n = 144024; seed 0; CI95 (no CI: is itself the block standard error); run `20260920T225238-fb89d0-none`
- **FeSi / NEB E_a (eV)** — reference mace/PBE; E0 foundation; head Default; n = 7; seed 0; CI95 (no CI: deterministic NEB barrier (no sampling)); run `20260919T085607-fb89d0-none`

</details>
<!-- gen:end -->


### Agent evaluation

<!-- gen:start:agent -->
_Not yet run: no `agent.eval` numbers are published in `reports/numbers.json`._

_Twelve-task eval (`evals/agent_tasks.jsonl`, gold from `b20mlip screen`); backend and model id are in the provenance list. Mock-backend replays (CI) are never shown here._
<!-- gen:end -->


### One-line summary

<!-- gen:start:bullet -->
> Fine-tuned MACE-MPA-0 (equivariant GNN) on <!-- num:data.n_qe_frames -->pending<!-- /num --> in-house spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi/MnSi/CoSi): held-out force MAE <!-- num:eval.errors.T0.B0.mae_f -->65.59<!-- /num -->→<!-- num:eval.errors.T0.B2.mae_f -->59.07<!-- /num --> meV/Å, phonon ω-MAE vs same-code DFT <!-- num:eval.phonons.FeSi.B0.omega_mae_meV -->pending<!-- /num -->→<!-- num:eval.phonons.FeSi.B2.omega_mae_meV -->pending<!-- /num --> meV (FeSi), never-trained FeGe <!-- num:eval.errors.T2.B2.mae_f -->72.67<!-- /num --> meV/Å; forgetting quantified as paired ΔF1 = <!-- num:eval.discovery.B2.delta_f1 -->pending<!-- /num --> [CI pending] on a labelled 1,000-structure WBM sample (Matbench-Discovery protocol, no public ranking claimed); deployed in ASE MD (thermal expansion within <!-- num:md.ase.MnSi.a_dev_pct -->-0.01557<!-- /num --> % of experiment), umbrella-sampled a vacancy-hop free energy (ΔF = <!-- num:sampling.umbrella.FeSi.dF_eV -->-0.001228<!-- /num --> eV vs NEB <!-- num:sampling.neb.FeSi.Ea_eV -->0.6923<!-- /num --> eV), an active-learning round (pending); open-sourced (MIT) with a provenance-checked tool-calling agent.
<!-- gen:end -->

## How to reproduce

```bash
make setup                                   # uv sync --extra dev (Python 3.11, CPU torch)
make lint && make test                       # ruff, mypy, offline pytest (sockets blocked)
uv run b20mlip data pull --sources mptrj,omat24,wbm,phonondb
uv run b20mlip data sample --out data/frames/candidates_r0.extxyz
uv run b20mlip data filter --frames data/frames/candidates_r0.extxyz --out data/frames/r0.extxyz
uv run b20mlip data split --frames data/frames/r0.extxyz --out data/splits/v1.json
uv run b20mlip dft converge --compound MnSi
uv run b20mlip dft prep --frames data/frames/r0.extxyz --out dft/r0/
uv run b20mlip --executor slurm dft run --units dft/r0/
uv run b20mlip dft collect --units dft/r0/ --out data/frames/labelled_r0.extxyz
uv run b20mlip dft e0s --out configs/dft/E0s_qe.json
uv run b20mlip train --variant naive --split data/splits/v1.json --seed 0
uv run b20mlip train --variant replay --split data/splits/v1.json --seed 0
uv run b20mlip eval errors --model M --head Default --split data/splits/v1.json --tiers T0,T1,T2,T3,T4a
uv run b20mlip eval discovery --model M --head pt_head --sample data/wbm/sample_1000_s0.json
uv run b20mlip eval phonons --model M --compound FeSi --reference qe
uv run b20mlip md ase --model M --compound MnSi --ensemble npt --T 300 --ps 40 --natoms 64
uv run b20mlip md parity --model M --frames data/frames/parity.extxyz
uv run b20mlip sampling neb --model M --images 7
uv run b20mlip sampling umbrella --model M --windows 12 --ps 15
uv run b20mlip active select --models M1,M2,M3 --frames F --n 100 --out data/frames/candidates_r1.extxyz
uv run b20mlip agent eval --tasks evals/agent_tasks.jsonl --backend mock
uv run b20mlip report build --readme          # runs/ -> reports/{numbers.json,manifests/} -> README.md
uv run b20mlip report audit --strict          # honesty gates; exit 1 with a JSON list of violations
```

Every command writes `runs/<stage>/<run_id>/manifest.json` (git sha, `uv.lock` hash, package
versions, config hash, input and output sha256, seed, host, wall time) and exits zero only when
its status is ok; `--resume` reuses the last failed or partial run of the same configuration and
`--dry-run` writes a partial manifest without outputs. Tests never touch the network
(`tests/test_no_network.py` blocks sockets for the whole session) and never import
`matbench_discovery`; markers `network`, `slow` and `cluster` are excluded by default and run
explicitly with `uv run pytest -m network`.

## Honesty rules

- Every metric carries reference code, functional, pseudopotentials, E0 source, head, sample
  size, seed and a confidence interval; a table row cites one code and one functional, PBEsol
  appears only in a column flagged cross-functional, and PySCF only as a labelled lower-fidelity
  baseline, never as a headline reference.
- WBM results are a paired ΔF1 against B0 on a labelled, seeded thousand-structure sample at
  natural prevalence with a bootstrap CI. No discovery-acceleration factor, no combined score, no
  thermal-conductivity error and no public-ranking number is reported; MPA-0 derivatives are not
  eligible for public rankings anyway.
- "Unseen" is per frame: the relaxed parents of the B20 frames sit in MPtrj and were seen by
  MPA-0 (disclosed); strained, rattled, heated and vacancy frames are new to it.
- Absence from the public phonon databases (phonondb, MP-DFPT, MbD-103) is stated with the date it
  was checked, never as priority.
- Differences below the cross-code force noise floor (OMat24 frames relabelled by QE) are not
  claims; the timing-run validation swing of six frames is noise, not a result.
- The second MD engine is named only after the parity gate passes; until then only ASE results
  exist.
- Provenance ships with the repository: `report build` copies the manifest and the small
  `numbers.json` of every run that backs a published number into `reports/manifests/`, and the
  audit resolves each cited run from `runs/` or, when that is absent (as in CI), from that
  snapshot. The manifest must be complete and unchanged, and every output that is present
  locally must still match its recorded checksum; heavy outputs (models, trajectories, frame
  files) are never copied, so an output that is absent locally is noted as not local rather
  than counted as a violation.
- Protocol constants (from `configs/default.yaml` and SPEC.md), generated here so they cannot
  drift from the code:

<!-- gen:start:protocol -->
| Constant | Value |
| --- | --- |
| WBM sample size / seed | 1000 / 0 |
| Bootstrap resamples (over groups) | 2000 |
| Seeds per fine-tuning bracket | 3 |
| Noise-floor frames (OMat24 relabelled by QE) | 20 |
| Offset-map gate for B1 WBM energies (meV/atom) | 20.0 |
| Magnetic-branch tolerance (μB per TM atom) | 0.3 |
| Force cap on training frames (eV/Å) | 15.0 |
| Maximum training temperature (K); T1 is hotter | 600 |
| Parity gate: frames / max abs ΔF (eV/Å) / max abs ΔE (eV/atom) | 20 / 0.001 / 0.0001 |
| Phonon path q-points / imaginary threshold (meV) | 100 / -0.4 |
| Elastic strains (%) | 0.5 / 1 |
| QE k-spacing (1/Å) / MV smearing (Ry) | 0.25 / 0.01 |
<!-- gen:end -->

## Limitations and negative results

- Collinear ferromagnetic PBE potential-energy surfaces only: no spin degrees of freedom, no DMI,
  no helimagnetism; PBE overestimates the MnSi moment by roughly a factor of two.
- MPtrj B20 frames are near-equilibrium relaxation steps and are never used as a zero-shot versus
  fine-tune test; MnSi FM versus NM phonons are a stretch question, not a deliverable.
- B1 naive energies live on the QE scale; its WBM hull metrics appear only if the per-element
  offset map passes the gate, otherwise the table says `n/a (scale)` and forgetting is measured
  on forces only.
- Negative results ship: a bracket that does not improve on B0, a softening that fine-tuning does
  not fix, or a forgetting cost that outweighs the gain is reported in the same tables with the
  same CIs.
- Rows of the second MD engine are absent until the ASE-versus-LAMMPS parity gate passes; if the
  LAMMPS build fails on the cluster the project stays ASE-only and says so.

## Plan

- Cluster stages (Tillicum): QE production arrays and phonon displacement sets, multihead replay on
  the GPU partition, the LAMMPS `pair_style mace` build, the parity gate and, after it, the
  second-engine MD rows above.
- v0.1: data → fine-tune → eval → MD → phonons without the cluster; v1.0: QE round, brackets with
  CIs, phonons against the project's own QE, WBM sample for B0 and B2 (B1 if the scale gate
  passes), parity decided, one active-learning round, agent eval, audit green.

## Non-goals

- No "first" claims, no "state-of-the-art" or "SOTA" wording, no "leaderboard" submission (MPA-0
  derivatives are non-compliant), no thermal-conductivity, combined-score or acceleration-factor
  numbers, no spin degrees of freedom, no MD-engine source patches, no mixed-energy-scale
  training, no MP API key dependency.

## Data and licences

MPtrj (MIT) · sAlex/OMat24 (CC-BY-4.0) · WBM structures and summary as packaged by Matbench
Discovery (CC-BY-4.0) · phononDB-PBE-103 (CC-BY-4.0) · SSSP-efficiency pseudopotentials: cited
with md5 sums in run manifests, never redistributed · MACE-MPA-0 and MACE-MP-0 weights (MIT);
fine-tuned weights shipped here are MIT. Details: `MODEL_CARD.md`, `docs/DATA_CARD.md`, `NOTICE`.

## Cite

```bibtex
@software{b20mlip,
  author = {Chen, Wenqin},
  title  = {b20-mlip: fine-tuning MACE-MPA-0 on spin-polarised QE frames of B20 skyrmion hosts},
  year   = {2026},
  url    = {https://github.com/wenqin-chen/b20-mlip},
  note   = {MIT licence; numbers regenerated from reports/numbers.json and audited in CI}
}
```
