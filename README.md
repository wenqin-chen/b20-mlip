# b20-mlip

<!-- gen:start:oneliner -->
`b20-mlip` (package `b20mlip`, MIT): fine-tune MACE-MPA-0 (equivariant GNN) on in-house
spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi, CoSi, MnSi, FeGe); measure
what it fixes (force/phonon softening) and costs (forgetting); deploy in ASE and LAMMPS MD;
umbrella-sample a vacancy hop; drive it with a provenance-checked tool-calling agent.
<!-- gen:end -->

**Status: under construction.** <!-- gen:start:status -->370 numbers from 58 runs are published in `reports/numbers.json` (0 stale).<!-- gen:end -->
Every number on this page is regenerated from `reports/numbers.json`, which is itself built from
run manifests and gated by `b20mlip report audit --strict` (honesty gates A1–A11 in
`CONTRACTS.md`). A value that reads `pending` has not been produced by any run; nothing here is
typed by hand.

## What it does

- **Data.** In-house Quantum ESPRESSO PBE frames of the B20 hosts: eight-atom cells under strain,
  shear, rattle, equation-of-state scans and finite-temperature snapshots, plus vacancy and phonon
  supercells; OMat24 in-chemsys frames (forces and stress); MPtrj in-family frames for replay and
  forgetting; a labelled, seeded WBM sample. Every frame carries its energy scale (MP, OMat24-VASP
  or QE/SSSP) and the three are never mixed in one loss. The fine-tunes train on FeSi, MnSi and
  CoSi (plus the three relaxed CoGe cells of the offset map); FeGe and MnGe are labelled but never
  trained on.
- **Models (brackets).** B0 MACE-MPA-0 medium zero-shot; B0′ MACE-MP-0 medium zero-shot; B1 naive
  fine-tune with QE isolated-atom E0s; B2 multihead replay (QE data in head `Default`, a
  farthest-point subset of the MPtrj foundation data replayed in `pt_head`); B3 scratch MACE on
  identical frames. B1 and B2 are trained with three seeds and published as the seed mean with the
  seed range as its interval; B3 has one seed; every single-model number carries a bootstrap CI
  over groups.
- **Tiers.** T0 held-out groups of the trained compounds; T1 snapshots hotter than any training
  frame; T2 FeGe and MnGe, never trained on; T3 OMat24 VASP forces against a cross-code noise
  floor; T4a forces against the MPtrj VASP labels of the MPtrj B20 frames (forgetting: the relaxed
  parents among them were also relabelled by QE and trained on, so T4a measures drift away from
  the MP labels, not generalisation); T4b paired ΔF1 on the WBM sample.
- **Physics checks.** Same-code phonon and elastic references, softening index, NPT thermal
  expansion against experiment, an umbrella-sampled vacancy hop against NEB, and a
  committee-uncertainty selector for the next active-learning round.
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
| T0 held-out groups | <!-- num:eval.errors.T0.B0.mae_f -->24.24<!-- /num --> [12.3, 44.51] | <!-- num:eval.errors.T0.B0p.mae_f -->76.02<!-- /num --> [72.76, 79.51] | <!-- num:eval.errors.T0.B1.mae_f -->6.106<!-- /num --> [6.059, 6.156] | <!-- num:eval.errors.T0.B2.mae_f -->8.578<!-- /num --> [8.323, 8.87] | <!-- num:eval.errors.T0.B3.mae_f -->11.63<!-- /num --> [3.394, 26.76] |
| T1 hot snapshots | <!-- num:eval.errors.T1.B0.mae_f -->127.9<!-- /num --> [95.29, 159.6] | <!-- num:eval.errors.T1.B0p.mae_f -->330.7<!-- /num --> [265.9, 395] | <!-- num:eval.errors.T1.B1.mae_f -->65.4<!-- /num --> [63.57, 66.92] | <!-- num:eval.errors.T1.B2.mae_f -->63.2<!-- /num --> [62.47, 63.73] | <!-- num:eval.errors.T1.B3.mae_f -->268.7<!-- /num --> [34.23, 506.6] |
| T2 FeGe + MnGe (never trained) | <!-- num:eval.errors.T2.B0.mae_f -->90.02<!-- /num --> [54.93, 107.1] | <!-- num:eval.errors.T2.B0p.mae_f -->210.4<!-- /num --> [103.2, 267.5] | <!-- num:eval.errors.T2.B1.mae_f -->108.6<!-- /num --> [106.5, 110.4] | <!-- num:eval.errors.T2.B2.mae_f -->77.84<!-- /num --> [76.64, 78.52] | <!-- num:eval.errors.T2.B3.mae_f -->453.6<!-- /num --> [194.5, 601.8] |
| T3 OMat24 VASP | <!-- num:eval.errors.T3.B0.mae_f -->156.9<!-- /num --> [136.5, 183.5] | <!-- num:eval.errors.T3.B0p.mae_f -->273.4<!-- /num --> [238.3, 311.8] | <!-- num:eval.errors.T3.B1.mae_f -->167.3<!-- /num --> [165.8, 168.2] | <!-- num:eval.errors.T3.B2.mae_f -->156.9<!-- /num --> [156.7, 157.1] | <!-- num:eval.errors.T3.B3.mae_f -->617.3<!-- /num --> [524.2, 731.1] |
| T4a MPtrj forgetting | <!-- num:eval.errors.T4a.B0.mae_f -->36.47<!-- /num --> [23.37, 49.53] | <!-- num:eval.errors.T4a.B0p.mae_f -->63.72<!-- /num --> [45.6, 83.65] | <!-- num:eval.errors.T4a.B1.mae_f -->36.14<!-- /num --> [35.58, 37.12] | <!-- num:eval.errors.T4a.B2.mae_f -->25.82<!-- /num --> [25.14, 26.63] | <!-- num:eval.errors.T4a.B3.mae_f -->43.18<!-- /num --> [14.61, 80.31] |

_Force MAE in meV/Å over all force components with 95 % bootstrap CIs (2,000 resamples over groups). Each tier row cites one reference code + functional (T0–T2 the project's QE PBE, T3 OMat24 VASP PBE, T4a MPtrj VASP PBE; the provenance list says which). T3 cells carry the cross-code noise floor._

<details><summary>Provenance (one line per cell)</summary>

- **T0 held-out groups / B0 MPA-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 32; seed 0; CI95 [12.3, 44.51]; run `20260922T171855-fb89d0-none`
- **T0 held-out groups / B0′ MP-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 32; seed 0; CI95 [72.76, 79.51]; run `20260922T174254-fb89d0-none`
- **T0 held-out groups / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 32; seed 0; CI95 [6.059, 6.156]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174850-fb89d0-none`
- **T0 held-out groups / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 32; seed 0; CI95 [8.323, 8.87]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181411-fb89d0-none`
- **T0 held-out groups / B3 scratch** — reference qe/PBE (SSSP-efficiency-1.3); E0 estimated; head Default; n = 32; seed 0; CI95 [3.394, 26.76]; run `20260922T174348-fb89d0-none`
- **T1 hot snapshots / B0 MPA-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 49; seed 0; CI95 [95.29, 159.6]; run `20260922T171855-fb89d0-none`
- **T1 hot snapshots / B0′ MP-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 49; seed 0; CI95 [265.9, 395]; run `20260922T174254-fb89d0-none`
- **T1 hot snapshots / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 49; seed 0; CI95 [63.57, 66.92]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174850-fb89d0-none`
- **T1 hot snapshots / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 49; seed 0; CI95 [62.47, 63.73]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181411-fb89d0-none`
- **T1 hot snapshots / B3 scratch** — reference qe/PBE (SSSP-efficiency-1.3); E0 estimated; head Default; n = 49; seed 0; CI95 [34.23, 506.6]; run `20260922T174348-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B0 MPA-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 175; seed 0; CI95 [54.93, 107.1]; run `20260922T171855-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B0′ MP-0 zero-shot** — reference qe/PBE (SSSP-efficiency-1.3); E0 foundation; head Default; n = 175; seed 0; CI95 [103.2, 267.5]; run `20260922T174254-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 175; seed 0; CI95 [106.5, 110.4]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174850-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 175; seed 0; CI95 [76.64, 78.52]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181411-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B3 scratch** — reference qe/PBE (SSSP-efficiency-1.3); E0 estimated; head Default; n = 175; seed 0; CI95 [194.5, 601.8]; run `20260922T174348-fb89d0-none`
- **T3 OMat24 VASP / B0 MPA-0 zero-shot** — reference vasp/PBE (PAW (OMat24)); E0 foundation; head Default; n = 91; seed 0; CI95 [136.5, 183.5]; noise floor 21.53 meV/Å; run `20260920T225306-fb89d0-none`
- **T3 OMat24 VASP / B0′ MP-0 zero-shot** — reference vasp/PBE (PAW (OMat24)); E0 foundation; head Default; n = 91; seed 0; CI95 [238.3, 311.8]; noise floor 21.53 meV/Å; run `20260920T225328-fb89d0-none`
- **T3 OMat24 VASP / B1 naive fine-tune** — reference vasp/PBE (PAW (OMat24)); E0 E0s_qe.json; head Default; n = 91; seed 0; CI95 [165.8, 168.2]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; noise floor 21.53 meV/Å; run `20260922T174852-fb89d0-none`
- **T3 OMat24 VASP / B2 multihead replay** — reference vasp/PBE (PAW (OMat24)); E0 E0s_qe.json; head Default; n = 91; seed 0; CI95 [156.7, 157.1]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; noise floor 21.53 meV/Å; run `20260922T181413-fb89d0-none`
- **T3 OMat24 VASP / B3 scratch** — reference vasp/PBE (PAW (OMat24)); E0 estimated; head Default; n = 91; seed 0; CI95 [524.2, 731.1]; noise floor 21.53 meV/Å; run `20260922T174409-fb89d0-none`
- **T4a MPtrj forgetting / B0 MPA-0 zero-shot** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head Default; n = 65; seed 0; CI95 [23.37, 49.53]; run `20260919T075717-fb89d0-none`
- **T4a MPtrj forgetting / B0′ MP-0 zero-shot** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head Default; n = 65; seed 0; CI95 [45.6, 83.65]; run `20260919T075806-fb89d0-none`
- **T4a MPtrj forgetting / B1 naive fine-tune** — reference vasp/PBE (PAW (Materials Project)); E0 E0s_qe.json; head Default; n = 65; seed 0; CI95 [35.58, 37.12]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174851-fb89d0-none`
- **T4a MPtrj forgetting / B2 multihead replay** — reference vasp/PBE (PAW (Materials Project)); E0 foundation; head pt_head; n = 65; seed 0; CI95 [25.14, 26.63]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181412-fb89d0-none`
- **T4a MPtrj forgetting / B3 scratch** — reference vasp/PBE (PAW (Materials Project)); E0 estimated; head Default; n = 65; seed 0; CI95 [14.61, 80.31]; run `20260922T174402-fb89d0-none`

</details>
<!-- gen:end -->


### Energy errors by tier

<!-- gen:start:energies -->
| Tier | B0 MPA-0 zero-shot | B0′ MP-0 zero-shot | B1 naive fine-tune | B2 multihead replay | B3 scratch |
| --- | --- | --- | --- | --- | --- |
| T0 held-out groups | <!-- num:eval.errors.T0.B0.mae_e -->n/a (MP scale)<!-- /num --> | <!-- num:eval.errors.T0.B0p.mae_e -->n/a (MP scale)<!-- /num --> | <!-- num:eval.errors.T0.B1.mae_e -->6.721<!-- /num --> [6.548, 7.003] | <!-- num:eval.errors.T0.B2.mae_e -->5.164<!-- /num --> [4.168, 6.125] | <!-- num:eval.errors.T0.B3.mae_e -->4.881<!-- /num --> [2.54, 8.179] |
| T1 hot snapshots | <!-- num:eval.errors.T1.B0.mae_e -->n/a (MP scale)<!-- /num --> | <!-- num:eval.errors.T1.B0p.mae_e -->n/a (MP scale)<!-- /num --> | <!-- num:eval.errors.T1.B1.mae_e -->20.21<!-- /num --> [19.88, 20.73] | <!-- num:eval.errors.T1.B2.mae_e -->12.67<!-- /num --> [11.65, 13.37] | <!-- num:eval.errors.T1.B3.mae_e -->55.46<!-- /num --> [1.405, 149.8] |
| T2 FeGe + MnGe (never trained) | <!-- num:eval.errors.T2.B0.mae_e -->n/a (MP scale)<!-- /num --> | <!-- num:eval.errors.T2.B0p.mae_e -->n/a (MP scale)<!-- /num --> | <!-- num:eval.errors.T2.B1.mae_e -->52.05<!-- /num --> [50.68, 53.06] | <!-- num:eval.errors.T2.B2.mae_e -->26.67<!-- /num --> [25.41, 27.66] | <!-- num:eval.errors.T2.B3.mae_e -->137.4<!-- /num --> [71.53, 208.4] |

_Energy MAE in meV/atom against the project's QE labels. The fine-tuned models predict on the QE scale (B1 and B2 head `Default` with isolated-atom QE E0s, B3 with fitted E0s); the zero-shot models predict on the MP scale, so their cells read n/a instead of mixing scales. Only the QE-labelled tiers are listed (T3 and T4a energies would be on other codes' scales)._

<details><summary>Provenance (one line per cell)</summary>

- **T0 held-out groups / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 32; seed 0; CI95 [6.548, 7.003]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174850-fb89d0-none`
- **T0 held-out groups / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 32; seed 0; CI95 [4.168, 6.125]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181411-fb89d0-none`
- **T0 held-out groups / B3 scratch** — reference qe/PBE (SSSP-efficiency-1.3); E0 estimated; head Default; n = 32; seed 0; CI95 [2.54, 8.179]; run `20260922T174348-fb89d0-none`
- **T1 hot snapshots / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 49; seed 0; CI95 [19.88, 20.73]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174850-fb89d0-none`
- **T1 hot snapshots / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 49; seed 0; CI95 [11.65, 13.37]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181411-fb89d0-none`
- **T1 hot snapshots / B3 scratch** — reference qe/PBE (SSSP-efficiency-1.3); E0 estimated; head Default; n = 49; seed 0; CI95 [1.405, 149.8]; run `20260922T174348-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B1 naive fine-tune** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 175; seed 0; CI95 [50.68, 53.06]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T174850-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B2 multihead replay** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 175; seed 0; CI95 [25.41, 27.66]; mean of 3 models (training seeds 0,1,2); CI95 = seed min–max; run `20260922T181411-fb89d0-none`
- **T2 FeGe + MnGe (never trained) / B3 scratch** — reference qe/PBE (SSSP-efficiency-1.3); E0 estimated; head Default; n = 175; seed 0; CI95 [71.53, 208.4]; run `20260922T174348-fb89d0-none`

</details>
<!-- gen:end -->


### Forgetting on the WBM sample

<!-- gen:start:discovery -->
| Model | paired ΔF1 vs B0 | e_above_hull MAE (meV/atom) | RMSD (Å) |
| --- | --- | --- | --- |
| B0 MPA-0 zero-shot | — | <!-- num:eval.discovery.B0.mae_e_above_hull -->25.49<!-- /num --> [22.3, 29.41] | <!-- num:eval.discovery.B0.rmsd -->0.08885<!-- /num --> [0.08161, 0.09678] |
| B2 multihead replay | <!-- num:eval.discovery.B2.delta_f1 -->-0.1098<!-- /num --> [-0.1603, -0.06316] | <!-- num:eval.discovery.B2.mae_e_above_hull -->38.79<!-- /num --> [35.29, 42.86] | <!-- num:eval.discovery.B2.rmsd -->0.08826<!-- /num --> [0.08076, 0.09608] |

_Pending rows (no number published yet): B1 naive fine-tune, B3 scratch._

_Labelled, seeded 1,000-structure WBM sample at natural prevalence (15.2 % stable in this sample), vendored Matbench-Discovery metrics; F1 only as a paired difference vs B0 on the identical sample with a bootstrap CI. No public ranking is claimed or comparable._

<details><summary>Provenance (one line per cell)</summary>

- **B0 MPA-0 zero-shot / e_above_hull MAE (meV/atom)** — reference vasp/PBE (PAW (Materials Project settings, MP2020-corrected)); E0 foundation; head Default; n = 1000; seed 0; CI95 [22.3, 29.41]; run `20260921T025457-fd5fe0-0`
- **B0 MPA-0 zero-shot / RMSD (Å)** — reference vasp/PBE (PAW (Materials Project settings, MP2020-corrected)); E0 foundation; head Default; n = 1000; seed 0; CI95 [0.08161, 0.09678]; run `20260921T025457-fd5fe0-0`
- **B2 multihead replay / paired ΔF1 vs B0** — reference vasp/PBE (PAW (Materials Project settings, MP2020-corrected)); E0 foundation; head pt_head; n = 1000; seed 0; CI95 [-0.1603, -0.06316]; run `20260922T181942-fd5fe0-0`
- **B2 multihead replay / e_above_hull MAE (meV/atom)** — reference vasp/PBE (PAW (Materials Project settings, MP2020-corrected)); E0 foundation; head pt_head; n = 1000; seed 0; CI95 [35.29, 42.86]; run `20260922T181942-fd5fe0-0`
- **B2 multihead replay / RMSD (Å)** — reference vasp/PBE (PAW (Materials Project settings, MP2020-corrected)); E0 foundation; head pt_head; n = 1000; seed 0; CI95 [0.08076, 0.09608]; run `20260922T181942-fd5fe0-0`

</details>
<!-- gen:end -->


### Phonons

<!-- gen:start:phonons -->
| Compound / model | ω-MAE (meV) | softening index s | imaginary modes | ω-MAE vs PBEsol (cross-functional) |
| --- | --- | --- | --- | --- |
| FeSi / B0 | <!-- num:eval.phonons.FeSi.B0.omega_mae_meV -->8.321<!-- /num --> | <!-- num:eval.phonons.FeSi.B0.softening_index -->0.7775<!-- /num --> | <!-- num:eval.phonons.FeSi.B0.imaginary_count -->0<!-- /num --> | <!-- num:eval.phonons.FeSi.B0.omega_mae_meV_pbesol -->pending<!-- /num --> |
| FeSi / B1 | <!-- num:eval.phonons.FeSi.B1.omega_mae_meV -->0.5695<!-- /num --> | <!-- num:eval.phonons.FeSi.B1.softening_index -->1.002<!-- /num --> | <!-- num:eval.phonons.FeSi.B1.imaginary_count -->0<!-- /num --> | <!-- num:eval.phonons.FeSi.B1.omega_mae_meV_pbesol -->pending<!-- /num --> |
| FeSi / B2 | <!-- num:eval.phonons.FeSi.B2.omega_mae_meV -->0.6026<!-- /num --> | <!-- num:eval.phonons.FeSi.B2.softening_index -->1.004<!-- /num --> | <!-- num:eval.phonons.FeSi.B2.imaginary_count -->0<!-- /num --> | <!-- num:eval.phonons.FeSi.B2.omega_mae_meV_pbesol -->pending<!-- /num --> |
| CoSi / B0 | <!-- num:eval.phonons.CoSi.B0.omega_mae_meV -->2.242<!-- /num --> | <!-- num:eval.phonons.CoSi.B0.softening_index -->0.9331<!-- /num --> | <!-- num:eval.phonons.CoSi.B0.imaginary_count -->0<!-- /num --> | <!-- num:eval.phonons.CoSi.B0.omega_mae_meV_pbesol -->pending<!-- /num --> |
| CoSi / B1 | <!-- num:eval.phonons.CoSi.B1.omega_mae_meV -->0.4354<!-- /num --> | <!-- num:eval.phonons.CoSi.B1.softening_index -->1.002<!-- /num --> | <!-- num:eval.phonons.CoSi.B1.imaginary_count -->0<!-- /num --> | <!-- num:eval.phonons.CoSi.B1.omega_mae_meV_pbesol -->pending<!-- /num --> |
| CoSi / B2 | <!-- num:eval.phonons.CoSi.B2.omega_mae_meV -->0.4594<!-- /num --> | <!-- num:eval.phonons.CoSi.B2.softening_index -->1.003<!-- /num --> | <!-- num:eval.phonons.CoSi.B2.imaginary_count -->0<!-- /num --> | <!-- num:eval.phonons.CoSi.B2.omega_mae_meV_pbesol -->pending<!-- /num --> |

_Pending rows (no number published yet): FeSi / B3, CoSi / B3, MnSi / B0, MnSi / B1, MnSi / B2, MnSi / B3, FeGe / B0, FeGe / B1, FeGe / B2, FeGe / B3, phononDB103 / B0, phononDB103 / B1, phononDB103 / B2, phononDB103 / B3._

_ω-MAE over sorted branches at 100 seekpath q-points, model at the DFT cell; s = median ω_model/ω_ref; imaginary = ω < −0.4 meV. B20 rows cite the project's own QE PBE; phononDB103 rows cite VASP PBE (other code, same functional). The last column is the only place a PBEsol reference appears._

<details><summary>Provenance (one line per cell)</summary>

- **FeSi / B0 / ω-MAE (meV)** — reference qe/PBE (SSSP-efficiency-1.3); E0 unspecified; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205055-fb89d0-none`
- **FeSi / B0 / softening index s** — reference qe/PBE (SSSP-efficiency-1.3); E0 unspecified; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205055-fb89d0-none`
- **FeSi / B0 / imaginary modes** — reference qe/PBE (SSSP-efficiency-1.3); E0 unspecified; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205055-fb89d0-none`
- **FeSi / B1 / ω-MAE (meV)** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205105-fb89d0-none`
- **FeSi / B1 / softening index s** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205105-fb89d0-none`
- **FeSi / B1 / imaginary modes** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205105-fb89d0-none`
- **FeSi / B2 / ω-MAE (meV)** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205111-fb89d0-none`
- **FeSi / B2 / softening index s** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205111-fb89d0-none`
- **FeSi / B2 / imaginary modes** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205111-fb89d0-none`
- **CoSi / B0 / ω-MAE (meV)** — reference qe/PBE (SSSP-efficiency-1.3); E0 unspecified; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205118-fb89d0-none`
- **CoSi / B0 / softening index s** — reference qe/PBE (SSSP-efficiency-1.3); E0 unspecified; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205118-fb89d0-none`
- **CoSi / B0 / imaginary modes** — reference qe/PBE (SSSP-efficiency-1.3); E0 unspecified; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205118-fb89d0-none`
- **CoSi / B1 / ω-MAE (meV)** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205124-fb89d0-none`
- **CoSi / B1 / softening index s** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205124-fb89d0-none`
- **CoSi / B1 / imaginary modes** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205124-fb89d0-none`
- **CoSi / B2 / ω-MAE (meV)** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205130-fb89d0-none`
- **CoSi / B2 / softening index s** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205130-fb89d0-none`
- **CoSi / B2 / imaginary modes** — reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 100; seed 0; CI95 (no CI: deterministic harmonic quantity over a fixed q-path; no resampling); run `20260922T205130-fb89d0-none`

</details>
<!-- gen:end -->


### Molecular dynamics and thermal expansion

<!-- gen:start:parity -->
Parity gate (twenty frames, ASE against the second MD engine): **passed**;
max|ΔF| <!-- num:md.parity.max_dF_eVA -->3.908e-14<!-- /num --> eV/Å, max|ΔE| <!-- num:md.parity.max_dE_eV_atom -->7.994e-15<!-- /num --> eV/atom.
<!-- gen:end -->

<!-- gen:start:thermal -->
| Compound / engine | a(300 K) (Å) | a experiment (Å) | deviation (%) | α (1/K) |
| --- | --- | --- | --- | --- |
| FeSi / ASE | <!-- num:md.ase.FeSi.a_300K_A -->4.4346<!-- /num --> [4.4304, 4.4387] | <!-- num:md.ase.FeSi.a_exp_A -->4.489<!-- /num --> | <!-- num:md.ase.FeSi.a_dev_pct -->-1.213<!-- /num --> | <!-- num:md.ase.FeSi.alpha_per_K -->pending<!-- /num --> |
| CoSi / ASE | <!-- num:md.ase.CoSi.a_300K_A -->4.4279<!-- /num --> [4.4267, 4.4290] | <!-- num:md.ase.CoSi.a_exp_A -->4.444<!-- /num --> | <!-- num:md.ase.CoSi.a_dev_pct -->-0.3629<!-- /num --> | <!-- num:md.ase.CoSi.alpha_per_K -->pending<!-- /num --> |
| MnSi / ASE | <!-- num:md.ase.MnSi.a_300K_A -->4.5573<!-- /num --> [4.5543, 4.5603] | <!-- num:md.ase.MnSi.a_exp_A -->4.558<!-- /num --> | <!-- num:md.ase.MnSi.a_dev_pct -->-0.01557<!-- /num --> | <!-- num:md.ase.MnSi.alpha_per_K -->0.000006563<!-- /num --> |
| MnSi / LAMMPS | <!-- num:md.lammps.MnSi.a_300K_A -->4.5568<!-- /num --> [4.5567, 4.5569] | <!-- num:md.lammps.MnSi.a_exp_A -->4.558<!-- /num --> | <!-- num:md.lammps.MnSi.a_dev_pct -->-0.02683<!-- /num --> | <!-- num:md.lammps.MnSi.alpha_per_K -->pending<!-- /num --> |
| FeGe / ASE | <!-- num:md.ase.FeGe.a_300K_A -->4.7009<!-- /num --> [4.6956, 4.7062] | <!-- num:md.ase.FeGe.a_exp_A -->4.7<!-- /num --> | <!-- num:md.ase.FeGe.a_dev_pct -->0.01874<!-- /num --> | <!-- num:md.ase.FeGe.alpha_per_K -->pending<!-- /num --> |

_Pending rows (no number published yet): FeSi / LAMMPS, CoSi / LAMMPS, FeGe / LAMMPS._

_NPT thermal expansion at 300 K against experiment; cell size, trajectory length and time step of every row are in its provenance line. Every cell of a row cites reference code `experiment` (the literature lattice constant is itself a published number). Rows of a second engine appear only after the parity gate passes._

<details><summary>Provenance (one line per cell)</summary>

- **FeSi / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 [4.43, 4.439]; 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T224837-fd5fe0-0`
- **FeSi / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: tabulated experimental value); 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T224837-fd5fe0-0`
- **FeSi / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T224837-fd5fe0-0`
- **CoSi / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 [4.427, 4.429]; 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225220-fd5fe0-0`
- **CoSi / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: tabulated experimental value); 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225220-fd5fe0-0`
- **CoSi / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225220-fd5fe0-0`
- **MnSi / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 1501; seed 0; CI95 [4.554, 4.56]; 64 atoms, 40 ps at 2 fs incl. equilibration; run `20260920T230308-fd5fe0-0`
- **MnSi / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 1501; seed 0; CI95 (no CI: tabulated experimental value); 64 atoms, 40 ps at 2 fs incl. equilibration; run `20260920T230308-fd5fe0-0`
- **MnSi / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 1501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); 64 atoms, 40 ps at 2 fs incl. equilibration; run `20260920T230308-fd5fe0-0`
- **MnSi / ASE / α (1/K)** — reference experiment/None; E0 foundation; head Default; n = 3; seed 0; CI95 (no CI: fewer than 4 temperatures: no slope error); run `20260920T230308-fd5fe0-0`
- **MnSi / LAMMPS / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 1001; seed 0; CI95 [4.557, 4.557]; 512 atoms, 30 ps at 2 fs incl. equilibration; run `20260922T205327-5f3c7c-0`
- **MnSi / LAMMPS / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 1001; seed 0; CI95 (no CI: tabulated experimental value); 512 atoms, 30 ps at 2 fs incl. equilibration; run `20260922T205327-5f3c7c-0`
- **MnSi / LAMMPS / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 1001; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); 512 atoms, 30 ps at 2 fs incl. equilibration; run `20260922T205327-5f3c7c-0`
- **FeGe / ASE / a(300 K) (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 [4.696, 4.706]; 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225557-fd5fe0-0`
- **FeGe / ASE / a experiment (Å)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: tabulated experimental value); 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225557-fd5fe0-0`
- **FeGe / ASE / deviation (%)** — reference experiment/None; E0 foundation; head Default; n = 501; seed 0; CI95 (no CI: derived from a_300K_A and a_exp_A); 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225557-fd5fe0-0`

</details>
<!-- gen:end -->


<!-- gen:start:stability -->
| Compound / engine | NVE drift (meV/atom/ps) |
| --- | --- |
| MnSi / ASE | <!-- num:md.ase.MnSi.drift_meV_atom_ps -->0.0001657<!-- /num --> [0.00002303, 0.0003084] |
| MnSi / LAMMPS | <!-- num:md.lammps.MnSi.drift_meV_atom_ps -->0.00001423<!-- /num --> [-0.0001984, 0.0002268] |

_Pending rows (no number published yet): FeSi / ASE, FeSi / LAMMPS, CoSi / ASE, CoSi / LAMMPS, FeGe / ASE, FeGe / LAMMPS._

_NVE energy drift of separate NVE runs (cell size and length in the provenance lines); self-consistency numbers cite reference code `mace` with the training functional._

<details><summary>Provenance (one line per cell)</summary>

- **MnSi / ASE / NVE drift (meV/atom/ps)** — reference mace/PBE; E0 foundation; head Default; n = 501; seed 0; CI95 [0.00002303, 0.0003084]; 64 atoms, 20 ps at 2 fs incl. equilibration; run `20260920T225932-fd5fe0-0`
- **MnSi / LAMMPS / NVE drift (meV/atom/ps)** — reference mace/PBE; E0 foundation; head Default; n = 251; seed 0; CI95 [-0.0001984, 0.0002268]; 512 atoms, 6 ps at 2 fs incl. equilibration; run `20260922T190745-a730b0-0`

</details>
<!-- gen:end -->


### Vacancy hop: umbrella sampling versus NEB

<!-- gen:start:sampling -->
| Compound | umbrella barrier ΔF‡ (eV) | end-to-end ΔF (eV) | block error (eV) | NEB E_a (eV) |
| --- | --- | --- | --- | --- |
| FeSi | <!-- num:sampling.wham.FeSi.B0.dF_barrier_eV -->0.7417<!-- /num --> [0.718, 0.7654] | <!-- num:sampling.umbrella.FeSi.dF_eV -->-0.001228<!-- /num --> [-0.0199, 0.01744] | <!-- num:sampling.umbrella.FeSi.dF_err_eV -->0.006724<!-- /num --> | <!-- num:sampling.neb.FeSi.Ea_eV -->0.6923<!-- /num --> |

_Pending rows (no number published yet): CoSi, MnSi, FeGe._

_Vacancy hop at 300 K: 24 umbrella windows × 15 ps (Langevin), MBAR free-energy profile (WHAM cross-check) with block error, against the NEB barrier on the same model (reference code `mace`). The hop is between equivalent sites, so the end-to-end ΔF is a consistency check that should vanish; the barrier ΔF‡ is the number to compare with NEB._

<details><summary>Provenance (one line per cell)</summary>

- **FeSi / umbrella barrier ΔF‡ (eV)** — reference mace/PBE; E0 foundation; head Default; n = 144024; seed 0; CI95 [0.718, 0.7654]; run `20260920T225238-fb89d0-none`
- **FeSi / end-to-end ΔF (eV)** — reference mace/PBE; E0 foundation; head Default; n = 144024; seed 0; CI95 [-0.0199, 0.01744]; run `20260920T225238-fb89d0-none`
- **FeSi / block error (eV)** — reference mace/PBE; E0 foundation; head Default; n = 144024; seed 0; CI95 (no CI: is itself the block standard error); run `20260920T225238-fb89d0-none`
- **FeSi / NEB E_a (eV)** — reference mace/PBE; E0 foundation; head Default; n = 7; seed 0; CI95 (no CI: deterministic NEB barrier (no sampling)); run `20260919T085607-fb89d0-none`

</details>
<!-- gen:end -->


### Agent evaluation

<!-- gen:start:agent -->
| Metric | value |
| --- | --- |
| accuracy | <!-- num:agent.eval.accuracy -->1<!-- /num --> [1, 1] |
| invalid-call rate | <!-- num:agent.eval.invalid_call_rate -->0.02778<!-- /num --> [0, 0.08333] |
| DAG-valid order rate | <!-- num:agent.eval.dag_valid_rate -->1<!-- /num --> [1, 1] |
| provenance completeness | <!-- num:agent.eval.provenance_rate -->0.9167<!-- /num --> [0.75, 1] |
| recovery from injected failures | <!-- num:agent.eval.recovery_rate -->1<!-- /num --> [1, 1] |
| tokens per task | <!-- num:agent.eval.tokens_per_task -->13770<!-- /num --> [11940, 15670] |
| USD per task | <!-- num:agent.eval.usd_per_task -->0.08601<!-- /num --> [0.0741, 0.09874] |
| tasks | <!-- num:agent.eval.n_tasks -->12<!-- /num --> |

_Twelve-task eval (`evals/agent_tasks.jsonl`, gold from `b20mlip screen`); backend and model id are in the provenance list. Mock-backend replays (CI) are never shown here._

<details><summary>Provenance (one line per cell)</summary>

- **accuracy / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 [1, 1]; run `20260922T181438-26ca37-0`
- **invalid-call rate / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 [0, 0.08333]; run `20260922T181438-26ca37-0`
- **DAG-valid order rate / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 [1, 1]; run `20260922T181438-26ca37-0`
- **provenance completeness / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 [0.75, 1]; run `20260922T181438-26ca37-0`
- **recovery from injected failures / value** — reference mace/PBE; E0 foundation; head Default; n = 4; seed 0; CI95 [1, 1]; run `20260922T181438-26ca37-0`
- **tokens per task / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 [11940, 15670]; run `20260922T181438-26ca37-0`
- **USD per task / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 [0.0741, 0.09874]; run `20260922T181438-26ca37-0`
- **tasks / value** — reference mace/PBE; E0 foundation; head Default; n = 12; seed 0; CI95 (no CI: a count); run `20260922T181438-26ca37-0`

</details>
<!-- gen:end -->


### One-line summary

<!-- gen:start:bullet -->
> Fine-tuned MACE-MPA-0 (equivariant GNN) on <!-- num:data.n_train_frames -->193<!-- /num --> in-house spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi/MnSi/CoSi; <!-- num:data.n_qe_frames -->457<!-- /num --> labelled in total): held-out force MAE <!-- num:eval.errors.T0.B0.mae_f -->24.24<!-- /num -->→<!-- num:eval.errors.T0.B2.mae_f -->8.578<!-- /num --> meV/Å, phonon ω-MAE vs same-code DFT <!-- num:eval.phonons.FeSi.B0.omega_mae_meV -->8.321<!-- /num -->→<!-- num:eval.phonons.FeSi.B2.omega_mae_meV -->0.6026<!-- /num --> meV (FeSi), never-trained FeGe/MnGe <!-- num:eval.errors.T2.B0.mae_f -->90.02<!-- /num -->→<!-- num:eval.errors.T2.B2.mae_f -->77.84<!-- /num --> meV/Å; forgetting quantified as paired ΔF1 = <!-- num:eval.discovery.B2.delta_f1 -->-0.1098<!-- /num --> [-0.1603, -0.06316] on a labelled 1,000-structure WBM sample (Matbench-Discovery protocol, no public ranking claimed); deployed in ASE/LAMMPS MD (zero-shot MPA-0 MnSi 300 K lattice constant <!-- num:md.ase.MnSi.a_dev_pct -->-0.01557<!-- /num --> % from experiment), umbrella-sampled a vacancy-hop free-energy barrier (ΔF‡ = <!-- num:sampling.wham.FeSi.B0.dF_barrier_eV -->0.7417<!-- /num --> eV vs NEB <!-- num:sampling.neb.FeSi.Ea_eV -->0.6923<!-- /num --> eV), open-sourced (MIT) with a provenance-checked tool-calling agent.
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
- The second MD engine is LAMMPS (`pair_style mace`, Kokkos on a cluster GPU) running the same
  exported model; its only equivalence claim is the parity gate above, and its rows carry their
  own cell size and length in the provenance lines.
- Same-code phonon references exist for FeSi and CoSi only; the MnSi and FeGe displacement sets
  were not computed (cluster budget), so their phonon rows are absent rather than compared
  against another code.

## Plan

- Next, cluster budget permitting: QE phonon displacement sets for MnSi and FeGe; one
  committee-selected active-learning round (the selector is implemented, the QE labelling of its
  picks is not done); a forgetting tier restricted to MPtrj frames whose geometry was never
  trained on.
- Done for v1.0: the QE round, brackets with seed means, phonons against the project's own QE,
  the WBM sample for B0 and B2, the parity gate, the live agent eval and a green audit.

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
