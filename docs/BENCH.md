# Day-1 timing task: `b20mlip bench`

`b20mlip bench --out runs/bench/` (SPEC.md section 15, CONTRACTS.md row 15) measures, on this Mac
(M2 Pro, CPU float64, `compute.threads` = 6 threads) with the MACE-MPA-0 medium foundation model,
the five unit costs every schedule number of SPEC.md section 5 and section 10 depends on, and
re-derives that runtime table from them. Bench numbers are **not** README claims: they live in
`runs/bench/bench.json` (the resumable cache) and in each run's `numbers.json` under
`bench.<measurement>.<metric>`, never in a `<!-- num:... -->` marker.

## 1. Running it

```bash
uv run --no-sync b20mlip bench --out runs/bench/                       # all five, MPA-0 medium
uv run --no-sync b20mlip --set bench.tiers=b,c,d,e bench --out runs/bench/   # the fast four first
uv run --no-sync b20mlip --set bench.tiers=a bench --out runs/bench/         # the fine-tune epoch
uv run --no-sync b20mlip --set bench.force=true bench --out runs/bench/      # re-measure everything
uv run --no-sync b20mlip --set bench.model=runs/train/<id>/models/b20_naive.model bench --out runs/bench_b1/
uv run --no-sync b20mlip --dry-run bench --out runs/bench/                   # plan only (partial manifest)
```

The root CLI takes only `--out`; every knob is a `--set bench.<key>=<value>` (section `bench:` of
`configs/default.yaml`):

| key | default | meaning |
|---|---|---|
| `model` | `null` | MACE `.model` to time; `null` = the foundation named by `train.foundation` (MPA-0 medium) |
| `tiers` | `a,b,c,d,e` | comma list, letters or names (`finetune,md,relax,phonons,fwbw`) |
| `quick` | `false` | tiny-model sizes for CI: 8 + 5 frames, one epoch, 20 MD steps, 2 relaxations, 1x1x1 phonons |
| `force` | `false` | re-measure sub-tasks already cached in `<out>/bench.json` |
| `compound` | `FeSi` | reference cell (dft tier's `reference_frame`) for MD, phonons and the fwbw batch |
| `n_frames_8atom` / `n_frames_64atom` | 80 / 20 | size classes of measurement (a) |
| `finetune_epochs` | 2 | the last epoch is the measurement (epoch 0 carries warm-up) |
| `finetune_timeout_s` | 3600 | per `mace_run_train` process |
| `md_natoms` / `md_steps` / `md_warmup_steps` / `md_T` | `[64, 512]` / 200 / 10 / 300 | measurement (b) |
| `n_relax` | 20 | first ids of `data/wbm/sample_1000_s0.json` (measurement (c)) |
| `phonon_supercell` / `phonon_distance` | 2 / 0.03 | measurement (d) |
| `fwbw_batch` / `fwbw_natoms` / `fwbw_repeats` | 4 / 64 / 3 | measurement (e) |

Every measurement is a named sub-task. Its result (or `{"error": ...}`) is written into
`<out>/bench.json` the moment it finishes, so a crash loses at most one sub-task and a re-run skips
what is done (same model sha256, same `quick` flag, same parameters; `bench.force=true` overrides).
A different model or quick flag moves the old file to `bench.stale-<utc>.json` and starts fresh.
The stage returns `status="ok"` only when every requested sub-task has a number; otherwise
`"partial"` and the manifest says which failed. Nothing is ever estimated in place of a failed
measurement. Per run, `runs/bench/<run_id>/` holds the manifest, `plan.json`, `numbers.json` and
`bench_report.md` (the tables below); `<out>/finetune/<class>/` and `<out>/fwbw/` keep the MACE
logs, results files and frames of (a) and (e) (trained weights are deleted: they are not a result).

## 2. What is measured

| | sub-task | what runs | reported |
|---|---|---|---|
| (a) | `finetune` | one naive fine-tuning epoch per size class: `mace_run_train` via `train.finetune.build_args(variant="naive", energy_scale="mp")`, `--E0s foundation`, batch `train.batch_size`. 8-atom class: the MPtrj B20 frames with their DFT labels, cycled/rattled copies (0.05 A) labelled zero-shot by the same model beyond the source; 64-atom class: 2x2x2 supercells rattled by 0.02 A with zero-shot labels (`label_source="mace_zero_shot"`, `energy_scale="mp"`). Train part cut to a multiple of the batch size (MACE drops the last incomplete batch), the rest validates. `finetune_epochs` epochs; the last epoch is the measurement. | `s_per_frame_<class>` = wall of the measured epoch (from the timestamps of MACE's `Initial:`/`Epoch k:` validation lines: training + validation, no start-up) / n_train; `s_per_frame_steps` = sum of MACE's optimiser-step times / n_train; `startup_s` |
| (b) | `md` | NVT Langevin (`cfg.md` friction, `md.timestep_fs`) on the FeSi reference cell repeated to 64 and 512 atoms, seeded Maxwell-Boltzmann start | `s_per_step_<natoms>` = wall / (steps - warm-up), atom-steps per second |
| (c) | `relax` | `evaluate.discovery.relax` (FIRE + FrechetCellFilter, `eval.fmax`, `eval.max_steps`) on the first `n_relax` WBM-sample initial structures | mean/median s and steps per structure, `n_capped`, `s_per_relax_step` = total wall / total steps |
| (d) | `phonons` | `phonons.harmonic.compute` (phonopy finite displacements, seekpath path, DOS) on the reference cell, `phonon_supercell`^3, `phonon_distance` | wall `s`, `n_displacements`, `s_per_displacement` |
| (e) | `fwbw` | in a fresh Python process: the model loaded with torch, one batch of `fwbw_batch` rattled 64-atom cells built the way `mace_run_train` builds batches, forward with forces + `loss.backward()`, `fwbw_repeats` timed passes after a warm-up | `peak_rss_gb` (`ru_maxrss` of the child, GiB; bytes on macOS, kB on Linux), `s_per_batch`, edges |

## 3. Schedule derivation (`derive_schedule`)

The SPEC.md section 5 table is recomputed from the measurements with these formulas (also stored
as strings under `"schedule"` in `bench.json`; `epochs` = `train.epochs`, `dt` = `md.timestep_fs`,
temperatures = `data.temperatures_K`, windows = `sampling.windows` x `sampling.ps_per_window`,
`N_wbm` = `data.wbm_sample_n`, cap = `eval.max_steps`):

| entry | formula |
|---|---|
| `finetune_round0_epoch_s` | `600 * s_per_frame_8atom + 60 * s_per_frame_64atom` |
| `finetune_round0_per_seed_min` | `(epochs * finetune_round0_epoch_s + startup_s) / 60` |
| `finetune_round0_3seeds_h` | `3 * finetune_round0_per_seed_min / 60` |
| `md_nvt_64_40ps_per_T_h` | `(40 ps / dt) * s_per_step_64 / 3600` |
| `md_nvt_64_40ps_3T_h` | `n_temperatures * md_nvt_64_40ps_per_T_h` |
| `md_512_100ps_ase_h` | `(100 ps / dt) * s_per_step_512 / 3600` (the Mac/ASE cost of the LAMMPS parity job) |
| `umbrella_windows_64_h` | `windows * (ps_per_window / dt) * s_per_step_64 / 3600` |
| `wbm_relax_h` | `N_wbm * s_per_structure / 3600` |
| `wbm_relax_cap_h` | `N_wbm * cap * s_per_relax_step / 3600` (every structure at the step cap) |
| `phonondb103_h` | `103 * phonons_s / 3600` |
| `fwbw_ram_headroom_gib` | `host RAM (GiB) - peak_rss_gb` |

An entry whose inputs are missing (a sub-task not run or failed) carries `error` instead of a
value; nothing is back-filled from the pre-bench estimates.

## 4. Results

<!-- filled by `b20mlip bench` (bench_report of runs/bench/bench.json); regenerate with
     uv run --no-sync python -c "from b20mlip.bench import bench_report; print(bench_report('runs/bench/bench.json'))" -->

Model `mace-mpa-0-medium.model` (sha256 `75428afe3a1d`), host `Wenqins-MacBook-Pro-10.local` (arm64), 6 threads, float64, quick=False, status **ok**, updated 2026-09-19T07:42:32+00:00.

| Measurement | Metric | Value | Unit | n | Notes |
|---|---|---|---|---|---|
| (a) finetune | s_per_frame_8atom | 0.139 | s/frame/epoch | 72 | epoch wall 10 s, 18 batches of 4, steps-only 0.13 s/frame, epoch 1 of 2 |
| (a) finetune | s_per_frame_64atom | 2.47 | s/frame/epoch | 16 | epoch wall 39.5 s, 4 batches of 4, steps-only 2.36 s/frame, epoch 1 of 2 |
| (a) finetune | startup_s | 5.51 | s | 2 | mace_run_train start-up (model load, data) before validation 0 |
| (b) md | s_per_step_64 | 0.265 | s/step | 190 | nvt 300 K, 2 fs, 241 atom-steps/s |
| (b) md | s_per_step_512 | 9.69 | s/step | 190 | nvt 300 K, 2 fs, 52.9 atom-steps/s |
| (c) relax | s_per_structure | 1.14 | s | 20 | median 0.915 s; fmax 0.05, cap 500, capped 0 |
| (c) relax | steps_per_structure | 20.4 | steps | 20 | median 19; 0.0557 s per FIRE step |
| (d) phonons | FeSi 2x2x2 s | 2.66 | s | 4 | 4 displacements of 64 atoms, 0.03 A; 0 imaginary |
| (e) fwbw | peak_rss_gb | 4.58 | GiB | 3 | fresh process; before load 0.384, batch built 0.501 GiB |
| (e) fwbw | s_per_batch | 14.1 | s | 3 | 4 x 64 atoms, 18840 edges, forward+backward with forces |

| Schedule entry | Value | Formula | SPEC row |
|---|---|---|---|
| finetune_round0_epoch_s | 232 | `600 * s_per_frame_8atom + 60 * s_per_frame_64atom` | train naive round-0 (600 x 8 + 60 x 64 frames), s/epoch |
| finetune_round0_per_seed_min | 116 | `(30 epochs * finetune_round0_epoch_s + startup_s) / 60` | train naive: 30 epochs per seed |
| finetune_round0_3seeds_h | 5.8 | `3 seeds * finetune_round0_per_seed_min / 60` | train naive: 3 seeds overnight |
| md_nvt_64_40ps_per_T_h | 1.47 | `(40.0 ps / 2.0 fs) steps * s_per_step_64 / 3600` | md ase: 64 atoms, 40 ps per temperature |
| md_nvt_64_40ps_3T_h | 4.42 | `3 temperatures * md_nvt_64_40ps_per_T_h` | md ase: 40 ps x 3 temperatures overnight |
| md_512_100ps_ase_h | 135 | `(100.0 ps / 2.0 fs) steps * s_per_step_512 / 3600` | md: 100 ps at 512 atoms on the Mac with ASE (the LAMMPS/GPU job's local cost) |
| umbrella_windows_64_h | 6.63 | `12 windows * (15.0 ps / 2.0 fs) steps * s_per_step_64 / 3600` | sampling umbrella: 12 windows x 15 ps at 64 atoms |
| wbm_relax_h | 0.316 | `1000 structures * s_per_structure / 3600` | eval discovery: WBM-1000 per model (mean over the timed structures) |
| wbm_relax_cap_h | 7.73 | `1000 structures * 500 steps * s_per_relax_step / 3600` | eval discovery: worst case, every structure at the 500-step cap |
| phonondb103_h | 0.0761 | `103 compounds * phonons_s / 3600` | eval phonons: phononDB-103 per model (2x2x2 FeSi as the unit cost) |
| fwbw_ram_headroom_gib | 11.4 | `host RAM (GiB) - fwbw peak_rss_gb` | train: 64-atom batches fit the Mac (positive headroom) |

Run record (`runs/bench/bench.json`, `"runs"`): fwbw measured in run `20260919T065427-a92bc8-0`
(the md/relax/phonons sub-tasks of that first attempt failed because the foundation file names
its head `default`, fixed the same evening), md + relax + phonons in `20260919T065619-a92bc8-0`
(33 min, 31 of them the 512-atom MD), finetune in `20260919T074006-4a6dc7-0` (2.4 min). MACE
argv, logs and results files of (a): `runs/bench/finetune/<class>/`; the fwbw child log and batch:
`runs/bench/fwbw/`.

### 4.1 What the numbers say (2026-09-19, MPA-0 medium, 6 threads, float64)

* **8-atom frames cost 0.14 s/frame/epoch** (wall, 72 train + 8 valid frames; optimiser steps
  alone 0.13 s/frame): the 0.12 s/frame figure of `docs/design/facts_2026-09-18.md` holds.
* **64-atom frames cost 2.5 s/frame/epoch**, i.e. about 18x an 8-atom frame, not the 8x SPEC.md
  section 5 assumed: 2.2x more per atom (a batch of 4 x 64 atoms carries 18,840 edges and
  float64 intermediates of hundreds of MB, and the machine was swapping during the measurement,
  see the caveats). The round-0 epoch of 600 x 8 + 60 x 64 frames is therefore 232 s = 3.9 min
  (SPEC: 2.2 min), 30 epochs = 1.9 h per seed and 5.8 h for three seeds: still one overnight
  slot, but not two runs per night.
* **64-atom NVT MD: 0.265 s/step** (SPEC: 0.25): 40 ps per temperature = 1.5 h, the three
  temperatures 4.4 h (SPEC: 5 h); the umbrella windows (12 x 15 ps) 6.6 h (SPEC: 6 h).
* **512-atom MD does not fit this Mac**: the calculator needs about 8.8 GB for 512 atoms in
  float64 and the desktop session already used most of the 16 GB, so the 9.7 s/step (52
  atom-steps/s against 241 at 64 atoms) was measured while macOS was swapping (9 GB of swap in
  use, 60 % system CPU). It is the honest number for "this machine as it was", not a clean
  kernel timing; extrapolated linearly from 64 atoms the clean cost would be about 2.1 s/step. Either
  way the 100 ps / 512-atom parity job stays on the cluster (135 h here), as SPEC.md planned.
* **WBM relaxations are cheap**: 1.1 s per structure (median 0.9 s, 20 steps, none capped, 8
  atoms on average): WBM-1000 = 0.3 h per model; the 500-step cap bounds it at 7.7 h. SPEC's
  3-6 h/model was pessimistic by an order of magnitude for the initial-structure sample.
* **Phonons**: FeSi 2x2x2 needs four displacements (P2_1 3 symmetry) and 2.7 s; phononDB-103 at
  that unit cost is 5 min, so the 1 h/model of SPEC.md is dominated by the compounds with lower
  symmetry and larger cells, not by FeSi-like ones.
* **Memory**: a training batch of 4 x 64 atoms peaks at 4.6 GiB RSS in a fresh process (11.4 GiB
  of headroom on paper; in practice the desktop session leaves less). Forward + backward with
  forces took 14 s per batch there against 9.4 s per optimiser step (energy, forces and stress)
  inside `mace_run_train` 50 minutes later; the two ran under different memory loads and the
  bench does not resolve the difference, both are reported as measured.

Caveats of this run: another `pytest --cov` process (a sibling tier's build) used about half a
core during the fine-tune measurement, and swap was in use throughout (the 512-atom MD pushed the
machine into it). Re-run with `bench.force=true` on an idle machine to tighten the numbers; every
figure above is single-run (`ci95: null`, `ci95_reason: "single timing run"`).
