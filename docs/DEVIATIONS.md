# Deviations from SPEC v1.0 / CONTRACTS v1.0 (running log)

Recorded as they are discovered during the build; the SPEC and CONTRACTS files are frozen at v1.0 and are
not edited retroactively.

| Date | Item | SPEC/CONTRACTS said | Reality | Consequence |
|---|---|---|---|---|
| 2026-09-17 | `[omat]` extra | `fairchem-core` for OMat24 ASE-LMDB | every fairchem-core release needs `e3nn>=0.5`; `mace-torch==0.3.16` pins `e3nn==0.4.4`, so uv cannot lock both | OMat24 `.aselmdb` shards are read natively (`lmdb` + `orjson`); `[omat]` = `lmdb`, `orjson` |
| 2026-09-17 | PySCF baseline | optional lower-fidelity reference (Si 16-atom, FeSi Γ) | 8-atom FeSi Γ spin-polarised KRKS did not finish in 15 min on the Mac (probe killed at 40 min) | `dft/pyscf_pbc.py` stays a guarded optional stage; never on the v0.1 path |
| 2026-09-17 | Local QE | none promised | no `pw.x` on macOS (conda-forge `qe` has no osx-arm64 build; no brew formula) | golden `pw.out` fixture is synthetic (labelled) until the first Tillicum output replaces it |
| 2026-09-18 | Tillicum partitions | "CPU nodes for QE, H200 GPU for replay/LAMMPS" | Tillicum is GPU-only: partitions `gpu-h200` and `gpu-h200-mig`; every job needs `--gpus=1` (8 CPUs + 200 GB per GPU) | QE arrays run on `gpu-h200-mig` (0.143× cost); templates emit `--gpus`, `--qos`; `partition_cpu` is the MIG partition by user choice at bootstrap |
| 2026-09-18 | Scratch layout | `/gscratch/<group>` | that is the Klone layout; Tillicum uses `/gpfs/projects/<group>` or `/gpfs/scrubbed/<user>` (60-day purge) | bootstrap probes both, never assumes |
| 2026-09-18 | `hyakalloc` | account/partition discovery | Klone tool, may be absent on Tillicum | `sacctmgr` + `sinfo` path is the default; raw output kept in the manifest |
| 2026-09-18 | Remote venv | `uv sync --frozen` on the cluster | `uv.lock` pins the CPU torch wheel on Linux | GPU replay jobs install a CUDA torch wheel inside the job (train tier), recorded in the manifest |
| 2026-09-18 | MPtrj B20 set | "60 B20 frames" (facts file, by-mp-id pull) | spglib-198 selector: 65 frames — the old pull included 13 frames of non-cubic FeGe mp-22510 (C2/m, excluded per SPEC §3) and missed CoGe mp-10692 (4) and MnGe mp-1078464 (14; a = 4.762 Å) | `data/frames/mptrj_b20.extxyz` (65) is the offset/keyless set; MnGe needs no OPTIMADE search |
| 2026-09-18 | OMat24 in-family | "few hundred expected" | 656 in-chemsys frames of 1,171,309 scanned; 477 after the 15 eV/Å force cap (179 dropped); 0 are spg-198 | bootstrap/T3 set = 477 frames, stress already ASE-convention (r = +0.965 vs MPA-0) |
| 2026-09-18 | WBM sample prevalence | 16.7 % (population) | 15.2 % realised for seed 0, N = 1000; in-family stratum N = 312, 1.28 % stable | discovery captions report the realised sample prevalence |
| 2026-09-18 | Round-0 candidates | 400–600 × 8-atom | 420 generated → 300 after StructureMatcher dedupe (rotation-equivalent strains/shears merged) | QE round-0 array = 300 + phonon/vacancy/noise-floor units; more rattle seeds if the budget allows |

