# Judge: benchmark-maintainer view of the P3 designs

|design|real_value|honesty|production|2-week|hiring|total|
|---|---|---|---|---|---|---|
|impact|8|7|7|6|9|**37**|
|production|7|7|9|5|8|36|
|mvp|6|5|7|8|6|32|

Winner: **impact** (real questions, single-code label+reference chain, three MFA logins not eight). None ships as written.

## Fatal flaws
1. **Energy scale.** OMat24 is functional-consistent with MPtrj (PBE, same +U scheme; B20 has no O/F so no U anywhere) but uses a different PAW set, so its total energies are off the MP scale; QE/SSSP energies are a third scale. Naive FT with `--E0s estimated` (impact), isolated-atom E0s (production) or `--E0s foundation` on OMat24 (mvp: plainly wrong) shifts the Fe/Mn/Co/Si/Ge readout; every WBM structure with those elements moves relative to the MP hull, so "forgetting" in the in-chemsys stratum is partly a reference artefact, and MP2020 corrections on such energies are meaningless. Fix: QE single points on the MPtrj B20 frames (same structures, both codes) → per-element offsets → QE energies on the MP scale, foundation E0s kept, residual published; residual >~20 meV/atom → no hull metric for that model. Replay models: evaluate WBM with the MP head and say so. Always add force-only forgetting on held-out MPtrj frames.
2. **F1 on a subset.** 50/50 stability strata (impact) or e_above_hull bins (production) change prevalence (WBM ≈17 % stable): precision, F1 and DAF become incomparable with any leaderboard value; DAF is capped at 1/prevalence. Report F1 only as paired Δ vs MPA-0 zero-shot on the identical seeded sample with bootstrap CI, at natural prevalence or inverse-probability-weighted; never DAF on a stratified sample; never a leaderboard number in the same table; nothing on <103 materials is "κ_SRME" (mvp stretch); no CPS.
3. **"Phonon MAE vs DFT".** phonondb and MP-DFPT are PBEsol; MbD-103 and own QE are PBE; PySCF is GTH/Gaussian. One pooled MAE (mvp bullet) is indefensible. One reference code+functional per table row, geometry stated (DFT cell vs model-relaxed), softening index only vs same-functional PBE, PBEsol in a separate "cross-functional" column. Impact alone keeps labels and references in one code; preserve that.
4. **Leakage.** Per-frame sha split (production) puts rattles of one parent in train and test; use mvp's group split by parent plus OOD tiers. Disclose that MPA-0 saw the MPtrj equilibrium frames of mp-871/1431/7577/21255 and, via sAlex, the relaxed parents of OMat24 frames: "unseen" holds per frame only.
5. **Overclaims.** "First public PBE phonon references" (impact) → "absent from phonondb, MP-DFPT and the Alexandria/Loew PBE phonon set as of <date>". "No licence entanglement" (production) → SSSP mixes GPL and CC-BY pseudopotentials: cite + md5, never redistribute. "OMat24 structures of B20 chiral magnets" (mvp) only with spglib-verified P2₁3 counts, else "Mn–Fe–Co–Si–Ge space". mvp Day 1 calls MPRester without a key.

## Data fork
**In-house QE, primary.** Only it yields targeted B20 frames (strain, rattle, finite-T, vacancy), same-code PBE phonon/elastic references, and a real closed loop (OMat24 cannot label new frames, so mvp's "active learning" keyword is unbacked). **OMat24: external, never-trained-on VASP-PBE force/stress test set** for in-chemsys frames (state subset and max|F| filter); energies only after offset fitting. No three-scale hybrid training; if OMat24 frames are ever trained on, per-config `energy_weight=0`, stated. Publish a cross-code force noise floor (20 QE single points on OMat24/MPtrj structures); differences below it are not claims. PySCF is a labelled lower-fidelity fallback, never the headline reference.

## Grafts into the final spec
From production: manifest schema + `--resume`; `report --audit` CI gate; `dft converge` thresholds; DFT noise floor and failure counts; bootstrap CIs per tier; StructureMatcher dedupe; MODEL_CARD/DATA_CARD; scripted planner + trace replay in CI; agent budget caps and `submit_dft` dry-run gate; tiny fixture model for offline tests.
From mvp: group split by parent; LOCO tier; 300/600/all data curve; `--llm mock`; v0.1 vertical slice tagged by Day 4 so a bullet exists if the cluster dies.

## Extra honesty rules
- Every metric names reference code, functional, pseudopotential set, E0 source, head evaluated, N, seed, CI.
- Bullet: "labelled 1,000-structure WBM sample, not the leaderboard"; no "first"; "LAMMPS" only after parity.
- Magnetism: collinear-FM PES, no spin DOF/DMI; PBE moment overestimate (MnSi) disclosed.
- Fine-tuned weights MIT with MPtrj/sAlex (+OMat24 CC-BY) attribution in NOTICE.
- Claude model id pinned from the claude-api reference, recorded in manifests, never hard-coded in claims.
- Negative results ship.
