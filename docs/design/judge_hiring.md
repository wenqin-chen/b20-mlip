# P3 judge — hiring-manager verdict (applied AI-for-science)

## Scores (1–10, total /50)

| Design | real_value | honesty | prod_quality | feasibility_2w | hiring_signal | Total |
|---|---|---|---|---|---|---|
| impact | 8 | 8 | 7 | 6 | 9 | **38** |
| production | 7 | 9 | 9 | 4 | 8 | 37 |
| mvp | 5 | 7 | 7 | 8 | 6 | 33 |

impact: sharpest questions (softening index, transfer, forgetting, FM-vs-NM MnSi) and the one citable artifact — first public PBE phonon/elastic references for B20 FeSi/MnSi/FeGe. production: best engineering discipline, but 9 of 14 days are cluster-gated. mvp: ships without the cluster, but fine-tuning MPA-0 on OMat24 mostly re-derives MACE-OMAT-0, and "DFT tools"/magnetism are barely backed.

## Fork → hybrid, QE as the headline

- In-house spin-polarised QE labels are the only way to back "running DFT on realistic systems", a controlled FM state, and unpublished B20 references. Spine.
- OMat24 in-chemsys frames enter Day 1 as bracket B4 ("~2k public non-equilibrium frames vs ~600 targeted in-house frames") and as insurance: vertical slice (fine-tune → eval → MD → phonons vs phonondb) tagged v0.1 by Day 4, no cluster.
- Never mix VASP/OMat24 and QE frames in one head with foundation E0s: QE data gets isolated-atom `E0s.json` and its own multihead head (or naive with estimated E0s). QE frames are first an independent cross-code test set, then training data.
- User does the MFA login Day 1; convergence + QE array submitted on that login; only LAMMPS parity and round-1 labels may depend on a later login.
- Login slips past Day 5 → ship v0.1 on OMat24 + PySCF FeSi Γ references; bullet drops "Quantum ESPRESSO"/"LAMMPS". Honest, still competitive.

## Fatal / serious flaws

- **impact**: `--E0s="foundation"` for replay on QE labels is wrong (QE absolute energies ≠ MP/VASP) — use QE isolated-atom E0s. PySCF fallback (150 KUKS SCFs, 3×3×3 k, 8-atom Fe cell, Mac) takes days — replace with the OMat24 leg. WBM-1000 "≈3 h/model" is optimistic for MPA-0 medium; budget 4–6 h, resumable.
- **production**: Days 1–3 all cluster-gated; mypy --strict, Executor protocol, Jinja templates, two agent backends and a second cluster round trip for live AL kill a 14-day scope. Keep the discipline, drop the abstractions.
- **mvp**: B20 frames in OMat24 unverified → "OMat24 structures of B20 chiral magnets" may be inflated; per-frame magnetic states uncontrolled (FM/NM mixtures for MnSi/FeGe); `mpr.materials.summary.search` needs the absent MP key; 6.8 GB tarball breaks the 5 GB budget; Si-vacancy umbrella is off-family; DFT backed only by PySCF Si.
- **all**: matbench-discovery git needs Python 3.14 (env is 3.11) → vendor metrics + protocol with checksummed references. LAMMPS is cluster-only; no parity run, no "LAMMPS" in the bullet.

## Grafts into the final spec (base: impact)

From production: `dft converge` stage with thresholds (E ≤1 meV/atom, F ≤5 meV/Å) and isolated-atom `E0s.json`; DFT noise floor (20 frames, tighter settings); bootstrap 95 % CIs per config_type; StructureMatcher dedupe + hash split; per-frame `--resume` QE array with failure counts; `report --audit` CI gate; agent `submit_dft` dry-run unless `--approve-cluster`; MnGe as never-trained composition; MODEL_CARD/DATA_CARD; Alexandria keyless fallback.

From mvp: Day-1 OMat24 leg (stream-filter, group split by parent id, LOCO) with spglib-198 B20 count reported; v0.1 vertical slice by Day 4; `--llm mock` replay so CI needs no API key; data curve 300/1k/3k with seeds stated; 103-compound PBE phonondb harness with PBEsol-vs-PBE labelling; experimental anchors (MnSi a = 4.558 Å, INS phonons; FeSi Raman); PLUMED only if `brew install plumed` works, pure-Python umbrella + pymbar primary.
