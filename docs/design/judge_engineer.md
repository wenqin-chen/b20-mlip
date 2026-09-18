# P3 judge — engineer (2026-09-17)

## Scores (1–10)
| Design | value | honesty | prod-quality | 14-day | hiring | Total |
|---|---|---|---|---|---|---|
| impact | 8 | 9 | 6 | 3 | 8 | 34 |
| production | 7 | 9 | 9 | 5 | 8 | **38** |
| mvp | 5 | 8 | 7 | 7 | 6 | 33 |

Winner: production's architecture, mvp's schedule, impact's science.

## Runtime sanity
Basis: 48–59 ms/32 atoms is MP-0 *small*; MPA-0 medium (L=1) ≈ 2–3× slower on CPU; a training step ≈ 3× a forward. The facts' "1 h/100 epochs on 3k" line contradicts this — re-time one epoch on medium, Day 1.
- Naive fine-tune: impact "10 min/100 ep" (7.8k atoms/epoch) → 1.1 h small / 2.6 h medium, **6–16× low**. production ✓. mvp ✓.
- Replay: impact 10k MPtrj frames "35 min/20 ep" → 8 h small / 20 h medium, **15–35× low**. production 5k×30 ep "0.5–1.5 h" → 7/16 h, ×3 seeds, **5–10× low**. mvp 2k samples overnight ✓. Replay belongs on the Tillicum GPU.
- WBM-1000 (~20 atoms, 150–250 FIRE steps typical): 3–6 h/model medium, 11 h at the 500-step cap. impact "3 h" ×4 models on D12, **≥3× low**; production "2–6 h, resumable" ✓; mvp ✓.
- 64-atom MD ≈ 0.1 s/step: impact ✓, but its 512-atom ASE runs → ~25 h, **8×**; production 150 ps "2–3 h" → 4.6/10 h, **2–4×**; mvp ✓.
- Umbrella (63 atoms): impact 180 ps "6 h" ✓ small, 11–17 h medium; production 240 ps "3–4 h" → 7/16 h, **2–5×**; mvp ✓.
- phonopy 2×2×2: seconds, all ✓.
- QE: impact's 0.15 Å⁻¹ on P1 8-atom frames ≈ 500 k-points needs a GPU node for "5 min/frame"; production's scan → ≤0.25 Å⁻¹, ~3 min/frame ✓.
- impact's no-cluster fallback (PySCF, 150 FeSi frames) = 150 × hours: **infeasible**.

## Data fork
Verified (OPTIMADE): Alexandria PBE holds only **803** {Mn,Fe,Co}×{Si,Ge} binaries of 5.78M entries. OMat24 samples Alexandria, so val + 1M (~2% of train) hold a few hundred in-chemsys frames, not 2–3k; 3k means streaming ≥20 GB — not a primary training set.
Recommend **hybrid, QE-primary**: Day 1 pull the ~300 OMat24 in-chemsys frames → bring up fine-tune/eval/MD/agent end-to-end without the cluster; keep them as a VASP-labelled *held-out forces tier*. Login #1 (D2): convergence scan + 400–600 8-atom frames + FeSi/MnSi/CoSi phonopy displacement sets (~40 node-h); login #2 (D5–6) collect. QE frames → multihead ft head with own E0s. If the cluster slips: ship v0.1 on OMat24-300 + PySCF, "QE round pending".

## Technical checks
- `--pt_train_file mp`: verified in `multihead_tools.py`; branch enters for `pt_train_file in ["mp","omat","matpes_pbe","matpes_r2scan"]` regardless of foundation path; downloads `mace_mp_0b/mp_traj_combined.xyz`. Replay is MPtrj-only (MPA-0 also saw sAlex).
- Anthropic names correct: `client.beta.messages.tool_runner`, `@beta_tool`, `strict: true` (top-level on the tool), `claude-opus-5`. impact lacks a mock backend for CI.
- Naive fine-tuning on QE/OMat24 energies with refit E0s shifts the MPtrj scale and confounds WBM forgetting; evaluate WBM with `MACECalculator(..., head="pt_head")` (verified kwarg).
- MnSi/FeGe have several PBE moment branches; frames on different branches make a discontinuous PES. Log total magnetization per frame, reject off-branch. No design does.
- PySCF PBC: Si 16-atom ✓; FeSi 8-atom with k-mesh + smearing → GDF exceeds 16 GB, hours/SCF; FM KUKS MnSi not viable. Baseline only, never labels.
- `matbench-discovery` git needs Python ≥3.14 (Mac: 3.11): impact's dep unresolvable; vendor metrics + MP elemental refs.
- MP API key absent: mvp's Day-1 MPRester call fails; use OPTIMADE. mvp's `--E0s foundation` on OMat24 → `estimated`.
- QE on Tillicum unverified: `module avail`, else micromamba conda-forge `qe`. Build LAMMPS-MACE on login #1, not Day 10.
- impact's balanced 250×4 WBM split changes the base rate; sample proportionally.

## Fatal flaws
impact: 6–35×-low training estimates, three cluster logins on the critical path, infeasible fallback. mvp: 3k-frame data plan contradicted by the 803 count; DFT story too thin. production: none fatal; cut scope (replay seeds on GPU only; agent `submit_dft` dry-run + one manual AL round).

## Grafts
impact: softening index + imaginary-mode count vs own QE; brackets B0′ (MP-0), B3 (scratch); H2 900 K holdout; H3 FeGe/CoSi never trained; LAMMPS parity gate (20 frames, |ΔF|<1e-3); Receipts CI; agent eval with injected failures.
mvp: vertical slice tagged v0.1 by Day 4; cluster-optional ordering; OMat24 frames as bootstrap + held-out tier; mock-LLM CI; MODEL_CARD.
production (base): executors, manifests, `--resume`, convergence scan + DFT noise floor extended cross-code (20 frames labelled by QE and OMat24-VASP); `report --audit`; MnGe holdout.
New: per-config `config_energy_weight` (verified in `data/utils.py`) for forces-only mixing; magnetic-branch filter.
