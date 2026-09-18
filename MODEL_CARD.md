# Model card — b20-mlip fine-tuned MACE models

Values on this card are `<!-- num:key -->` placeholders resolved from `reports/numbers.json`
by `b20mlip report build`; `pending` means no run has produced the number yet. Nothing on this
card is typed by hand.

## Model details

- **Family:** MACE (equivariant message-passing GNN), foundation model MACE-MPA-0 medium
  (MIT licence, github.com/ACEsuit/mace-foundations); MACE-MP-0 medium is the second zero-shot
  baseline (B0′).
- **Brackets shipped:** B1 naive fine-tune (QE energy scale, `--E0s configs/dft/E0s_qe.json`),
  B2 multihead replay (QE data in head `Default`, MPtrj replay in `pt_head`), B3 scratch MACE
  on identical frames; every checkpoint records its foundation sha256, E0 source, heads, seed,
  learning rate, epochs and split id (`CheckpointInfo`).
- **Licence:** weights MIT (inherited from MACE-MPA-0), code MIT (see `LICENSE`, `NOTICE`).
- **Contact:** Wenqin Chen, wenqinchenphy@gmail.com.

## Intended use

Collinear-ferromagnetic PBE potential-energy surfaces of the B20 hosts FeSi, CoSi, MnSi and FeGe
(MnGe as a stretch): forces, energies and stresses for MD, phonons, elastic constants and
vacancy-hop sampling near the training distribution (strain, shear, rattle, equation of state,
finite temperature up to the training temperature). Out of scope: spin degrees of freedom, DMI,
helimagnetism, non-B20 chemistries, public-ranking submissions.

## Training data

See `docs/DATA_CARD.md`. In short: in-house Quantum ESPRESSO frames (SSSP-efficiency
pseudopotentials, cited with md5 sums, never redistributed), OMat24 in-chemsys frames
(CC-BY-4.0, forces and stress only, energy weight zero), MPtrj replay frames (MIT).

| Quantity | Value |
| --- | --- |
| QE frames, round 0 | <!-- num:data.n_qe_frames -->pending<!-- /num --> |
| QE frames, round 1 (active learning) | <!-- num:data.n_qe_frames_r1 -->pending<!-- /num --> |
| OMat24 frames (bootstrap and tier T3) | <!-- num:data.n_omat24_frames -->pending<!-- /num --> |
| Frames rejected by the magnetic-branch filter | <!-- num:data.n_branch_rejected -->pending<!-- /num --> |

## Energy scales and heads

Three energy scales exist (MP/foundation, OMat24-VASP, QE/SSSP) and are never mixed in one loss.
B1 predicts on the QE scale; its WBM hull metrics are reported only if the per-element offset
map residual <!-- num:offsets.residual_meV_atom -->pending<!-- /num --> meV/atom passes the gate.
B2 serves B20 properties through `head="Default"` and WBM through `head="pt_head"`.

## Evaluation

| Metric | B0 | B1 | B2 |
| --- | --- | --- | --- |
| T0 force MAE (meV/Å) | <!-- num:eval.errors.T0.B0.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T0.B1.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T0.B2.mae_f -->pending<!-- /num --> |
| T2 FeGe force MAE (meV/Å) | <!-- num:eval.errors.T2.B0.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T2.B1.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T2.B2.mae_f -->pending<!-- /num --> |
| T3 OMat24 force MAE (meV/Å) | <!-- num:eval.errors.T3.B0.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T3.B1.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T3.B2.mae_f -->pending<!-- /num --> |
| T4a MPtrj force MAE (meV/Å) | <!-- num:eval.errors.T4a.B0.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T4a.B1.mae_f -->pending<!-- /num --> | <!-- num:eval.errors.T4a.B2.mae_f -->pending<!-- /num --> |
| paired ΔF1 vs B0 (WBM sample) | — | <!-- num:eval.discovery.B1.delta_f1 -->pending<!-- /num --> | <!-- num:eval.discovery.B2.delta_f1 -->pending<!-- /num --> |
| FeSi phonon ω-MAE vs own QE (meV) | <!-- num:eval.phonons.FeSi.B0.omega_mae_meV -->pending<!-- /num --> | <!-- num:eval.phonons.FeSi.B1.omega_mae_meV -->pending<!-- /num --> | <!-- num:eval.phonons.FeSi.B2.omega_mae_meV -->pending<!-- /num --> |

Full tables with reference, E0 source, head, sample size, seed and CI per cell: `README.md`.

## Limitations

- Trained on collinear PBE data; PBE overestimates the MnSi moment by roughly a factor of two.
- Differences below the cross-code force noise floor are not claims.
- The model is not validated outside the B20 family or above the training temperature.

## Citation

See `README.md` (Cite).
