# Test fixtures

## `wbm_mini.csv.gz`

Ten rows copied verbatim (all 18 columns) from the Matbench-Discovery WBM summary
`data/raw/references/wbm-summary.csv.gz` (Wang, Botti, Marques 2021, as distributed by
Riebesell et al., CC-BY-4.0; 256,963 rows), written with pandas on 2026-09-18.
sha256 `de50c5455b209301beb110e6cdc83ea423709f31dec183ec5f344332dfd77f70`, 909 bytes.

| material_id | formula | e_above_hull_mp2020_corrected_ppd_mp (eV/atom) | why |
|---|---|---|---|
| wbm-1-16845 | Co1 Fe2 Ge1 | -0.007782 | in-family stable (one of the four in WBM) |
| wbm-1-29035 | Fe1 Ge1 Mn2 | -0.009194 | in-family stable |
| wbm-1-55036 | Fe1 Mn2 Si1 | -0.012518 | in-family stable |
| wbm-3-42176 | Ge3 Mn3 | -0.009444 | in-family stable |
| wbm-2-17064 | Fe3 Ge1 | 0.011279 | in-family unstable |
| wbm-1-29652 | Ge4 Mn4 | 0.042386 | in-family unstable |
| wbm-2-27173 | Fe1 Mn2 | 0.080757 | in-family unstable |
| wbm-4-18301 | Fe1 Ge1 Ir1 Li1 | 0.025428 | first id of the seed-0 sample |
| wbm-1-1 | Ac6 U2 | 0.544310 | first row of the file |
| wbm-5-6641 | F6 Hg1 Pa1 | -2.728363 | large MP2020 correction (-0.3465 eV/atom) |

`tests/evaluate/test_mbd_vendored.py` scores a fixed synthetic prediction vector against
these truths and asserts the hand-computed `stable_metrics` values (TP 4, FN 1, FP 1, TN 4,
F1 0.8, MAE 0.024392 eV/atom, RMSE 0.027954 eV/atom).

## `golden/`

Real Quantum ESPRESSO input/output of the 8-atom MnSi cell (see `golden/README.md`).
