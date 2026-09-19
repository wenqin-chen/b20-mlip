# templates/qe

Quantum ESPRESSO input templates (Jinja2), owned by the dft tier (CONTRACTS.md row 6):

- `pw.in.j2` — `pw.x` SCF input rendered by `b20mlip.dft.qe.render_pw_input(frame, cfg)`:
  PBE, SSSP-efficiency 1.3.0 pseudopotentials (`cfg.dft.pseudos` file names in
  `cfg.dft.pseudo_dir`; md5s from `cfg.dft.pseudo_md5s` are recorded in every manifest; the UPF
  files are never redistributed), `ecutwfc`/`ecutrho` from `cfg.dft`, Marzari–Vanderbilt smearing
  (`degauss` in Ry), an unshifted k-mesh with `k_i = ceil(|b_i| / k_spacing_inv_A)` (|b_i| includes
  2π, the ASE/VASP `KSPACING` convention), `nspin` per compound and `starting_magnetization` per
  species when `nspin = 2`, `tprnfor` and `tstress` on, `outdir = './tmp'` (deleted by the unit
  script after a converged run unless `B20_KEEP_TMP=1`).

`tests/dft/test_dft_qe.py` compares the render of the golden 8-atom MnSi frame
(`tests/fixtures/golden/mnsi_8atom.frame.json`) against the snapshot `tests/fixtures/golden/pw.in`.
Phonon displacement sets and isolated-atom (E0) units reuse the same template through
`render_pw_input(..., overrides=...)` (`kpoints="gamma"`, `nspin`, `starting_magnetization`, cutoffs).
