# templates/qe

Quantum ESPRESSO input templates (Jinja2), owned by the dft tier (CONTRACTS.md row 6):

- `pw.in.j2` — `pw.x` SCF input rendered by `b20mlip.dft.qe.render_pw_input(frame, cfg)`:
  PBE, SSSP-efficiency pseudopotentials (`cfg.dft.pseudo_dir`, md5s recorded in the manifest,
  files never redistributed), `ecutwfc`/`ecutrho` from `cfg.dft`, Marzari–Vanderbilt smearing
  (`degauss` in Ry), k-spacing <= 0.25 1/Å, `nspin` and `starting_magnetization` per compound,
  `tprnfor` and `tstress` on.
- `ph_disp.in.j2` — (optional) inputs for phonopy displacement sets.

Snapshot tests compare renders against `tests/golden/pw.in`. Nothing exists here yet.
