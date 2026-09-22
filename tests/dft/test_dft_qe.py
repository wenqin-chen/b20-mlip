"""qe.py: render snapshot, k-mesh rule, parser exactness on the golden files, planning, collect."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import units

from b20mlip.config import DFTConfig, Settings
from b20mlip.dft import qe
from b20mlip.executors import JobSpec
from b20mlip.io import read_frames, write_frames
from b20mlip.models import DFTFrame, Frame

GOLDEN_ENERGY_RY = -890.78158231  # real pw.x 7.5 run, Tillicum job 303018, 2026-09-18
GOLDEN_STRESS_RY_BOHR3 = -0.00038447


# --- rendering ---------------------------------------------------------------------------------


def test_render_snapshot(golden_frame: Frame, golden_cfg: Settings, golden_dir: Path) -> None:
    text = qe.render_pw_input(golden_frame, golden_cfg)
    assert text == (golden_dir / "pw.in").read_text()
    assert (
        "calculation = 'scf'" in text and "tprnfor = .true." in text and "tstress = .true." in text
    )
    assert "ecutwfc = 90.0" in text and "ecutrho = 1080.0" in text
    assert "smearing = 'mv'" in text and "degauss = 0.01" in text and "nspin = 2" in text
    assert "starting_magnetization(1) = 0.50" in text and "starting_magnetization(2) = 0.00" in text
    assert "  Mn 54.9380 mn_pbe_v1.5.uspp.F.UPF" in text
    assert "  Si 28.0850 Si.pbe-n-rrkjus_psl.1.0.0.UPF" in text
    assert "K_POINTS automatic\n  6 6 6 0 0 0\n" in text
    assert f"prefix = '{golden_frame.frame_id}'" in text and "outdir = './tmp'" in text
    assert text.count("\n") == len(text.splitlines())  # trailing newline, no blank lines from jinja


def test_render_overrides(golden_frame: Frame, golden_cfg: Settings, b20_frame) -> None:
    text = qe.render_pw_input(
        golden_frame,
        golden_cfg,
        overrides={"kpoints": "gamma", "nspin": 1, "ecut_ry": 40, "ecut_rho": 480, "prefix": "x"},
    )
    assert "K_POINTS gamma" in text and "K_POINTS automatic" not in text
    assert "nspin = 1" in text and "starting_magnetization" not in text
    assert "ecutwfc = 40.0" in text and "ecutrho = 480.0" in text and "prefix = 'x'" in text
    fesi = b20_frame("FeSi")
    text = qe.render_pw_input(fesi, golden_cfg, unit_id="u1")
    assert "nspin = 1" in text and "prefix = 'u1'" in text
    assert "  Fe 55.8450 Fe.pbe-spn-kjpaw_psl.0.2.1.UPF" in text
    fege = qe.render_pw_input(b20_frame("FeGe"), golden_cfg)
    assert "nspin = 2" in fege and "  Ge 72.6300 ge_pbe_v1.4.uspp.F.UPF" in fege
    with pytest.raises(ValueError, match="unknown pw.in overrides"):
        qe.pw_parameters(golden_frame, golden_cfg, overrides={"ecut": 1})
    with pytest.raises(ValueError, match="kpoints"):
        qe.pw_parameters(golden_frame, golden_cfg, overrides={"kpoints": "grid"})
    with pytest.raises(ValueError, match="nspin"):
        qe.pw_parameters(golden_frame, golden_cfg, overrides={"nspin": 4})
    no_pseudo = golden_cfg.model_copy(
        update={"dft": golden_cfg.dft.model_copy(update={"pseudos": {"Si": "Si.UPF"}})}
    )
    with pytest.raises(ValueError, match="no pseudopotential file configured for \\['Mn'\\]"):
        qe.pw_parameters(golden_frame, no_pseudo)


def test_kmesh_rule_and_helpers(golden_frame: Frame, golden_cfg: Settings, tmp_path: Path) -> None:
    cubic = np.diag([4.56, 4.56, 4.56])
    assert qe.kmesh(cubic, 0.25) == (6, 6, 6) and qe.kmesh(cubic, 0.35) == (4, 4, 4)
    assert qe.kmesh(cubic, 0.3) == (5, 5, 5) and qe.kmesh(cubic, 0.2) == (7, 7, 7)
    # |b| = 2π/a = 1.37789: exactly 1.37789/0.229648 = 6.0000 must give 6, not 7 (tolerance)
    assert qe.kmesh(cubic, 2 * np.pi / 4.56 / 6) == (6, 6, 6)
    assert qe.kmesh(np.diag([4.56, 9.12, 13.68]), 0.25) == (6, 3, 2)
    assert qe.kmesh(np.diag([100.0, 100.0, 100.0]), 0.25) == (1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        qe.kmesh(cubic, 0.0)
    assert qe.species_order([25, 25, 14, 14, 25]) == ["Mn", "Si"]
    assert qe.nspin_for("MnSi", golden_cfg) == 2 and qe.nspin_for("FeSi", golden_cfg) == 1
    assert qe.nspin_for("Fe3Si", golden_cfg, [26, 26, 26, 14]) == 2
    assert qe.nspin_for("SiGe", golden_cfg, [14, 32]) == 1
    assert qe.resolve_pseudo_dir("/abs/p") == "/abs/p"
    assert (
        qe.resolve_pseudo_dir("rel/p").endswith("/rel/p")
        and Path(qe.resolve_pseudo_dir("rel/p")).is_absolute()
    )
    assert qe.sssp_cutoffs(["Mn", "Si"], golden_cfg) == (65.0, 780.0, 12.0)
    assert qe.sssp_cutoffs(["Fe", "Ge"], golden_cfg) == (90.0, 1080.0, 12.0)
    assert qe.sssp_cutoffs(["Co", "Si"], golden_cfg) == (45.0, 360.0, 8.0)
    fallback = (golden_cfg.dft.ecut_ry, golden_cfg.dft.ecut_rho, 12.0)
    assert qe.sssp_cutoffs(["Xe"], golden_cfg) == fallback
    absent = golden_cfg.model_copy(
        update={"dft": golden_cfg.dft.model_copy(update={"sssp_json": str(tmp_path / "no.json")})}
    )
    assert qe.sssp_cutoffs(["Mn", "Si"], absent) == fallback
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps({"Mn": {"cutoff_wfc": 1, "cutoff_rho": 8, "filename": "x", "md5": "y"}})
    )
    assert qe.load_sssp(raw)["Mn"]["cutoff_rho"] == 8


# --- parsing -----------------------------------------------------------------------------------


def test_parse_golden_exact(golden_frame: Frame, golden_cfg: Settings, golden_dir: Path) -> None:
    """Exact numbers of the REAL 8-atom MnSi pw.x 7.5 output (Tillicum, 2 MPI ranks, 2026-09-18)."""
    frame = qe.parse_pw_output(golden_dir / "pw.out", golden_frame, dft=golden_cfg.dft, unit_id="g")
    assert isinstance(frame, DFTFrame) and frame.converged and frame.scf_steps == 24
    assert frame.energy == pytest.approx(GOLDEN_ENERGY_RY * units.Ry, rel=1e-12)
    assert frame.energy == pytest.approx(-12119.70075, abs=2e-5)
    f = units.Ry / units.Bohr
    assert frame.forces is not None and len(frame.forces) == 8
    assert frame.forces[0] == pytest.approx([-0.00586276 * f] * 3, rel=1e-12)
    assert frame.forces[4] == pytest.approx([-0.00221139 * f] * 3, rel=1e-12)
    assert frame.forces[7] == pytest.approx([-0.00221139 * f, 0.00221139 * f, 0.00221139 * f])
    assert frame.forces[0][0] == pytest.approx(-0.150738, abs=1e-5)  # eV/Å
    assert np.allclose(np.sum(frame.forces, axis=0), 0.0, atol=1e-6)
    s = units.Ry / units.Bohr**3
    assert frame.stress == pytest.approx([-GOLDEN_STRESS_RY_BOHR3 * s] * 3 + [0.0] * 3, rel=1e-12)
    assert frame.stress[0] > 0  # QE P = -56.56 kbar (tensile) -> ASE sigma_xx = +0.0353 eV/Å^3
    assert frame.stress[0] == pytest.approx(0.0353, abs=1e-4)
    assert frame.info["pressure_kbar"] == -56.56
    assert frame.stress[0] == pytest.approx(56.56 * qe.KBAR_TO_EV_A3, rel=2e-3)
    assert frame.fermi_eV == 15.5131
    assert frame.total_magnetization == 4.05 and frame.abs_magnetization == 4.77
    assert frame.magmoms == [1.0381] * 4 + [-0.0456] * 4
    assert frame.wall_seconds == pytest.approx(5 * 60 + 34.96) and frame.info["pw_version"] == "7.5"
    assert frame.info["total_energy_ry"] == GOLDEN_ENERGY_RY
    assert frame.code == "qe" and frame.functional == "PBE" and frame.unit_id == "g"
    assert frame.label_source == "qe" and frame.energy_scale == "qe"
    assert frame.ecut_ry == 90.0 and frame.k_spacing == 0.25 and frame.nspin == 2
    assert frame.smearing == "mv" and frame.degauss_ry == 0.01
    assert frame.pseudo_md5s == {
        "Mn": "82ef2b46521d7a7d9e736dc3972e4928",
        "Si": "0b0bb1205258b0d07b9f9672cf965d36",
    }
    assert frame.info["kmesh"] == [6, 6, 6] and frame.info["ecut_rho"] == 1080.0
    assert frame.branch_ok is None and "parse_warnings" not in frame.info
    assert frame.frame_id == golden_frame.frame_id and frame.compound == "MnSi"


def test_parse_unconverged_missing_and_truncated(
    golden_frame: Frame, golden_cfg: Settings, golden_dir: Path, tmp_path: Path
) -> None:
    frame = qe.parse_pw_output(golden_dir / "pw_unconverged.out", golden_frame, dft=golden_cfg.dft)
    assert not frame.converged and frame.scf_steps == 100
    assert frame.energy is None and frame.forces is None and frame.stress is None
    assert frame.total_magnetization == 4.05 and frame.abs_magnetization == 4.52
    assert frame.magmoms is None and frame.fermi_eV is None
    assert frame.wall_seconds == pytest.approx(11 * 60 + 4.35)
    assert frame.unit_id == "golden"  # parent directory name

    missing = qe.parse_pw_output(tmp_path / "nope" / "pw.out", golden_frame, dft=golden_cfg.dft)
    assert not missing.converged and missing.scf_steps == 0 and missing.energy is None
    assert missing.info["parse_warnings"] == "pw.out missing" and missing.wall_seconds == 0.0

    text = (golden_dir / "pw.out").read_text()
    cut = text[: text.index("iteration #  2")]
    trunc = tmp_path / "pw.out"
    trunc.write_text(cut)
    t = qe.parse_pw_output(trunc, golden_frame, dft=golden_cfg.dft)
    assert not t.converged and t.scf_steps == 1 and t.energy is None
    assert "truncated" in t.info["parse_warnings"]
    assert qe.parse_pw_text("")["converged"] is False


def test_parse_helpers_and_formats() -> None:
    block = (
        "          total   stress  (Ry/bohr**3)                   (kbar)     P=        1.23\n"
        "   0.00001000   0.00000600   0.00000500            1.47        0.88        0.74\n"
        "   0.00000600   0.00002000   0.00000400            0.88        2.94        0.59\n"
        "   0.00000500   0.00000400   0.00003000            0.74        0.59        4.41\n"
    )
    rows, pressure = qe.parse_stress_ry_bohr3(block)
    assert pressure == 1.23 and rows[2] == [0.000005, 0.000004, 0.00003]
    v = qe.voigt6_ase_from_qe(rows)
    s = units.Ry / units.Bohr**3
    # Voigt order [xx, yy, zz, yz, xz, xy] with the sign flipped
    assert v == pytest.approx([-1e-5 * s, -2e-5 * s, -3e-5 * s, -4e-6 * s, -5e-6 * s, -6e-6 * s])
    assert qe.parse_stress_ry_bohr3("nothing") is None
    assert qe.parse_stress_ry_bohr3(block.splitlines()[0] + "\n garbage\n") is None

    old = (
        "     Magnetic moment per site:\n"
        "     atom:    1    charge:   14.2891    magn:    2.2011    constr:    0.0000\n"
        "     atom:    2    charge:   14.2891    magn:   -2.2011    constr:    0.0000\n\n"
    )
    assert qe.parse_site_moments(old) == [2.2011, -2.2011]
    assert qe.parse_site_moments("no block") is None
    assert qe.parse_site_moments("     Magnetic moment per site\n  nothing here\n") is None
    two = (
        old
        + "     Magnetic moment per site  (integrated on atomic sphere of radius R)\n"
        + "     atom   1 (R=0.357)  charge= 12.0000  magn=  0.5000\n"
    )
    assert qe.parse_site_moments(two) == [0.5]  # the LAST block wins

    assert qe.parse_duration("16.12s") == 16.12
    assert qe.parse_duration("2m30.12s") == pytest.approx(150.12)
    assert qe.parse_duration("1h 5m") == 3900.0 and qe.parse_duration("  1h 2m 3.5s") == 3723.5
    assert qe.parse_duration("bogus") is None and qe.parse_duration("") is None

    forces = (
        "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
        "     atom    2 type  1   force =     0.10000000    0.20000000    0.30000000\n"
        "     atom    1 type  1   force =    -0.10000000   -0.20000000   -0.30000000\n"
        "     The non-local contrib.  to forces\n"
        "     atom    1 type  1   force =     9.00000000    9.00000000    9.00000000\n"
    )
    assert qe.parse_forces_ry_bohr(forces) == [[-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]]
    assert qe.parse_forces_ry_bohr("no forces") is None
    assert (
        qe.parse_forces_ry_bohr("Forces acting on atoms (cartesian axes, Ry/au):\n\n bad\n") is None
    )

    assert qe.branch_ok([1.0, 1.1, -0.04, -0.02], [25, 25, 14, 14], 1.0, 0.3) is True
    assert qe.branch_ok([0.5, 0.5, 0.0, 0.0], [25, 25, 14, 14], 1.0, 0.3) is False
    assert (
        qe.branch_ok(None, [25], 1.0, 0.3) is None and qe.branch_ok([1.0], [14], 1.0, 0.3) is None
    )
    assert qe.branch_ok([1.0], [25], None, 0.3) is None


def test_parse_uses_unit_json_and_branch_reference(
    golden_frame: Frame, golden_cfg: Settings, golden_dir: Path, tmp_path: Path
) -> None:
    cfg = golden_cfg.model_copy(
        update={"dft": golden_cfg.dft.model_copy(update={"m_ref_muB": {"MnSi": 1.0}})}
    )
    [uid] = qe.plan_units([golden_frame], tmp_path, cfg)
    unit = tmp_path / uid
    shutil.copy(golden_dir / "pw.out", unit / "pw.out")
    frame = qe.parse_pw_output(unit / "pw.out", golden_frame)  # parameters from unit.json
    assert frame.branch_ok is True and frame.unit_id == uid and frame.ecut_ry == 90.0
    assert frame.pseudo_md5s["Mn"] == "82ef2b46521d7a7d9e736dc3972e4928"
    strict = qe.parse_pw_output(
        unit / "pw.out", golden_frame, dft={**qe.read_unit(unit)["parameters"], "m_ref_muB": 0.4}
    )
    assert strict.branch_ok is False
    (unit / "unit.json").write_text("{not json")
    fallback = qe.parse_pw_output(unit / "pw.out", golden_frame)
    assert fallback.ecut_ry == DFTConfig().ecut_ry and fallback.converged


# --- planning and collection ---------------------------------------------------------------------


def test_plan_units_and_collect(
    b20_frame, golden_cfg: Settings, golden_dir: Path, tmp_path: Path
) -> None:
    frames = [
        b20_frame("MnSi"),
        b20_frame("MnSi", "strain", scale=1.02),
        b20_frame("MnSi", "rattle", rattle=0.05),
    ]
    root = tmp_path / "r0"
    ids = qe.plan_units(frames, root, golden_cfg)
    assert ids == [f.frame_id for f in frames] and qe.list_units(root) == ids
    unit = qe.unit_dir(frames[0], root)
    assert unit == root / frames[0].frame_id and (unit / "pw.in").is_file()
    record = qe.read_unit(unit)
    assert record["schema"] == "b20mlip.qe_unit.v1" and record["unit_id"] == ids[0]
    assert Frame.model_validate(record["frame"]) == frames[0]
    assert record["parameters"]["kmesh"] == [6, 6, 6] and record["parameters"]["nspin"] == 2
    assert record["parameters"]["pseudo_md5s"] == {
        "Mn": "82ef2b46521d7a7d9e736dc3972e4928",
        "Si": "0b0bb1205258b0d07b9f9672cf965d36",
    }
    assert record["dft"]["ecut_ry"] == 90.0
    assert (unit / "pw.in").read_text() == qe.render_pw_input(frames[0], golden_cfg)
    assert json.loads((root / "units.json").read_text())["units"] == ids
    assert qe.unit_status(unit) == "pending" and qe.pending_units(root) == ids

    # outputs: unit 0 converged, unit 1 unconverged output, unit 2 failed without output
    shutil.copy(golden_dir / "pw.out", unit / "pw.out")
    (unit / ".done").write_text("{}")
    shutil.copy(golden_dir / "pw_unconverged.out", root / ids[1] / "pw.out")
    (root / ids[1] / ".failed").write_text("{}")
    (root / ids[2] / ".failed").write_text("{}")
    extra = qe.plan_units([b20_frame("FeSi")], root, golden_cfg)  # a 4th, never run: missing
    assert qe.list_units(root) == ids + extra
    parsed, counts = qe.collect(root)
    assert counts == {"planned": 4, "done": 1, "failed": 1, "unconverged": 1, "missing": 1}
    assert (
        [f.unit_id for f in parsed] == ids[:2] and parsed[0].converged and not parsed[1].converged
    )
    assert parsed[0].energy == pytest.approx(GOLDEN_ENERGY_RY * units.Ry)
    assert qe.pending_units(root) == ids[1:] + extra
    assert qe.unit_status(root / ids[1]) == "failed"

    # idempotent re-plan keeps outputs; changed parameters clear them
    qe.plan_units(frames, root, golden_cfg)
    assert (unit / "pw.out").is_file() and (unit / ".done").is_file()
    denser = golden_cfg.model_copy(
        update={"dft": golden_cfg.dft.model_copy(update={"k_spacing_inv_A": 0.2})}
    )
    qe.plan_units(frames[:1], root, denser)
    assert not (unit / "pw.out").exists() and not (unit / ".done").exists()
    assert qe.read_unit(unit)["parameters"]["kmesh"] == [7, 7, 7]
    assert "7 7 7 0 0 0" in (unit / "pw.in").read_text()

    with pytest.raises(ValueError, match="duplicate unit ids"):
        qe.plan_units([frames[0], frames[0]], tmp_path / "dup", golden_cfg)
    with pytest.raises(ValueError, match="unit ids for"):
        qe.plan_units(frames, tmp_path / "x", golden_cfg, unit_ids=["a"])
    with pytest.raises(ValueError, match="one overrides mapping per frame"):
        qe.plan_units(frames, tmp_path / "x", golden_cfg, overrides=[{}])
    with pytest.raises(ValueError, match="marker safe"):
        qe.plan_units(frames[:1], tmp_path / "x", golden_cfg, unit_ids=["a/b"])
    assert qe.collect(tmp_path / "absent") == (
        [],
        {"planned": 0, "done": 0, "failed": 0, "unconverged": 0, "missing": 0},
    )
    assert qe.list_units(tmp_path / "absent") == []
    (root / "units.json").write_text("garbage")
    assert sorted(qe.list_units(root)) == sorted(ids + extra)  # falls back to a directory scan
    ids_default_cfg = qe.plan_units(
        [b20_frame("CoSi")], tmp_path / "defaults"
    )  # cfg=None -> load_config
    assert "nspin = 1" in (tmp_path / "defaults" / ids_default_cfg[0] / "pw.in").read_text()


def test_to_plain_frame_round_trip(
    golden_frame: Frame, golden_cfg: Settings, golden_dir: Path, tmp_path: Path
) -> None:
    df = qe.parse_pw_output(golden_dir / "pw.out", golden_frame, dft=golden_cfg.dft, unit_id="u")
    plain = qe.to_plain_frame(df)
    assert type(plain) is Frame and plain.energy == df.energy and plain.forces == df.forces
    assert plain.info["dft_code"] == "qe" and plain.info["dft_converged"] is True
    assert plain.info["dft_unit_id"] == "u" and plain.info["dft_ecut_ry"] == 90.0
    assert plain.info["dft_pseudo_md5s"]["Si"] == "0b0bb1205258b0d07b9f9672cf965d36"
    path = tmp_path / "qe.extxyz"
    write_frames([plain], path)
    [back] = read_frames(path)
    assert back.energy == pytest.approx(df.energy) and back.label_source == "qe"
    assert back.energy_scale == "qe" and back.info["dft_scf_steps"] == 24
    assert np.allclose(back.stress, df.stress) and back.magmoms == pytest.approx(df.magmoms)
    assert back.total_magnetization == 4.05 and back.info["dft_fermi_eV"] == 15.5131
    assert json.loads(back.info["dft_pseudo_md5s"])["Mn"] == "82ef2b46521d7a7d9e736dc3972e4928"


def test_job_spec_and_script(golden_cfg: Settings, tmp_path: Path) -> None:
    cfg = golden_cfg.model_copy(
        update={
            "cluster": golden_cfg.cluster.model_copy(update={"qe_cmd": "mpirun -np 8 pw.x -nk 2"})
        }
    )
    spec = qe.job_spec(tmp_path / "r0", ["a", "b"], cfg, resources={"time": "00:20:00"})
    assert isinstance(spec, JobSpec) and spec.units == ["a", "b"]
    assert spec.name.startswith("qe-r0-") and len(spec.name) == len("qe-r0-") + 8
    assert spec.name == qe.job_name(tmp_path / "r0") != qe.job_name(tmp_path / "r1")
    assert spec.env == {
        "B20_UNITS_ROOT": str((tmp_path / "r0").resolve()),
        "QE_CMD": "mpirun -np 8 pw.x -nk 2",
    }
    assert spec.resources == {"template": "qe_array", "time": "00:20:00"}
    assert spec.script == qe.UNIT_SCRIPT
    for needle in (
        'cd "$unit_dir"',
        "$QE_CMD -in pw.in > pw.out",
        "convergence has been achieved",
        "> .done",
        "> .failed",
        "exit 1",
        "rm -rf tmp",
    ):
        assert needle in spec.script
    assert "srun" not in spec.script
    remote = qe.job_spec(
        tmp_path / "r0",
        ["a"],
        golden_cfg,
        name="n",
        template="qe_phonons",
        units_root="/gscratch/x/r0",
        env={"B20_KEEP_TMP": "1"},
    )
    assert remote.name == "n" and remote.resources["template"] == "qe_phonons"
    assert remote.env == {
        "B20_UNITS_ROOT": "/gscratch/x/r0",
        "B20_KEEP_TMP": "1",
    }  # no qe_cmd configured


def test_job_spec_merges_cluster_resource_defaults(golden_cfg: Settings, tmp_path: Path) -> None:
    """Site defaults from cluster.resources[template] apply; explicit resources win."""
    cfg = golden_cfg.model_copy(
        update={
            "cluster": golden_cfg.cluster.model_copy(
                update={
                    "resources": {
                        "qe_array": {"ntasks": 2, "gpus": 1, "mem": "30G", "max_parallel": 24}
                    }
                }
            )
        }
    )
    spec = qe.job_spec(tmp_path, ["u1"], cfg, template="qe_array", resources={"mem": "60G"})
    assert spec.resources == {
        "template": "qe_array",
        "ntasks": 2,
        "gpus": 1,
        "mem": "60G",
        "max_parallel": 24,
    }
    other = qe.job_spec(tmp_path, ["u1"], cfg, template="qe_phonons")
    assert other.resources == {"template": "qe_phonons"}


def test_parser_flags_a_diverged_scf(
    golden_frame: Frame, golden_cfg: Settings, tmp_path: Path
) -> None:
    """A run that 'converged' after thousands of electrons of negative density is not converged."""
    text = (Path("tests/fixtures/golden/pw.out")).read_text(encoding="utf-8")
    bad = text.replace(
        "     iteration #  2",
        "     negative rho (up, down):  1.330E+03 0.000E+00\n     iteration #  2",
        1,
    )
    path = tmp_path / "pw.out"
    path.write_text(bad, encoding="utf-8")
    frame = qe.parse_pw_output(path, golden_frame, dft=golden_cfg.dft, unit_id="d")
    assert frame.converged is False and frame.info["diverged"] is True
    assert frame.info["max_negative_rho"] == pytest.approx(1330.0)
    ok = qe.parse_pw_output(Path("tests/fixtures/golden/pw.out"), golden_frame, dft=golden_cfg.dft)
    assert ok.converged and ok.info["max_negative_rho"] < qe.MAX_NEGATIVE_RHO


def test_parser_rejects_a_run_killed_before_its_forces(
    golden_frame: Frame, golden_cfg: Settings, tmp_path: Path
) -> None:
    """Tillicum 2026-09-21: a 64-atom unit converged its SCF and was killed by the walltime
    before printing forces; that is not a usable label (converged=False, flagged)."""
    text = Path("tests/fixtures/golden/pw.out").read_text(encoding="utf-8")
    cut = text[: text.index("Forces acting on atoms")]
    path = tmp_path / "pw.out"
    path.write_text(cut, encoding="utf-8")
    frame = qe.parse_pw_output(path, golden_frame, dft=golden_cfg.dft)
    assert frame.converged is False and frame.forces is None
    assert frame.info["killed_before_forces"] is True
    assert frame.energy is not None  # the SCF energy is still reported, just not used
