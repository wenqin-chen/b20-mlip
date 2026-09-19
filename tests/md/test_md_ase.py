"""ase_md: ensembles, drift, a(T), VDOS, RDF, thermal expansion, the md.ase stage (tiny MACE)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.lj import LennardJones
from ase.io import read as ase_read

from b20mlip.config import Settings
from b20mlip.md import ase_md, common
from b20mlip.models import MDResult
from b20mlip.provenance import RunContext, read_manifest, run_stage, sha256_file


def _ctx(cfg: Settings, seed: int | None = 0) -> RunContext:
    return RunContext("md.ase", cfg, seed=seed)


def _result(**overrides: object) -> MDResult:
    base = dict(
        engine="ase", model_sha256="x", head="Default", compound="MnSi", natoms=64,
        ensemble="npt", temperature_K=300.0, pressure_GPa=0.0, timestep_fs=2.0, steps=10,
        drift_meV_atom_ps=None, a_mean_A=4.56, a_std_A=0.01, alpha_per_K=None,
        traj_sha256="t", vdos_path=None, rdf_path=None, run_id="r",
    )  # fmt: skip
    base.update(overrides)
    return MDResult.model_validate(base)


# --- helpers ------------------------------------------------------------------------------------


def test_supercell_reps_and_lattice_parameter(argon_cell: Atoms, b20_atoms) -> None:
    assert common.supercell_reps(8, 64) == 2 and common.supercell_reps(8, 512) == 4
    assert common.supercell_reps(8, 8) == 1 and common.supercell_reps(8, 9) == 2
    assert common.supercell_reps(4, 32) == 2 and common.supercell_reps(8, 1) == 1
    cell, reps = common.make_supercell(b20_atoms("MnSi"), 64)
    assert len(cell) == 64 and reps == 2
    assert common.lattice_parameter(cell.get_volume(), reps) == pytest.approx(4.56)
    same, reps1 = common.make_supercell(argon_cell, None)
    assert len(same) == 4 and reps1 == 1
    assert common.elements_of_atoms(cell) == ["Si", "Mn"]  # by atomic number
    assert common.elements_of_compound("MnSi") == ["Mn", "Si"]
    with pytest.raises(ValueError):
        common.elements_of_compound("Xx")
    assert common.fmt_T(300.0) == "300" and common.fmt_T(300.5) == "300.5"
    assert common.sanitize_key_segment("mace-mpa-0 medium/x") == "mace-mpa-0-medium-x"
    assert common.sanitize_key_segment("///") == "model"
    with pytest.raises(ValueError):
        common.supercell_reps(0, 8)


def test_n_steps_and_ensemble_checks() -> None:
    assert ase_md.n_steps(0.04, 2.0) == 20 and ase_md.n_steps(40.0, 2.0) == 20000
    assert ase_md.n_steps(0.0001, 2.0) == 1
    with pytest.raises(ValueError):
        ase_md.n_steps(0.0, 2.0)
    with pytest.raises(ValueError, match="ensemble"):
        ase_md.check_ensemble("nph")
    assert ase_md.thermostat_label("npt", Settings.model_validate({})).startswith("Nose-Hoover")


def test_linear_fit_and_block_ci() -> None:
    slope, intercept, se = common.linear_fit([0, 1, 2, 3], [1, 3, 5, 7])
    assert (
        slope == pytest.approx(2.0) and intercept == pytest.approx(1.0) and se == pytest.approx(0.0)
    )
    assert common.linear_fit([0, 1], [0, 2])[2] is None
    with pytest.raises(ValueError):
        common.linear_fit([1, 1], [0, 1])
    assert common.block_ci95([1.0, 2.0, 3.0]) is None
    lo, hi = common.block_ci95(np.linspace(0, 1, 50))  # type: ignore[misc]
    assert lo < 0.5 < hi


# --- MD runs -------------------------------------------------------------------------------------


def test_nve_drift_is_finite_and_tiny_with_lj(
    md_settings: Settings, argon_cell: Atoms, lj_calc: LennardJones
) -> None:
    cfg = md_settings.model_copy(
        update={"md": md_settings.md.model_copy(update={"timestep_fs": 1.0})}
    )
    ctx = _ctx(cfg)
    stats: dict[str, object] = {}
    res = ase_md.run(argon_cell, lj_calc, "nve", 100.0, 0.04, 1.0, cfg, ctx, natoms=32, stats=stats)
    assert res.engine == "ase" and res.ensemble == "nve" and res.natoms == 32 and res.steps == 40
    assert res.drift_meV_atom_ps is not None and np.isfinite(res.drift_meV_atom_ps)
    assert abs(res.drift_meV_atom_ps) < 1.0  # meV/atom/ps: LJ argon at 1 fs conserves energy
    assert res.a_mean_A is None and res.a_std_A is None and res.pressure_GPa is None
    assert res.alpha_per_K is None and res.run_id == ctx.run_id
    assert stats["n_production"] == 21 and stats["production_window"].startswith("production")
    assert stats["drift_ci95"] is not None and stats["reps"] == 2
    thermo = ctx.out_dir / "thermo_nve_100K.csv"
    with open(thermo, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(ase_md.THERMO_COLUMNS) and len(rows) == 21
    assert float(rows[0]["T_K"]) == pytest.approx(100.0, abs=3.0)  # MB init, COM removed
    assert float(rows[-1]["step"]) == 40 and float(rows[-1]["time_ps"]) == pytest.approx(0.04)
    e_tot = np.array([float(r["E_tot_eV"]) for r in rows])
    assert np.ptp(e_tot) / 32 * 1000 < 0.5  # meV/atom
    assert res.traj_sha256 == sha256_file(ctx.out_dir / "md_nve_100K.traj")
    images = ase_read(str(ctx.out_dir / "md_nve_100K.traj"), index=":")
    assert len(images) == 9 and len(images[0]) == 32  # every 5 steps incl. step 0


def test_nvt_with_tiny_mace_writes_every_file(md_settings: Settings, tiny_mace, b20_atoms) -> None:  # type: ignore[no-untyped-def]
    cfg = md_settings
    ctx = _ctx(cfg, seed=3)
    calc = tiny_mace.calculator()
    stats: dict[str, object] = {}
    res = ase_md.run(
        b20_atoms("MnSi"), calc, "nvt", 300.0, 0.04, 2.0, cfg, ctx, natoms=8, seed=3,
        compound="MnSi", model_sha256=tiny_mace.sha256, stats=stats,
    )  # fmt: skip
    assert res.natoms == 8 and res.steps == 20 and res.ensemble == "nvt"
    assert res.model_sha256 == tiny_mace.sha256 and res.compound == "MnSi"
    assert res.drift_meV_atom_ps is None and res.a_mean_A is None
    MDResult.model_validate_json(res.model_dump_json())  # round trip
    names = {p.name for p in ctx.out_dir.iterdir()}
    assert {
        "md_nvt_300K.traj",
        "thermo_nvt_300K.csv",
        "vdos_nvt_300K.json",
        "rdf_nvt_300K.json",
    } <= names
    vd = json.loads(Path(res.vdos_path).read_text())  # type: ignore[arg-type]
    assert len(vd["freq_meV"]) == len(vd["dos"]) and vd["n_frames"] == 21 and vd["dt_fs"] == 2.0
    assert vd["freq_meV"][0] == 0.0 and np.trapezoid(vd["dos"], vd["freq_meV"]) == pytest.approx(
        1.0
    )
    rd = json.loads(Path(res.rdf_path).read_text())  # type: ignore[arg-type]
    assert len(rd["r_A"]) == cfg.md.rdf_nbins == 50 and rd["rmax_A"] < 4.56 / 2  # clipped
    assert rd["n_frames"] == 5 and max(rd["g"]) > 0
    kinds = {Path(a.path).name: a.kind for a in ctx.outputs}
    assert kinds["md_nvt_300K.traj"] == "traj" and kinds["vdos_nvt_300K.json"] == "json"
    assert kinds["thermo_nvt_300K.csv"] == "other"
    assert stats["T_mean_K"] > 0 and stats["s_per_step"] > 0


def test_npt_returns_a_mean_on_the_fixture_cell(
    md_settings: Settings, tiny_mace, b20_atoms
) -> None:  # type: ignore[no-untyped-def]
    cfg = md_settings
    ctx = _ctx(cfg)
    stats: dict[str, object] = {}
    res = ase_md.run(
        b20_atoms("FeSi"), tiny_mace.calculator(), "npt", 300.0, 0.02, 2.0, cfg, ctx, natoms=8,
        stats=stats,
    )  # fmt: skip
    assert res.ensemble == "npt" and res.pressure_GPa == 0.0 and res.steps == 10
    assert res.a_mean_A is not None and abs(res.a_mean_A - 4.48) < 0.2
    assert res.a_std_A is not None and res.a_std_A >= 0.0 and res.drift_meV_atom_ps is None
    assert stats["n_production"] == 6 and stats["a_ci95"] is None  # < 10 samples: no block CI
    with open(ctx.out_dir / "thermo_npt_300K.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(np.isfinite(float(r["pressure_GPa"])) for r in rows[1:])  # NPT computes stress
    assert float(rows[0]["a_A"]) == pytest.approx(4.48)


def test_equilibration_window_and_short_run_fallback(
    md_settings: Settings, argon_cell: Atoms, lj_calc: LennardJones
) -> None:
    half = md_settings.model_copy(
        update={"md": md_settings.md.model_copy(update={"equil_ps": 0.02})}
    )
    ctx = _ctx(half)
    stats: dict[str, object] = {}
    res = ase_md.run(argon_cell, lj_calc, "npt", 50.0, 0.04, 2.0, half, ctx, natoms=32, stats=stats)
    assert stats["n_production"] == 6 and stats["n_thermo"] == 11 and res.a_mean_A is not None
    vd = json.loads(Path(res.vdos_path).read_text())  # type: ignore[arg-type]
    assert vd["n_frames"] == 11  # velocity frames at t >= 0.02 ps (steps 10..20)
    too_long = md_settings.model_copy(
        update={"md": md_settings.md.model_copy(update={"equil_ps": 10.0})}
    )
    ctx = _ctx(too_long)
    stats = {}
    res = ase_md.run(
        argon_cell, lj_calc, "nve", 50.0, 0.04, 2.0, too_long, ctx, natoms=32, stats=stats
    )
    assert stats["production_window"].startswith("all samples") and stats["n_production"] == 11
    assert res.drift_meV_atom_ps is not None


def test_npt_rotates_a_non_triangular_cell(md_settings: Settings, lj_calc: LennardJones) -> None:
    atoms = Atoms("Ar4", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
                  cell=[[5.26, 0, 0], [0.3, 5.26, 0], [0.2, 0.1, 5.26]], pbc=True)  # fmt: skip
    ctx = _ctx(md_settings)
    res = ase_md.run(atoms, lj_calc, "npt", 50.0, 0.004, 2.0, md_settings, ctx, natoms=4)
    assert res.a_mean_A is not None and res.natoms == 4
    tri = ase_md._triangular(atoms)
    cell = np.asarray(tri.cell[:])
    assert (
        cell[0, 1] == cell[0, 2] == cell[1, 2] == 0.0
        or cell[1, 0] == cell[2, 0] == cell[2, 1] == 0.0
    )
    assert tri.get_volume() == pytest.approx(atoms.get_volume())


# --- analysis helpers ---------------------------------------------------------------------------


def test_thermal_expansion_on_synthetic_a_of_T() -> None:
    alpha_true = 1.2e-5
    results = [
        _result(temperature_K=T, a_mean_A=4.56 * (1 + alpha_true * (T - 300.0)))
        for T in (100.0, 300.0, 500.0)
    ]
    fit = ase_md.thermal_expansion_fit(results)
    assert fit["alpha_per_K"] == pytest.approx(alpha_true, rel=1e-6)
    assert fit["n"] == 3 and fit["temperatures_K"] == [100.0, 300.0, 500.0]
    assert fit["a_ref_A"] == pytest.approx(4.56) and fit["alpha_ci95"] is None
    assert ase_md.thermal_expansion(results) == pytest.approx(alpha_true, rel=1e-6)
    # T_ref changes the normalisation only
    assert ase_md.thermal_expansion(results, T_ref=100.0) == pytest.approx(
        alpha_true / (1 + alpha_true * (100.0 - 300.0)), rel=1e-6
    )
    four = results + [_result(temperature_K=700.0, a_mean_A=4.56 * (1 + alpha_true * 400.0))]
    assert ase_md.thermal_expansion_fit(four)["alpha_ci95"] is not None
    with pytest.raises(ValueError, match=">= 3 temperatures"):
        ase_md.thermal_expansion(results[:2])
    with pytest.raises(ValueError):  # NVE results carry no a(T)
        ase_md.thermal_expansion([_result(ensemble="nve", a_mean_A=None, pressure_GPa=None)] * 3)


def test_vdos_recovers_a_harmonic_frequency(tmp_path: Path) -> None:
    dt, n, f0 = 1.0, 1000, 0.01  # fs, frames, 1/fs (10 THz)
    t = np.arange(n) * dt
    vel = np.zeros((n, 2, 3))
    vel[:, 0, 0] = np.cos(2 * np.pi * f0 * t)
    vel[:, 1, 1] = 0.5 * np.sin(2 * np.pi * f0 * t)
    freq, dos = ase_md.vdos(vel, dt)
    assert freq[np.argmax(dos)] == pytest.approx(10.0 * ase_md.MEV_PER_THZ, abs=1e-9)
    assert np.trapezoid(dos, freq) == pytest.approx(1.0)
    # mass weighting changes the DOS of a two-mode signal, not its peak
    freq_m, dos_m = ase_md.vdos(vel, dt, masses=[1.0, 100.0])
    assert freq_m[np.argmax(dos_m)] == pytest.approx(10.0 * ase_md.MEV_PER_THZ, abs=1e-9)
    # from Atoms with velocities and from a trajectory file
    images = []
    for k in range(64):
        a = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
        a.set_velocities(vel[k] * units.Ang / units.fs)
        images.append(a)
    f1, d1 = ase_md.vdos(images, dt)
    assert f1.shape == d1.shape == (33,)
    path = tmp_path / "v.traj"
    from ase.io import write

    write(path, images)
    f2, d2 = ase_md.vdos(path, dt)
    assert np.allclose(d1, d2)
    with pytest.raises(ValueError):
        ase_md.vdos(np.zeros((1, 2, 3)), dt)
    with pytest.raises(ValueError):
        ase_md.vdos(np.zeros((5, 2)), dt)
    with pytest.raises(ValueError):
        ase_md.vdos([], dt)


def test_rdf_clips_rmax_to_the_cell(argon_cell: Atoms) -> None:
    r, g, rmax = ase_md.rdf([argon_cell.repeat(2)], 20.0, 40)
    assert rmax < 10.52 / 2 and len(r) == len(g) == 40
    nn = r[np.argmax(g)]
    assert nn == pytest.approx(5.26 / np.sqrt(2), abs=0.2)  # fcc nearest neighbour
    with pytest.raises(ValueError):
        ase_md.rdf([], 5.0, 10)


# --- provenance and labels ---------------------------------------------------------------------


def test_model_provenance_and_default_labels(tmp_path: Path, tiny_mace) -> None:  # type: ignore[no-untyped-def]
    prov = common.model_provenance(tiny_mace.model_path)
    assert prov["sha256"] == tiny_mace.sha256 and prov["lammps_sha256"] == tiny_mace.lammps_sha256
    assert prov["e0_source"] == "unknown" and prov["energy_scale"] == "none"
    assert prov["label"] == "tiny_b20" and prov["heads"] == ["Default"]
    # a -lammps.pt maps back to its .model
    via_lammps = common.model_provenance(tiny_mace.lammps_path)
    assert via_lammps["model_path"] == str(tiny_mace.model_path)
    # train-run layout: runs/train/<id>/models/x.model + checkpoint.json two levels up
    run_dir = tmp_path / "runs" / "train" / "r1"
    (run_dir / "models").mkdir(parents=True)
    model = run_dir / "models" / "b20_naive.model"
    model.write_bytes(b"weights")
    ckpt = {
        "model_path": str(model), "sha256": sha256_file(model), "lammps_path": None,
        "variant": "naive", "foundation": "medium-mpa-0", "foundation_sha256": "f" * 64,
        "heads": ["Default"], "e0_source": "E0s_qe.json", "energy_scale": "qe", "seed": 1,
        "epochs": 30, "lr": 1e-4, "batch_size": 4, "split_id": "v1", "replay_samples": None,
        "train_run_id": "r1", "val_metrics": {},
    }  # fmt: skip
    (run_dir / "checkpoint.json").write_text(json.dumps(ckpt))
    prov = common.model_provenance(model)
    assert prov["label"] == "B1" and prov["e0_source"] == "E0s_qe.json"
    assert prov["energy_scale"] == "qe" and prov["train_run_id"] == "r1"
    for variant, bracket in (("replay", "B2"), ("scratch", "B3"), ("bootstrap", "B4")):
        assert common.default_label({"variant": variant, "model_path": "x.model"}) == bracket
    foundation = tmp_path / "mace-mpa-0-medium.model"
    foundation.write_bytes(b"w")
    prov = common.model_provenance(foundation)
    assert (
        prov["label"] == "B0" and prov["e0_source"] == "foundation" and prov["energy_scale"] == "mp"
    )
    assert common.model_provenance(tmp_path / "absent.model")["sha256"] is None
    assert common.lammps_path_for("m.model").name == "m.model-lammps.pt"
    assert common.base_model_for("m.model-lammps.pt").name == "m.model"


# --- the stage ----------------------------------------------------------------------------------


def _numbers(run_dir: Path) -> dict:
    return json.loads((run_dir / "numbers.json").read_text())


def test_stage_npt_three_temperatures_end_to_end(
    md_settings: Settings, tiny_mace, structure_file: Path
) -> None:  # type: ignore[no-untyped-def]
    cfg = md_settings
    calc = tiny_mace.calculator()
    result = run_stage(
        "md.ase", cfg, ase_md.stage, seed=0, model=tiny_mace.model_path, compound="MnSi",
        ensemble="npt", T=[100.0, 300.0, 500.0], ps=0.02, natoms=8, structure=structure_file,
        calc=calc,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    s = result.summary
    assert (
        s["n_runs"] == 3
        and s["steps"] == 10
        and s["natoms"] == 8
        and s["model_label"] == "tiny_b20"
    )
    assert "alpha_per_K" in s and s["a_exp_A"] == 4.558 and "a_mean_A_300K" in s
    run_dir = Path(result.manifest_path).parent
    names = {Path(a.path).name for a in result.outputs}
    for T in ("100", "300", "500"):
        assert {f"md_npt_{T}K.traj", f"thermo_npt_{T}K.csv", f"md_result_npt_{T}K.json"} <= names
        res = MDResult.model_validate_json((run_dir / f"md_result_npt_{T}K.json").read_text())
        assert res.alpha_per_K == s["alpha_per_K"] and res.a_mean_A is not None
        assert res.model_sha256 == tiny_mace.sha256 and res.run_id == result.run_id
    assert "numbers.json" in names and "plan.json" in names
    numbers = _numbers(run_dir)
    keys = {k for k in numbers if not k.endswith("@meta")}
    assert {
        "md.ase.MnSi.tiny_b20.npt.100.a_mean_A", "md.ase.MnSi.tiny_b20.npt.300.a_std_A",
        "md.ase.MnSi.tiny_b20.npt.500.alpha_per_K", "md.ase.MnSi.a_300K_A", "md.ase.MnSi.a_exp_A",
        "md.ase.MnSi.a_dev_pct", "md.ase.MnSi.alpha_per_K",
    } <= keys  # fmt: skip
    assert numbers["md.ase.MnSi.a_exp_A"] == 4.558
    assert numbers["md.ase.MnSi.a_dev_pct"] == pytest.approx(
        100 * (numbers["md.ase.MnSi.a_300K_A"] - 4.558) / 4.558
    )
    meta = numbers["md.ase.MnSi.tiny_b20.npt.300.a_mean_A@meta"]
    assert meta["reference"] == {
        "code": "mace", "functional": "PBE", "pseudos": None, "e0_source": "unknown",
        "cross_functional": False,
    }  # fmt: skip
    assert meta["head"] == "Default" and meta["n"] == 6 and meta["seed"] == 0
    assert meta["ci95"] is None and meta["ci95_reason"] and meta["engine"] == "ase"
    assert meta["ensemble"] == "npt" and meta["T"] == 300.0 and meta["model_label"] == "tiny_b20"
    assert meta["energy_scale"] == "none" and meta["model_sha256"] == tiny_mace.sha256
    assert numbers["md.ase.MnSi.a_300K_A@meta"]["reference"]["code"] == "experiment"
    assert numbers["md.ase.MnSi.alpha_per_K@meta"]["temperatures_K"] == [100.0, 300.0, 500.0]
    for key in keys:  # every value is a finite number
        assert np.isfinite(float(numbers[key])), key
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["engine"] == "ase" and manifest.extras["timestep_fs"] == 2.0
    assert manifest.extras["thermostat"].startswith("Nose-Hoover") and manifest.seed == 0
    assert manifest.extras["thermal_expansion"]["n"] == 3
    assert {Path(a.path).name for a in manifest.inputs} == {
        tiny_mace.model_path.name,
        structure_file.name,
    }
    report_numbers = pytest.importorskip("b20mlip.report.numbers")
    parsed = report_numbers.parse_numbers_file(numbers)
    assert set(parsed) == keys and parsed["md.ase.MnSi.a_300K_A"][1]["head"] == "Default"


def test_stage_nve_and_nvt_numbers(md_settings: Settings, tiny_mace, structure_file: Path) -> None:  # type: ignore[no-untyped-def]
    calc = tiny_mace.calculator()
    nve = run_stage(
        "md.ase", md_settings, ase_md.stage, seed=1, model=tiny_mace.model_path, compound="MnSi",
        ensemble="nve", T=300.0, ps=0.02, natoms=8, structure=structure_file, calc=calc,
        label="B1",
    )  # fmt: skip
    assert nve.status == "ok" and "drift_meV_atom_ps_300K" in nve.summary
    numbers = _numbers(Path(nve.manifest_path).parent)
    assert set(k for k in numbers if not k.endswith("@meta")) == {
        "md.ase.MnSi.B1.nve.300.drift_meV_atom_ps", "md.ase.MnSi.drift_meV_atom_ps",
    }  # fmt: skip
    assert numbers["md.ase.MnSi.drift_meV_atom_ps@meta"]["model_label"] == "B1"
    nvt = run_stage(
        "md.ase", md_settings, ase_md.stage, model=tiny_mace.model_path, compound="MnSi",
        ensemble="nvt", T=[300.0], ps=0.01, natoms=8, structure=structure_file, calc=calc,
    )  # fmt: skip
    assert nvt.status == "ok" and _numbers(Path(nvt.manifest_path).parent) == {}
    assert read_manifest(nvt.manifest_path).extras["thermostat"].startswith("Langevin")


def test_stage_dry_run_missing_model_and_bad_head(
    md_settings: Settings, tiny_mace, structure_file: Path, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    dry = run_stage(
        "md.ase", md_settings, ase_md.stage, dry_run=True, model=tiny_mace.model_path,
        compound="MnSi", ensemble="nvt", T=300.0, ps=1.0, structure=structure_file, natoms=64,
    )  # fmt: skip
    assert (
        dry.status == "partial"
        and dry.summary == {"planned": 1, "steps": 500}
        and dry.outputs == []
    )
    plan = json.loads((Path(dry.manifest_path).parent / "plan.json").read_text())
    assert plan["natoms"] == 64 and plan["reps"] == 2 and plan["dry_run"] is True
    missing = run_stage(
        "md.ase", md_settings, ase_md.stage, model=tmp_path / "nope.model", compound="MnSi",
        ensemble="nvt", T=300.0, ps=0.01, structure=structure_file,
    )  # fmt: skip
    assert missing.status == "failed" and "model not found" in missing.summary["error"]
    bad_head = run_stage(
        "md.ase", md_settings, ase_md.stage, model=tiny_mace.model_path, compound="MnSi",
        ensemble="nvt", T=300.0, ps=0.01, natoms=8, structure=structure_file, head="pt_head",
    )  # fmt: skip
    assert bad_head.status == "failed" and "not 'pt_head'" in bad_head.summary["error"]
    no_T = run_stage(
        "md.ase", md_settings, ase_md.stage, model=tiny_mace.model_path, compound="MnSi",
        ensemble="nvt", T=[], ps=0.01, structure=structure_file,
    )  # fmt: skip
    assert no_T.status == "failed" and "temperature" in no_T.summary["error"]
    no_structure = run_stage(
        "md.ase", md_settings, ase_md.stage, model=tiny_mace.model_path, compound="MnSi",
        ensemble="nvt", T=300.0, ps=0.01, structure=tmp_path / "absent.extxyz",
    )  # fmt: skip
    assert no_structure.status == "failed"
