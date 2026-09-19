"""`data sample`: derived configurations (strain/shear/rattle/eos/vacancy/phonon_disp/md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.calculators.lj import LennardJones
from helpers_data import b20_frame

from b20mlip.config import Settings
from b20mlip.data import sample
from b20mlip.io import read_frames, write_frames
from b20mlip.models import Frame
from b20mlip.provenance import RunContext, read_manifest, run_stage


def small_options(**kw) -> sample.SampleOptions:  # type: ignore[no-untyped-def]
    base = dict(
        rattle_levels=2, rattle_seeds=2, n_eos=3, md_equil_steps=4, md_snapshots=2, md_stride=2
    )
    base.update(kw)
    return sample.SampleOptions(**base)


def _parent(compound: str = "FeSi") -> Frame:
    return b20_frame(compound, labelled=False)


def test_select_parents_prefers_smallest_forces() -> None:
    noisy = b20_frame("FeSi", seed=1)
    calm = b20_frame("FeSi", seed=2, scale=1.01)
    calm = calm.model_copy(update={"forces": (np.asarray(calm.forces) * 1e-3).tolist()})
    big = b20_frame("FeSi", labelled=False)
    big = big.model_copy(
        update={"numbers": big.numbers * 2, "positions": big.positions * 2, "frame_id": "x" * 16}
    )
    stats: dict = {}
    parents = sample.select_parents([noisy, big, calm], ["FeSi", "MnGe"], stats=stats)
    assert [p.frame_id for p in parents] == [calm.frame_id]
    assert stats["missing"] == ["MnGe"] and stats["parents"] == {"FeSi": calm.frame_id}
    assert sample.max_abs_force(big) == float("inf")


def test_generate_counts_and_conventions() -> None:
    cfg = Settings.model_validate({})
    parent = _parent()
    rng = np.random.default_rng(0)
    calc = LennardJones(sigma=2.0, epsilon=0.1, rc=6.0)
    frames = sample.generate(cfg, None, [parent], rng, calc=calc, md=True, options=small_options())
    counts: dict[str, int] = {}
    for f in frames:
        counts[f.config_type] = counts.get(f.config_type, 0) + 1
    assert counts == {
        "strain": 24, "shear": 12, "rattle": 4, "eos": 3, "vacancy": 2, "phonon_disp": 5, "md": 6
    }  # fmt: skip
    assert len({f.frame_id for f in frames}) == len(frames)
    for f in frames:
        assert f.label_source == "none" and f.energy_scale == "none"
        assert f.forces is None and f.energy is None and f.stress is None
        assert f.parent_id == parent.frame_id and f.compound == "FeSi"
        assert f.group_id == f"FeSi/{f.config_type}/{parent.frame_id}"
    cell = np.asarray(parent.cell)
    strain = [f for f in frames if f.config_type == "strain"]
    iso = next(f for f in strain if f.info["strain_axis"] == "iso" and f.info["strain"] == 0.02)
    assert np.allclose(np.asarray(iso.cell), cell * 1.02)
    uni = next(f for f in strain if f.info["strain_axis"] == "y" and f.info["strain"] == -0.04)
    assert np.allclose(np.asarray(uni.cell)[1], cell[1] * 0.96)
    assert np.allclose(np.asarray(uni.cell)[0], cell[0])
    shear = next(f for f in frames if f.config_type == "shear" and f.info["shear_plane"] == "xy")
    assert not np.allclose(np.asarray(shear.cell), np.diag(np.diag(np.asarray(shear.cell))))
    rattle = [f for f in frames if f.config_type == "rattle"]
    assert sorted({f.info["rattle_sigma_A"] for f in rattle}) == [0.075, 0.15]
    assert all(np.allclose(np.asarray(f.cell), cell) for f in rattle)
    eos = [f.info["eos_volume_scale"] for f in frames if f.config_type == "eos"]
    assert eos == pytest.approx([0.92, 1.0, 1.08])
    vac = [f for f in frames if f.config_type == "vacancy"]
    assert [len(f.numbers) for f in vac] == [63, 63]
    assert {f.info["vacancy_species"] for f in vac} == {"Fe", "Si"}
    assert vac[0].info["supercell"] == "2x2x2"
    ph = [f for f in frames if f.config_type == "phonon_disp"]
    assert [len(f.numbers) for f in ph] == [64] * 5
    assert [f.info["phonopy_disp_number"] for f in ph] == [-1, 0, 1, 2, 3]
    assert all(np.linalg.norm(f.info["phonopy_disp_vector"]) == pytest.approx(0.03) for f in ph[1:])
    assert ph[1].info["phonopy_distance"] == 0.03 and "phonopy_disp_atom" in ph[1].info
    md = [f for f in frames if f.config_type == "md"]
    assert [f.temperature_K for f in md] == [300.0, 300.0, 600.0, 600.0, 900.0, 900.0]
    assert [f.info["md_step"] for f in md] == [6, 8, 6, 8, 6, 8]
    assert (
        md[0].info["md_calculator"] == "LennardJones" and md[0].info["md_kinetic_temperature_K"] > 0
    )
    assert md[0].info["md_timestep_fs"] == cfg.md.timestep_fs and md[0].info["supercell"] == "1x1x1"
    assert len(md[0].numbers) == 8


def test_generate_options_and_errors() -> None:
    cfg = Settings.model_validate({})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="calculator"):
        sample.generate(cfg, None, [_parent()], rng, md=True)
    only = sample.generate(
        cfg, None, [_parent()], rng, options=small_options(config_types=("eos",))
    )
    assert {f.config_type for f in only} == {"eos"} and len(only) == 3
    # md requested but not in config_types -> no md frames, no calculator needed
    none = sample.generate(
        cfg, None, [_parent()], rng, md=True, calc=LennardJones(),
        options=small_options(config_types=("shear",)),
    )  # fmt: skip
    assert {f.config_type for f in none} == {"shear"}
    two = sample.generate(cfg, None, [_parent("FeSi"), _parent("MnSi")], rng,
                          options=small_options(config_types=("vacancy",)))  # fmt: skip
    assert [f.compound for f in two] == ["FeSi", "FeSi", "MnSi", "MnSi"]
    assert {f.info["vacancy_species"] for f in two if f.compound == "MnSi"} == {"Mn", "Si"}
    md_sc = sample.generate(
        cfg, None, [_parent()], rng, md=True, calc=LennardJones(sigma=2.0, epsilon=0.1, rc=6.0),
        options=small_options(config_types=("md",), md_supercell=(2, 1, 1), md_snapshots=1),
    )  # fmt: skip
    assert [len(f.numbers) for f in md_sc] == [16, 16, 16]


def test_generate_dry_run_logs_plan(data_settings: Settings) -> None:
    ctx = RunContext("data.sample", data_settings, seed=0, dry_run=True)
    assert sample.generate(data_settings, ctx, [_parent()], np.random.default_rng(0)) == []
    plan = ctx.extras["plan"]
    assert plan["parents"] == 1 and plan["strain"] == 24 and plan["md"] == 0
    assert plan["total_planned"] == 24 + 12 + 30 + 11 + 2 + 5
    assert sample.plan(data_settings, [_parent()], sample.SampleOptions(), True)["md"] == 30


def test_deform_math() -> None:
    from helpers_data import b20_cell

    atoms = b20_cell("CoSi")
    same = sample.deform(atoms, np.eye(3))
    assert np.allclose(same.cell[:], atoms.cell[:]) and np.allclose(same.positions, atoms.positions)
    F = np.eye(3)
    F[0, 1] = 0.1
    sheared = sample.deform(atoms, F)
    assert np.allclose(sheared.cell[:], atoms.cell[:] @ F.T)
    assert np.allclose(sheared.get_scaled_positions(), atoms.get_scaled_positions())


def test_run_stage_end_to_end(data_settings: Settings, tmp_path: Path) -> None:
    parents = tmp_path / "parents.extxyz"
    write_frames([b20_frame("FeSi", seed=3), b20_frame("CoSi", seed=4), b20_frame("MnSi", seed=5)],
                 parents)  # fmt: skip
    out = tmp_path / "cand.extxyz"
    result = run_stage(
        "data.sample", data_settings, sample.run, seed=0, out=out, parents=parents,
        compounds=["FeSi", "CoSi", "MnGe"], options=small_options(config_types=("strain", "eos")),
    )  # fmt: skip
    assert result.status == "ok", result.summary
    assert result.summary == {"parents": 2, "frames": 54, "n_strain": 48, "n_eos": 6}
    frames = read_frames(out)
    assert len(frames) == 54 and {f.compound for f in frames} == {"FeSi", "CoSi"}
    assert frames[0].info["strain_axis"] == "iso" and frames[0].label_source == "none"
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["parent_selection"]["missing"] == ["MnGe"]
    assert manifest.extras["sample_counts"] == {"strain": 48, "eos": 6}
    assert {Path(a.path).name for a in manifest.outputs} == {"cand.extxyz"}
    assert [Path(a.path).name for a in manifest.inputs] == ["parents.extxyz"]
    dry = run_stage("data.sample", data_settings, sample.run, seed=0, dry_run=True, out=out,
                    parents=parents)  # fmt: skip
    assert dry.status == "partial" and dry.summary["total_planned"] > 0
    missing = run_stage("data.sample", data_settings, sample.run, out=out,
                        parents=tmp_path / "nope.extxyz")  # fmt: skip
    assert missing.status == "failed" and "parents file not found" in missing.summary["error"]
    nob20 = tmp_path / "nob20.extxyz"
    write_frames([b20_frame("FeGe")], nob20)
    bad = run_stage("data.sample", data_settings, sample.run, out=out, parents=nob20,
                    compounds=["FeSi"])  # fmt: skip
    assert bad.status == "failed" and "no B20 parent" in bad.summary["error"]


def test_load_mace_calculator_checks_file_first(data_settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="foundation model"):
        sample.load_mace_calculator(data_settings)
    with pytest.raises(FileNotFoundError):
        sample.load_mace_calculator(data_settings, tmp_path / "x.model")
