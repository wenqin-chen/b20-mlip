"""Discovery: relax, MP-scale e_form + MP2020 corrections, hull shortcut, RMSD, resumable
run_sample with stubbed WBM loaders, sample metrics, paired ΔF1, the run() stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from b20mlip.config import Settings
from b20mlip.data import wbm
from b20mlip.evaluate import discovery as disc
from b20mlip.evaluate.mbd_vendored import stable_metrics
from b20mlip.io import frame_to_atoms
from b20mlip.models import Frame, Metric
from b20mlip.provenance import RunContext, read_manifest, run_stage

from .conftest import BOOT_N

SAMPLE_IDS = ["wbm-1-16845", "wbm-1-55036", "wbm-2-27173"]
TRUTH = {
    # e_form_per_atom of the first id is a synthetic +5.0 so the (physically meaningless) tiny
    # model predicts it stable: precision is then defined (the other two are predicted unstable)
    "wbm-1-16845": {
        "formula": "Fe4 Si4",
        "n_sites": 8,
        "e_form_per_atom": 5.0,
        "e_above_hull": -0.01,
        "stable": True,
    },
    "wbm-1-55036": {
        "formula": "Co4 Si4",
        "n_sites": 8,
        "e_form_per_atom": -0.20,
        "e_above_hull": 0.00,
        "stable": True,
    },
    "wbm-2-27173": {
        "formula": "Mn4 Si4",
        "n_sites": 8,
        "e_form_per_atom": -0.10,
        "e_above_hull": 0.08,
        "stable": False,
    },
}


def _sample(path: Path, ids: list[str] = SAMPLE_IDS) -> Path:
    data = {
        "schema": 1, "n": len(ids), "seed": 0, "ids": ids, "prevalence": 2 / 3,
        "in_family_ids": ids, "truth": {i: TRUTH[i] for i in ids},
        "checksums": {"summary_csv": {"sha256": "x"}, "atoms_zip": {"sha256": "y"}},
    }  # fmt: skip
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _structures(tiny_frames: list[Frame]) -> dict[str, Atoms]:
    out: dict[str, Atoms] = {}
    for wbm_id, compound in zip(SAMPLE_IDS, ("FeSi", "CoSi", "MnSi"), strict=True):
        frame = next(f for f in tiny_frames if f.compound == compound and f.config_type == "rattle")
        atoms = frame_to_atoms(frame)
        atoms.calc = None
        atoms.info = {}
        out[wbm_id] = atoms
    return out


@pytest.fixture
def structures(tiny_frames: list[Frame]) -> dict[str, Atoms]:
    return _structures(tiny_frames)


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    return _sample(tmp_path / "sample.json")


@pytest.fixture
def stub_wbm(monkeypatch: pytest.MonkeyPatch, structures: dict[str, Atoms], tmp_path: Path) -> Path:
    """``wbm.atoms_for_ids`` serves the fixture cells; the "zip" is an empty placeholder file."""
    calls: list[list[str]] = []

    def fake(atoms_zip: Path, ids: Any) -> dict[str, Atoms]:
        ids = list(ids)
        calls.append(ids)
        return {i: structures[i].copy() for i in ids}

    monkeypatch.setattr(wbm, "atoms_for_ids", fake)
    monkeypatch.setattr(wbm, "_test_calls", calls, raising=False)
    zip_path = tmp_path / "wbm-initial-atoms.extxyz.zip"
    zip_path.write_bytes(b"")
    return zip_path


# --- relax, e_form, corrections, rmsd ------------------------------------------------------------


def test_relax_rattled_fixture_cell(fesi_atoms: Atoms, tiny_calc: Any) -> None:
    rattled = fesi_atoms.copy()
    rattled.rattle(0.05, seed=3)
    relaxed, steps = disc.relax(rattled, tiny_calc, fmax=1e-5, steps=25)
    assert steps > 0 and len(relaxed) == 8
    f_max = disc.max_force(relaxed)
    assert f_max <= 1e-5 or steps == 25  # fmax reached or the cap hit
    assert relaxed.calc is tiny_calc and np.isfinite(relaxed.get_potential_energy())
    assert rattled.positions is not relaxed.positions  # a copy was relaxed
    assert disc.rmsd_internal(rattled, relaxed) >= 0.0
    # a physical case converges: EMT Al from a stretched cell
    al = bulk("Al", "fcc", a=4.2, cubic=True)
    relaxed_al, steps_al = disc.relax(al, EMT(), fmax=0.02, steps=200)
    assert 0 < steps_al < 200 and disc.max_force(relaxed_al) <= 0.02
    assert abs(relaxed_al.cell.lengths()[0] - 4.0) < 0.1
    sym, _ = disc.relax(al, EMT(), fmax=0.02, steps=50, fix_symmetry=True)
    assert len(sym.constraints) == 1


def test_rmsd_internal_is_strain_invariant(fesi_atoms: Atoms) -> None:
    strained = fesi_atoms.copy()
    strained.set_cell(fesi_atoms.cell[:] * 1.05, scale_atoms=True)
    assert disc.rmsd_internal(fesi_atoms, strained) == pytest.approx(0.0, abs=1e-12)
    moved = fesi_atoms.copy()
    moved.positions[0] += [0.1, 0.0, 0.0]
    assert disc.rmsd_internal(fesi_atoms, moved) == pytest.approx(0.1 / np.sqrt(8))
    wrapped = fesi_atoms.copy()
    wrapped.positions[0] += fesi_atoms.cell[0]  # a lattice translation is no displacement
    assert disc.rmsd_internal(fesi_atoms, wrapped) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="atom count"):
        disc.rmsd_internal(fesi_atoms, fesi_atoms.repeat((2, 1, 1)))


def test_elemental_references_and_hubbards() -> None:
    refs = disc.mp_elemental_reference_energies()
    assert len(refs) == 89 and refs["Fe"] == pytest.approx(-8.47002121)
    assert refs["Si"] == pytest.approx(-5.425318, abs=1e-5) and refs["Mn"] == pytest.approx(
        -9.162015, abs=1e-5
    )
    from pymatgen.core import Composition

    assert disc.mp_hubbards(Composition("Fe2O3")) == {"Fe": 5.3}
    assert disc.mp_hubbards(Composition("FeSi")) == {}
    assert disc.mp_hubbards(Composition("LiCoF3")) == {"Co": 3.32}
    assert disc.mp_hubbards(Composition("NaCl")) == {}


def test_e_form_and_mp2020_corrections(fesi_atoms: Atoms) -> None:
    refs = disc.mp_elemental_reference_energies()
    e_ref = 4 * refs["Fe"] + 4 * refs["Si"]
    corr = disc.mp2020_correction(fesi_atoms, -60.0)
    assert corr["run_type"] == "GGA" and corr["hubbards"] == {}
    assert corr["adjustments"] == ["MP2020 anion correction (Si)"]
    assert corr["correction_eV"] == pytest.approx(0.284)  # 4 Si x 0.071 eV
    form = disc.e_form_per_atom(fesi_atoms, -60.0, refs)
    assert form["e_form_per_atom"] == pytest.approx((-60.0 + 0.284 - e_ref) / 8)
    assert form["correction_per_atom"] == pytest.approx(0.284 / 8)
    raw = disc.e_form_per_atom(fesi_atoms, -60.0, refs, corrections=False)
    assert raw["e_form_per_atom"] == pytest.approx((-60.0 - e_ref) / 8) and raw["adjustments"] == []
    # a GGA+U composition gets the mixing correction through the MPRelaxSet rule
    fe2o3 = Atoms(
        "Fe4O6",
        positions=np.random.default_rng(0).uniform(0, 4, (10, 3)),
        cell=np.eye(3) * 5,
        pbc=True,
    )
    corr = disc.mp2020_correction(fe2o3, -80.0)
    assert corr["run_type"] == "GGA+U" and corr["hubbards"] == {"Fe": 5.3}
    assert any("GGA/GGA+U mixing correction (Fe)" in a for a in corr["adjustments"])
    with pytest.raises(ValueError, match="no MP elemental reference"):
        disc.e_form_per_atom(fesi_atoms, -60.0, {"Fe": -8.0})
    assert disc.offset_shift(fesi_atoms, {"Fe": 1.5, "Si": -0.75}) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="no coefficient"):
        disc.offset_shift(fesi_atoms, {"Fe": 1.5})


def test_evaluate_structure_hull_shortcut(fesi_atoms: Atoms, tiny_calc: Any) -> None:
    refs = disc.mp_elemental_reference_energies()
    truth = {"e_form_per_atom": -0.3, "e_above_hull": -0.01}
    rec = disc.evaluate_structure("wbm-x", fesi_atoms, tiny_calc, truth, refs, fmax=0.05, steps=5)
    assert (
        rec["id"] == "wbm-x" and rec["n_sites"] == 8 and rec["steps"] <= 5 and rec["error"] is None
    )
    assert rec["e_above_hull_pred"] == pytest.approx(-0.01 + (rec["e_form_pred"] - (-0.3)))
    assert rec["e_form_true"] == -0.3 and rec["e_above_hull_true"] == -0.01
    assert rec["run_type"] == "GGA" and rec["correction_per_atom"] == pytest.approx(0.284 / 8)
    assert rec["runtime_s"] > 0 and rec["converged"] == (rec["fmax_final_eVA"] <= 0.05)
    no_truth = disc.evaluate_structure(
        "wbm-y", fesi_atoms, tiny_calc, None, refs, fmax=0.05, steps=1
    )
    assert no_truth["e_form_pred"] is not None and no_truth["e_above_hull_pred"] is None
    shifted = disc.evaluate_structure(
        "wbm-z",
        fesi_atoms,
        tiny_calc,
        truth,
        refs,
        fmax=0.05,
        steps=1,
        offsets={"Fe": 1.0, "Si": 0.0},
    )
    assert shifted["offset_shift_eV"] == pytest.approx(4.0)
    assert shifted["e_form_pred"] == pytest.approx(no_truth["e_form_pred"] + 0.5)
    broken = disc.evaluate_structure(
        "wbm-w", fesi_atoms, tiny_calc, truth, {"Fe": 0.0}, fmax=0.05, steps=1
    )
    assert (
        broken["error"] and "no MP elemental reference" in broken["error"] and broken["rmsd_A"] >= 0
    )


# --- metrics and paired dF1 ---------------------------------------------------------------------


def _records(preds: dict[str, float | None]) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "e_above_hull_pred": p,
            "rmsd_A": 0.01 * (k + 1),
            "steps": 3,
            "converged": k != 0,
            "runtime_s": 0.5,
        }
        for k, (i, p) in enumerate(preds.items())
    ]


def test_sample_metrics_matches_vendored_and_drops_daf() -> None:
    preds = {"wbm-1-16845": -0.02, "wbm-1-55036": 0.05, "wbm-2-27173": None}
    scored = disc.sample_metrics(_records(preds), TRUTH, BOOT_N, 0)
    expected = stable_metrics([-0.01, 0.0, 0.08], [-0.02, 0.05, float("nan")])
    m = scored["metrics"]
    assert m["f1"]["value"] == pytest.approx(expected["F1"]) and m["f1"]["unit"] == "F1"
    assert m["precision"]["value"] == pytest.approx(expected["Precision"])
    assert m["recall"]["value"] == pytest.approx(expected["Recall"]) == pytest.approx(0.5)
    assert m["mae_e_above_hull"]["value"] == pytest.approx(expected["MAE"] * 1e3)
    assert m["rmse_e_above_hull"]["value"] == pytest.approx(expected["RMSE"] * 1e3)
    assert m["mae_e_above_hull"]["n"] == 2 and m["f1"]["n"] == 3
    assert m["rmsd"]["value"] == pytest.approx(0.02) and m["rmsd"]["ci95"] is not None
    assert "DAF" not in scored["vendored"] and "daf" not in json.dumps(scored).lower()
    assert (
        scored["vendored"]["TP"] == 1
        and scored["vendored"]["FN"] == 1
        and scored["vendored"]["TN"] == 1
    )
    assert scored["counts"] == {
        "n_ids": 3,
        "n_with_truth": 3,
        "n_with_prediction": 2,
        "n_converged": 2,
        "n_capped": 1,
        "steps_mean": 3.0,
        "runtime_s_mean": 0.5,
    }
    assert scored["prevalence"] == pytest.approx(2 / 3) and scored["stability_threshold"] == 0.0
    for metric in m.values():
        if metric["ci95"] is not None:
            assert metric["ci95"][0] <= metric["value"] <= metric["ci95"][1] + 1e-12
    # degenerate: no positive prediction -> F1 undefined upstream -> null here (never 0.0 published)
    none = disc.sample_metrics(
        _records({"wbm-1-16845": 0.5, "wbm-1-55036": 0.5, "wbm-2-27173": 0.5}), TRUTH, 10, 0
    )
    assert none["metrics"]["f1"]["value"] is None and none["metrics"]["precision"]["value"] is None
    with pytest.raises(ValueError, match="no records"):
        disc.sample_metrics([], TRUTH, 10, 0)


def test_paired_delta_f1_known_difference() -> None:
    rng = np.random.default_rng(0)
    n = 60
    truth = {f"id{i}": {"e_above_hull": float(rng.normal(0.0, 0.1))} for i in range(n)}
    ids = list(truth)
    pred_a = {i: truth[i]["e_above_hull"] + 0.05 for i in ids}  # biased up: misses stable ones
    pred_b = {i: truth[i]["e_above_hull"] for i in ids}  # perfect
    true = np.array([truth[i]["e_above_hull"] for i in ids])
    f1_a = stable_metrics(true, np.array([pred_a[i] for i in ids]))["F1"]
    exact = 1.0 - f1_a
    metric = disc.paired_delta_f1(pred_a, pred_b, truth, 300, 1)
    assert isinstance(metric, Metric) and metric.unit == "ΔF1" and metric.n == n
    assert metric.value == pytest.approx(exact)
    assert metric.ci95 is not None and metric.ci95[0] <= exact <= metric.ci95[1]
    assert metric == disc.paired_delta_f1(pred_a, pred_b, truth, 300, 1)  # deterministic
    # identical predictions -> exactly zero with a zero-width interval; truth may be floats
    zero = disc.paired_delta_f1(pred_b, pred_b, {i: truth[i]["e_above_hull"] for i in ids}, 50, 0)
    assert zero.value == 0.0 and zero.ci95 == (0.0, 0.0)
    # None predictions count as unstable; ids without truth are dropped from n
    dropped = disc.paired_delta_f1(
        {"a": None, "b": -0.1}, {"a": -0.1, "b": -0.1}, {"a": -0.05, "b": float("nan")}, 10, 0
    )
    assert dropped.n == 1 and dropped.value == pytest.approx(1.0) and dropped.ci95 is None
    with pytest.raises(ValueError, match="identical ids"):
        disc.paired_delta_f1({"a": 0.0}, {"b": 0.0}, truth, 10, 0)
    with pytest.raises(ValueError, match="no truth"):
        disc.paired_delta_f1({"zz": 0.0}, {"zz": 0.0}, truth, 10, 0)
    with pytest.raises(ValueError, match="finite truth"):
        disc.paired_delta_f1({"a": 0.0}, {"a": 0.0}, {"a": None}, 10, 0)


# --- run_sample and run() -------------------------------------------------------------------------


def test_run_sample_resumes_and_scores(
    tiny_mace: Any,
    tiny_calc: Any,
    eval_settings: Settings,
    sample_path: Path,
    structures: dict[str, Atoms],
) -> None:
    sample = wbm.load_sample(sample_path)
    ctx = RunContext("eval.discovery", eval_settings, seed=0)
    first = disc.run_sample(
        tiny_mace.model_path,
        "Default",
        sample,
        eval_settings,
        ctx,
        atoms=structures,
        limit=2,
        calc=tiny_calc,
        n_boot=20,
    )
    assert (
        first["n_ids"] == 2
        and first["n_new"] == 2
        and first["n_resumed"] == 0
        and first["limit"] == 2
    )
    results = ctx.out_dir / "results.jsonl"
    assert len(results.read_text().splitlines()) == 2
    second = disc.run_sample(
        tiny_mace.model_path,
        "Default",
        sample,
        eval_settings,
        ctx,
        resume=True,
        atoms=structures,
        calc=tiny_calc,
        n_boot=20,
    )
    assert second["n_ids"] == 3 and second["n_new"] == 1 and second["n_resumed"] == 2
    records = disc.read_results(results)
    assert list(records) == SAMPLE_IDS and all(
        r["e_above_hull_pred"] is not None for r in records.values()
    )
    assert all(
        r["e_above_hull_pred"]
        == pytest.approx(TRUTH[i]["e_above_hull"] + r["e_form_pred"] - TRUTH[i]["e_form_per_atom"])
        for i, r in records.items()
    )
    assert (
        second["metrics"]["mae_e_above_hull"]["n"] == 3
        and second["hull_column"] == disc.HULL_COLUMN
    )
    assert second["energy_scale"] == "mp" and second["sample_seed"] == 0 and second["sample_n"] == 3
    assert second["fmax"] == 0.05 and second["max_steps"] == 30 and second["bootstrap_n"] == 20
    # resume=False starts over
    third = disc.run_sample(
        tiny_mace.model_path,
        "Default",
        sample,
        eval_settings,
        ctx,
        resume=False,
        atoms=structures,
        calc=tiny_calc,
        n_boot=20,
    )
    assert third["n_new"] == 3 and len(results.read_text().splitlines()) == 3
    with pytest.raises(KeyError, match="no structure"):
        disc.run_sample(
            tiny_mace.model_path,
            "Default",
            sample,
            eval_settings,
            RunContext("eval.discovery", eval_settings),
            atoms={},
            calc=tiny_calc,
        )


def test_run_sample_energy_scale_rules(
    tiny_mace: Any,
    tiny_calc: Any,
    eval_settings: Settings,
    sample_path: Path,
    structures: dict[str, Atoms],
) -> None:
    sample = wbm.load_sample(sample_path)
    ctx = RunContext("eval.discovery", eval_settings)
    with pytest.raises(ValueError, match="rule R3"):
        disc.run_sample(
            tiny_mace.model_path,
            "Default",
            sample,
            eval_settings,
            ctx,
            atoms=structures,
            energy_scale="qe",
            calc=tiny_calc,
        )
    with pytest.raises(ValueError, match="no route to the MP hull"):
        disc.run_sample(
            tiny_mace.model_path,
            "Default",
            sample,
            eval_settings,
            ctx,
            atoms=structures,
            energy_scale="omat24",
            calc=tiny_calc,
        )
    bad = {"coefficients": {"Fe": 0.0, "Si": 0.0, "Co": 0.0, "Mn": 0.0}, "residual_meV_atom": 35.0}
    with pytest.raises(ValueError, match="> gate 20"):
        disc.run_sample(
            tiny_mace.model_path,
            "Default",
            sample,
            eval_settings,
            ctx,
            atoms=structures,
            energy_scale="qe",
            offsets=bad,
            calc=tiny_calc,
        )
    good = {
        **bad,
        "residual_meV_atom": 5.0,
        "coefficients": {"Fe": 1.0, "Si": 0.5, "Co": 0.0, "Mn": 0.0},
    }
    scored = disc.run_sample(
        tiny_mace.model_path,
        "Default",
        sample,
        eval_settings,
        ctx,
        atoms=structures,
        energy_scale="qe",
        offsets=good,
        limit=1,
        calc=tiny_calc,
        n_boot=10,
    )
    assert scored["offsets_residual_meV_atom"] == 5.0
    rec = next(iter(disc.read_results(ctx.out_dir / "results.jsonl").values()))
    assert rec["offset_shift_eV"] == pytest.approx(4 * 1.0 + 4 * 0.5)
    ctx2 = RunContext("eval.discovery", eval_settings)
    with pytest.raises(FileNotFoundError, match="archive not found"):
        disc.run_sample(
            tiny_mace.model_path, "Default", sample, eval_settings, ctx2, calc=tiny_calc
        )


def test_run_stage_numbers_and_paired_baseline(
    tiny_mace: Any, eval_settings: Settings, sample_path: Path, stub_wbm: Path, tmp_path: Path
) -> None:
    base = run_stage(
        "eval.discovery", eval_settings, disc.run, model=tiny_mace.model_path, head="Default",
        sample=sample_path, atoms_zip=stub_wbm, energy_scale="mp", label="B0",
        e0_source="foundation", bootstrap_n=20,
    )  # fmt: skip
    assert base.status == "ok", base.summary
    base_dir = Path(base.manifest_path).parent
    numbers = json.loads((base_dir / "numbers.json").read_text())
    keys = {k for k in numbers if not k.endswith("@meta")}
    assert {
        "eval.discovery.B0.precision",
        "eval.discovery.B0.recall",
        "eval.discovery.B0.mae_e_above_hull",
        "eval.discovery.B0.rmse_e_above_hull",
        "eval.discovery.B0.rmsd",
        "eval.discovery.B0.n_ids",
        "eval.discovery.B0.n_converged",
        "eval.discovery.B0.steps_mean",
        "eval.discovery.B0.runtime_s_mean",
    } == keys
    assert "eval.discovery.B0.f1" not in numbers and "daf" not in json.dumps(numbers).lower()
    meta = numbers["eval.discovery.B0.mae_e_above_hull@meta"]
    assert (
        meta["reference"]["code"] == "vasp"
        and meta["reference"]["functional"] == "PBE"
        and meta["reference"]["e0_source"] == "foundation"
    )
    assert (
        meta["head"] == "Default"
        and meta["n"] == 3
        and meta["seed"] == 0
        and meta["prevalence"] == "natural"
    )
    assert (
        meta["sample_seed"] == 0
        and meta["energy_scale"] == "mp"
        and meta["model_label"] == "B0"
        and meta["tier"] == "T4b"
    )
    assert meta["ci95"] and meta["unit"] == "meV/atom" and meta["e0_source"] == "foundation"
    assert (
        numbers["eval.discovery.B0.n_ids@meta"]["ci95"] is None
        and "ci95_reason" in numbers["eval.discovery.B0.n_ids@meta"]
    )
    manifest = read_manifest(base.manifest_path)
    assert (
        manifest.extras["tier"] == "T4b"
        and manifest.extras["n"] == 3
        and manifest.extras["sample_seed"] == 0
    )
    assert {Path(a.path).name for a in manifest.outputs} >= {
        "results.jsonl",
        "discovery.json",
        "numbers.json",
        "plan.json",
    }
    summary = json.loads((base_dir / "discovery.json").read_text())
    assert (
        summary["model_label"] == "B0"
        and summary["sample_sha256"]
        and "DAF" not in summary["vendored"]
    )

    paired = run_stage(
        "eval.discovery", eval_settings, disc.run, model=tiny_mace.model_path, head="Default",
        sample=sample_path, atoms_zip=stub_wbm, energy_scale="mp", label="B3",
        baseline_run=base_dir, bootstrap_n=20,
    )  # fmt: skip
    assert paired.status == "ok", paired.summary
    assert paired.summary["delta_f1_vs_B0"] == 0.0  # same model twice
    assert base.summary["precision"] == 1.0 and base.summary["recall"] == 0.5
    numbers = json.loads((Path(paired.manifest_path).parent / "numbers.json").read_text())
    meta = numbers["eval.discovery.B3.delta_f1@meta"]
    assert numbers["eval.discovery.B3.delta_f1"] == 0.0 and meta["paired_vs"] == "B0"
    assert (
        meta["baseline_run_id"] == base.run_id
        and meta["prevalence"] == "natural"
        and meta["sample_seed"] == 0
    )
    assert meta["n"] == 3 and meta["ci95"] == [0.0, 0.0]
    assert numbers["eval.discovery.B3.n_ids"] == 3.0


def test_run_stage_resume_dry_run_and_failures(
    tiny_mace: Any,
    eval_settings: Settings,
    sample_path: Path,
    stub_wbm: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry = run_stage(
        "eval.discovery",
        eval_settings,
        disc.run,
        dry_run=True,
        model=tiny_mace.model_path,
        sample=sample_path,
        energy_scale="mp",
        limit=2,
    )
    assert dry.status == "partial" and dry.summary["n_ids"] == 2 and dry.outputs == []
    # a crash after the first id leaves results.jsonl with one record; --resume picks it up
    original = disc.evaluate_structure
    seen: list[str] = []

    def crashing(wbm_id: str, *args: Any, **kw: Any) -> dict[str, Any]:
        if len(seen) == 1:
            raise RuntimeError("boom")
        seen.append(wbm_id)
        return original(wbm_id, *args, **kw)

    monkeypatch.setattr(disc, "evaluate_structure", crashing)
    failed = run_stage(
        "eval.discovery",
        eval_settings,
        disc.run,
        seed=0,
        model=tiny_mace.model_path,
        head="Default",
        sample=sample_path,
        atoms_zip=stub_wbm,
        energy_scale="mp",
        bootstrap_n=10,
    )
    assert failed.status == "failed" and "boom" in failed.summary["error"]
    assert len((Path(failed.manifest_path).parent / "results.jsonl").read_text().splitlines()) == 1
    monkeypatch.setattr(disc, "evaluate_structure", original)
    resumed = run_stage(
        "eval.discovery",
        eval_settings,
        disc.run,
        seed=0,
        resume=True,
        model=tiny_mace.model_path,
        head="Default",
        sample=sample_path,
        atoms_zip=stub_wbm,
        energy_scale="mp",
        bootstrap_n=10,
    )
    assert (
        resumed.status == "ok"
        and resumed.run_id == failed.run_id
        and resumed.summary["n_resumed"] == 1
    )
    assert (
        wbm._test_calls[-1] == SAMPLE_IDS[1:]
    )  # only the missing ids were fetched  # type: ignore[attr-defined]

    def fail(**kw: Any) -> Any:
        kw.setdefault("head", "Default")
        return run_stage(
            "eval.discovery",
            eval_settings,
            disc.run,
            model=tiny_mace.model_path,
            sample=sample_path,
            atoms_zip=stub_wbm,
            **kw,
        )

    r = fail()
    assert r.status == "failed" and "energy scale unknown" in r.summary["error"]
    r = fail(energy_scale="mp", head="other")
    assert r.status == "failed" and "head" in r.summary["error"]
    r = fail(energy_scale="mp", head="pt_head")
    assert r.status == "failed" and "not 'pt_head'" in r.summary["error"]
    r = fail(energy_scale="mp", baseline_run=tmp_path / "nope")
    assert r.status == "failed"
    r = run_stage(
        "eval.discovery",
        eval_settings,
        disc.run,
        model=tmp_path / "absent.model",
        sample=sample_path,
    )
    assert r.status == "failed" and "model not found" in r.summary["error"]
    with pytest.raises(ValueError, match="not an offsets.json"):
        disc.read_offsets(_sample(tmp_path / "s2.json"))
    with pytest.raises(ValueError, match="not an ok eval.discovery run"):
        disc.load_baseline(Path(dry.manifest_path).parent)
