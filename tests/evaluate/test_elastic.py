"""Elastic constants / EOS: strain algebra, EMT Al physics, the tiny model on the fixture cell
(finite, symmetric tensor), the eval.elastic stage and its numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT

from b20mlip.config import Settings
from b20mlip.evaluate import elastic
from b20mlip.provenance import read_manifest, run_stage


def test_strain_tensor_and_deformation(fesi_atoms: Atoms) -> None:
    eps = elastic.strain_tensor([0.01, 0.0, 0.0, 0.02, 0.0, 0.0])
    assert (
        eps[0, 0] == 0.01 and eps[1, 2] == eps[2, 1] == 0.01 and eps[0, 1] == 0.0
    )  # engineering shear
    with pytest.raises(ValueError, match="6 components"):
        elastic.strain_tensor([0.01])
    stretched = elastic.strained(fesi_atoms, [0.01, 0, 0, 0, 0, 0])
    assert stretched.cell.lengths()[0] == pytest.approx(fesi_atoms.cell.lengths()[0] * 1.01)
    assert stretched.cell.lengths()[1] == pytest.approx(fesi_atoms.cell.lengths()[1])
    np.testing.assert_allclose(stretched.get_scaled_positions(), fesi_atoms.get_scaled_positions())
    sheared = elastic.strained(fesi_atoms, [0, 0, 0, 0.02, 0, 0])
    assert sheared.get_volume() == pytest.approx(fesi_atoms.get_volume(), rel=1e-3)
    assert (
        elastic.EV_A3_TO_GPA == pytest.approx(1.0 / units.GPa) and 160 < elastic.EV_A3_TO_GPA < 161
    )
    assert elastic._is_cubic(np.eye(3) * 4.0) and not elastic._is_cubic(np.diag([4.0, 4.0, 5.0]))


def test_emt_aluminium_is_physical() -> None:
    al = bulk("Al", "fcc", a=4.0, cubic=True)
    out = elastic.constants(al, EMT(), full=True)
    assert (
        out["C11"] > out["C12"] > 0
        and out["C44"] > 0
        and out["B"] == pytest.approx((out["C11"] + 2 * out["C12"]) / 3)
    )
    assert 20 < out["B"] < 60 and out["max_asymmetry_GPa"] < 0.5  # EMT Al: B ~ 35-40 GPa
    assert out["C11_yy"] == pytest.approx(out["C11"], abs=0.5) and out["C44_xz"] == pytest.approx(
        out["C44"], abs=0.5
    )
    assert (
        out["C12_yy"] == pytest.approx(out["C12_zz"], abs=0.5)
        and out["converged"] == 1.0
        and out["n_points"] == 24.0
    )
    fit = elastic.eos(al, EMT(), 8.0)
    assert (
        3.95 < fit["a0_A"] < 4.05
        and 30 < fit["B0_GPa"] < 50
        and fit["v0_in_range"]
        and fit["cubic"]
    )
    assert fit["residual_meV_atom"] < 0.5 and fit["n_points"] == 9 and len(fit["volumes_A3"]) == 9
    assert fit["B0_GPa"] == pytest.approx(out["B"], rel=0.25)  # EOS bulk modulus vs Cij within 25 %
    assert 1.0 < fit["B0_prime"] < 8.0 and fit["converged"]
    with pytest.raises(ValueError, match="positive"):
        elastic.constants(al, EMT(), strains=(0.0,))
    with pytest.raises(ValueError, match="pct"):
        elastic.eos(al, EMT(), 0.0)


def test_tiny_model_on_fixture_cell_finite_and_symmetric(fesi_atoms: Atoms, tiny_calc: Any) -> None:
    out = elastic.constants(fesi_atoms, tiny_calc, full=True)
    for key in ("C11", "C12", "C44", "B"):
        assert np.isfinite(out[key])
    C = np.array([[out[f"C{i + 1}{j + 1}_GPa"] for j in range(6)] for i in range(6)])
    assert C.shape == (6, 6) and np.all(np.isfinite(C))
    scale = max(1.0, float(np.max(np.abs(C))))
    assert out["max_asymmetry_GPa"] <= 0.02 * scale  # symmetric within 2 % of the largest entry
    tensor = elastic.elastic_tensor(fesi_atoms, tiny_calc, (0.01,), modes=(0,))
    assert (
        np.isnan(tensor["C_GPa"][3, 3])
        and np.isfinite(tensor["C_GPa"][0, 0])
        and len(tensor["points"]) == 2
    )
    stress, energy, converged = elastic.stress_gpa(fesi_atoms, tiny_calc)
    assert stress.shape == (6,) and np.isfinite(energy) and isinstance(converged, bool)


def test_run_stage_numbers(
    tiny_mace: Any, fesi_structure_path: Path, eval_settings: Settings
) -> None:
    result = run_stage(
        "eval.elastic", eval_settings, elastic.run, model=tiny_mace.model_path, compound="FeSi",
        structure=fesi_structure_path, label="B3", e0_source="estimated", strains=(0.005,),
        npoints=5,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    for key in ("C11_GPa", "C12_GPa", "C44_GPa", "B_GPa"):
        assert isinstance(result.summary[key], float)
    run_dir = Path(result.manifest_path).parent
    numbers = json.loads((run_dir / "numbers.json").read_text())
    keys = {k for k in numbers if not k.endswith("@meta")}
    expected = {f"eval.elastic.FeSi.B3.{m}" for m in ("C11", "C12", "C44", "B")}
    if "a0_A" in result.summary:  # the EOS of a random model may or may not fit
        expected |= {"eval.elastic.FeSi.B3.a0", "eval.elastic.FeSi.B3.B0"}
    assert keys == expected
    meta = numbers["eval.elastic.FeSi.B3.C11@meta"]
    assert meta["reference"] == {
        "code": "mace",
        "functional": "PBE",
        "pseudos": None,
        "e0_source": "estimated",
        "cross_functional": False,
    }
    assert (
        meta["head"] == "Default"
        and meta["n"] == 4
        and meta["seed"] == 0
        and meta["ci95"] is None
        and meta["ci95_reason"]
    )
    assert (
        meta["e0_source"] == "estimated"
        and meta["model_label"] == "B3"
        and meta["unit"] == "GPa"
        and meta["cell_source"] == "dft"
    )
    results = json.loads((run_dir / "elastic.json").read_text())
    assert results["constants"]["C11"] == numbers["eval.elastic.FeSi.B3.C11"] and results[
        "strains"
    ] == [0.005]
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["compound"] == "FeSi" and manifest.extras["reference"]["code"] == "mace"
    assert {Path(a.path).name for a in manifest.inputs} >= {"fesi.extxyz"}
    # model-relaxed cell, constants only
    result = run_stage(
        "eval.elastic",
        eval_settings,
        elastic.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        structure=fesi_structure_path,
        cell="model",
        do_eos=False,
        strains=(0.005,),
    )
    assert (
        result.status == "ok"
        and result.summary["cell_source"] == "model_relaxed"
        and "a0_A" not in result.summary
    )
    results = json.loads((Path(result.manifest_path).parent / "elastic.json").read_text())
    assert "a_model_A" in results and results["eos"] is None if "eos" in results else True


def test_run_eos_failure_is_recorded_not_fatal(
    tiny_mace: Any,
    fesi_structure_path: Path,
    eval_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr(elastic, "eos", broken)
    result = run_stage(
        "eval.elastic",
        eval_settings,
        elastic.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        structure=fesi_structure_path,
        do_constants=False,
    )
    assert result.status == "ok" and result.summary["eos"].startswith("failed: RuntimeError")
    run_dir = Path(result.manifest_path).parent
    assert json.loads((run_dir / "numbers.json").read_text()) == {}
    assert read_manifest(result.manifest_path).extras["eos_error"].startswith("RuntimeError")


def test_run_dry_run_and_failures(
    tiny_mace: Any, fesi_structure_path: Path, eval_settings: Settings, tmp_path: Path
) -> None:
    dry = run_stage(
        "eval.elastic",
        eval_settings,
        elastic.run,
        dry_run=True,
        model=tiny_mace.model_path,
        compound="FeSi",
        structure=fesi_structure_path,
    )
    assert dry.status == "partial" and dry.summary["planned"] == 1 and dry.outputs == []
    r = run_stage(
        "eval.elastic",
        eval_settings,
        elastic.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        structure=fesi_structure_path,
        cell="x",
    )
    assert r.status == "failed" and "cell must be" in r.summary["error"]
    r = run_stage(
        "eval.elastic", eval_settings, elastic.run, model=tmp_path / "absent.model", compound="FeSi"
    )
    assert r.status == "failed" and "model not found" in r.summary["error"]
    r = run_stage(
        "eval.elastic",
        eval_settings,
        elastic.run,
        model=tiny_mace.model_path,
        compound="MnGe",
        structure=fesi_structure_path,
    )
    assert r.status == "failed" and "no frame of compound" in r.summary["error"]
    r = run_stage(
        "eval.elastic", eval_settings, elastic.run, model=tiny_mace.model_path, compound="FeSi"
    )
    assert r.status == "failed" and "no structure file" in r.summary["error"]
