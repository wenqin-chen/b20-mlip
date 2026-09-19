"""phonon_compare on PhononResults from harmonic.compute: identical -> 0/1, softened -> s < 1,
incompatible inputs raise; the eval.phonons stage for qe / pbesol / phonondb103 references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from b20mlip.config import Settings
from b20mlip.evaluate import phonon_compare as pc
from b20mlip.models import PhononResult, Reference
from b20mlip.phonons import harmonic
from b20mlip.provenance import read_manifest, run_stage

QE_REF = Reference(code="qe", functional="PBE", pseudos="SSSP-efficiency-1.3", e0_source=None)
NPTS = 24


@pytest.fixture(scope="module")
def model_result(tiny_calc: Any, tiny_frames: list) -> PhononResult:
    from b20mlip.io import frame_to_atoms

    frame = next(f for f in tiny_frames if f.compound == "FeSi" and f.config_type == "relax")
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    return harmonic.compute(
        atoms, tiny_calc, 1, 0.03, cell_source="dft", npoints=NPTS, mesh=None, run_id="m"
    )


def _scaled(result: PhononResult, factor: float, **update: Any) -> PhononResult:
    freqs = (np.asarray(result.frequencies_meV) * factor).tolist()
    return result.model_copy(
        update={"frequencies_meV": freqs, "reference": QE_REF, "source_label": "qe", **update}
    )


def _force_sets(atoms: Atoms, calc: Any, path: Path, scale: float = 1.0) -> Path:
    phonon = harmonic.new_phonopy(atoms, 1)
    phonon.generate_displacements(distance=0.03)
    forces = []
    for cell in phonon.supercells_with_displacements:
        sc = harmonic.from_phonopy(cell)
        sc.calc = calc
        forces.append(sc.get_forces() * scale)
    phonon.forces = forces
    doc = harmonic.force_sets_document(atoms, phonon, reference=QE_REF, source_run_id="qe-run")
    return harmonic.write_force_sets(doc, path)


def test_identical_results_give_zero_mae_and_unit_index(model_result: PhononResult) -> None:
    ref = model_result.model_copy(update={"reference": QE_REF, "source_label": "qe"})
    assert pc.omega_mae(model_result, ref) == 0.0
    assert pc.softening_index(model_result, ref) == 1.0
    assert pc.imaginary_delta(model_result, ref) == 0
    compared = pc.compare(model_result, ref)
    assert compared.omega_mae_meV == 0.0 and compared.softening_index == 1.0
    assert compared.reference == QE_REF and compared.source_label == "mace"
    assert compared.frequencies_meV == model_result.frequencies_meV
    s = pc.summary(model_result, ref)
    assert s["n_qpoints"] == NPTS and s["n_branches"] == 24 and s["omega_max_abs_delta_meV"] == 0.0
    assert s["reference"]["code"] == "qe" and s["cell_source"] == "dft"


def test_softened_reference_scaling(model_result: PhononResult) -> None:
    ref = _scaled(model_result, 1.0)
    soft = _scaled(model_result, 0.9)  # model 10 % softer than the reference
    assert pc.softening_index(soft, ref) == pytest.approx(0.9)
    assert pc.omega_mae(soft, ref) == pytest.approx(
        0.1 * np.mean(np.abs(np.asarray(ref.frequencies_meV)))
    )
    assert pc.softening_index(_scaled(model_result, 1.2), ref) == pytest.approx(1.2)
    # branches are re-sorted before pairing; tiny reference frequencies are excluded from s
    shuffled = ref.model_copy(
        update={"frequencies_meV": [list(reversed(f)) for f in ref.frequencies_meV]}
    )
    assert pc.omega_mae(shuffled, ref) == 0.0
    tiny = ref.model_copy(
        update={"frequencies_meV": (np.asarray(ref.frequencies_meV) * 1e-3).tolist()}
    )
    with pytest.raises(ValueError, match="softening index undefined"):
        pc.softening_index(model_result, tiny)


def test_incompatible_inputs_raise(model_result: PhononResult) -> None:
    ref = _scaled(model_result, 1.0)
    fewer = ref.model_copy(
        update={
            "qpoints": ref.qpoints[:-1],
            "frequencies_meV": ref.frequencies_meV[:-1],
            "qpath_labels": ref.qpath_labels[:-1],
        }
    )
    with pytest.raises(ValueError, match="q-point count differs"):
        pc.omega_mae(model_result, fewer)
    moved = ref.model_copy(update={"qpoints": [[q[0] + 0.01, q[1], q[2]] for q in ref.qpoints]})
    with pytest.raises(ValueError, match="q-points differ"):
        pc.omega_mae(model_result, moved)
    bigger = ref.model_copy(update={"frequencies_meV": [f + [1.0] for f in ref.frequencies_meV]})
    with pytest.raises(ValueError, match="branch count differs"):
        pc.softening_index(model_result, bigger)
    empty = ref.model_copy(update={"frequencies_meV": [], "qpoints": [], "qpath_labels": []})
    with pytest.raises(ValueError, match="non-empty"):
        pc.omega_mae(model_result, empty)
    short = ref.model_copy(update={"frequencies_meV": ref.frequencies_meV[:-1]})
    with pytest.raises(ValueError, match="disagree in length"):
        pc.check_comparable(model_result, short)


def test_load_reference_round_trip(model_result: PhononResult, tmp_path: Path) -> None:
    path = harmonic.to_json(_scaled(model_result, 1.0), tmp_path / "ref.json")
    assert pc.load_reference(path) == _scaled(model_result, 1.0)
    pbesol = harmonic.to_json(
        _scaled(
            model_result,
            1.0,
            reference=QE_REF.model_copy(update={"code": "abinit", "functional": "PBEsol"}),
        ),
        tmp_path / "pbesol.json",
    )
    loaded = pc.load_labelled_reference("pbesol", pbesol)
    assert loaded.reference.cross_functional is True and loaded.reference.functional == "PBEsol"
    assert pc.load_labelled_reference("phonondb103", path).reference.cross_functional is False
    out = pc.write_summary({"a": 1}, tmp_path / "sub" / "s.json")
    assert json.loads(out.read_text()) == {"a": 1}


# --- the stage ------------------------------------------------------------------------------------


def test_run_qe_reference_end_to_end(
    tiny_mace: Any, tiny_calc: Any, fesi_atoms: Atoms, eval_settings: Settings, tmp_path: Path
) -> None:
    fs = _force_sets(fesi_atoms, tiny_calc, tmp_path / "force_sets.json")
    atoms, matrix = pc.cell_from_force_sets(fs)
    assert len(atoms) == 8 and np.array_equal(matrix, np.eye(3, dtype=int))
    result = run_stage(
        "eval.phonons", eval_settings, pc.run, model=tiny_mace.model_path, compound="FeSi",
        reference="qe", force_sets=fs, npoints=NPTS, mesh=None, label="B1", e0_source="E0s_qe.json",
    )  # fmt: skip
    assert result.status == "ok", result.summary
    assert result.summary["omega_mae_meV"] == 0.0 and result.summary["softening_index"] == 1.0
    assert result.summary["cell_source"] == "dft" and result.summary["reference"] == "qe"
    run_dir = Path(result.manifest_path).parent
    numbers = json.loads((run_dir / "numbers.json").read_text())
    assert set(k for k in numbers if not k.endswith("@meta")) == {
        "eval.phonons.FeSi.B1.omega_mae_meV", "eval.phonons.FeSi.B1.softening_index",
        "eval.phonons.FeSi.B1.imaginary_count",
    }  # fmt: skip
    meta = numbers["eval.phonons.FeSi.B1.omega_mae_meV@meta"]
    assert (
        meta["reference"] == QE_REF.model_dump(mode="json")
        and meta["n"] == NPTS
        and meta["seed"] == 0
    )
    assert meta["ci95"] is None and meta["ci95_reason"] and meta["head"] == "Default"
    assert (
        meta["e0_source"] == "E0s_qe.json"
        and meta["model_label"] == "B1"
        and meta["cross_functional"] is False
    )
    assert meta["supercell"] == [1, 1, 1] and meta["cell_source"] == "dft" and meta["unit"] == "meV"
    model_json = harmonic.from_json(run_dir / "phonons_model.json")
    assert (
        model_json.omega_mae_meV == 0.0
        and model_json.reference == QE_REF
        and model_json.run_id == result.run_id
    )
    ref_json = harmonic.from_json(run_dir / "phonons_reference.json")
    assert ref_json.source_label == "qe:PBE" and ref_json.run_id == "qe-run"
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["reference_label"] == "qe" and manifest.extras["n"] == NPTS
    assert {Path(a.path).name for a in manifest.outputs} >= {
        "phonons_model.json",
        "phonons_reference.json",
        "phonon_compare.json",
        "numbers.json",
    }
    # a softened reference (forces x 0.81 -> omega x 0.9) is detected
    softer = _force_sets(fesi_atoms, tiny_calc, tmp_path / "soft.json", scale=1 / 0.81)
    result = run_stage(
        "eval.phonons",
        eval_settings,
        pc.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        reference="qe",
        force_sets=softer,
        npoints=NPTS,
        mesh=None,
    )
    assert result.status == "ok" and result.summary["softening_index"] == pytest.approx(
        0.9, abs=1e-6
    )
    assert result.summary["omega_mae_meV"] > 0
    # the model-relaxed cell path
    result = run_stage(
        "eval.phonons",
        eval_settings,
        pc.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        reference="qe",
        force_sets=fs,
        cell="model",
        npoints=NPTS,
        mesh=None,
    )
    assert result.status == "ok" and result.summary["cell_source"] == "model_relaxed"
    assert (
        json.loads((Path(result.manifest_path).parent / "phonon_compare.json").read_text())[
            "relax_steps"
        ]
        is not None
    )


def test_run_pbesol_and_phonondb103_references(
    tiny_mace: Any,
    model_result: PhononResult,
    fesi_structure_path: Path,
    eval_settings: Settings,
    tmp_path: Path,
) -> None:
    ref_json = harmonic.to_json(
        _scaled(
            model_result,
            1.0,
            reference=Reference(code="abinit", functional="PBEsol", pseudos="PAW", e0_source=None),
        ),
        tmp_path / "pbesol.json",
    )
    result = run_stage(
        "eval.phonons", eval_settings, pc.run, model=tiny_mace.model_path, compound="FeSi",
        reference="pbesol", reference_json=ref_json, structure=fesi_structure_path,
        supercell=(1, 1, 1), mesh=None, label="B0",
    )  # fmt: skip
    assert result.status == "ok", result.summary
    numbers = json.loads((Path(result.manifest_path).parent / "numbers.json").read_text())
    assert "eval.phonons.FeSi.B0.omega_mae_meV_pbesol" in numbers
    meta = numbers["eval.phonons.FeSi.B0.omega_mae_meV_pbesol@meta"]
    assert meta["cross_functional"] is True and meta["reference"]["cross_functional"] is True
    assert meta["reference"]["functional"] == "PBEsol" and meta["reference"]["code"] == "abinit"
    assert numbers["eval.phonons.FeSi.B0.omega_mae_meV_pbesol"] == pytest.approx(
        0.0, abs=1e-6
    )  # extxyz rounding
    # phonondb103 without a reference JSON fails with the explicit "not downloaded" message
    result = run_stage(
        "eval.phonons",
        eval_settings,
        pc.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        reference="phonondb103",
        structure=fesi_structure_path,
    )
    assert result.status == "failed" and "not downloaded" in result.summary["error"]
    vasp = harmonic.to_json(
        _scaled(
            model_result,
            1.0,
            reference=Reference(code="vasp", functional="PBE", pseudos="PAW", e0_source=None),
        ),
        tmp_path / "phdb.json",
    )
    result = run_stage(
        "eval.phonons",
        eval_settings,
        pc.run,
        model=tiny_mace.model_path,
        compound="FeSi",
        reference="phonondb103",
        reference_json=vasp,
        structure=fesi_structure_path,
        mesh=None,
    )
    assert result.status == "ok"
    numbers = json.loads((Path(result.manifest_path).parent / "numbers.json").read_text())
    assert "eval.phonons.FeSi.tiny_b20.softening_index_phonondb103" in numbers
    assert (
        numbers["eval.phonons.FeSi.tiny_b20.softening_index_phonondb103@meta"]["reference"]["code"]
        == "vasp"
    )


def test_run_failures_dry_run_and_force_set_lookup(
    tiny_mace: Any, tiny_calc: Any, fesi_atoms: Atoms, eval_settings: Settings, tmp_path: Path
) -> None:
    fs = _force_sets(fesi_atoms, tiny_calc, tmp_path / "force_sets.json")
    dry = run_stage(
        "eval.phonons",
        eval_settings,
        pc.run,
        dry_run=True,
        model=tiny_mace.model_path,
        compound="FeSi",
        force_sets=fs,
    )
    assert dry.status == "partial" and dry.summary["planned"] == 1 and dry.outputs == []

    def fail(**kw: Any) -> Any:
        kw.setdefault("compound", "FeSi")
        return run_stage("eval.phonons", eval_settings, pc.run, model=tiny_mace.model_path, **kw)

    r = fail(reference="dfpt")
    assert r.status == "failed" and "reference must be" in r.summary["error"]
    r = fail(cell="both", force_sets=fs)
    assert r.status == "failed" and "cell must be" in r.summary["error"]
    r = fail(reference="pbesol")
    assert r.status == "failed" and "--reference-json" in r.summary["error"]
    r = fail(force_sets=tmp_path / "missing.json")
    assert r.status == "failed" and "not found" in r.summary["error"]
    r = fail()  # no default force sets under the tmp data dir and no dft.phonons runs
    assert r.status == "failed" and "no QE force sets for FeSi" in r.summary["error"]
    r = fail(compound="CoSi", force_sets=fs)
    assert r.status == "failed" and "force sets are for FeSi" in r.summary["error"]
    r = run_stage(
        "eval.phonons", eval_settings, pc.run, model=tmp_path / "absent.model", compound="FeSi"
    )
    assert r.status == "failed" and "model not found" in r.summary["error"]
    # the default location <data_dir>/phonons/force_sets_<compound>.json is found
    default = Path(eval_settings.paths.data_dir) / "phonons" / "force_sets_FeSi.json"
    default.parent.mkdir(parents=True)
    default.write_text(fs.read_text())
    assert pc.find_force_sets(eval_settings, "FeSi") == default
    with pytest.raises(ValueError, match="schema"):
        pc.cell_from_force_sets(_write(tmp_path / "bad.json", {"schema": "x"}))


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path
