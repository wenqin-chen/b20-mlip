"""dft phonons: displacement frames, the force_sets.json schema (phonopy reads it back), e2e."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.config import Settings
from b20mlip.dft import phonons, qe
from b20mlip.executors import LocalExecutor
from b20mlip.io import write_frames
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage

SCHEMA_KEYS = {
    "schema", "compound", "unitcell", "supercell_matrix", "displacement_distance", "dataset",
    "reference", "units", "source_run_id",
}  # fmt: skip


def _lj_forces(frame: Frame) -> np.ndarray:
    atoms = Atoms(numbers=frame.numbers, positions=frame.positions, cell=frame.cell, pbc=True)
    atoms.calc = LennardJones(sigma=2.0, epsilon=0.5, rc=5.0, smooth=True)
    return np.asarray(atoms.get_forces())


def test_displacement_frames(b20_frame) -> None:
    ref = b20_frame("FeSi")
    frames, dataset = phonons.displacement_frames(ref, (2, 2, 2), 0.03)
    assert dataset["natom"] == 64 and len(dataset["first_atoms"]) == 4 == len(frames)
    assert [d["number"] for d in dataset["first_atoms"]] == [0, 0, 32, 32]
    assert dataset["first_atoms"][0]["displacement"] == pytest.approx([0.03, 0.0, 0.0])
    assert dataset["first_atoms"][1]["displacement"] == pytest.approx([-0.03, 0.0, 0.0])
    for i, f in enumerate(frames):
        assert len(f.numbers) == 64 and f.config_type == "phonon_disp" and f.compound == "FeSi"
        assert f.parent_id == ref.frame_id and f.group_id == f"FeSi/phonon_disp/{ref.frame_id}"
        assert (
            f.info["disp_index"] == i and f.info["disp_atom"] == dataset["first_atoms"][i]["number"]
        )
        assert f.info["supercell"] == [2, 2, 2] and f.info["displacement_distance"] == 0.03
        assert np.allclose(np.asarray(f.cell), 2 * np.asarray(ref.cell))
    assert len({f.frame_id for f in frames}) == 4
    moved = np.asarray(frames[0].positions) - np.asarray(frames[1].positions)
    assert np.allclose(moved[0], [0.06, 0.0, 0.0]) and np.allclose(moved[1:], 0.0)
    assert phonons.supercell_matrix((2, 2, 2)) == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert phonons.unit_ids_for("FeSi", 2) == ["FeSi_disp000", "FeSi_disp001"]
    with pytest.raises(ValueError, match="three positive integers"):
        phonons.supercell_matrix((2, 0, 2))
    with pytest.raises(ValueError, match="three positive integers"):
        phonons.displacement_frames(ref, (2, 2), 0.03)


def test_force_sets_schema_round_trip(b20_frame, dft_settings: Settings, tmp_path: Path) -> None:
    ref = b20_frame("FeSi")
    frames, dataset = phonons.displacement_frames(ref, (2, 2, 2), 0.03)
    forces = [_lj_forces(f) for f in frames]
    data = phonons.build_force_sets(
        ref, (2, 2, 2), 0.03, dataset, forces, compound="FeSi", run_id="run-1", cfg=dft_settings
    )
    assert set(data) == SCHEMA_KEYS and data["schema"] == "b20mlip.force_sets.v1"
    assert data["compound"] == "FeSi" and data["source_run_id"] == "run-1"
    assert (
        data["supercell_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        and data["displacement_distance"] == 0.03
    )
    assert (
        set(data["unitcell"]) == {"numbers", "positions", "cell"}
        and data["unitcell"]["numbers"] == ref.numbers
    )
    assert data["unitcell"]["positions"] == ref.positions and data["unitcell"]["cell"] == ref.cell
    assert set(data["dataset"]) == {"natom", "first_atoms"} and data["dataset"]["natom"] == 64
    first = data["dataset"]["first_atoms"][0]
    assert set(first) == {"number", "displacement", "forces"} and len(first["forces"]) == 64
    assert first["forces"] == forces[0].tolist()
    assert data["reference"] == {
        "code": "qe",
        "functional": "PBE",
        "pseudos": "SSSP-efficiency-1.3",
        "e0_source": None,
        "cross_functional": False,
    }
    assert data["units"] == {"forces": "eV/A", "positions": "A"}

    path = phonons.write_force_sets(data, tmp_path / "fs" / "force_sets.json")
    back = phonons.read_force_sets(path)
    assert back == json.loads(json.dumps(data))
    ph = phonons.phonopy_from_force_sets(back)
    assert ph.force_constants.shape == (64, 64, 3, 3)
    ph.symmetrize_force_constants()
    ph.run_qpoints([[0.0, 0.0, 0.0]])
    freqs = np.asarray(ph.qpoints.frequencies)[0]  # THz, sorted
    assert len(freqs) == 24
    acoustic = np.sort(np.abs(freqs))[:3]
    assert np.all(acoustic < 0.05), acoustic  # acoustic sum rule after symmetrisation
    (tmp_path / "bad.json").write_text(json.dumps({**data, "schema": "other"}))
    with pytest.raises(ValueError, match="schema"):
        phonons.read_force_sets(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="force sets for"):
        phonons.build_force_sets(
            ref, (2, 2, 2), 0.03, dataset, forces[:2], compound="FeSi", run_id="r", cfg=dft_settings
        )
    with pytest.raises(ValueError, match="shape"):
        phonons.build_force_sets(
            ref,
            (2, 2, 2),
            0.03,
            dataset,
            [f[:8] for f in forces],
            compound="FeSi",
            run_id="r",
            cfg=dft_settings,
        )


def test_phonons_end_to_end_with_fake_pw(
    dft_settings: Settings, b20_frame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PW_MODE", "model")
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_PW_CALLS", str(calls))
    structure = tmp_path / "fesi.extxyz"
    write_frames([b20_frame("FeSi")], structure)
    cfg = dft_settings
    out = tmp_path / "data" / "phonons" / "force_sets_FeSi.json"
    res = run_stage(
        "dft.phonons",
        cfg,
        phonons.run,
        executor=LocalExecutor(cfg),
        compound="FeSi",
        supercell=(2, 2, 2),
        distance=0.03,
        structure=structure,
        out=out,
    )
    assert res.status == "ok", res.summary
    assert (
        res.summary["n_displacements"] == 4
        and res.summary["natom_supercell"] == 64
        and res.summary["done"] == 4
    )
    assert calls.read_text().split() == [
        "FeSi_disp000",
        "FeSi_disp001",
        "FeSi_disp002",
        "FeSi_disp003",
    ]
    data = phonons.read_force_sets(out)
    assert (
        set(data) == SCHEMA_KEYS
        and data["source_run_id"] == res.run_id
        and data["compound"] == "FeSi"
    )
    local = Path(res.manifest_path).parent / "force_sets.json"
    assert local.is_file() and json.loads(local.read_text()) == data
    # the fake pw.x labels with LJ forces: what QE "printed" round-trips through the parser
    frames, _ = phonons.displacement_frames(b20_frame("FeSi"), (2, 2, 2), 0.03)
    expected = _lj_forces(frames[0]) * (1 + 0.001 * np.exp(-5.0))
    assert np.allclose(np.asarray(data["dataset"]["first_atoms"][0]["forces"]), expected, atol=2e-5)
    root = cfg.paths.dft_dir / "phonons_FeSi"
    plan = json.loads((root / "phonon_plan.json").read_text())
    assert plan["units"] == qe.list_units(root) and plan["dataset"]["natom"] == 64
    text = (root / "FeSi_disp000" / "pw.in").read_text()
    assert (
        "nat = 64" in text and "nspin = 1" in text and "3 3 3 0 0 0" in text
    )  # 0.25/A on the 8.96 A supercell
    ph = phonons.phonopy_from_force_sets(data)
    assert ph.force_constants.shape == (64, 64, 3, 3)
    manifest = read_manifest(res.manifest_path)
    assert (
        manifest.extras["n_units"] == 4
        and manifest.extras["supercell"] == [2, 2, 2]
        and manifest.extras["pw_version"] == "7.3"
    )
    assert {Path(a.path).name for a in manifest.outputs} >= {
        "force_sets.json",
        "force_sets_FeSi.json",
    }

    # one unit lost: collect-only cannot assemble the set (partial), rerun repairs it
    (root / "FeSi_disp002" / "pw.out").unlink()
    (root / "FeSi_disp002" / ".done").unlink()
    part = run_stage(
        "dft.phonons",
        cfg,
        phonons.run,
        executor=LocalExecutor(cfg),
        compound="FeSi",
        structure=structure,
        out=out,
        collect_only=True,
    )
    assert (
        part.status == "partial"
        and part.summary["first_missing_unit"] == "FeSi_disp002"
        and part.summary["missing"] == 1
    )
    fixed = run_stage(
        "dft.phonons",
        cfg,
        phonons.run,
        executor=LocalExecutor(cfg),
        compound="FeSi",
        structure=structure,
        out=out,
    )
    assert fixed.status == "ok" and len(calls.read_text().split()) == 5
    monkeypatch.setenv("FAKE_PW_FAIL_UNITS", "FeSi_disp001")
    (root / "FeSi_disp001" / ".done").unlink()
    failed = run_stage(
        "dft.phonons",
        cfg,
        phonons.run,
        executor=LocalExecutor(cfg),
        compound="FeSi",
        structure=structure,
        out=out,
    )
    assert failed.status == "failed" and failed.summary["unconverged"] == 1


def test_phonons_dry_run_and_cli(
    dft_settings: Settings, b20_frame, tmp_path: Path, fake_qe_cmd: str, cli_args
) -> None:
    structure = tmp_path / "cosi.extxyz"
    write_frames([b20_frame("CoSi")], structure)
    cfg = dft_settings
    dry = run_stage(
        "dft.phonons",
        cfg,
        phonons.run,
        executor=LocalExecutor(cfg),
        dry_run=True,
        compound="CoSi",
        structure=structure,
        supercell=(1, 1, 1),
    )
    assert dry.status == "partial" and dry.summary["natom_supercell"] == 8 and dry.outputs == []
    assert (
        len(qe.list_units(cfg.paths.dft_dir / "phonons_CoSi"))
        == dry.summary["n_displacements"]
        == 4
    )
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            *cli_args(tmp_path, fake_qe_cmd),
            "--dry-run", "dft", "phonons", "--compound", "CoSi", "--supercell", "1", "1", "1",
            "--distance", "0.02", "--structure", str(structure),
        ],
    )  # fmt: skip
    assert (
        res.exit_code == 1 and '"status": "partial"' in res.output and '"dft.phonons"' in res.output
    )
    frames = json.loads(
        (tmp_path / "dft" / "phonons_CoSi" / "CoSi_disp000" / "unit.json").read_text()
    )
    assert frames["frame"]["info"]["displacement_distance"] == 0.02
