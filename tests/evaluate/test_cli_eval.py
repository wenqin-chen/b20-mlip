"""`b20mlip eval errors|discovery|phonons|elastic` through the root CLI, and the report tier's
harvest + honesty gates on what they publish."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from b20mlip.cli import REGISTERED, app
from b20mlip.data import wbm
from b20mlip.evaluate.cli import COMMANDS
from b20mlip.io import frame_to_atoms
from b20mlip.models import Frame
from b20mlip.phonons import harmonic
from b20mlip.report import audit
from b20mlip.report import numbers as nums

runner = CliRunner()


def _payload(output: str) -> dict[str, Any]:
    """The StageResult JSON (torch/MACE warnings may surround it in the captured output)."""
    obj, _ = json.JSONDecoder().raw_decode(output[output.index("{") :])
    return obj


def test_registration_and_help() -> None:
    assert (
        REGISTERED["eval"]
        == set(COMMANDS)
        == {"aggregate", "errors", "discovery", "phonons", "elastic"}
    )
    for name, needle in (
        ("errors", "--tiers"),
        ("discovery", "--baseline-run"),
        ("phonons", "--reference"),
        ("elastic", "--strains"),
    ):
        result = runner.invoke(app, ["eval", name, "--help"])
        assert result.exit_code == 0 and needle in result.output and "[stub]" not in result.output
    assert runner.invoke(app, ["eval", "errors"]).exit_code == 2  # --model is required


def test_errors_cli_end_to_end_and_report_harvest(
    tiny_mace: Any, qe_frames_path: Path, overlay: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["--config", str(overlay), "--seed", "1", "eval", "errors",
         "--model", str(tiny_mace.model_path), "--frames", str(qe_frames_path), "--tier", "T0",
         "--energy-scale", "qe", "--label", "B1", "--e0-source", "E0s_qe.json",
         "--bootstrap-n", "30"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["stage"] == "eval.errors" and payload["status"] == "ok"
    assert (
        payload["summary"]["T0.energy_metrics"] == "ok"
        and payload["summary"]["model_label"] == "B1"
    )
    assert any(p.endswith("numbers.json") for p in payload["outputs"])
    runs_dir = tmp_path / "runs"
    harvest = nums.harvest(runs_dir)
    assert harvest.n_ok == 1 and harvest.stale == []
    keys = set(harvest.entries)
    assert {"eval.errors.T0.B1.mae_f", "eval.errors.T0.B1.mae_e", "eval.errors.T0.B1.mae_s"} <= keys
    entry = harvest.entries["eval.errors.T0.B1.mae_f"]
    assert (
        entry.run_id == payload["run_id"]
        and entry.meta["seed"] == 1
        and entry.meta["reference"]["code"] == "qe"
    )
    readme = tmp_path / "README.md"
    readme.write_text("# t\n\nno numbers here\n", encoding="utf-8")
    violations = audit.run(readme, harvest.as_json(), runs_dir)
    assert [v for v in violations if v.startswith(("A2:", "A3:", "A4:", "A8:", "A9:"))] == []
    # a dry run exits 1 with a partial manifest
    result = runner.invoke(
        app,
        [
            "--config",
            str(overlay),
            "--dry-run",
            "eval",
            "errors",
            "--model",
            str(tiny_mace.model_path),
            "--frames",
            str(qe_frames_path),
        ],
    )
    assert result.exit_code == 1 and _payload(result.output)["status"] == "partial"


def test_discovery_cli_with_stubbed_wbm(
    tiny_mace: Any,
    tiny_frames: list[Frame],
    overlay: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = ["wbm-1-16845", "wbm-1-55036"]
    cells = {}
    for wbm_id, compound in zip(ids, ("FeSi", "CoSi"), strict=True):
        frame = next(f for f in tiny_frames if f.compound == compound and f.config_type == "relax")
        atoms = frame_to_atoms(frame)
        atoms.calc = None
        cells[wbm_id] = atoms
    monkeypatch.setattr(wbm, "atoms_for_ids", lambda z, i: {k: cells[k].copy() for k in i})
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({
        "n": 2, "seed": 0, "ids": ids, "prevalence": 0.5, "in_family_ids": ids,
        "truth": {ids[0]: {"e_form_per_atom": 5.0, "e_above_hull": -0.01},
                  ids[1]: {"e_form_per_atom": -0.2, "e_above_hull": 0.05}},
        "checksums": {},
    }))  # fmt: skip
    zip_path = tmp_path / "atoms.zip"
    zip_path.write_bytes(b"")
    common = ["--config", str(overlay), "eval", "discovery", "--model", str(tiny_mace.model_path),
              "--head", "Default", "--sample", str(sample), "--atoms-zip", str(zip_path),
              "--energy-scale", "mp", "--bootstrap-n", "20"]  # fmt: skip
    base = runner.invoke(app, [*common, "--label", "B0"])
    assert base.exit_code == 0, base.output
    base_payload = _payload(base.output)
    assert base_payload["summary"]["n_ids"] == 2 and base_payload["summary"]["precision"] == 1.0
    base_dir = Path(base_payload["manifest"]).parent
    paired = runner.invoke(app, [*common, "--label", "B2", "--baseline-run", str(base_dir)])
    assert paired.exit_code == 0, paired.output
    payload = _payload(paired.output)
    assert payload["summary"]["delta_f1_vs_B0"] == 0.0
    numbers = json.loads((Path(payload["manifest"]).parent / "numbers.json").read_text())
    assert numbers["eval.discovery.B2.delta_f1@meta"]["paired_vs"] == "B0"
    harvest = nums.harvest(tmp_path / "runs")
    readme = tmp_path / "README.md"
    readme.write_text("# t\n", encoding="utf-8")
    violations = audit.run(readme, harvest.as_json(), tmp_path / "runs")
    assert [v for v in violations if v.startswith(("A2:", "A3:", "A8:", "A9:"))] == []
    a4 = [v for v in violations if v.startswith("A4:")]
    assert (
        len(a4) == 1 and "n != 1000" in a4[0] and "delta_f1" in a4[0]
    )  # a 2-id smoke run is not the protocol
    assert "daf" not in json.dumps(harvest.as_json()).lower()
    # --limit and a missing archive
    limited = runner.invoke(app, [*common, "--label", "B0", "--limit", "1"])
    assert limited.exit_code == 0 and _payload(limited.output)["summary"]["n_ids"] == 1
    missing = runner.invoke(
        app, [*common[:-4], "--energy-scale", "mp", "--atoms-zip", str(tmp_path / "nope.zip")]
    )
    assert (
        missing.exit_code == 1
        and "archive not found" in _payload(missing.output)["summary"]["error"]
    )


def test_phonons_and_elastic_cli(
    tiny_mace: Any,
    tiny_calc: Any,
    fesi_structure_path: Path,
    tiny_frames: list[Frame],
    overlay: Path,
    tmp_path: Path,
) -> None:
    frame = next(f for f in tiny_frames if f.compound == "FeSi" and f.config_type == "relax")
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    phonon = harmonic.new_phonopy(atoms, 1)
    phonon.generate_displacements(distance=0.03)
    forces = []
    for cell in phonon.supercells_with_displacements:
        sc = harmonic.from_phonopy(cell)
        sc.calc = tiny_calc
        forces.append(sc.get_forces())
    phonon.forces = forces
    ref = {"code": "qe", "functional": "PBE", "pseudos": "SSSP-efficiency-1.3", "e0_source": None}
    fs = harmonic.write_force_sets(
        harmonic.force_sets_document(atoms, phonon, reference=ref), tmp_path / "fs.json"
    )
    result = runner.invoke(
        app, ["--config", str(overlay), "eval", "phonons", "--model", str(tiny_mace.model_path),
              "--compound", "FeSi", "--reference", "qe", "--force-sets", str(fs),
              "--npoints", "20", "--mesh", "0", "--label", "B1"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert (
        payload["summary"]["omega_mae_meV"] == 0.0 and payload["summary"]["softening_index"] == 1.0
    )
    result = runner.invoke(
        app,
        [
            "--config",
            str(overlay),
            "eval",
            "phonons",
            "--model",
            str(tiny_mace.model_path),
            "--compound",
            "FeSi",
            "--reference",
            "phonondb103",
            "--structure",
            str(fesi_structure_path),
        ],
    )
    assert result.exit_code == 1 and "not downloaded" in _payload(result.output)["summary"]["error"]

    result = runner.invoke(
        app, ["--config", str(overlay), "eval", "elastic", "--model", str(tiny_mace.model_path),
              "--compound", "FeSi", "--structure", str(fesi_structure_path), "--strains", "0.005",
              "--npoints", "5", "--label", "B1"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert (
        isinstance(payload["summary"]["C11_GPa"], float)
        and payload["summary"]["cell_source"] == "dft"
    )
    bad = runner.invoke(
        app,
        [
            "--config",
            str(overlay),
            "eval",
            "elastic",
            "--model",
            str(tiny_mace.model_path),
            "--compound",
            "FeSi",
            "--strains",
            "x",
        ],
    )
    assert bad.exit_code == 2 and "comma list of floats" in bad.output
    harvest = nums.harvest(tmp_path / "runs")
    assert {"eval.phonons.FeSi.B1.omega_mae_meV", "eval.elastic.FeSi.B1.C11"} <= set(
        harvest.entries
    )
    readme = tmp_path / "README.md"
    readme.write_text("# t\n", encoding="utf-8")
    violations = audit.run(readme, harvest.as_json(), tmp_path / "runs")
    assert [v for v in violations if v.startswith(("A2:", "A3:", "A4:", "A5:", "A8:", "A9:"))] == []
