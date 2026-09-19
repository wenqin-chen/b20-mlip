"""dft e0s: isolated-atom units and the MACE --E0s output format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.config import Settings
from b20mlip.dft import e0s, qe
from b20mlip.dft.structures import elements_of
from b20mlip.executors import LocalExecutor
from b20mlip.provenance import read_manifest, run_stage


def test_isolated_atom_frame_and_plan(dft_settings: Settings, tmp_path: Path) -> None:
    frame = e0s.isolated_atom_frame("Fe", 12.0)
    assert (
        frame.numbers == [26]
        and frame.positions == [[6.0, 6.0, 6.0]]
        and frame.cell[1] == [0.0, 12.0, 0.0]
    )
    assert frame.compound == "Fe" and frame.info == {"isolated_atom": True, "box_A": 12.0}
    assert frame.config_type == "offset" and frame.parent_id == "E0_Fe"
    with pytest.raises(ValueError, match="unknown element"):
        e0s.isolated_atom_frame("Xx", 12.0)
    assert e0s.overrides_for("Mn") == {
        "nspin": 2,
        "starting_magnetization": {"Mn": 1.0},
        "kpoints": "gamma",
    }
    root = tmp_path / "e0s"
    ids = e0s.plan(dft_settings, ["Fe", "Si"], root)
    assert ids == ["E0_Fe", "E0_Si"] and qe.list_units(root) == ids
    text = (root / "E0_Si" / "pw.in").read_text()
    assert (
        "K_POINTS gamma" in text
        and "nspin = 2" in text
        and "starting_magnetization(1) = 1.00" in text
    )
    assert "nat = 1" in text and "ntyp = 1" in text and "12.0000000000" in text
    assert "ecutwfc = 90.0" in text and "smearing = 'mv'" in text
    assert elements_of(["FeSi", "CoSi", "MnSi", "FeGe"]) == ["Fe", "Si", "Co", "Mn", "Ge"]
    assert elements_of("MnGe") == ["Mn", "Ge"]
    with pytest.raises(ValueError, match="unknown element"):
        elements_of("Xx2")


def test_isolated_atoms_end_to_end(
    dft_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PW_MODE", "model")
    cfg = dft_settings
    out = tmp_path / "configs" / "E0s_qe.json"
    res = run_stage("dft.e0s", cfg, e0s.run, executor=LocalExecutor(cfg), out=out)
    assert res.status == "ok", res.summary
    assert res.summary["elements"] == "Fe,Si,Co,Mn,Ge" and res.summary["done"] == 5
    data = json.loads(out.read_text())
    assert list(data) == ["14", "25", "26", "27", "32"]  # atomic numbers, sorted, MACE --E0s format
    assert all(isinstance(v, float) for v in data.values())
    # single LJ atom in a box: E_LJ = 0, so E = 0.050 exp(-5) + 0.200 (Gamma: n_k = 1) eV
    assert data["26"] == pytest.approx(0.05 * 2.718281828**-5 + 0.2, abs=1e-6)
    assert res.summary["E0_Fe"] == data["26"]
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert (
        meta["units"] == "eV"
        and meta["settings"]["kpoints"] == "gamma"
        and meta["settings"]["nspin"] == 2
    )
    assert meta["settings"]["pseudo_md5s"]["Fe"] == "e86618425769142926afa95317d90200"
    assert (
        meta["elements"]["Mn"]["status"] == "done"
        and meta["elements"]["Mn"]["total_magnetization_muB"] == 1.0
    )
    manifest = read_manifest(res.manifest_path)
    assert manifest.extras["n_units"] == 5 and manifest.extras["n_failed"] == 0
    assert {Path(a.path).name for a in manifest.outputs} >= {
        "E0s_qe.json",
        "E0s_qe.meta.json",
        "counts.json",
    }
    root = cfg.paths.dft_dir / "e0s"
    assert qe.list_units(root) == ["E0_Fe", "E0_Si", "E0_Co", "E0_Mn", "E0_Ge"]

    # the plain API: plan + collect existing outputs without an executor
    energies = e0s.isolated_atoms(cfg, ["Fe", "Ge"], root=root)
    assert set(energies) == {"26", "32"} and energies["26"] == data["26"]
    assert e0s.isolated_atoms(cfg, ["Co"], root=tmp_path / "fresh") == {}  # planned, not run

    # one element unconverged -> failed, no file rewritten
    monkeypatch.setenv("FAKE_PW_FAIL_UNITS", "E0_Ge")
    (root / "E0_Ge" / ".done").unlink()
    out2 = tmp_path / "configs" / "E0s_2.json"
    res2 = run_stage(
        "dft.e0s", cfg, e0s.run, executor=LocalExecutor(cfg), out=out2, elements=["Fe", "Ge"]
    )
    assert res2.status == "failed" and res2.summary["missing"] == "Ge" and not out2.exists()
    dry = run_stage(
        "dft.e0s",
        cfg,
        e0s.run,
        executor=LocalExecutor(cfg),
        dry_run=True,
        out=out2,
        elements=["Co"],
        units=tmp_path / "dry",
    )
    assert (
        dry.status == "partial"
        and dry.summary["n_units"] == 1
        and qe.list_units(tmp_path / "dry") == ["E0_Co"]
    )


def test_e0s_cli(
    dft_settings: Settings,
    tmp_path: Path,
    fake_qe_cmd: str,
    monkeypatch: pytest.MonkeyPatch,
    cli_args,
) -> None:
    monkeypatch.setenv("FAKE_PW_MODE", "model")
    out = tmp_path / "E0s_qe.json"
    res = CliRunner().invoke(
        app,
        [
            *cli_args(tmp_path, fake_qe_cmd),
            "dft", "e0s", "--out", str(out), "--elements", "Si,Fe",
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["stage"] == "dft.e0s" and list(json.loads(out.read_text())) == ["14", "26"]
    assert str(out) in payload["outputs"]
