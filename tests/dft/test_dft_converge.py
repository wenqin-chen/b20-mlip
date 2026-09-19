"""dft converge: scan planning, the selection logic, and the end-to-end run with the fake pw.x."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.config import DFTThresholds, Settings
from b20mlip.dft import converge, qe
from b20mlip.executors import LocalExecutor
from b20mlip.io import write_frames
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage


def test_scan_points_and_plan(dft_settings: Settings, golden_frame: Frame, tmp_path: Path) -> None:
    points = converge.scan_points(dft_settings, ["Mn", "Si"], golden_frame.cell)
    assert len(points) == 24
    assert points[0] == {
        "ecut_ry": 40.0,
        "ecut_rho": 480.0,
        "k_spacing_inv_A": 0.35,
        "kmesh": [4, 4, 4],
    }
    assert points[-1] == {
        "ecut_ry": 90.0,
        "ecut_rho": 1080.0,
        "k_spacing_inv_A": 0.2,
        "kmesh": [7, 7, 7],
    }
    assert all(p["ecut_rho"] == 12 * p["ecut_ry"] for p in points)  # SSSP dual of Mn (GBRV USPP)
    assert (
        converge.unit_id("MnSi", 40, 0.35) == "MnSi_ec40_k0.35"
        and converge.unit_id("MnSi", 90.0, 0.2) == "MnSi_ec90_k0.2"
    )
    root = tmp_path / "scan"
    planned = converge.plan(dft_settings, "MnSi", root, golden_frame)
    assert [p["unit_id"] for p in planned][:2] == ["MnSi_ec40_k0.35", "MnSi_ec40_k0.3"]
    assert qe.list_units(root) == [p["unit_id"] for p in planned]
    text = (root / "MnSi_ec50_k0.25" / "pw.in").read_text()
    assert "ecutwfc = 50.0" in text and "ecutrho = 600.0" in text and "6 6 6 0 0 0" in text
    scan = json.loads((root / "scan.json").read_text())
    assert scan["compound"] == "MnSi" and len(scan["points"]) == 24
    assert (
        converge.cost(80, [6, 6, 6]) < converge.cost(90, [6, 6, 6]) < converge.cost(80, [7, 7, 7])
    )


def _point(
    uid: str,
    ecut: float,
    spacing: float,
    kmesh: list[int],
    e: float | None,
    f: float | None,
    converged: bool = True,
) -> dict:
    forces = None if f is None else [[f, 0.0, 0.0], [-f, 0.0, 0.0]]
    return {
        "unit_id": uid, "ecut_ry": ecut, "ecut_rho": 12 * ecut, "k_spacing_inv_A": spacing,
        "kmesh": kmesh, "converged": converged, "energy_atom_eV": e, "forces": forces,
    }  # fmt: skip


def test_analyse_selection_logic() -> None:
    ref_e, ref_f = -10.0, 0.100
    points = [
        _point("a_ec40_k0.3", 40, 0.3, [5, 5, 5], ref_e + 0.0200, ref_f + 0.020),  # E fails
        _point("a_ec60_k0.3", 60, 0.3, [5, 5, 5], ref_e + 0.0008, ref_f + 0.009),  # E ok, F fails
        _point(
            "a_ec60_k0.2", 60, 0.2, [7, 7, 7], ref_e + 0.0009, ref_f + 0.004
        ),  # meets, cost 60^1.5*343
        _point(
            "a_ec80_k0.3", 80, 0.3, [5, 5, 5], ref_e + 0.0003, ref_f + 0.002
        ),  # meets, cost 80^1.5*125 (cheapest)
        _point("a_ec80_k0.2", 80, 0.2, [7, 7, 7], ref_e + 0.0001, ref_f + 0.001),  # meets, dearer
        _point(
            "a_ec90_k0.3", 90, 0.3, [5, 5, 5], None, None, converged=False
        ),  # unconverged: excluded
        _point("a_ec90_k0.2", 90, 0.2, [7, 7, 7], ref_e, ref_f),  # densest = reference
    ]
    out = converge.analyse(points, (1.0, 5.0))
    assert out["reference_unit"] == "a_ec90_k0.2" and out["thresholds"] == {
        "E_meV_atom": 1.0,
        "F_meV_A": 5.0,
    }
    rows = {r["unit_id"]: r for r in out["points"]}
    assert (
        rows["a_ec40_k0.3"]["dE_meV_atom"] == pytest.approx(20.0)
        and rows["a_ec40_k0.3"]["meets"] is False
    )
    assert (
        rows["a_ec60_k0.3"]["dF_meV_A"] == pytest.approx(9.0)
        and rows["a_ec60_k0.3"]["meets"] is False
    )
    assert (
        rows["a_ec60_k0.2"]["meets"]
        and rows["a_ec80_k0.3"]["meets"]
        and rows["a_ec80_k0.2"]["meets"]
    )
    assert rows["a_ec90_k0.3"]["dE_meV_atom"] is None and rows["a_ec90_k0.3"]["meets"] is False
    assert rows["a_ec90_k0.2"]["dE_meV_atom"] == 0.0 and rows["a_ec90_k0.2"][
        "max_force_eVA"
    ] == pytest.approx(0.1)
    assert out["selected"]["unit_id"] == "a_ec80_k0.3"
    assert out["selected"]["ecut_ry"] == 80 and out["selected"]["k_spacing_inv_A"] == 0.3
    # tighter thresholds: only the reference itself qualifies
    tight = converge.analyse(points, (0.05, 0.5))
    assert tight["selected"]["unit_id"] == "a_ec90_k0.2"
    # impossible thresholds -> no selection
    assert converge.analyse(points, (-1.0, -1.0))["selected"] is None
    # explicit reference, and a reference without forces disables the force criterion
    alt = converge.analyse(points, (1.0, 5.0), reference="a_ec80_k0.2")
    assert alt["reference_unit"] == "a_ec80_k0.2" and alt["selected"]["unit_id"] == "a_ec80_k0.3"
    nof = [dict(p, forces=None) for p in points]
    assert converge.analyse(nof, (1.0, 5.0))["selected"] is None
    with pytest.raises(ValueError, match="missing or unconverged"):
        converge.analyse(points, (1.0, 5.0), reference="a_ec90_k0.3")
    with pytest.raises(ValueError, match="nothing to analyse"):
        converge.analyse([dict(p, converged=False) for p in points], (1.0, 5.0))


def test_converge_end_to_end_with_fake_pw(
    dft_settings: Settings, b20_frame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PW_MODE", "model")
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_PW_CALLS", str(calls))
    structure = tmp_path / "mnsi.extxyz"
    write_frames([b20_frame("MnSi")], structure)
    out = tmp_path / "configs" / "qe_MnSi_converged.json"
    cfg = dft_settings
    res = run_stage(
        "dft.converge",
        cfg,
        converge.run,
        executor=LocalExecutor(cfg),
        compound="MnSi",
        structure=structure,
        out=out,
    )
    assert res.status == "ok", res.summary
    assert len(calls.read_text().split()) == 24
    # fake model: dE = 50 meV exp(-(ecut-40)/10) + 200 meV / n_k relative to (90 Ry, 7x7x7)
    assert res.summary["selected_unit"] == "MnSi_ec80_k0.25"
    assert (
        res.summary["ecut_ry"] == 80.0
        and res.summary["ecut_rho"] == 960.0
        and res.summary["k_spacing_inv_A"] == 0.25
    )
    assert res.summary["reference_unit"] == "MnSi_ec90_k0.2" and res.summary["done"] == 24
    assert 0.9 < res.summary["dE_meV_atom"] < 1.0 and res.summary["dF_meV_A"] < 5.0
    data = json.loads(out.read_text())
    assert data["schema"] == "b20mlip.qe_converged.v1" and data["compound"] == "MnSi"
    assert (
        data["ecut_ry"] == 80.0
        and data["kmesh"] == [6, 6, 6]
        and data["source_run_id"] == res.run_id
    )
    assert len(data["points"]) == 24 and sum(p["meets"] for p in data["points"]) == 4
    assert {p["unit_id"] for p in data["points"] if p["meets"]} == {
        "MnSi_ec80_k0.25", "MnSi_ec90_k0.25", "MnSi_ec80_k0.2", "MnSi_ec90_k0.2"
    }  # fmt: skip
    assert data["thresholds"] == {"E_meV_atom": 1.0, "F_meV_A": 5.0}
    manifest = read_manifest(res.manifest_path)
    assert manifest.extras["selected"] == "MnSi_ec80_k0.25" and manifest.extras["n_units"] == 24
    assert {Path(a.path).name for a in manifest.outputs} >= {
        "converge.json",
        "qe_MnSi_converged.json",
        "counts.json",
    }
    assert manifest.inputs[0].path == str(structure)
    root = cfg.paths.dft_dir / "converge_MnSi"
    assert qe.list_units(root)[0] == "MnSi_ec40_k0.35"

    # --collect: re-analyse without running anything
    again = run_stage(
        "dft.converge",
        cfg,
        converge.run,
        executor=LocalExecutor(cfg),
        compound="MnSi",
        structure=structure,
        out=out,
        collect_only=True,
    )
    assert again.status == "ok" and again.summary["selected_unit"] == "MnSi_ec80_k0.25"
    assert len(calls.read_text().split()) == 24

    # thresholds nothing but the reference meets: the densest point is selected (dE = dF = 0)
    strict = cfg.model_copy(
        update={
            "dft": cfg.dft.model_copy(
                update={"thresholds": DFTThresholds(E_meV_atom=0.01, F_meV_A=0.001)}
            )
        }
    )
    res2 = run_stage(
        "dft.converge",
        strict,
        converge.run,
        executor=LocalExecutor(strict),
        compound="MnSi",
        structure=structure,
        out=out,
        units=root,
        collect_only=True,
    )
    assert res2.status == "ok" and res2.summary["selected_unit"] == "MnSi_ec90_k0.2"
    assert json.loads(out.read_text())["selected_unit"] == "MnSi_ec90_k0.2"
    assert sum(p["meets"] for p in json.loads(out.read_text())["points"]) == 1


def test_converge_dry_run_partial_and_cli(
    dft_settings: Settings,
    b20_frame,
    tmp_path: Path,
    fake_qe_cmd: str,
    monkeypatch: pytest.MonkeyPatch,
    cli_args,
) -> None:
    structure = tmp_path / "mnsi.extxyz"
    write_frames([b20_frame("MnSi")], structure)
    cfg = dft_settings
    dry = run_stage(
        "dft.converge",
        cfg,
        converge.run,
        executor=LocalExecutor(cfg),
        dry_run=True,
        compound="MnSi",
        structure=structure,
    )
    assert dry.status == "partial" and dry.summary["n_points"] == 24 and dry.outputs == []
    root = cfg.paths.dft_dir / "converge_MnSi"
    assert len(qe.list_units(root)) == 24
    # nothing has run: collect-only is partial with 24 missing
    part = run_stage(
        "dft.converge",
        cfg,
        converge.run,
        executor=LocalExecutor(cfg),
        compound="MnSi",
        structure=structure,
        collect_only=True,
    )
    assert part.status == "partial" and part.summary["missing"] == 24
    missing_structure = run_stage(
        "dft.converge",
        cfg,
        converge.run,
        executor=LocalExecutor(cfg),
        compound="MnSi",
        structure=tmp_path / "nope.extxyz",
    )
    assert missing_structure.status == "failed"

    monkeypatch.setenv("FAKE_PW_MODE", "model")
    runner = CliRunner()
    out = tmp_path / "conv.json"
    res = runner.invoke(
        app,
        [
            *cli_args(tmp_path, fake_qe_cmd),
            "dft", "converge", "--compound", "MnSi",
            "--structure", str(structure), "--out", str(out),
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert (
        payload["stage"] == "dft.converge"
        and payload["summary"]["selected_unit"] == "MnSi_ec80_k0.25"
    )
    assert str(out) in payload["outputs"]
