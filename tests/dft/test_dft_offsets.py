"""dft offsets: pairing by frame_id, the per-element fit (exact and noisy), the residual gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.config import Settings
from b20mlip.dft import offsets
from b20mlip.dft.structures import strip_labels
from b20mlip.executors import LocalExecutor
from b20mlip.io import frame_id_for, write_frames
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage

TRUE = {"Fe": 1.5, "Si": -0.75, "Co": 2.25, "Mn": 0.5, "Ge": -1.0}


def _pairs(b20_frame, *, noise: float = 0.0, seed: int = 0) -> tuple[list[Frame], list[Frame]]:
    """QE/MP frame pairs with E_mp = E_qe + Σ n_e c_e (+ noise); one off-stoichiometric frame."""
    rng = np.random.default_rng(seed)
    bases = [b20_frame(c, config_type="offset") for c in ("FeSi", "CoSi", "MnSi", "FeGe")]
    bases.append(b20_frame("FeSi", config_type="rattle", rattle=0.02))
    vac = bases[-1]
    vac = vac.model_copy(  # Fe3Si4 vacancy: off-stoichiometric -> 5 independent element columns
        update={
            "numbers": vac.numbers[1:],
            "positions": vac.positions[1:],
            "frame_id": frame_id_for(vac.numbers[1:], vac.positions[1:], vac.cell),
        }
    )
    bases.append(vac)
    qe_frames, mp_frames = [], []
    from ase.data import chemical_symbols

    for i, base in enumerate(bases):
        e_qe = -1000.0 - 3.0 * i
        offset = sum(TRUE[chemical_symbols[z]] for z in base.numbers)
        e_mp = e_qe + offset + rng.normal(0.0, noise) * len(base.numbers)
        qe_frames.append(
            base.model_copy(update={"energy": e_qe, "label_source": "qe", "energy_scale": "qe"})
        )
        mp_frames.append(
            base.model_copy(update={"energy": e_mp, "label_source": "mptrj", "energy_scale": "mp"})
        )
    return qe_frames, mp_frames


def test_fit_recovers_known_offsets(b20_frame) -> None:
    qe_frames, mp_frames = _pairs(b20_frame)
    out = offsets.fit(qe_frames, mp_frames)
    assert out["n"] == 6 and out["rank"] == 5 and out["unique"] is True and out["n_elements"] == 5
    assert out["coefficients"] == pytest.approx(TRUE, abs=1e-9)
    assert out["residual_meV_atom"] == pytest.approx(0.0, abs=1e-8)
    assert out["max_abs_residual_meV_atom"] == pytest.approx(0.0, abs=1e-8)
    assert out["schema"] == "b20mlip.offsets.v1" and len(out["pairs"]) == 6
    assert out["pairs"][-1]["n_atoms"] == 7 and out["pairs"][0]["compound"] == "FeSi"
    assert out["pairs"][0]["e_mp_eV"] - out["pairs"][0]["e_qe_eV"] == pytest.approx(
        4 * 1.5 - 4 * 0.75
    )

    # stoichiometric compounds only: the residual is still exact but the coefficients are not unique
    out4 = offsets.fit(qe_frames[:4], mp_frames[:4])
    assert (
        out4["rank"] == 4
        and out4["unique"] is False
        and out4["residual_meV_atom"] == pytest.approx(0.0, abs=1e-8)
    )
    c = out4["coefficients"]
    assert c["Fe"] + c["Si"] == pytest.approx(TRUE["Fe"] + TRUE["Si"])
    assert c["Mn"] + c["Si"] == pytest.approx(TRUE["Mn"] + TRUE["Si"])


def test_fit_with_noise_and_pairing(b20_frame) -> None:
    qe_frames, mp_frames = _pairs(b20_frame, noise=0.010)  # 10 meV/atom noise
    out = offsets.fit(qe_frames, mp_frames)
    assert 1.0 < out["residual_meV_atom"] < 20.0 and out["coefficients"]["Fe"] == pytest.approx(
        TRUE["Fe"], abs=0.1
    )
    # pairing: unmatched ids, missing energies and duplicates are skipped
    extra = qe_frames[0].model_copy(update={"frame_id": "nomatch"})
    no_energy = mp_frames[1].model_copy(update={"energy": None})
    pairs = offsets.pair_frames([extra, *qe_frames, qe_frames[0]], [no_energy, *mp_frames])
    assert [p[0].frame_id for p in pairs] == [f.frame_id for f in qe_frames]
    assert pairs[1][1].energy is not None  # the first (energy-less) copy of frame 1 is ignored
    changed = mp_frames[2].model_copy(update={"numbers": [14] * 8})
    assert len(offsets.pair_frames(qe_frames, [changed])) == 0  # same id, different composition
    with pytest.raises(ValueError, match="no frame_id pairs"):
        offsets.fit(qe_frames, [strip_labels(f) for f in mp_frames])


def test_offsets_stage_and_cli(dft_settings: Settings, b20_frame, tmp_path: Path) -> None:
    qe_frames, mp_frames = _pairs(b20_frame, noise=0.002)
    qe_path, mp_path = tmp_path / "qe.extxyz", tmp_path / "mp.extxyz"
    write_frames(qe_frames, qe_path)
    write_frames(mp_frames, mp_path)
    out = tmp_path / "configs" / "offsets.json"
    cfg = dft_settings
    res = run_stage(
        "dft.offsets",
        cfg,
        offsets.run,
        executor=LocalExecutor(cfg),
        qe=qe_path,
        mp=mp_path,
        out=out,
    )
    assert res.status == "ok" and res.summary["passed"] == 1 and res.summary["n"] == 6
    assert res.summary["gate_meV_atom"] == 20.0 and res.summary["residual_meV_atom"] < 20.0
    data = json.loads(out.read_text())
    assert (
        data["passed"] is True
        and data["source_run_id"] == res.run_id
        and data["gate_meV_atom"] == 20.0
    )
    manifest = read_manifest(res.manifest_path)
    assert manifest.extras["passed"] is True and manifest.extras["n"] == 6
    assert {a.kind for a in manifest.inputs} == {"frames"} and len(manifest.inputs) == 2
    strict = cfg.model_copy(
        update={"eval": cfg.eval.model_copy(update={"offset_residual_gate_meV": 1e-6})}
    )
    res2 = run_stage(
        "dft.offsets",
        strict,
        offsets.run,
        executor=LocalExecutor(strict),
        qe=qe_path,
        mp=mp_path,
        out=out,
    )
    assert (
        res2.status == "ok"
        and res2.summary["passed"] == 0
        and json.loads(out.read_text())["passed"] is False
    )

    cli = CliRunner().invoke(
        app,
        [
            "--set",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "dft",
            "offsets",
            "--qe",
            str(qe_path),
            "--mp",
            str(mp_path),
            "--out",
            str(out),
        ],
    )
    assert cli.exit_code == 0, cli.output
    payload = json.loads(cli.output)
    assert (
        payload["stage"] == "dft.offsets"
        and payload["summary"]["n"] == 6
        and str(out) in payload["outputs"]
    )
    bad = CliRunner().invoke(
        app,
        [
            "--set",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "dft",
            "offsets",
            "--qe",
            str(qe_path),
            "--mp",
            str(tmp_path / "none.extxyz"),
            "--out",
            str(out),
        ],
    )
    assert bad.exit_code == 1 and '"status": "failed"' in bad.output
