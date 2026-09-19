"""`data filter`: StructureMatcher dedupe, force cap and the magnetic-branch filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from helpers_data import b20_frame

from b20mlip.config import Settings
from b20mlip.data import filters
from b20mlip.io import read_frames, write_frames
from b20mlip.models import DFTFrame, Frame
from b20mlip.provenance import read_manifest, run_stage


def _dft(frame: Frame, **kw: Any) -> DFTFrame:
    fields: dict[str, Any] = dict(
        code="qe", functional="PBE", pseudo_md5s={}, ecut_ry=90.0, k_spacing=0.25, nspin=2,
        smearing="mv", degauss_ry=0.01, converged=True, scf_steps=12, abs_magnetization=None,
        fermi_eV=10.0, branch_ok=None, wall_seconds=1.0, unit_id="u",
    )  # fmt: skip
    fields.update(kw)
    return DFTFrame(**frame.model_dump(), **fields)


def test_dedupe_keeps_first_and_distinguishes_derivatives() -> None:
    a = b20_frame("FeSi", seed=1)
    jitter = a.model_copy(
        update={"positions": (np.asarray(a.positions) + 1e-5).tolist(), "frame_id": "j" * 16}
    )
    strained = b20_frame("FeSi", scale=1.02, seed=2)
    rattled = b20_frame("FeSi", rattle=0.1, seed=3)
    cosi = b20_frame("CoSi", seed=4)
    stats: dict = {}
    kept = filters.dedupe([a, a, jitter, strained, rattled, cosi], stats=stats)
    assert [f.frame_id for f in kept] == [
        a.frame_id,
        strained.frame_id,
        rattled.frame_id,
        cosi.frame_id,
    ]
    assert stats["n_in"] == 6 and stats["n_kept"] == 4
    assert stats["n_duplicate_ids"] == 1 and stats["n_duplicate_structures"] == 1
    logged: list[dict] = []
    filters.dedupe([a], log=lambda **kw: logged.append(kw))
    assert logged[0]["dedupe"]["n_kept"] == 1


def test_force_cap() -> None:
    small = b20_frame("FeSi", seed=1)
    big = small.model_copy(
        update={"forces": (np.asarray(small.forces) * 100).tolist(), "frame_id": "b" * 16}
    )
    unlabelled = b20_frame("MnSi", labelled=False)
    stats: dict = {}
    kept = filters.force_cap([small, big, unlabelled], 15.0, stats=stats)
    assert [f.frame_id for f in kept] == [small.frame_id, unlabelled.frame_id]
    assert stats == {"n_kept": 2, "n_dropped_force_cap": 1, "n_unlabelled": 1, "cap_eVA": 15.0}
    assert filters.max_force_norm(unlabelled) is None


def test_magnetic_branch_on_dftframes() -> None:
    m_ref = {"MnSi": 1.0, "FeSi": 0.0}
    good = _dft(b20_frame("MnSi", seed=1), abs_magnetization=4.0)  # 4 Mn -> 1.0 muB each
    wrong = _dft(b20_frame("MnSi", seed=2), abs_magnetization=8.4)  # 2.1 muB each
    unconv = _dft(b20_frame("MnSi", seed=3), abs_magnetization=4.0, converged=False)
    fesi = _dft(b20_frame("FeSi", seed=4), abs_magnetization=0.02)
    noref = _dft(b20_frame("CoGe", seed=5), abs_magnetization=0.0)
    kept, counts = filters.magnetic_branch([good, wrong, unconv, fesi, noref], m_ref, 0.3)
    assert [f.frame_id for f in kept] == [good.frame_id, fesi.frame_id, noref.frame_id]
    assert kept[0].branch_ok is True and kept[1].branch_ok is True and kept[2].branch_ok is None
    assert counts["n_in"] == 5 and counts["n_kept"] == 3
    assert counts["n_wrong_branch"] == 1 and counts["n_unconverged"] == 1
    assert counts["n_no_reference"] == 1 and counts["tol_muB"] == 0.3
    reasons = {r["frame_id"]: r["reason"] for r in counts["rejected"]}
    assert reasons == {wrong.frame_id: "wrong_branch", unconv.frame_id: "unconverged"}
    assert next(r for r in counts["rejected"] if r["reason"] == "wrong_branch")["m_per_TM"] == 2.1


def test_magnetic_branch_on_plain_frames_via_info() -> None:
    m_ref = {"MnSi": 1.0}
    via_abs = b20_frame("MnSi", seed=1, info={"converged": True, "abs_magnetization": 4.2})
    via_total = b20_frame("MnSi", seed=2).model_copy(update={"total_magnetization": -3.9})
    unknown = b20_frame("MnSi", seed=3)
    unconv = b20_frame("MnSi", seed=4, info={"converged": False})
    si = Frame(
        frame_id="s" * 16,
        group_id="Si/relax/x",
        compound="Si",
        config_type="relax",
        parent_id="x",
        numbers=[14, 14],
        positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
        cell=(np.eye(3) * 3.87).tolist(),
    )
    kept, counts = filters.magnetic_branch([via_abs, via_total, unknown, unconv, si], m_ref, 0.3)
    assert [f.frame_id for f in kept] == [
        via_abs.frame_id,
        via_total.frame_id,
        unknown.frame_id,
        "s" * 16,
    ]
    assert kept[0].info["branch_ok"] is True and kept[1].info["branch_ok"] is True
    assert "branch_ok" not in kept[2].info
    assert counts["n_no_magnetization"] == 1 and counts["n_no_transition_metal"] == 1
    assert counts["n_unconverged"] == 1
    assert filters.m_per_tm(via_total) == pytest.approx(3.9 / 4) and filters.m_per_tm(si) == 0.0
    assert filters.m_per_tm(unknown) is None


def test_load_m_ref(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"MnSi": 1, "FeSi": 0.0}))
    assert filters.load_m_ref(p) == {"MnSi": 1.0, "FeSi": 0.0}
    p.write_text("[1]")
    with pytest.raises(ValueError):
        filters.load_m_ref(p)


def test_run_stage_pipeline(data_settings: Settings, tmp_path: Path) -> None:
    a = b20_frame("MnSi", seed=1, info={"converged": True, "abs_magnetization": 4.0})
    dup = a.model_copy(update={"frame_id": "d" * 16})
    wrong = b20_frame(
        "MnSi", seed=2, scale=1.02, info={"converged": True, "abs_magnetization": 9.0}
    )
    big = b20_frame("FeSi", seed=3)
    big = big.model_copy(update={"forces": (np.asarray(big.forces) * 200).tolist()})
    fine = b20_frame("FeSi", seed=4, scale=1.01, info={"converged": True, "abs_magnetization": 0.0})
    src = tmp_path / "in.extxyz"
    write_frames([a, dup, wrong, big, fine], src)
    m_ref = tmp_path / "m_ref.json"
    m_ref.write_text(json.dumps({"MnSi": 1.0, "FeSi": 0.0}))
    out = tmp_path / "out.extxyz"
    result = run_stage("data.filter", data_settings, filters.run, frames=src, out=out, m_ref=m_ref)
    assert result.status == "ok", result.summary
    assert result.summary == {
        "n_in": 5, "n_after_dedupe": 4, "n_after_force_cap": 3, "n_after_magnetic_branch": 2,
        "n_wrong_branch": 1, "n_unconverged": 0, "n_out": 2,
    }  # fmt: skip
    assert [f.frame_id for f in read_frames(out)] == [a.frame_id, fine.frame_id]
    manifest = read_manifest(result.manifest_path)
    # `dup` keeps its own stored frame_id through the extxyz round trip: a structural duplicate
    assert manifest.extras["dedupe"]["n_duplicate_ids"] == 0
    assert manifest.extras["dedupe"]["n_duplicate_structures"] == 1
    assert manifest.extras["force_cap"]["n_dropped_force_cap"] == 1
    assert manifest.extras["magnetic_branch"]["n_wrong_branch"] == 1
    # no m_ref anywhere -> the magnetic filter is skipped and logged
    result = run_stage(
        "data.filter", data_settings, filters.run, frames=src, out=out, do_dedupe=False, cap=1000.0
    )
    assert result.summary["n_out"] == 5 and "n_after_dedupe" not in result.summary
    assert read_manifest(result.manifest_path).extras["magnetic_branch"].startswith("skipped")
    # m_ref from the config (dft.m_ref_muB) when the config carries it
    cfg = data_settings.model_copy(deep=True)
    if hasattr(cfg.dft, "m_ref_muB"):
        cfg.dft.m_ref_muB = {"MnSi": 1.0}
        result = run_stage("data.filter", cfg, filters.run, frames=src, out=out, do_dedupe=False)
        assert result.summary["n_wrong_branch"] == 1
    dry = run_stage("data.filter", data_settings, filters.run, frames=src, out=out, dry_run=True)
    assert dry.status == "partial" and dry.summary == {"planned": 1}
