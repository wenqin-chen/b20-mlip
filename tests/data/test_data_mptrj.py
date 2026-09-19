"""MPtrj archive streaming: element screen, family extraction, B20 subset, conventions."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from helpers_data import REAL_MPTRJ, repo_path

from b20mlip.data import mptrj
from b20mlip.io import frame_id_for, read_frames


def test_member_ids() -> None:
    assert mptrj.member_mp_id("mp-871.extxyz") == "mp-871"
    assert mptrj.member_mp_id("sub/mp-12.extxyz") == "mp-12"
    assert mptrj.member_mp_id("README.txt") is None
    assert mptrj.member_mp_id("mvc-1.extxyz") is None


def test_screen_and_family_members(mptrj_zip: Path) -> None:
    with zipfile.ZipFile(mptrj_zip) as zf:
        assert mptrj.screen_member(zf, "mp-871.extxyz") == frozenset({"Fe", "Si"})
        assert mptrj.screen_member(zf, "mp-19381.extxyz") == frozenset({"Fe", "Si", "O"})
        assert mptrj.screen_member(zf, "README.txt") == frozenset()
        stats: dict = {}
        members = list(mptrj.iter_family_members(zf, stats=stats))
    assert members == ["mp-871.extxyz", "mp-1431.extxyz", "mp-22510.extxyz"]  # sorted by id
    assert stats["members_scanned"] == 6 and stats["members_in_family"] == 3
    assert stats["members_below_min_elements"] == 1  # elemental Si skipped and counted
    with zipfile.ZipFile(mptrj_zip) as zf:
        assert len(list(mptrj.iter_family_members(zf, min_elements=1))) == 4


def test_extract_family_conventions(mptrj_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "family.extxyz"
    stats: dict = {}
    frames = mptrj.extract_family(mptrj_zip, out=out, stats=stats)
    assert len(frames) == 6 and stats["frames"] == 6
    assert stats["per_element_set"] == {"Fe-Ge": 1, "Fe-Si": 3, "Mn-Si": 2}
    f = frames[0]
    assert f.label_source == "mptrj" and f.energy_scale == "mp" and f.config_type == "relax"
    assert f.compound == "FeSi" and f.parent_id == "mp-871" and f.group_id == "FeSi/mptrj/mp-871"
    assert f.energy is not None and f.forces is not None and len(f.forces) == 8
    assert f.stress is None and len(f.info["mptrj_stress_raw_voigt6"]) == 6
    assert f.info["mp_id"] == "mp-871" and f.info["mptrj_frame_id"] == "mp-871-0-0"
    assert f.info["task_id"] == "mp-871-task" and f.info["ionic_step"] == 0
    assert f.info["mp2020_corrected_energy"] == pytest.approx(f.energy - 0.284)
    assert f.frame_id == frame_id_for(f.numbers, f.positions, f.cell)  # geometry hash, not MPtrj's
    mnsi = [x for x in frames if x.compound == "MnSi"]
    assert len(mnsi) == 2 and mnsi[0].magmoms is not None and len(mnsi[0].magmoms) == 8
    assert all(x.magmoms is None for x in frames if x.compound == "FeSi")
    # the written file round-trips with identical ids and metadata
    back = read_frames(out)
    assert [b.frame_id for b in back] == [x.frame_id for x in frames]
    assert back[0].info["mp_id"] == "mp-871" and back[0].stress is None
    assert back[0].label_source == "mptrj"
    # restricting the element set drops MnSi and FeGe
    only = mptrj.extract_family(mptrj_zip, ["Fe", "Si"])
    assert {x.compound for x in only} == {"FeSi"}


def test_b20_subset_excludes_non_198_polymorph(mptrj_zip: Path) -> None:
    b20 = mptrj.b20_frames(mptrj_zip)
    assert mptrj.mp_ids(b20) == {"mp-1431": 2, "mp-871": 3}
    assert all(len(f.numbers) == 8 for f in b20)
    assert "mp-22510" not in mptrj.mp_ids(b20)  # sheared FeGe member is not P2_1 3


def test_frame_from_mptrj_atoms_requires_material_id() -> None:
    atoms = Atoms("Fe4Si4", positions=np.zeros((8, 3)), cell=np.eye(3) * 4.5, pbc=True)
    with pytest.raises(ValueError, match="material_id"):
        mptrj.frame_from_mptrj_atoms(atoms)
    frame = mptrj.frame_from_mptrj_atoms(atoms, "mp-1")
    assert frame.parent_id == "mp-1" and frame.energy is None and frame.forces is None


@pytest.mark.slow
def test_real_archive_counts() -> None:
    """Measured 2026-09-18: 1,460 in-family frames / 154 materials; 65 strict-B20 frames."""
    zip_path = repo_path(REAL_MPTRJ)
    if not zip_path.is_file():
        pytest.skip("real MPtrj archive not present")
    stats: dict = {}
    fam = mptrj.extract_family(zip_path, stats=stats)
    assert len(fam) == 1460 and stats["members_in_family"] == 154
    b20 = mptrj.b20_subset(fam)
    assert mptrj.mp_ids(b20) == {
        "mp-10692": 4, "mp-1078464": 14, "mp-1431": 12, "mp-21255": 8, "mp-7577": 13, "mp-871": 14,
    }  # fmt: skip
    old_path = repo_path(Path("data/raw/mptrj/b20_mptrj.extxyz"))
    if old_path.is_file():
        old = ase_read(old_path, index=":")
        old_ids = {a.info["frame_id"] for a in old}
        new_ids = {f.info["mptrj_frame_id"] for f in b20}
        # the 60-frame by-id extraction minus its 13 spg-12 mp-22510 frames = 47 shared frames
        assert len(old_ids & new_ids) == 47
        assert {a.info["material_id"] for a in old if a.info["frame_id"] not in new_ids} == {
            "mp-22510"
        }
