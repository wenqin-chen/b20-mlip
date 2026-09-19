"""OMat24 ASE-LMDB reader and stream filter on synthetic shards (directory and tarball)."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest
from helpers_data import REAL_OMAT24, b20_frame, repo_path

from b20mlip.data import omat24
from b20mlip.io import read_frames

EXPECTED = {
    "scanned": 6,
    "in_family": 4,
    "kept": 3,
    "dropped_force_cap": 1,
    "dropped_min_elements": 1,
}


def test_find_shards_and_labels(omat24_dir: Path) -> None:
    shards = omat24.find_shards(omat24_dir)
    assert [omat24.shard_label(s, omat24_dir) for s in shards] == [
        "test/rattled-300",
        "train/aimd-from-PBE-1000-npt",
    ]
    assert omat24.find_shards(shards[0]) == [shards[0]]
    assert omat24.shard_label(Path("x.aselmdb"), None) == "x"


def test_iter_shard_rows_skips_nextid(omat24_dir: Path) -> None:
    shard = omat24.find_shards(omat24_dir)[0]
    rows = list(omat24.iter_shard_rows(shard))
    assert [k for k, _ in rows] == [1, 2, 3, 4]
    assert set(rows[0][1]) >= {"numbers", "positions", "cell", "energy", "forces", "stress", "data"}
    assert omat24.decode_row(orjson.dumps({"a": 1})) == {"a": 1}  # uncompressed fallback
    assert omat24.decode_row(zlib.compress(orjson.dumps({"a": 2}))) == {"a": 2}
    assert omat24.missing_dependencies() == []


def test_stream_filter_directory(omat24_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "omat.extxyz"
    frames, counts = omat24.stream_filter(omat24_dir, out=out, force_cap=15.0)
    assert {k: counts[k] for k in EXPECTED} == EXPECTED
    assert counts["per_element_set"] == {"Co-Fe-Si": 1, "Fe-Ge": 1, "Fe-Si": 1}
    assert counts["per_shard"] == {"test/rattled-300": 2, "train/aimd-from-PBE-1000-npt": 1}
    assert counts["per_calc_id"] == {"aimd-from-PBE-1000-npt": 1, "rattled-300": 2}
    assert counts["n_b20_spg198"] == 2 and counts["n_spg198_8atom"] == 2
    fesi, co2fesi, fege = frames
    assert fesi.compound == "FeSi" and fesi.config_type == "omat24"
    assert fesi.label_source == "omat24" and fesi.energy_scale == "omat24"
    assert fesi.group_id == "FeSi/omat24/agm000001_AB_8_spg198"
    assert fesi.parent_id == "agm000001_AB_8_spg198"
    assert fesi.info["omat24_shard"] == "test/rattled-300" and fesi.info["omat24_row"] == 1
    assert (
        fesi.info["omat24_calc_id"] == "rattled-300" and fesi.info["omat24_task_type"] == "Static"
    )
    assert fesi.energy == pytest.approx(-44.0) and fesi.info["energy_corrected_mp2020"] == -44.1
    assert len(fesi.stress) == 6 and fesi.forces is not None and fesi.weights == {}
    # 3x3 stress rows are converted to Voigt-6 [xx, yy, zz, yz, xz, xy]
    assert len(co2fesi.stress) == 6 and co2fesi.compound == "Co2FeSi"
    # missing data["parent_id"] -> Alexandria id prefix of the sid; no corrected energy key
    assert fege.parent_id == "agm000006" and fege.group_id == "FeGe/omat24/agm000006"
    assert "energy_corrected_mp2020" not in fege.info
    back = read_frames(out)
    assert [b.frame_id for b in back] == [f.frame_id for f in frames]
    assert back[0].stress == pytest.approx(fesi.stress) and back[0].label_source == "omat24"


def test_stream_filter_tarball_matches_directory(omat24_dir: Path, omat24_tarball: Path) -> None:
    a, ca = omat24.stream_filter(omat24_dir)
    b, cb = omat24.stream_filter(omat24_tarball)
    assert [f.frame_id for f in a] == [f.frame_id for f in b]
    assert {k: ca[k] for k in EXPECTED} == {k: cb[k] for k in EXPECTED}
    assert cb["per_shard"] == ca["per_shard"]


def test_stream_filter_options(omat24_dir: Path) -> None:
    frames, counts = omat24.stream_filter(omat24_dir, force_cap=100.0, min_elements=1)
    assert counts["kept"] == 5 and counts["dropped_force_cap"] == 0
    assert counts["dropped_min_elements"] == 0
    frames, counts = omat24.stream_filter(omat24_dir, ["Fe", "Si"])
    assert [f.compound for f in frames] == ["FeSi"]
    with pytest.raises(FileNotFoundError):
        list(omat24.iter_source(omat24_dir / "missing.tar.gz"))


def test_row_round_trip_and_stress_shapes(tmp_path: Path) -> None:
    frame = b20_frame("CoSi", label_source="omat24", energy_scale="omat24")
    rows = omat24.frames_to_rows([frame])
    row = rows[0]
    assert row["data"]["parent_id"] == "CoSi-p0"
    back = omat24.row_to_frame(row, shard="s", key=7)
    assert back.frame_id == frame.frame_id and back.stress == pytest.approx(frame.stress)
    assert back.forces == pytest.approx(np.asarray(frame.forces))
    row9 = dict(row)
    row9["stress"] = np.asarray(row["stress"])[[0, 5, 4, 5, 1, 3, 4, 3, 2]].tolist()  # 3x3 as 9
    assert omat24.row_to_frame(row9).stress == pytest.approx(frame.stress)
    row_none = dict(row)
    row_none.update(stress=None, forces=None, energy=None)
    f_none = omat24.row_to_frame(row_none)
    assert f_none.stress is None and f_none.forces is None and f_none.energy is None
    assert omat24.max_force_norm(f_none) is None
    shard = omat24.write_shard(rows, tmp_path / "rt.aselmdb")
    assert [k for k, _ in omat24.iter_shard_rows(shard)] == [1]


def test_ndarray_encoded_rows_decode(omat24_rows: dict[str, list[dict[str, Any]]]) -> None:
    """Some shards store arrays as ase.db ``{"__ndarray__": [shape, dtype, flat]}`` objects."""
    row = omat24_rows["test/rattled-300"][0]
    encoded = dict(row)
    for key in ("numbers", "positions", "cell", "forces", "stress"):
        arr = np.asarray(row[key])
        encoded[key] = {"__ndarray__": [list(arr.shape), str(arr.dtype), arr.ravel().tolist()]}
    decoded = omat24.decode_row(zlib.compress(orjson.dumps(encoded)))
    assert decoded["numbers"] == row["numbers"] and decoded["cell"] == row["cell"]
    assert omat24.row_to_frame(decoded).frame_id == omat24.row_to_frame(row).frame_id
    assert omat24.decode_ase_json({"a": [1, {"__ndarray__": [[2], "int64", [1, 2]]}]}) == {
        "a": [1, [1, 2]]
    }


def test_missing_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omat24, "lmdb", None)
    assert omat24.missing_dependencies() == ["lmdb"]
    with pytest.raises(ImportError, match="lmdb"):
        list(omat24.iter_shard_rows("x.aselmdb"))


@pytest.mark.slow
def test_real_subsplit_counts() -> None:
    """Measured 2026-09-18 on omat24_1M (1,171,309 rows): see docs/DATA_CARD.md."""
    root = repo_path(REAL_OMAT24)
    if not root.is_dir():
        pytest.skip("real OMat24 shards not present")
    frames, counts = omat24.stream_filter(root, force_cap=15.0)
    assert counts["scanned"] == 1_171_309
    assert counts["in_family"] == 656  # the earlier uncapped in-chemsys extraction
    assert counts["kept"] == len(frames) == counts["in_family"] - counts["dropped_force_cap"]
    assert counts["n_b20_spg198"] == 0
    assert isinstance(counts["per_element_set"], dict) and counts["dropped_min_elements"] > 0


def test_frames_to_rows_keeps_sid(omat24_rows: dict[str, list[dict[str, Any]]]) -> None:
    frame = omat24.row_to_frame(omat24_rows["test/rattled-300"][0])
    row = omat24.frames_to_rows([frame])[0]
    assert row["data"]["sid"] == frame.info["omat24_sid"]
    assert row["data"]["calc_id"] == "rattled-300"
