"""`data split`: group-hash determinism, no leakage, fractions, tier assignment, exclusions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers_data import b20_frame

from b20mlip.config import Settings
from b20mlip.data import split as sp
from b20mlip.data import wbm
from b20mlip.io import write_frames
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage, sha256_frames

COMPOUNDS = ("FeSi", "CoSi", "MnSi")


def qe_frames(n_groups: int = 300, per_group: int = 3) -> list[Frame]:
    frames: list[Frame] = []
    for g in range(n_groups):
        compound = COMPOUNDS[g % 3]
        parent = f"{compound}-p{g}"
        for k in range(per_group):
            frames.append(
                b20_frame(
                    compound,
                    "rattle",
                    rattle=0.05,
                    seed=1000 * g + k,
                    label_source="qe",
                    energy_scale="qe",
                    parent=parent,
                )  # fmt: skip
            )
    return frames


def test_group_hash_and_buckets() -> None:
    u = sp.group_hash("FeSi/rattle/p0", 0)
    assert 0.0 <= u < 1.0 and u == sp.group_hash("FeSi/rattle/p0", 0)
    assert u != sp.group_hash("FeSi/rattle/p0", 1) != sp.group_hash("FeSi/rattle/p1", 1)
    values = [sp.group_hash(f"g{i}", 0) for i in range(2000)]
    assert abs(float(np.mean(values)) - 0.5) < 0.03
    fr = (0.8, 0.1, 0.1)
    assert sp.bucket_of(0.1, fr) == "train" and sp.bucket_of(0.85, fr) == "val"
    assert sp.bucket_of(0.95, fr) == "test" and sp.bucket_of(0.8, fr) == "val"


def test_split_is_deterministic_leak_free_and_balanced() -> None:
    frames = qe_frames()
    s1 = sp.group_split(frames, seed=0)
    s2 = sp.group_split(list(frames), seed=0)
    assert s1 == s2 and s1.split_id == s2.split_id and len(s1.split_id) == 12
    s3 = sp.group_split(frames, seed=1)
    assert s3.split_id != s1.split_id and set(s3.train) != set(s1.train)
    assert s1.frames_sha256 == sha256_frames(frames) and s1.policy == "group_hash"
    ids = {f.frame_id for f in frames}
    assert set(s1.train) | set(s1.val) | set(s1.test) == ids
    assert not (set(s1.train) & set(s1.val)) and not (set(s1.train) & set(s1.test))
    assert not (set(s1.val) & set(s1.test))
    buckets = {"train": set(s1.train), "val": set(s1.val), "test": set(s1.test)}
    for gid, members in s1.groups.items():
        where = {b for b, ids_ in buckets.items() if set(members) & ids_}
        assert len(where) == 1, f"group {gid} leaks across {where}"
    n_groups = len(s1.groups)
    assert n_groups == 300
    frac = {b: sum(1 for m in s1.groups.values() if m[0] in buckets[b]) / n_groups for b in buckets}
    assert abs(frac["train"] - 0.8) < 0.06 and abs(frac["val"] - 0.1) < 0.05
    assert abs(frac["test"] - 0.1) < 0.05
    assert set(s1.tiers["T0"]) == set(s1.val) | set(s1.test)
    assert all(s1.tiers[t] == [] for t in ("T1", "T2", "T3", "T4a", "T4b"))
    assert s1.fractions == (0.8, 0.1, 0.1) and s1.holdout_compounds == ["FeGe", "MnGe"]


def _train_group_seed(group_id: str, want: str = "train") -> int:
    for seed in range(200):
        if sp.bucket_of(sp.group_hash(group_id, seed), (0.8, 0.1, 0.1)) == want:
            return seed
    raise AssertionError("no seed found")


def test_exclusions_and_tiers() -> None:
    hot_parent = "MnSi-md"
    md_group = f"MnSi/md/{hot_parent}"
    seed = _train_group_seed(md_group)
    cool = b20_frame("MnSi", "md", rattle=0.05, seed=1, label_source="qe", energy_scale="qe",
                     parent=hot_parent, temperature_K=300.0)  # fmt: skip
    hot = b20_frame("MnSi", "md", rattle=0.05, seed=2, label_source="qe", energy_scale="qe",
                    parent=hot_parent, temperature_K=900.0)  # fmt: skip
    fege = [b20_frame("FeGe", "strain", scale=1.0 + 0.01 * k, seed=10 + k, label_source="qe",
                      energy_scale="qe", parent="FeGe-p") for k in range(3)]  # fmt: skip
    omat = [b20_frame("FeSi", "omat24", rattle=0.1, seed=20 + k, label_source="omat24",
                      energy_scale="omat24", parent=f"agm{k}") for k in range(3)]  # fmt: skip
    mptrj = [b20_frame("CoSi", "relax", scale=1.0 + 0.01 * k, seed=30 + k, label_source="mptrj",
                       energy_scale="mp", parent="mp-7577") for k in range(3)]  # fmt: skip
    unlabelled = [b20_frame("FeSi", "rattle", rattle=0.05, seed=40 + k, labelled=False,
                            parent="FeSi-cand") for k in range(3)]  # fmt: skip
    frames = [cool, hot, *fege, *omat, *mptrj, *unlabelled]
    s = sp.group_split(frames, seed=seed, wbm_ids=["wbm-1-1", "wbm-2-2"])
    assert cool.frame_id in s.train  # the cool frame of a train-bucket group stays trainable
    assert hot.frame_id in s.test and s.tiers["T1"] == [hot.frame_id]
    assert set(s.tiers["T2"]) == {f.frame_id for f in fege}
    assert not ({f.frame_id for f in fege} & set(s.train))
    assert set(s.tiers["T3"]) == {f.frame_id for f in omat} and not (
        set(s.tiers["T3"]) & set(s.train)
    )
    assert set(s.tiers["T4a"]) == {f.frame_id for f in mptrj} and not (
        set(s.tiers["T4a"]) & set(s.train)
    )
    assert not ({f.frame_id for f in unlabelled} & set(s.train))
    assert s.tiers["T4b"] == ["wbm-1-1", "wbm-2-2"]
    qe_ids = {f.frame_id for f in frames if f.label_source == "qe"}
    assert set(s.tiers["T0"]) == {i for i in s.val + s.test if i in qe_ids}
    assert s.max_train_T == 600.0 and s.holdout_compounds == ["FeGe", "MnGe"]
    # raising the temperature ceiling puts the hot frame back into train
    s2 = sp.group_split(frames, seed=seed, max_train_T=1000.0)
    assert hot.frame_id in s2.train and s2.tiers["T1"] == [] and s2.split_id != s.split_id
    # allowing omat24 as a train source still keeps it out (never-trained tier by policy)
    s3 = sp.group_split(frames, seed=seed, train_sources=("qe", "omat24"))
    assert not (set(s3.tiers["T3"]) & set(s3.train))


def test_invalid_inputs() -> None:
    frames = qe_frames(3, 1)
    with pytest.raises(ValueError, match="fractions"):
        sp.group_split(frames, fractions=(0.7, 0.1, 0.1))
    with pytest.raises(ValueError, match="duplicate"):
        sp.group_split(frames + [frames[0]])


def test_round_trip_and_frames_by_bucket(tmp_path: Path) -> None:
    frames = qe_frames(20, 2)
    s = sp.group_split(frames, seed=3)
    path = sp.write_split(s, tmp_path / "splits" / "s.json")
    assert sp.read_split(path) == s
    by = sp.frames_by_bucket(s, frames)
    assert [f.frame_id for f in by["train"]] == s.train and len(by["val"]) == len(s.val)
    assert sp.split_id_for("a", 0, (0.8, 0.1, 0.1)) != sp.split_id_for("a", 0, (0.7, 0.2, 0.1))


def test_run_stage(data_settings: Settings, tmp_path: Path, wbm_files: tuple[Path, Path]) -> None:
    frames = qe_frames(30, 2) + [b20_frame("FeGe", label_source="qe", energy_scale="qe")]
    src = tmp_path / "frames.extxyz"
    write_frames(frames, src)
    csv_path, zip_path = wbm_files
    sample_json = tmp_path / "wbm_sample.json"
    wbm.sample(csv_path, zip_path, 3, 0, out=sample_json)
    result = run_stage(
        "data.split", data_settings, sp.run, seed=0, frames=src, out=tmp_path / "splits",
        wbm_sample=sample_json,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    split_id = result.summary["split_id"]
    out = tmp_path / "splits" / f"{split_id}.json"
    assert out.is_file() and sp.read_split(out).split_id == split_id
    assert result.summary["n_frames"] == 61 and result.summary["n_groups"] == 31
    assert result.summary["n_T4b"] == 3 and result.summary["n_T2"] == 1
    assert result.summary["n_train"] + result.summary["n_val"] + result.summary["n_test"] == 61
    manifest = read_manifest(result.manifest_path)
    assert {Path(a.path).name for a in manifest.inputs} == {"frames.extxyz", "wbm_sample.json"}
    assert [Path(a.path).name for a in manifest.outputs] == [f"{split_id}.json"]
    explicit = run_stage("data.split", data_settings, sp.run, seed=0, frames=src,
                         out=tmp_path / "v1.json")  # fmt: skip
    assert (tmp_path / "v1.json").is_file() and explicit.summary["n_T4b"] == 0
    dry = run_stage("data.split", data_settings, sp.run, frames=src, out=tmp_path, dry_run=True)
    assert dry.status == "partial" and dry.summary == {"planned": 1}
