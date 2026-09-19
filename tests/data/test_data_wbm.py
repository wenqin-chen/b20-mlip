"""WBM seeded natural-prevalence sample on a 12-row mini summary and a mini atoms archive."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from helpers_data import REAL_WBM_CSV, REAL_WBM_ZIP, WBM_IN_FAMILY, WBM_ROWS, WBM_STABLE, repo_path

from b20mlip.data import wbm
from b20mlip.provenance import sha256_file


def test_formula_elements() -> None:
    assert wbm.formula_elements("Fe4 Si4") == {"Fe", "Si"}
    assert wbm.formula_elements("Li1 Fe1 Ge1") == {"Li", "Fe", "Ge"}
    assert wbm.formula_elements("Cu1") == {"Cu"}


def test_read_summary_requires_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("material_id,formula\nwbm-1-1,Fe1\n")
    with pytest.raises(ValueError):
        wbm.read_summary(bad)


def test_sample_protocol_truth_and_checksums(wbm_files: tuple[Path, Path], tmp_path: Path) -> None:
    csv_path, zip_path = wbm_files
    out = tmp_path / "sample.json"
    r = wbm.sample(csv_path, zip_path, 5, 0, out=out)
    ids = sorted(row[0] for row in WBM_ROWS)
    expected = [ids[i] for i in np.random.default_rng(0).permutation(len(ids))[:5]]
    assert r["ids"] == expected and r["n"] == 5 and r["seed"] == 0
    assert r["population_n"] == 12 and r["population_stable"] == 5
    assert r["population_prevalence"] == pytest.approx(5 / 12)
    assert r["population_missing_hull"] == 1
    assert r["in_family_ids"] == WBM_IN_FAMILY and r["in_family_n"] == 6
    assert r["in_family_stable"] == 2 and r["in_family_stable_fraction"] == pytest.approx(2 / 6)
    assert r["n_stable"] == len(set(expected) & WBM_STABLE)
    assert r["prevalence"] == pytest.approx(r["n_stable"] / 5)
    assert set(r["truth"]) == set(expected) | set(WBM_IN_FAMILY)
    assert r["truth"]["wbm-3-1"] == {
        "formula": "Mn1 Si1", "n_sites": 2, "e_form_per_atom": -0.3, "e_above_hull": None,
        "stable": False,
    }  # fmt: skip
    assert (
        r["truth"]["wbm-1-1"]["stable"] is True and r["truth"]["wbm-1-1"]["e_form_per_atom"] == -0.5
    )
    assert r["hull_column"] == "e_above_hull_mp2020_corrected_ppd_mp"
    assert r["checksums"]["summary_csv"]["sha256"] == sha256_file(csv_path)
    assert r["checksums"]["atoms_zip"]["bytes"] == zip_path.stat().st_size
    loaded = wbm.load_sample(out)
    assert loaded == json.loads(json.dumps(r))
    assert wbm.sample(csv_path, zip_path, 5, 0)["ids"] == expected  # deterministic
    assert wbm.sample(csv_path, zip_path, 5, 1)["ids"] != expected
    with pytest.raises(ValueError, match="exceeds"):
        wbm.sample(csv_path, zip_path, 13, 0)


def test_load_sample_validation(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"ids": ["a"]}))
    with pytest.raises(ValueError, match="missing"):
        wbm.load_sample(p)
    p.write_text(
        json.dumps({"ids": ["a"], "prevalence": 0, "in_family_ids": [], "truth": {},
                    "checksums": {}, "seed": 0, "n": 2})
    )  # fmt: skip
    with pytest.raises(ValueError, match="ids but n=2"):
        wbm.load_sample(p)


def test_atoms_and_frames_for_ids(wbm_files: tuple[Path, Path]) -> None:
    _, zip_path = wbm_files
    atoms = wbm.atoms_for_ids(zip_path, ["wbm-1-1", "wbm-3-2"])
    assert list(atoms) == ["wbm-1-1", "wbm-3-2"]
    assert len(atoms["wbm-1-1"]) == 8 and len(atoms["wbm-3-2"]) == 1
    assert "material_id" not in atoms["wbm-1-1"].info
    frames = wbm.frames_for_ids(zip_path, ["wbm-1-1"])
    (f,) = frames
    assert f.config_type == "wbm" and f.group_id == "FeSi/wbm/wbm-1-1" and f.parent_id == "wbm-1-1"
    assert f.info == {"wbm_id": "wbm-1-1"} and f.label_source == "none"
    with pytest.raises(KeyError):
        wbm.atoms_for_ids(zip_path, ["wbm-9-9"])


@pytest.mark.slow
def test_real_wbm_numbers() -> None:
    """Measured 2026-09-18: 256,963 rows, 16.67 % stable; 312 in-family (1.28 %); sample 15.2 %."""
    csv_path, zip_path = repo_path(REAL_WBM_CSV), repo_path(REAL_WBM_ZIP)
    if not (csv_path.is_file() and zip_path.is_file()):
        pytest.skip("real WBM files not present")
    r = wbm.sample(csv_path, zip_path, 1000, 0)
    assert r["population_n"] == 256_963 and r["population_prevalence"] == pytest.approx(
        0.16666, abs=1e-4
    )
    assert r["in_family_n"] == 312 and r["in_family_stable"] == 4
    assert r["n_stable"] == 152 and r["ids"][0] == "wbm-4-18301"
