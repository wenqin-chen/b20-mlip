"""Fixtures for the data tier: synthetic mini MPtrj zip, OMat24 shards/tarball, WBM csv+zip and
a phononDB extxyz. Everything is built in-test (offline, fast); helpers live in helpers_data.py."""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write
from helpers_data import WBM_ROWS, b20_cell, mptrj_member_bytes, omat_row

from b20mlip.config import Settings
from b20mlip.data import omat24


@pytest.fixture(scope="session")
def mptrj_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Members: FeSi x3 (B20), MnSi x2 (B20, magmoms), FeGe x1 sheared (not 198), Si (elemental),
    NaCl and Fe2SiO4 (out of family), plus a non-material text member."""
    path = tmp_path_factory.mktemp("mptrj") / "mptrj-mini.extxyz.zip"
    fesi = [b20_cell("FeSi", 1.0 + 0.01 * k) for k in range(3)]
    mnsi = [b20_cell("MnSi", 1.0), b20_cell("MnSi", 1.02)]
    fege = b20_cell("FeGe")
    F = np.eye(3)
    F[0, 1] = 0.08  # shear breaks P2_1 3
    fege.set_cell(fege.cell[:] @ F.T, scale_atoms=True)
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=np.eye(3) * 3.87, pbc=True)
    nacl = Atoms("NaCl", positions=[[0, 0, 0], [2.8, 2.8, 2.8]], cell=np.eye(3) * 5.6, pbc=True)
    rng = np.random.default_rng(1)
    feo = Atoms("Fe2SiO4", positions=rng.uniform(0, 4, (7, 3)), cell=np.eye(3) * 5, pbc=True)
    members = [
        ("mp-871", fesi, "Fe4 Si4", False),
        ("mp-1431", mnsi, "Mn4 Si4", True),
        ("mp-22510", [fege], "Fe4 Ge4", True),
        ("mp-149", [si], "Si2", False),
        ("mp-22862", [nacl], "Na1 Cl1", False),
        ("mp-19381", [feo], "Fe2 Si1 O4", False),
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mp_id, images, formula, magnetic in members:
            zf.writestr(
                f"{mp_id}.extxyz", mptrj_member_bytes(images, mp_id, formula, magnetic=magnetic)
            )
        zf.writestr("README.txt", b"not a material\n")
    return path


@pytest.fixture(scope="session")
def omat24_rows() -> dict[str, list[dict[str, Any]]]:
    """Two shards; expected: scanned 6, in_family 4, kept 3, force-cap 1, elemental 1, spg198 2."""
    rng = np.random.default_rng(3)
    fesi = b20_cell("FeSi")
    co2fesi = Atoms(
        "Co2FeSi",
        positions=[[0, 0, 0], [1.4, 1.4, 0], [1.4, 0, 1.4], [0, 1.4, 1.4]],
        cell=np.eye(3) * 2.8,
        pbc=True,
    )
    mnsi = b20_cell("MnSi")
    mnsi.positions = mnsi.positions + rng.normal(0.0, 0.1, mnsi.positions.shape)
    nacl = Atoms("NaCl", positions=[[0, 0, 0], [2.8, 2.8, 2.8]], cell=np.eye(3) * 5.6, pbc=True)
    fe = Atoms("Fe2", positions=[[0, 0, 0], [1.43, 1.43, 1.43]], cell=np.eye(3) * 2.87, pbc=True)
    fege = b20_cell("FeGe", 1.03)
    shard_a = [
        omat_row(
            fesi, sid="agm000001_AB_8_spg198_0_0_rattled-300_aaaa", parent="agm000001_AB_8_spg198"
        ),
        omat_row(
            co2fesi,
            sid="agm000002_ABC2_4_spg225_0_0_rattled-300_bbbb",
            parent="agm000002_ABC2_4_spg225",
            stress_3x3=True,
            seed=1,
        ),
        omat_row(
            mnsi,
            sid="agm000003_AB_8_spg198_0_0_rattled-300_cccc",
            parent="agm000003_AB_8_spg198",
            fmax=20.0,
        ),
        omat_row(
            nacl, sid="agm000004_AB_2_spg225_0_0_rattled-300_dddd", parent="agm000004_AB_2_spg225"
        ),
    ]
    shard_b = [
        omat_row(
            fe,
            sid="agm000005_A_2_spg229_0_aimd-from-PBE-1000-npt_eeee_1",
            parent="agm000005_A_2_spg229",
            calc_id="aimd-from-PBE-1000-npt",
        ),
        omat_row(
            fege,
            sid="agm000006_AB_8_spg198_0_aimd-from-PBE-1000-npt_ffff_2",
            parent=None,
            calc_id="aimd-from-PBE-1000-npt",
            corrected=False,
            seed=2,
        ),
    ]
    return {"test/rattled-300": shard_a, "train/aimd-from-PBE-1000-npt": shard_b}


@pytest.fixture(scope="session")
def omat24_dir(
    tmp_path_factory: pytest.TempPathFactory, omat24_rows: dict[str, list[dict[str, Any]]]
) -> Path:
    root = tmp_path_factory.mktemp("omat24") / "omat24_mini"
    for label_, rows in omat24_rows.items():
        split = label_.split("/")[0]
        omat24.write_shard(rows, root / label_ / f"{split}.aselmdb")
    return root


@pytest.fixture(scope="session")
def omat24_tarball(tmp_path_factory: pytest.TempPathFactory, omat24_dir: Path) -> Path:
    path = tmp_path_factory.mktemp("omat24tar") / "omat24_mini.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(omat24_dir, arcname="omat24_mini")
    return path


@pytest.fixture(scope="session")
def wbm_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("wbm")
    csv_path = root / "wbm-summary-mini.csv.gz"
    header = (
        "material_id,formula,n_sites,volume,uncorrected_energy,e_form_per_atom_wbm,"
        "e_above_hull_wbm,bandgap_pbe,e_form_per_atom_mp2020_corrected,"
        "e_above_hull_mp2020_corrected_ppd_mp,unique_prototype\n"
    )
    lines = [header]
    for mid, formula, n_sites, e_hull, e_form in WBM_ROWS:
        hull = "" if e_hull is None else f"{e_hull}"
        lines.append(
            f"{mid},{formula},{float(n_sites)},100.0,-10.0,0.1,0.1,0.0,{e_form},{hull},True\n"
        )
    with gzip.open(csv_path, "wt", encoding="utf-8") as fh:
        fh.writelines(lines)
    zip_path = root / "wbm-initial-atoms-mini.extxyz.zip"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mid, formula, n_sites, _, _ in WBM_ROWS:
            symbols = "".join(formula.split())
            atoms = Atoms(
                symbols, positions=rng.uniform(0, 4, (n_sites, 3)), cell=np.eye(3) * 5, pbc=True
            )
            atoms.info["material_id"] = mid
            buf = io.StringIO()
            ase_write(buf, atoms, format="extxyz")
            zf.writestr(f"{mid}.extxyz", buf.getvalue())
    return csv_path, zip_path


@pytest.fixture(scope="session")
def phonondb_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("phonondb") / "phononDB-mini.extxyz"
    images = [b20_cell("FeSi"), b20_cell("CoSi")]
    for atoms, mid in zip(images, ("mp-871", "mp-7577"), strict=True):
        atoms.info.update(material_id=mid, name=atoms.get_chemical_formula(empirical=True))
    ase_write(path, images, format="extxyz")
    return path


@pytest.fixture
def data_settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "runs_dir": str(tmp_path / "runs"),
                "models_dir": str(tmp_path / "models"),
                "dft_dir": str(tmp_path / "dft"),
                "reports_dir": str(tmp_path / "reports"),
            }
        }
    )
