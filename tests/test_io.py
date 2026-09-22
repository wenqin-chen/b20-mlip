"""Frame <-> Atoms round trip incl. weights, stress units (Voigt 6, eV/A^3), extxyz I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.stress import voigt_6_to_full_3x3_stress

from b20mlip.io import frame_from_atoms, frame_id_for, frame_to_atoms, read_frames, write_frames
from b20mlip.models import Frame
from b20mlip.provenance import sha256_file


def _meta(frame: Frame) -> dict[str, object]:
    return dict(
        group_id=frame.group_id,
        compound=frame.compound,
        config_type=frame.config_type,
        parent_id=frame.parent_id,
        label_source=frame.label_source,
        energy_scale=frame.energy_scale,
    )


def test_fixture_shape(tiny_frames: list[Frame]) -> None:
    assert len(tiny_frames) == 15
    assert all(len(f.numbers) == 8 for f in tiny_frames)
    assert {f.compound for f in tiny_frames} == {"FeSi", "CoSi", "MnSi", "FeGe"}
    base = [f for f in tiny_frames if f.label_source == "none"]
    assert len(base) == 12 and {f.config_type for f in base} == {"relax", "strain", "rattle"}
    qe = [f for f in tiny_frames if f.energy_scale == "qe"]
    assert len(qe) == 3 and all(f.weights == {"energy": 0.0} for f in qe)
    assert all(f.label_source == "qe" and f.config_type == "noise_floor" for f in qe)
    assert (
        len({f.frame_id for f in tiny_frames}) == 15
        and len({f.group_id for f in tiny_frames}) == 15
    )
    assert all(f.group_id == f"{f.compound}/{f.config_type}/{f.parent_id}" for f in tiny_frames)
    assert all(
        f.energy is not None and f.forces is not None and f.stress is not None for f in tiny_frames
    )
    assert all((f.magmoms is not None) == (f.compound in {"MnSi", "FeGe"}) for f in tiny_frames)
    assert all((f.temperature_K == 300.0) == (f.config_type == "rattle") for f in tiny_frames)


def test_in_memory_round_trip_is_exact(tiny_frames: list[Frame]) -> None:
    for frame in tiny_frames:
        atoms = frame_to_atoms(frame)
        assert atoms.calc is not None and len(atoms) == 8 and atoms.pbc.all()
        back = frame_from_atoms(atoms, **_meta(frame))  # type: ignore[arg-type]
        assert back == frame


def test_weights_become_config_weight_info(tiny_frames: list[Frame]) -> None:
    qe = next(f for f in tiny_frames if f.energy_scale == "qe")
    atoms = frame_to_atoms(qe)
    assert atoms.info["config_energy_weight"] == 0.0
    assert atoms.info["energy_scale"] == "qe" and atoms.info["label_source"] == "qe"
    plain = next(f for f in tiny_frames if f.energy_scale == "none")
    assert not any(
        k.startswith("config_") and k.endswith("_weight") for k in frame_to_atoms(plain).info
    )
    multi = qe.model_copy(update={"weights": {"energy": 0.0, "forces": 2.0, "stress": 0.5}})
    info = frame_to_atoms(multi).info
    assert (info["config_forces_weight"], info["config_stress_weight"]) == (2.0, 0.5)
    assert frame_from_atoms(frame_to_atoms(multi), **_meta(multi)).weights == multi.weights  # type: ignore[arg-type]


def test_stress_is_voigt6_in_ev_per_A3() -> None:
    atoms = Atoms("FeSi", positions=[[0, 0, 0], [1.2, 1.2, 1.2]], cell=np.eye(3) * 4.0, pbc=True)
    voigt = np.array([0.01, 0.02, 0.03, 0.004, 0.005, 0.006])  # xx yy zz yz xz xy
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, stress=voigt_6_to_full_3x3_stress(voigt))
    frame = frame_from_atoms(
        atoms, group_id="g", compound="FeSi", config_type="relax", parent_id="p"
    )
    assert frame.stress is not None
    np.testing.assert_allclose(frame.stress, voigt)  # 3x3 -> Voigt 6, no unit conversion
    assert np.asarray(frame_to_atoms(frame).calc.results["stress"]).shape == (6,)
    atoms.info["stress"] = list(voigt)
    atoms.calc = None
    assert frame_from_atoms(
        atoms, group_id="g", compound="FeSi", config_type="relax", parent_id="p"
    ).stress == list(voigt)
    atoms.info["stress"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="Voigt-6"):
        frame_from_atoms(atoms, group_id="g", compound="FeSi", config_type="relax", parent_id="p")


def test_frame_id_is_stable_hex16() -> None:
    numbers, cell = [26, 14], np.eye(3) * 4.0
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    fid = frame_id_for(numbers, pos, cell)
    assert len(fid) == 16 and int(fid, 16) >= 0
    assert frame_id_for(numbers, pos + 1e-9, cell) == fid  # below extxyz print precision
    assert frame_id_for(numbers, pos - 1e-9, cell) == fid  # -0.0 normalised
    assert frame_id_for(numbers, pos + 1e-3, cell) != fid
    assert frame_id_for([14, 26], pos, cell) != fid


def test_extxyz_round_trip(tiny_frames: list[Frame], tiny_b20_path: Path) -> None:
    art = write_frames(tiny_frames, tiny_b20_path)
    assert art.kind == "frames" and art.path == str(tiny_b20_path)
    assert art.sha256 == sha256_file(tiny_b20_path) and art.bytes == tiny_b20_path.stat().st_size

    back = read_frames(tiny_b20_path)
    assert len(back) == len(tiny_frames)
    for a, b in zip(tiny_frames, back, strict=True):
        assert (a.frame_id, a.group_id, a.compound, a.config_type, a.parent_id) == (
            b.frame_id, b.group_id, b.compound, b.config_type, b.parent_id,
        )  # fmt: skip
        assert (a.label_source, a.energy_scale, a.weights, a.pbc) == (
            b.label_source, b.energy_scale, b.weights, b.pbc,
        )  # fmt: skip
        assert a.numbers == b.numbers
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-7)
        np.testing.assert_allclose(a.cell, b.cell, atol=1e-7)
        assert a.energy is not None and b.energy is not None
        assert abs(a.energy - b.energy) < 1e-7
        np.testing.assert_allclose(a.forces, b.forces, atol=1e-7)
        np.testing.assert_allclose(a.stress, b.stress, atol=1e-7)
        if a.magmoms is None:
            assert b.magmoms is None
        else:
            np.testing.assert_allclose(a.magmoms, b.magmoms, atol=1e-7)
        assert a.temperature_K == b.temperature_K
        if a.total_magnetization is None:
            assert b.total_magnetization is None
        else:
            assert b.total_magnetization is not None
            assert abs(a.total_magnetization - b.total_magnetization) < 1e-7
        assert a.info == b.info  # e.g. lineage / qe_unit survive; no metadata leaks into info
    # a second round trip is a fixed point (no info-key growth)
    twice = tiny_b20_path.with_name("twice.extxyz")
    write_frames(back, twice)
    assert [f.model_dump() for f in read_frames(twice)] == [f.model_dump() for f in back]


def test_mace_readable_keys(tiny_b20_path: Path) -> None:
    """What MACE's loader sees: energy/forces on the calculator, weights in info."""
    images = ase_read(str(tiny_b20_path), index=":", format="extxyz")
    qe = [a for a in images if a.info.get("energy_scale") == "qe"]
    assert len(qe) == 3
    for atoms in qe:
        assert atoms.info["config_energy_weight"] == 0.0
        assert atoms.calc is not None and atoms.get_forces().shape == (8, 3)
        assert np.asarray(atoms.get_stress()).shape == (6,)


def test_read_foreign_extxyz_with_lenient_defaults(tmp_path: Path) -> None:
    atoms = Atoms(
        "Fe4Si4",
        positions=np.random.default_rng(1).random((8, 3)) * 4,
        cell=np.eye(3) * 4.5,
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(atoms, energy=-40.0, forces=np.zeros((8, 3)))
    atoms.info["mp_id"] = "mp-871"
    path = tmp_path / "foreign.extxyz"
    ase_write(str(path), [atoms], format="extxyz")
    (frame,) = read_frames(path)
    assert frame.compound == "FeSi" and frame.config_type == "relax"
    assert frame.parent_id == frame.frame_id
    assert frame.group_id == f"FeSi/relax/{frame.frame_id}"
    assert frame.label_source == "none" and frame.energy_scale == "none"
    assert frame.energy == -40.0 and frame.info == {"mp_id": "mp-871"}
    (frame2,) = read_frames(
        path, defaults={"config_type": "wbm", "label_source": "mptrj", "energy_scale": "mp"}
    )
    assert (frame2.config_type, frame2.label_source, frame2.energy_scale) == ("wbm", "mptrj", "mp")


def test_non_scalar_info_is_json_encoded(tiny_frames: list[Frame], tmp_path: Path) -> None:
    frame = tiny_frames[0].model_copy(update={"info": {"tags": ["a", "b"], "n": 2, "flag": True}})
    atoms = frame_to_atoms(frame)
    assert (
        atoms.info["tags"] == '["a", "b"]' and atoms.info["n"] == 2 and atoms.info["flag"] is True
    )
    (back,) = read_frames(write_frames([frame], tmp_path / "one.extxyz").path)
    assert back.info == {"tags": '["a", "b"]', "n": 2, "flag": True}


def test_write_empty(tmp_path: Path) -> None:
    art = write_frames([], tmp_path / "empty.extxyz")
    assert art.bytes == 0 and read_frames(art.path) == []


def test_resolve_head_accepts_the_foundation_lowercase_head() -> None:
    """MACE-MPA-0 / MP-0 name their single head 'default'; fine-tuned models write 'Default'."""
    from b20mlip.io import resolve_head

    assert resolve_head("Default", ["default"]) == "default"
    assert resolve_head("Default", ["Default", "pt_head"]) == "Default"
    assert resolve_head("pt_head", ["pt_head", "Default"]) == "pt_head"
    assert resolve_head("Default", None) == "Default"
    with pytest.raises(ValueError, match="not 'pt_head'"):
        resolve_head("pt_head", ["default"])


def test_contract_head_labels_the_foundation_head_default() -> None:
    """Published numbers carry the CONTRACTS label (gate A3), whatever the model calls its head."""
    from b20mlip.io import contract_head
    from b20mlip.md.common import number_meta

    assert contract_head("default") == "Default" and contract_head("Default") == "Default"
    assert contract_head("pt_head") == "pt_head"
    meta = number_meta(
        provenance={"e0_source": "foundation"}, head="default", n=20, seed=0,
        engine="lammps-vs-ase", ensemble=None, T=None, ci95=None, ci95_reason="deterministic",
    )  # fmt: skip
    assert meta["head"] == "Default"
