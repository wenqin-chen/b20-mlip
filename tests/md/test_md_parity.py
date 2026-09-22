"""parity: the ASE-vs-LAMMPS gate on the tiny model's own forces (pass) and a perturbed copy
(fail), the LAMMPS single-point inputs + collection, and the md.parity stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from b20mlip.config import Settings
from b20mlip.executors import LocalExecutor
from b20mlip.io import write_frames
from b20mlip.md import parity
from b20mlip.models import Frame
from b20mlip.provenance import read_manifest, run_stage

from .conftest import GOLDEN_PARITY


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PARITY.read_text())


@pytest.fixture
def frames(b20_frames) -> list[Frame]:  # type: ignore[no-untyped-def]
    return b20_frames("MnSi", n=3)


@pytest.fixture
def frames_file(tmp_path: Path, frames: list[Frame]) -> Path:
    path = tmp_path / "parity_frames.extxyz"
    write_frames(frames, path)
    return path


@pytest.fixture
def reference(tiny_mace, frames: list[Frame]) -> dict:  # type: ignore[no-untyped-def]
    """The tiny model's own ASE energies/forces in the parity JSON layout."""
    return parity.ase_reference(None, frames, calc=tiny_mace.calculator())


def perturbed(reference: dict, golden: dict, frames: list[Frame]) -> dict:
    pert = golden["perturbation"]
    out = json.loads(json.dumps(reference))
    fid = frames[pert["frame_index"]].frame_id
    out[fid]["forces"][pert["atom"]][pert["component"]] += pert["dF_eVA"]
    out[fid]["energy"] += pert["dE_eV"]
    return out


# --- check ----------------------------------------------------------------------------------------


def test_check_passes_on_identical_forces_and_fails_on_the_golden_perturbation(
    tiny_mace, frames: list[Frame], reference: dict, golden: dict, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    calc = tiny_mace.calculator()
    ok = parity.check(tiny_mace.model_path, frames, reference, calc=calc)
    exp = golden["expected_passing"]
    assert ok["passed"] is True and ok["n_frames"] == 3 and ok["n_missing"] == 0
    assert ok["max_dF_eVA"] <= exp["max_dF_eVA_below"]
    assert ok["max_dE_eV_atom"] <= exp["max_dE_eV_atom_below"]
    assert ok["tol_f"] == 1e-3 and ok["tol_e"] == 1e-4 and ok["enough_frames"] is False
    assert [r["frame_id"] for r in ok["frames"]] == [f.frame_id for f in frames]
    assert all(r["passed"] and r["natoms"] == 8 for r in ok["frames"])
    assert ok["head"] == "Default" and ok["model_path"] == str(tiny_mace.model_path)

    bad = perturbed(reference, golden, frames)
    path = tmp_path / "lammps.json"
    path.write_text(json.dumps({"@meta": {"ignored": True}, **bad}))
    fail = parity.check(None, frames, path, calc=calc)
    exp = golden["expected_failing"]
    assert fail["passed"] is False
    assert fail["max_dF_eVA"] == pytest.approx(exp["max_dF_eVA"], abs=1e-12)
    assert fail["max_dE_eV_atom"] == pytest.approx(exp["max_dE_eV_atom"], abs=1e-12)
    rows = {r["frame_id"]: r for r in fail["frames"]}
    victim = frames[golden["perturbation"]["frame_index"]].frame_id
    assert not rows[victim]["passed"] and rows[victim]["rms_dF_eVA"] > 0
    assert all(rows[f.frame_id]["passed"] for f in frames if f.frame_id != victim)
    # loose tolerances let the perturbed copy pass; a missing frame never passes
    assert parity.check(None, frames, bad, tol_f=0.01, tol_e=0.001, calc=calc)["passed"]
    partial = {k: v for k, v in reference.items() if k != frames[0].frame_id}
    miss = parity.check(None, frames, partial, calc=calc)
    assert miss["passed"] is False and miss["missing"] == [frames[0].frame_id]
    assert miss["n_frames"] == 2 and miss["n_missing"] == 1
    with pytest.raises(ValueError, match="shapes"):
        parity.compare(
            reference, {k: {**v, "forces": v["forces"][:4]} for k, v in reference.items()}
        )
    with pytest.raises(ValueError):
        parity.check(None, frames, [1, 2], calc=calc)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_path or calc"):
        parity.ase_reference(None, frames)
    with pytest.raises(TypeError):
        parity._as_atoms([object()])
    empty = parity.compare({}, {})
    assert empty["passed"] is False and np.isnan(empty["max_dF_eVA"]) and empty["n_frames"] == 0


def test_ase_reference_accepts_atoms_and_files(
    tiny_mace, frames: list[Frame], frames_file: Path, reference: dict
) -> None:  # type: ignore[no-untyped-def]
    from b20mlip.io import frame_to_atoms

    calc = tiny_mace.calculator()
    from_file = parity.ase_reference(None, frames_file, calc=calc)
    assert from_file.keys() == reference.keys()
    for fid in reference:
        assert from_file[fid]["energy"] == pytest.approx(reference[fid]["energy"], abs=1e-9)
    atoms = [frame_to_atoms(f) for f in frames]
    for a in atoms:
        a.calc = None
    from_atoms = parity.ase_reference(None, atoms, calc=calc)
    assert from_atoms.keys() == reference.keys()
    anonymous = parity.ase_reference(None, [atoms[0].copy()], calc=calc)
    assert list(anonymous) == ["frame000"] or list(anonymous)[0] == frames[0].frame_id
    # the calculator is built from the model when none is given
    built = parity.ase_reference(tiny_mace.model_path, frames[:1], cfg=Settings.model_validate({}))
    assert built[frames[0].frame_id]["energy"] == pytest.approx(
        reference[frames[0].frame_id]["energy"], abs=1e-9
    )


# --- LAMMPS single points -------------------------------------------------------------------------


def test_parity_inputs_and_collect(tmp_path: Path, frames: list[Frame], reference: dict) -> None:
    ids = parity.parity_lammps_inputs(frames, "../../m-lammps.pt", tmp_path / "frames")
    assert ids == [f.frame_id for f in frames]
    unit = tmp_path / "frames" / ids[0]
    text = (unit / "in.parity").read_text()
    assert text.startswith(f"# b20mlip-parity frame_id={ids[0]} natoms=8 compound=MnSi\n")
    for line in (
        "units           metal", "atom_style      atomic", "atom_modify     map yes",
        "newton          on",
        "read_data       data.lmp", "pair_style      mace no_domain_decomposition",
        "pair_coeff      * * ../../m-lammps.pt Si Mn", "thermo_style    custom step pe",
        "dump            1 all custom 1 forces.dump id type fx fy fz", "run             0",
    ):  # fmt: skip
        assert line in text, line
    assert (unit / "data.lmp").is_file() and "2 atom types" in (unit / "data.lmp").read_text()
    units_json = json.loads((tmp_path / "frames" / "units.json").read_text())
    assert units_json == {"units": ids, "elements": ["Si", "Mn"]}
    explicit = parity.parity_lammps_inputs(
        frames[:1], "m.pt", tmp_path / "f2", elements=["Mn", "Si"]
    )
    assert (
        "pair_coeff      * * m.pt Mn Si"
        in (tmp_path / "f2" / explicit[0] / "in.parity").read_text()
    )
    with pytest.raises(ValueError):
        parity.parity_lammps_inputs([], "m.pt", tmp_path / "f3")

    # simulate two finished units (log + forces.dump) and one without output
    for fid in ids[:2]:
        ref = reference[fid]
        (tmp_path / "frames" / fid / "log.lammps").write_text(
            "LAMMPS (22 Jul 2025 - Update 6)\n   Step         PotEng     \n"
            f"         0 {ref['energy']:.15g}\n"
            "Loop time of 1e-06 on 1 procs for 0 steps with 8 atoms\n"
        )
        lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", "8", "ITEM: BOX BOUNDS pp pp pp",
                 "0 1", "0 1", "0 1", "ITEM: ATOMS id type fx fy fz"]  # fmt: skip
        for i, f in reversed(list(enumerate(ref["forces"]))):  # unsorted ids on purpose
            lines.append(f"{i + 1} 1 {f[0]:.15g} {f[1]:.15g} {f[2]:.15g}")
        (tmp_path / "frames" / fid / "forces.dump").write_text("\n".join(lines) + "\n")
    collected = parity.collect_parity(tmp_path, tmp_path / "out.json")
    assert collected["@missing"] == [ids[2]] and set(collected) == {ids[0], ids[1], "@missing"}
    for fid in ids[:2]:
        assert collected[fid]["energy"] == pytest.approx(reference[fid]["energy"], abs=1e-12)
        assert np.allclose(collected[fid]["forces"], reference[fid]["forces"], atol=1e-12)
        assert collected[fid]["natoms"] == 8
    assert json.loads((tmp_path / "out.json").read_text()) == collected
    assert parity.load_lammps_json(tmp_path / "out.json").keys() == {ids[0], ids[1]}
    direct = parity.collect_parity(tmp_path / "frames")
    assert direct[ids[0]] == collected[ids[0]]
    with pytest.raises(ValueError, match="no atoms"):
        parity.parse_forces_dump(tmp_path / "frames" / "units.json")
    (tmp_path / "bad.log").write_text(
        "   Step Temp\n 0 1\nLoop time of 1 on 1 procs for 0 steps with 8 atoms\n"
    )
    with pytest.raises(ValueError, match="pe column"):
        parity.parse_single_point_energy(tmp_path / "bad.log")
    with pytest.raises(ValueError, match="no thermo row"):
        parity.parse_single_point_energy(tmp_path / "frames" / "units.json")
    with pytest.raises(ValueError, match="must be an object"):
        parity.load_lammps_json(_write(tmp_path / "list.json", "[]"))
    assert parity.parity_unit_script("lmp").startswith(
        'cd "$B20_WORKDIR"\nfor d in frames/*/; do\n'
    )
    assert "lmp -in in.parity -log log.lammps" in parity.parity_unit_script("lmp")


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# --- the stage ------------------------------------------------------------------------------------


def _numbers(result) -> dict:  # type: ignore[no-untyped-def]
    return json.loads((Path(result.manifest_path).parent / "numbers.json").read_text())


def test_stage_with_a_lammps_json(
    md_settings: Settings,
    tiny_mace,
    frames: list[Frame],
    frames_file: Path,
    reference: dict,
    golden: dict,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    calc = tiny_mace.calculator()
    good = tmp_path / "good.json"
    good.write_text(json.dumps(reference))
    result = run_stage(
        "md.parity", md_settings, parity.run, seed=0, model=tiny_mace.model_path,
        frames=frames_file, lammps_json=good, calc=calc,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    s = result.summary
    assert s["parity_passed"] == 1 and s["n_frames"] == 3 and s["enough_frames"] == 0
    floor = golden["expected_passing"]
    assert s["max_dF_eVA"] <= floor["max_dF_eVA_below"] and s["result"] == "parity.json"
    assert s["max_dE_eV_atom"] <= floor["max_dE_eV_atom_below"]
    numbers = _numbers(result)
    assert numbers["parity.tiny_b20.passed"] == 1 and numbers["parity.tiny_b20.n_frames"] == 3
    assert (
        numbers["parity.tiny_b20.max_dF_eVA"] <= 1e-9
        and numbers["md.parity.tiny_b20.max_dE_eV_atom"] <= 1e-9
    )
    meta = numbers["parity.tiny_b20.passed@meta"]
    assert meta["reference"]["code"] == "mace" and meta["n"] == 3 and meta["seed"] == 0
    assert meta["ci95"] is None and meta["ci95_reason"] and meta["enough_frames"] is False
    assert (
        meta["engine"] == "lammps-vs-ase"
        and meta["head"] == "Default"
        and meta["tol_f_eVA"] == 1e-3
    )
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["parity_passed"] is True and manifest.extras["n_compared"] == 3
    assert {Path(a.path).name for a in manifest.outputs} == {"parity.json", "numbers.json"}
    assert {Path(a.path).name for a in manifest.inputs} == {
        frames_file.name,
        tiny_mace.model_path.name,
        "good.json",
    }
    verdict = json.loads((Path(result.manifest_path).parent / "parity.json").read_text())
    assert verdict["run_id"] == result.run_id and verdict["lammps_source"] == str(good)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(perturbed(reference, golden, frames)))
    failed_gate = run_stage(
        "md.parity", md_settings, parity.run, model=tiny_mace.model_path, frames=frames_file,
        lammps_json=bad, calc=calc, label="B2",
    )  # fmt: skip
    assert (
        failed_gate.status == "ok" and failed_gate.summary["parity_passed"] == 0
    )  # negative results ship
    numbers = _numbers(failed_gate)
    assert numbers["parity.B2.passed"] == 0 and numbers["parity.B2.max_dF_eVA"] == pytest.approx(
        0.0025
    )
    assert numbers["parity.B2.passed@meta"]["model_label"] == "B2"
    report_numbers = pytest.importorskip("b20mlip.report.numbers")
    parsed = report_numbers.parse_numbers_file(numbers)
    assert parsed["parity.B2.passed"][0] == 0.0 and "md.parity.B2.max_dF_eVA" in parsed


def test_stage_runs_the_lammps_single_points_through_the_local_executor(
    lammps_settings: Settings,
    tiny_mace,
    frames: list[Frame],
    frames_file: Path,
    reference: dict,
    golden: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = lammps_settings
    calc = tiny_mace.calculator()
    good = tmp_path / "fake_lammps_side.json"
    good.write_text(json.dumps(reference))
    monkeypatch.setenv("FAKE_LMP_PARITY_JSON", str(good))
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_LMP_CALLS", str(calls))
    result = run_stage(
        "md.parity", cfg, parity.run, executor=LocalExecutor(cfg), model=tiny_mace.model_path,
        frames=frames_file, calc=calc,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    assert result.summary["parity_passed"] == 1 and result.summary["n_missing"] == 0
    run_dir = Path(result.manifest_path).parent
    manifest = read_manifest(result.manifest_path)
    job_dir = Path(manifest.extras["job_dir"])
    assert job_dir == LocalExecutor(cfg).workdir(f"md-parity-{result.run_id}")
    assert (job_dir / tiny_mace.lammps_path.name).is_file()
    assert sorted(p.name for p in (job_dir / "frames").iterdir() if p.is_dir()) == sorted(
        f.frame_id for f in frames
    )
    assert len(calls.read_text().splitlines()) == 3  # one fake lmp call per frame
    assert (
        manifest.extras["units"] == [f.frame_id for f in frames] and manifest.extras["gpu"] is False
    )
    names = {Path(a.path).name for a in manifest.outputs}
    assert {"parity.json", "parity_lammps.json", "numbers.json", "handle.json"} <= names
    collected = json.loads((run_dir / "parity_lammps.json").read_text())
    assert set(collected) == {f.frame_id for f in frames}
    assert (run_dir / "job" / "frames" / frames[0].frame_id / "forces.dump").is_file()
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["units"] == ["parity"] and spec["resources"]["template"] == "lammps"
    assert "input" not in spec["resources"] and "for d in frames/*/" in spec["script"]

    # the perturbed LAMMPS side fails the gate; a frame the fake cannot answer is "missing"
    bad = tmp_path / "bad_side.json"
    bad.write_text(json.dumps(perturbed(reference, golden, frames)))
    monkeypatch.setenv("FAKE_LMP_PARITY_JSON", str(bad))
    failed_gate = run_stage(
        "md.parity", cfg, parity.run, executor=LocalExecutor(cfg), model=tiny_mace.model_path,
        frames=frames_file, calc=calc,
    )  # fmt: skip
    assert failed_gate.status == "ok" and failed_gate.summary["parity_passed"] == 0
    assert failed_gate.summary["max_dF_eVA"] == pytest.approx(
        golden["expected_failing"]["max_dF_eVA"]
    )
    partial_side = tmp_path / "partial_side.json"
    partial_side.write_text(
        json.dumps({k: v for k, v in reference.items() if k != frames[1].frame_id})
    )
    monkeypatch.setenv("FAKE_LMP_PARITY_JSON", str(partial_side))
    missing = run_stage(
        "md.parity", cfg, parity.run, executor=LocalExecutor(cfg), model=tiny_mace.model_path,
        frames=frames_file, calc=calc,
    )  # fmt: skip
    assert missing.status == "ok" and missing.summary["parity_passed"] == 0
    assert missing.summary["n_missing"] == 1 and missing.summary["n_frames"] == 2
    assert read_manifest(missing.manifest_path).extras["lammps_missing"] == [frames[1].frame_id]
    # --resume: a partial (no-wait) run whose job directory was synced later is re-collected
    # without resubmitting (RunContext reuses the latest failed/partial run of this config)
    import shutil

    from b20mlip.executors import JobHandle, JobSpec

    class NoWait:
        def submit(self, spec: JobSpec) -> JobHandle:
            return JobHandle(job_ids=["9"], workdir="/gscratch/x")

        def wait(self, handle: JobHandle, poll_s: float = 60):  # type: ignore[no-untyped-def]
            raise AssertionError("not waited")

        def fetch(self, handle: JobHandle, dest: Path) -> list:
            raise AssertionError("not fetched")

    partial = run_stage(
        "md.parity", cfg, parity.run, executor=NoWait(), model=tiny_mace.model_path,
        frames=frames_file, calc=calc, wait=False,
    )  # fmt: skip
    assert partial.status == "partial"
    shutil.copytree(  # the unknown executor staged its inputs under <run dir>/job already
        Path(missing.manifest_path).parent / "job",
        Path(partial.manifest_path).parent / "job",
        dirs_exist_ok=True,
    )
    calls.write_text("")
    resumed = run_stage(
        "md.parity", cfg, parity.run, executor=NoWait(), resume=True, model=tiny_mace.model_path,
        frames=frames_file, calc=calc,
    )  # fmt: skip
    assert resumed.status == "ok" and resumed.run_id == partial.run_id and calls.read_text() == ""
    assert resumed.summary["n_missing"] == 1 and resumed.summary["parity_passed"] == 0
    assert read_manifest(resumed.manifest_path).extras["resumed"] is True


def test_stage_without_lammps_stages_inputs_and_ends_partial(
    md_settings: Settings, tiny_mace, frames: list[Frame], frames_file: Path, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    result = run_stage(
        "md.parity", md_settings, parity.run, executor=LocalExecutor(md_settings),
        model=tiny_mace.model_path, frames=frames_file,
    )  # fmt: skip
    assert result.status == "partial" and result.summary["staged"] == 3
    assert (
        "lammps_cmd not found" in result.summary["note"]
        and "--lammps-json" in result.summary["note"]
    )
    inputs = Path(result.summary["job_dir"])
    assert inputs == Path(result.manifest_path).parent / "parity_inputs"
    assert (inputs / "frames" / frames[0].frame_id / "in.parity").is_file()
    assert (inputs / tiny_mace.lammps_path.name).is_file()
    dry = run_stage(
        "md.parity", md_settings, parity.run, executor=LocalExecutor(md_settings), dry_run=True,
        model=tiny_mace.model_path, frames=frames_file,
    )  # fmt: skip
    assert dry.status == "partial" and dry.summary["planned"] == 3 and dry.outputs == []
    no_exec = run_stage(
        "md.parity",
        md_settings,
        parity.run,
        executor=None,
        model=tiny_mace.model_path,
        frames=frames_file,
    )
    assert no_exec.status == "failed" and "no executor" in no_exec.summary["error"]
    absent = run_stage(
        "md.parity",
        md_settings,
        parity.run,
        model=tiny_mace.model_path,
        frames=tmp_path / "none.extxyz",
    )
    assert absent.status == "failed" and "frames file not found" in absent.summary["error"]
    (tmp_path / "empty.extxyz").write_text("")
    empty = run_stage(
        "md.parity",
        md_settings,
        parity.run,
        model=tiny_mace.model_path,
        frames=tmp_path / "empty.extxyz",
    )
    assert empty.status == "failed" and "no frames" in empty.summary["error"]
    no_export = run_stage(
        "md.parity", md_settings, parity.run, executor=LocalExecutor(md_settings),
        model=tmp_path / "other.model", frames=frames_file,
    )  # fmt: skip
    assert no_export.status == "failed" and "b20mlip export" in no_export.summary["error"]


def test_no_wait_with_a_slurm_like_executor(
    lammps_settings: Settings, tiny_mace, frames_file: Path
) -> None:  # type: ignore[no-untyped-def]
    from b20mlip.executors import JobHandle, JobSpec

    class FakeSlurm:
        specs: list[JobSpec] = []

        def submit(self, spec: JobSpec) -> JobHandle:
            self.specs.append(spec)
            return JobHandle(job_ids=["7"], workdir="/gscratch/x")

        def wait(self, handle: JobHandle, poll_s: float = 60):  # type: ignore[no-untyped-def]
            raise AssertionError("not waited")

        def fetch(self, handle: JobHandle, dest: Path) -> list:
            raise AssertionError("not fetched")

    result = run_stage(
        "md.parity", lammps_settings, parity.run, executor=FakeSlurm(), model=tiny_mace.model_path,
        frames=frames_file, wait=False, gpu=True,
    )  # fmt: skip
    assert (
        result.status == "partial"
        and result.summary["submitted"] == 3
        and result.summary["waited"] == 0
    )
    spec = FakeSlurm.specs[-1]
    assert spec.resources["gpu"] is True and spec.env["LAMMPS_CMD"].endswith("neigh half")
    assert spec.resources["time"] == "00:30:00" and spec.units == ["parity"]
