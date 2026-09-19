"""dft prep / run / collect end to end with the LocalExecutor and the fake pw.x (unit protocol)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import units
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.config import Settings
from b20mlip.dft import qe, stages
from b20mlip.executors import JobHandle, LocalExecutor, SlurmExecutor
from b20mlip.io import read_frames, write_frames
from b20mlip.models import Frame, SlurmInfo
from b20mlip.provenance import read_manifest, run_stage

GOLDEN_E = -1178.40219853 * units.Ry


@pytest.fixture
def three_frames(b20_frame) -> list[Frame]:
    return [
        b20_frame("MnSi"),
        b20_frame("MnSi", "strain", scale=1.03),
        b20_frame("MnSi", "rattle", rattle=0.05),
    ]


def _run(cfg: Settings, stage: str, fn, **kw):
    return run_stage(stage, cfg, fn, executor=LocalExecutor(cfg), **kw)  # type: ignore[arg-type]


def test_prep_run_collect_end_to_end(
    dft_settings: Settings,
    three_frames: list[Frame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = dft_settings
    frames_path = tmp_path / "frames.extxyz"
    write_frames(three_frames, frames_path)
    root = cfg.paths.dft_dir / "r0"
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_PW_CALLS", str(calls))

    prep = _run(cfg, "dft.prep", stages.prep, frames=frames_path, out=root)
    assert prep.status == "ok" and prep.summary == {"n_units": 3, "units_root": str(root)}
    ids = qe.list_units(root)
    assert ids == [f.frame_id for f in three_frames]
    manifest = read_manifest(prep.manifest_path)
    assert manifest.extras["n_units"] == 3 and manifest.inputs[0].kind == "frames"
    assert manifest.extras["pseudo_md5s"]["Mn"] == "82ef2b46521d7a7d9e736dc3972e4928"

    run = _run(cfg, "dft.run", stages.run, units=root)
    assert run.status == "ok", run.summary
    assert (
        run.summary["n_submitted"] == 3 and run.summary["n_done"] == 3 and run.summary["done"] == 3
    )
    assert calls.read_text().split() == ids  # one fake pw.x call per unit, in order
    for uid in ids:
        unit = root / uid
        assert (
            (unit / "pw.out").is_file()
            and (unit / ".done").is_file()
            and qe.unit_status(unit) == "done"
        )
        marker = json.loads((unit / ".done").read_text())
        assert marker["unit"] == uid and marker["state"] == "done" and marker["returncode"] == 0
        assert not (unit / "tmp").exists()
    manifest = read_manifest(run.manifest_path)
    assert manifest.status == "ok" and manifest.extras["pw_version"] == "7.3"
    assert manifest.extras["n_units"] == 3 and manifest.extras["n_failed"] == 0
    assert manifest.extras["n_branch_rejected"] == 0 and manifest.extras["template"] == "qe_array"
    assert manifest.extras["pseudo_md5s"] == {
        "Mn": "82ef2b46521d7a7d9e736dc3972e4928",
        "Si": "0b0bb1205258b0d07b9f9672cf965d36",
    }
    assert manifest.extras["job_ids"][0].startswith("local:qe-r0-")
    assert {Path(a.path).name for a in manifest.outputs} == {"handle.json", "counts.json"}
    workdir = Path(manifest.extras["job_workdir"])
    assert (workdir / "units.txt").read_text().split() == ids and (workdir / "script.sh").is_file()
    spec = json.loads((Path(run.manifest_path).parent / "spec.json").read_text())
    assert (
        spec["env"]["B20_UNITS_ROOT"] == str(root.resolve())
        and spec["env"]["QE_CMD"] == cfg.cluster.qe_cmd
    )

    # resume: nothing pending, no new pw.x calls
    again = _run(cfg, "dft.run", stages.run, units=root)
    assert (
        again.status == "ok"
        and again.summary["n_submitted"] == 0
        and again.summary["note"] == "nothing pending"
    )
    assert calls.read_text().split() == ids

    out = tmp_path / "labelled.extxyz"
    coll = _run(cfg, "dft.collect", stages.collect, units=root, out=out)
    assert coll.status == "ok" and coll.summary["n_frames"] == 3 and coll.summary["done"] == 3
    labelled = read_frames(out)
    assert [f.frame_id for f in labelled] == ids
    assert all(f.energy == pytest.approx(GOLDEN_E) for f in labelled)
    assert all(f.label_source == "qe" and f.energy_scale == "qe" for f in labelled)
    assert all(f.info["dft_converged"] and f.info["dft_unit_id"] == f.frame_id for f in labelled)
    assert labelled[1].config_type == "strain" and labelled[2].temperature_K is None
    manifest = read_manifest(coll.manifest_path)
    assert manifest.extras["counts"] == {
        "planned": 3,
        "done": 3,
        "failed": 0,
        "unconverged": 0,
        "missing": 0,
    }
    assert {Path(a.path).name for a in manifest.outputs} == {"labelled.extxyz", "counts.json"}

    # a unit that stops converging: rerun only that unit, it is marked failed, collect drops it
    victim = ids[1]
    (root / victim / ".done").unlink()
    monkeypatch.setenv("FAKE_PW_FAIL_UNITS", victim)
    failed = _run(cfg, "dft.run", stages.run, units=root)
    assert (
        failed.status == "failed"
        and failed.summary["n_submitted"] == 1
        and failed.summary["n_failed"] == 1
    )
    assert qe.unit_status(root / victim) == "failed" and calls.read_text().split() == ids + [victim]
    assert json.loads((root / victim / ".failed").read_text())["state"] == "failed"
    coll2 = _run(cfg, "dft.collect", stages.collect, units=root, out=out)
    assert (
        coll2.status == "ok"
        and coll2.summary["n_frames"] == 2
        and coll2.summary["unconverged"] == 1
    )
    assert [f.frame_id for f in read_frames(out)] == [ids[0], ids[2]]
    manifest = read_manifest(coll2.manifest_path)
    assert manifest.extras["n_failed"] == 1

    # retry after clearing the failure: the failed marker is cleared and the unit reruns
    monkeypatch.delenv("FAKE_PW_FAIL_UNITS")
    fixed = _run(cfg, "dft.run", stages.run, units=root)
    assert fixed.status == "ok" and qe.unit_status(root / victim) == "done"


def test_run_limit_dry_run_and_errors(
    dft_settings: Settings,
    three_frames: list[Frame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = dft_settings
    root = tmp_path / "units"
    ids = qe.plan_units(three_frames, root, cfg)
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_PW_CALLS", str(calls))

    dry = run_stage(
        "dft.run", cfg, stages.run, executor=LocalExecutor(cfg), dry_run=True, units=root
    )
    assert dry.status == "partial" and dry.summary["dry_run"] == 1 and dry.outputs == []
    assert (Path(dry.manifest_path).parent / "spec.json").is_file() and not calls.exists()
    assert read_manifest(dry.manifest_path).extras["n_units"] == 3

    one = _run(cfg, "dft.run", stages.run, units=root, limit=1, parallel=2)
    assert one.status == "ok" and one.summary["n_submitted"] == 1 and one.summary["missing"] == 2
    assert calls.read_text().split() == ids[:1]
    rest = _run(cfg, "dft.run", stages.run, units=root, limit=5)
    assert (
        rest.status == "ok"
        and rest.summary["n_submitted"] == 2
        and calls.read_text().split() == ids
    )

    empty = _run(cfg, "dft.run", stages.run, units=tmp_path / "nothing")
    assert empty.status == "failed" and "no planned units" in empty.summary["error"]
    no_cmd = cfg.model_copy(update={"cluster": cfg.cluster.model_copy(update={"qe_cmd": None})})
    qe.plan_units(three_frames[:1], tmp_path / "u2", cfg)
    res = _run(no_cmd, "dft.run", stages.run, units=tmp_path / "u2")
    assert res.status == "failed" and "cluster.qe_cmd is unset" in res.summary["error"]
    res = run_stage("dft.run", cfg, stages.run, executor=None, units=tmp_path / "u2")
    assert res.status == "failed" and "no executor" in res.summary["error"]
    res = _run(
        cfg, "dft.collect", stages.collect, units=tmp_path / "nothing", out=tmp_path / "o.extxyz"
    )
    assert res.status == "failed" and "no planned units" in res.summary["error"]
    nothing_done = _run(
        cfg, "dft.collect", stages.collect, units=tmp_path / "u2", out=tmp_path / "o.extxyz"
    )
    assert nothing_done.status == "failed" and nothing_done.summary["missing"] == 1
    assert not (tmp_path / "o.extxyz").exists()
    res = _run(cfg, "dft.prep", stages.prep, frames=tmp_path / "missing.extxyz", out=tmp_path / "p")
    assert res.status == "failed"
    (tmp_path / "empty.extxyz").write_text("")
    res = _run(cfg, "dft.prep", stages.prep, frames=tmp_path / "empty.extxyz", out=tmp_path / "p")
    assert res.status == "failed" and "no frames" in res.summary["error"]


def test_no_wait_and_slurm_accounting(
    dft_settings: Settings, three_frames: list[Frame], tmp_path: Path
) -> None:
    """A SLURM-like executor: `wait` returns accounting; outputs arrive with `cluster sync`."""
    cfg = dft_settings
    root = tmp_path / "units"
    ids = qe.plan_units(three_frames, root, cfg)

    class FakeSlurm:
        def __init__(self) -> None:
            self.specs = []

        def submit(self, spec):
            self.specs.append(spec)
            return JobHandle(job_ids=["4242"], workdir="/gscratch/b20/b20-mlip/jobs/x")

        def wait(self, handle, poll_s=60):
            return SlurmInfo(
                job_ids=handle.job_ids,
                account="a",
                partition="p",
                nodes=1,
                wall="00:10:00",
                units_done=2,
                units_failed=1,
            )

        def fetch(self, handle, dest):
            return []

    ex = FakeSlurm()
    res = run_stage("dft.run", cfg, stages.run, executor=ex, units=root)  # type: ignore[arg-type]
    assert res.status == "partial" and res.summary["n_done"] == 2 and res.summary["n_failed"] == 1
    assert res.summary["missing"] == 3  # nothing local yet
    manifest = read_manifest(res.manifest_path)
    assert (
        manifest.slurm is not None
        and manifest.slurm.job_ids == ["4242"]
        and manifest.extras["job_ids"] == ["4242"]
    )
    assert ex.specs[0].units == ids and ex.specs[0].env["B20_UNITS_ROOT"] == str(root.resolve())
    nowait = run_stage("dft.run", cfg, stages.run, executor=ex, units=root, wait=False)  # type: ignore[arg-type]
    assert nowait.status == "partial" and nowait.summary["waited"] == 0
    assert read_manifest(nowait.manifest_path).slurm is None


def test_units_root_for(
    dft_settings: Settings, slurm_settings: Settings, repo: Path, tmp_path: Path
) -> None:
    local = LocalExecutor(dft_settings)
    assert stages.units_root_for(local, "dft/r0", dft_settings) == str(
        (Path.cwd() / "dft/r0").resolve()
    )
    slurm = SlurmExecutor(slurm_settings, runner=lambda argv: None)  # type: ignore[arg-type]
    assert (
        stages.units_root_for(slurm, repo / "dft" / "r0", slurm_settings)
        == "/gscratch/b20/b20-mlip/dft/r0"
    )
    assert (
        stages.units_root_for(slurm, tmp_path / "elsewhere", slurm_settings)
        == "/gscratch/b20/b20-mlip/elsewhere"
    )
    no_scratch = slurm_settings.model_copy(
        update={"cluster": slurm_settings.cluster.model_copy(update={"scratch": None})}
    )
    with pytest.raises(ValueError, match="cluster bootstrap"):
        stages.units_root_for(SlurmExecutor(no_scratch, runner=lambda argv: None), "x", no_scratch)  # type: ignore[arg-type]


def test_cli_prep_run_collect(
    dft_settings: Settings, three_frames: list[Frame], tmp_path: Path, fake_qe_cmd: str
) -> None:
    runner = CliRunner()
    frames_path = tmp_path / "frames.extxyz"
    write_frames(three_frames, frames_path)
    root = tmp_path / "r0"
    common = [
        "--set", f"paths.runs_dir={tmp_path / 'runs'}",
        "--set", f"paths.dft_dir={tmp_path / 'dft'}",
        "--set", f"cluster.qe_cmd={fake_qe_cmd}",
        "--set", f"dft.pseudo_dir={tmp_path / 'pseudos'}",
    ]  # fmt: skip
    res = runner.invoke(
        app, [*common, "dft", "prep", "--frames", str(frames_path), "--out", str(root)]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert (
        payload["stage"] == "dft.prep"
        and payload["status"] == "ok"
        and payload["summary"]["n_units"] == 3
    )
    res = runner.invoke(app, [*common, "--dry-run", "dft", "run", "--units", str(root)])
    assert res.exit_code == 1 and '"status": "partial"' in res.output
    res = runner.invoke(app, [*common, "dft", "run", "--units", str(root), "--limit", "2"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["summary"]["n_submitted"] == 2
    res = runner.invoke(app, [*common, "dft", "run", "--units", str(root), "--parallel", "2"])
    assert res.exit_code == 0 and json.loads(res.output)["summary"]["n_submitted"] == 1
    out = tmp_path / "labelled.extxyz"
    res = runner.invoke(app, [*common, "dft", "collect", "--units", str(root), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert len(read_frames(out)) == 3 and str(out) in json.loads(res.output)["outputs"]
    res = runner.invoke(
        app, [*common, "dft", "collect", "--units", str(tmp_path / "nope"), "--out", str(out)]
    )
    assert res.exit_code == 1 and '"status": "failed"' in res.output
    assert runner.invoke(app, ["dft", "run"]).exit_code == 2  # missing --units: usage error
