"""`b20mlip cluster bootstrap | sync | status` through the root CLI with a fake runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from b20mlip.cli import REGISTERED, app
from b20mlip.cluster import cli as cluster_cli
from b20mlip.cluster.remote import COMMANDS
from b20mlip.config import read_yaml
from b20mlip.executors import JobHandle
from b20mlip.provenance import find_runs

from .conftest import SCRATCH, FakeRunner, Scenario, simple_scenario, tillicum_scenario

runner = CliRunner()


def _use(monkeypatch: pytest.MonkeyPatch, fake: FakeRunner) -> None:
    monkeypatch.setattr(cluster_cli, "RUNNER_FACTORY", lambda: fake)


def _globals(tmp_path: Path) -> list[str]:
    return [
        "--set",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "--set",
        f"cluster.control_path={tmp_path / 'cm.sock'}",
    ]


def test_registered_and_help() -> None:
    assert REGISTERED["cluster"] == {"bootstrap", "sync", "status"}
    result = runner.invoke(app, ["cluster", "--help"])
    assert result.exit_code == 0
    for name in ("bootstrap", "sync", "status"):
        assert name in result.output
    assert "not implemented" not in runner.invoke(app, ["cluster", "bootstrap", "--help"]).output
    assert cluster_cli.default_runner().__name__ == "subprocess_runner"


def test_bootstrap_cli_ok_and_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yaml_path: Path
) -> None:
    fake = FakeRunner(simple_scenario())
    _use(monkeypatch, fake)
    result = runner.invoke(
        app,
        [
            *_globals(tmp_path),
            "cluster",
            "bootstrap",
            "--config-out",
            str(yaml_path),
            "--skip",
            "repo",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["stage"] == "cluster.bootstrap" and payload["status"] == "ok"
    assert payload["summary"]["account"] == "b20" and payload["summary"]["scratch"] == SCRATCH
    assert read_yaml(yaml_path)["cluster"]["partition_gpu"] == "gpu-h200"
    assert not any(c[0] == "rsync" for c in fake.calls)
    assert (
        find_runs("cluster.bootstrap", tmp_path / "runs")[0].extras["steps"]["repo"]["status"]
        == "skipped"
    )

    _use(monkeypatch, FakeRunner(Scenario(socket_ok=False)))
    result = runner.invoke(
        app, [*_globals(tmp_path), "cluster", "bootstrap", "--config-out", str(yaml_path)]
    )
    assert result.exit_code == 1
    assert "run `ssh -fN tillicum` (MFA) then retry" in result.output
    assert '"status": "failed"' in result.output

    _use(monkeypatch, FakeRunner(tillicum_scenario()))
    result = runner.invoke(
        app,
        [*_globals(tmp_path), "--dry-run", "cluster", "bootstrap", "--config-out", str(yaml_path)],
    )
    assert result.exit_code == 1 and '"status": "partial"' in result.output


def test_bootstrap_cli_flags_reach_the_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yaml_path: Path
) -> None:
    fake = FakeRunner(tillicum_scenario(uv_present=False))
    _use(monkeypatch, fake)
    result = runner.invoke(
        app,
        [
            *_globals(tmp_path),
            "cluster",
            "bootstrap",
            "--config-out",
            str(yaml_path),
            "--build-lammps",
            "--install-qe",
            "--partition",
            "gpu-h200-mig",
        ],
    )
    assert result.exit_code == 0, result.output
    data = read_yaml(yaml_path)["cluster"]
    assert data["qe_cmd"] == f"{SCRATCH}/qe/bin/pw.x"
    manifest = find_runs("cluster.bootstrap", tmp_path / "runs")[-1]
    assert manifest.extras["steps"]["lammps"]["partition"] == "gpu-h200-mig"
    assert manifest.extras["steps"]["lammps"]["cuda"] is True
    assert manifest.slurm is not None and manifest.slurm.job_ids == ["4242"]


def test_sync_and_status_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_root: Path
) -> None:
    job = remote_root / "jobs" / "qe_r0"
    (job / "units").mkdir(parents=True)
    (job / "units.txt").write_text("u1\n")
    (job / "units" / "u1.done").write_text("{}")
    dest = tmp_path / "runs" / "slurm" / "qe_r0"
    dest.mkdir(parents=True)
    (dest / "handle.json").write_text(
        JobHandle(job_ids=["4243"], workdir=f"{SCRATCH}/b20-mlip/jobs/qe_r0").model_dump_json()
    )
    sacct = "JobID|State|ExitCode|Elapsed|NodeList|\n4243|COMPLETED|0:0|00:03:10|g004|\n"
    scratch_args = ["--set", f"cluster.scratch={SCRATCH}"]

    _use(monkeypatch, FakeRunner(Scenario(sacct=[sacct]), remote_root=remote_root))
    result = runner.invoke(
        app, [*_globals(tmp_path), *scratch_args, "cluster", "sync", "--jobs", "qe_r0"]
    )
    assert result.exit_code == 0, result.output
    assert '"units_done": 1' in result.output and '"status": "ok"' in result.output

    _use(monkeypatch, FakeRunner(Scenario(socket_ok=False)))
    result = runner.invoke(app, [*_globals(tmp_path), *scratch_args, "cluster", "sync"])
    assert result.exit_code == 1 and "run `ssh -fN tillicum` (MFA) then retry" in result.output

    _use(monkeypatch, FakeRunner(Scenario()))
    result = runner.invoke(app, [*_globals(tmp_path), "cluster", "status"])
    assert result.exit_code == 0, result.output
    assert "JOBID" in result.output and "build_lammps" in result.output and "qe_r0" in result.output
    result = runner.invoke(app, [*_globals(tmp_path), "cluster", "status", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["local"]["qe_r0"] == {"units": 1, "done": 1, "failed": 0, "pending": 0}
    assert "commands" not in report

    _use(monkeypatch, FakeRunner(Scenario(socket_ok=False)))
    result = runner.invoke(app, [*_globals(tmp_path), "cluster", "status"])
    assert result.exit_code == 1 and "MFA" in result.output


def test_every_command_is_exercised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yaml_path: Path, remote_root: Path
) -> None:
    """The union of COMMANDS keys used by bootstrap (both presets), sync and status is complete."""
    from b20mlip.cluster.discover import bootstrap
    from b20mlip.cluster.status import show
    from b20mlip.cluster.sync import pull
    from b20mlip.config import Settings
    from b20mlip.provenance import run_stage

    cfg = Settings.model_validate(
        {
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "cluster": {"alias": "tillicum", "control_path": str(tmp_path / "cm.sock")},
        }
    )
    used: set[str] = set()

    def keys_of(result) -> set[str]:  # type: ignore[no-untyped-def]
        from b20mlip.provenance import read_manifest

        return {c["key"] for c in read_manifest(result.manifest_path).extras["commands"]}

    used |= keys_of(
        run_stage(
            "cluster.bootstrap",
            cfg,
            bootstrap,
            runner=FakeRunner(simple_scenario()),
            yaml_path=yaml_path,
        )
    )
    used |= keys_of(
        run_stage(
            "cluster.bootstrap",
            cfg,
            bootstrap,
            runner=FakeRunner(tillicum_scenario(uv_present=False), remote_root=remote_root),
            yaml_path=yaml_path,
            build_lammps=True,
            install_qe=True,
            partition="gpu-h200",
        )
    )
    discovered = Settings.model_validate(
        read_yaml(yaml_path) | {"paths": {"runs_dir": str(tmp_path / "runs")}}
    )
    discovered = discovered.model_copy(
        update={
            "cluster": discovered.cluster.model_copy(
                update={"control_path": str(tmp_path / "cm.sock")}
            )
        }
    )
    used |= keys_of(
        run_stage(
            "cluster.sync",
            discovered,
            pull,
            runner=FakeRunner(
                Scenario(
                    sacct=[
                        "JobID|State|ExitCode|Elapsed|NodeList|\n4242|RUNNING|0:0|00:01:00|g001|\n"
                    ]
                ),
                remote_root=remote_root,
            ),
        )
    )
    used |= {c["key"] for c in show(discovered, runner=FakeRunner(Scenario()))["commands"]}
    assert used == set(COMMANDS), sorted(set(COMMANDS) - used)
