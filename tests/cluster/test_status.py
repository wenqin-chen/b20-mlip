"""cluster status: squeue rows plus local marker counts, rendered as a table."""

from __future__ import annotations

from pathlib import Path

import pytest

from b20mlip.cluster.status import format_table, local_marker_counts, show
from b20mlip.config import Settings
from b20mlip.executors import ClusterUnreachable, CommandResult

from .conftest import FakeRunner, Scenario


def _staging(cfg: Settings) -> Path:
    root = Path(cfg.paths.runs_dir) / "slurm"
    qe = root / "qe_r0"
    (qe / "units").mkdir(parents=True)
    (qe / "units.txt").write_text("u1\nu2\nu3\nu4\n")
    (qe / "units" / "u1.done").write_text("{}")
    (qe / "units" / "u2.failed").write_text("{}")
    build = root / "build_lammps"
    (build / "units").mkdir(parents=True)
    (build / "units" / "build.done").write_text("{}")
    (root / "stray").mkdir()  # no units: ignored
    (root / "file.txt").write_text("")
    return root


def test_show_and_format(discovered_cfg: Settings) -> None:
    _staging(discovered_cfg)
    fake = FakeRunner(Scenario())
    report = show(discovered_cfg, runner=fake)
    assert report["alias"] == "tillicum" and report["squeue_error"] is None
    assert [r["job_id"] for r in report["queue"]] == ["4242", "4243_[1-40]", "4243_41"]
    assert report["local"] == {
        "build_lammps": {"units": 1, "done": 1, "failed": 0, "pending": 0},
        "qe_r0": {"units": 4, "done": 1, "failed": 1, "pending": 2},
    }
    assert [c["key"] for c in report["commands"]] == ["socket_check", "squeue"]
    assert fake.remote_commands() == ['squeue -u "$USER" -o "%i %j %T %M %P %R"']
    table = format_table(report)
    assert "JOBID" in table and "4243_[1-40]" in table and "(Priority)" in table
    assert "JOB" in table and "qe_r0" in table and "build_lammps" in table
    lines = table.splitlines()
    row = next(ln for ln in lines if ln.startswith("qe_r0"))
    assert row.split() == ["qe_r0", "4", "1", "1", "2"]


def test_status_errors(discovered_cfg: Settings, tmp_path: Path) -> None:
    with pytest.raises(ClusterUnreachable, match="MFA"):
        show(discovered_cfg, runner=FakeRunner(Scenario(socket_ok=False)))

    class SqueueFails(FakeRunner):
        def _respond(self, cmd, tup):  # type: ignore[no-untyped-def]
            if cmd.startswith("squeue"):
                return CommandResult(tup, 1, "", "squeue: error: slurm_load_jobs")
            return super()._respond(cmd, tup)

    report = show(discovered_cfg, runner=SqueueFails(Scenario()))
    assert report["queue"] == [] and "slurm_load_jobs" in report["squeue_error"]
    text = format_table(report)
    assert "squeue failed" in text and "(no synced jobs)" in text
    assert local_marker_counts(tmp_path / "absent") == {}
    empty = format_table({"alias": "x", "queue": [], "local": {}})
    assert "(no jobs)" in empty
