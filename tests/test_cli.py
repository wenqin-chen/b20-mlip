"""CLI skeleton: --version, stubs exit 2, bench/report audit wired to their (future) modules."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from b20mlip import __version__
from b20mlip.cli import CLIState, ExecutorKind, app
from b20mlip.executors import LocalExecutor, SlurmExecutor
from b20mlip.models import StageResult

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and f"b20mlip {__version__}" in result.output
    assert __version__ == "0.1.0"


def test_help_lists_every_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "data",
        "dft",
        "eval",
        "md",
        "sampling",
        "active",
        "agent",
        "cluster",
        "report",
        "train",
        "export",
        "screen",
        "bench",
    ):
        assert name in result.output
    for sub in ("data", "dft", "report"):
        assert runner.invoke(app, [sub, "--help"]).exit_code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "pull", "--sources", "mptrj,wbm"],
        ["dft", "run", "--units", "dft/r0/", "--limit", "3"],
        ["train", "--variant", "naive", "--split", "S", "--seed", "0"],
        ["eval", "errors", "--model", "M"],
        ["md", "lammps"],
        ["sampling", "wham", "--run", "R"],
        ["active", "select"],
        ["agent", "run", "--backend", "mock"],
        ["cluster", "bootstrap"],
        ["report", "build", "--readme"],
        ["screen"],
        ["export", "--model", "M"],
    ],
)
def test_stubs_exit_2_with_message(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "not implemented in this tier" in result.output


def test_bench_and_audit_exit_2_until_their_modules_exist() -> None:
    assert "b20mlip.bench" not in sys.modules and "b20mlip.report.audit" not in sys.modules
    result = runner.invoke(app, ["bench", "--out", "runs/bench"])
    assert result.exit_code == 2 and "b20mlip.bench" in result.output
    result = runner.invoke(app, ["report", "audit", "--strict"])
    assert result.exit_code == 2 and "b20mlip.report.audit" in result.output


def test_global_options_build_state(tmp_path: Path, repo: Path) -> None:
    state = CLIState(
        config_paths=[repo / "configs" / "cluster" / "tillicum.yaml"],
        overrides=["compute.threads=2"],
        seed=3,
        executor=ExecutorKind.slurm,
    )
    cfg = state.settings()
    assert (
        cfg.compute.threads == 2 and cfg.cluster.alias == "tillicum" and cfg.cluster.scratch is None
    )
    assert isinstance(state.make_executor(cfg), SlurmExecutor)
    assert isinstance(CLIState().make_executor(cfg), LocalExecutor)


def test_bench_wiring_with_a_test_double(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command runs `bench.run(cfg, ctx, out=...)` under run_stage and exits on its status."""
    seen: dict[str, object] = {}

    def run(cfg, ctx, *, out):  # type: ignore[no-untyped-def]
        seen["out"] = out
        seen["seed"] = ctx.seed
        seen["executor"] = type(ctx.executor).__name__
        (ctx.out_dir / "bench.json").write_text("{}")
        ctx.add_output(ctx.out_dir / "bench.json")
        return {"s_per_frame": 0.12}

    monkeypatch.setitem(sys.modules, "b20mlip.bench", types.SimpleNamespace(run=run))
    result = runner.invoke(
        app,
        ["--set", f"paths.runs_dir={tmp_path / 'runs'}", "--seed", "5", "bench", "--out", "x/y"],
    )
    assert result.exit_code == 0, result.output
    assert '"status": "ok"' in result.output and "bench.json" in result.output
    assert seen == {"out": Path("x/y"), "seed": 5, "executor": "LocalExecutor"}
    assert (tmp_path / "runs" / "bench").is_dir()

    def failing(cfg, ctx, *, out):  # type: ignore[no-untyped-def]
        raise RuntimeError("no model cached")

    monkeypatch.setitem(sys.modules, "b20mlip.bench", types.SimpleNamespace(run=failing))
    result = runner.invoke(app, ["--set", f"paths.runs_dir={tmp_path / 'runs'}", "bench"])
    assert result.exit_code == 1 and '"status": "failed"' in result.output

    def partial(cfg, ctx, *, out):  # type: ignore[no-untyped-def]
        return StageResult(
            stage="bench", run_id=ctx.run_id, manifest_path="", status="partial", outputs=[]
        )

    monkeypatch.setitem(sys.modules, "b20mlip.bench", types.SimpleNamespace(run=partial))
    result = runner.invoke(
        app, ["--set", f"paths.runs_dir={tmp_path / 'runs'}", "--dry-run", "bench"]
    )
    assert result.exit_code == 1 and '"status": "partial"' in result.output


def test_report_audit_wiring_with_a_test_double(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def make(violations: list[str]):  # type: ignore[no-untyped-def]
        def run(*, readme, numbers, runs_dir, strict):  # type: ignore[no-untyped-def]
            calls.append(
                {"readme": readme, "numbers": numbers, "runs_dir": runs_dir, "strict": strict}
            )
            return violations

        return types.SimpleNamespace(run=run)

    monkeypatch.setitem(sys.modules, "b20mlip.report.audit", make([]))
    result = runner.invoke(app, ["report", "audit", "--strict", "--readme", "docs/README.md"])
    assert result.exit_code == 0 and "OK" in result.output
    assert calls[-1] == {
        "readme": Path("docs/README.md"),
        "numbers": Path("reports/numbers.json"),
        "runs_dir": Path("runs"),
        "strict": True,
    }
    monkeypatch.setitem(sys.modules, "b20mlip.report.audit", make(["A7: forbidden string 'first'"]))
    result = runner.invoke(app, ["report", "audit"])
    assert result.exit_code == 1 and "A7" in result.output and calls[-1]["strict"] is False
