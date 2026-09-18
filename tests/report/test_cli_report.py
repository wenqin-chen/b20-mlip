"""``b20mlip report build [--readme]`` and ``report audit [--strict]`` end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from b20mlip.cli import app
from b20mlip.config import Settings
from b20mlip.report import audit

from .conftest import FixtureRuns

runner = CliRunner()


def test_report_build_and_audit_cli(
    report_root: Path, cfg: Settings, fixture_runs: FixtureRuns, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(report_root)
    common = [
        "--set",
        f"paths.runs_dir={cfg.paths.runs_dir}",
        "--set",
        f"report.numbers_path={cfg.report.numbers_path}",
    ]
    result = runner.invoke(app, [*common, "report", "build", "--readme"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["stage"] == "report" and payload["status"] == "ok"
    assert payload["summary"]["n_numbers"] == 4 and payload["summary"]["n_stale"] == 1
    assert (report_root / "README.md").is_file() and Path(cfg.report.numbers_path).is_file()
    assert (report_root / "runs" / "report").is_dir()

    result = runner.invoke(app, [*common, "report", "audit", "--strict"])
    assert result.exit_code == 0 and "OK" in result.output, result.output

    readme = report_root / "README.md"
    readme.write_text(
        readme.read_text().replace("## What it does", "## What it does\n\nWe used 3 seeds.", 1)
    )
    result = runner.invoke(app, [*common, "report", "audit"])
    assert result.exit_code == 1 and "A1:" in result.output and "'3'" in result.output

    result = runner.invoke(app, [*common, "--dry-run", "report", "build"])
    assert result.exit_code == 1 and '"status": "partial"' in result.output
    assert "--readme" in runner.invoke(app, ["report", "build", "--help"]).output


def test_repo_readme_passes_strict_audit(repo: Path) -> None:
    """CONTRACTS.md section 7: the generated README passes; skipped until the inputs exist."""
    numbers = repo / "reports" / "numbers.json"
    if not numbers.is_file():
        pytest.skip("reports/numbers.json not built yet (run `b20mlip report build --readme`)")
    if audit.card_locations(repo)["DATA_CARD.md"] is None:
        pytest.skip("DATA_CARD.md is owned by the data tier and is not written yet")
    assert (
        audit.run(readme=repo / "README.md", numbers=numbers, runs_dir=repo / "runs", strict=True)
        == []
    )
