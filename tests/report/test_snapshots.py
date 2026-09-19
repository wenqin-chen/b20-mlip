"""Provenance snapshots: ``report build`` ships manifest.json + numbers.json of every cited run
under reports/manifests/; the audit resolves from runs/ first, then from the snapshot."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from b20mlip.config import Settings
from b20mlip.provenance import RunContext, run_stage, sha256_file
from b20mlip.report import audit, build
from b20mlip.report import numbers as nums
from b20mlip.report.numbers import SNAPSHOTS_KEY

from .conftest import FixtureRuns, RunMaker


def _snapshots(cfg: Settings) -> Path:
    return nums.snapshot_dir(cfg.report.numbers_path)


def test_write_numbers_snapshots_cited_runs_only(cfg: Settings, fixture_runs: FixtureRuns) -> None:
    out = Path(cfg.report.numbers_path)
    nums.collect(cfg.paths.runs_dir, out=out)
    snap = _snapshots(cfg)
    cited = {("eval.errors", fixture_runs.ok.run_id), ("eval.errors", fixture_runs.newer.run_id)}
    found = {(p.parent.parent.name, p.parent.name) for p in snap.glob("*/*/manifest.json")}
    assert found == cited  # failed, partial and stale runs are never snapshotted
    for stage, run_id in cited:
        run_dir = cfg.paths.runs_dir / stage / run_id
        for name in ("manifest.json", "numbers.json"):
            assert (snap / stage / run_id / name).read_bytes() == (run_dir / name).read_bytes()
        assert sorted(p.name for p in (snap / stage / run_id).iterdir()) == [
            "manifest.json",
            "numbers.json",
        ]
    data = json.loads(out.read_text())
    assert list(data)[:2] == ["@stale", SNAPSHOTS_KEY]
    assert data[SNAPSHOTS_KEY] == {
        f"eval.errors/{run_id}": f"reports/manifests/eval.errors/{run_id}"
        for _, run_id in sorted(cited)
    }
    assert nums.load_entries(out)[0].keys() == nums.load(out).keys()  # @snapshots is skipped


def test_snapshots_of_uncited_runs_are_pruned(cfg: Settings, make_run: RunMaker) -> None:
    out = Path(cfg.report.numbers_path)
    first = make_run("bench", {"bench.x": 1.0})
    nums.collect(cfg.paths.runs_dir, out=out)
    snap = _snapshots(cfg)
    assert (snap / "bench" / first.run_id / "manifest.json").is_file()
    second = make_run("bench", {"bench.x": 2.0})  # newer run of the same key wins
    (snap / "orphan" / "x").mkdir(parents=True)
    nums.collect(cfg.paths.runs_dir, out=out)
    assert not (snap / "bench" / first.run_id).exists()
    assert (snap / "bench" / second.run_id / "numbers.json").is_file()
    assert not (snap / "orphan").exists()
    assert nums.write_snapshots(nums.Harvest(), cfg.paths.runs_dir, out) == {}
    assert not any(snap.iterdir())


def test_audit_passes_from_snapshot_after_runs_deleted(
    cfg: Settings, fixture_runs: FixtureRuns, report_root: Path
) -> None:
    readme = report_root / "README.md"
    assert run_stage("report", cfg, build.run, readme=True, readme_path=readme).status == "ok"
    numbers = Path(cfg.report.numbers_path)
    runs = cfg.paths.runs_dir
    with_runs, notes = audit.run_report(readme, numbers, runs, strict=True)
    assert with_runs == []
    assert all("resolved from runs" in n for n in notes if "resolved" in n)
    shutil.rmtree(runs)  # CI checks out the repository without runs/
    from_snapshot, notes = audit.run_report(readme, numbers, runs, strict=True)
    assert from_snapshot == []
    assert notes and all("resolved from snapshot" in n for n in notes if "resolved" in n)
    assert not any("not local" in n for n in notes)  # numbers.json is always in the snapshot
    assert audit.run(readme, numbers, runs, strict=True) == []


def test_tampered_or_missing_snapshot_fails_a2(
    cfg: Settings, fixture_runs: FixtureRuns, report_root: Path
) -> None:
    readme = report_root / "README.md"
    out = Path(cfg.report.numbers_path)
    nums.collect(cfg.paths.runs_dir, out=out)
    readme.write_text(build.render(json.loads(out.read_text())), encoding="utf-8")
    snap = _snapshots(cfg) / "eval.errors" / fixture_runs.newer.run_id
    runs = cfg.paths.runs_dir

    def a2(**kw: object) -> list[str]:
        return [v for v in audit.run(readme, out, runs, strict=True, **kw) if v.startswith("A2:")]  # type: ignore[arg-type]

    # while runs/ exists the snapshot must agree with it
    manifest_bytes = (snap / "manifest.json").read_bytes()
    (snap / "manifest.json").write_text(manifest_bytes.decode().replace('"ok"', '"partial"'))
    assert any("snapshot" in v and "disagrees with runs/" in v for v in a2())
    (snap / "manifest.json").write_bytes(manifest_bytes)
    assert a2() == []
    shutil.rmtree(runs)
    # tampered numbers.json in the snapshot
    (snap / "numbers.json").write_text(json.dumps({"eval.errors.T0.B1.mae_f": 1.0}))
    violations = a2()
    assert len(violations) == 1 and "tampered snapshot" in violations[0]
    assert violations[0].endswith("no longer matches its sha256")
    # missing numbers.json in the snapshot is a violation (it is never "not local")
    (snap / "numbers.json").unlink()
    assert any("numbers.json artifact is missing (not in snapshot)" in v for v in a2())
    # a tampered manifest (sha differs from numbers.json)
    (snap / "numbers.json").write_bytes(b"{}")
    (snap / "manifest.json").write_text(manifest_bytes.decode().replace("\n", "\n "))
    assert any("manifest sha256 (snapshot) differs" in v for v in a2())
    # no snapshot at all
    shutil.rmtree(snap)
    assert any("has no manifest under" in v and "manifests" in v for v in a2())
    # an explicit snapshot root can be given
    assert any("has no manifest" in v for v in a2(snapshots=report_root / "elsewhere"))


def test_absent_heavy_outputs_are_notes_not_violations(
    cfg: Settings, make_run: RunMaker, report_root: Path
) -> None:
    def stage(cfg: Settings, ctx: RunContext) -> dict[str, int]:
        (ctx.out_dir / "numbers.json").write_text(json.dumps({"bench.x": 1.0}))
        ctx.add_output(ctx.out_dir / "numbers.json", "json")
        (ctx.out_dir / "model.pt").write_bytes(b"heavy")
        ctx.add_output(ctx.out_dir / "model.pt", "model")
        return {}

    result = run_stage("bench", cfg, stage, seed=0)
    out = Path(cfg.report.numbers_path)
    nums.collect(cfg.paths.runs_dir, out=out)
    readme = report_root / "README.md"
    readme.write_text(build.render(json.loads(out.read_text())), encoding="utf-8")
    runs = cfg.paths.runs_dir
    snap = _snapshots(cfg) / "bench" / result.run_id
    assert sorted(p.name for p in snap.iterdir()) == ["manifest.json", "numbers.json"]
    # locally present heavy output: sha is checked
    (runs / "bench" / result.run_id / "model.pt").write_bytes(b"heavier")
    violations, _ = audit.run_report(readme, out, runs)
    assert any("model.pt no longer matches its sha256" in v for v in violations)
    (runs / "bench" / result.run_id / "model.pt").write_bytes(b"heavy")
    assert audit.run(readme, out, runs) == []
    # after deleting runs/ the heavy output is simply not local
    shutil.rmtree(runs)
    violations, notes = audit.run_report(readme, out, runs, strict=True)
    assert violations == []
    assert any("artifact not local, sha unchecked" in n and "model.pt" in n for n in notes)
    assert sha256_file(snap / "numbers.json")  # still verified
