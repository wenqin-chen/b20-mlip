"""Fixtures for the report tier: a scratch project root with licence cards, synthetic runs.

``fixture_runs`` writes real manifests through ``provenance.run_stage``: an ok ``eval.errors``
run publishing numbers, a failed ``train`` run (ignored), a partial ``eval.discovery`` dry run
(ignored), a newer ok ``eval.errors`` run whose value must win, and an ok ``md.ase`` run whose
``numbers.json`` is modified after the manifest was written (stale).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from b20mlip.config import Settings
from b20mlip.models import StageResult
from b20mlip.provenance import RunContext, run_stage

REF_QE = {
    "code": "qe",
    "functional": "PBE",
    "pseudos": "SSSP-efficiency-1.3",
    "e0_source": "E0s_qe.json",
}
REF_VASP = {"code": "vasp", "functional": "PBE", "pseudos": "PAW", "e0_source": "foundation"}
REF_EXP = {"code": "experiment", "functional": None, "pseudos": None, "e0_source": None}
REF_MACE = {"code": "mace", "functional": "PBE", "pseudos": None, "e0_source": "E0s_qe.json"}
META_QE: dict[str, Any] = {
    "reference": REF_QE,
    "e0_source": "E0s_qe.json",
    "head": "Default",
    "n": 120,
    "seed": 0,
    "ci95": [1.0, 2.0],
}
DATA_CARD_TEXT = """# Data card (test fixture)

MPtrj (MIT); sAlex / OMat24 / WBM (CC-BY-4.0); SSSP-efficiency pseudopotentials are cited by
md5 and never redistributed.
"""


def meta(**overrides: Any) -> dict[str, Any]:
    """Valid A3 meta for a metric key, with overrides."""
    return {**META_QE, **overrides}


@pytest.fixture
def report_root(tmp_path: Path, repo: Path) -> Path:
    """A project root with MODEL_CARD.md and NOTICE copied from the repo and a docs/DATA_CARD.md."""
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "reports").mkdir()
    shutil.copy(repo / "MODEL_CARD.md", root / "MODEL_CARD.md")
    shutil.copy(repo / "NOTICE", root / "NOTICE")
    (root / "docs" / "DATA_CARD.md").write_text(DATA_CARD_TEXT, encoding="utf-8")
    return root


@pytest.fixture
def cfg(report_root: Path) -> Settings:
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(report_root / "data"),
                "runs_dir": str(report_root / "runs"),
                "models_dir": str(report_root / "models"),
                "dft_dir": str(report_root / "dft"),
                "reports_dir": str(report_root / "reports"),
            },
            "report": {"numbers_path": str(report_root / "reports" / "numbers.json")},
        }
    )


RunMaker = Callable[..., StageResult]


@pytest.fixture
def make_run(cfg: Settings) -> RunMaker:
    """``make_run(stage, numbers, seed=0, status="ok", extras=None)`` -> StageResult."""

    def _make(
        stage: str,
        numbers: dict[str, Any] | None,
        *,
        seed: int | None = 0,
        status: str = "ok",
        extras: dict[str, Any] | None = None,
        register: bool = True,
    ) -> StageResult:
        def fn(cfg: Settings, ctx: RunContext) -> dict[str, int]:
            if numbers is not None:
                path = ctx.out_dir / "numbers.json"
                path.write_text(json.dumps(numbers), encoding="utf-8")
                if register:
                    ctx.add_output(path, "json")
            if extras:
                ctx.log(**extras)
            if status == "failed":
                raise RuntimeError("stage exploded")
            return {"n": 1}

        return run_stage(stage, cfg, fn, seed=seed, dry_run=(status == "partial"))  # type: ignore[arg-type]

    return _make


@dataclass
class FixtureRuns:
    ok: StageResult
    failed: StageResult
    partial: StageResult
    newer: StageResult
    stale: StageResult


@pytest.fixture
def fixture_runs(make_run: RunMaker) -> FixtureRuns:
    ok = make_run(
        "eval.errors",
        {
            "@meta": meta(tier="T0"),
            "eval.errors.T0.B0.mae_f": 61.2,
            "eval.errors.T0.B0.mae_f@meta": {"ci95": [59.0, 63.4], "bracket": "B0"},
            "eval.errors.T0.B1.mae_f": 32.1,
            "eval.errors.T0.B1.mae_f@meta": {"ci95": [30.0, 34.2], "bracket": "B1"},
            "eval.errors.T3.B0.mae_f": 71.2,
            "eval.errors.T3.B0.mae_f@meta": {
                "reference": REF_VASP,
                "tier": "T3",
                "noise_floor_f": 8.0,
                "ci95": None,
                "ci95_reason": "single seed",
            },
            "parity.passed": False,
        },
        seed=0,
    )
    failed = make_run("train", {"eval.errors.T0.B1.mae_f": 1.0}, seed=0, status="failed")
    partial = make_run("eval.discovery", {"eval.discovery.B1.delta_f1": 0.5}, status="partial")
    newer = make_run(
        "eval.errors",
        {
            "@meta": meta(tier="T0"),
            "eval.errors.T0.B1.mae_f": 30.5,
            "eval.errors.T0.B1.mae_f@meta": {"ci95": [28.9, 32.0], "bracket": "B1"},
        },
        seed=1,
    )
    stale = make_run(
        "md.ase",
        {"@meta": meta(reference=REF_EXP), "md.ase.MnSi.a_300K_A": 4.571},
        seed=0,
    )
    stale_numbers = Path(stale.outputs[0].path)
    stale_numbers.write_text(json.dumps({"md.ase.MnSi.a_300K_A": 9.999}), encoding="utf-8")
    assert ok.status == "ok" and failed.status == "failed" and partial.status == "partial"
    assert newer.status == "ok" and stale.status == "ok"
    return FixtureRuns(ok=ok, failed=failed, partial=partial, newer=newer, stale=stale)
