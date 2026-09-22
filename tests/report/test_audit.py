"""Gates A1–A11: at least one failing and one passing case each (CONTRACTS.md section 8)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from b20mlip.config import Settings
from b20mlip.report import audit, build
from b20mlip.report import numbers as nums

from .conftest import REF_EXP, REF_QE, REF_VASP, FixtureRuns, RunMaker, meta

RUN = "20260918T100000-abcdef-0"
SHA = "b" * 64
MARK = "<!-- num:{k} -->{v}<!-- /num -->"


def entry(value: float, m: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"value": value, "run_id": RUN, "manifest_sha256": SHA, "stage": "s", "meta": m or {}}


def base_numbers() -> dict[str, Any]:
    t3 = {"tier": "T3", "reference": REF_VASP, "noise_floor_f": 8.0}
    return {
        "eval.errors.T0.B0.mae_f": entry(61.2, meta(tier="T0")),
        "eval.errors.T0.B1.mae_f": entry(32.1, meta(tier="T0")),
        "eval.errors.T3.B0.mae_f": entry(71.2, meta(**t3)),
        "eval.errors.T3.B1.mae_f": entry(66.2, meta(**t3)),
        "eval.discovery.B2.delta_f1": entry(
            0.012,
            meta(
                prevalence="natural",
                paired_vs="B0",
                sample_seed=0,
                n=1000,
                head="pt_head",
                energy_scale="mp",
                reference=REF_VASP,
            ),
        ),  # fmt: skip
        "agent.eval.accuracy": entry(
            0.83, {"backend": "anthropic", "model_id": "claude-opus-5", "trace_path": "t.jsonl"}
        ),
        "parity.passed": entry(1.0),
        "data.n_qe_frames": entry(512),
    }


def doc(
    pre: str = "", results: str = "", limitations: str = "", plan: str = "", nongoals: str = ""
) -> str:
    return (
        f"# b20-mlip\n\n{pre}\n\n## Results\n\n{results}\n\n## Limitations\n\n{limitations}\n\n"
        f"## Plan\n\n{plan}\n\n## Non-goals\n\n{nongoals}\n"
    )


def gate(violations: list[str], k: int) -> list[str]:
    return [v for v in violations if v.startswith(f"A{k}:")]


@pytest.fixture
def check(report_root: Path):  # type: ignore[no-untyped-def]
    """``check(text, numbers, strict=False, runs_dir=None)`` -> violations of a README in root."""

    def _check(
        text: str,
        numbers: Any,
        *,
        strict: bool = False,
        runs_dir: Path | None = None,
    ) -> list[str]:
        readme = report_root / "README.md"
        readme.write_text(text, encoding="utf-8")
        return audit.run(
            readme=readme, numbers=numbers, runs_dir=runs_dir or report_root / "runs", strict=strict
        )

    return _check


# --- whole pipeline ---------------------------------------------------------------------------


def test_rendered_readme_passes_every_gate_but_a2(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    violations = check(build.render(numbers), numbers, strict=True)
    assert violations and all(v.startswith("A2:") for v in violations)  # synthetic run ids only


def test_run_missing_readme_and_numbers(tmp_path: Path) -> None:
    assert audit.run(tmp_path / "nope.md", tmp_path / "n.json", tmp_path) == [
        f"A1: README not found: {tmp_path / 'nope.md'}"
    ]
    readme = tmp_path / "README.md"
    readme.write_text("# t\n" + MARK.format(k="a.b", v="1") + "\n")
    violations = audit.run(readme, tmp_path / "missing.json", tmp_path)
    assert any("absent from numbers.json" in v for v in gate(violations, 1))


# --- A1 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pre", "needle"),
    [
        ("three seeds and 3 seeds", "unwrapped numeral '3'"),
        ("MAE " + MARK.format(k="zz.missing", v="61.2") + " meV/Å", "absent from numbers.json"),
        ("MAE " + MARK.format(k="eval.errors.T0.B0.mae_f", v="60") + " meV/Å", "shows '60'"),
        (MARK.format(k="eval.errors.T0.B0.mae_f", v="61.2 ± 1"), "must show one number"),
        ("<!-- gen:start:x -->never closed", "malformed gen blocks"),
        ("8-atom cells, 1,000-structure sample, ±0.5 Å, 1e-3", "unwrapped numeral '1e-3'"),
    ],
)
def test_a1_fails(check, pre: str, needle: str) -> None:  # type: ignore[no-untyped-def]
    assert any(needle in v for v in gate(check(doc(pre=pre), base_numbers()), 1))


def test_a1_incomplete_entry(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    del numbers["eval.errors.T0.B0.mae_f"]["manifest_sha256"]
    text = doc(pre=MARK.format(k="eval.errors.T0.B0.mae_f", v="61.2"))
    assert any("incomplete" in v for v in gate(check(text, numbers), 1))


@pytest.mark.parametrize(
    "pre",
    [
        "MACE-MPA-0, mp-871, v0.1, Python 3.11, 2026-09-18, https://x.org/1/2, T0–T4b, CC-BY-4.0",
        "`E0s_qe.json` and mace-torch==0.3.16 and ![cov](https://b/95.svg) and [d](docs/3.md)",
        "```bash\nb20mlip md ase --T 300 --ps 40\n```",
        "<!-- gen:start:x -->3 seeds, 95 % CI<!-- gen:end -->",
        MARK.format(k="zz.missing", v="pending")
        + " and "
        + MARK.format(k="eval.errors.T0.B0.mae_f", v="61.2"),
        "<!-- a hidden 42 comment --> ΔF1 P2₁3 sha256 float64",
    ],
)
def test_a1_passes(check, pre: str) -> None:  # type: ignore[no-untyped-def]
    assert gate(check(doc(pre=pre), base_numbers()), 1) == []


# --- A2 ---------------------------------------------------------------------------------------


def test_a2_passes_on_real_runs(cfg: Settings, fixture_runs: FixtureRuns, check) -> None:  # type: ignore[no-untyped-def]
    refs = nums.collect(cfg.paths.runs_dir, out=cfg.report.numbers_path)
    text = build.render(json.loads(Path(cfg.report.numbers_path).read_text()))
    for strict in (False, True):
        violations = check(
            text, cfg.report.numbers_path, strict=strict, runs_dir=cfg.paths.runs_dir
        )
        assert gate(violations, 2) == [] and len(refs) == 4


def test_a2_fails(cfg: Settings, fixture_runs: FixtureRuns, check) -> None:  # type: ignore[no-untyped-def]
    runs = cfg.paths.runs_dir
    nums.collect(runs, out=cfg.report.numbers_path)
    entries, stale = nums.load_entries(cfg.report.numbers_path)
    text = build.render({**entries, "@stale": stale})

    def with_run(key: str, stage: str, run_id: str) -> dict[str, Any]:
        numbers = {**entries, "@stale": stale}
        numbers[key] = {**entries[key], "stage": stage, "run_id": run_id, "manifest_sha256": SHA}
        return numbers

    def a2(numbers: Any, strict: bool = False) -> list[str]:
        return gate(check(text, numbers, runs_dir=runs, strict=strict), 2)

    bogus = with_run("parity.passed", "eval.errors", "20200101T000000-000000-0")
    assert any("has no manifest" in v for v in a2(bogus))
    # run ids are unique per stage only: without a stage an id found under two stages is
    # ambiguous (built explicitly here; two stages share an id only within one second)
    twin = runs / "bench" / fixture_runs.ok.run_id
    shutil.copytree(runs / "eval.errors" / fixture_runs.ok.run_id, twin)
    ambiguous = with_run("parity.passed", "", fixture_runs.ok.run_id)
    assert any("is ambiguous without a stage" in v for v in a2(ambiguous))
    shutil.rmtree(twin)
    unique = with_run("parity.passed", "", fixture_runs.newer.run_id)
    assert not any("ambiguous" in v for v in a2(unique))
    failed = with_run("parity.passed", "train", fixture_runs.failed.run_id)
    assert any("status=failed" in v for v in a2(failed))
    assert any("manifest sha256 (runs) differs from numbers.json" in v for v in a2(failed))
    partial = with_run("parity.passed", "eval.discovery", fixture_runs.partial.run_id)
    assert not any("partial" in v for v in a2(partial))
    assert any("status=partial (--strict)" in v for v in a2(partial, strict=True))
    # a stale key that an older fresh run still publishes, or a stale file of a cited run
    numbers = {**entries, "@stale": [{"key": "eval.errors.T0.B0.mae_f", "run_id": "r"}]}
    assert any("is stale" in v for v in a2(numbers))
    same_run = [{"key": "other", "run_id": fixture_runs.ok.run_id, "stage": "eval.errors"}]
    assert any("is stale" in v for v in a2({**entries, "@stale": same_run}))
    # outputs that changed or vanished after the run, and an unreadable manifest
    out = Path(fixture_runs.ok.outputs[0].path)
    original = out.read_text()
    out.write_text(original + " ")
    assert any("no longer matches" in v for v in a2(cfg.report.numbers_path))
    out.unlink()  # the snapshot copy still backs the number: no violation, verified there
    assert a2(cfg.report.numbers_path) == []
    out.write_text(original)
    Path(fixture_runs.ok.manifest_path).write_text("{not json")
    assert any("unreadable manifest" in v for v in a2(cfg.report.numbers_path))


# --- A3 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "m", "needle"),
    [
        ("eval.errors.T0.B0.mae_f", {}, "['reference', 'e0_source', 'head', 'n', 'seed', 'ci95']"),
        (
            "md.ase.MnSi.a_300K_A",
            meta(reference={**REF_QE, "functional": None}),
            "reference.functional",
        ),
        (
            "phonons.FeSi.B0.omega_mae_meV",
            meta(reference={**REF_QE, "code": "castep"}),
            "reference.code",
        ),
        ("sampling.neb.FeSi.Ea_eV", meta(ci95=None), "ci95 (interval, or null with ci95_reason)"),
        ("eval.errors.T0.B0.mae_f", meta(n=True, seed=1.5, head="both"), "['head', 'n', 'seed']"),
        (
            "eval.errors.T0.B0.mae_f",
            meta(reference="qe/PBE", e0_source=""),
            "['reference', 'e0_source']",
        ),
    ],
)
def test_a3_fails(check, key: str, m: dict[str, Any], needle: str) -> None:  # type: ignore[no-untyped-def]
    violations = gate(check(doc(), {key: entry(1.0, m)}), 3)
    assert len(violations) == 1 and key in violations[0] and needle in violations[0]


def test_a3_passes(check) -> None:  # type: ignore[no-untyped-def]
    numbers = {
        "eval.errors.T0.B0.mae_f": entry(1.0, meta()),
        "md.ase.MnSi.a_exp_A": entry(
            4.558, meta(reference=REF_EXP, ci95=None, ci95_reason="literature")
        ),
        "data.n_frames": entry(3),  # not a metric namespace: no meta required
        "parity.passed": entry(1),
    }
    assert gate(check(doc(), numbers), 3) == []


# --- A4 ---------------------------------------------------------------------------------------


def test_a4_f1_meta(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    assert gate(check(doc(), numbers), 4) == []
    bad = dict(numbers["eval.discovery.B2.delta_f1"]["meta"])
    bad.update(n=999, prevalence="stratified", paired_vs="B1")
    del bad["sample_seed"]
    numbers["eval.discovery.B2.delta_f1"]["meta"] = bad
    numbers["eval.discovery.B0.f1"] = entry(0.5, meta(n=1000))
    violations = gate(check(doc(), numbers), 4)
    assert len(violations) == 2
    assert 'prevalence != "natural"' in violations[1] and "n != 1000" in violations[1]
    assert "sample_seed missing" in violations[1] and "paired_vs" in violations[1]


def test_a4_forbidden_tokens(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    numbers["eval.discovery.B1.daf"] = entry(2.0, meta(n=1000))
    numbers["parity.passed"]["meta"] = {"note": "not the leaderboard"}
    violations = gate(check(doc(results="we report CPS and kappa_srme"), numbers), 4)
    assert any("'daf' appears in numbers.json" in v for v in violations)
    assert any("'leaderboard' appears in numbers.json" in v for v in violations)
    assert any("'cps' in README claims" in v for v in violations)
    assert any("'kappa_srme' in README claims" in v for v in violations)
    ok = doc(results="biceps and daffodils", nongoals='no "leaderboard" submission')
    assert gate(check(ok, base_numbers()), 4) == []


# --- A5 ---------------------------------------------------------------------------------------


def table(*rows: str) -> str:
    body = "\n".join(rows)
    return f"<!-- gen:start:t -->\n| Model | a | b |\n| --- | --- | --- |\n{body}\n<!-- gen:end -->"


def test_a5_fails(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    numbers["eval.errors.T0.B0.mae_f_pbesol"] = entry(
        1.0, meta(reference={**REF_QE, "functional": "PBEsol"})
    )
    numbers["eval.errors.T0.B0.mae_f_pyscf"] = entry(
        2.0, meta(reference={**REF_QE, "code": "pyscf"})
    )
    mixed = table(
        "| B0 | "
        + MARK.format(k="eval.errors.T0.B0.mae_f", v="61.2")
        + " | "
        + MARK.format(k="eval.errors.T3.B0.mae_f", v="71.2")
        + " |",
        "| B1 | "
        + MARK.format(k="eval.errors.T0.B0.mae_f_pbesol", v="1")
        + " | "
        + MARK.format(k="eval.errors.T0.B0.mae_f_pyscf", v="2")
        + " |",
    )
    violations = gate(check(doc(results=mixed), numbers), 5)
    assert any(
        "row 'B0' of block 't' cites several references: qe/PBE, vasp/PBE" in v for v in violations
    )
    assert any("PBEsol value eval.errors.T0.B0.mae_f_pbesol in column 'a'" in v for v in violations)
    assert any("PySCF value eval.errors.T0.B0.mae_f_pyscf in row 'B1'" in v for v in violations)


def test_a5_passes(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    numbers["eval.errors.T0.B0.mae_f_pbesol"] = entry(
        1.0, meta(reference={**REF_QE, "functional": "PBEsol"}, cross_functional=True)
    )
    numbers["eval.errors.T0.B0.mae_f_pyscf"] = entry(
        2.0, meta(reference={**REF_QE, "code": "pyscf"}, lower_fidelity=True)
    )
    qe = MARK.format(k="eval.errors.T0.B0.mae_f", v="61.2")
    pbesol = MARK.format(k="eval.errors.T0.B0.mae_f_pbesol", v="1")
    pyscf = MARK.format(k="eval.errors.T0.B0.mae_f_pyscf", v="2")
    vasp = MARK.format(k="eval.errors.T3.B0.mae_f", v="71.2")
    pending = MARK.format(k="zz.pending", v="pending")
    fine = table(
        f"| B0 | {qe} | {pbesol} |",
        f"| B1 (PySCF) | {pyscf} | — |",
        f"| T3 | {vasp} | {pending} |",
    )
    assert gate(check(doc(results=fine), numbers), 5) == []


# --- A6 ---------------------------------------------------------------------------------------


def test_a6(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    claim = doc(results="Deployed in LAMMPS MD.\nMore LAMMPS.", plan="LAMMPS after parity")
    assert gate(check(claim, numbers), 6) == []  # parity.passed == 1
    numbers["parity.passed"] = entry(0)
    violations = gate(check(claim, numbers), 6)
    assert len(violations) == 2 and "line 7" in violations[0]
    del numbers["parity.passed"]
    assert len(gate(check(claim, numbers), 6)) == 2
    numbers["parity.B2.passed"] = entry(1)  # a model-labelled gate (runs from 2026-09-22) counts
    assert gate(check(claim, numbers), 6) == []
    del numbers["parity.B2.passed"]
    assert gate(check(doc(plan="LAMMPS later", results="```\nmd lammps\n```"), numbers), 6) == []
    bullet = doc(limitations="<!-- gen:start:bullet -->ASE/LAMMPS MD<!-- gen:end -->")
    assert len(gate(check(bullet, numbers), 6)) == 1  # the bullet is claim text wherever it sits


# --- A7 ---------------------------------------------------------------------------------------


def test_a7_fails(check) -> None:  # type: ignore[no-untyped-def]
    text = doc(
        results="the first model; state of the art; SOTA results; the leaderboard; "
        "outperforms  all",
        limitations="first without quotes",
    )
    violations = gate(check(text, base_numbers()), 7)
    labels = [v.split("'")[1] for v in violations]
    assert labels == [
        "first",
        "state-of-the-art",
        "SOTA",
        "leaderboard",
        "outperforms all",
        "first",
    ]
    assert "outside quotes in a non-claim section" in violations[-1]


def test_a7_passes(check) -> None:  # type: ignore[no-untyped-def]
    text = doc(
        pre="firstly, ab initio\n\n```\n--first\n```",
        nongoals='no "first" claims, no `SOTA`, no “leaderboard” submission, "state-of-the-art"',
        plan="no `outperforms all` wording",
    )
    assert gate(check(text, base_numbers()), 7) == []


# --- A8 ---------------------------------------------------------------------------------------


def test_a8(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    key = "eval.discovery.B1.mae_e_above_hull"
    numbers[key] = entry(40.0, meta(n=1000, reference=REF_VASP, head="Default", energy_scale="qe"))
    assert any("needs offsets.residual_meV_atom" in v for v in gate(check(doc(), numbers), 8))
    numbers["offsets.residual_meV_atom"] = entry(25.0)
    assert any("= 25.0 > 20" in v for v in gate(check(doc(), numbers), 8))
    numbers["offsets.residual_meV_atom"] = entry(11.0)
    assert gate(check(doc(), numbers), 8) == []
    numbers[key]["meta"].pop("energy_scale")
    assert any("does not declare meta.energy_scale" in v for v in gate(check(doc(), numbers), 8))
    numbers[key]["meta"]["energy_scale"] = "mp"
    numbers["eval.discovery.B1.rmsd"] = entry(0.1, meta(n=1000, reference=REF_VASP))  # exempt
    del numbers["offsets.residual_meV_atom"]
    assert gate(check(doc(), numbers), 8) == []


# --- A9 ---------------------------------------------------------------------------------------


def test_a9_meta_and_delta_keys(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    assert gate(check(doc(), numbers), 9) == []
    numbers["eval.errors.T3.B1.mae_f"]["meta"].pop("noise_floor_f")
    numbers["eval.errors.T4a.B1.mae_f"] = entry(1.0, meta(tier="T3"))  # tier by meta
    numbers["eval.errors.T3.B1.delta_mae_f"] = entry(-2.9, meta(tier="T3", noise_floor_f=8.0))
    numbers["eval.errors.T3.B2.delta_mae_f"] = entry(-9.5, meta(tier="T3", noise_floor_f=8.0))
    violations = gate(check(doc(), numbers), 9)
    assert any("eval.errors.T3.B1.mae_f lacks meta.noise_floor_f" in v for v in violations)
    assert any("eval.errors.T4a.B1.mae_f lacks meta.noise_floor_f" in v for v in violations)
    assert any("eval.errors.T3.B1.delta_mae_f = -2.9 is below" in v for v in violations)
    assert not any("B2.delta_mae_f" in v for v in violations)


def test_a9_prose_pairs(check) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    b0 = MARK.format(k="eval.errors.T3.B0.mae_f", v="71.2")
    b1 = MARK.format(k="eval.errors.T3.B1.mae_f", v="66.2")
    same_paragraph = doc(results=f"T3 force MAE improves from {b0} to {b1} meV/Å.")
    assert any("not a claimable improvement" in v for v in gate(check(same_paragraph, numbers), 9))
    in_bullet = doc(results=f"<!-- gen:start:bullet -->{b0}→{b1}<!-- gen:end -->")
    assert len(gate(check(in_bullet, numbers), 9)) == 1
    in_table = doc(results=table(f"| T3 | {b0} | {b1} |"))
    assert gate(check(in_table, numbers), 9) == []
    apart = doc(results=f"B0: {b0}\n\nB1: {b1}")
    assert gate(check(apart, numbers), 9) == []
    numbers["eval.errors.T3.B1.mae_f"]["value"] = 50.0  # improvement above the floor
    assert gate(check(same_paragraph.replace("66.2", "50"), numbers), 9) == []


# --- A10 --------------------------------------------------------------------------------------


def test_a10(check, cfg: Settings, make_run: RunMaker) -> None:  # type: ignore[no-untyped-def]
    numbers = base_numbers()
    cited = doc(results=MARK.format(k="agent.eval.accuracy", v="0.83"))
    assert gate(check(cited, numbers), 10) == []
    numbers["agent.eval.accuracy"]["meta"] = {"backend": "mock", "model_id": "m", "trace_path": "t"}
    assert gate(check(cited, numbers), 10) == [
        "A10: mock-backend number agent.eval.accuracy is cited in README"
    ]
    assert gate(check(doc(), numbers), 10) == []  # mock numbers may exist, just not in README
    numbers["agent.eval.tokens"] = entry(10, {"backend": "gpt", "model_id": "", "trace_path": None})
    violations = gate(check(doc(), numbers), 10)
    assert violations == [
        "A10: agent key agent.eval.tokens: missing or invalid meta "
        "['backend', 'model_id', 'trace_path']"
    ]
    # manifest cross-check: meta says scripted, the manifest recorded anthropic
    make_run(
        "agent.eval",
        {
            "agent.eval.accuracy": 0.9,
            "agent.eval.accuracy@meta": {"backend": "scripted", "model_id": "x", "trace_path": "t"},
        },
        extras={"backend": "anthropic", "model_id": "claude-opus-5"},
    )
    refs = nums.collect(cfg.paths.runs_dir, out=cfg.report.numbers_path)
    assert refs["agent.eval.accuracy"].value == 0.9
    violations = gate(check(doc(), cfg.report.numbers_path, runs_dir=cfg.paths.runs_dir), 10)
    assert len(violations) == 2 and "meta.backend 'scripted' disagrees" in violations[0]
    assert "meta.model_id 'x' disagrees" in violations[1]


# --- A11 --------------------------------------------------------------------------------------


def test_a11(check, report_root: Path) -> None:  # type: ignore[no-untyped-def]
    assert gate(check(doc(), {}), 11) == []
    found = audit.card_locations(report_root)
    assert found["DATA_CARD.md"] == report_root / "docs" / "DATA_CARD.md"
    assert found["NOTICE"] == report_root / "NOTICE"
    notice = report_root / "NOTICE"
    notice.write_text(notice.read_text().replace("SSSP", "XXXX"))
    assert gate(check(doc(), {}), 11) == ["A11: NOTICE (found at NOTICE) does not mention 'SSSP'"]
    card = report_root / "docs" / "DATA_CARD.md"
    text = card.read_text()
    card.unlink()
    assert any(
        v.startswith("A11: DATA_CARD.md not found (looked at DATA_CARD.md, docs/DATA_CARD.md")
        for v in gate(check(doc(), {}), 11)
    )
    (report_root / "DATA_CARD.md").write_text(text.replace("never redistributed", "shipped"))
    violations = gate(check(doc(), {}), 11)
    assert (
        "A11: DATA_CARD.md (found at DATA_CARD.md) does not mention 'not redistributed'"
        in violations
    )
    (report_root / "MODEL_CARD.md").unlink()
    assert any(v.startswith("A11: MODEL_CARD.md not found") for v in gate(check(doc(), {}), 11))
