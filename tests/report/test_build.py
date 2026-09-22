"""render(): pending everywhere with empty numbers, tables/captions/provenance with numbers,
gating (parity, offsets, bullet variants); write_readme prose preservation; the stage function."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from b20mlip.config import Settings
from b20mlip.models import NumberRef
from b20mlip.provenance import read_manifest, run_stage
from b20mlip.report import build
from b20mlip.report.markers import find_gen_blocks, find_num_markers

from .conftest import REF_EXP, REF_MACE, REF_QE, REF_VASP, FixtureRuns, meta

RUN = "20260918T100000-abcdef-0"
SHA = "a" * 64
BLOCKS = [
    "oneliner",
    "status",
    "forces",
    "energies",
    "discovery",
    "phonons",
    "parity",
    "thermal",
    "stability",
    "sampling",
    "agent",
    "bullet",
    "protocol",
]


def entry(value: float, m: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"value": value, "run_id": RUN, "manifest_sha256": SHA, "stage": "x", "meta": m or {}}


def full_numbers() -> dict[str, Any]:
    n: dict[str, Any] = {}
    for b, v in (("B0", 61.2), ("B1", 32.1), ("B2", 35.0)):
        n[f"eval.errors.T0.{b}.mae_f"] = entry(v, meta(tier="T0", ci95=[v - 2, v + 2]))
        n[f"eval.errors.T0.{b}.mae_e"] = entry(v / 10, meta(tier="T0"))
        n[f"eval.errors.T2.{b}.mae_f"] = entry(v + 5, meta(tier="T2"))
        n[f"eval.errors.T3.{b}.mae_f"] = entry(
            v + 10,
            meta(
                tier="T3",
                reference=REF_VASP,
                noise_floor_f=8.0,
                ci95=None,
                ci95_reason="single seed",
            ),
        )
    n["eval.discovery.B2.delta_f1"] = entry(
        0.012,
        meta(prevalence="natural", paired_vs="B0", sample_seed=0, n=1000, head="pt_head",
             energy_scale="mp", reference=REF_VASP),
    )  # fmt: skip
    n["eval.phonons.FeSi.B0.omega_mae_meV"] = entry(2.5, meta(n=100))
    n["eval.phonons.FeSi.B2.omega_mae_meV"] = entry(1.1, meta(n=100))
    n["eval.phonons.FeSi.B2.omega_mae_meV_pbesol"] = entry(
        1.9, meta(n=100, reference={**REF_QE, "functional": "PBEsol"}, cross_functional=True)
    )
    n["md.ase.MnSi.a_300K_A"] = entry(4.571, meta(n=2000, reference=REF_EXP))
    n["md.ase.MnSi.a_exp_A"] = entry(
        4.558, meta(n=1, reference=REF_EXP, ci95=None, ci95_reason="literature value")
    )
    n["md.ase.MnSi.a_dev_pct"] = entry(0.29, meta(n=2000, reference=REF_EXP))
    n["md.ase.MnSi.drift_meV_atom_ps"] = entry(0.02, meta(n=2000, reference=REF_MACE))
    n["parity.passed"] = entry(1)
    n["md.parity.max_dF_eVA"] = entry(
        0.0004, meta(n=20, ci95=None, ci95_reason="max", reference=REF_MACE)
    )
    n["sampling.umbrella.FeSi.dF_eV"] = entry(0.01, meta(n=12, reference=REF_MACE))
    n["sampling.wham.FeSi.B0.dF_barrier_eV"] = entry(
        0.83, meta(n=12, reference=REF_MACE, n_windows=24, ps_per_window=15.0)
    )
    n["sampling.neb.FeSi.Ea_eV"] = entry(
        0.79, meta(n=7, ci95=None, ci95_reason="NEB", reference=REF_MACE)
    )
    n["agent.eval.accuracy"] = entry(
        0.83, {"backend": "anthropic", "model_id": "claude-opus-5", "trace_path": "t.jsonl"}
    )
    n["agent.eval.n_tasks"] = entry(12, {"backend": "mock", "model_id": "mock", "trace_path": "t"})
    n["data.n_qe_frames"] = entry(512)
    n["data.n_train_frames"] = entry(400)
    n["active.n_selected"] = entry(96)
    n["offsets.residual_meV_atom"] = entry(11.0)
    return n


def block(text: str, name: str) -> str:
    m = re.search(rf"<!-- gen:start:{name} -->(.*?)<!-- gen:end -->", text, re.S)
    assert m is not None, name
    return m.group(1)


def test_render_empty_is_all_pending() -> None:
    text = build.render({})
    markers = find_num_markers(text)
    assert markers and all(m.text == "pending" for m in markers)
    assert [b.name for b in find_gen_blocks(text)] == BLOCKS
    assert text.count("_Not yet run: no `") == 8
    assert "deploy in ASE MD" in text and "LAMMPS" not in text.split("## Limitations")[0]
    assert "0 numbers from 0 runs are published" in text
    assert "**not yet run**" in text and text.endswith("\n")


def test_render_with_numbers_formats_cis_and_provenance() -> None:
    text = build.render(full_numbers())
    assert "<!-- num:eval.errors.T0.B1.mae_f -->32.1<!-- /num --> [30.1, 34.1]" in text
    assert "(no CI: single seed)" in text
    assert (
        "reference qe/PBE (SSSP-efficiency-1.3); E0 E0s_qe.json; head Default; n = 120; seed 0; "
        "CI95 [30.1, 34.1]; run `" + RUN + "`"
    ) in text
    assert "noise floor 8 meV/Å" in text and "cross-functional column" in text
    assert "deploy in ASE and LAMMPS MD" in text and "**passed**" in text
    assert "FeSi / LAMMPS" in block(text, "thermal")  # second engine rows listed once parity passed
    assert "| accuracy | <!-- num:agent.eval.accuracy -->0.83<!-- /num -->" in block(text, "agent")
    assert "agent.eval.n_tasks" not in text  # mock-backend numbers never rendered
    assert "_Pending rows (no number published yet): FeSi / B1" in block(text, "phonons")
    bullet = block(text, "bullet")
    assert "→<!-- num:eval.errors.T0.B2.mae_f -->35<!-- /num --> meV/Å" in bullet  # B2 preferred
    assert "one committee-uncertainty active-learning round" in bullet
    assert "<!-- num:data.n_train_frames -->400<!-- /num --> in-house" in bullet
    assert "<!-- num:data.n_qe_frames -->512<!-- /num --> labelled in total" in bullet
    assert "ΔF‡ = <!-- num:sampling.wham.FeSi.B0.dF_barrier_eV -->0.83<!-- /num --> eV" in bullet
    assert "24 umbrella windows × 15 ps" in block(text, "sampling")
    assert "<!-- num:md.parity.max_dF_eVA -->0.0004<!-- /num -->" in block(text, "parity")


def test_render_gates_and_variants() -> None:
    n = full_numbers()
    n["parity.passed"] = entry(0)
    n["offsets.residual_meV_atom"] = entry(25.0)
    n.pop("data.n_qe_frames")
    n.pop("data.n_train_frames")
    n["data.n_omat24_frames"] = entry(300)
    n.pop("active.n_selected")
    text = build.render(n)
    assert "deploy in ASE MD" in text and "ASE/LAMMPS" not in text and "**not passed**" in text
    assert "LAMMPS" not in block(text, "thermal")
    assert "<!-- num:eval.discovery.B1.delta_f1 -->n/a (scale)<!-- /num -->" in text
    assert "OMat24 DFT frames" in text and "active-learning" not in block(text, "bullet")
    n["eval.phonons.CoSi.B0.omega_mae_meV"] = entry(
        3.3, meta(reference={**REF_QE, "code": "pyscf"}, lower_fidelity=True, n=100)
    )
    text = build.render(n)
    assert "| CoSi / B0 (lower fidelity) |" in text and "lower fidelity; run" in text


def test_normalize_and_number_view() -> None:
    refs: dict[str, Any] = {
        "a.b": NumberRef(key="a.b", value=1.5, run_id=RUN, manifest_sha256=SHA),
        "@stale": [{"key": "x"}],
        "@meta": {},
    }
    entries, stale = build.normalize(refs)
    assert entries == {"a.b": {"value": 1.5, "run_id": RUN, "manifest_sha256": SHA, "meta": {}}}
    assert stale == [{"key": "x"}]
    with pytest.raises(TypeError):
        build.normalize({"a.b": 3})
    view = build.NumberView(entries)
    assert view.marker("a.b") == "<!-- num:a.b -->1.5<!-- /num -->"
    assert view.marker("zz") == "<!-- num:zz -->pending<!-- /num -->"
    assert view.marker("zz", missing="n/a") == "<!-- num:zz -->n/a<!-- /num -->"
    assert view.ci("zz") == "" and view.ci("a.b") == "(no CI: no CI given)"
    assert view.value("zz", 7.0) == 7.0 and view.value("a.b") == 1.5
    assert view.meta("a.b", "reference", "code", default="?") == "?"
    assert view.cell("a.b") == "<!-- num:a.b -->1.5<!-- /num -->"  # the reason is in prov
    assert view.prov("a.b").startswith("reference ?/?; E0 ?; head ?; n = ?; seed ?; CI95 (no CI")


def test_write_readme_preserves_prose_and_refreshes_markers(tmp_path: Path) -> None:
    numbers = {"a.b": entry(1.5)}
    rendered = (
        "# T\n<!-- gen:start:one -->NEW1<!-- gen:end -->\n\n"
        "<!-- gen:start:two -->NEW2<!-- gen:end -->\n"
    )
    path = tmp_path / "README.md"
    assert build.write_readme(rendered, path).read_text() == rendered  # no file: whole render
    hand = (
        "# Mine\nkeep <!-- num:a.b -->pending<!-- /num --> and "
        "<!-- num:zz -->pending<!-- /num -->\n"
        "<!-- gen:start:one -->OLD1<!-- gen:end -->\nmore prose\n"
    )
    path.write_text(hand)
    out = build.write_readme(rendered, path, numbers=numbers).read_text()
    assert out.startswith(
        "# Mine\nkeep <!-- num:a.b -->1.5<!-- /num --> and <!-- num:zz -->pending<!-- /num -->\n"
        "<!-- gen:start:one -->NEW1<!-- gen:end -->\nmore prose\n"
    )
    assert out.rstrip().endswith("<!-- gen:start:two -->NEW2<!-- gen:end -->")
    path.write_text("# Old\n<!-- gen:start:x -->unclosed\n")  # malformed: overwritten wholesale
    assert build.write_readme(rendered, path).read_text() == rendered
    path.write_text("no blocks at all\n")
    assert build.write_readme(rendered, path).read_text() == rendered


def test_merge_keeps_unknown_existing_blocks() -> None:
    existing = "<!-- gen:start:old -->keep<!-- gen:end -->\n"
    merged = build.merge_gen_blocks(existing, "<!-- gen:start:new -->n<!-- gen:end -->\n")
    assert merged == existing + "\n<!-- gen:start:new -->n<!-- gen:end -->\n"


def test_run_stage_writes_numbers_and_readme(
    cfg: Settings, fixture_runs: FixtureRuns, report_root: Path
) -> None:
    readme = report_root / "README.md"
    result = run_stage("report", cfg, build.run, readme=True, readme_path=readme)
    assert result.status == "ok"
    assert result.summary["n_numbers"] == 4 and result.summary["n_stale"] == 1
    assert result.summary["n_manifests_ok"] == 3 and result.summary["readme"] == str(readme)
    assert [Path(a.path).name for a in result.outputs] == ["numbers.json", "README.md"]
    assert read_manifest(result.manifest_path).extras["stale_keys"] == ["md.ase.MnSi.a_300K_A"]
    text = readme.read_text()
    assert "<!-- num:eval.errors.T0.B1.mae_f -->30.5<!-- /num -->" in text
    readme.write_text(text.replace("## What it does", "## What it does\n\nHand-written note.", 1))
    again = run_stage("report", cfg, build.run, readme=True, readme_path=readme)
    assert again.status == "ok" and "Hand-written note." in readme.read_text()
    numbers_path = Path(cfg.report.numbers_path)
    numbers_path.unlink()
    dry = run_stage("report", cfg, build.run, dry_run=True, readme=True, readme_path=readme)
    assert dry.status == "partial" and dry.outputs == [] and not numbers_path.exists()
    plain = run_stage("report", cfg, build.run)
    assert plain.summary["readme"] == "" and numbers_path.is_file()


def test_make_table_formats_values_and_their_cis() -> None:
    key = "md.lammps.MnSi.a_300K_A"
    view = build.NumberView(
        {key: {"value": 4.556777, "run_id": "r", "meta": {"ci95": [4.55672, 4.55686]}}}
    )
    table = build.make_table(
        view, "t", "md", ["Compound", "a"], "caption", [build.Row("MnSi", [key])],
        formats={"a_300K_A": "{:.4f}"},
    )  # fmt: skip
    assert table.rows[0]["cells"] == [f"<!-- num:{key} -->4.5568<!-- /num --> [4.5567, 4.5569]"]


def test_model_labelled_md_rows_need_their_own_parity() -> None:
    """Runs from 2026-09-22 publish md.<engine>.<compound>.<label>.* and parity.<label>.*; the
    legacy unlabelled keys stay B0's, and a model's LAMMPS rows need that model's own gate."""
    n = full_numbers()  # legacy B0 keys: md.ase.MnSi.*, parity.passed, md.parity.*
    b2 = meta(n=1001, reference=REF_EXP, model_label="B2", natoms=512, steps=15000, timestep_fs=2)
    n["md.lammps.MnSi.B2.a_300K_A"] = entry(4.5312, b2)
    n["md.lammps.MnSi.B2.a_exp_A"] = entry(4.558, b2)
    n["md.lammps.MnSi.B2.a_dev_pct"] = entry(-0.588, b2)
    text = build.render(n)
    assert "MnSi / LAMMPS / B2" not in block(text, "thermal")  # no B2 parity yet
    assert "MnSi / ASE / B0" in block(text, "thermal")  # legacy rows survive
    assert "zero-shot MPA-0" in block(text, "bullet")
    n["parity.B2.passed"] = entry(1, meta(model_label="B2"))
    n["md.parity.B2.max_dF_eVA"] = entry(2e-14, meta(n=20, reference=REF_MACE, model_label="B2"))
    text = build.render(n)
    thermal = block(text, "thermal")
    assert "MnSi / LAMMPS / B2" in thermal and "MnSi / ASE / B0" in thermal
    assert "<!-- num:md.lammps.MnSi.B2.a_300K_A -->4.5312<!-- /num -->" in thermal
    parity = block(text, "parity")
    assert "B0 MPA-0 zero-shot" in parity and "B2 multihead replay" in parity
    assert "<!-- num:md.parity.B2.max_dF_eVA -->2e-14<!-- /num -->" in parity
    bullet = block(text, "bullet")
    assert "the fine-tuned model in LAMMPS: MnSi 300 K lattice constant" in bullet
    assert "<!-- num:md.lammps.MnSi.B2.a_dev_pct -->-0.588<!-- /num -->" in bullet
