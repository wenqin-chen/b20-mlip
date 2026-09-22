"""Tools: strict schemas, every tool returns JSON with run ids and never raises, sub-run
provenance (result.json + manifest), write_report rejects unresolvable numbers, dry runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from b20mlip.agent import tools
from b20mlip.agent.tools import (
    RESULT_FILE,
    STAGE_NUMBERS_FILE,
    TOOL_NAMES,
    TOOLS,
    TOOLS_BY_NAME,
    ToolContext,
    find_run_dir,
    resolve_number,
    use_context,
)
from b20mlip.config import Settings
from b20mlip.provenance import read_manifest
from b20mlip.report import numbers as nums


def invoke(name: str, **args: Any) -> dict[str, Any]:
    """Call a tool the way the runner does (validated arguments, JSON string back)."""
    text = TOOLS_BY_NAME[name].call(args)
    assert isinstance(text, str)
    data = json.loads(text)
    assert isinstance(data, dict)
    return data


def test_schemas_are_strict_and_named() -> None:
    assert tuple(t.name for t in TOOLS) == TOOL_NAMES
    for tool in TOOLS:
        d = tool.to_dict()
        schema = d["input_schema"]
        assert d["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]), tool.name
        assert d["description"], tool.name
        for prop in schema["properties"].values():
            assert prop.get("type") or prop.get("$ref") or prop.get("items"), tool.name
    assert tools.list_data.to_dict()["input_schema"]["properties"]["kind"]["enum"]
    assert "shell" not in TOOL_NAMES and "bash" not in TOOL_NAMES


def test_tools_need_a_context() -> None:
    tools.set_context(None)
    data = json.loads(tools.list_data.call({"kind": "models"}))
    assert "error" in data and "ToolContext" in data["error"]


def test_list_data_and_get_structure(tool_context: ToolContext) -> None:
    with use_context(tool_context):
        listed = invoke("list_data", kind="all")
        assert listed["models"] == ["tiny_b20", "tiny_copy"] and listed["n_models"] == 2
        assert {"labelled_r0", "candidates_r0_filtered", "mptrj_b20"} <= set(listed["datasets"])
        assert listed["references"] == ["CoSi:qe", "FeSi:qe", "MnSi:qe"]
        assert listed["compounds"] == ["FeSi", "CoSi", "MnSi", "FeGe"] and listed["run_id"]
        s = invoke("get_structure", compound="FeSi")
        assert s["formula"] == "FeSi" and s["elements"] == ["Fe", "Si"] and s["natoms"] == 8
        assert s["a_A"] == pytest.approx(4.48) and s["spacegroup"] == 198 and s["frame_id"]
        assert s["run_id"] and s["status"] == "ok"
        assert tool_context.frame_elements(s["frame_id"]) == ["Fe", "Si"]
        assert tool_context.frame_natoms(s["frame_id"]) == 8
        bad = invoke("get_structure", compound="XxSi")
        assert "error" in bad and "unknown element" in bad["error"]
        missing = invoke("get_structure", compound="MnGe")
        assert "error" in missing


def test_relax_phonons_compare_and_provenance(tool_context: ToolContext) -> None:
    with use_context(tool_context):
        s = invoke("get_structure", compound="MnSi")
        r = invoke("relax", compound="MnSi", frame_id=s["frame_id"], fmax=1e-6, steps=3)
        assert "error" not in r and r["a_A"] > 0 and r["natoms"] == 8 and r["run_id"]
        assert r["frame_id"] != s["frame_id"] and r["steps"] == 3 and r["converged"] == 0
        run_dir = find_run_dir(tool_context.runs_dir, r["run_id"])
        assert run_dir is not None
        manifest = read_manifest(run_dir)
        assert manifest.stage == "agent.run" and manifest.status == "ok"
        assert manifest.extras["tool"] == "relax" and manifest.extras["args"]["fmax"] > 0
        names = {Path(a.path).name for a in manifest.outputs}
        assert {RESULT_FILE, "relaxed.extxyz"} <= names
        record = json.loads((run_dir / RESULT_FILE).read_text())
        assert record["tool"] == "relax" and record["numbers"]["a_A"] == r["a_A"]
        ref = resolve_number(tool_context.runs_dir, "a_A", r["a_A"], r["run_id"])
        assert ref.key == "a_A" and ref.manifest_sha256
        with pytest.raises(ValueError, match="does not match"):
            resolve_number(tool_context.runs_dir, "a_A", r["a_A"] + 1.0, r["run_id"])
        with pytest.raises(ValueError, match="has no manifest"):
            resolve_number(tool_context.runs_dir, "a_A", r["a_A"], "nope")
        # a key the run does not know is matched by value
        by_value = resolve_number(tool_context.runs_dir, "lattice", r["a_A"], r["run_id"])
        assert by_value.value == r["a_A"]
        with pytest.raises(ValueError, match="not a number produced"):
            resolve_number(tool_context.runs_dir, "lattice", 12345.678, r["run_id"])
        # unknown frame ids and a relax at the relaxed cell
        unknown = invoke("relax", compound="MnSi", frame_id="deadbeef", fmax=0.0, steps=0)
        assert "error" in unknown and "get_structure" in unknown["error"]
        p = invoke(
            "phonons", compound="MnSi", frame_id=s["frame_id"], supercell=[], distance=0.0, asr=True
        )
        assert "error" not in p and p["n_branches"] == 24 and p["n_qpoints"] == 100
        assert p["asr_applied"] == 1 and p["asr_violation"] == 0 and p["natoms_supercell"] == 8
        assert p["gamma_acoustic_max_abs_meV"] < 0.1 and p["cell_source"] == "dft"
        no_asr = invoke(
            "phonons", compound="MnSi", frame_id="", supercell=[1, 1, 1], distance=0.03, asr=False
        )
        assert no_asr["asr_applied"] == 0 and no_asr["run_id"] != p["run_id"]
        bad_cell = invoke(
            "phonons", compound="MnSi", frame_id="", supercell=[1, 1], distance=0.03, asr=True
        )
        assert "error" in bad_cell
        c = invoke("compare_phonons", run_id_model=p["run_id"], reference="qe")
        assert "error" not in c and c["compound"] == "MnSi" and c["reference_code"] == "mace"
        assert c["softening_index"] == pytest.approx(1.0) and c["omega_mae_meV"] == pytest.approx(
            0.0, abs=1e-6
        )
        assert invoke("compare_phonons", run_id_model=r["run_id"], reference="qe")["error"]
        assert (
            "no 'vasp'"
            in invoke("compare_phonons", run_id_model=p["run_id"], reference="vasp")["error"]
        )
        relaxed_ph = invoke(
            "phonons", compound="", frame_id=r["frame_id"], supercell=[], distance=0.0, asr=True
        )
        assert relaxed_ph["cell_source"] == "model_relaxed"


def test_evaluate_select_dft_and_md(tool_context: ToolContext) -> None:
    with use_context(tool_context):
        e = invoke("evaluate_errors", model="", frames="labelled_r0", split="", tier="T0")
        assert "error" not in e and e["mae_f"] > 0 and e["n_frames"] == 15 and e["run_id"]
        assert "mae_f_ci95_lo" in e and e["tier"] == "T0" and e["model"] == "tiny_b20"
        # the documented explicit labels ("primary", "none") mean the same as the empty strings
        explicit = invoke(
            "evaluate_errors", model="primary", frames="labelled_r0", split="none", tier="T0"
        )
        assert explicit["mae_f"] == e["mae_f"] and explicit["n_frames"] == e["n_frames"]
        run_dir = find_run_dir(tool_context.runs_dir, e["run_id"])
        assert run_dir is not None
        names = {Path(a.path).name for a in read_manifest(run_dir).outputs}
        assert STAGE_NUMBERS_FILE in names and "numbers.json" not in names  # never harvested
        assert nums.harvest(tool_context.runs_dir).entries == {}
        assert (
            "unknown dataset"
            in invoke("evaluate_errors", model="", frames="nope", split="", tier="T0")["error"]
        )
        assert (
            "unknown model"
            in invoke("evaluate_errors", model="B9", frames="labelled_r0", split="", tier="T0")[
                "error"
            ]
        )
        assert "error" in invoke("evaluate_errors", model="", frames="", split="", tier="T0")
        sel = invoke(
            "select_frames", models=["tiny_b20", "tiny_copy"], frames="candidates_r0_filtered", n=4
        )
        assert (
            sel["n_selected"] == 4
            and sel["n_candidates"] == 15
            and sel["sigma_f_max"] == pytest.approx(0.0)
        )
        assert "error" in invoke(
            "select_frames", models=["tiny_b20"], frames="candidates_r0_filtered", n=4
        )
        plan = invoke(
            "submit_dft", frames="candidates_r0_filtered", root="agent_r1", n_frames=3, submit=False
        )
        assert plan["n_units"] == 3 and plan["node_hours_estimate"] == pytest.approx(0.3)
        assert (
            plan["submitted"] == 0
            and (Path(tool_context.cfg.paths.dft_dir) / "agent_r1" / "units.json").is_file()
        )
        denied = invoke(
            "submit_dft", frames="candidates_r0_filtered", root="agent_r1", n_frames=3, submit=True
        )
        assert "not approved" in denied["error"]
        assert "error" in invoke(
            "submit_dft", frames="candidates_r0_filtered", root="../x", n_frames=1, submit=False
        )
        md = invoke(
            "run_md", compound="CoSi", frame_id="", ensemble="nve", T=300.0, ps=0.02, natoms=0
        )
        assert (
            "error" not in md
            and md["steps"] == 10
            and md["natoms"] == 8
            and "drift_meV_atom_ps" in md
        )
        assert md["T_mean_K"] > 0 and md["run_id"]
        assert "error" in invoke(
            "run_md", compound="CoSi", frame_id="", ensemble="nve", T=0.0, ps=0.02, natoms=0
        )


def test_write_report_validates_every_number(
    tool_context: ToolContext, agent_settings: Settings
) -> None:
    from b20mlip.provenance import RunContext

    parent = RunContext("agent.run", agent_settings, seed=0, runs_dir=agent_settings.paths.runs_dir)
    tool_context.ctx = parent
    with use_context(tool_context):
        s = invoke("get_structure", compound="FeSi")
        r = invoke("relax", compound="FeSi", frame_id=s["frame_id"], fmax=0.0, steps=0)
        rejected = invoke(
            "write_report", title="t", text="x",
            numbers=[{"key": "a_A", "value": r["a_A"], "run_id": r["run_id"]},
                     {"key": "a_A", "value": 9.99, "run_id": r["run_id"]}],
        )  # fmt: skip
        assert "rejected" in rejected["error"] and len(rejected["rejected"]) == 1
        assert not (parent.out_dir / "report.md").exists()
        assert "at least one" in invoke("write_report", title="t", text="x", numbers=[])["error"]
        ghost = invoke(
            "write_report",
            title="t",
            text="x",
            numbers=[{"key": "a_A", "value": r["a_A"], "run_id": "ghost"}],
        )
        assert "has no manifest" in ghost["rejected"][0]
        ok = invoke(
            "write_report", title="FeSi relaxation",
            text="a from run sk-ant-api03-abcdefghijklmnop",
            numbers=[{"key": "a_A", "value": r["a_A"], "run_id": r["run_id"]},
                     {"key": "energy_eV", "value": r["energy_eV"], "run_id": r["run_id"]}],
        )  # fmt: skip
        assert ok["n_numbers"] == 2 and ok["run_id"] and ok["report_sha256"]
        md = (parent.out_dir / "report.md").read_text()
        assert "# FeSi relaxation" in md and r["run_id"] in md and "sk-ant" not in md
        report = json.loads((parent.out_dir / "report.json").read_text())
        assert [n["key"] for n in report["numbers"]] == ["a_A", "energy_eV"]
        assert {Path(a.path).name for a in parent.outputs} == {"report.md", "report.json"}


def test_dry_run_plans_only(tool_context: ToolContext) -> None:
    tool_context.dry_run = True
    with use_context(tool_context):
        s = invoke("get_structure", compound="FeSi")
        assert s["planned"] == 1 and s["status"] == "partial" and "a_A" not in s
        run_dir = find_run_dir(tool_context.runs_dir, s["run_id"])
        assert run_dir is not None and read_manifest(run_dir).status == "partial"
        assert not (run_dir / RESULT_FILE).exists()


def test_tool_error_writes_failed_manifest(
    tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from b20mlip.evaluate import discovery

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr(discovery, "relax", boom)
    with use_context(tool_context):
        s = invoke("get_structure", compound="FeSi")
        r = invoke("relax", compound="FeSi", frame_id=s["frame_id"], fmax=0.0, steps=0)
    assert "optimizer exploded" in r["error"] and r["run_id"]
    run_dir = find_run_dir(tool_context.runs_dir, r["run_id"])
    assert run_dir is not None and read_manifest(run_dir).status == "failed"
    with pytest.raises(ValueError, match="status 'failed'"):
        resolve_number(tool_context.runs_dir, "a_A", 1.0, r["run_id"])


def test_context_helpers(tool_context: ToolContext, tmp_path: Path) -> None:
    tc = tool_context
    assert tc.model_label() == "tiny_b20" and tc.resolve_model("") == tc.model
    assert tc.resolve_model("tiny_copy").name == "tiny_copy.model"
    with pytest.raises(KeyError):
        tc.resolve_model("nope")
    assert tc.frames_natoms("labelled_r0") == [8] * 15 and tc.frames_elements("labelled_r0") == [
        "Co",
        "Fe",
        "Ge",
        "Mn",
        "Si",
    ]
    with pytest.raises(KeyError):
        tc.resolve_frames("nope")
    assert set(tc.split_files()) == {"v2"}
    assert tc.reference_path("FeSi", "qe").name == "phonons_FeSi_qe.json"
    with pytest.raises(KeyError):
        tc.resolve_split("nope")
    assert tc.frame_elements("unknown") is None and tc.frame_natoms("unknown") is None
    with pytest.raises(ValueError, match="compound or a frame_id"):
        tc.structure("", "")
    with pytest.raises(RuntimeError, match="no ToolContext"):
        tools.set_context(None)
        tools.current()
    bare = ToolContext(tc.cfg)
    with pytest.raises(FileNotFoundError):
        bare.calculator()
    assert bare.provenance()["label"] == "none" and bare.models() == {
        "tiny_copy": Path(tc.cfg.paths.models_dir) / "tiny_copy.model"
    }
