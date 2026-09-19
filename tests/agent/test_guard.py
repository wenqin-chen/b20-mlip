"""Guard: one test per Budget field, the element allowlist, unknown tools, counters."""

from __future__ import annotations

import pytest

from b20mlip.agent.guard import (
    DEFAULT_ALLOWLIST,
    Guard,
    GuardViolation,
    node_hours_estimate,
)
from b20mlip.agent.tools import TOOL_NAMES
from b20mlip.agent.trace import ToolCall
from b20mlip.models import Budget


def call(name: str, **args: object) -> ToolCall:
    return ToolCall(name, dict(args))


def test_defaults_and_node_hours() -> None:
    guard = Guard(Budget())
    assert guard.tool_names == frozenset(TOOL_NAMES) and guard.allowlist == DEFAULT_ALLOWLIST
    assert node_hours_estimate([8] * 10) == pytest.approx(1.0)
    assert node_hours_estimate([64, 8]) == pytest.approx(1.1)
    assert guard.remaining()["calls"] == 25 and guard.remaining()["approve_cluster"] is False


def test_unknown_tool_and_no_shell() -> None:
    guard = Guard(Budget())
    with pytest.raises(GuardViolation, match="unknown_tool") as info:
        guard.check(call("shell", cmd="rm -rf /"))
    assert info.value.rule == "unknown_tool" and "no shell" in str(info.value)
    assert guard.violations and guard.calls == 1
    restricted = Guard(Budget(), tool_names={"list_data"})
    with pytest.raises(GuardViolation, match="unknown_tool"):
        restricted.check(call("relax", compound="FeSi", frame_id="", fmax=0.0, steps=0))


def test_max_calls_counts_every_attempt() -> None:
    guard = Guard(Budget(max_calls=2))
    guard.check(call("list_data", kind="models"))
    guard.check(call("list_data", kind="datasets"))
    with pytest.raises(GuardViolation, match="max_calls"):
        guard.check(call("list_data", kind="all"))
    with pytest.raises(GuardViolation, match="max_calls"):  # still blocked afterwards
        guard.check(call("get_structure", compound="FeSi"))
    assert guard.calls == 4 and len(guard.violations) == 2


def test_max_wall_s() -> None:
    now = [100.0]
    guard = Guard(Budget(max_wall_s=10), clock=lambda: now[0])
    guard.check(call("list_data", kind="models"))
    now[0] = 111.0
    with pytest.raises(GuardViolation, match="max_wall_s"):
        guard.check(call("list_data", kind="models"))
    assert guard.elapsed() == pytest.approx(11.0)


def test_allowlist_on_compound_frame_id_and_dataset() -> None:
    guard = Guard(
        Budget(),
        structure_resolver=lambda fid: ["Cu", "Si"] if fid == "bad" else ["Fe", "Si"],
        frames_elements_resolver=lambda label: ["Ni", "Si"] if label == "alien" else ["Fe", "Si"],
    )
    guard.check(call("get_structure", compound="FeSi"))
    with pytest.raises(GuardViolation, match="allowlist") as info:
        guard.check(call("get_structure", compound="NiSi"))
    assert "['Ni']" in str(info.value)
    with pytest.raises(GuardViolation, match="allowlist"):
        guard.check(call("relax", compound="CoSi", frame_id="bad", fmax=0.0, steps=0))
    guard.check(call("relax", compound="CoSi", frame_id="good", fmax=0.0, steps=0))
    with pytest.raises(GuardViolation, match="allowlist"):
        guard.check(call("submit_dft", frames="alien", root="r", n_frames=0, submit=False))
    guard.check(call("submit_dft", frames="fine", root="r", n_frames=0, submit=False))
    guard.check(call("get_structure", compound="FeGe"))  # Ge is in the default allowlist


def test_narrow_allowlist() -> None:
    guard = Guard(Budget(), allowlist=frozenset({"Fe", "Si"}))
    guard.check(call("get_structure", compound="FeSi"))
    with pytest.raises(GuardViolation, match="allowlist"):
        guard.check(call("get_structure", compound="MnSi"))


def test_max_atoms_md_and_phonons() -> None:
    guard = Guard(Budget(max_atoms=64), structure_natoms_resolver=lambda fid: 8)
    guard.check(
        call("run_md", compound="FeSi", frame_id="", ensemble="nvt", T=300.0, ps=1.0, natoms=64)
    )
    with pytest.raises(GuardViolation, match="max_atoms"):
        guard.check(
            call("run_md", compound="FeSi", frame_id="", ensemble="nvt", T=300.0, ps=1.0, natoms=65)
        )
    guard.check(
        call("phonons", compound="FeSi", frame_id="x", supercell=[2, 2, 2], distance=0.03, asr=True)
    )
    with pytest.raises(GuardViolation, match="max_atoms"):
        guard.check(
            call(
                "phonons",
                compound="FeSi",
                frame_id="x",
                supercell=[3, 3, 3],
                distance=0.03,
                asr=True,
            )
        )


def test_max_md_ps_per_call() -> None:
    guard = Guard(Budget(max_md_ps=2.0))
    guard.check(
        call("run_md", compound="FeSi", frame_id="", ensemble="nve", T=300.0, ps=2.0, natoms=8)
    )
    assert guard.md_ps == pytest.approx(2.0)
    with pytest.raises(GuardViolation, match="max_md_ps"):
        guard.check(
            call("run_md", compound="FeSi", frame_id="", ensemble="nve", T=300.0, ps=2.5, natoms=8)
        )


def test_max_dft_frames_cumulative() -> None:
    guard = Guard(Budget(max_dft_frames=12), frames_resolver=lambda label: [8] * 10)
    guard.check(call("submit_dft", frames="pool", root="r", n_frames=0, submit=False))
    assert guard.dft_frames == 10
    with pytest.raises(GuardViolation, match="max_dft_frames"):
        guard.check(call("submit_dft", frames="pool", root="r", n_frames=5, submit=False))
    assert guard.dft_frames == 10  # a blocked call commits nothing
    guard.check(call("submit_dft", frames="pool", root="r", n_frames=2, submit=False))
    assert guard.dft_frames == 12


def test_max_node_hours_counts_large_cells() -> None:
    guard = Guard(Budget(max_node_hours=2.5), frames_resolver=lambda label: [64, 64, 8])
    guard.check(call("submit_dft", frames="pool", root="r", n_frames=0, submit=False))
    assert guard.node_hours == pytest.approx(2.1)
    with pytest.raises(GuardViolation, match="max_node_hours"):
        guard.check(call("submit_dft", frames="pool", root="r", n_frames=2, submit=False))
    unknown = Guard(Budget(max_node_hours=0.5))  # no resolver: 8-atom units assumed
    with pytest.raises(GuardViolation, match="max_node_hours"):
        unknown.check(call("submit_dft", frames="pool", root="r", n_frames=6, submit=False))


def test_approve_cluster() -> None:
    guard = Guard(Budget(approve_cluster=False), frames_resolver=lambda label: [8])
    guard.check(call("submit_dft", frames="pool", root="r", n_frames=1, submit=False))
    with pytest.raises(GuardViolation, match="approve_cluster"):
        guard.check(call("submit_dft", frames="pool", root="r", n_frames=1, submit=True))
    approved = Guard(Budget(approve_cluster=True), frames_resolver=lambda label: [8])
    approved.check(call("submit_dft", frames="pool", root="r", n_frames=1, submit=True))


def test_every_budget_field_has_a_rule() -> None:
    from b20mlip.agent.guard import RULES

    for name in Budget.model_fields:
        assert name in RULES, name
