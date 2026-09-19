"""Guardrails (SPEC.md section 7): every :class:`~b20mlip.models.Budget` field, the element
allowlist and the tool allowlist, checked *before* a tool call executes.

Rules (binding; one violation string per blocked call, fed back to the model as the tool result)

* ``unknown_tool`` — the tool is not one of :data:`~b20mlip.agent.tools.TOOL_NAMES` or not in the
  task's ``tools_allowed``; there is no shell tool, so a shell can never be requested.
* ``max_calls`` — every *attempted* call counts (blocked ones too: the model spent the turn).
* ``max_wall_s`` — wall clock since the guard was created.
* ``allowlist`` — every structure-bearing argument (``compound``, ``compounds``, ``elements``,
  ``formula``; a ``frame_id`` resolved through the tool context; a ``frames``/``structure``
  dataset resolved to its atoms) may contain only {Fe, Mn, Co, Si, Ge}.
* ``max_atoms`` — ``run_md.natoms``; ``phonons`` supercells (8-atom B20 cell x prod(supercell)
  unless the structure resolver knows the cell size).
* ``max_md_ps`` — ``run_md.ps`` per call.
* ``max_dft_frames`` / ``max_node_hours`` — cumulative over ``submit_dft`` calls; the node-hour
  estimate is :func:`node_hours_estimate` (0.1 node-h per frame below 32 atoms, 1 node-h above:
  SPEC.md section 5, 8-atom ~3 min, 64-atom ~1 h, rounded up to the SPEC budget unit).
* ``approve_cluster`` — ``submit_dft(submit=True)`` needs ``budget.approve_cluster``; without it
  the call is blocked (the tool itself never submits without the flag either).

Counters (``dft_frames``, ``node_hours``, ``md_ps``) are committed only when a call passes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from b20mlip.agent.trace import ToolCall
from b20mlip.dft.structures import elements_of
from b20mlip.models import Budget

DEFAULT_ALLOWLIST: frozenset[str] = frozenset({"Fe", "Mn", "Co", "Si", "Ge"})
B20_CELL_ATOMS = 8
LARGE_CELL_ATOMS = 32  # frames at or above this count as "64-atom" units for node hours
NODE_HOURS_SMALL = 0.1  # per 8-atom frame (SPEC section 5: ~3 min, budgeted at 0.1 node-h)
NODE_HOURS_LARGE = 1.0  # per 64-atom frame (~1 h)
STRUCTURE_ARGS: tuple[str, ...] = ("compound", "compounds", "elements", "formula")
FRAME_ID_ARGS: tuple[str, ...] = ("frame_id",)
FRAMES_ARGS: tuple[str, ...] = ("frames", "frames_path", "structure")
RULES: tuple[str, ...] = (
    "unknown_tool",
    "max_calls",
    "max_wall_s",
    "allowlist",
    "max_atoms",
    "max_md_ps",
    "max_dft_frames",
    "max_node_hours",
    "approve_cluster",
)


class GuardViolation(Exception):
    """A blocked call; ``rule`` names the Budget field / rule that fired."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"guard violation ({rule}): {message}")
        self.rule = rule
        self.detail = message


def node_hours_estimate(natoms_per_frame: Iterable[int]) -> float:
    """Node-hour estimate of a QE round: 0.1 per small (< 32-atom) frame, 1.0 per large one."""
    total = 0.0
    for natoms in natoms_per_frame:
        total += NODE_HOURS_LARGE if int(natoms) >= LARGE_CELL_ATOMS else NODE_HOURS_SMALL
    return round(float(total), 6)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _prod(values: Iterable[Any]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


@dataclass
class Guard:
    """Budget + allowlist checks (see the module docstring); ``check`` raises on a violation."""

    budget: Budget
    allowlist: frozenset[str] = DEFAULT_ALLOWLIST
    tool_names: frozenset[str] | None = None
    frames_resolver: Callable[[str], list[int]] | None = None  # dataset label -> atoms per frame
    frames_elements_resolver: Callable[[str], list[str]] | None = None  # dataset -> elements
    structure_resolver: Callable[[str], list[str] | None] | None = None  # frame_id -> elements
    structure_natoms_resolver: Callable[[str], int | None] | None = None  # frame_id -> natoms
    clock: Callable[[], float] = time.monotonic
    calls: int = 0
    dft_frames: int = 0
    node_hours: float = 0.0
    md_ps: float = 0.0
    violations: list[str] = field(default_factory=list)
    t0: float = field(init=False)

    def __post_init__(self) -> None:
        self.allowlist = frozenset(self.allowlist)
        if self.tool_names is None:
            from b20mlip.agent.tools import TOOL_NAMES  # noqa: PLC0415 - avoids an import cycle

            self.tool_names = frozenset(TOOL_NAMES)
        else:
            self.tool_names = frozenset(self.tool_names)
        self.t0 = float(self.clock())

    # -- bookkeeping -------------------------------------------------------------------------

    def elapsed(self) -> float:
        return float(self.clock()) - self.t0

    def remaining(self) -> dict[str, float | int | bool]:
        b = self.budget
        return {
            "calls": max(0, b.max_calls - self.calls),
            "wall_s": max(0.0, float(b.max_wall_s) - self.elapsed()),
            "dft_frames": max(0, b.max_dft_frames - self.dft_frames),
            "node_hours": max(0.0, float(b.max_node_hours) - self.node_hours),
            "approve_cluster": bool(b.approve_cluster),
        }

    # -- the check ---------------------------------------------------------------------------

    def check(self, call: ToolCall) -> None:
        """Count the call, verify every rule, commit the counters; raise :class:`GuardViolation`."""
        self.calls += 1
        try:
            commit = self._check(call)
        except GuardViolation as exc:
            self.violations.append(str(exc))
            raise
        for key, value in commit.items():
            setattr(self, key, getattr(self, key) + value)

    def _check(self, call: ToolCall) -> dict[str, Any]:
        b = self.budget
        assert self.tool_names is not None
        if call.name not in self.tool_names:
            raise GuardViolation(
                "unknown_tool",
                f"{call.name!r} is not an available tool (no shell; allowed: "
                f"{sorted(self.tool_names)})",
            )
        if self.calls > b.max_calls:
            raise GuardViolation(
                "max_calls",
                f"call budget exhausted ({b.max_calls} calls); answer with the numbers you have",
            )
        elapsed = self.elapsed()
        if elapsed > b.max_wall_s:
            raise GuardViolation(
                "max_wall_s", f"wall-clock budget exhausted ({elapsed:.0f} s > {b.max_wall_s} s)"
            )
        bad = sorted(set(self.structure_elements(call)) - self.allowlist)
        if bad:
            raise GuardViolation(
                "allowlist",
                f"elements {bad} are outside the allowlist {sorted(self.allowlist)}; "
                "discard this structure and fetch a valid one",
            )
        commit: dict[str, Any] = {}
        args = call.args
        if call.name == "run_md":
            natoms = int(args.get("natoms") or 0)
            if natoms > b.max_atoms:
                raise GuardViolation(
                    "max_atoms", f"run_md natoms={natoms} exceeds the budget ({b.max_atoms})"
                )
            ps = float(args.get("ps") or 0.0)
            if ps > b.max_md_ps:
                raise GuardViolation(
                    "max_md_ps", f"run_md ps={ps} exceeds the budget ({b.max_md_ps} ps per call)"
                )
            commit["md_ps"] = ps
        elif call.name == "phonons":
            cell = self._cell_atoms(args)
            natoms = cell * _prod(_as_list(args.get("supercell")) or [1, 1, 1])
            if natoms > b.max_atoms:
                raise GuardViolation(
                    "max_atoms",
                    f"phonons supercell holds {natoms} atoms, more than the budget ({b.max_atoms})",
                )
        elif call.name == "submit_dft":
            natoms_list = self._frames_natoms(args)
            requested = int(args.get("n_frames") or 0)
            if requested > 0:
                natoms_list = natoms_list[:requested]
            n = len(natoms_list) if natoms_list else requested
            if self.dft_frames + n > b.max_dft_frames:
                raise GuardViolation(
                    "max_dft_frames",
                    f"{n} DFT frames requested with {self.dft_frames} already planned exceeds the "
                    f"budget ({b.max_dft_frames})",
                )
            hours = (
                node_hours_estimate(natoms_list)
                if natoms_list
                else node_hours_estimate([B20_CELL_ATOMS] * n)
            )
            if self.node_hours + hours > b.max_node_hours:
                raise GuardViolation(
                    "max_node_hours",
                    f"estimated {hours:.2f} node-h (plus {self.node_hours:.2f} already planned) "
                    f"exceeds the budget ({b.max_node_hours} node-h)",
                )
            if bool(args.get("submit")) and not b.approve_cluster:
                raise GuardViolation(
                    "approve_cluster",
                    "cluster submission is not approved (run with --approve-cluster); "
                    "plan with submit=false instead",
                )
            commit["dft_frames"] = n
            commit["node_hours"] = hours
        return commit

    # -- structure-bearing arguments ---------------------------------------------------------

    def structure_elements(self, call: ToolCall) -> list[str]:
        """Every element symbol the call's arguments refer to (compounds, frame ids, datasets)."""
        found: list[str] = []
        args = call.args
        for name in STRUCTURE_ARGS:
            for item in _as_list(args.get(name)):
                if isinstance(item, str) and item:
                    found.extend(elements_of(item))
        for name in FRAME_ID_ARGS:
            frame_id = args.get(name)
            if isinstance(frame_id, str) and frame_id and self.structure_resolver is not None:
                found.extend(self.structure_resolver(frame_id) or [])
        for name in FRAMES_ARGS:
            label = args.get(name)
            if isinstance(label, str) and label and self.frames_elements_resolver is not None:
                try:
                    found.extend(self.frames_elements_resolver(label))
                except Exception:  # noqa: BLE001 - an unknown dataset fails inside the tool
                    continue
        return list(dict.fromkeys(found))

    def _cell_atoms(self, args: dict[str, Any]) -> int:
        frame_id = args.get("frame_id")
        if isinstance(frame_id, str) and frame_id and self.structure_natoms_resolver is not None:
            natoms = self.structure_natoms_resolver(frame_id)
            if natoms:
                return int(natoms)
        return B20_CELL_ATOMS

    def _frames_natoms(self, args: dict[str, Any]) -> list[int]:
        label = args.get("frames")
        if isinstance(label, str) and label and self.frames_resolver is not None:
            try:
                return [int(n) for n in self.frames_resolver(label)]
            except Exception:  # noqa: BLE001 - unknown dataset: the tool reports it
                return []
        return []


__all__ = [
    "B20_CELL_ATOMS",
    "DEFAULT_ALLOWLIST",
    "FRAMES_ARGS",
    "LARGE_CELL_ATOMS",
    "NODE_HOURS_LARGE",
    "NODE_HOURS_SMALL",
    "RULES",
    "STRUCTURE_ARGS",
    "Guard",
    "GuardViolation",
    "node_hours_estimate",
]
