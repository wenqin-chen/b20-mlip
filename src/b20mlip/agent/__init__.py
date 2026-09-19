"""Agent tier (CONTRACTS.md row 13, SPEC.md section 7): a budgeted, provenance-checked
tool-calling agent over the pipeline's stage functions.

Modules (build order): ``trace`` (JSONL event log with argument/result/manifest hashes),
``guard`` (the :class:`~b20mlip.models.Budget` and element-allowlist guardrails), ``tools``
(one ``@beta_tool`` per SPEC tool, each a thin wrapper around a stage function that returns
numbers and run ids only), ``backends`` (``anthropic`` live runs through the SDK tool runner,
``mock`` trace replays for CI, ``scripted`` deterministic DAG planner = the gold script),
``runner`` (guard + trace + injected failures around a backend; produces an
:class:`~b20mlip.models.AgentReport`), ``tasks`` (the JSONL task file), ``eval`` (scoring and
the 12-task aggregate), ``screen`` (gold generation) and ``cli`` (``agent run``, ``agent eval``,
``screen``).

Every tool call runs as its own ``agent.run`` sub-stage (``runs/agent.run/<run_id>/`` with a
manifest, ``result.json`` and the stage's own outputs); the parent agent run records the trace,
``report.json`` and its ``numbers.json``. Tool sub-runs never publish ``numbers.json`` (their
stage numbers are stored as ``stage_numbers.json``), so an agent can never write README numbers
through a side door: only ``agent.run`` / ``agent.eval`` publish ``agent.*`` keys, and gate A10
keeps mock-backend numbers out of the README.
"""

from __future__ import annotations

__all__ = [
    "backends",
    "cli",
    "eval",
    "guard",
    "runner",
    "screen",
    "tasks",
    "tools",
    "trace",
]
