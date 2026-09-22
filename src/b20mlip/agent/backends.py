"""Agent backends (SPEC.md section 7): ``anthropic`` (live, SDK tool runner), ``mock`` (trace
replay, CI) and ``scripted`` (deterministic DAG planner = the ``b20mlip screen`` gold script).

Every backend has ``run(task, tools) -> BackendResult`` where ``tools`` is the runner's
dispatcher (:class:`ToolDispatcher`): ``tools.call(name, args)`` executes one guarded, traced
tool call and returns its result dict; ``tools.beta_tools(allowed)`` hands the live backend
the guarded ``@beta_tool`` objects; ``tools.context`` is the :class:`~b20mlip.agent.tools.
ToolContext`. Backends never touch the guard or the trace themselves.

Answer format (shared by every backend; the live model is asked for it in :data:`PROMPT`)::

    {"answer": {"<key>": <number or string>, ...},
     "numbers": [{"key": "<key>", "value": <number>, "run_id": "<run id of the tool call>"}],
     "text": "<one paragraph>"}

``answer`` holds the values the task asks for, ``numbers`` cites the tool run every numeric
value came from (the runner's grounding check resolves each citation against that run's
manifest and ``result.json``).

Prices (USD per million tokens) come from :data:`PRICES_USD_PER_MTOK`; the live backend
counts ``usage.input_tokens`` / ``usage.output_tokens`` of every turn.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from b20mlip.agent.guard import DEFAULT_ALLOWLIST
from b20mlip.agent.trace import Trace
from b20mlip.models import AgentTask

DEFAULT_MODEL_ID = "claude-opus-5"
# USD per 1M tokens (input, output); Anthropic first-party API rates as listed in the claude-api
# skill's model table (cached 2026-06-24) / https://platform.claude.com/pricing. Cache reads and
# writes are billed differently and are not modelled (the agent sends no cache_control).
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # SPEC section 7: fallbacks="default"
DEFAULT_MAX_TOKENS = 16000
API_KEY_ENV = "ANTHROPIC_API_KEY"
BACKEND_NAMES: tuple[str, ...] = ("anthropic", "mock", "scripted")

PROMPT = """You are a careful computational materials scientist driving the b20-mlip pipeline
(machine-learning interatomic potentials for B20 compounds FeSi, CoSi, MnSi, FeGe) through tools.

Tool contract (binding):
- Every number you report MUST come from a tool result in this conversation; never estimate,
  recall or extrapolate a value. Cite the run_id of the tool call that produced each number.
- Order: call get_structure (or list_data) before relax, phonons or run_md; compare_phonons only
  with the run_id of an earlier phonons call; evaluate_errors before select_frames; write_report,
  when the task asks for a report, is the last call.
- Only Fe, Mn, Co, Si and Ge are allowed. If a structure contains any other element, discard it
  and call get_structure again.
- If a phonons result says asr_violation = true, re-run phonons with asr = true before using it.
- A tool result with an "error" field failed: fix the arguments or retry once, then move on.
- Budget: {max_calls} tool calls, {max_wall_s} s, {max_dft_frames} DFT frames, {max_node_hours}
  node-hours, {max_atoms} atoms, {max_md_ps} ps of MD per call; cluster submission
  {cluster_note}. A blocked call returns a guard violation: answer with what you have.
- Defaults unless the task says otherwise: phonon supercell {supercell}, displacement
  {distance} A, fmax {fmax} eV/A, {max_steps} relaxation steps, {natoms} atoms for MD.

Final answer: reply with ONE JSON object and nothing else:
{{"answer": {{"<key>": <number or short string>, ...}},
 "numbers": [{{"key": "<key>", "value": <number>, "run_id": "<run id>"}}, ...],
 "text": "<one or two sentences>"}}
Use the exact keys the task names (e.g. imaginary_count, a_A, softening_index, mae_f,
n_selected, node_hours_estimate, drift_meV_atom_ps, n_models); leave a key out rather than
guess it."""


class BackendUnavailable(RuntimeError):
    """The backend cannot be constructed here (no API key, no trace file, ...)."""


@dataclass
class BackendResult:
    answer: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_id: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    stop_reason: str | None = None
    error: str | None = None


class ToolDispatcher(Protocol):
    """What the runner hands a backend (see :class:`b20mlip.agent.runner.Dispatcher`)."""

    context: Any
    trace: Trace

    def call(
        self, name: str, args: Mapping[str, Any], *, stored_result: Any = None
    ) -> dict[str, Any]: ...

    def beta_tools(self, allowed: Iterable[str] | None = None) -> list[Any]: ...


# --- helpers ------------------------------------------------------------------------------------


def cost_usd(model_id: str | None, tokens_in: int, tokens_out: int) -> float:
    """USD for the given token counts (0 for unknown / non-billed model ids)."""
    if not model_id or model_id not in PRICES_USD_PER_MTOK:
        return 0.0
    price_in, price_out = PRICES_USD_PER_MTOK[model_id]
    return tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _last_json_object(text: str) -> dict[str, Any] | None:
    """The last balanced ``{...}`` in ``text`` that parses as a JSON object."""
    end = len(text)
    while True:
        close = text.rfind("}", 0, end)
        if close < 0:
            return None
        depth = 0
        start = -1
        for i in range(close, -1, -1):
            ch = text[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start < 0:
            return None
        try:
            data = json.loads(text[start : close + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        end = close


def parse_answer(text: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """``(answer, citations, text)`` from a final message (see the module docstring)."""
    data: dict[str, Any] | None = None
    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
        data = loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        data = None
    if data is None:
        for m in _FENCE_RE.finditer(stripped):
            try:
                loaded = json.loads(m.group(1))
                if isinstance(loaded, dict):
                    data = loaded
            except json.JSONDecodeError:
                continue
    if data is None:
        data = _last_json_object(stripped)
    if data is None:
        return {}, [], stripped
    if "answer" in data and isinstance(data["answer"], dict):
        answer = dict(data["answer"])
        raw = data.get("numbers", data.get("citations", []))
        body = str(data.get("text", ""))
    else:
        answer = {k: v for k, v in data.items() if k not in ("numbers", "citations", "text")}
        raw = data.get("numbers", data.get("citations", []))
        body = str(data.get("text", ""))
    citations: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and {"key", "value", "run_id"} <= set(item):
            citations.append(
                {"key": str(item["key"]), "value": item["value"], "run_id": str(item["run_id"])}
            )
    return answer, citations, body or stripped


def format_answer(
    answer: Mapping[str, Any], citations: Sequence[Mapping[str, Any]], text: str
) -> str:
    return json.dumps(
        {"answer": dict(answer), "numbers": list(citations), "text": text}, sort_keys=True
    )


def system_prompt(tc: Any, budget: Any) -> str:
    cfg = tc.cfg
    return PROMPT.format(
        max_calls=budget.max_calls,
        max_wall_s=budget.max_wall_s,
        max_dft_frames=budget.max_dft_frames,
        max_node_hours=budget.max_node_hours,
        max_atoms=budget.max_atoms,
        max_md_ps=budget.max_md_ps,
        cluster_note="is approved" if budget.approve_cluster else "is NOT approved (plan only)",
        supercell=list(cfg.agent.phonon_supercell),
        distance=cfg.agent.phonon_distance,
        fmax=cfg.eval.fmax,
        max_steps=cfg.eval.max_steps,
        natoms=cfg.md.natoms,
    )


# --- anthropic ----------------------------------------------------------------------------------


class AnthropicBackend:
    """Live runs through ``client.beta.messages.tool_runner`` with the guarded ``@beta_tool``s.

    Constructible only when ``ANTHROPIC_API_KEY`` is set (or a client is injected): CI never
    builds an ``anthropic.Anthropic`` client (CONTRACTS section 7).
    """

    name = "anthropic"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        client: Any | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fallbacks: Any = "default",
        betas: Sequence[str] = (FALLBACK_BETA,),
        system: str | None = None,
    ) -> None:
        if client is None:
            if not os.environ.get(API_KEY_ENV):
                raise BackendUnavailable(
                    f"{API_KEY_ENV} is not set: the anthropic backend is unavailable here; use "
                    "--backend mock (trace replay) or --backend scripted"
                )
            import anthropic  # noqa: PLC0415 - only when a key exists

            client = anthropic.Anthropic()
        self.client = client
        self.model_id = model_id
        self.max_tokens = int(max_tokens)
        self.fallbacks = fallbacks
        self.betas = list(betas)
        self.system = system

    def run(self, task: AgentTask, tools: ToolDispatcher) -> BackendResult:
        system = self.system or system_prompt(tools.context, task.budget)
        kwargs: dict[str, Any] = {}
        if self.fallbacks is not None:
            kwargs["fallbacks"] = self.fallbacks
        if self.betas:
            kwargs["betas"] = list(self.betas)
        runner = self.client.beta.messages.tool_runner(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": task.prompt}],
            tools=tools.beta_tools(task.tools_allowed or None),
            max_iterations=int(task.budget.max_calls) + 2,
            **kwargs,
        )
        tokens_in = tokens_out = 0
        last: Any = None
        turn = 0
        for message in runner:
            turn += 1
            last = message
            usage = getattr(message, "usage", None)
            t_in = int(getattr(usage, "input_tokens", 0) or 0)
            t_out = int(getattr(usage, "output_tokens", 0) or 0)
            tokens_in += t_in
            tokens_out += t_out
            tools.trace.usage(t_in, t_out, model_id=self.model_id, turn=turn)
        text = ""
        stop_reason = None
        if last is not None:
            stop_reason = getattr(last, "stop_reason", None)
            text = "".join(
                str(getattr(block, "text", ""))
                for block in last.content
                if getattr(block, "type", "") == "text"
            )
        answer, citations, body = parse_answer(text)
        tools.trace.message("assistant", text, stop_reason=stop_reason)
        return BackendResult(
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd(self.model_id, tokens_in, tokens_out),
            model_id=self.model_id,
            citations=citations,
            text=body,
            stop_reason=str(stop_reason) if stop_reason is not None else None,
            error=None if answer else "no JSON answer in the final message",
        )


# --- mock ---------------------------------------------------------------------------------------


_ID_KEYS: tuple[str, ...] = ("frame_id", "run_id", "run_id_model")


def _remap(obj: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _remap(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remap(v, mapping) for v in obj]
    if isinstance(obj, str) and obj in mapping:
        return mapping[obj]
    return obj


class MockBackend:
    """Replay a recorded trace: the same tool calls in order, then the recorded final answer.

    ``trace_path`` is a trace file or a directory holding ``<task_id>.jsonl``. By default the
    tools are re-executed (fresh sub-runs; recorded ids in later arguments and in the answer's
    citations are re-mapped to the fresh ones and cited values refreshed from the fresh
    results); with ``replay_results=True`` the stored results are returned instead and nothing
    is computed (the citations then point at runs that do not exist, so such a report is never
    grounded).
    """

    name = "mock"

    def __init__(self, trace_path: str | Path, *, replay_results: bool = False) -> None:
        self.trace_path = Path(trace_path)
        self.replay_results = bool(replay_results)

    def trace_for(self, task: AgentTask) -> Path:
        if self.trace_path.is_dir():
            return self.trace_path / f"{task.task_id}.jsonl"
        return self.trace_path

    def run(self, task: AgentTask, tools: ToolDispatcher) -> BackendResult:
        path = self.trace_for(task)
        if not path.is_file():
            raise BackendUnavailable(f"no recorded trace for task {task.task_id!r} at {path}")
        events = Trace.read(path)
        header = next(
            (e for e in events if e["kind"] == "message" and e.get("role") == "system"), {}
        )
        model_id = header.get("model_id") or "mock"
        calls = Trace.calls(events)
        stored = Trace.results(events)
        id_map: dict[str, str] = {}
        fresh: list[dict[str, Any]] = []
        for call, recorded in zip(calls, stored, strict=True):
            args = _remap(call.args, id_map)
            stored_result = (
                recorded.get("result") if recorded and recorded["kind"] == "result" else None
            )
            if self.replay_results:
                result = tools.call(call.name, args, stored_result=stored_result)
            else:
                result = tools.call(call.name, args)
                if isinstance(stored_result, dict):
                    for key in _ID_KEYS:
                        old, new = stored_result.get(key), result.get(key)
                        if isinstance(old, str) and isinstance(new, str) and old and new:
                            id_map[old] = new
            fresh.append(result)
        final = next(
            (
                e
                for e in reversed(events)
                if e["kind"] == "message" and e.get("role") == "assistant"
            ),
            None,
        )
        text = str(final.get("text", "")) if final else ""
        answer, citations, body = parse_answer(text)
        if not self.replay_results:
            answer, citations = self._refresh(answer, citations, id_map, fresh)
        tokens_in, tokens_out = Trace.usage_totals(events)
        tools.trace.message(
            "assistant", format_answer(answer, citations, body), replayed_from=str(path)
        )
        return BackendResult(
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd(model_id, tokens_in, tokens_out),
            model_id=str(model_id),
            citations=citations,
            text=body,
            stop_reason="end_turn",
            error=None if answer else "recorded trace holds no final answer",
        )

    @staticmethod
    def _refresh(
        answer: dict[str, Any],
        citations: list[dict[str, Any]],
        id_map: Mapping[str, str],
        fresh: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        by_run = {r["run_id"]: r for r in fresh if isinstance(r.get("run_id"), str)}
        new_citations: list[dict[str, Any]] = []
        new_answer = dict(answer)
        for cite in citations:
            run_id = id_map.get(cite["run_id"], cite["run_id"])
            value = cite["value"]
            result = by_run.get(run_id)
            if (
                result is not None
                and cite["key"] in result
                and isinstance(result[cite["key"]], (int, float))
            ):
                value = result[cite["key"]]
                if cite["key"] in new_answer:
                    new_answer[cite["key"]] = value
            new_citations.append({"key": cite["key"], "value": value, "run_id": run_id})
        return new_answer, new_citations


# --- scripted ------------------------------------------------------------------------------------

KINDS: tuple[str, ...] = (
    "phonons_imaginary",
    "relax_a0",
    "softening_rank",
    "force_mae",
    "select_frames",
    "dft_plan",
    "md_drift",
    "relax_report",
    "inventory_relax_report",
)
KNOWN_COMPOUNDS: tuple[str, ...] = ("FeSi", "CoSi", "MnSi", "FeGe", "MnGe", "CoGe")
_COMPOUND_RE = re.compile(r"\b(FeSi|CoSi|MnSi|FeGe|MnGe|CoGe)\b")
# kind -> alternative needle groups; a prompt matches a kind when every word of one group occurs
_KIND_RULES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("inventory_relax_report", (("write", "report", "list"),)),
    ("relax_report", (("write", "report"),)),
    ("softening_rank", (("softening",),)),
    ("phonons_imaginary", (("imaginary",),)),
    ("force_mae", (("mae",), ("force error",))),
    ("select_frames", (("select",), ("uncertainty",))),
    ("dft_plan", (("node-hour",), ("node hour",), ("qe round",), ("quantum espresso",))),
    ("md_drift", ((" md ",), ("molecular dynamics",), ("drift",))),
    ("relax_a0", (("relax",),)),
)


@dataclass
class TaskPlan:
    """Which canonical tool sequence a task follows and with which parameters."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    tool: str
    args: Callable[[dict[str, Any]], dict[str, Any]]
    key: str
    required: bool = True


def infer_kind(prompt: str) -> str:
    """Keyword map from a prompt to a task kind (the ``--prompt`` path; JSONL tasks name theirs)."""
    text = f" {prompt.lower()} "
    for kind, groups in _KIND_RULES:
        if any(all(word in text for word in group) for group in groups):
            return kind
    raise ValueError(f"cannot infer a task kind from the prompt {prompt!r}; give kind= explicitly")


def parse_params(prompt: str) -> dict[str, Any]:
    """Compounds, counts, temperatures, lengths, tiers, model/dataset names from a prompt."""
    params: dict[str, Any] = {}
    compounds = list(dict.fromkeys(_COMPOUND_RE.findall(prompt)))
    if compounds:
        params["compounds"] = compounds
        params["compound"] = compounds[0]
    m = re.search(r"\b(nve|nvt|npt)\b", prompt, re.I)
    if m:
        params["ensemble"] = m.group(1).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*ps\b", prompt, re.I)
    if m:
        params["ps"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*K\b", prompt)
    if m:
        params["T"] = float(m.group(1))
    m = re.search(r"(\d+)\s*atoms?\b", prompt, re.I)
    if m:
        params["natoms"] = int(m.group(1))
    m = re.search(r"\b(T[0-4][ab]?)\b", prompt)
    if m:
        params["tier"] = m.group(1)
    m = re.search(r"\bmodel\s+([A-Za-z0-9][A-Za-z0-9_.-]*)", prompt)
    if m and m.group(1).lower() not in ("and", "the", "with", "to", "for", "of"):
        params["model"] = m.group(1)
    m = re.search(r"\bdataset\s+([A-Za-z0-9][A-Za-z0-9_.-]*)", prompt)
    if m:
        params["frames"] = m.group(1)
    m = re.search(r"\b(\d+)\s+(?:frames|highest|most)", prompt, re.I)
    if m:
        params["n"] = int(m.group(1))
    m = re.search(r"\b(qe|phonondb103|pbesol)\b", prompt, re.I)
    if m:
        params["reference"] = m.group(1).lower()
    return params


def _ok(result: Mapping[str, Any] | None) -> bool:
    return bool(result) and "error" not in result  # type: ignore[operator]


def build_plan(kind: str, params: Mapping[str, Any], tc: Any) -> list[Step]:
    """The canonical DAG-respecting tool sequence of a task kind."""
    p = dict(params)
    cfg = tc.cfg
    supercell = [int(x) for x in p.get("supercell") or []]
    distance = float(p.get("distance") or 0.0)
    fmax = float(p.get("fmax") or 0.0)
    steps = int(p.get("steps") or 0)
    compound = str(p.get("compound") or (p.get("compounds") or [""])[0])

    def structure(c: str) -> Step:
        return Step("get_structure", lambda s: {"compound": c}, f"structure:{c}")

    def frame_id(s: dict[str, Any], c: str) -> str:
        res = s.get(f"structure:{c}") or {}
        return str(res.get("frame_id", "")) if _ok(res) else ""

    def relax_step(c: str) -> Step:
        return Step(
            "relax",
            lambda s: {"compound": c, "frame_id": frame_id(s, c), "fmax": fmax, "steps": steps},
            f"relax:{c}",
        )

    def phonons_step(c: str) -> Step:
        return Step(
            "phonons",
            lambda s: {
                "compound": c, "frame_id": frame_id(s, c), "supercell": supercell,
                "distance": distance, "asr": True,
            },
            f"phonons:{c}",
        )  # fmt: skip

    if kind == "phonons_imaginary":
        return [structure(compound), phonons_step(compound)]
    if kind == "relax_a0":
        return [structure(compound), relax_step(compound)]
    if kind == "softening_rank":
        compounds = list(p.get("compounds") or [compound])
        reference = str(p.get("reference") or "qe")
        plan: list[Step] = []
        for c in compounds:
            plan += [structure(c), phonons_step(c)]
            plan.append(
                Step(
                    "compare_phonons",
                    lambda s, c=c: {  # type: ignore[misc]
                        "run_id_model": str((s.get(f"phonons:{c}") or {}).get("run_id", "")),
                        "reference": reference,
                    },
                    f"compare:{c}",
                )
            )
        return plan
    if kind == "force_mae":
        tier = str(p.get("tier") or "T0")
        return [
            Step("list_data", lambda s: {"kind": "models"}, "models", required=False),
            Step(
                "evaluate_errors",
                lambda s: {
                    "model": str(p.get("model") or ""),
                    "frames": str(p.get("frames") or ""),
                    "split": str(p.get("split") or ""),
                    "tier": tier,
                },
                "errors",
            ),  # fmt: skip
        ]
    if kind == "select_frames":
        tier = str(p.get("tier") or "T0")
        n = int(p.get("n") or 5)

        def models_of(s: dict[str, Any]) -> list[str]:
            given = p.get("models")
            if given:
                return [str(m) for m in given]
            inventory = s.get("models") or {}
            committee = inventory.get("committee") or []
            if len(committee) >= 2:  # the configured committee, never the primary model
                return [str(m) for m in committee][:3]
            listed = inventory.get("models") or []
            return [str(m) for m in listed][:3]

        return [
            Step("list_data", lambda s: {"kind": "models"}, "models", required=False),
            Step("list_data", lambda s: {"kind": "datasets"}, "datasets", required=False),
            Step(
                "evaluate_errors",
                lambda s: {
                    "model": (models_of(s) or [""])[0],
                    "frames": str(p.get("eval_frames") or p.get("frames") or ""),
                    "split": str(p.get("split") or ""),
                    "tier": tier,
                },
                "errors",
            ),  # fmt: skip
            Step(
                "select_frames",
                lambda s: {"models": models_of(s), "frames": str(p.get("frames") or ""), "n": n},
                "select",
            ),
        ]
    if kind == "dft_plan":
        return [
            Step("list_data", lambda s: {"kind": "datasets"}, "datasets", required=False),
            Step(
                "submit_dft",
                lambda s: {
                    "frames": str(p.get("frames") or ""),
                    "root": str(p.get("root") or "agent_plan"),
                    "n_frames": int(p.get("n_frames") or p.get("n") or 0),
                    "submit": bool(p.get("submit", False)),
                },
                "dft",
            ),  # fmt: skip
        ]
    if kind == "md_drift":
        return [
            structure(compound),
            Step(
                "run_md",
                lambda s: {
                    "compound": compound,
                    "frame_id": frame_id(s, compound),
                    "ensemble": str(p.get("ensemble") or "nve"),
                    "T": float(p.get("T") or 300.0),
                    "ps": float(p.get("ps") or 1.0),
                    "natoms": int(p.get("natoms") or cfg.md.natoms),
                },
                "md",
            ),  # fmt: skip
        ]
    if kind in ("relax_report", "inventory_relax_report"):
        title = str(p.get("title") or f"{compound} relaxation with the configured MLIP")

        def report_args(s: dict[str, Any]) -> dict[str, Any]:
            res = s.get(f"relax:{compound}") or {}
            numbers = [
                {"key": k, "value": res[k], "run_id": res["run_id"]}
                for k in ("a_A", "energy_eV")
                if _ok(res) and isinstance(res.get(k), (int, float))
            ]
            text = (
                f"{compound}: relaxed lattice parameter a = {res.get('a_A', float('nan')):.4f} A, "
                f"energy {res.get('energy_eV', float('nan')):.4f} eV "
                f"(run {res.get('run_id', '?')})."
            )
            return {"title": title, "numbers": numbers, "text": text}

        plan = [
            structure(compound),
            relax_step(compound),
            Step("write_report", report_args, "report", required=False),
        ]
        if kind == "inventory_relax_report":
            plan.insert(
                0, Step("list_data", lambda s: {"kind": "models"}, "models", required=False)
            )
        return plan
    raise ValueError(f"unknown task kind {kind!r}; expected one of {KINDS}")


def compose_answer(
    kind: str, params: Mapping[str, Any], state: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """``(answer, citations, text)`` from the executed plan's results."""
    answer: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    notes: list[str] = []

    def cite(result: Mapping[str, Any] | None, key: str, as_key: str | None = None) -> None:
        if not _ok(result) or result is None or not result.get("run_id"):
            return
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        name = as_key or key
        answer[name] = value
        citations.append({"key": name, "value": value, "run_id": str(result["run_id"])})

    compound = str(params.get("compound") or (params.get("compounds") or [""])[0])
    if kind == "phonons_imaginary":
        res = state.get(f"phonons:{compound}")
        cite(res, "imaginary_count")
        cite(res, "omega_max_meV")
        notes.append(f"{compound}: imaginary_count from phonons run {(res or {}).get('run_id')}")
    elif kind in ("relax_a0", "relax_report", "inventory_relax_report"):
        res = state.get(f"relax:{compound}")
        cite(res, "a_A")
        cite(res, "energy_eV")
        cite(res, "steps")
        if kind == "inventory_relax_report":
            cite(state.get("models"), "n_models")
        if kind != "relax_a0":
            cite(state.get("report"), "n_numbers", "report_numbers")
        notes.append(f"{compound}: a_A from relax run {(res or {}).get('run_id')}")
    elif kind == "softening_rank":
        best: tuple[str, float] | None = None
        for c in params.get("compounds") or [compound]:
            res = state.get(f"compare:{c}")
            if _ok(res) and isinstance((res or {}).get("softening_index"), (int, float)):
                s = float(res["softening_index"])  # type: ignore[index]
                cite(res, "softening_index", f"softening_index_{c}")
                if best is None or s > best[1]:
                    best = (c, s)
        if best is not None:
            answer["compound"] = best[0]
            cite(state.get(f"compare:{best[0]}"), "softening_index")
        notes.append("largest softening index = " + (best[0] if best else "undetermined"))
    elif kind == "force_mae":
        res = state.get("errors")
        for key in ("mae_f", "rmse_f", "n_frames"):
            cite(res, key)
    elif kind == "select_frames":
        cite(state.get("select"), "n_selected")
        cite(state.get("select"), "sigma_f_max")
        cite(state.get("errors"), "mae_f")
    elif kind == "dft_plan":
        cite(state.get("dft"), "n_units")
        cite(state.get("dft"), "node_hours_estimate")
        cite(state.get("dft"), "submitted")
    elif kind == "md_drift":
        res = state.get("md")
        cite(res, "T_mean_K")
        cite(res, "drift_meV_atom_ps")
        cite(res, "E_pot_mean_eV_atom")
    errors = [
        f"{k}: {v.get('error')}" for k, v in state.items() if isinstance(v, dict) and "error" in v
    ]
    text = "; ".join(notes + errors) if (notes or errors) else "no results"
    return answer, citations, text


class ScriptedBackend:
    """Deterministic DAG planner (``b20mlip screen``): canonical tool sequence per task kind,
    gold answer composed from the tool results; recovers from tool errors (retry once),
    disallowed structures (re-fetch), sum-rule flags (re-run with asr) and budget exhaustion
    (stop and answer with what it has)."""

    name = "scripted"

    def __init__(
        self,
        plans: Mapping[str, TaskPlan] | None = None,
        *,
        allowlist: Iterable[str] = DEFAULT_ALLOWLIST,
        max_retries: int = 1,
    ) -> None:
        self.plans: dict[str, TaskPlan] = dict(plans or {})
        self.allowlist = frozenset(allowlist)
        self.max_retries = int(max_retries)

    def plan_for(self, task: AgentTask) -> TaskPlan:
        plan = self.plans.get(task.task_id)
        parsed = parse_params(task.prompt)
        if plan is None:
            return TaskPlan(infer_kind(task.prompt), parsed)
        return TaskPlan(plan.kind, {**parsed, **plan.params})

    def run(self, task: AgentTask, tools: ToolDispatcher) -> BackendResult:
        plan = self.plan_for(task)
        steps = build_plan(plan.kind, plan.params, tools.context)
        state: dict[str, Any] = {}
        calls: list[dict[str, Any]] = []
        exhausted = False
        for step in steps:
            if exhausted and not step.required:
                continue
            args = step.args(state)
            result = self._execute(tools, step, args, state, calls)
            state[step.key] = result
            if isinstance(result, dict) and "error" in result and _budget_gone(result["error"]):
                exhausted = True
        answer, citations, text = compose_answer(plan.kind, plan.params, state)
        tools.trace.message(
            "assistant", format_answer(answer, citations, text), kind_of_task=plan.kind
        )
        return BackendResult(
            answer=answer,
            tool_calls=calls,
            model_id="scripted",
            citations=citations,
            text=text,
            stop_reason="end_turn",
            error=None if answer else "the plan produced no answer",
        )

    def _execute(
        self,
        tools: ToolDispatcher,
        step: Step,
        args: dict[str, Any],
        state: dict[str, Any],
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """One plan step with bounded recovery (see the class docstring)."""

        def call(tool: str, a: dict[str, Any]) -> dict[str, Any]:
            r = tools.call(tool, a)
            calls.append({"tool": tool, "args": a})
            return r

        result = call(step.tool, args)
        for _ in range(self.max_retries + 1):
            if step.tool == "get_structure" and _ok(result):
                if set(result.get("elements") or []) - self.allowlist:  # bad structure: re-fetch
                    result = call(step.tool, args)
                    continue
                break
            if step.tool == "phonons" and _ok(result) and result.get("asr_violation"):
                args = {**args, "asr": True}  # sum-rule flag: re-run with the ASR enforced
                result = call(step.tool, args)
                continue
            if _ok(result):
                break
            error = str(result.get("error", ""))
            if _budget_gone(error) or not _retryable(error):
                break
            if "(allowlist)" in error and args.get("compound"):  # a blocked frame: fetch again
                fresh = call("get_structure", {"compound": args["compound"]})
                state[f"structure:{args['compound']}"] = fresh
                args = {**args, "frame_id": str(fresh.get("frame_id", "")) if _ok(fresh) else ""}
            result = call(step.tool, args)  # transient tool error: retry once
        return result


def _budget_gone(error: str) -> bool:
    return "(max_calls)" in error or "(max_wall_s)" in error


_PERMANENT_ERRORS: tuple[str, ...] = (
    "KeyError",
    "FileNotFoundError",
    "ValueError",
    "PermissionError",
    "TypeError",
    "guard violation (unknown_tool)",
    "guard violation (max_atoms)",
    "guard violation (max_md_ps)",
    "guard violation (max_dft_frames)",
    "guard violation (max_node_hours)",
    "guard violation (approve_cluster)",
)


def _retryable(error: str) -> bool:
    """Transient failures (runtime errors, injected outages) are retried; argument, lookup and
    budget errors are not (the same call would fail the same way)."""
    return not error.startswith(_PERMANENT_ERRORS)


def make_backend(
    name: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    trace_path: str | Path | None = None,
    replay_results: bool = False,
    plans: Mapping[str, TaskPlan] | None = None,
    client: Any | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AnthropicBackend | MockBackend | ScriptedBackend:
    """Backend by name; raises :class:`BackendUnavailable` (never builds a client without a key)."""
    if name == "anthropic":
        return AnthropicBackend(model_id, client=client, max_tokens=max_tokens)
    if name == "mock":
        if trace_path is None:
            raise BackendUnavailable(
                "the mock backend needs a recorded trace (--trace or --trace-dir)"
            )
        return MockBackend(trace_path, replay_results=replay_results)
    if name == "scripted":
        return ScriptedBackend(plans)
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKEND_NAMES}")


__all__ = [
    "API_KEY_ENV",
    "BACKEND_NAMES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_ID",
    "FALLBACK_BETA",
    "KINDS",
    "KNOWN_COMPOUNDS",
    "PRICES_USD_PER_MTOK",
    "PROMPT",
    "AnthropicBackend",
    "BackendResult",
    "BackendUnavailable",
    "MockBackend",
    "ScriptedBackend",
    "Step",
    "TaskPlan",
    "ToolDispatcher",
    "build_plan",
    "compose_answer",
    "cost_usd",
    "format_answer",
    "infer_kind",
    "make_backend",
    "parse_answer",
    "parse_params",
    "system_prompt",
]
