"""Agent trace: one JSON line per event (CONTRACTS.md row 13; SPEC.md section 7 "every call
logged with argument and manifest hashes").

Format (binding; the mock backend replays it and ``tests/fixtures/traces/*.jsonl`` are written
in exactly this shape)::

    {"t": <unix seconds>, "kind": "call"|"result"|"violation"|"message"|"usage",
     "tool": <tool name or null>, "args_sha256": <sha256 of the canonical args JSON or null>,
     "args": <redacted args or null>, "result_sha256": <sha256 of the canonical result JSON or
     null>, "manifest_sha256": <sha256 of the tool sub-run's manifest.json or null>,
     "run_id": <sub-run id or null>, "tokens_in": <int or null>, "tokens_out": <int or null>,
     ...}

The ten fields above are present on every line; kinds add their own: ``result`` carries
``result`` (the redacted tool result, what ``MockBackend(replay_results=True)`` returns),
``violation`` carries ``message`` (the guard's reason), ``message`` carries ``role`` and ``text``
(``system`` = the run header with ``backend``, ``model_id``, ``task_id``, ``budget``;
``assistant`` = the final answer), ``usage`` carries ``model_id`` and ``turn``.

``redact`` strips anything that looks like a key before it is written: values of keys named
like ``api_key``/``token``/``secret``, ``sk-ant-...``/``sk-...`` strings and ``ANTHROPIC_*=...``
assignments. Hashes are computed over the *redacted* payloads, so a trace can be verified
without ever holding the secret.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVENT_KINDS: tuple[str, ...] = ("call", "result", "violation", "message", "usage")
FIELDS: tuple[str, ...] = (
    "t",
    "kind",
    "tool",
    "args_sha256",
    "args",
    "result_sha256",
    "manifest_sha256",
    "run_id",
    "tokens_in",
    "tokens_out",
)
REDACTED = "[REDACTED]"
SECRET_KEY_NAMES: tuple[str, ...] = (
    "api_key",
    "apikey",
    "api-key",
    "auth_token",
    "authorization",
    "password",
    "secret",
    "token",
)
_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"ANTHROPIC_[A-Z0-9_]*(?:KEY|TOKEN|SECRET)[A-Z0-9_]*\s*[=:]\s*\S+"),
)


# --- hashing and redaction ---------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Sorted-keys, compact JSON (the payload every hash is computed over)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def redact(obj: Any) -> Any:
    """Recursively strip secrets: key-named fields, ``sk-...`` tokens, ``ANTHROPIC_*=`` values."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            name = str(key).lower()
            if any(marker in name for marker in SECRET_KEY_NAMES):
                out[str(key)] = REDACTED
            else:
                out[str(key)] = redact(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        text = obj
        for pattern in _KEY_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text
    return obj


# --- events ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation as the guard and the trace see it."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    @property
    def args_sha256(self) -> str:
        return sha256_json(redact(self.args))


class Trace:
    """Append-only JSONL trace of one agent run (see the module docstring for the format)."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.n_events = 0

    # -- writing -----------------------------------------------------------------------------

    def record(
        self,
        call: ToolCall | None,
        result_hash: str | None = None,
        manifest_sha: str | None = None,
        *,
        kind: str = "call",
        run_id: str | None = None,
        result: Any | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append one event and return it (CONTRACTS: ``record(call, result_hash, manifest_sha)``).

        ``kind`` selects the event type; ``extra`` keys (``role``, ``text``, ``message``,
        ``model_id``, ``turn``, ...) are stored redacted next to the fixed fields.
        """
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown trace event kind {kind!r}; expected one of {EVENT_KINDS}")
        event: dict[str, Any] = {
            "t": float(self.clock()),
            "kind": kind,
            "tool": call.name if call is not None else None,
            "args_sha256": call.args_sha256 if call is not None else None,
            "args": redact(call.args) if call is not None else None,
            "result_sha256": result_hash,
            "manifest_sha256": manifest_sha,
            "run_id": run_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        if result is not None:
            event["result"] = redact(result)
        for key, value in extra.items():
            event[key] = redact(value)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(event) + "\n")
        self.n_events += 1
        return event

    def call(self, call: ToolCall) -> dict[str, Any]:
        return self.record(call, kind="call")

    def result(
        self,
        call: ToolCall,
        result: Any,
        *,
        manifest_sha: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        clean = redact(result)
        return self.record(
            call, sha256_json(clean), manifest_sha, kind="result", run_id=run_id, result=clean
        )

    def violation(self, call: ToolCall, message: str) -> dict[str, Any]:
        return self.record(call, kind="violation", message=message)

    def message(self, role: str, text: str, **extra: Any) -> dict[str, Any]:
        return self.record(None, kind="message", role=role, text=text, **extra)

    def usage(
        self,
        tokens_in: int,
        tokens_out: int,
        *,
        model_id: str | None = None,
        turn: int | None = None,
    ) -> dict[str, Any]:
        return self.record(
            None,
            kind="usage",
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            model_id=model_id,
            turn=turn,
        )

    # -- reading -----------------------------------------------------------------------------

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        """Every event of a trace file, in order (blank lines skipped)."""
        p = Path(path)
        events: list[dict[str, Any]] = []
        for number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{number}: not a JSON object ({exc})") from exc
            if not isinstance(event, dict) or event.get("kind") not in EVENT_KINDS:
                raise ValueError(f"{p}:{number}: not a trace event")
            missing = [f for f in FIELDS if f not in event]
            if missing:
                raise ValueError(f"{p}:{number}: trace event lacks {missing}")
            events.append(event)
        return events

    def events(self) -> list[dict[str, Any]]:
        return self.read(self.path) if self.path.is_file() else []

    @staticmethod
    def calls(events: Iterable[dict[str, Any]]) -> list[ToolCall]:
        """The ``call`` events as :class:`ToolCall` objects (index = position among calls)."""
        out: list[ToolCall] = []
        for event in events:
            if event.get("kind") == "call":
                out.append(ToolCall(str(event["tool"]), dict(event.get("args") or {}), len(out)))
        return out

    @staticmethod
    def results(events: Sequence[dict[str, Any]]) -> list[dict[str, Any] | None]:
        """One entry per call: the ``result``/``violation`` event that followed it (or ``None``)."""
        out: list[dict[str, Any] | None] = []
        pending = False
        for event in events:
            kind = event.get("kind")
            if kind == "call":
                if pending:
                    out.append(None)
                pending = True
            elif kind in ("result", "violation") and pending:
                out.append(event)
                pending = False
        if pending:
            out.append(None)
        return out

    @staticmethod
    def usage_totals(events: Iterable[dict[str, Any]]) -> tuple[int, int]:
        tokens_in = tokens_out = 0
        for event in events:
            if event.get("kind") == "usage":
                tokens_in += int(event.get("tokens_in") or 0)
                tokens_out += int(event.get("tokens_out") or 0)
        return tokens_in, tokens_out


__all__ = [
    "EVENT_KINDS",
    "FIELDS",
    "REDACTED",
    "SECRET_KEY_NAMES",
    "ToolCall",
    "Trace",
    "canonical_json",
    "redact",
    "sha256_json",
]
