"""Trace: record / read round trip, the fixed field set, redaction of anything key-like."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from b20mlip.agent.trace import (
    EVENT_KINDS,
    FIELDS,
    REDACTED,
    ToolCall,
    Trace,
    canonical_json,
    redact,
    sha256_json,
)


def test_record_and_read_round_trip(tmp_path: Path) -> None:
    clock = iter(range(1, 100))
    trace = Trace(tmp_path / "t" / "trace.jsonl", clock=lambda: float(next(clock)))
    call = ToolCall("get_structure", {"compound": "FeSi"}, 0)
    trace.message("system", "run 1", backend="scripted", model_id="scripted", task_id="t")
    trace.call(call)
    trace.result(call, {"a_A": 4.48, "run_id": "r1"}, manifest_sha="m" * 64, run_id="r1")
    shell = ToolCall("shell", {}, 1)
    trace.call(shell)
    trace.violation(shell, "guard violation (unknown_tool): no shell")
    trace.usage(120, 30, model_id="claude-opus-5", turn=1)
    trace.message("assistant", '{"answer": {"a_A": 4.48}}')
    events = Trace.read(trace.path)
    assert [e["kind"] for e in events] == [
        "message", "call", "result", "call", "violation", "usage", "message"
    ]  # fmt: skip
    assert all(set(FIELDS) <= set(e) for e in events)
    assert [e["t"] for e in events] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    recorded_call = events[1]
    assert recorded_call["tool"] == "get_structure" and recorded_call["args"] == {
        "compound": "FeSi"
    }
    assert recorded_call["args_sha256"] == sha256_json({"compound": "FeSi"}) == call.args_sha256
    result = events[2]
    assert result["result"] == {"a_A": 4.48, "run_id": "r1"} and result["run_id"] == "r1"
    assert result["result_sha256"] == sha256_json({"a_A": 4.48, "run_id": "r1"})
    assert result["manifest_sha256"] == "m" * 64
    assert events[4]["message"].startswith("guard violation") and events[4]["tool"] == "shell"
    assert events[5]["tokens_in"] == 120 and events[5]["tokens_out"] == 30
    assert Trace.usage_totals(events) == (120, 30)
    calls = Trace.calls(events)
    assert [c.name for c in calls] == ["get_structure", "shell"] and calls[1].index == 1
    results = Trace.results(events)
    assert results[0] is not None and results[0]["kind"] == "result"
    assert results[1] is not None and results[1]["kind"] == "violation"
    assert trace.events() == events and trace.n_events == 7
    assert set(EVENT_KINDS) == {"call", "result", "violation", "message", "usage"}


def test_read_rejects_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "call"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        Trace.read(path)
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        Trace.read(path)
    path.write_text('{"kind": "party", "t": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a trace event"):
        Trace.read(path)
    with pytest.raises(ValueError, match="unknown trace event kind"):
        Trace(tmp_path / "x.jsonl").record(None, kind="party")


def test_redact_strips_keys_everywhere(tmp_path: Path) -> None:
    payload = {
        "api_key": "sk-ant-api03-abcdefghijklmnop",
        "nested": {"Authorization": "Bearer abc", "token": "t", "fine": "value"},
        "text": "set ANTHROPIC_API_KEY=sk-ant-1234567890abcdef then sk-abcdefghijklmnopqrstuvwxyz",
        "list": ["sk-ant-secretsecretsecret", 3],
        "n": 3,
    }
    clean = redact(payload)
    assert clean["api_key"] == REDACTED and clean["nested"]["Authorization"] == REDACTED
    assert clean["nested"]["token"] == REDACTED and clean["nested"]["fine"] == "value"
    assert "sk-" not in clean["text"] and "ANTHROPIC_API_KEY=" not in clean["text"]
    assert clean["list"] == [REDACTED, 3] and clean["n"] == 3
    trace = Trace(tmp_path / "trace.jsonl")
    trace.call(ToolCall("write_report", {"title": "t", "api_key": "sk-ant-zzzzzzzzzzzz"}))
    trace.message("assistant", "my key is sk-ant-api03-0123456789abcdef", note={"secret": "x"})
    raw = trace.path.read_text(encoding="utf-8")
    assert "sk-ant" not in raw and "zzzz" not in raw and REDACTED in raw
    assert json.loads(raw.splitlines()[1])["note"] == {"secret": REDACTED}
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
