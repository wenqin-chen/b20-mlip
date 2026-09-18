"""The whole session is offline and never imports matbench_discovery."""

from __future__ import annotations

import importlib
import importlib.util
import socket
import sys

import pytest


def test_sockets_are_blocked() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("example.com", 80), timeout=0.1)
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.socket().connect(("127.0.0.1", 9))


def test_matbench_discovery_absent() -> None:
    for name in ("models", "config", "provenance", "io", "executors", "cli"):
        importlib.import_module(f"b20mlip.{name}")
    assert "matbench_discovery" not in sys.modules
    assert importlib.util.find_spec("matbench_discovery") is None, (
        "matbench_discovery must never be installed (CONTRACTS.md header)"
    )
