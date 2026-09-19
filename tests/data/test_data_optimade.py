"""Keyless OPTIMADE search against a mocked httpx transport (canned JSON, pagination, errors)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs

import httpx
import numpy as np
import pytest
from helpers_data import b20_cell, mock_client, optimade_entry

from b20mlip.data import optimade

MP = "https://optimade.materialsproject.org/v1"
ALEX = "https://alexandria.icams.rub.de/pbe/v1"


def test_build_filter() -> None:
    assert optimade.build_filter(["Ge", "Fe"], 2, 198, "mp") == (
        'elements HAS ALL "Fe","Ge" AND nelements=2 AND _mp_space_group_number=198'
    )
    assert optimade.build_filter(["Fe", "Ge"], 2, 198, "alexandria").endswith(
        "_alexandria_space_group=198"
    )
    assert optimade.build_filter(["Fe", "Ge"], None, None, "mp") == 'elements HAS ALL "Fe","Ge"'
    assert optimade.build_filter(["Fe", "Ge"], 2, 198, "other") == (
        'elements HAS ALL "Fe","Ge" AND nelements=2'
    )
    assert "_mp" not in optimade.build_filter(["Fe"], 1, 198, "mp", use_spacegroup_field=False)


def test_entry_to_atoms_variants() -> None:
    cell = b20_cell("FeGe")
    atoms = optimade.entry_to_atoms(optimade_entry("1", cell))
    assert atoms is not None and atoms.get_chemical_formula() == "Fe4Ge4"
    assert np.allclose(atoms.cell[:], cell.cell[:]) and all(atoms.pbc)
    assert optimade.entry_to_atoms(optimade_entry("2", cell, disordered=True)) is None
    entry = optimade_entry("3", cell)
    del entry["attributes"]["lattice_vectors"]
    assert optimade.entry_to_atoms(entry) is None
    entry = optimade_entry("4", cell)
    entry["attributes"]["species"][0]["chemical_symbols"] = ["X"]
    assert optimade.entry_to_atoms(entry) is None
    entry = optimade_entry("5", cell)
    entry["attributes"]["species_at_sites"][0] = "Unknown"
    assert optimade.entry_to_atoms(entry) is None
    assert optimade.entry_to_frame(optimade_entry("6", cell, disordered=True), "mp") is None


def _payloads() -> dict[str, Any]:
    good = b20_cell("FeGe")
    sheared = b20_cell("FeGe")
    F = np.eye(3)
    F[0, 1] = 0.08
    sheared.set_cell(sheared.cell[:] @ F.T, scale_atoms=True)
    bigger = b20_cell("FeGe", 1.02)
    return {
        "mp1": [
            optimade_entry("mp-21255", good, {"_mp_stability": 0.0, "_mp_nested": {"a": 1}}),
            optimade_entry("mp-22510", sheared),
        ],
        "mp2": [optimade_entry("mp-999", bigger)],
        "alex": [
            optimade_entry("agm1", b20_cell("MnGe"), {"_alexandria_hull_distance": 0.01}),
            optimade_entry("agm2", b20_cell("MnGe"), disordered=True),
        ],
    }


def _handler(payloads: dict[str, Any], *, reject_field: bool = False, alex_down: bool = False):  # type: ignore[no-untyped-def]
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        params = parse_qs(request.url.query.decode())
        filt = params.get("filter", [""])[0]
        if request.url.host == "optimade.materialsproject.org":
            if reject_field and "_mp_space_group_number" in filt:
                return httpx.Response(400, json={"errors": [{"detail": "unknown property"}]})
            if "page_offset" in params:
                return httpx.Response(200, json={"data": payloads["mp2"], "links": {"next": None}})
            nxt = f"{MP}/structures?filter=x&page_limit=2&page_offset=2"
            return httpx.Response(
                200, json={"data": payloads["mp1"], "links": {"next": {"href": nxt}}}
            )
        if request.url.host == "alexandria.icams.rub.de":
            if alex_down:
                raise httpx.ConnectError("down", request=request)
            return httpx.Response(200, json={"data": payloads["alex"], "links": {}})
        return httpx.Response(404)

    handle.calls = calls  # type: ignore[attr-defined]
    return handle


def test_search_paginates_filters_and_converts() -> None:
    handler = _handler(_payloads())
    stats: dict[str, Any] = {}
    frames = optimade.search(["Fe", "Ge"], client=mock_client(handler), stats=stats)
    assert [f.parent_id for f in frames] == ["mp:mp-21255", "mp:mp-999", "alexandria:agm1"]
    assert stats["providers"]["mp"] == {"returned": 3, "kept": 2, "skipped_disordered": 0}
    assert stats["providers"]["alexandria"] == {"returned": 2, "kept": 1, "skipped_disordered": 1}
    assert stats["errors"] == {}
    f = frames[0]
    assert f.label_source == "none" and f.energy_scale == "none" and f.config_type == "relax"
    assert f.compound == "FeGe" and f.group_id == "FeGe/relax/mp:mp-21255"
    assert f.info["optimade_provider"] == "mp" and f.info["optimade_id"] == "mp-21255"
    assert f.info["spacegroup_spglib"] == 198 and f.info["mp_stability"] == 0.0
    assert "mp_nested" not in f.info and f.info["optimade_formula"] == "FeGe"
    assert f.forces is None and f.energy is None
    first = handler.calls[0]
    q = parse_qs(first.url.query.decode())
    assert (
        q["filter"][0]
        == 'elements HAS ALL "Fe","Ge" AND nelements=2 AND _mp_space_group_number=198'
    )
    assert q["page_limit"] == ["100"]
    # without the space-group post-filter the sheared entry is kept as well
    frames = optimade.search(["Fe", "Ge"], spacegroup=None, providers=("mp",),
                             client=mock_client(_handler(_payloads())))  # fmt: skip
    assert len(frames) == 3


def test_search_retries_without_spacegroup_field() -> None:
    handler = _handler(_payloads(), reject_field=True)
    stats: dict[str, Any] = {}
    frames = optimade.search(
        ["Fe", "Ge"], providers=("mp",), client=mock_client(handler), stats=stats
    )
    assert [f.parent_id for f in frames] == ["mp:mp-21255", "mp:mp-999"]
    assert stats["retried_without_spacegroup_field"] == ["mp"] and stats["errors"] == {}
    filters = [parse_qs(c.url.query.decode()).get("filter", [""])[0] for c in handler.calls]
    assert "_mp_space_group_number" in filters[0] and "_mp_space_group_number" not in filters[1]


def test_search_records_provider_errors() -> None:
    handler = _handler(_payloads(), alex_down=True)
    stats: dict[str, Any] = {}
    frames = optimade.search(["Fe", "Ge"], client=mock_client(handler), stats=stats)
    assert [f.parent_id for f in frames] == ["mp:mp-21255", "mp:mp-999"]
    assert "ConnectError" in stats["errors"]["alexandria"]
    stats = {}
    assert (
        optimade.search(["Fe"], providers=("nope",), client=mock_client(handler), stats=stats) == []
    )
    assert stats["errors"] == {"nope": "unknown provider"}
    # a persistent HTTP error (not tied to the space-group field) is recorded, not raised
    always_500 = mock_client(lambda request: httpx.Response(500))
    stats = {}
    assert (
        optimade.search(["Fe", "Ge"], providers=("alexandria",), client=always_500, stats=stats)
        == []
    )
    assert stats["errors"]["alexandria"].startswith("HTTP 500")


def test_query_provider_limits_pages_and_custom_base() -> None:
    handler = _handler(_payloads())
    client = mock_client(handler)
    entries = optimade.query_provider(client, "mp", "x", max_pages=1)
    assert len(entries) == 2
    entries = optimade.query_provider(client, "custom", "x", base_url=f"{ALEX}/")
    assert len(entries) == 2 and handler.calls[-1].url.host == "alexandria.icams.rub.de"
    payload = json.loads(httpx.Response(200, json={"links": {"next": "u"}}).text)
    assert optimade._next_link(payload) == "u" and optimade._next_link({}) is None


@pytest.mark.network
def test_live_alexandria_mnge() -> None:  # pragma: no cover - opt-in
    frames = optimade.search(["Mn", "Ge"], providers=("alexandria",))
    assert any(f.compound == "MnGe" for f in frames)
