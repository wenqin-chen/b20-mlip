"""Keyless OPTIMADE structure search (Materials Project and Alexandria providers).

``search`` issues ``GET <base>/v1/structures?filter=...`` with the standard OPTIMADE filter
``elements HAS ALL "Fe","Ge" AND nelements=2`` plus a provider-specific space-group clause
(``_mp_space_group_number`` / ``_alexandria_space_group``) when the provider supports it.
If a provider rejects that clause (HTTP 4xx) the query is retried without it; the space
group is always re-checked with spglib on the returned structures, so the result never
depends on the provider field. Disordered entries (species with more than one symbol or a
concentration below 1) are skipped and counted.

Frames: ``label_source="none"``, ``energy_scale="none"``, ``config_type="relax"``,
``parent_id="<provider>:<id>"``, ``group_id="<compound>/relax/<provider>:<id>"``, ``info`` with
``optimade_provider``, ``optimade_id``, ``optimade_formula``, ``spacegroup_spglib`` and any
scalar provider-prefixed attribute (``_mp_*``, ``_alexandria_*``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx
import numpy as np
from ase import Atoms

from b20mlip.data._common import (
    DEFAULT_SYMPREC,
    clean_info,
    compound_name,
    normalize_elements,
    spacegroup_number,
)
from b20mlip.io import frame_from_atoms
from b20mlip.models import Frame

PROVIDERS: dict[str, str] = {
    "mp": "https://optimade.materialsproject.org/v1",
    "alexandria": "https://alexandria.icams.rub.de/pbe/v1",
}
SPACEGROUP_FIELDS: dict[str, str] = {
    "mp": "_mp_space_group_number",
    "alexandria": "_alexandria_space_group",
}
DEFAULT_PAGE_LIMIT = 100
DEFAULT_MAX_PAGES = 20
USER_AGENT = "b20-mlip/0.1 (+https://github.com/wenqin-chen/b20-mlip)"


def build_filter(
    elements: Iterable[str | int],
    nelements: int | None = 2,
    spacegroup: int | None = 198,
    provider: str | None = None,
    *,
    use_spacegroup_field: bool = True,
) -> str:
    """OPTIMADE filter string; the space-group clause only for providers that expose one."""
    els = ",".join(f'"{e}"' for e in sorted(normalize_elements(elements)))
    parts = [f"elements HAS ALL {els}"]
    if nelements is not None:
        parts.append(f"nelements={int(nelements)}")
    field = SPACEGROUP_FIELDS.get(provider or "")
    if spacegroup is not None and field and use_spacegroup_field:
        parts.append(f"{field}={int(spacegroup)}")
    return " AND ".join(parts)


def entry_to_atoms(entry: dict[str, Any]) -> Atoms | None:
    """One OPTIMADE ``structures`` entry -> ``Atoms``; ``None`` for disordered/incomplete."""
    attrs = entry.get("attributes") or {}
    lattice = attrs.get("lattice_vectors")
    positions = attrs.get("cartesian_site_positions")
    species_at_sites = attrs.get("species_at_sites")
    species = attrs.get("species") or []
    if lattice is None or positions is None or species_at_sites is None:
        return None
    symbol_of: dict[str, str] = {}
    for sp in species:
        symbols = sp.get("chemical_symbols") or []
        conc = sp.get("concentration") or [1.0] * len(symbols)
        if len(symbols) != 1 or (conc and abs(float(conc[0]) - 1.0) > 1e-6):
            return None  # disordered / vacancy-containing site
        if symbols[0] in ("X", "vacancy"):
            return None
        symbol_of[str(sp.get("name"))] = str(symbols[0])
    try:
        symbols_list = [symbol_of[str(name)] for name in species_at_sites]
    except KeyError:
        return None
    pbc = attrs.get("dimension_types") or [1, 1, 1]
    atoms = Atoms(
        symbols=symbols_list,
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(lattice, dtype=float),
        pbc=[bool(int(b)) for b in pbc],
    )
    return atoms


def entry_to_frame(
    entry: dict[str, Any], provider: str, *, symprec: float = DEFAULT_SYMPREC
) -> Frame | None:
    atoms = entry_to_atoms(entry)
    if atoms is None:
        return None
    attrs = entry.get("attributes") or {}
    entry_id = str(entry.get("id"))
    compound = compound_name(atoms)
    lineage = f"{provider}:{entry_id}"
    info: dict[str, Any] = {
        "optimade_provider": provider,
        "optimade_id": entry_id,
        "optimade_formula": attrs.get("chemical_formula_reduced"),
        "spacegroup_spglib": spacegroup_number(atoms, symprec),
    }
    for key, value in attrs.items():
        if key.startswith("_") and isinstance(value, (int, float, str, bool)):
            info[key.lstrip("_")] = value
    frame = frame_from_atoms(
        atoms,
        group_id=f"{compound}/relax/{lineage}",
        compound=compound,
        config_type="relax",
        parent_id=lineage,
    )
    return frame.model_copy(update={"info": clean_info(info)})


def _next_link(payload: dict[str, Any]) -> str | None:
    nxt = (payload.get("links") or {}).get("next")
    if isinstance(nxt, dict):
        nxt = nxt.get("href")
    return str(nxt) if nxt else None


def query_provider(
    client: httpx.Client,
    provider: str,
    filter_: str,
    *,
    base_url: str | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> list[dict[str, Any]]:
    """All ``data`` entries of a paginated ``/structures`` query."""
    base = (base_url or PROVIDERS[provider]).rstrip("/")
    url: str | None = f"{base}/structures"
    params: dict[str, Any] | None = {"filter": filter_, "page_limit": page_limit}
    entries: list[dict[str, Any]] = []
    for _ in range(max_pages):
        if url is None:
            break
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        entries.extend(payload.get("data") or [])
        url = _next_link(payload)
        params = None  # the next link carries its own query string
    return entries


def search(
    elements: Iterable[str | int],
    *,
    nelements: int | None = 2,
    spacegroup: int | None = 198,
    providers: Iterable[str] = ("mp", "alexandria"),
    client: httpx.Client | None = None,
    base_urls: dict[str, str] | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    symprec: float = DEFAULT_SYMPREC,
    stats: dict[str, Any] | None = None,
) -> list[Frame]:
    """Keyless structure search over ``providers``; frames post-filtered with spglib.

    A provider that fails (network error, HTTP error on both query variants) is recorded in
    ``stats["errors"]`` and skipped; ``stats`` also gets per-provider ``returned``, ``kept``
    and ``skipped_disordered`` counts.
    """
    owned = client is None
    http = client or httpx.Client(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    frames: list[Frame] = []
    info: dict[str, Any] = {"errors": {}, "providers": {}}
    try:
        for provider in providers:
            base = (base_urls or {}).get(provider)
            if base is None and provider not in PROVIDERS:
                info["errors"][provider] = "unknown provider"
                continue
            entries: list[dict[str, Any]] | None = None
            for use_field in (True, False):
                filt = build_filter(
                    elements, nelements, spacegroup, provider, use_spacegroup_field=use_field
                )
                try:
                    entries = query_provider(
                        http, provider, filt, base_url=base,
                        page_limit=page_limit, max_pages=max_pages,
                    )  # fmt: skip
                    break
                except httpx.HTTPStatusError as exc:
                    info["errors"][provider] = f"HTTP {exc.response.status_code} for {filt!r}"
                    if not use_field or SPACEGROUP_FIELDS.get(provider) is None:
                        break
                except httpx.HTTPError as exc:
                    info["errors"][provider] = repr(exc)
                    break
            if entries is None:
                continue
            kept = 0
            skipped = 0
            for entry in entries:
                frame = entry_to_frame(entry, provider, symprec=symprec)
                if frame is None:
                    skipped += 1
                    continue
                if spacegroup is not None and frame.info.get("spacegroup_spglib") != spacegroup:
                    continue
                frames.append(frame)
                kept += 1
            info["providers"][provider] = {
                "returned": len(entries),
                "kept": kept,
                "skipped_disordered": skipped,
            }
            if provider in info["errors"]:  # the retry without the space-group field worked
                info["retried_without_spacegroup_field"] = info.get(
                    "retried_without_spacegroup_field", []
                ) + [provider]
                info["errors"].pop(provider)
    finally:
        if owned:
            http.close()
    if stats is not None:
        stats.update(info)
    return frames


__all__ = [
    "PROVIDERS",
    "SPACEGROUP_FIELDS",
    "build_filter",
    "entry_to_atoms",
    "entry_to_frame",
    "query_provider",
    "search",
]
