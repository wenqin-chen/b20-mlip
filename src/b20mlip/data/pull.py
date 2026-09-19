"""``data pull``: fetch (or verify) the raw sources and optionally run the local extractions.

Sources (CONTRACTS.md section 3 ``--sources mptrj,omat24,wbm,phonondb``; sizes measured in
docs/design/facts_2026-09-18.md):

* ``mptrj``    figshare files/49034296 -> ``<data_dir>/raw/mptrj/mptrj.extxyz.zip`` (MIT)
* ``wbm``      figshare files/48169597 (initial atoms) and files/64706751 (summary csv.gz)
               -> ``<data_dir>/raw/references/`` (CC-BY-4.0)
* ``phonondb`` figshare files/52179965 -> ``<data_dir>/raw/references/`` (CC-BY-4.0)
* ``omat24``   ``cfg.data.omat24_url`` (1M subsplit tarball) -> ``<data_dir>/raw/omat24/``;
               an already extracted ``omat24_1M/`` directory of ``.aselmdb`` shards counts as
               present (CC-BY-4.0)

Rules: a file that exists with the expected size (and sha256 when ``cfg.data.omat24_sha256``
or a registry sha is set) is never downloaded again, only hashed and recorded; figshare
rejects ``HEAD`` (403), so only ranged ``GET`` requests are used and a ``.part`` file is
resumed with ``Range: bytes=<offset>-``. ``sources.json`` (url, bytes, sha256, licence,
status) is written to ``ctx.out_dir``. With ``extract=True`` the MPtrj family/B20 extraction,
the OMat24 stream filter and the WBM sample are produced under ``<data_dir>/frames`` and
``<data_dir>/wbm``; ``delete_raw=True`` removes the OMat24 raw data *after* a successful
filter (never by default).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from b20mlip.config import Settings
from b20mlip.data._common import FAMILY_ELEMENTS
from b20mlip.models import StageResult
from b20mlip.provenance import RunContext, sha256_file

USER_AGENT = "b20-mlip/0.1 (+https://github.com/wenqin-chen/b20-mlip)"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=120.0)
CHUNK_SIZE = 1 << 20
PART_SUFFIX = ".part"
SOURCE_KEYS: tuple[str, ...] = ("mptrj", "omat24", "wbm", "phonondb")
Log = Callable[..., None] | None


class DownloadError(RuntimeError):
    """A download finished with the wrong size or checksum, or the server refused a range."""


@dataclass(frozen=True)
class Source:
    """One downloadable file; ``relpath`` is relative to ``cfg.paths.data_dir``."""

    key: str  # CLI group: mptrj | omat24 | wbm | phonondb
    name: str  # unique file name in sources.json
    url: str
    relpath: str
    bytes: int | None
    licence: str
    sha256: str | None = None
    extracted_dir: str | None = None  # relative dir that counts as "present" (omat24)
    note: str = ""

    def path(self, data_dir: str | Path) -> Path:
        return Path(data_dir) / self.relpath


SOURCES: dict[str, list[Source]] = {
    "mptrj": [
        Source(
            key="mptrj",
            name="mptrj_extxyz_zip",
            url="https://ndownloader.figshare.com/files/49034296",
            relpath="raw/mptrj/mptrj.extxyz.zip",
            bytes=1_521_713_089,
            licence="MIT",
            note="2024-09-03-mp-trj.extxyz.zip (figshare 23713842), 145,923 per-material members",
        )
    ],
    "wbm": [
        Source(
            key="wbm",
            name="wbm_initial_atoms_zip",
            url="https://ndownloader.figshare.com/files/48169597",
            relpath="raw/references/wbm-initial-atoms.extxyz.zip",
            bytes=98_164_028,
            licence="CC-BY-4.0",
            note="2024-08-04-wbm-initial-atoms.extxyz.zip (matbench-discovery), 256,963 members",
        ),
        Source(
            key="wbm",
            name="wbm_summary_csv_gz",
            url="https://ndownloader.figshare.com/files/64706751",
            relpath="raw/references/wbm-summary.csv.gz",
            bytes=12_745_623,
            licence="CC-BY-4.0",
            note="2023-12-13-wbm-summary.csv.gz (matbench-discovery), 256,963 rows",
        ),
    ],
    "phonondb": [
        Source(
            key="phonondb",
            name="phonondb_pbe_103_structures",
            url="https://ndownloader.figshare.com/files/52179965",
            relpath="raw/references/phononDB-PBE-103-structures.extxyz",
            bytes=76_121,
            licence="CC-BY-4.0",
            note="2024-11-09-phononDB-PBE-103-structures.extxyz (matbench-discovery)",
        )
    ],
    "omat24": [
        Source(
            key="omat24",
            name="omat24_1M_tar_gz",
            url="",  # filled from cfg.data.omat24_url
            relpath="raw/omat24/omat24_1M_251210.tar.gz",
            bytes=2_274_568_000,
            licence="CC-BY-4.0",
            extracted_dir="raw/omat24/omat24_1M",
            note="OMat24 1M subsplit (33 .aselmdb shards, 1,171,309 frames)",
        )
    ],
}


def parse_sources(spec: str | Iterable[str] | None) -> list[str]:
    """``"mptrj,wbm"`` / ``["mptrj", "wbm"]`` / ``None`` (= all) -> validated key list."""
    if spec is None:
        return list(SOURCE_KEYS)
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    keys: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key:
            continue
        if key == "all":
            return list(SOURCE_KEYS)
        if key not in SOURCES:
            raise ValueError(f"unknown source {key!r}; choose from {', '.join(SOURCE_KEYS)}")
        if key not in keys:
            keys.append(key)
    return keys


def sources_for(cfg: Settings, keys: Iterable[str]) -> list[Source]:
    """Registry entries for ``keys`` with the config-provided OMat24 url/sha applied."""
    out: list[Source] = []
    for key in keys:
        for src in SOURCES[key]:
            if key == "omat24":
                src = replace(src, url=cfg.data.omat24_url, sha256=cfg.data.omat24_sha256)
            out.append(src)
    return out


def make_client(timeout: httpx.Timeout | float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})


# --- download --------------------------------------------------------------------------------


def download(
    client: httpx.Client,
    url: str,
    dest: str | Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_attempts: int = 3,
    log: Log = None,
) -> dict[str, Any]:
    """Ranged, resumable GET into ``dest`` (via ``dest.part``); verifies size and sha256.

    A transport error mid-stream is retried up to ``max_attempts`` times, resuming from the
    bytes already on disk. Returns ``{"bytes", "sha256", "resumed_from", "attempts"}``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + PART_SUFFIX)
    first_offset = part.stat().st_size if part.exists() else 0
    attempts = 0
    last_error: Exception | None = None
    while attempts < max_attempts:
        attempts += 1
        offset = part.stat().st_size if part.exists() else 0
        try:
            _stream_to(client, url, part, offset, chunk_size, log)
        except httpx.TransportError as exc:  # connection dropped: resume on the next attempt
            last_error = exc
            if log is not None:
                log(event="retry", url=url, attempt=attempts, error=repr(exc))
            continue
        size = part.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            if size > expected_bytes or attempts >= max_attempts:
                part.unlink(missing_ok=True)
                raise DownloadError(f"{url}: downloaded {size} bytes, expected {expected_bytes}")
            last_error = DownloadError(f"short read: {size} < {expected_bytes}")
            continue
        digest = sha256_file(part)
        if expected_sha256 and digest != expected_sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(f"{url}: sha256 {digest} != expected {expected_sha256}")
        os.replace(part, dest)
        return {
            "bytes": size,
            "sha256": digest,
            "resumed_from": first_offset,
            "attempts": attempts,
        }
    raise DownloadError(f"{url}: giving up after {attempts} attempts: {last_error!r}")


def _stream_to(
    client: httpx.Client, url: str, part: Path, offset: int, chunk_size: int, log: Log
) -> None:
    headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
    with client.stream("GET", url, headers=headers) as resp:
        if resp.status_code == 416:  # range not satisfiable: the .part is already complete
            return
        if resp.status_code == 200:
            mode = "wb"  # full body (fresh download, or the server ignored the range)
        elif resp.status_code == 206:
            mode = "ab"
        else:
            raise DownloadError(f"{url}: HTTP {resp.status_code}")
        with open(part, mode) as fh:
            done = 0 if mode == "wb" else offset
            for chunk in resp.iter_bytes(chunk_size):
                fh.write(chunk)
                done += len(chunk)
        if log is not None:
            log(event="downloaded", url=url, bytes=done, resumed_from=offset if mode == "ab" else 0)


# --- verification ----------------------------------------------------------------------------


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_extracted_dir(root: Path) -> dict[str, Any]:
    """Hash every ``.aselmdb`` shard below ``root``; combined sha over ``(relpath, sha)`` pairs."""
    import hashlib

    from b20mlip.data.omat24 import find_shards

    shards = find_shards(root)
    combined = hashlib.sha256()
    total = 0
    for shard in shards:
        digest = sha256_file(shard)
        total += shard.stat().st_size
        combined.update(f"{shard.relative_to(root)}\t{digest}\n".encode())
    return {
        "path": str(root),
        "shards": len(shards),
        "bytes": total,
        "sha256": combined.hexdigest() if shards else None,
    }


def ensure_source(
    src: Source,
    data_dir: Path,
    *,
    client: httpx.Client | None,
    dry_run: bool,
    log: Log = None,
) -> dict[str, Any]:
    """Verify-or-download one source; returns its ``sources.json`` record."""
    dest = src.path(data_dir)
    record: dict[str, Any] = {
        "key": src.key,
        "url": src.url,
        "path": str(dest),
        "licence": src.licence,
        "expected_bytes": src.bytes,
        "expected_sha256": src.sha256,
        "note": src.note,
    }
    if dest.is_file():
        size = dest.stat().st_size
        if src.bytes is None or size == src.bytes:
            digest = sha256_file(dest)
            if src.sha256 and digest != src.sha256:
                record.update(status="sha256_mismatch", bytes=size, sha256=digest)
                if dry_run:
                    record["status"] = "planned_redownload"
                    return record
                dest.unlink()
            else:
                record.update(status="verified", bytes=size, sha256=digest)
                return record
        else:
            # wrong size: treat it as a partial download and resume it
            part = dest.with_name(dest.name + PART_SUFFIX)
            if size < (src.bytes or 0) and not part.exists():
                os.replace(dest, part)
                record["resumed_partial"] = size
            else:
                dest.unlink()
    if src.extracted_dir is not None:
        root = data_dir / src.extracted_dir
        if root.is_dir():
            record.update(status="extracted_present", extracted=verify_extracted_dir(root))
            record.update(bytes=None, sha256=None)
            return record
    if dry_run:
        record["status"] = "planned_download"
        return record
    if client is None:
        raise DownloadError(f"{src.name}: not present under {data_dir} and no HTTP client given")
    if not src.url:
        raise DownloadError(f"{src.name}: no url configured")
    result = download(
        client,
        src.url,
        dest,
        expected_bytes=src.bytes,
        expected_sha256=src.sha256,
        log=log,
    )
    record.update(status="downloaded" if not result["resumed_from"] else "resumed", **result)
    return record


# --- stage -----------------------------------------------------------------------------------


def fetch(
    cfg: Settings,
    ctx: RunContext,
    sources: str | Iterable[str] | None = None,
    *,
    client: httpx.Client | None = None,
    extract: bool = False,
    delete_raw: bool = False,
    force_cap: float | None = None,
) -> StageResult:
    """Stage ``data.pull`` (CONTRACTS.md section 6 ``pull.fetch(cfg, ctx, sources)``)."""
    keys = parse_sources(sources)
    data_dir = Path(cfg.paths.data_dir)
    records: dict[str, Any] = {}
    summary: dict[str, float | int | str] = {"sources": ",".join(keys)}
    owned_client = client is None and not ctx.dry_run
    http = client
    if owned_client:
        http = make_client()
    try:
        for src in sources_for(cfg, keys):
            rec = ensure_source(src, data_dir, client=http, dry_run=ctx.dry_run, log=ctx.log)
            records[src.name] = rec
            summary[f"{src.name}.status"] = str(rec["status"])
            if rec.get("bytes") is not None:
                summary[f"{src.name}.bytes"] = int(rec["bytes"])
    finally:
        if owned_client and http is not None:
            http.close()

    sources_json = ctx.out_dir / "sources.json"
    sources_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "data_dir": str(data_dir),
                "sources": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ctx.add_output(sources_json, "json")
    for rec in records.values():
        p = Path(rec["path"])
        if p.is_file():
            ctx.add_input(p)

    if extract and not ctx.dry_run:
        extra = run_extractions(cfg, ctx, keys, records, delete_raw=delete_raw, force_cap=force_cap)
        summary.update(extra)
    elif extract:
        ctx.log(extract="planned (dry run)")

    failed = [name for name, rec in records.items() if rec["status"].startswith("sha256")]
    status = "failed" if failed else "ok"
    return StageResult(
        stage="data.pull",
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status=status,
        outputs=list(ctx.outputs),
        summary=summary,
    )


def run_extractions(
    cfg: Settings,
    ctx: RunContext,
    keys: Iterable[str],
    records: dict[str, Any],
    *,
    delete_raw: bool = False,
    force_cap: float | None = None,
) -> dict[str, float | int | str]:
    """MPtrj family/B20 frames, OMat24 in-family frames, WBM sample; counts to the manifest."""
    from b20mlip.data import mptrj, omat24, wbm
    from b20mlip.io import write_frames

    data_dir = Path(cfg.paths.data_dir)
    frames_dir = data_dir / "frames"
    summary: dict[str, float | int | str] = {}
    keys = list(keys)

    if "mptrj" in keys:
        zip_path = Path(records["mptrj_extxyz_zip"]["path"])
        stats: dict[str, Any] = {}
        family = mptrj.extract_family(zip_path, FAMILY_ELEMENTS, None, stats=stats)
        b20 = mptrj.b20_subset(family)
        fam_path = frames_dir / "mptrj_infamily.extxyz"
        b20_path = frames_dir / "mptrj_b20.extxyz"
        write_frames(family, fam_path)
        write_frames(b20, b20_path)
        ctx.add_output(fam_path, "frames")
        ctx.add_output(b20_path, "frames")
        stats.update(b20_frames=len(b20), b20_mp_ids=mptrj.mp_ids(b20))
        ctx.log(mptrj=stats)
        summary.update(
            mptrj_infamily_frames=len(family),
            mptrj_infamily_materials=int(stats.get("members_in_family", 0)),
            mptrj_b20_frames=len(b20),
        )

    if "omat24" in keys:
        rec = records["omat24_1M_tar_gz"]
        if rec["status"] == "extracted_present":
            source = Path(rec["extracted"]["path"])
        else:
            source = Path(rec["path"])
        cap = cfg.data.force_cap_eVA if force_cap is None else force_cap
        out_path = frames_dir / "omat24_infamily.extxyz"
        frames, counts = omat24.stream_filter(source, FAMILY_ELEMENTS, out_path, cap, log=ctx.log)
        ctx.add_output(out_path, "frames")
        ctx.log(omat24=counts)
        summary.update(
            omat24_scanned=int(counts["scanned"]),
            omat24_kept=int(counts["kept"]),
            omat24_dropped_force_cap=int(counts["dropped_force_cap"]),
            omat24_n_b20_spg198=int(counts["n_b20_spg198"]),
        )
        if delete_raw and frames:
            removed = delete_omat24_raw(source)
            ctx.log(omat24_deleted_raw=removed)
            summary["omat24_deleted_raw"] = removed

    if "wbm" in keys:
        csv_path = Path(records["wbm_summary_csv_gz"]["path"])
        zip_path = Path(records["wbm_initial_atoms_zip"]["path"])
        n, seed = cfg.data.wbm_sample_n, cfg.data.wbm_seed
        out_json = data_dir / "wbm" / f"sample_{n}_s{seed}.json"
        result = wbm.sample(csv_path, zip_path, n, seed, FAMILY_ELEMENTS, out=out_json)
        ctx.add_output(out_json, "json")
        ctx.log(
            wbm={
                k: result[k]
                for k in (
                    "n",
                    "seed",
                    "prevalence",
                    "population_prevalence",
                    "population_n",
                    "in_family_n",
                    "in_family_stable_fraction",
                    "hull_column",
                )
            }
        )
        summary.update(
            wbm_sample_n=int(result["n"]),
            wbm_prevalence=float(result["prevalence"]),
            wbm_in_family_n=int(result["in_family_n"]),
            wbm_in_family_stable_fraction=float(result["in_family_stable_fraction"]),
        )

    if "phonondb" in keys:
        from b20mlip.io import read_frames

        path = Path(records["phonondb_pbe_103_structures"]["path"])
        n_structures = len(read_frames(path))
        ctx.log(phonondb={"structures": n_structures})
        summary["phonondb_structures"] = n_structures
    return summary


def delete_omat24_raw(source: Path) -> str:
    """Remove the OMat24 tarball or extracted shard directory (only on explicit request)."""
    if source.is_dir():
        shutil.rmtree(source)
    elif source.is_file():
        source.unlink()
    return str(source)


__all__ = [
    "SOURCES",
    "SOURCE_KEYS",
    "DownloadError",
    "Source",
    "delete_omat24_raw",
    "download",
    "ensure_source",
    "fetch",
    "make_client",
    "parse_sources",
    "run_extractions",
    "sources_for",
    "verify_extracted_dir",
]
