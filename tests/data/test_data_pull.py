"""`data pull`: ranged/resumable downloads (mocked httpx), verification, sources.json and the
local extractions on the synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from helpers_data import RangeServer, mock_client

from b20mlip.config import Settings
from b20mlip.data import pull
from b20mlip.provenance import read_manifest, run_stage, sha256_file

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes
SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://files.test/payload.bin"


def _src(**kw) -> pull.Source:  # type: ignore[no-untyped-def]
    base = dict(
        key="phonondb", name="p", url=URL, relpath="raw/p.bin", bytes=len(PAYLOAD), licence="MIT"
    )
    base.update(kw)
    return pull.Source(**base)


def test_parse_sources_and_registry() -> None:
    assert pull.parse_sources(None) == list(pull.SOURCE_KEYS)
    assert pull.parse_sources("all") == list(pull.SOURCE_KEYS)
    assert pull.parse_sources("wbm, mptrj,wbm") == ["wbm", "mptrj"]
    assert pull.parse_sources(["phonondb"]) == ["phonondb"]
    with pytest.raises(ValueError, match="unknown source"):
        pull.parse_sources("mptrj,bogus")
    cfg = Settings.model_validate(
        {"data": {"omat24_url": "https://x.test/o.tar.gz", "omat24_sha256": "ab" * 32}}
    )
    (omat,) = pull.sources_for(cfg, ["omat24"])
    assert omat.url == "https://x.test/o.tar.gz" and omat.sha256 == "ab" * 32
    assert omat.extracted_dir == "raw/omat24/omat24_1M"
    names = [s.name for s in pull.sources_for(cfg, pull.SOURCE_KEYS)]
    assert len(names) == len(set(names)) == 5
    assert all(s.url.startswith("https://ndownloader.figshare.com/files/") for s in
               pull.sources_for(cfg, ["mptrj", "wbm", "phonondb"]))  # fmt: skip


def test_download_full_get(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD)
    dest = tmp_path / "out.bin"
    res = pull.download(mock_client(server), URL, dest, expected_bytes=len(PAYLOAD),
                        expected_sha256=SHA)  # fmt: skip
    assert dest.read_bytes() == PAYLOAD and not dest.with_name("out.bin.part").exists()
    assert res == {"bytes": len(PAYLOAD), "sha256": SHA, "resumed_from": 0, "attempts": 1}
    assert "Range" not in server.requests[0].headers


def test_download_resumes_partial_file(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD)
    dest = tmp_path / "out.bin"
    dest.with_name("out.bin.part").write_bytes(PAYLOAD[:4000])
    res = pull.download(mock_client(server), URL, dest, expected_bytes=len(PAYLOAD))
    assert server.requests[0].headers["Range"] == "bytes=4000-"
    assert res["resumed_from"] == 4000 and res["sha256"] == SHA and dest.read_bytes() == PAYLOAD


def test_download_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD, honour_range=False)
    dest = tmp_path / "out.bin"
    dest.with_name("out.bin.part").write_bytes(b"garbage" * 100)
    res = pull.download(mock_client(server), URL, dest, expected_bytes=len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD and res["sha256"] == SHA


def test_download_416_means_complete(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD)
    dest = tmp_path / "out.bin"
    dest.with_name("out.bin.part").write_bytes(PAYLOAD)
    res = pull.download(mock_client(server), URL, dest, expected_sha256=SHA)
    assert res["sha256"] == SHA and dest.is_file()


def test_download_errors(tmp_path: Path) -> None:
    dest = tmp_path / "out.bin"
    with pytest.raises(pull.DownloadError, match="expected"):
        pull.download(mock_client(RangeServer(PAYLOAD)), URL, dest, expected_bytes=10)
    assert not dest.with_name("out.bin.part").exists()
    with pytest.raises(pull.DownloadError, match="sha256"):
        pull.download(mock_client(RangeServer(PAYLOAD)), URL, dest, expected_sha256="0" * 64)
    assert not dest.exists()
    with pytest.raises(pull.DownloadError, match="HTTP 403"):
        pull.download(mock_client(RangeServer(PAYLOAD, status=403)), URL, dest)


def test_download_retries_after_transport_error(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD, fail_after=3000)
    dest = tmp_path / "out.bin"
    events: list[dict] = []
    res = pull.download(
        mock_client(server),
        URL,
        dest,
        expected_bytes=len(PAYLOAD),
        chunk_size=1024,  # the failure lands mid-chunk: the two complete chunks are kept
        log=lambda **kw: events.append(kw),
    )
    assert res["attempts"] == 2 and res["sha256"] == SHA
    assert server.requests[1].headers["Range"] == "bytes=2048-"
    assert any(e.get("event") == "retry" for e in events)


def test_download_gives_up_on_persistent_short_reads(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD, truncate_to=5000)
    dest = tmp_path / "out.bin"
    with pytest.raises(pull.DownloadError, match="giving up|downloaded 5000"):
        pull.download(mock_client(server), URL, dest, expected_bytes=len(PAYLOAD), max_attempts=2)
    assert len(server.requests) == 2


def test_ensure_source_states(tmp_path: Path, omat24_dir: Path) -> None:
    data_dir = tmp_path / "data"
    src = _src()
    dest = src.path(data_dir)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD)
    rec = pull.ensure_source(src, data_dir, client=None, dry_run=False)
    assert rec["status"] == "verified" and rec["sha256"] == SHA and rec["bytes"] == len(PAYLOAD)
    assert rec["licence"] == "MIT" and rec["url"] == URL
    # sha mismatch on disk -> re-download (server has the good bytes)
    dest.write_bytes(bytes(len(PAYLOAD)))
    rec = pull.ensure_source(
        replace(src, sha256=SHA), data_dir, client=mock_client(RangeServer(PAYLOAD)), dry_run=False
    )
    assert rec["status"] == "downloaded" and dest.read_bytes() == PAYLOAD
    dest.write_bytes(bytes(len(PAYLOAD)))
    rec = pull.ensure_source(replace(src, sha256=SHA), data_dir, client=None, dry_run=True)
    assert rec["status"] == "planned_redownload"
    # smaller file: resumed as a partial download
    dest.write_bytes(PAYLOAD[:4000])
    server = RangeServer(PAYLOAD)
    rec = pull.ensure_source(src, data_dir, client=mock_client(server), dry_run=False)
    assert rec["status"] == "resumed" and rec["resumed_partial"] == 4000
    assert server.requests[0].headers["Range"] == "bytes=4000-" and dest.read_bytes() == PAYLOAD
    # larger file: discarded and downloaded again
    dest.write_bytes(PAYLOAD + b"x")
    rec = pull.ensure_source(src, data_dir, client=mock_client(RangeServer(PAYLOAD)), dry_run=False)
    assert rec["status"] == "downloaded" and dest.read_bytes() == PAYLOAD
    dest.unlink()
    assert (
        pull.ensure_source(src, data_dir, client=None, dry_run=True)["status"] == "planned_download"
    )
    with pytest.raises(pull.DownloadError, match="no HTTP client"):
        pull.ensure_source(src, data_dir, client=None, dry_run=False)
    with pytest.raises(pull.DownloadError, match="no url"):
        pull.ensure_source(replace(src, url=""), data_dir, client=mock_client(RangeServer(b"")),
                           dry_run=False)  # fmt: skip
    # an extracted directory counts as present (OMat24 shards)
    shutil.copytree(omat24_dir, data_dir / "raw" / "omat24_mini")
    osrc = _src(name="o", relpath="raw/o.tar.gz", extracted_dir="raw/omat24_mini")
    rec = pull.ensure_source(osrc, data_dir, client=None, dry_run=False)
    assert rec["status"] == "extracted_present" and rec["extracted"]["shards"] == 2
    assert len(rec["extracted"]["sha256"]) == 64 and rec["extracted"]["bytes"] > 0
    assert pull.verify_extracted_dir(tmp_path / "empty")["sha256"] is None


def _registry(mptrj_zip: Path, omat24_dir: Path, wbm_files: tuple[Path, Path],
              phonondb_file: Path) -> dict[str, list[pull.Source]]:  # fmt: skip
    csv_path, zip_path = wbm_files
    return {
        "mptrj": [pull.Source("mptrj", "mptrj_extxyz_zip", "https://f.test/mptrj",
                              "raw/mptrj/mptrj.extxyz.zip", mptrj_zip.stat().st_size, "MIT")],
        "wbm": [
            pull.Source("wbm", "wbm_initial_atoms_zip", "https://f.test/wbm-atoms",
                        "raw/references/wbm-initial-atoms.extxyz.zip", zip_path.stat().st_size,
                        "CC-BY-4.0"),
            pull.Source("wbm", "wbm_summary_csv_gz", "https://f.test/wbm-summary",
                        "raw/references/wbm-summary.csv.gz", csv_path.stat().st_size, "CC-BY-4.0"),
        ],
        "phonondb": [pull.Source("phonondb", "phonondb_pbe_103_structures", "https://f.test/ph",
                                 "raw/references/phononDB-PBE-103-structures.extxyz",
                                 phonondb_file.stat().st_size, "CC-BY-4.0")],
        "omat24": [pull.Source("omat24", "omat24_1M_tar_gz", "", "raw/omat24/omat24_1M.tar.gz",
                               1, "CC-BY-4.0", extracted_dir="raw/omat24/omat24_1M")],
    }  # fmt: skip


def _stage_settings(tmp_path: Path, wbm_n: int = 5) -> Settings:
    return Settings.model_validate(
        {
            "paths": {"data_dir": str(tmp_path / "data"), "runs_dir": str(tmp_path / "runs")},
            "data": {"wbm_sample_n": wbm_n, "wbm_seed": 0},
        }
    )


def _place_fixtures(data_dir: Path, mptrj_zip: Path, omat24_dir: Path,
                    wbm_files: tuple[Path, Path], phonondb_file: Path) -> None:  # fmt: skip
    csv_path, zip_path = wbm_files
    for src, rel in (
        (mptrj_zip, "raw/mptrj/mptrj.extxyz.zip"),
        (zip_path, "raw/references/wbm-initial-atoms.extxyz.zip"),
        (csv_path, "raw/references/wbm-summary.csv.gz"),
        (phonondb_file, "raw/references/phononDB-PBE-103-structures.extxyz"),
    ):
        (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, data_dir / rel)
    shutil.copytree(omat24_dir, data_dir / "raw" / "omat24" / "omat24_1M")


def test_fetch_verifies_and_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mptrj_zip: Path, omat24_dir: Path,
    wbm_files: tuple[Path, Path], phonondb_file: Path,
) -> None:  # fmt: skip
    cfg = _stage_settings(tmp_path)
    data_dir = Path(cfg.paths.data_dir)
    _place_fixtures(data_dir, mptrj_zip, omat24_dir, wbm_files, phonondb_file)
    monkeypatch.setattr(pull, "SOURCES", _registry(mptrj_zip, omat24_dir, wbm_files, phonondb_file))
    result = run_stage("data.pull", cfg, pull.fetch, sources="all", extract=True)
    assert result.status == "ok", result.summary
    manifest = read_manifest(result.manifest_path)
    sources_json = Path(manifest.outputs[0].path)
    assert sources_json.name == "sources.json"
    records = json.loads(sources_json.read_text())["sources"]
    assert {k: v["status"] for k, v in records.items()} == {
        "mptrj_extxyz_zip": "verified",
        "wbm_initial_atoms_zip": "verified",
        "wbm_summary_csv_gz": "verified",
        "phonondb_pbe_103_structures": "verified",
        "omat24_1M_tar_gz": "extracted_present",
    }
    assert records["mptrj_extxyz_zip"]["sha256"] == sha256_file(mptrj_zip)
    assert records["mptrj_extxyz_zip"]["licence"] == "MIT"
    s = result.summary
    assert s["mptrj_infamily_frames"] == 6 and s["mptrj_b20_frames"] == 5
    assert s["mptrj_infamily_materials"] == 3
    assert s["omat24_kept"] == 3 and s["omat24_dropped_force_cap"] == 1
    assert s["omat24_n_b20_spg198"] == 2 and s["omat24_scanned"] == 6
    assert s["wbm_sample_n"] == 5 and s["wbm_in_family_n"] == 6
    assert s["wbm_in_family_stable_fraction"] == pytest.approx(2 / 6)
    assert s["phonondb_structures"] == 2
    outputs = {Path(a.path).name for a in manifest.outputs}
    assert outputs >= {"sources.json", "mptrj_infamily.extxyz", "mptrj_b20.extxyz",
                       "omat24_infamily.extxyz", "sample_5_s0.json"}  # fmt: skip
    assert (data_dir / "frames" / "mptrj_b20.extxyz").is_file()
    assert (data_dir / "wbm" / "sample_5_s0.json").is_file()
    assert manifest.extras["mptrj"]["b20_mp_ids"] == {"mp-1431": 2, "mp-871": 3}
    assert manifest.extras["omat24"]["kept"] == 3 and manifest.extras["wbm"]["in_family_n"] == 6
    assert {Path(a.path).name for a in manifest.inputs} >= {"mptrj.extxyz.zip"}
    # raw shards survive unless --delete-raw is given
    assert (data_dir / "raw" / "omat24" / "omat24_1M").is_dir()
    result = run_stage(
        "data.pull", cfg, pull.fetch, sources="omat24", extract=True, delete_raw=True
    )
    assert result.status == "ok" and "omat24_deleted_raw" in result.summary
    assert not (data_dir / "raw" / "omat24" / "omat24_1M").exists()


def test_fetch_dry_run_plans_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  phonondb_file: Path) -> None:  # fmt: skip
    cfg = _stage_settings(tmp_path)
    monkeypatch.setattr(pull, "SOURCES", {"phonondb": [_src(relpath="raw/ph.extxyz")]})
    result = run_stage("data.pull", cfg, pull.fetch, sources="phonondb", extract=True, dry_run=True)
    assert result.status == "partial" and result.outputs == []
    assert result.summary["p.status"] == "planned_download"
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["extract"].startswith("planned")


def test_fetch_downloads_missing_files_with_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _stage_settings(tmp_path)
    monkeypatch.setattr(pull, "SOURCES", {"phonondb": [_src(relpath="raw/ph.bin")]})
    server = RangeServer(PAYLOAD)
    result = run_stage("data.pull", cfg, pull.fetch, sources="phonondb", client=mock_client(server))
    assert result.status == "ok" and result.summary["p.status"] == "downloaded"
    assert (Path(cfg.paths.data_dir) / "raw" / "ph.bin").read_bytes() == PAYLOAD


def test_fetch_without_network_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _stage_settings(tmp_path)
    monkeypatch.setattr(pull, "SOURCES", {"phonondb": [_src(relpath="raw/ph.bin")]})
    result = run_stage("data.pull", cfg, pull.fetch, sources="phonondb")  # sockets are blocked
    assert result.status == "failed" and "error" in result.summary


def test_fetch_reports_sha_mismatch_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _stage_settings(tmp_path)
    src = _src(relpath="raw/ph.bin", sha256="0" * 64)
    monkeypatch.setattr(pull, "SOURCES", {"phonondb": [src]})
    dest = src.path(Path(cfg.paths.data_dir))
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD)
    result = run_stage("data.pull", cfg, pull.fetch, sources="phonondb", dry_run=True)
    assert result.summary["p.status"] == "planned_redownload"


def test_make_client_is_configured() -> None:
    client = pull.make_client()
    assert client.follow_redirects and "b20-mlip" in client.headers["User-Agent"]
    client.close()


@pytest.mark.network
def test_figshare_rejects_head_but_serves_ranges() -> None:  # pragma: no cover - opt-in
    with httpx.Client(follow_redirects=True) as client:
        assert client.head(pull.SOURCES["phonondb"][0].url).status_code == 403
        resp = client.get(pull.SOURCES["phonondb"][0].url, headers={"Range": "bytes=0-9"})
        assert resp.status_code == 206 and len(resp.content) == 10
