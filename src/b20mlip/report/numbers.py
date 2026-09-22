"""Harvest published numbers from run manifests into ``reports/numbers.json`` (CONTRACTS row 14).

Publishing convention (binding for every tier that wants a number in the README)
================================================================================
A stage publishes numbers by writing **``numbers.json`` in its own run directory**
(``ctx.out_dir / "numbers.json"``) and registering it with ``ctx.add_output(path, "json")`` so
the manifest carries its sha256. Only manifests with ``status == "ok"`` are harvested; a file
whose sha256 no longer matches the manifest is *stale* and its keys are listed under
``"@stale"`` instead of being published (gate A2 flags stale keys that are still cited).

File format::

    {
      "@meta": {"reference": {...}, "head": "Default", "seed": 0},   # file-level defaults (opt.)
      "eval.errors.T0.B1.mae_f": 12.3,                               # key -> finite number
      "eval.errors.T0.B1.mae_f@meta": {"n": 120, "ci95": [11.8, 12.9], "tier": "T0"}
    }

* Keys are dotted, ``[A-Za-z0-9][A-Za-z0-9_.-]*``; the first segment is the namespace:
  ``eval`` (``eval.errors.<tier>.<bracket>.<metric>``, ``eval.discovery.<bracket>.<metric>``,
  ``eval.phonons.<compound>.<bracket>.<metric>``, ``eval.elastic...``), ``md`` (``md.ase.<compound>
  .<label>.<metric>``, ``md.lammps...``, ``md.parity.<label>...``; runs before 2026-09-22 omitted
  ``<label>``), ``sampling`` (``sampling.umbrella.<compound>
  .dF_eV``, ``sampling.neb.<compound>.Ea_eV``), ``agent`` (``agent.eval.<metric>``), ``data``
  (``data.n_qe_frames`` ...), ``offsets.residual_meV_atom``, ``parity.<label>.passed`` (0/1) and
  ``active.n_selected``. Brackets are ``B0, B0p (= B0′), B1, B2, B3, B4``; tiers ``T0..T4a``.
  ``templates/README.md.j2`` documents exactly which keys each README table reads.
* ``<key>@meta`` is a JSON object merged over the file-level ``"@meta"`` (shallow). The gates need:
  - every ``eval.*``/``md.*``/``phonons.*``/``sampling.*`` key (A3): ``reference`` = ``{"code",
    "functional", "pseudos", "e0_source"}`` (``functional`` may be null only for
    ``code == "experiment"``), ``e0_source``, ``head`` (``Default``/``pt_head``), ``n`` (int ≥ 1),
    ``seed`` (int), ``ci95`` = ``[lo, hi]`` or ``null`` together with ``ci95_reason``;
  - F1 keys (A4): ``prevalence: "natural"``, ``paired_vs: "B0"``, ``sample_seed``, ``n: 1000``;
  - table cells (A5): ``cross_functional: true`` on PBEsol columns, ``lower_fidelity: true`` on
    PySCF rows;
  - ``eval.discovery.*`` energy metrics (A8): ``energy_scale`` and ``head`` of the model;
  - tier-T3 keys (A9): ``tier: "T3"`` and ``noise_floor_f`` (meV/Å); improvement claims are keys
    whose last segment starts with ``delta_``;
  - ``agent.*`` keys (A10): ``backend`` (``anthropic``/``mock``/``scripted``), ``model_id``,
    ``trace_path``.
  Free extra fields (``unit``, ``bracket``, ``tier``, ``compound``, ``source``) are kept verbatim.
* The strings ``daf``, ``cps``, ``kappa_srme`` and ``leaderboard`` may appear nowhere (A4).
* A newer ok run of the same key replaces an older one (manifest ``created_at``, then run_id).

Output format (``reports/numbers.json``, sorted keys)::

    {"@stale": [{"key", "run_id", "stage", "path", "expected_sha256", "actual_sha256"}],
     "@snapshots": {"<stage>/<run_id>": "reports/manifests/<stage>/<run_id>"},
     "<key>": {"value": float, "run_id": str, "manifest_sha256": str, "stage": str, "meta": {}}}

``stage`` is part of every entry because ``run_id`` is unique per stage directory only; the
``NumberRef`` model (CONTRACTS section 2) stays unchanged and ``load(path)`` returns that view
(key, value, run_id, manifest_sha256); ``load_entries(path)`` returns the raw entries plus the
stale list.

Provenance snapshots (shipped with the repository)
==================================================
``runs/`` is gitignored, so the manifests that back the README could not be checked in CI.
``write_numbers`` therefore copies, for every run cited by a published number, that run's
``manifest.json`` and its (small) ``numbers.json`` into
``reports/manifests/<stage>/<run_id>/``; nothing else is copied (models, trajectories and frame
files are never snapshotted). The snapshot directory is tracked (``.gitignore`` un-ignores it),
snapshots of runs that are no longer cited are removed on the next write, and the audit falls
back to it when ``runs/`` is absent (:mod:`b20mlip.report.audit`, gate A2).
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from b20mlip.models import Manifest, NumberRef, StageName
from b20mlip.provenance import MANIFEST_NAME, find_runs, sha256_file
from b20mlip.report.markers import KEY_RE

NUMBERS_FILE = "numbers.json"
DEFAULT_OUT = Path("reports") / NUMBERS_FILE
SNAPSHOT_DIRNAME = "manifests"
STALE_KEY = "@stale"
SNAPSHOTS_KEY = "@snapshots"
FILE_META_KEY = "@meta"
META_SUFFIX = "@meta"
SKIPPED_STAGES: frozenset[str] = frozenset({"report"})  # never harvest our own output


@dataclass
class Entry:
    key: str
    value: float
    run_id: str
    manifest_sha256: str
    stage: str
    created_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "stage": self.stage,
            "meta": self.meta,
        }

    def as_ref(self) -> NumberRef:
        return NumberRef(
            key=self.key,
            value=self.value,
            run_id=self.run_id,
            manifest_sha256=self.manifest_sha256,
        )


@dataclass
class Harvest:
    entries: dict[str, Entry] = field(default_factory=dict)
    stale: list[dict[str, Any]] = field(default_factory=list)
    n_manifests: int = 0
    n_ok: int = 0

    def as_json(self, snapshots: dict[str, str] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {STALE_KEY: sorted(self.stale, key=lambda s: (s["key"], s["run_id"]))}
        if snapshots is not None:
            out[SNAPSHOTS_KEY] = dict(sorted(snapshots.items()))
        for key in sorted(self.entries):
            out[key] = self.entries[key].as_json()
        return out

    def cited_runs(self) -> dict[tuple[str, str], list[str]]:
        """``{(stage, run_id): [keys]}`` of every run that backs a published number."""
        runs: dict[tuple[str, str], list[str]] = {}
        for key, entry in self.entries.items():
            runs.setdefault((entry.stage, entry.run_id), []).append(key)
        return runs

    def refs(self) -> dict[str, NumberRef]:
        return {key: entry.as_ref() for key, entry in sorted(self.entries.items())}


# --- reading a stage's numbers.json -------------------------------------------------------------


def _coerce_value(key: str, raw: Any, where: str) -> float:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, int | float) and math.isfinite(float(raw)):
        return float(raw)
    raise ValueError(f"{where}: key {key!r} must be a finite number, got {raw!r}")


def parse_numbers_file(
    data: Any, where: str = NUMBERS_FILE
) -> dict[str, tuple[float, dict[str, Any]]]:
    """Validate one stage file: ``{key: (value, meta)}`` with file-level ``@meta`` merged in."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: top level must be an object")
    file_meta = data.get(FILE_META_KEY, {})
    if not isinstance(file_meta, dict):
        raise ValueError(f"{where}: {FILE_META_KEY!r} must be an object")
    values: dict[str, float] = {}
    metas: dict[str, dict[str, Any]] = {}
    for raw_key, raw in data.items():
        if raw_key == FILE_META_KEY:
            continue
        if raw_key.endswith(META_SUFFIX):
            key = raw_key[: -len(META_SUFFIX)]
            if not isinstance(raw, dict):
                raise ValueError(f"{where}: {raw_key!r} must be an object")
            metas[key] = raw
            continue
        if not KEY_RE.match(raw_key):
            raise ValueError(f"{where}: invalid key {raw_key!r}")
        values[raw_key] = _coerce_value(raw_key, raw, where)
    orphan = sorted(set(metas) - set(values))
    if orphan:
        raise ValueError(f"{where}: meta without a number: {orphan}")
    return {key: (value, {**file_meta, **metas.get(key, {})}) for key, value in values.items()}


def _numbers_artifact(manifest: Manifest) -> Any | None:
    for art in manifest.outputs:
        if Path(art.path).name == NUMBERS_FILE:
            return art
    return None


def harvest(runs_dir: str | Path) -> Harvest:
    """Walk ``runs/<stage>/<run_id>/manifest.json`` for every stage and merge fresh numbers."""
    runs = Path(runs_dir)
    result = Harvest()
    for stage in get_args(StageName):
        if stage in SKIPPED_STAGES:
            continue
        for manifest in find_runs(stage, runs):
            result.n_manifests += 1
            if manifest.status != "ok":
                continue
            result.n_ok += 1
            art = _numbers_artifact(manifest)
            if art is None:
                continue
            run_dir = runs / stage / manifest.run_id
            path = run_dir / NUMBERS_FILE
            actual = sha256_file(path) if path.is_file() else None
            manifest_sha = sha256_file(run_dir / MANIFEST_NAME)
            if actual != art.sha256:
                data = json.loads(path.read_text(encoding="utf-8")) if actual else {}
                keys = [k for k in data if k != FILE_META_KEY and not k.endswith(META_SUFFIX)]
                for key in keys or ["*"]:
                    result.stale.append(
                        {
                            "key": key,
                            "run_id": manifest.run_id,
                            "stage": stage,
                            "path": str(path),
                            "expected_sha256": art.sha256,
                            "actual_sha256": actual,
                        }
                    )
                continue
            parsed = parse_numbers_file(
                json.loads(path.read_text(encoding="utf-8")), where=f"{stage}/{manifest.run_id}"
            )
            for key, (value, meta) in parsed.items():
                entry = Entry(
                    key=key,
                    value=value,
                    run_id=manifest.run_id,
                    manifest_sha256=manifest_sha,
                    stage=stage,
                    created_at=manifest.created_at,
                    meta=meta,
                )
                current = result.entries.get(key)
                if current is None or (entry.created_at, entry.run_id) >= (
                    current.created_at,
                    current.run_id,
                ):
                    result.entries[key] = entry
    return result


# --- reports/numbers.json I/O --------------------------------------------------------------------


def snapshot_dir(numbers_path: str | Path) -> Path:
    """``reports/manifests/`` next to ``reports/numbers.json``."""
    return Path(numbers_path).parent / SNAPSHOT_DIRNAME


def write_snapshots(result: Harvest, runs_dir: str | Path, out: str | Path) -> dict[str, str]:
    """Copy ``manifest.json`` + ``numbers.json`` of every cited run into the snapshot dir.

    Returns ``{"<stage>/<run_id>": "<snapshot dir>"}`` (paths relative to the numbers file's
    parent's parent when possible, i.e. ``reports/manifests/...`` for the default layout).
    Snapshots of runs that are no longer cited are deleted so the tracked directory never
    grows stale. Heavy outputs are never copied.
    """
    root = snapshot_dir(out)
    base = Path(out).parent.parent
    wanted: dict[str, str] = {}
    for stage, run_id in sorted(result.cited_runs()):
        src = Path(runs_dir) / stage / run_id
        dst = root / stage / run_id
        dst.mkdir(parents=True, exist_ok=True)
        for name in (MANIFEST_NAME, NUMBERS_FILE):
            shutil.copyfile(src / name, dst / name)
        try:
            rel = dst.relative_to(base)
        except ValueError:
            rel = dst
        wanted[f"{stage}/{run_id}"] = str(rel)
    if root.is_dir():
        for stage_dir in root.iterdir():
            if not stage_dir.is_dir():
                continue
            for run_dir in stage_dir.iterdir():
                if run_dir.is_dir() and f"{stage_dir.name}/{run_dir.name}" not in wanted:
                    shutil.rmtree(run_dir)
            if not any(stage_dir.iterdir()):
                stage_dir.rmdir()
    return wanted


def write_numbers(result: Harvest, out: str | Path, runs_dir: str | Path | None = None) -> Path:
    """Write ``numbers.json``; with ``runs_dir`` also refresh the provenance snapshots."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = write_snapshots(result, runs_dir, path) if runs_dir is not None else None
    path.write_text(
        json.dumps(result.as_json(snapshots), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return path


def collect(runs_dir: str | Path, out: str | Path | None = DEFAULT_OUT) -> dict[str, NumberRef]:
    """Harvest ``runs_dir`` and write ``reports/numbers.json`` (``out=None`` skips the write)."""
    result = harvest(runs_dir)
    if out is not None:
        write_numbers(result, out, runs_dir=runs_dir)
    return result.refs()


def load_entries(path: str | Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Raw ``{key: {value, run_id, manifest_sha256, stage, meta}}`` plus the stale list."""
    p = Path(path)
    if not p.is_file():
        return {}, []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be an object")
    stale = data.get(STALE_KEY, [])
    entries = {k: v for k, v in data.items() if not k.startswith("@")}
    return entries, list(stale) if isinstance(stale, list) else []


def load(path: str | Path) -> dict[str, NumberRef]:
    entries, _ = load_entries(path)
    refs: dict[str, NumberRef] = {}
    for key, raw in entries.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: entry {key!r} must be an object")
        refs[key] = NumberRef(
            key=key,
            value=raw["value"],
            run_id=raw["run_id"],
            manifest_sha256=raw["manifest_sha256"],
        )
    return refs


__all__ = [
    "DEFAULT_OUT",
    "FILE_META_KEY",
    "META_SUFFIX",
    "NUMBERS_FILE",
    "SKIPPED_STAGES",
    "SNAPSHOTS_KEY",
    "SNAPSHOT_DIRNAME",
    "STALE_KEY",
    "Entry",
    "Harvest",
    "collect",
    "harvest",
    "load",
    "load_entries",
    "parse_numbers_file",
    "snapshot_dir",
    "write_numbers",
    "write_snapshots",
]
