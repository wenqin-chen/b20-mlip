"""Honesty gates A1–A11 (CONTRACTS.md section 8) for README.md + reports/numbers.json.

``run(readme, numbers, runs_dir, strict)`` returns violations as ``"A<k>: <detail>"`` strings;
``[]`` means pass. The CLI (``b20mlip report audit [--strict]``) prints them as a JSON list and
exits 1. Definitions used by the gates (all binding):

* **Marker grammar and the numeral allowlist** are those of :mod:`b20mlip.report.markers`
  (fenced code, HTML comments, URLs/links/badges, ``v0.1``/``Python 3.11``/``0.3.16``/``name==1.2``
  versions, ISO dates and letter-led identifier tokens such as ``T0``, ``MACE-MPA-0``, ``mp-871``).
  Inline code is *not* exempt. A number marker must show exactly one numeral that equals the
  published value up to its own rounding, or a numeral-free placeholder (``pending``).
* **Claim sections** (A4/A6/A7): every level-2 section whose heading does not start with
  ``Limitations``, ``Plan`` or ``Non-goals``, including the preamble and the gen blocks inside
  them, minus fenced code — plus the ``bullet`` gen block wherever it sits. This is a superset of
  "prose outside gen blocks": generated tables are held to the same words as prose.
* **Metric keys** (A3): every key whose first segment is ``eval``, ``md``, ``phonons`` or
  ``sampling``.
* **F1 keys** (A4): keys with a segment ``f1`` or ``delta_f1``.
* **WBM energy metrics** (A8): every ``eval.discovery.*`` key except the geometric/count ones
  (last segment starting with ``rmsd``, ``n_`` or ``runtime``); the gate value is 20 meV/atom.
* **Tier-T3 keys** (A9): ``meta.tier == "T3"`` or a key segment ``T3``; an improvement claim is a
  key whose last segment starts with ``delta_``, or two same-metric T3 markers of different
  brackets in one paragraph of claim prose (outside gen blocks) or in the bullet.
* **Cards** (A11): ``MODEL_CARD.md`` and ``NOTICE`` at the README's directory, ``DATA_CARD.md``
  there or under ``docs/``; the violation text names which location was checked.
* **Provenance resolution (A2).** A cited ``(stage, run_id)`` is resolved from
  ``runs/<stage>/<run_id>/manifest.json`` first and, when that is absent (CI checks out the
  repository without ``runs/``), from the tracked snapshot
  ``reports/manifests/<stage>/<run_id>/`` that ``report build`` writes next to
  ``numbers.json`` (manifest plus the run's own ``numbers.json``, nothing heavy). The manifest
  must be ``ok`` (``partial`` only fails under ``--strict``) and its sha256 must equal the one
  recorded in ``numbers.json``. The sha check is mandatory for the ``numbers.json`` artifact,
  which is always present in the snapshot, and for every other output that exists locally;
  outputs that are absent locally (models, trajectories, frame files that were never copied)
  are recorded as ``artifact not local`` in the informational notes (:func:`run_report`) and
  are **not** a violation. When both ``runs/`` and a snapshot exist they must agree.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from b20mlip.config import repo_root
from b20mlip.models import Manifest
from b20mlip.provenance import MANIFEST_NAME, read_manifest, sha256_file
from b20mlip.report.build import normalize
from b20mlip.report.markers import (
    NUMERAL_RE,
    GenBlock,
    MarkerError,
    NumMarker,
    allowlist_spans,
    cell_index,
    fenced_code_spans,
    find_gen_blocks,
    find_num_markers,
    find_numerals,
    html_comment_spans,
    line_of,
    line_text,
    markdown_tables,
    mask_spans,
    sections,
    value_matches,
)
from b20mlip.report.numbers import NUMBERS_FILE, STALE_KEY, load_entries, snapshot_dir

log = logging.getLogger(__name__)

METRIC_NAMESPACES: tuple[str, ...] = ("eval", "md", "phonons", "sampling")
REFERENCE_CODES: frozenset[str] = frozenset({"qe", "vasp", "abinit", "pyscf", "experiment", "mace"})
HEADS: frozenset[str] = frozenset({"Default", "pt_head"})
BACKENDS: frozenset[str] = frozenset({"anthropic", "mock", "scripted"})
NON_CLAIM_PREFIXES: tuple[str, ...] = ("limitations", "plan", "non-goals", "non goals", "nongoals")
BULLET_BLOCK = "bullet"
PARITY_KEY = "parity.passed"
OFFSET_KEY = "offsets.residual_meV_atom"
OFFSET_GATE_MEV_ATOM = 20.0
F1_RE = re.compile(r"(?<![a-z0-9])(?:delta_)?f1(?![a-z0-9])")
FORBIDDEN_TOKEN_RE = re.compile(r"(?<![a-z0-9])(daf|cps|kappa_srme|leaderboard)(?![a-z0-9])")
FORBIDDEN_CLAIMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("first", re.compile(r"\bfirst\b", re.I)),
    ("state-of-the-art", re.compile(r"\bstate[\s-]+of[\s-]+the[\s-]+art\b", re.I)),
    ("SOTA", re.compile(r"\bsota\b", re.I)),
    ("leaderboard", re.compile(r"leaderboard", re.I)),
    ("outperforms all", re.compile(r"\boutperforms\s+all\b", re.I)),
)
LAMMPS_RE = re.compile(r"LAMMPS")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORCE_METRICS: frozenset[str] = frozenset({"mae_f", "rmse_f", "delta_mae_f", "delta_rmse_f"})
REDISTRIBUTION_RE = r"\b(?:never|not)\s+(?:be\s+)?redistribut"
CARD_CHECKS: tuple[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    ("MODEL_CARD.md", ("MODEL_CARD.md",), (("MIT", r"\bMIT\b"),)),
    (
        "DATA_CARD.md",
        ("DATA_CARD.md", "docs/DATA_CARD.md"),
        (
            ("MIT", r"\bMIT\b"),
            ("CC-BY-4.0", r"CC-BY-4\.0"),
            ("SSSP", r"\bSSSP\b"),
            ("not redistributed", REDISTRIBUTION_RE),
        ),
    ),
    (
        "NOTICE",
        ("NOTICE",),
        (
            ("MIT", r"\bMIT\b"),
            ("MPtrj", r"\bMPtrj\b"),
            ("CC-BY-4.0", r"CC-BY-4\.0"),
            ("sAlex", r"\bsAlex\b"),
            ("OMat24", r"\bOMat24\b"),
            ("WBM", r"\bWBM\b"),
            ("SSSP", r"\bSSSP\b"),
            ("not redistributed", REDISTRIBUTION_RE),
        ),
    ),
)


@dataclass
class Audit:
    """Everything the gates share: the README, its markers and blocks, the numbers, the runs."""

    readme_path: Path
    text: str
    entries: dict[str, dict[str, Any]]
    stale: list[dict[str, Any]]
    numbers_text: str
    runs_dir: Path
    strict: bool
    snapshots_dir: Path | None = None
    blocks: list[GenBlock] = field(default_factory=list)
    block_error: str | None = None
    markers: list[NumMarker] = field(default_factory=list)
    manifests: dict[tuple[str, str], Manifest | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # informational, never violations

    def __post_init__(self) -> None:
        try:
            self.blocks = find_gen_blocks(self.text)
        except MarkerError as exc:
            self.block_error = str(exc)
            self.blocks = []
        self.markers = find_num_markers(self.text)

    # -- geometry --------------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.readme_path.resolve().parent

    def block_at(self, pos: int) -> GenBlock | None:
        for block in self.blocks:
            if block.start <= pos < block.end:
                return block
        return None

    def bullet_block(self) -> GenBlock | None:
        return next((b for b in self.blocks if b.name == BULLET_BLOCK), None)

    def cited_keys(self) -> set[str]:
        return {m.key for m in self.markers if m.key in self.entries}

    def value(self, key: str) -> float | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        raw = entry.get("value")
        return float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None

    def meta(self, key: str) -> dict[str, Any]:
        meta = self.entries.get(key, {}).get("meta")
        return meta if isinstance(meta, dict) else {}

    def claim_text(self, *, include_blocks: bool = True) -> str:
        """The README with everything that is not claim text blanked (offsets preserved)."""
        spans = fenced_code_spans(self.text)
        for section in sections(self.text):
            if section.normalized.startswith(NON_CLAIM_PREFIXES):
                spans.append((section.start, section.end))
        if not include_blocks:
            spans.extend((b.start, b.end) for b in self.blocks if b.name != BULLET_BLOCK)
        masked = mask_spans(self.text, spans)
        bullet = self.bullet_block()
        if bullet is not None:
            masked = (
                masked[: bullet.body_start]
                + self.text[bullet.body_start : bullet.body_end]
                + masked[bullet.body_end :]
            )
        return masked

    def non_claim_text(self) -> str:
        spans = fenced_code_spans(self.text)
        for section in sections(self.text):
            if not section.normalized.startswith(NON_CLAIM_PREFIXES):
                spans.append((section.start, section.end))
        return mask_spans(self.text, spans)

    def where(self, pos: int) -> str:
        return f"line {line_of(self.text, pos)}: {line_text(self.text, pos).strip()[:90]!r}"


# --- loading -------------------------------------------------------------------------------------


def load_numbers(
    numbers: str | Path | Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    """Entries, stale list and the raw JSON text (for the A4 substring scan)."""
    if isinstance(numbers, Mapping):
        entries, stale = normalize(numbers)
        return entries, stale, json.dumps({STALE_KEY: stale, **entries}, default=str)
    path = Path(numbers)
    if not path.is_file():
        return {}, [], "{}"
    entries, stale = load_entries(path)
    return entries, stale, path.read_text(encoding="utf-8")


def find_manifest_paths(root: Path, run_id: str, stage: str | None = None) -> list[Path]:
    """Manifests of ``run_id`` under ``root`` (``runs/`` or ``reports/manifests/``): exactly
    ``<root>/<stage>/<run_id>/manifest.json`` when the stage is known, else every stage
    directory holding that id (run ids are unique per stage only)."""
    if not root.is_dir():
        return []
    if stage:
        candidate = root / stage / run_id / MANIFEST_NAME
        return [candidate] if candidate.is_file() else []
    return [
        stage_dir / run_id / MANIFEST_NAME
        for stage_dir in sorted(root.iterdir())
        if (stage_dir / run_id / MANIFEST_NAME).is_file()
    ]


def resolve_run(a: Audit, stage: str, run_id: str) -> tuple[list[Path], list[Path], str]:
    """``(live paths, snapshot paths, source)`` where source is ``runs``, ``snapshot`` or ``""``."""
    live = find_manifest_paths(a.runs_dir, run_id, stage or None)
    snap = (
        find_manifest_paths(a.snapshots_dir, run_id, stage or None)
        if a.snapshots_dir is not None
        else []
    )
    source = "runs" if live else ("snapshot" if snap else "")
    return live, snap, source


def resolve_artifact(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    alt = repo_root() / p
    return alt if alt.exists() else p


def _entry_problems(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return ["entry is not an object"]
    problems = []
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        problems.append("value")
    if not isinstance(entry.get("run_id"), str) or not entry["run_id"]:
        problems.append("run_id")
    sha = entry.get("manifest_sha256")
    if not isinstance(sha, str) or not SHA256_RE.match(sha):
        problems.append("manifest_sha256")
    return problems


# --- gates ---------------------------------------------------------------------------------------


def gate_a1(a: Audit) -> list[str]:
    """Every numeral outside gen blocks sits in a marker whose key is published and consistent."""
    v: list[str] = []
    if a.block_error:
        v.append(f"A1: malformed gen blocks: {a.block_error}")
    for m in a.markers:
        numerals = find_numerals(m.text)
        if not numerals:
            continue  # placeholder such as "pending"
        if len(numerals) != 1 or not NUMERAL_RE.fullmatch(m.text.strip()):
            v.append(
                f"A1: marker num:{m.key} must show one number or a placeholder ({a.where(m.start)})"
            )
            continue
        entry = a.entries.get(m.key)
        if entry is None:
            v.append(
                f"A1: marker num:{m.key} cites a key absent from numbers.json ({a.where(m.start)})"
            )
            continue
        problems = _entry_problems(entry)
        if problems:
            v.append(f"A1: numbers.json entry {m.key} is incomplete: missing/invalid {problems}")
            continue
        if not value_matches(m.text, float(entry["value"])):
            v.append(
                f"A1: marker num:{m.key} shows {m.text.strip()!r} but numbers.json has "
                f"{entry['value']!r} ({a.where(m.start)})"
            )
    spans = [(b.start, b.end) for b in a.blocks]
    spans += [(m.start, m.end) for m in a.markers]
    spans += fenced_code_spans(a.text) + html_comment_spans(a.text)
    masked = mask_spans(a.text, spans)
    masked = mask_spans(masked, allowlist_spans(masked))
    for numeral, start, _ in find_numerals(masked):
        v.append(f"A1: unwrapped numeral {numeral!r} outside a num marker ({a.where(start)})")
    return v


def gate_a2(a: Audit) -> list[str]:
    """Every cited (stage, run_id) resolves, from runs/ or the tracked snapshot, to an ok
    manifest whose sha256 matches numbers.json and whose local outputs still match their sha."""
    v: list[str] = []
    by_run: dict[tuple[str, str], list[str]] = {}
    for key, entry in a.entries.items():
        run_id = entry.get("run_id") if isinstance(entry, dict) else None
        if isinstance(run_id, str) and run_id:
            stage = entry.get("stage")
            by_run.setdefault((stage if isinstance(stage, str) else "", run_id), []).append(key)
    for (stage, run_id), keys in sorted(by_run.items()):
        label = f"run {run_id} (keys {', '.join(sorted(keys)[:3])}{'…' if len(keys) > 3 else ''})"
        live, snap, source = resolve_run(a, stage, run_id)
        paths = live or snap
        if not paths:
            a.manifests[(stage, run_id)] = None
            where = str(a.runs_dir / stage) if stage else str(a.runs_dir)
            if a.snapshots_dir is not None:
                where += f" or {a.snapshots_dir / stage if stage else a.snapshots_dir}"
            v.append(f"A2: {label} has no manifest under {where}")
            continue
        if len(paths) > 1:
            a.manifests[(stage, run_id)] = None
            stages = ", ".join(p.parent.parent.name for p in paths)
            v.append(f"A2: {label} is ambiguous without a stage (found under {stages})")
            continue
        path = paths[0]
        try:
            manifest = read_manifest(path)
        except ValueError as exc:
            a.manifests[(stage, run_id)] = None
            v.append(f"A2: {label}: unreadable manifest {path}: {exc}")
            continue
        a.manifests[(stage, run_id)] = manifest
        a.notes.append(f"A2: {label} resolved from {source} ({path})")
        actual = sha256_file(path)
        expected = {a.entries[k].get("manifest_sha256") for k in keys}
        if expected != {actual}:
            v.append(
                f"A2: {label}: manifest sha256 ({source}) differs from numbers.json; "
                "rerun `b20mlip report build`"
            )
        if live and len(snap) == 1 and sha256_file(snap[0]) != actual:
            v.append(
                f"A2: {label}: snapshot {snap[0]} disagrees with runs/; "
                "rerun `b20mlip report build`"
            )
        if manifest.status == "failed":
            v.append(f"A2: {label} has status=failed")
        elif manifest.status == "partial" and a.strict:
            v.append(f"A2: {label} has status=partial (--strict)")
        v.extend(_check_outputs(a, label, manifest, path.parent, source, snap))
    cited = a.cited_keys()
    for s in a.stale:
        key, run_id = str(s.get("key")), str(s.get("run_id"))
        same_run = (str(s.get("stage", "")), run_id) in by_run
        if key in a.entries or key in cited or same_run:
            v.append(f"A2: key {key} of run {run_id} is stale (numbers.json changed on disk)")
    return v


def _check_outputs(
    a: Audit, label: str, manifest: Manifest, run_dir: Path, source: str, snap: list[Path]
) -> list[str]:
    """Sha-check every output that can be found; ``numbers.json`` must always be found.

    ``numbers.json`` is looked up at its recorded path, in the resolved run directory and in the
    snapshot (a checksum-verified copy anywhere proves the artifact intact); any other output is
    checked only where the manifest recorded it and is otherwise noted as not local.
    """
    v: list[str] = []
    snapshot_dirs = [p.parent for p in snap]
    for art in manifest.outputs:
        name = Path(art.path).name
        candidates = [resolve_artifact(art.path)]
        if name == NUMBERS_FILE:
            candidates += [d / NUMBERS_FILE for d in (run_dir, *snapshot_dirs)]
        local = next((p for p in candidates if p.is_file()), None)
        if local is None:
            if name == NUMBERS_FILE:
                where = source if source == "snapshot" else "runs or snapshot"
                v.append(f"A2: {label}: {NUMBERS_FILE} artifact is missing (not in {where})")
            else:
                a.notes.append(f"A2: {label}: artifact not local, sha unchecked: {art.path}")
            continue
        if sha256_file(local) != art.sha256:
            what = "tampered snapshot" if local.parent in snapshot_dirs else "output"
            v.append(f"A2: {label}: {what} {local} no longer matches its sha256")
    return v


def gate_a3(a: Audit) -> list[str]:
    """Metric keys carry reference.code/functional, e0_source, head, n, seed, ci95 (or a reason)."""
    v: list[str] = []
    for key in sorted(a.entries):
        if key.split(".")[0] not in METRIC_NAMESPACES:
            continue
        meta = a.meta(key)
        problems: list[str] = []
        ref = meta.get("reference")
        if not isinstance(ref, dict):
            problems.append("reference")
        else:
            code = ref.get("code")
            if code not in REFERENCE_CODES:
                problems.append("reference.code")
            if "functional" not in ref:
                problems.append("reference.functional")
            else:
                functional = ref["functional"]
                ok = isinstance(functional, str) and bool(functional.strip())
                if not ok and not (functional is None and code == "experiment"):
                    problems.append("reference.functional")
        e0 = meta.get("e0_source")
        if not isinstance(e0, str) or not e0.strip():
            problems.append("e0_source")
        if meta.get("head") not in HEADS:
            problems.append("head")
        n = meta.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            problems.append("n")
        seed = meta.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            problems.append("seed")
        if "ci95" not in meta:
            problems.append("ci95")
        else:
            ci = meta["ci95"]
            interval = (
                isinstance(ci, list | tuple)
                and len(ci) == 2
                and all(isinstance(x, int | float) and not isinstance(x, bool) for x in ci)
            )
            reason = meta.get("ci95_reason")
            explicit_null = ci is None and isinstance(reason, str) and bool(reason.strip())
            if not (interval or explicit_null):
                problems.append("ci95 (interval, or null with ci95_reason)")
        if problems:
            v.append(f"A3: {key}: missing or invalid meta {problems}")
    return v


def gate_a4(a: Audit) -> list[str]:
    """F1 entries are paired, natural-prevalence, n=1000; daf/cps/kappa_srme/leaderboard nowhere."""
    v: list[str] = []
    for key in sorted(a.entries):
        if not F1_RE.search(key.lower()):
            continue
        meta = a.meta(key)
        problems: list[str] = []
        if meta.get("prevalence") != "natural":
            problems.append('prevalence != "natural"')
        if meta.get("paired_vs") != "B0":
            problems.append('paired_vs != "B0"')
        if "sample_seed" not in meta:
            problems.append("sample_seed missing")
        if meta.get("n") != 1000 or isinstance(meta.get("n"), bool):
            problems.append("n != 1000")
        if problems:
            v.append(f"A4: F1 key {key}: {problems}")
    for token in sorted({m.group(1) for m in FORBIDDEN_TOKEN_RE.finditer(a.numbers_text.lower())}):
        v.append(f"A4: forbidden token {token!r} appears in numbers.json")
    claims = a.claim_text()
    for m in FORBIDDEN_TOKEN_RE.finditer(claims.lower()):
        v.append(f"A4: forbidden token {m.group(1)!r} in README claims ({a.where(m.start())})")
    return v


def _marker_reference(a: Audit, key: str) -> tuple[str | None, str | None, bool, bool]:
    meta = a.meta(key)
    raw = meta.get("reference")
    ref: dict[str, Any] = raw if isinstance(raw, dict) else {}
    cross = bool(meta.get("cross_functional")) or bool(ref.get("cross_functional"))
    return ref.get("code"), ref.get("functional"), cross, bool(meta.get("lower_fidelity"))


def gate_a5(a: Audit) -> list[str]:
    """Table rows in gen blocks cite one code+functional; PBEsol/PySCF cells are flagged."""
    v: list[str] = []
    for block in a.blocks:
        for table in markdown_tables(a.text, block.body_start, block.body_end):
            for row in table.rows:
                label = row.cells[0] if row.cells else "?"
                refs: set[tuple[str | None, str | None]] = set()
                for m in a.markers:
                    if not (row.start <= m.start < row.end) or m.key not in a.entries:
                        continue
                    code, functional, cross, lower = _marker_reference(a, m.key)
                    col = cell_index(a.text[row.start : row.end], m.start - row.start)
                    header = table.header[col] if col < len(table.header) else f"column {col}"
                    if functional == "PBEsol" and not cross:
                        v.append(
                            f"A5: PBEsol value {m.key} in column {header!r} of block "
                            f"{block.name!r} is not flagged cross_functional"
                        )
                    if code == "pyscf" and not lower:
                        v.append(
                            f"A5: PySCF value {m.key} in row {label!r} of block {block.name!r} "
                            "is not flagged lower_fidelity"
                        )
                    if not cross:
                        refs.add((code, functional))
                if len(refs) > 1:
                    pretty = ", ".join(f"{c}/{f}" for c, f in sorted(refs, key=str))
                    v.append(
                        f"A5: row {label!r} of block {block.name!r} cites several "
                        f"references: {pretty}"
                    )
    return v


def gate_a6(a: Audit) -> list[str]:
    """The string LAMMPS in claim sections or the bullet needs parity.passed == 1."""
    if a.value(PARITY_KEY) == 1.0:
        return []
    v: list[str] = []
    seen: set[int] = set()
    for m in LAMMPS_RE.finditer(a.claim_text()):
        line = line_of(a.text, m.start())
        if line in seen:
            continue
        seen.add(line)
        v.append(
            f"A6: 'LAMMPS' in a claim section without parity.passed == 1 ({a.where(m.start())})"
        )
    return v


def _quoted(line: str, start: int, end: int) -> bool:
    before, after = line[:start], line[end:]
    if before.count('"') % 2 == 1 and '"' in after:
        return True
    if before.count("`") % 2 == 1 and "`" in after:
        return True
    return before.rfind("“") > before.rfind("”") and "”" in after


def gate_a7(a: Audit) -> list[str]:
    """Forbidden words in claims; in Limitations/Plan/Non-goals only inside quotes."""
    v: list[str] = []
    claims = a.claim_text()
    for label, pattern in FORBIDDEN_CLAIMS:
        for m in pattern.finditer(claims):
            v.append(f"A7: forbidden string {label!r} in a claim section ({a.where(m.start())})")
    other = a.non_claim_text()
    for label, pattern in FORBIDDEN_CLAIMS:
        for m in pattern.finditer(other):
            line_start = other.rfind("\n", 0, m.start()) + 1
            line = line_text(a.text, m.start())
            if not _quoted(line, m.start() - line_start, m.end() - line_start):
                v.append(
                    f"A7: forbidden string {label!r} outside quotes in a non-claim section "
                    f"({a.where(m.start())})"
                )
    return v


def _is_wbm_energy_metric(key: str) -> bool:
    if not key.startswith("eval.discovery."):
        return False
    last = key.rsplit(".", 1)[-1]
    return not last.startswith(("rmsd", "n_", "runtime"))


def gate_a8(a: Audit) -> list[str]:
    """WBM energy metrics off the MP scale and off pt_head need the offset residual <= 20."""
    v: list[str] = []
    residual = a.value(OFFSET_KEY)
    for key in sorted(a.entries):
        if not _is_wbm_energy_metric(key):
            continue
        meta = a.meta(key)
        scale, head = meta.get("energy_scale"), meta.get("head")
        if scale is None:
            v.append(f"A8: WBM metric {key} does not declare meta.energy_scale")
            continue
        if scale == "mp" or head == "pt_head":
            continue
        if residual is None:
            v.append(f"A8: WBM metric {key} (scale {scale!r}, head {head!r}) needs {OFFSET_KEY}")
        elif residual > OFFSET_GATE_MEV_ATOM:
            v.append(
                f"A8: WBM metric {key} (scale {scale!r}, head {head!r}) but {OFFSET_KEY} = "
                f"{residual} > {OFFSET_GATE_MEV_ATOM:g}"
            )
    return v


def _is_t3(a: Audit, key: str) -> bool:
    return a.meta(key).get("tier") == "T3" or "T3" in key.split(".")


def _floor(a: Audit, key: str) -> float | None:
    raw = a.meta(key).get("noise_floor_f")
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        return None
    return float(raw)


def gate_a9(a: Audit) -> list[str]:
    """Tier-T3 keys carry noise_floor_f; improvement claims below the floor are flagged."""
    v: list[str] = []
    t3 = [key for key in sorted(a.entries) if _is_t3(a, key)]
    for key in t3:
        floor = _floor(a, key)
        if floor is None:
            v.append(f"A9: tier-T3 key {key} lacks meta.noise_floor_f")
            continue
        last = key.rsplit(".", 1)[-1]
        value = a.value(key)
        if last.startswith("delta_") and value is not None and abs(value) < floor:
            v.append(
                f"A9: improvement claim {key} = {value} is below the T3 noise floor {floor} meV/Å"
            )
    prose = a.claim_text(include_blocks=False)
    for start, end in _paragraphs(prose):
        cited = [
            m
            for m in a.markers
            if start <= m.start < end
            and m.key in a.entries
            and _is_t3(a, m.key)
            and _in_claim_prose(a, m)
        ]
        for i, m1 in enumerate(cited):
            for m2 in cited[i + 1 :]:
                p1, p2 = m1.key.split("."), m2.key.split(".")
                if p1[-1] != p2[-1] or p1[-1] not in FORCE_METRICS or p1 == p2:
                    continue
                f1, f2 = _floor(a, m1.key), _floor(a, m2.key)
                v1, v2 = a.value(m1.key), a.value(m2.key)
                if f1 is None or f2 is None or v1 is None or v2 is None:
                    continue
                floor = max(f1, f2)
                if abs(v1 - v2) < floor:
                    v.append(
                        f"A9: {m1.key} vs {m2.key} differ by {abs(v1 - v2):.4g} < noise floor "
                        f"{floor:g} meV/Å; not a claimable improvement ({a.where(m1.start)})"
                    )
    return v


def _in_claim_prose(a: Audit, m: NumMarker) -> bool:
    """Outside every gen block, or inside the bullet block (the only generated claim text)."""
    block = a.block_at(m.start)
    return block is None or block.name == BULLET_BLOCK


def _paragraphs(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n[ \t]*\n", text):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(text)))
    return out


def gate_a10(a: Audit) -> list[str]:
    """Agent numbers carry backend/model_id/trace_path; mock-backend numbers never reach README."""
    v: list[str] = []
    cited = a.cited_keys()
    for key in sorted(a.entries):
        if not key.startswith("agent."):
            continue
        meta = a.meta(key)
        problems: list[str] = []
        backend = meta.get("backend")
        if backend not in BACKENDS:
            problems.append("backend")
        for name in ("model_id", "trace_path"):
            if not isinstance(meta.get(name), str) or not meta[name].strip():
                problems.append(name)
        if problems:
            v.append(f"A10: agent key {key}: missing or invalid meta {problems}")
        entry = a.entries[key]
        manifest = a.manifests.get((str(entry.get("stage", "")), str(entry.get("run_id"))))
        if manifest is not None:
            for name in ("backend", "model_id"):
                recorded = manifest.extras.get(name)
                if recorded is not None and name in meta and recorded != meta[name]:
                    v.append(
                        f"A10: agent key {key}: meta.{name} {meta[name]!r} disagrees with the "
                        f"manifest ({recorded!r})"
                    )
        if backend == "mock" and key in cited:
            v.append(f"A10: mock-backend number {key} is cited in README")
    return v


def card_locations(root: Path) -> dict[str, Path | None]:
    """Which candidate path of each licence file exists (``None`` when none does)."""
    found: dict[str, Path | None] = {}
    for name, candidates, _ in CARD_CHECKS:
        found[name] = next((root / c for c in candidates if (root / c).is_file()), None)
    return found


def gate_a11(a: Audit) -> list[str]:
    """MODEL_CARD.md, DATA_CARD.md (root or docs/) and NOTICE exist and name the licences."""
    v: list[str] = []
    found = card_locations(a.root)
    for name, candidates, required in CARD_CHECKS:
        path = found[name]
        if path is None:
            v.append(f"A11: {name} not found (looked at {', '.join(candidates)} under {a.root})")
            continue
        rel = path.relative_to(a.root)
        text = path.read_text(encoding="utf-8")
        for label, pattern in required:
            if re.search(pattern, text, re.I if label == "not redistributed" else 0) is None:
                v.append(f"A11: {name} (found at {rel}) does not mention {label!r}")
        log.info("A11: %s found at %s", name, rel)
    return v


GATES = (
    gate_a1,
    gate_a2,
    gate_a3,
    gate_a4,
    gate_a5,
    gate_a6,
    gate_a7,
    gate_a8,
    gate_a9,
    gate_a10,
    gate_a11,
)


def run_report(
    readme: str | Path,
    numbers: str | Path | Mapping[str, Any],
    runs_dir: str | Path,
    strict: bool = False,
    *,
    snapshots: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Gates A1–A11: ``(violations, notes)``. Notes are informational (provenance sources,
    artifacts that are not local); only violations fail the audit. ``snapshots`` defaults to
    ``reports/manifests/`` next to the numbers file when ``numbers`` is a path."""
    readme_path = Path(readme)
    if not readme_path.is_file():
        return [f"A1: README not found: {readme_path}"], []
    entries, stale, numbers_text = load_numbers(numbers)
    snapshots_dir: Path | None
    if snapshots is not None:
        snapshots_dir = Path(snapshots)
    elif not isinstance(numbers, Mapping):
        snapshots_dir = snapshot_dir(numbers)
    else:
        snapshots_dir = None
    audit = Audit(
        readme_path=readme_path,
        text=readme_path.read_text(encoding="utf-8"),
        entries=entries,
        stale=stale,
        numbers_text=numbers_text,
        runs_dir=Path(runs_dir),
        strict=strict,
        snapshots_dir=snapshots_dir,
    )
    violations: list[str] = []
    for gate in GATES:
        violations.extend(gate(audit))
    return violations, list(audit.notes)


def run(
    readme: str | Path,
    numbers: str | Path | Mapping[str, Any],
    runs_dir: str | Path,
    strict: bool = False,
    *,
    snapshots: str | Path | None = None,
) -> list[str]:
    """Run gates A1–A11; return violations (``[]`` = pass); ``numbers`` is a path or mapping."""
    return run_report(readme, numbers, runs_dir, strict, snapshots=snapshots)[0]


__all__ = [
    "BACKENDS",
    "BULLET_BLOCK",
    "CARD_CHECKS",
    "FORBIDDEN_CLAIMS",
    "FORBIDDEN_TOKEN_RE",
    "GATES",
    "HEADS",
    "METRIC_NAMESPACES",
    "NON_CLAIM_PREFIXES",
    "OFFSET_GATE_MEV_ATOM",
    "OFFSET_KEY",
    "PARITY_KEY",
    "REFERENCE_CODES",
    "Audit",
    "card_locations",
    "find_manifest_paths",
    "gate_a1",
    "gate_a2",
    "gate_a3",
    "gate_a4",
    "gate_a5",
    "gate_a6",
    "gate_a7",
    "gate_a8",
    "gate_a9",
    "gate_a10",
    "gate_a11",
    "load_numbers",
    "resolve_artifact",
    "resolve_run",
    "run",
    "run_report",
]
