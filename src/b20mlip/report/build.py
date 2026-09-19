"""Render ``templates/README.md.j2`` from ``reports/numbers.json`` and write ``README.md``.

``render(numbers, template)`` (CONTRACTS row 14) turns the published numbers into the README:
every number is emitted as ``<!-- num:<key> -->VALUE<!-- /num -->`` (grammar in
:mod:`b20mlip.report.markers`), every table sits in a gen block
(``<!-- gen:start:<name> -->…<!-- gen:end -->``), a missing key renders ``pending`` and a table
without any number renders an explicit "not yet run" line. Nothing is ever typed by hand: the
template is rendered with
``StrictUndefined`` from a context built here (:func:`build_context`).

``write_readme(text, path)`` replaces only the gen blocks of an existing README (hand-written
prose outside them survives, number markers in that prose are refreshed) and writes the whole
rendered file when the README has no gen blocks yet. ``run(cfg, ctx, readme=...)`` is the
``report build`` stage function.

README key table (what each gen block reads; brackets ``B0 B0p B1 B2 B3``, tiers ``T0..T4a``,
compounds ``FeSi CoSi MnSi FeGe``):

===============  ==================================================================================
block            keys
===============  ==================================================================================
status           (counts of the numbers file itself)
oneliner         ``parity.passed`` (== 1 unlocks the second MD engine in the one-liner)
forces           ``eval.errors.<tier>.<bracket>.mae_f``            [meV/Å]
energies         ``eval.errors.<tier>.<bracket>.mae_e``            [meV/atom]
discovery        ``eval.discovery.<bracket>.{delta_f1, mae_e_above_hull, rmsd}``;
                 ``offsets.residual_meV_atom`` (> 20 renders "n/a (scale)" for B1)
phonons          ``eval.phonons.<compound|phononDB103>.<bracket>.{omega_mae_meV, softening_index,
                 imaginary_count, omega_mae_meV_pbesol}`` (the last one is the cross-functional col)
thermal          ``md.<ase|lammps>.<compound>.{a_300K_A, a_exp_A, a_dev_pct, alpha_per_K}``
                 (all referenced to ``experiment``); ``parity.passed`` unlocks lammps rows
stability        ``md.<ase|lammps>.<compound>.drift_meV_atom_ps`` (reference code ``mace``)
parity           ``md.parity.{max_dF_eVA, max_dE_eV_atom}``; ``parity.passed``
sampling         ``sampling.umbrella.<compound>.{dF_eV, dF_err_eV}``;
                 ``sampling.neb.<compound>.Ea_eV``
agent            ``agent.eval.{accuracy, invalid_call_rate, dag_valid_rate, provenance_rate,
                 recovery_rate, tokens_per_task, usd_per_task, n_tasks}`` (mock backend hidden)
bullet           ``data.n_qe_frames`` | ``data.n_omat24_frames``, ``active.n_selected`` + the above
===============  ==================================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from b20mlip.config import Settings, repo_root
from b20mlip.models import NumberRef, StageResult
from b20mlip.provenance import RunContext
from b20mlip.report.markers import (
    MarkerError,
    find_gen_blocks,
    find_num_markers,
    format_value,
    num_marker,
)
from b20mlip.report.numbers import STALE_KEY, Harvest, harvest, write_numbers

TEMPLATE_NAME = "README.md.j2"
PENDING = "pending"
BRACKETS: tuple[tuple[str, str], ...] = (
    ("B0", "B0 MPA-0 zero-shot"),
    ("B0p", "B0′ MP-0 zero-shot"),
    ("B1", "B1 naive fine-tune"),
    ("B2", "B2 multihead replay"),
    ("B3", "B3 scratch"),
)
TIERS: tuple[tuple[str, str], ...] = (
    ("T0", "T0 held-out groups"),
    ("T1", "T1 hot snapshots"),
    ("T2", "T2 FeGe (never trained)"),
    ("T3", "T3 OMat24 VASP"),
    ("T4a", "T4a MPtrj forgetting"),
)
COMPOUNDS: tuple[str, ...] = ("FeSi", "CoSi", "MnSi", "FeGe")
FINE_TUNED: tuple[str, ...] = ("B2", "B1", "B3")
OFFSET_GATE_MEV_ATOM = 20.0
# Protocol constants that are not in Settings (SPEC.md section 6); rendered inside a gen block.
PROTOCOL_CONSTANTS: dict[str, Any] = {
    "parity_frames": 20,
    "parity_tol_f_eVA": 1e-3,
    "parity_tol_e_eV_atom": 1e-4,
    "imaginary_threshold_meV": -0.4,
    "seeds": 3,
    "qpoints": 100,
    "elastic_strains_pct": "0.5 / 1",
}


def templates_dir() -> Path:
    return repo_root() / "templates"


# --- number view used by the template -------------------------------------------------------------


def normalize(numbers: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Accept ``{key: NumberRef}`` or the raw ``numbers.json`` mapping; return entries + stale."""
    entries: dict[str, dict[str, Any]] = {}
    stale: list[dict[str, Any]] = []
    for key, raw in numbers.items():
        if key == STALE_KEY:
            stale = list(raw) if isinstance(raw, list) else []
            continue
        if key.startswith("@"):
            continue
        if isinstance(raw, NumberRef):
            entries[key] = {
                "value": raw.value,
                "run_id": raw.run_id,
                "manifest_sha256": raw.manifest_sha256,
                "meta": {},
            }
        elif isinstance(raw, Mapping):
            entry = dict(raw)
            entry.setdefault("meta", {})
            entries[key] = entry
        else:
            raise TypeError(f"numbers[{key!r}] must be a NumberRef or a mapping, got {type(raw)}")
    return entries, stale


class NumberView:
    """Template-facing accessors: markers, CIs and provenance captions for published keys."""

    def __init__(self, entries: Mapping[str, Mapping[str, Any]]) -> None:
        self.entries = entries

    def has(self, key: str) -> bool:
        return key in self.entries

    def value(self, key: str, default: float | None = None) -> float | None:
        entry = self.entries.get(key)
        return float(entry["value"]) if entry is not None else default

    def meta(self, key: str, *path: str, default: Any = None) -> Any:
        node: Any = self.entries.get(key, {}).get("meta", {})
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def marker(self, key: str, fmt: str | None = None, missing: str = PENDING) -> str:
        entry = self.entries.get(key)
        text = format_value(float(entry["value"]), fmt) if entry is not None else missing
        return num_marker(key, text)

    def ci(self, key: str) -> str:
        if key not in self.entries:
            return ""
        ci = self.meta(key, "ci95")
        if isinstance(ci, list | tuple) and len(ci) == 2:
            return f"[{format_value(float(ci[0]))}, {format_value(float(ci[1]))}]"
        reason = self.meta(key, "ci95_reason", default="no CI given")
        return f"(no CI: {reason})"

    def cell(self, key: str, fmt: str | None = None, missing: str = PENDING) -> str:
        ci = self.ci(key)
        return f"{self.marker(key, fmt, missing)} {ci}".rstrip()

    def prov(self, key: str) -> str:
        """Caption fragment demanded by gate A3 (reference, E0 source, head, n, seed, CI)."""
        ref = self.meta(key, "reference", default={}) or {}
        code = ref.get("code", "?") if isinstance(ref, Mapping) else "?"
        functional = ref.get("functional", "?") if isinstance(ref, Mapping) else "?"
        pseudos = ref.get("pseudos") if isinstance(ref, Mapping) else None
        parts = [
            f"reference {code}/{functional}" + (f" ({pseudos})" if pseudos else ""),
            f"E0 {self.meta(key, 'e0_source', default='?')}",
            f"head {self.meta(key, 'head', default='?')}",
            f"n = {self.meta(key, 'n', default='?')}",
            f"seed {self.meta(key, 'seed', default='?')}",
            f"CI95 {self.ci(key) or '?'}",
        ]
        floor = self.meta(key, "noise_floor_f")
        if floor is not None:
            parts.append(f"noise floor {format_value(float(floor))} meV/Å")
        if self.meta(key, "cross_functional") or (
            isinstance(ref, Mapping) and ref.get("cross_functional")
        ):
            parts.append("cross-functional column")
        if self.meta(key, "lower_fidelity"):
            parts.append("lower fidelity")
        return "; ".join(parts) + f"; run `{self.entries[key].get('run_id', '?')}`"


# --- tables ---------------------------------------------------------------------------------------


@dataclass
class Row:
    label: str
    keys: list[str | None]


@dataclass
class Table:
    name: str
    namespace: str
    header: list[str]
    caption: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    pending_rows: list[str] = field(default_factory=list)
    prov: list[str] = field(default_factory=list)
    present: bool = False


def make_table(
    view: NumberView,
    name: str,
    namespace: str,
    header: list[str],
    caption: str,
    rows: list[Row],
    *,
    missing: dict[str, str] | None = None,
    compact: bool = False,
) -> Table:
    """A cell whose key is absent renders ``pending`` (or ``missing[key]``); with ``compact``,
    rows without any published number are listed by label instead of rendered."""
    table = Table(name=name, namespace=namespace, header=header, caption=caption)
    for row in rows:
        cells: list[str] = []
        lower_fidelity = False
        row_present = False
        for col, key in zip(header[1:], row.keys, strict=True):
            if key is None:
                cells.append("—")
                continue
            cells.append(view.cell(key, missing=(missing or {}).get(key, PENDING)))
            if view.has(key):
                row_present = True
                table.prov.append(f"**{row.label} / {col}** — {view.prov(key)}")
                lower_fidelity = lower_fidelity or bool(view.meta(key, "lower_fidelity"))
        table.present = table.present or row_present
        if compact and not row_present:
            table.pending_rows.append(row.label)
            continue
        label = row.label + (" (lower fidelity)" if lower_fidelity else "")
        table.rows.append({"label": label, "cells": cells})
    return table


def _tier_rows(metric: str) -> list[Row]:
    """One row per tier so that every row cites a single reference code + functional (A5)."""
    return [
        Row(label, [f"eval.errors.{tier}.{bracket}.{metric}" for bracket, _ in BRACKETS])
        for tier, label in TIERS
    ]


def build_tables(view: NumberView) -> dict[str, Table]:
    bracket_cols = [label for _, label in BRACKETS]
    tables: dict[str, Table] = {}
    tables["forces"] = make_table(
        view,
        "forces",
        "eval.errors",
        ["Tier", *bracket_cols],
        "Force MAE in meV/Å over all force components with 95 % bootstrap CIs (2,000 resamples "
        "over groups). Each tier row cites one reference code + functional (T0–T2 the project's "
        "QE PBE, T3 OMat24 VASP PBE, T4a MPtrj VASP PBE; the provenance list says which). "
        "T3 cells carry the cross-code noise floor.",
        _tier_rows("mae_f"),
    )
    tables["energies"] = make_table(
        view,
        "energies",
        "eval.errors",
        ["Tier", *bracket_cols],
        "Energy MAE in meV/atom on the reference's own energy scale; B1 energies are on the QE "
        "scale (E0s from isolated-atom QE), B0/B0′/B2 `pt_head` numbers on the MP scale.",
        _tier_rows("mae_e"),
    )
    b1_scale_na = (view.value("offsets.residual_meV_atom") or 0.0) > OFFSET_GATE_MEV_ATOM
    missing = {f"eval.discovery.B1.{m}": "n/a (scale)" for m in ("delta_f1", "mae_e_above_hull")}
    discovery_metrics = ("delta_f1", "mae_e_above_hull", "rmsd")
    tables["discovery"] = make_table(
        view,
        "discovery",
        "eval.discovery",
        ["Model", "paired ΔF1 vs B0", "e_above_hull MAE (meV/atom)", "RMSD (Å)"],
        "Labelled, seeded 1,000-structure WBM sample at natural prevalence (16.7 % stable), "
        "vendored Matbench-Discovery metrics; F1 only as a paired difference vs B0 on the "
        "identical sample with a bootstrap CI. No public ranking is claimed or comparable.",
        [
            Row(label, [f"eval.discovery.{b}.{m}" for m in discovery_metrics])
            for b, label in BRACKETS
            if b != "B0p"
        ],
        missing=missing if b1_scale_na else None,
    )
    phonon_metrics = ("omega_mae_meV", "softening_index", "imaginary_count", "omega_mae_meV_pbesol")
    phonon_rows = [
        Row(f"{compound} / {b}", [f"eval.phonons.{compound}.{b}.{m}" for m in phonon_metrics])
        for compound in (*COMPOUNDS, "phononDB103")
        for b, _ in BRACKETS
        if b != "B0p"
    ]
    tables["phonons"] = make_table(
        view,
        "phonons",
        "eval.phonons",
        [
            "Compound / model",
            "ω-MAE (meV)",
            "softening index s",
            "imaginary modes",
            "ω-MAE vs PBEsol (cross-functional)",
        ],
        "ω-MAE over sorted branches at 100 seekpath q-points, model at the DFT cell; "
        "s = median ω_model/ω_ref; imaginary = ω < −0.4 meV. B20 rows cite the project's own "
        "QE PBE; phononDB103 rows cite VASP PBE (other code, same functional). The last column "
        "is the only place a PBEsol reference appears.",
        phonon_rows,
        compact=True,
    )
    parity_ok = view.value("parity.passed") == 1.0
    engines = ["ase", "lammps"] if parity_ok else ["ase"]
    thermal_metrics = ("a_300K_A", "a_exp_A", "a_dev_pct", "alpha_per_K")
    tables["thermal"] = make_table(
        view,
        "thermal",
        "md",
        ["Compound / engine", "a(300 K) (Å)", "a experiment (Å)", "deviation (%)", "α (1/K)"],
        "NPT thermal expansion at 300 K, 64-atom cells, 2 fs steps, against experiment: every "
        "cell of a row cites reference code `experiment` (the literature lattice constant is "
        "itself a published number). Rows of a second engine appear only after the parity gate "
        "passes.",
        [
            Row(
                f"{compound} / {engine.upper()}",
                [f"md.{engine}.{compound}.{m}" for m in thermal_metrics],
            )
            for compound in COMPOUNDS
            for engine in engines
        ],
        compact=True,
    )
    tables["stability"] = make_table(
        view,
        "stability",
        "md",
        ["Compound / engine", "NVE drift (meV/atom/ps)"],
        "NVE energy drift of the same trajectories; self-consistency numbers cite reference "
        "code `mace` with the training functional.",
        [
            Row(f"{compound} / {engine.upper()}", [f"md.{engine}.{compound}.drift_meV_atom_ps"])
            for compound in COMPOUNDS
            for engine in engines
        ],
        compact=True,
    )
    tables["sampling"] = make_table(
        view,
        "sampling",
        "sampling",
        ["Compound", "umbrella ΔF (eV)", "block error (eV)", "NEB E_a (eV)"],
        "Vacancy hop at 300 K: 12 umbrella windows × 15 ps, MBAR/WHAM free energy with block "
        "error, against the NEB barrier on the same model (reference code `mace`).",
        [
            Row(
                compound,
                [
                    f"sampling.umbrella.{compound}.dF_eV",
                    f"sampling.umbrella.{compound}.dF_err_eV",
                    f"sampling.neb.{compound}.Ea_eV",
                ],
            )
            for compound in COMPOUNDS
        ],
        compact=True,
    )
    agent_metrics = (
        ("accuracy", "accuracy"),
        ("invalid_call_rate", "invalid-call rate"),
        ("dag_valid_rate", "DAG-valid order rate"),
        ("provenance_rate", "provenance completeness"),
        ("recovery_rate", "recovery from injected failures"),
        ("tokens_per_task", "tokens per task"),
        ("usd_per_task", "USD per task"),
        ("n_tasks", "tasks"),
    )
    live_rows = [
        Row(label, [f"agent.eval.{m}"])
        for m, label in agent_metrics
        if view.meta(f"agent.eval.{m}", "backend") != "mock"
    ]
    tables["agent"] = make_table(
        view,
        "agent",
        "agent.eval",
        ["Metric", "value"],
        "Twelve-task eval (`evals/agent_tasks.jsonl`, gold from `b20mlip screen`); backend and "
        "model id are in the provenance list. Mock-backend replays (CI) are never shown here.",
        live_rows,
    )
    return tables


# --- bullet and parity ----------------------------------------------------------------------------


def _first_present(view: NumberView, keys: list[str], fallback: str) -> str:
    for key in keys:
        if view.has(key):
            return key
    return fallback


def build_bullet(view: NumberView) -> str:
    """SPEC.md section 13 bullet; every placeholder is a number marker (``pending`` if absent)."""
    ft = next((b for b in FINE_TUNED if view.has(f"eval.errors.T0.{b}.mae_f")), "B1")
    phonon_compound = next(
        (c for c in ("FeSi", "MnSi", "CoSi") if view.has(f"eval.phonons.{c}.{ft}.omega_mae_meV")),
        "FeSi",
    )
    if view.has("data.n_qe_frames") or not view.has("data.n_omat24_frames"):
        data_clause = (
            f"on {view.marker('data.n_qe_frames')} in-house spin-polarised Quantum ESPRESSO "
            "frames of B20 skyrmion hosts (FeSi/MnSi/CoSi)"
        )
    else:
        data_clause = (
            f"on {view.marker('data.n_omat24_frames')} OMat24 DFT frames of the "
            "Mn–Fe–Co–Si–Ge space (forces+stress; QE round pending)"
        )
    engines = "ASE/LAMMPS" if view.value("parity.passed") == 1.0 else "ASE"
    thermal = _first_present(
        view,
        [f"md.ase.{c}.a_dev_pct" for c in ("MnSi", "FeSi", "CoSi", "FeGe")],
        "md.ase.MnSi.a_dev_pct",
    )
    umbrella = _first_present(
        view, [f"sampling.umbrella.{c}.dF_eV" for c in COMPOUNDS], "sampling.umbrella.FeSi.dF_eV"
    )
    neb = umbrella.replace("sampling.umbrella.", "sampling.neb.").replace(".dF_eV", ".Ea_eV")
    active = (
        "one committee-uncertainty active-learning round"
        if view.has("active.n_selected")
        else "an active-learning round (pending)"
    )
    delta_f1 = f"eval.discovery.{ft}.delta_f1"
    ci = view.ci(delta_f1) or "[CI pending]"
    f_before = view.marker("eval.errors.T0.B0.mae_f")
    f_after = view.marker(f"eval.errors.T0.{ft}.mae_f")
    w_before = view.marker(f"eval.phonons.{phonon_compound}.B0.omega_mae_meV")
    w_after = view.marker(f"eval.phonons.{phonon_compound}.{ft}.omega_mae_meV")
    fege = view.marker(f"eval.errors.T2.{ft}.mae_f")
    return (
        f"Fine-tuned MACE-MPA-0 (equivariant GNN) {data_clause}: held-out force MAE "
        f"{f_before}→{f_after} meV/Å, phonon ω-MAE vs same-code DFT {w_before}→{w_after} meV "
        f"({phonon_compound}), never-trained FeGe {fege} meV/Å; forgetting quantified as paired "
        f"ΔF1 = {view.marker(delta_f1)} {ci} on a labelled 1,000-structure WBM sample "
        f"(Matbench-Discovery protocol, no public ranking claimed); deployed in {engines} MD "
        f"(thermal expansion within {view.marker(thermal)} % of experiment), umbrella-sampled a "
        f"vacancy-hop free energy (ΔF = {view.marker(umbrella)} eV vs NEB {view.marker(neb)} eV), "
        f"{active}; open-sourced (MIT) with a provenance-checked tool-calling agent."
    )


def build_parity(view: NumberView) -> dict[str, str]:
    passed = view.value("parity.passed")
    if passed is None:
        state = "not yet run"
    elif passed == 1.0:
        state = "passed"
    else:
        state = "not passed"
    return {
        "state": state,
        "max_df": view.cell("md.parity.max_dF_eVA"),
        "max_de": view.cell("md.parity.max_dE_eV_atom"),
    }


# --- rendering ------------------------------------------------------------------------------------


def build_context(numbers: Mapping[str, Any], cfg: Settings | None = None) -> dict[str, Any]:
    entries, stale = normalize(numbers)
    view = NumberView(entries)
    settings = cfg if cfg is not None else Settings.model_validate({})
    parity_ok = view.value("parity.passed") == 1.0
    return {
        "N": view,
        "numbers": entries,
        "engines": "ASE and LAMMPS" if parity_ok else "ASE",
        "parity_ok": parity_ok,
        "status": {
            "n_numbers": len(entries),
            "n_runs": len({e.get("run_id") for e in entries.values()}),
            "n_stale": len(stale),
        },
        "tables": build_tables(view),
        "parity": build_parity(view),
        "bullet": build_bullet(view),
        "protocol": {
            **PROTOCOL_CONSTANTS,
            "wbm_sample_n": settings.data.wbm_sample_n,
            "wbm_seed": settings.data.wbm_seed,
            "bootstrap_n": settings.eval.bootstrap_n,
            "noise_floor_n": settings.dft.noise_floor_n,
            "offset_gate_meV": settings.eval.offset_residual_gate_meV,
            "branch_tol_muB": settings.dft.branch_tol_muB,
            "force_cap_eVA": settings.data.force_cap_eVA,
            "k_spacing": settings.dft.k_spacing_inv_A,
            "degauss_ry": settings.dft.degauss_ry,
            "max_train_T": 600,
        },
    }


def environment(directory: str | Path | None = None) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(directory or templates_dir())),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render(
    numbers: Mapping[str, Any],
    template: str = TEMPLATE_NAME,
    *,
    cfg: Settings | None = None,
    directory: str | Path | None = None,
) -> str:
    """Render the README from ``{key: NumberRef}`` or the raw ``numbers.json`` mapping."""
    env = environment(directory)
    text = env.get_template(template).render(**build_context(numbers, cfg))
    find_gen_blocks(text)  # the template must produce well-formed blocks
    return text if text.endswith("\n") else text + "\n"


def merge_gen_blocks(existing: str, rendered: str) -> str:
    """Replace the gen blocks of ``existing`` by their namesakes in ``rendered``; append new."""
    new_blocks = {b.name: rendered[b.start : b.end] for b in find_gen_blocks(rendered)}
    out: list[str] = []
    pos = 0
    seen: set[str] = set()
    for block in find_gen_blocks(existing):
        out.append(existing[pos : block.start])
        out.append(new_blocks.get(block.name, existing[block.start : block.end]))
        seen.add(block.name)
        pos = block.end
    out.append(existing[pos:])
    text = "".join(out)
    missing = [name for name in new_blocks if name not in seen]
    if missing:
        text = text.rstrip("\n") + "\n\n" + "\n\n".join(new_blocks[name] for name in missing) + "\n"
    return text


def refresh_markers(text: str, entries: Mapping[str, Mapping[str, Any]]) -> str:
    """Outside gen blocks, re-emit every number marker whose key is published (prose stays)."""
    blocks = find_gen_blocks(text)
    out: list[str] = []
    pos = 0
    for marker in find_num_markers(text):
        if any(b.start <= marker.start < b.end for b in blocks) or marker.key not in entries:
            continue
        out.append(text[pos : marker.start])
        out.append(num_marker(marker.key, format_value(float(entries[marker.key]["value"]))))
        pos = marker.end
    out.append(text[pos:])
    return "".join(out)


def write_readme(text: str, path: str | Path, numbers: Mapping[str, Any] | None = None) -> Path:
    """Write the README, preserving hand-written prose outside gen blocks of an existing file."""
    p = Path(path)
    if p.is_file():
        existing = p.read_text(encoding="utf-8")
        try:
            has_blocks = bool(find_gen_blocks(existing))
        except MarkerError:
            has_blocks = False
        if has_blocks:
            merged = merge_gen_blocks(existing, text)
            if numbers is not None:
                merged = refresh_markers(merged, normalize(numbers)[0])
            p.write_text(merged, encoding="utf-8")
            return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --- stage function (``b20mlip report build``) ----------------------------------------------------


def run(
    cfg: Settings,
    ctx: RunContext,
    *,
    readme: bool = False,
    readme_path: str | Path = "README.md",
    template: str = TEMPLATE_NAME,
) -> StageResult:
    """Harvest ``runs/`` into ``cfg.report.numbers_path``; ``readme=True`` also writes README."""
    result: Harvest = harvest(cfg.paths.runs_dir)
    summary: dict[str, float | int | str] = {
        "n_numbers": len(result.entries),
        "n_stale": len(result.stale),
        "n_manifests_ok": result.n_ok,
        "readme": "",
    }
    ctx.log(
        n_numbers=len(result.entries),
        n_stale=len(result.stale),
        stale_keys=sorted({s["key"] for s in result.stale}),
    )
    if not ctx.dry_run:
        numbers_path = write_numbers(result, cfg.report.numbers_path, runs_dir=cfg.paths.runs_dir)
        ctx.add_output(numbers_path, "json")
        if readme:
            data = result.as_json()
            written = write_readme(render(data, template, cfg=cfg), readme_path, numbers=data)
            ctx.add_output(written, "other")
            summary["readme"] = str(written)
    return StageResult(
        stage="report",
        run_id=ctx.run_id,
        manifest_path=str(ctx.manifest_path),
        status="ok",
        outputs=list(ctx.outputs),
        summary=summary,
    )


__all__ = [
    "BRACKETS",
    "COMPOUNDS",
    "OFFSET_GATE_MEV_ATOM",
    "PENDING",
    "PROTOCOL_CONSTANTS",
    "TEMPLATE_NAME",
    "TIERS",
    "NumberView",
    "Row",
    "Table",
    "build_bullet",
    "build_context",
    "build_parity",
    "build_tables",
    "environment",
    "make_table",
    "merge_gen_blocks",
    "normalize",
    "refresh_markers",
    "render",
    "run",
    "templates_dir",
    "write_readme",
]
