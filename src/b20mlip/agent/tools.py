"""Agent tools (SPEC.md section 7): one ``@beta_tool`` per tool, 1:1 with a stage function,
returning **numbers and run ids only** as a JSON string, never raising.

Contract (binding for every tool)

* Strict JSON schema: typed arguments only (strings, numbers, booleans, lists of them or of
  small typed objects), every argument required, ``additionalProperties: false`` (the SDK
  generates the schema from the signature; ``strict=True`` is sent to the API).
* Every computation runs as its own ``agent.run`` sub-stage under
  :func:`~b20mlip.provenance.run_stage` (``runs/agent.run/<run_id>/``): the manifest hashes the
  stage outputs, ``result.json`` holds ``{"tool", "args", "numbers", "info", ...}`` and the tool
  returns ``{**numbers, **info, "run_id": ...}``. ``write_report`` and the runner's grounding
  check resolve every cited number back to that file through :func:`resolve_number`.
* Stage numbers written by the wrapped stage functions (``numbers.json`` of ``eval.errors``,
  ``active.select``) are renamed to ``stage_numbers.json`` so the report tier never harvests
  an agent-driven computation into README numbers.
* Exceptions become ``{"error": "<Type>: <message>"}`` (plus ``run_id`` when a sub-run manifest
  with ``status="failed"`` was written); the runner counts them as invalid calls.
* Configuration, model and run directories come from the module-level :class:`ToolContext`
  (``use_context`` / ``set_context``), so the decorated signatures stay model-facing.
* ``ctx.dry_run`` is honoured: every sub-run then writes a partial manifest and no numbers.

Structures are addressed by ``compound`` (the reference cell of the MPtrj B20 extract or the
``--structure`` file) or by a ``frame_id`` that ``get_structure``/``relax`` returned; datasets,
models and splits by the labels ``list_data`` reports (never by path).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from anthropic import beta_tool
from ase import Atoms
from pydantic import BaseModel, ConfigDict

from b20mlip.agent.trace import redact
from b20mlip.config import Settings
from b20mlip.dft.structures import elements_of, reference_frame
from b20mlip.io import frame_from_atoms, frame_to_atoms, read_frames, write_frames
from b20mlip.md.common import model_provenance, sanitize_key_segment
from b20mlip.models import Budget, Frame, NumberRef, PhononResult, Reference
from b20mlip.phonons import harmonic
from b20mlip.provenance import MANIFEST_NAME, RunContext, read_manifest, run_stage, sha256_file

TOOL_NAMES: tuple[str, ...] = (
    "list_data",
    "get_structure",
    "relax",
    "phonons",
    "compare_phonons",
    "evaluate_errors",
    "run_md",
    "select_frames",
    "submit_dft",
    "write_report",
)
RESULT_FILE = "result.json"
STAGE_NUMBERS_FILE = "stage_numbers.json"
PHONONS_FILE = "phonons_model.json"
COMPARE_FILE = "phonon_compare.json"
REPORT_MD = "report.md"
REPORT_JSON = "report.json"
ASR_TOL_MEV = 0.1  # |omega_acoustic(Gamma)| above this = acoustic-sum-rule violation
DataKind = Literal["compounds", "datasets", "models", "splits", "references", "all"]
Ensemble = Literal["nve", "nvt", "npt"]
TierName = Literal["T0", "T1", "T2", "T3", "T4a", "T4b"]
_ROOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_STRUCTURE_SOURCES = ("reference", "relaxed", "injected")


# --- context -----------------------------------------------------------------------------------


@dataclass
class ToolContext:
    """What the tools need beyond their arguments (set by the runner; one per agent run)."""

    cfg: Settings
    ctx: RunContext | None = None  # the parent agent run (reports are copied into its out_dir)
    model: Path | None = None  # the MLIP the agent drives (labels resolve through list_data)
    head: str = "Default"
    committee: list[Path] = field(default_factory=list)
    frames: dict[str, Path] = field(default_factory=dict)  # dataset label -> extxyz
    splits: dict[str, Path] = field(default_factory=dict)  # split label -> json
    structure_path: Path | None = None  # extxyz with the reference cells (default: MPtrj extract)
    refs_dir: Path | None = None  # phonons_<compound>_<reference>.json (default cfg.agent.refs_dir)
    budget: Budget = field(default_factory=Budget)
    seed: int = 0
    dry_run: bool = False
    runs_dir: Path | None = None
    calc: Any = None  # a ready calculator (tests share one); built lazily otherwise
    # state filled while the agent runs
    structures: dict[str, tuple[Frame, str]] = field(
        default_factory=dict
    )  # frame_id -> (frame, source)
    runs: dict[str, Path] = field(default_factory=dict)  # sub-run id -> manifest path
    results: dict[str, dict[str, Any]] = field(default_factory=dict)  # sub-run id -> numbers

    def __post_init__(self) -> None:
        if self.runs_dir is None:
            self.runs_dir = Path(self.cfg.paths.runs_dir)
        if self.refs_dir is None:
            self.refs_dir = Path(self.cfg.agent.refs_dir)
        self.model = Path(self.model) if self.model is not None else None
        self.committee = [Path(p) for p in self.committee]

    # -- model -------------------------------------------------------------------------------

    def calculator(self) -> Any:
        if self.calc is None:
            if self.model is None or not Path(self.model).is_file():
                raise FileNotFoundError("no MLIP model configured (pass --model PATH)")
            from b20mlip.evaluate.errors import make_calculator  # noqa: PLC0415 - heavy import

            with contextlib.suppress(Exception):
                import torch  # noqa: PLC0415

                torch.set_num_threads(int(self.cfg.compute.threads))
            self.calc = make_calculator(
                self.model, self.head, device=self.cfg.compute.device, dtype=self.cfg.compute.dtype
            )
        return self.calc

    def provenance(self) -> dict[str, Any]:
        if self.model is None:
            return {"label": "none", "sha256": None, "e0_source": "unknown", "energy_scale": "none"}
        return model_provenance(self.model)

    def model_label(self) -> str:
        return str(self.provenance()["label"])

    def models(self) -> dict[str, Path]:
        """Model label -> path: the primary model, the committee, ``models_dir/**/*.model``."""
        out: dict[str, Path] = {}

        def add(label: str, path: Path) -> None:
            base = sanitize_key_segment(label)
            name = base
            k = 2
            while name in out and out[name].resolve() != path.resolve():
                name = f"{base}-{k}"
                k += 1
            out.setdefault(name, path)

        if self.model is not None and self.model.is_file():
            add(self.model_label(), self.model)
        for path in self.committee:
            if path.is_file():
                add(path.stem, path)
        models_dir = Path(self.cfg.paths.models_dir)
        if models_dir.is_dir():
            for path in sorted(models_dir.rglob("*.model")):
                add(path.stem, path)
        return out

    def resolve_model(self, label: str) -> Path:
        if not label or label in ("default", "primary", "model"):
            if self.model is None:
                raise FileNotFoundError("no MLIP model configured (pass --model PATH)")
            return self.model
        models = self.models()
        if label in models:
            return models[label]
        raise KeyError(f"unknown model {label!r}; list_data('models') gives {sorted(models)}")

    # -- datasets and splits -----------------------------------------------------------------

    def datasets(self) -> dict[str, Path]:
        out: dict[str, Path] = {k: Path(v) for k, v in self.frames.items()}
        frames_dir = Path(self.cfg.paths.data_dir) / "frames"
        if frames_dir.is_dir():
            for path in sorted(frames_dir.glob("*.extxyz")):
                out.setdefault(path.stem, path)
        return out

    def resolve_frames(self, label: str) -> Path:
        datasets = self.datasets()
        if label in datasets:
            return datasets[label]
        raise KeyError(f"unknown dataset {label!r}; list_data('datasets') gives {sorted(datasets)}")

    def frames_natoms(self, label: str) -> list[int]:
        return [len(f.numbers) for f in read_frames(self.resolve_frames(label))]

    def frames_elements(self, label: str) -> list[str]:
        symbols: set[str] = set()
        for frame in read_frames(self.resolve_frames(label)):
            symbols.update(elements_of(frame.compound))
        return sorted(symbols)

    def split_files(self) -> dict[str, Path]:
        out: dict[str, Path] = {k: Path(v) for k, v in self.splits.items()}
        splits_dir = Path(self.cfg.paths.data_dir) / "splits"
        if splits_dir.is_dir():
            for path in sorted(splits_dir.glob("*.json")):
                out.setdefault(path.stem, path)
        return out

    def resolve_split(self, label: str) -> Path:
        splits = self.split_files()
        if label in splits:
            return splits[label]
        raise KeyError(f"unknown split {label!r}; list_data('splits') gives {sorted(splits)}")

    # -- references and structures -----------------------------------------------------------

    def references(self) -> dict[str, Path]:
        """``"<compound>:<label>"`` -> reference PhononResult JSON under ``refs_dir``."""
        out: dict[str, Path] = {}
        refs = Path(self.refs_dir) if self.refs_dir is not None else None
        if refs is not None and refs.is_dir():
            for path in sorted(refs.glob("phonons_*_*.json")):
                _, compound, label = path.stem.split("_", 2)
                out[f"{compound}:{label}"] = path
        return out

    def reference_path(self, compound: str, label: str) -> Path:
        refs = Path(self.refs_dir) if self.refs_dir is not None else Path(self.cfg.agent.refs_dir)
        return refs / f"phonons_{compound}_{label}.json"

    def register_structure(self, frame: Frame, source: str) -> str:
        """Remember a frame under its id; a reference cell is never re-labelled as relaxed
        (a relaxation that did not move anything keeps the reference id and its provenance)."""
        if source not in _STRUCTURE_SOURCES:
            raise ValueError(f"structure source must be one of {_STRUCTURE_SOURCES}")
        current_entry = self.structures.get(frame.frame_id)
        if current_entry is None or current_entry[1] != "reference":
            self.structures[frame.frame_id] = (frame, source)
        return frame.frame_id

    def structure(self, compound: str, frame_id: str) -> tuple[Frame, str]:
        """``(frame, source)`` for a ``frame_id`` (registered earlier) or a compound's reference."""
        if frame_id:
            if frame_id not in self.structures:
                raise KeyError(f"unknown frame_id {frame_id!r}; call get_structure(compound) first")
            return self.structures[frame_id]
        if not compound:
            raise ValueError("give a compound or a frame_id")
        frame = reference_frame(self.cfg, compound, self.structure_path)
        self.register_structure(frame, "reference")
        return frame, "reference"

    def frame_elements(self, frame_id: str) -> list[str] | None:
        entry = self.structures.get(frame_id)
        if entry is None:
            return None
        return sorted({elements_of_number(z) for z in entry[0].numbers})

    def frame_natoms(self, frame_id: str) -> int | None:
        entry = self.structures.get(frame_id)
        return None if entry is None else len(entry[0].numbers)

    # -- sub-runs -----------------------------------------------------------------------------

    def register_run(self, run_id: str, manifest_path: Path, numbers: dict[str, Any]) -> None:
        self.runs[run_id] = Path(manifest_path)
        self.results[run_id] = dict(numbers)

    def run_dir(self, run_id: str) -> Path | None:
        path = self.runs.get(run_id)
        if path is not None and path.parent.is_dir():
            return path.parent
        assert self.runs_dir is not None
        return find_run_dir(self.runs_dir, run_id)


def elements_of_number(z: int) -> str:
    from ase.data import chemical_symbols  # noqa: PLC0415

    return str(chemical_symbols[int(z)])


_CURRENT: ToolContext | None = None


def set_context(tc: ToolContext | None) -> None:
    global _CURRENT
    _CURRENT = tc


def current() -> ToolContext:
    if _CURRENT is None:
        raise RuntimeError("no ToolContext set; the agent runner must set one before tools run")
    return _CURRENT


@contextlib.contextmanager
def use_context(tc: ToolContext) -> Iterator[ToolContext]:
    previous = _CURRENT
    set_context(tc)
    try:
        yield tc
    finally:
        set_context(previous)


# --- helpers -------------------------------------------------------------------------------------


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_jsonable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _safe(fn: Callable[..., dict[str, Any]], **kw: Any) -> str:
    """Run a tool body; never raise (``{"error": ...}`` instead); always return a JSON string."""
    try:
        return _dump(fn(**kw))
    except Exception as exc:  # noqa: BLE001 - the contract: tools never raise
        return _dump({"error": f"{type(exc).__name__}: {exc}"})


def _numbers_only(data: dict[str, Any]) -> dict[str, float | int]:
    """The finite numbers of a result (bools as 0/1, integers kept integral)."""
    out: dict[str, float | int] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def _demote_stage_numbers(ctx: RunContext) -> None:
    """Rename a wrapped stage's ``numbers.json`` so the report tier never harvests it."""
    src = ctx.out_dir / "numbers.json"
    if not src.is_file():
        return
    dst = ctx.out_dir / STAGE_NUMBERS_FILE
    os.replace(src, dst)
    ctx.outputs = [a for a in ctx.outputs if Path(a.path).resolve() != src.resolve()]
    ctx.add_output(dst, "json")


def _sub_run(
    tool: str, fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]], **kw: Any
) -> dict[str, Any]:
    """Run ``fn(cfg, ctx, **kw) -> (numbers, info)`` as an ``agent.run`` sub-stage."""
    tc = current()
    parent = tc.ctx.run_id if tc.ctx is not None else None

    def stage(cfg: Settings, ctx: RunContext, **inner: Any) -> dict[str, Any]:
        ctx.log(tool=tool, parent_run_id=parent, args=redact(inner), backend_tool=True)
        if ctx.dry_run:
            return {"planned": 1, "tool": tool}
        numbers, info = fn(cfg, ctx, **inner)
        clean = _numbers_only(numbers)
        record = {
            "schema": "b20mlip.agent.result.v1",
            "tool": tool,
            "run_id": ctx.run_id,
            "parent_run_id": parent,
            "args": redact(inner),
            "numbers": clean,
            "info": json.loads(_dump(info)),
        }
        path = ctx.out_dir / RESULT_FILE
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ctx.add_output(path, "json")
        _demote_stage_numbers(ctx)
        return {**clean, "tool": tool}

    res = run_stage(
        "agent.run",
        tc.cfg,
        stage,
        seed=tc.seed,
        runs_dir=tc.runs_dir,
        dry_run=tc.dry_run,
        executor=tc.ctx.executor if tc.ctx is not None else None,
        **kw,
    )
    manifest_path = Path(res.manifest_path)
    if res.status == "failed":
        tc.register_run(res.run_id, manifest_path, {})
        return {"error": str(res.summary.get("error", "stage failed")), "run_id": res.run_id}
    if res.status == "partial" or tc.dry_run:
        tc.register_run(res.run_id, manifest_path, {})
        return {"planned": 1, "run_id": res.run_id, "status": "partial"}
    record = json.loads((manifest_path.parent / RESULT_FILE).read_text(encoding="utf-8"))
    numbers = dict(record["numbers"])
    tc.register_run(res.run_id, manifest_path, numbers)
    return {**numbers, **record.get("info", {}), "run_id": res.run_id, "status": "ok"}


def _atoms_of(frame: Frame) -> Atoms:
    atoms = frame_to_atoms(frame)
    atoms.calc = None
    atoms.info = {}
    return atoms


def _cubic_a(atoms: Atoms) -> float:
    return float(atoms.get_volume()) ** (1.0 / 3.0)


# --- number resolution (write_report, runner grounding) ------------------------------------------


def find_run_dir(runs_dir: str | Path, run_id: str) -> Path | None:
    """``runs/<stage>/<run_id>`` for any stage holding ``run_id`` (run ids are unique per stage)."""
    root = Path(runs_dir)
    if not root.is_dir():
        return None
    for stage_dir in sorted(root.iterdir()):
        candidate = stage_dir / run_id
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    return None


def resolve_number(
    runs_dir: str | Path,
    key: str,
    value: float,
    run_id: str,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> NumberRef:
    """The :class:`NumberRef` of ``(key, value)`` in sub-run ``run_id``, or ``ValueError``.

    The run must have an ``ok`` manifest whose ``result.json`` output still matches its recorded
    sha256 and holds ``key`` with a value within tolerance (a key the run does not know is
    matched by value against every number it produced).
    """
    run_dir = find_run_dir(runs_dir, run_id)
    if run_dir is None:
        raise ValueError(f"{key}: run {run_id!r} has no manifest under {runs_dir}")
    manifest = read_manifest(run_dir)
    if manifest.status != "ok":
        raise ValueError(f"{key}: run {run_id!r} has status {manifest.status!r}, not ok")
    art = next((a for a in manifest.outputs if Path(a.path).name == RESULT_FILE), None)
    if art is None:
        raise ValueError(f"{key}: run {run_id!r} produced no {RESULT_FILE}")
    path = Path(art.path)
    if not path.is_file():
        path = run_dir / RESULT_FILE
    if not path.is_file() or sha256_file(path) != art.sha256:
        raise ValueError(f"{key}: {RESULT_FILE} of run {run_id!r} is missing or changed")
    numbers = json.loads(path.read_text(encoding="utf-8")).get("numbers", {})
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: value {value!r} is not a number") from exc
    if not math.isfinite(target):
        raise ValueError(f"{key}: value {value!r} is not finite")
    if key in numbers:
        if not math.isclose(float(numbers[key]), target, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(f"{key}={target} does not match {numbers[key]} in run {run_id!r}")
    elif not any(
        math.isclose(float(v), target, rel_tol=rel_tol, abs_tol=abs_tol) for v in numbers.values()
    ):
        raise ValueError(f"{key}={target} is not a number produced by run {run_id!r}")
    return NumberRef(
        key=key, value=target, run_id=run_id, manifest_sha256=sha256_file(run_dir / MANIFEST_NAME)
    )


# --- phonons without / with the acoustic sum rule ------------------------------------------------


def compute_phonons(
    atoms: Atoms,
    calc: Any,
    supercell: Sequence[int],
    distance: float,
    *,
    asr: bool = True,
    cell_source: str = "dft",
    npoints: int = harmonic.DEFAULT_NPOINTS,
    source_label: str = "mace",
    run_id: str = "",
) -> tuple[PhononResult, float]:
    """:func:`harmonic.compute` with an optional acoustic-sum-rule enforcement switch.

    Returns the result and ``max |omega|`` (meV) of the three acoustic branches at Gamma; with
    ``asr=False`` the force constants are used as produced, so the acoustic modes at Gamma do
    not vanish and the result carries the raw translational-invariance error.
    """
    phonon = harmonic.new_phonopy(atoms, list(supercell))
    phonon.generate_displacements(distance=float(distance))
    forces = []
    for cell in phonon.supercells_with_displacements or []:
        sc = harmonic.from_phonopy(cell)
        sc.calc = calc
        forces.append(np.asarray(sc.get_forces(), dtype=float))
    phonon.forces = forces
    phonon.produce_force_constants()
    if asr:
        harmonic.enforce_acoustic_sum_rule(phonon)
    qpoints, labels, freqs = harmonic.band_structure(phonon, "seekpath", npoints)
    result = PhononResult(
        compound=harmonic.compound_of(atoms),
        source_label=source_label,
        reference=Reference(code="mace", functional=None, pseudos=None, e0_source=None),
        supercell=harmonic.supercell_field(phonon.supercell_matrix),
        displacement=float(distance),
        cell_source=cast(Any, cell_source),
        qpath_labels=labels,
        qpoints=qpoints,
        frequencies_meV=freqs,
        dos_meV=None,
        dos=None,
        imaginary_count=harmonic.imaginary_count(freqs),
        softening_index=None,
        omega_mae_meV=None,
        run_id=run_id,
    )
    gamma = harmonic.gamma_frequencies(result)
    acoustic = sorted(gamma, key=abs)[:3]
    gamma_max = max((abs(w) for w in acoustic), default=0.0)
    return result, float(gamma_max)


def write_phonon_reference(
    cfg: Settings,
    compound: str,
    calc: Any,
    out_dir: str | Path,
    *,
    label: str = "qe",
    supercell: Sequence[int] | None = None,
    distance: float | None = None,
    structure: str | Path | None = None,
    source_label: str = "synthetic",
    reference: Reference | None = None,
) -> Path:
    """Write ``phonons_<compound>_<label>.json`` from an ASE calculator at the reference cell.

    This is the **test path** (a synthetic reference from the tiny model); production references
    come from the dft tier's QE force sets (``harmonic.from_force_sets``, see evals/README.md).
    """
    atoms = _atoms_of(reference_frame(cfg, compound, structure))
    result, _ = compute_phonons(
        atoms,
        calc,
        supercell or list(cfg.agent.phonon_supercell),
        distance if distance is not None else cfg.agent.phonon_distance,
        source_label=source_label,
    )
    if reference is not None:
        result = result.model_copy(update={"reference": reference})
    path = Path(out_dir) / f"phonons_{compound}_{label}.json"
    harmonic.to_json(result, path)
    return path


# --- tool bodies --------------------------------------------------------------------------------


def _list_data(kind: str) -> dict[str, Any]:
    tc = current()

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        info: dict[str, Any] = {"kind": kind}
        if kind in ("compounds", "all"):
            info["compounds"] = list(cfg.data.compounds)
        if kind in ("datasets", "all"):
            info["datasets"] = sorted(tc.datasets())
        if kind in ("models", "all"):
            info["models"] = sorted(tc.models())
            info["primary_model"] = tc.model_label() if tc.model is not None else ""
        if kind in ("splits", "all"):
            info["splits"] = sorted(tc.split_files())
        if kind in ("references", "all"):
            info["references"] = sorted(tc.references())
        numbers = {
            f"n_{name}": len(info[name])
            for name in ("compounds", "datasets", "models", "splits", "references")
            if name in info
        }
        return numbers, info

    return _sub_run("list_data", fn, kind=kind)


def _get_structure(compound: str) -> dict[str, Any]:
    tc = current()
    elements_of(compound)  # validates the symbols
    frame, source = tc.structure(compound, "")

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.data import spacegroup_number  # noqa: PLC0415 - spglib import

        atoms = _atoms_of(frame)
        try:
            spacegroup = int(spacegroup_number(atoms))
        except Exception:  # noqa: BLE001 - symmetry search is informational
            spacegroup = 0
        numbers = {
            "natoms": len(frame.numbers),
            "a_A": _cubic_a(atoms),
            "volume_A3": float(atoms.get_volume()),
            "spacegroup": spacegroup,
        }
        info = {
            "compound": compound,
            "formula": atoms.get_chemical_formula("metal", empirical=True),
            "elements": sorted({elements_of_number(z) for z in frame.numbers}),
            "frame_id": frame.frame_id,
            "source": source,
        }
        ctx.log(compound=compound, frame_id=frame.frame_id)
        return numbers, info

    return _sub_run("get_structure", fn, compound=compound)


def _relax(compound: str, frame_id: str, fmax: float, steps: int) -> dict[str, Any]:
    tc = current()
    frame, _ = tc.structure(compound, frame_id)
    fmax = float(fmax) if fmax > 0 else float(tc.cfg.eval.fmax)
    steps = int(steps) if steps > 0 else int(tc.cfg.eval.max_steps)

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.evaluate.discovery import max_force, relax  # noqa: PLC0415

        atoms = _atoms_of(frame)
        relaxed, n = relax(atoms, tc.calculator(), fmax=fmax, steps=steps, fix_symmetry=True)
        energy = float(relaxed.get_potential_energy())
        f_final = float(max_force(relaxed))
        relaxed.set_constraint()
        out = frame_from_atoms(
            relaxed,
            group_id=frame.group_id,
            compound=frame.compound,
            config_type="relax",
            parent_id=frame.frame_id,
            label_source="mace_zero_shot",
        )
        write_frames([out], ctx.out_dir / "relaxed.extxyz")
        ctx.add_output(ctx.out_dir / "relaxed.extxyz", "frames")
        tc.register_structure(out, "relaxed")
        numbers = {
            "energy_eV": energy,
            "energy_eV_atom": energy / len(relaxed),
            "a_A": _cubic_a(relaxed),
            "volume_A3": float(relaxed.get_volume()),
            "steps": int(n),
            "fmax_final_eVA": f_final,
            "converged": f_final <= fmax,
            "natoms": len(relaxed),
        }
        info = {"compound": frame.compound, "frame_id": out.frame_id, "fmax": fmax}
        ctx.log(
            compound=frame.compound,
            fmax=fmax,
            max_steps=steps,
            model_sha256=tc.provenance()["sha256"],
        )
        return numbers, info

    return _sub_run(
        "relax", fn, compound=frame.compound, frame_id=frame.frame_id, fmax=fmax, steps=steps
    )


def _phonons(
    compound: str, frame_id: str, supercell: list[int], distance: float, asr: bool
) -> dict[str, Any]:
    tc = current()
    frame, source = tc.structure(compound, frame_id)
    cell = [int(x) for x in supercell] if supercell else list(tc.cfg.agent.phonon_supercell)
    if len(cell) != 3 or any(x < 1 for x in cell):
        raise ValueError(f"supercell must be three positive integers, got {supercell}")
    dist = float(distance) if distance > 0 else float(tc.cfg.agent.phonon_distance)
    cell_source = "model_relaxed" if source == "relaxed" else "dft"

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        atoms = _atoms_of(frame)
        result, gamma_max = compute_phonons(
            atoms, tc.calculator(), cell, dist, asr=bool(asr), cell_source=cell_source,
            source_label=tc.model_label(), run_id=ctx.run_id,
        )  # fmt: skip
        harmonic.to_json(result, ctx.out_dir / PHONONS_FILE)
        ctx.add_output(ctx.out_dir / PHONONS_FILE, "json")
        freqs = np.asarray(result.frequencies_meV, dtype=float)
        numbers = {
            "imaginary_count": int(result.imaginary_count),
            "omega_min_meV": float(freqs.min()),
            "omega_max_meV": float(freqs.max()),
            "gamma_acoustic_max_abs_meV": gamma_max,
            "asr_applied": bool(asr),
            "asr_violation": gamma_max > ASR_TOL_MEV,
            "n_qpoints": int(freqs.shape[0]),
            "n_branches": int(freqs.shape[1]),
            "natoms_supercell": len(frame.numbers) * int(np.prod(cell)),
        }
        info = {"compound": frame.compound, "cell_source": cell_source, "supercell": cell}
        ctx.log(
            compound=frame.compound, supercell=cell, distance=dist, asr=bool(asr),
            cell_source=cell_source, model_sha256=tc.provenance()["sha256"],
        )  # fmt: skip
        return numbers, info

    return _sub_run(
        "phonons", fn, compound=frame.compound, frame_id=frame.frame_id, supercell=cell,
        distance=dist, asr=bool(asr),
    )  # fmt: skip


def _compare_phonons(run_id_model: str, reference: str) -> dict[str, Any]:
    tc = current()
    run_dir = tc.run_dir(run_id_model)
    if run_dir is None or not (run_dir / PHONONS_FILE).is_file():
        raise KeyError(f"{run_id_model!r} is not the run_id of an earlier phonons call")
    model = harmonic.from_json(run_dir / PHONONS_FILE)
    label = reference or "qe"
    ref_path = tc.reference_path(model.compound, label)
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"no {label!r} phonon reference for {model.compound} (expected {ref_path.name} under "
            f"the references directory; list_data('references') lists what exists)"
        )

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.evaluate import phonon_compare  # noqa: PLC0415

        ctx.add_input(ref_path, "json")
        ctx.add_input(run_dir / PHONONS_FILE, "json")
        ref = harmonic.from_json(ref_path)
        result = phonon_compare.summary(model, ref)
        compared = phonon_compare.compare(model, ref)
        harmonic.to_json(compared, ctx.out_dir / PHONONS_FILE)
        ctx.add_output(ctx.out_dir / PHONONS_FILE, "json")
        phonon_compare.write_summary(result, ctx.out_dir / COMPARE_FILE)
        ctx.add_output(ctx.out_dir / COMPARE_FILE, "json")
        numbers = {
            "omega_mae_meV": float(result["omega_mae_meV"]),
            "softening_index": float(result["softening_index"]),
            "omega_max_abs_delta_meV": float(result["omega_max_abs_delta_meV"]),
            "imaginary_count_model": int(result["imaginary_count_model"]),
            "imaginary_count_reference": int(result["imaginary_count_reference"]),
            "imaginary_delta": int(result["imaginary_delta"]),
            "n_qpoints": int(result["n_qpoints"]),
        }
        info = {
            "compound": model.compound,
            "reference": label,
            "reference_code": ref.reference.code,
            "reference_functional": ref.reference.functional,
            "model_run_id": run_id_model,
        }
        ctx.log(compound=model.compound, reference=ref.reference.model_dump(mode="json"))
        return numbers, info

    return _sub_run("compare_phonons", fn, run_id_model=run_id_model, reference=label)


def _evaluate_errors(model: str, frames: str, split: str, tier: str) -> dict[str, Any]:
    tc = current()
    model_path = tc.resolve_model(model)
    frames_path = tc.resolve_frames(frames) if frames else None
    split_path = tc.resolve_split(split) if split else None
    if frames_path is None and split_path is None:
        raise ValueError("evaluate_errors needs a dataset label (frames) and/or a split label")
    prov = model_provenance(model_path)

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.evaluate import errors  # noqa: PLC0415

        scale = prov.get("energy_scale")
        summary = errors.run(
            cfg, ctx, model=model_path, head=tc.head, split=split_path,
            tiers=[tier] if split_path is not None else None,
            frames_path=frames_path, tier=tier,
            energy_scale=None if scale in (None, "none") else str(scale),
            label=str(prov["label"]), e0_source=str(prov.get("e0_source") or "unspecified"),
            seed=tc.seed,
        )  # fmt: skip
        table_path = ctx.out_dir / errors.TABLE_FILE.format(tier=tier)
        table = json.loads(table_path.read_text(encoding="utf-8"))
        numbers: dict[str, Any] = {
            "n_frames": int(table["n_frames"]),
            "n_atoms": int(table["n_atoms"]),
        }
        for name, metric in table["metrics"].items():
            numbers[name] = float(metric["value"])
            if metric.get("ci95"):
                numbers[f"{name}_ci95_lo"] = float(metric["ci95"][0])
                numbers[f"{name}_ci95_hi"] = float(metric["ci95"][1])
        info = {
            "model": str(prov["label"]),
            "tier": tier,
            "energy_metrics": summary.get(f"{tier}.energy_metrics", ""),
            "units": {k: v for k, v in errors.UNITS.items() if k in table["metrics"]},
        }
        return numbers, info

    return _sub_run(
        "evaluate_errors", fn, model=str(prov["label"]), frames=frames, split=split, tier=tier
    )


def _run_md(
    compound: str, frame_id: str, ensemble: str, T: float, ps: float, natoms: int
) -> dict[str, Any]:
    tc = current()
    frame, _ = tc.structure(compound, frame_id)
    target = int(natoms) if natoms > 0 else int(tc.cfg.md.natoms)
    if T <= 0 or ps <= 0:
        raise ValueError("T (K) and ps must be positive")

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.md import ase_md  # noqa: PLC0415

        stats: dict[str, Any] = {}
        prov = tc.provenance()
        res = ase_md.run(
            _atoms_of(frame), tc.calculator(), ensemble, float(T), float(ps), cfg.md.timestep_fs,
            cfg, ctx, natoms=target, seed=tc.seed, head=cast(Any, tc.head), compound=frame.compound,
            model_sha256=prov.get("sha256") or "", stats=stats,
        )  # fmt: skip
        (ctx.out_dir / "md_result.json").write_text(
            res.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        ctx.add_output(ctx.out_dir / "md_result.json", "json")
        numbers: dict[str, Any] = {
            "T_mean_K": float(stats["T_mean_K"]),
            "E_pot_mean_eV_atom": float(stats["E_pot_mean_eV_atom"]),
            "steps": int(res.steps),
            "natoms": int(res.natoms),
            "timestep_fs": float(res.timestep_fs),
            "n_production": int(stats["n_production"]),
        }
        if res.drift_meV_atom_ps is not None:
            numbers["drift_meV_atom_ps"] = float(res.drift_meV_atom_ps)
        if res.a_mean_A is not None:
            numbers["a_mean_A"] = float(res.a_mean_A)
            numbers["a_std_A"] = float(res.a_std_A or 0.0)
        info = {
            "compound": frame.compound,
            "ensemble": ensemble,
            "T_K": float(T),
            "ps": float(ps),
            "drift_meV_atom_ps": res.drift_meV_atom_ps,
            "production_window": stats.get("production_window"),
        }
        ctx.log(
            engine="ase",
            timestep_fs=cfg.md.timestep_fs,
            thermostat=ase_md.thermostat_label(ensemble, cfg),
        )
        return numbers, info

    return _sub_run(
        "run_md", fn, compound=frame.compound, frame_id=frame.frame_id, ensemble=ensemble,
        T=float(T), ps=float(ps), natoms=target,
    )  # fmt: skip


def _select_frames(models: list[str], frames: str, n: int) -> dict[str, Any]:
    tc = current()
    paths = [tc.resolve_model(m) for m in models]
    if len(paths) < 2:
        raise ValueError("select_frames needs at least two committee model labels")
    frames_path = tc.resolve_frames(frames)
    if n < 1:
        raise ValueError("n must be >= 1")

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.active import committee  # noqa: PLC0415

        summary = committee.run(
            cfg, ctx, models=paths, frames_path=frames_path, n=int(n),
            out=ctx.out_dir / "candidates.extxyz", head=tc.head,
        )  # fmt: skip
        numbers = {
            "n_selected": int(summary["n_selected"]),
            "n_candidates": int(summary["n_candidates"]),
            "n_models": int(summary["n_models"]),
            "sigma_f_median": float(summary["sigma_f_median"]),
            "sigma_f_max": float(summary["sigma_f_max"]),
        }
        info = {"dataset": frames, "models": list(models), "unit": "eV/A"}
        return numbers, info

    return _sub_run("select_frames", fn, models=list(models), frames=frames, n=int(n))


def _submit_dft(frames: str, root: str, n_frames: int, submit: bool) -> dict[str, Any]:
    tc = current()
    from b20mlip.agent.guard import node_hours_estimate  # noqa: PLC0415 - avoids a cycle

    frames_path = tc.resolve_frames(frames)
    if not _ROOT_RE.match(root or ""):
        raise ValueError("root must be a plain directory name (letters, digits, '_', '-', '.')")
    if submit and not tc.budget.approve_cluster:
        raise PermissionError(
            "cluster submission is not approved (run with --approve-cluster); planned only"
        )
    root_dir = Path(tc.cfg.paths.dft_dir) / root

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from b20mlip.dft import qe, stages  # noqa: PLC0415

        loaded = read_frames(frames_path)
        if n_frames > 0:
            loaded = loaded[: int(n_frames)]
        if not loaded:
            raise ValueError(f"no frames in dataset {frames!r}")
        subset = ctx.out_dir / "dft_frames.extxyz"
        write_frames(loaded, subset)
        ctx.add_output(subset, "frames")
        plan = stages.prep(cfg, ctx, frames=subset, out=root_dir)
        pending = qe.pending_units(root_dir)
        spec = qe.job_spec(root_dir, pending, cfg)
        (ctx.out_dir / "spec.json").write_text(
            spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        ctx.add_output(ctx.out_dir / "spec.json", "json")
        natoms = [len(f.numbers) for f in loaded]
        hours = node_hours_estimate(natoms)
        submitted = False
        if submit and tc.budget.approve_cluster:
            res = stages.run(cfg, ctx, units=root_dir, wait=True)
            if res.status != "ok":
                raise RuntimeError(f"dft run finished with status {res.status!r}")
            submitted = True
        numbers = {
            "n_units": int(plan["n_units"]),
            "n_pending": len(pending),
            "node_hours_estimate": hours,
            "natoms_total": int(sum(natoms)),
            "submitted": submitted,
        }
        info = {"dataset": frames, "root": root, "executor": "cluster" if submitted else "planned"}
        ctx.log(dft_root=str(root_dir), submitted=submitted, node_hours_estimate=hours)
        return numbers, info

    return _sub_run(
        "submit_dft", fn, frames=frames, root=root, n_frames=int(n_frames), submit=bool(submit)
    )


class NumberItem(BaseModel):
    """One cited number: the key and value a tool returned and that tool call's ``run_id``."""

    model_config = ConfigDict(extra="forbid")

    key: str
    value: float
    run_id: str


def _write_report(title: str, numbers: list[NumberItem], text: str) -> dict[str, Any]:
    tc = current()
    assert tc.runs_dir is not None
    if not numbers:
        raise ValueError("a report must cite at least one number")
    items = [NumberItem.model_validate(x) if isinstance(x, dict) else x for x in numbers]
    refs: list[NumberRef] = []
    rejected: list[str] = []
    for item in items:
        try:
            refs.append(resolve_number(tc.runs_dir, item.key, item.value, item.run_id))
        except ValueError as exc:
            rejected.append(str(exc))
    if rejected:
        return {
            "error": f"report rejected: {len(rejected)} number(s) do not resolve to a run",
            "rejected": rejected,
        }
    clean_title = redact(title)
    clean_text = redact(text)

    def fn(cfg: Settings, ctx: RunContext, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        lines = [
            f"# {clean_title}",
            "",
            clean_text.strip(),
            "",
            "| key | value | run_id | manifest_sha256 |",
            "|---|---|---|---|",
        ]
        for ref in refs:
            lines.append(
                f"| {ref.key} | {ref.value:.6g} | {ref.run_id} | {ref.manifest_sha256[:12]} |"
            )
        md_path = ctx.out_dir / REPORT_MD
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        payload = {
            "title": clean_title,
            "text": clean_text,
            "numbers": [r.model_dump(mode="json") for r in refs],
            "parent_run_id": tc.ctx.run_id if tc.ctx is not None else None,
        }
        json_path = ctx.out_dir / REPORT_JSON
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        ctx.add_output(md_path, "other")
        ctx.add_output(json_path, "json")
        if tc.ctx is not None:
            for src in (md_path, json_path):
                dst = tc.ctx.out_dir / src.name
                shutil.copyfile(src, dst)
                tc.ctx.add_output(dst, "json" if dst.suffix == ".json" else "other")
        numbers_out = {"n_numbers": len(refs), "report_bytes": md_path.stat().st_size}
        info = {"report_sha256": sha256_file(md_path), "title": clean_title}
        return numbers_out, info

    return _sub_run(
        "write_report", fn, title=clean_title, n_numbers=len(refs),
        keys=[r.key for r in refs],
    )  # fmt: skip


# --- the decorated tools ------------------------------------------------------------------------


@beta_tool(strict=True)
def list_data(kind: DataKind) -> str:
    """List what is available: compounds, datasets, models, splits and phonon references.

    Call this before evaluate_errors / select_frames / submit_dft to learn the labels those tools
    take (labels, never paths). Returns the labels and their counts.

    Args:
        kind: "compounds" | "datasets" | "models" | "splits" | "references" | "all".
    """
    return _safe(_list_data, kind=kind)


@beta_tool(strict=True)
def get_structure(compound: str) -> str:
    """Reference (relaxed DFT) B20 cell of a compound and its frame_id.

    Returns formula, elements, natoms, the lattice parameter a_A (A), the space group and the
    frame_id to pass to relax / phonons / run_md. Always call this before relax, phonons or
    run_md on a compound. Only Fe, Mn, Co, Si and Ge are allowed: a structure holding any other
    element must be discarded.

    Args:
        compound: B20 compound name, e.g. FeSi, CoSi, MnSi or FeGe.
    """
    return _safe(_get_structure, compound=compound)


@beta_tool(strict=True)
def relax(compound: str, frame_id: str, fmax: float, steps: int) -> str:
    """Relax a cell with the configured MLIP (FIRE, positions + cell, space group kept).

    Returns energy_eV, energy_eV_atom, the lattice parameter a_A, steps, the final maximum force
    (fmax_final_eVA), converged (0/1), the relaxed frame_id and the run_id.

    Args:
        compound: B20 compound name (used when frame_id is empty).
        frame_id: frame id from get_structure (or an earlier relax); "" = the reference cell of
            compound.
        fmax: force convergence criterion in eV/A; 0 = the configured default (0.05).
        steps: maximum optimizer steps; 0 = the configured default (500).
    """
    return _safe(_relax, compound=compound, frame_id=frame_id, fmax=fmax, steps=steps)


@beta_tool(strict=True)
def phonons(compound: str, frame_id: str, supercell: list[int], distance: float, asr: bool) -> str:
    """Harmonic phonons of a cell with the configured MLIP (phonopy displacements, seekpath path).

    Returns imaginary_count, omega_min_meV, omega_max_meV, the acoustic-sum-rule check
    (asr_applied, asr_violation, gamma_acoustic_max_abs_meV) and the run_id. If the result says
    asr_violation = true the force constants broke translational invariance: re-run with
    asr = true before using its numbers. Pass the run_id to compare_phonons for the omega-MAE and
    the softening index against a reference.

    Args:
        compound: B20 compound name (used when frame_id is empty).
        frame_id: frame id from get_structure (DFT cell) or relax (model-relaxed cell); "" = the
            reference cell of compound.
        supercell: three integers, e.g. [2, 2, 2]; [] = the configured default.
        distance: finite displacement in A; 0 = the configured default (0.03).
        asr: enforce the acoustic sum rule (true unless you are diagnosing a violation).
    """
    return _safe(
        _phonons,
        compound=compound,
        frame_id=frame_id,
        supercell=supercell,
        distance=distance,
        asr=asr,
    )


@beta_tool(strict=True)
def compare_phonons(run_id_model: str, reference: str) -> str:
    """Compare an earlier phonons run against a stored reference of the same compound.

    Returns omega_mae_meV, softening_index (median omega_model / omega_ref; below 1 = the model
    is softer than the reference), the imaginary counts of both and the run_id.

    Args:
        run_id_model: the run_id returned by phonons.
        reference: reference label, e.g. "qe" (own Quantum ESPRESSO PBE) or "phonondb103";
            list_data("references") lists what exists.
    """
    return _safe(_compare_phonons, run_id_model=run_id_model, reference=reference)


@beta_tool(strict=True)
def evaluate_errors(model: str, frames: str, split: str, tier: TierName) -> str:
    """Energy / force / stress error table of a model on labelled frames.

    Returns mae_f and rmse_f (meV/A) with bootstrap CIs (mae_f_ci95_lo/hi), mae_e (meV/atom)
    when the energy scales match, n_frames and the run_id.

    Args:
        model: model label from list_data("models"); "" = the primary model.
        frames: dataset label from list_data("datasets") holding labelled frames ("" only with
            a split).
        split: split label from list_data("splits") selecting the tier's frames; "" = evaluate
            the whole dataset as `tier`.
        tier: evaluation tier T0 | T1 | T2 | T3 | T4a | T4b.
    """
    return _safe(_evaluate_errors, model=model, frames=frames, split=split, tier=tier)


@beta_tool(strict=True)
def run_md(
    compound: str, frame_id: str, ensemble: Ensemble, T: float, ps: float, natoms: int
) -> str:
    """ASE molecular dynamics with the configured MLIP (2 fs steps).

    Returns T_mean_K, E_pot_mean_eV_atom, the NVE total-energy drift drift_meV_atom_ps (nve
    only), a_mean_A (npt only), steps, natoms and the run_id. Budget: at most 20 ps per call
    and 512 atoms.

    Args:
        compound: B20 compound name (used when frame_id is empty).
        frame_id: frame id from get_structure; "" = the reference cell of compound.
        ensemble: "nve" | "nvt" | "npt".
        T: temperature in K (the initial temperature for nve).
        ps: trajectory length in picoseconds (<= 20).
        natoms: supercell size in atoms (the cell is repeated isotropically up to this count);
            0 = the configured default.
    """
    return _safe(
        _run_md, compound=compound, frame_id=frame_id, ensemble=ensemble, T=T, ps=ps, natoms=natoms
    )


@beta_tool(strict=True)
def select_frames(models: list[str], frames: str, n: int) -> str:
    """Committee-uncertainty selection of the n frames with the largest force disagreement.

    sigma_F (eV/A) is the disagreement between the given models (at least two). Returns
    n_selected, sigma_f_median, sigma_f_max and the run_id. Run evaluate_errors first so the
    selection is informed by the error level.

    Args:
        models: two or more model labels from list_data("models").
        frames: dataset label from list_data("datasets") (the candidate pool).
        n: number of frames to select.
    """
    return _safe(_select_frames, models=models, frames=frames, n=n)


@beta_tool(strict=True)
def submit_dft(frames: str, root: str, n_frames: int, submit: bool) -> str:
    """Plan a Quantum ESPRESSO round for frames of a dataset (one pw.x unit per frame).

    Returns n_units, node_hours_estimate, submitted (0/1) and the run_id. Nothing is submitted
    unless submit = true AND the run was started with --approve-cluster; without approval the
    call is a dry plan.

    Args:
        frames: dataset label from list_data("datasets").
        root: name of the unit directory under the DFT root, e.g. "agent_r1".
        n_frames: plan only the first n frames of the dataset; 0 = all.
        submit: true to submit through the cluster executor (needs approval), false to plan.
    """
    return _safe(_submit_dft, frames=frames, root=root, n_frames=n_frames, submit=submit)


@beta_tool(strict=True)
def write_report(title: str, numbers: list[NumberItem], text: str) -> str:
    """Write the final report (report.md + report.json); call this last.

    Every cited number must be a value an earlier tool call returned together with that call's
    run_id; the report is rejected if any number does not resolve to a run manifest.

    Args:
        title: report title.
        numbers: the cited numbers as {key, value, run_id} objects (key and value exactly as
            the tool returned them).
        text: the report body (plain text or Markdown), citing run ids.
    """
    return _safe(_write_report, title=title, numbers=numbers, text=text)


TOOLS: list[Any] = [
    list_data,
    get_structure,
    relax,
    phonons,
    compare_phonons,
    evaluate_errors,
    run_md,
    select_frames,
    submit_dft,
    write_report,
]
TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in TOOLS}
assert tuple(TOOLS_BY_NAME) == TOOL_NAMES


def parse_result(text: str) -> dict[str, Any]:
    """The dict a tool returned (tools always return a JSON object string)."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("tool results must be JSON objects")
    return data


__all__ = [
    "ASR_TOL_MEV",
    "COMPARE_FILE",
    "PHONONS_FILE",
    "REPORT_JSON",
    "REPORT_MD",
    "RESULT_FILE",
    "STAGE_NUMBERS_FILE",
    "TOOLS",
    "TOOLS_BY_NAME",
    "TOOL_NAMES",
    "NumberItem",
    "ToolContext",
    "compare_phonons",
    "compute_phonons",
    "current",
    "elements_of_number",
    "evaluate_errors",
    "find_run_dir",
    "get_structure",
    "list_data",
    "parse_result",
    "phonons",
    "relax",
    "resolve_number",
    "run_md",
    "select_frames",
    "set_context",
    "submit_dft",
    "use_context",
    "write_phonon_reference",
    "write_report",
]
