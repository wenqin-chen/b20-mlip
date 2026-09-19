"""Helpers shared by the data-tier tests (importable as ``helpers_data`` under pytest)."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from ase.spacegroup import crystal

from b20mlip.io import frame_from_atoms
from b20mlip.models import Frame

# compound -> (a [A], u_TM, u_X, TM, X)
B20: dict[str, tuple[float, float, float, str, str]] = {
    "FeSi": (4.48, 0.137, 0.842, "Fe", "Si"),
    "CoSi": (4.44, 0.144, 0.843, "Co", "Si"),
    "MnSi": (4.56, 0.138, 0.846, "Mn", "Si"),
    "FeGe": (4.70, 0.135, 0.842, "Fe", "Ge"),
    "MnGe": (4.79, 0.135, 0.842, "Mn", "Ge"),
    "CoGe": (4.64, 0.140, 0.840, "Co", "Ge"),
}
REAL_MPTRJ = Path("data/raw/mptrj/mptrj.extxyz.zip")
REAL_OMAT24 = Path("data/raw/omat24/omat24_1M")
REAL_WBM_CSV = Path("data/raw/references/wbm-summary.csv.gz")
REAL_WBM_ZIP = Path("data/raw/references/wbm-initial-atoms.extxyz.zip")

# WBM mini summary rows: material_id, formula, n_sites, e_above_hull (None = NaN), e_form
WBM_ROWS: list[tuple[str, str, int, float | None, float]] = [
    ("wbm-1-1", "Fe4 Si4", 8, 0.0, -0.5),
    ("wbm-1-2", "Mn2 Ge2", 4, 0.05, -0.1),
    ("wbm-1-3", "Co1 Si1", 2, -0.01, -0.4),
    ("wbm-1-4", "Na1 Cl1", 2, 0.0, -2.0),
    ("wbm-1-5", "Ac6 U2", 8, 0.55, 0.54),
    ("wbm-2-1", "Fe2 O3", 5, 0.1, -1.5),
    ("wbm-2-2", "Ge1 Si1", 2, 0.2, 0.1),
    ("wbm-2-3", "Li1 Fe1 Ge1", 3, 0.3, 0.0),
    ("wbm-3-1", "Mn1 Si1", 2, None, -0.3),
    ("wbm-3-2", "Cu1", 1, 0.0, 0.0),
    ("wbm-3-3", "Fe1 Co1", 2, 0.02, -0.05),
    ("wbm-3-4", "Al1 Ni1", 2, -0.05, -0.6),
]
WBM_STABLE = {"wbm-1-1", "wbm-1-3", "wbm-1-4", "wbm-3-2", "wbm-3-4"}
WBM_IN_FAMILY = ["wbm-1-1", "wbm-1-2", "wbm-1-3", "wbm-2-2", "wbm-3-1", "wbm-3-3"]


def repo_path(rel: Path) -> Path:
    return Path(__file__).resolve().parents[2] / rel


def b20_cell(compound: str, scale: float = 1.0) -> Atoms:
    a, u, v, tm, x = B20[compound]
    atoms = crystal(
        [tm, x],
        basis=[(u, u, u), (v, v, v)],
        spacegroup=198,
        cellpar=[a * scale] * 3 + [90, 90, 90],
        primitive_cell=False,
    )
    atoms.info.clear()
    return atoms


def label(
    atoms: Atoms,
    rng: np.random.Generator,
    *,
    magnetic: bool = False,
    fscale: float = 0.2,
    stress_3x3: bool = False,
) -> Atoms:
    n = len(atoms)
    results: dict[str, Any] = {
        "energy": float(-6.0 * n + rng.normal(0.0, 0.05)),
        "forces": rng.normal(0.0, fscale, (n, 3)),
    }
    s6 = rng.normal(0.0, 0.005, 6)
    if stress_3x3:
        from ase.stress import voigt_6_to_full_3x3_stress

        results["stress"] = voigt_6_to_full_3x3_stress(s6)
    else:
        results["stress"] = s6
    if magnetic:
        symbols = np.asarray(atoms.get_chemical_symbols())
        results["magmoms"] = np.where(np.isin(symbols, ["Si", "Ge"]), 0.0, 1.0)
    atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def b20_frame(
    compound: str,
    config_type: str = "relax",
    *,
    scale: float = 1.0,
    rattle: float = 0.0,
    seed: int = 0,
    label_source: str = "none",
    energy_scale: str = "none",
    labelled: bool = True,
    parent: str | None = None,
    temperature_K: float | None = None,
    magnetic: bool = False,
    info: dict[str, Any] | None = None,
) -> Frame:
    rng = np.random.default_rng(seed)
    atoms = b20_cell(compound, scale)
    if rattle:
        atoms.positions = atoms.positions + rng.normal(0.0, rattle, atoms.positions.shape)
    if labelled:
        label(atoms, rng, magnetic=magnetic)
    if temperature_K is not None:
        atoms.info["temperature_K"] = temperature_K
    parent_id = parent or f"{compound}-p0"
    frame = frame_from_atoms(
        atoms,
        group_id=f"{compound}/{config_type}/{parent_id}",
        compound=compound,
        config_type=config_type,  # type: ignore[arg-type]
        parent_id=parent_id,
        label_source=label_source,  # type: ignore[arg-type]
        energy_scale=energy_scale,  # type: ignore[arg-type]
    )
    if info:
        frame = frame.model_copy(update={"info": {**frame.info, **info}})
    return frame


def mptrj_member_bytes(images: list[Atoms], mp_id: str, formula: str, *, magnetic: bool) -> bytes:
    """A multi-frame extxyz member in the MPtrj export layout."""
    rng = np.random.default_rng(int(mp_id.split("-")[1]) % 1000)
    for k, atoms in enumerate(images):
        label(atoms, rng, magnetic=magnetic, stress_3x3=True)
        energy = atoms.calc.results["energy"]
        atoms.info.update(
            material_id=mp_id,
            formula=formula,
            task_id=f"{mp_id}-task",
            calc_id=0,
            ionic_step=k,
            frame_id=f"{mp_id}-0-{k}",
            mp2020_corrected_energy=float(energy - 0.284),
        )
    buf = io.StringIO()
    ase_write(buf, images, format="extxyz")
    return buf.getvalue().encode("utf-8")


def omat_row(
    atoms: Atoms,
    *,
    sid: str,
    parent: str | None,
    calc_id: str = "rattled-300",
    fmax: float = 0.5,
    stress_3x3: bool = False,
    corrected: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """One OMat24 ase.db row (the layout measured on the real shards)."""
    rng = np.random.default_rng(seed)
    n = len(atoms)
    forces = rng.normal(0.0, 0.1, (n, 3))
    forces[0] = [fmax, 0.0, 0.0]
    s6 = rng.normal(0.0, 0.01, 6).tolist()
    stress: Any = s6
    if stress_3x3:
        from ase.stress import voigt_6_to_full_3x3_stress

        stress = voigt_6_to_full_3x3_stress(np.asarray(s6)).tolist()
    data: dict[str, Any] = {
        "sid": sid,
        "calc_id": calc_id,
        "task_type": "Static" if calc_id.startswith("rattled") else "Molecular Dynamics",
        "composition_reduced": atoms.get_chemical_formula("hill", empirical=True),
        "prototype_label": "AB_cP8_198_a_a",
        "prototype_error": "",
        "correction_warnings": [],
        "elements": "".join(sorted(set(atoms.get_chemical_symbols()))),
    }
    if parent is not None:
        data["parent_id"] = parent
        data["parent_prototype_label"] = "AB_cP8_198_a_a"
    energy = float(-5.5 * n)
    if corrected:
        data["energy_corrected_mp2020"] = energy - 0.1
        data["energy_correction_uncertainty_mp2020"] = 0.0
        data["energy_adjustments_mp2020"] = []
    return {
        "numbers": [int(z) for z in atoms.numbers],
        "positions": atoms.get_positions().tolist(),
        "unique_id": f"uid-{sid}",
        "pbc": [True, True, True],
        "cell": atoms.cell[:].tolist(),
        "calculator": "unknown",
        "calculator_parameters": {},
        "energy": energy,
        "forces": forces.tolist(),
        "stress": stress,
        "ctime": 1.0,
        "user": "test",
        "mtime": 1.0,
        "data": data,
    }


def optimade_entry(
    entry_id: str,
    atoms: Atoms,
    extra: dict[str, Any] | None = None,
    *,
    disordered: bool = False,
) -> dict[str, Any]:
    symbols = atoms.get_chemical_symbols()
    species = [
        {"name": s, "chemical_symbols": [s], "concentration": [1.0]} for s in sorted(set(symbols))
    ]
    if disordered:
        species[0] = {
            "name": species[0]["name"],
            "chemical_symbols": ["Fe", "Mn"],
            "concentration": [0.5, 0.5],
        }
    attrs: dict[str, Any] = {
        "lattice_vectors": atoms.cell[:].tolist(),
        "cartesian_site_positions": atoms.get_positions().tolist(),
        "species_at_sites": symbols,
        "species": species,
        "elements": sorted(set(symbols)),
        "nelements": len(set(symbols)),
        "chemical_formula_reduced": atoms.get_chemical_formula("hill", empirical=True),
        "dimension_types": [1, 1, 1],
    }
    attrs.update(extra or {})
    return {"id": entry_id, "type": "structures", "attributes": attrs}


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


class RangeServer:
    """Serves ``payload`` with optional Range support, truncation and mid-stream failures."""

    def __init__(
        self,
        payload: bytes,
        *,
        honour_range: bool = True,
        fail_after: int | None = None,
        truncate_to: int | None = None,
        status: int | None = None,
    ) -> None:
        self.payload = payload
        self.honour_range = honour_range
        self.fail_after = fail_after
        self.truncate_to = truncate_to
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status is not None:
            return httpx.Response(self.status)
        rng = request.headers.get("Range")
        offset = 0
        code = 200
        if rng and self.honour_range:
            offset = int(rng.split("=", 1)[1].rstrip("-"))
            if offset >= len(self.payload):
                return httpx.Response(416)
            code = 206
        body = self.payload[offset:]
        if self.truncate_to is not None:
            body = body[: max(0, self.truncate_to - offset)]
        if self.fail_after is not None:
            n = self.fail_after
            self.fail_after = None  # fail once
            return httpx.Response(code, stream=FailingStream(body, n))
        return httpx.Response(code, content=body)


class FailingStream(httpx.SyncByteStream):
    def __init__(self, data: bytes, n: int) -> None:
        self.data = data
        self.n = n

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.data[: self.n]
        raise httpx.ReadError("connection reset by peer")
