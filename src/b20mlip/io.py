"""``Frame`` <-> ``ase.Atoms`` conversion and extxyz I/O (CONTRACTS.md section 6).

Conventions (binding for every tier that touches frames):

* energies in eV, forces in eV/Å, stress as Voigt-6 ``[xx, yy, zz, yz, xz, xy]`` in eV/Å^3 —
  the ASE convention, so no unit conversion happens here;
* forces/energy/stress/magmoms travel on a ``SinglePointCalculator`` (what ASE's extxyz writer
  and MACE's loader expect); ``Frame.weights`` becomes ``info["config_<name>_weight"]``
  (``{"energy": 0.0}`` -> ``config_energy_weight=0``, MACE's per-config loss weights);
* frame metadata (``frame_id``, ``group_id``, ``compound``, ``config_type``, ``parent_id``,
  ``label_source``, ``energy_scale``, ``temperature_K``, ``total_magnetization``) lives in
  ``atoms.info`` under those exact keys; whatever else is in ``atoms.info`` round-trips through
  ``Frame.info`` (scalars only; other values are JSON-encoded strings).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.stress import full_3x3_to_voigt_6_stress

from b20mlip.models import Artifact, ConfigType, EnergyScale, Frame, LabelSource
from b20mlip.provenance import artifact_for

META_KEYS: tuple[str, ...] = (
    "frame_id",
    "group_id",
    "compound",
    "config_type",
    "parent_id",
    "label_source",
    "energy_scale",
)
OPTIONAL_META_KEYS: tuple[str, ...] = ("temperature_K", "total_magnetization")
WEIGHT_PREFIX = "config_"
WEIGHT_SUFFIX = "_weight"
FRAME_ID_DECIMALS = 6


def frame_id_for(
    numbers: Iterable[int], positions: Iterable[Iterable[float]], cell: Iterable[Iterable[float]]
) -> str:
    """``sha256(numbers, positions, cell)[:16]`` with coordinates rounded to 1e-6 Å.

    Rounding keeps the id stable across an extxyz round trip (8 printed decimals); ``+ 0.0``
    normalises ``-0.0`` so the byte representation is unique.
    """
    h = hashlib.sha256()
    h.update(np.asarray(list(numbers), dtype=np.int64).tobytes())
    for block in (positions, cell):
        arr = np.round(np.asarray(block, dtype=np.float64), FRAME_ID_DECIMALS) + 0.0
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _info_value(value: Any) -> Any:
    """extxyz info values must be scalars (or 1-d numeric arrays); encode the rest as JSON."""
    value = _jsonable(value)
    if isinstance(value, str):
        return " ".join(value.splitlines())  # the format is line-based: no embedded newlines
    if isinstance(value, (bool, int, float)):
        return value
    return json.dumps(value, sort_keys=True)


def _stress_voigt6(stress: Any) -> list[float] | None:
    if stress is None:
        return None
    arr = np.asarray(stress, dtype=np.float64)
    if arr.shape == (3, 3):
        arr = full_3x3_to_voigt_6_stress(arr)
    elif arr.shape == (9,):
        arr = full_3x3_to_voigt_6_stress(arr.reshape(3, 3))
    if arr.shape != (6,):
        raise ValueError(f"stress must be Voigt-6, 3x3 or 9 values, got shape {arr.shape}")
    return [float(x) for x in arr]


def frame_from_atoms(
    atoms: Atoms,
    *,
    group_id: str,
    compound: str,
    config_type: ConfigType,
    parent_id: str,
    label_source: LabelSource = "none",
    energy_scale: EnergyScale = "none",
    frame_id: str | None = None,
) -> Frame:
    """Convert an ``Atoms`` (labels on its calculator or in ``info``/``arrays``) to a ``Frame``.

    Explicit keyword metadata wins over same-named ``atoms.info`` keys, which are consumed
    (not duplicated into ``Frame.info``).
    """
    info: dict[str, Any] = dict(atoms.info)
    results: dict[str, Any] = dict(atoms.calc.results) if atoms.calc is not None else {}

    numbers = [int(z) for z in atoms.numbers]
    positions = np.asarray(atoms.get_positions(), dtype=np.float64).tolist()
    cell = np.asarray(atoms.cell[:], dtype=np.float64).tolist()
    pbc = tuple(bool(b) for b in atoms.pbc)

    energy = results.get("energy", info.pop("energy", None))
    forces = results.get("forces")
    if forces is None:
        forces = atoms.arrays.get("forces")
    stress = results.get("stress", info.pop("stress", None))
    magmoms = results.get("magmoms")
    if magmoms is None:
        magmoms = atoms.arrays.get("magmoms")
    total_mag = info.pop("total_magnetization", results.get("magmom"))
    temperature = info.pop("temperature_K", None)

    weights: dict[str, float] = {}
    for key in list(info):
        if key.startswith(WEIGHT_PREFIX) and key.endswith(WEIGHT_SUFFIX):
            name = key[len(WEIGHT_PREFIX) : -len(WEIGHT_SUFFIX)]
            if name:
                weights[name] = float(info.pop(key))

    stored_id = info.pop("frame_id", None)
    for key in META_KEYS:
        info.pop(key, None)

    return Frame(
        frame_id=frame_id
        or (str(stored_id) if stored_id else frame_id_for(numbers, positions, cell)),
        group_id=group_id,
        compound=compound,
        config_type=config_type,
        parent_id=parent_id,
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=(pbc[0], pbc[1], pbc[2]),
        energy=None if energy is None else float(energy),
        forces=None if forces is None else np.asarray(forces, dtype=np.float64).tolist(),
        stress=_stress_voigt6(stress),
        magmoms=None if magmoms is None else [float(m) for m in np.asarray(magmoms).ravel()],
        total_magnetization=None if total_mag is None else float(total_mag),
        label_source=label_source,
        energy_scale=energy_scale,
        temperature_K=None if temperature is None else float(temperature),
        weights=weights,
        info={str(k): _jsonable(v) for k, v in info.items()},
    )


def frame_to_atoms(frame: Frame) -> Atoms:
    """Inverse of :func:`frame_from_atoms`; labels go on a ``SinglePointCalculator``."""
    atoms = Atoms(
        numbers=frame.numbers, positions=frame.positions, cell=frame.cell, pbc=list(frame.pbc)
    )
    info: dict[str, Any] = {k: _info_value(v) for k, v in frame.info.items() if v is not None}
    info.update(
        frame_id=frame.frame_id,
        group_id=frame.group_id,
        compound=frame.compound,
        config_type=frame.config_type,
        parent_id=frame.parent_id,
        label_source=frame.label_source,
        energy_scale=frame.energy_scale,
    )
    if frame.temperature_K is not None:
        info["temperature_K"] = float(frame.temperature_K)
    if frame.total_magnetization is not None:
        info["total_magnetization"] = float(frame.total_magnetization)
    for name, weight in frame.weights.items():
        info[f"{WEIGHT_PREFIX}{name}{WEIGHT_SUFFIX}"] = float(weight)
    atoms.info.update(info)

    results: dict[str, Any] = {}
    if frame.energy is not None:
        results["energy"] = float(frame.energy)
    if frame.forces is not None:
        results["forces"] = np.asarray(frame.forces, dtype=np.float64)
    if frame.stress is not None:
        results["stress"] = np.asarray(frame.stress, dtype=np.float64)  # Voigt 6, eV/Å^3
    if frame.magmoms is not None:
        results["magmoms"] = np.asarray(frame.magmoms, dtype=np.float64)
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def read_frames(path: str | Path, *, defaults: Mapping[str, Any] | None = None) -> list[Frame]:
    """Read every configuration of an extxyz file as ``Frame``s.

    Metadata keys missing from ``atoms.info`` are filled from ``defaults``, then from lenient
    fallbacks (``compound`` = empirical formula, ``config_type="relax"``, ``parent_id`` =
    ``frame_id``, ``group_id = compound/config_type/parent_id``, ``label_source`` and
    ``energy_scale`` = ``"none"``). Files written by :func:`write_frames` need no defaults.
    """
    images = ase_read(str(path), index=":", format="extxyz")
    fallback = dict(defaults or {})
    frames: list[Frame] = []
    for atoms in images:
        info = atoms.info
        meta = {key: info.get(key, fallback.get(key)) for key in META_KEYS}
        if not meta["compound"]:
            meta["compound"] = atoms.get_chemical_formula("metal", empirical=True)
        if not meta["config_type"]:
            meta["config_type"] = "relax"
        if not meta["frame_id"]:
            meta["frame_id"] = frame_id_for(atoms.numbers, atoms.get_positions(), atoms.cell[:])
        if not meta["parent_id"]:
            meta["parent_id"] = meta["frame_id"]
        if not meta["group_id"]:
            meta["group_id"] = f"{meta['compound']}/{meta['config_type']}/{meta['parent_id']}"
        if not meta["label_source"]:
            meta["label_source"] = "none"
        if not meta["energy_scale"]:
            meta["energy_scale"] = "none"
        frames.append(
            frame_from_atoms(
                atoms,
                group_id=str(meta["group_id"]),
                compound=str(meta["compound"]),
                config_type=cast(ConfigType, meta["config_type"]),
                parent_id=str(meta["parent_id"]),
                label_source=cast(LabelSource, meta["label_source"]),
                energy_scale=cast(EnergyScale, meta["energy_scale"]),
                frame_id=str(meta["frame_id"]),
            )
        )
    return frames


def write_frames(frames: Iterable[Frame], path: str | Path) -> Artifact:
    """Write frames as extxyz and return the hashed ``Artifact`` (kind ``"frames"``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    images = [frame_to_atoms(f) for f in frames]
    if images:
        ase_write(str(p), images, format="extxyz")
    else:
        p.write_text("", encoding="utf-8")
    return artifact_for(p, "frames")


__all__ = [
    "resolve_head",
    "META_KEYS",
    "frame_from_atoms",
    "frame_id_for",
    "frame_to_atoms",
    "read_frames",
    "write_frames",
]


def resolve_head(requested: str, available: list[str] | tuple[str, ...] | None) -> str:
    """The model head to use for ``requested`` (CONTRACTS ``Head`` = "Default" | "pt_head").

    Foundation checkpoints name their single head ``"default"`` (lower case; MACE-MPA-0, MP-0) while
    fine-tuned models write ``"Default"``: a case-insensitive unique match is accepted, anything
    else raises (MACE itself would silently fall back to the last head).
    """
    if not available:
        return requested
    if requested in available:
        return requested
    matches = [h for h in available if h.lower() == requested.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"model has heads {list(available)}, not {requested!r}")
