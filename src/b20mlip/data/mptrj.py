"""MPtrj (``2024-09-03-mp-trj.extxyz.zip``, MIT): in-family extraction and the B20 subset.

The archive holds 145,923 flat members ``mp-<id>.extxyz``, one per material, each a
multi-frame extxyz of that material's relaxation trajectories (~6 GB uncompressed). Every
frame's comment line carries ``material_id``, ``formula``, ``task_id``, ``calc_id``,
``ionic_step``, ``frame_id`` (MPtrj's own ``<task>-<calc>-<step>`` id), ``energy`` (raw PBE/
PBE+U total energy), ``mp2020_corrected_energy`` and a 3x3 ``stress`` whose units are not
documented in the export; forces and (for spin-polarised tasks) per-site ``magmoms`` are
per-atom columns.

Streaming strategy: only the first frame of each member is decompressed to read its element
set (all frames of a member share one composition), so the whole archive is screened in
seconds; only in-family members are parsed with ASE.

Frame conventions produced here (binding):

* ``label_source="mptrj"``, ``energy_scale="mp"``, ``config_type="relax"``;
* ``group_id="<compound>/mptrj/<mp-id>"``, ``parent_id="<mp-id>"``;
* ``energy`` = raw MPtrj energy (eV), ``forces`` (eV/Å), ``magmoms`` when present;
* ``stress`` is **None**: the export's ``stress`` key is kept verbatim (Voigt-6 of the 3x3)
  in ``info["mptrj_stress_raw_voigt6"]`` because its unit and sign convention are not
  verified (the MPtrj JSON stores kBar, VASP sign); never train or evaluate on it;
* ``info``: ``mp_id``, ``task_id``, ``calc_id``, ``ionic_step``, ``mptrj_frame_id``,
  ``formula``, ``mp2020_corrected_energy`` (when present);
* ``frame_id`` is the project's geometry hash (:func:`b20mlip.io.frame_id_for`), not
  MPtrj's ``frame_id``.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from b20mlip.data._common import (
    DEFAULT_SYMPREC,
    FAMILY_ELEMENTS,
    clean_info,
    compound_name,
    element_key,
    element_set,
    is_b20_cell,
    normalize_elements,
)
from b20mlip.io import frame_from_atoms, frame_to_atoms, write_frames
from b20mlip.models import Frame

MEMBER_RE = re.compile(r"^(?:.*/)?(mp-\d+)\.extxyz$")
MPTRJ_INFO_KEYS: tuple[str, ...] = (
    "task_id",
    "calc_id",
    "ionic_step",
    "formula",
    "mp2020_corrected_energy",
)
Log = Callable[..., None] | None


def member_mp_id(name: str) -> str | None:
    """``"mp-871.extxyz"`` -> ``"mp-871"``; ``None`` for members that are not MPtrj materials."""
    m = MEMBER_RE.match(name)
    return m.group(1) if m else None


def _mp_sort_key(name: str) -> int:
    mp_id = member_mp_id(name) or "mp-0"
    return int(mp_id.split("-", 1)[1])


def screen_member(zf: zipfile.ZipFile, name: str) -> frozenset[str]:
    """Element set of a member's first frame (decompresses only its head)."""
    with zf.open(name) as fh:
        first = fh.readline()
        try:
            natoms = int(first.strip())
        except ValueError:
            return frozenset()
        fh.readline()  # comment line
        symbols: set[str] = set()
        for _ in range(natoms):
            line = fh.readline()
            if not line:
                break
            symbols.add(line.split(None, 1)[0].decode("ascii"))
    return frozenset(symbols)


def iter_family_members(
    zf: zipfile.ZipFile,
    elements: Iterable[str | int] = FAMILY_ELEMENTS,
    *,
    min_elements: int = 2,
    stats: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Yield member names whose element set is a subset of ``elements`` (sorted by mp id)."""
    family = normalize_elements(elements)
    scanned = 0
    elemental = 0
    hits: list[str] = []
    for name in zf.namelist():
        if member_mp_id(name) is None:
            continue
        scanned += 1
        symbols = screen_member(zf, name)
        if not symbols or not symbols <= family:
            continue
        if len(symbols) < min_elements:
            elemental += 1
            continue
        hits.append(name)
    hits.sort(key=_mp_sort_key)
    if stats is not None:
        stats.update(
            members_scanned=scanned,
            members_in_family=len(hits),
            members_below_min_elements=elemental,
            min_elements=min_elements,
            elements=sorted(family),
        )
    yield from hits


def parse_member(zf: zipfile.ZipFile, name: str) -> list[Atoms]:
    text = zf.read(name).decode("utf-8")
    images = ase_read(io.StringIO(text), index=":", format="extxyz")
    return list(images) if isinstance(images, list) else [images]


def frame_from_mptrj_atoms(atoms: Atoms, mp_id: str | None = None) -> Frame:
    """Convert one parsed MPtrj image to a :class:`Frame` (conventions in the module doc)."""
    atoms = atoms.copy() if atoms.calc is None else _copy_with_calc(atoms)
    info = dict(atoms.info)
    material_id = str(info.pop("material_id", mp_id or ""))
    if not material_id:
        raise ValueError("MPtrj frame without material_id")
    mptrj_frame_id = info.pop("frame_id", None)
    info.pop("stress", None)
    raw_stress: list[float] | None = None
    if atoms.calc is not None and "stress" in atoms.calc.results:
        raw_stress = _voigt6(atoms.calc.results.pop("stress"))
    atoms.info = {}
    compound = compound_name(atoms)
    frame = frame_from_atoms(
        atoms,
        group_id=f"{compound}/mptrj/{material_id}",
        compound=compound,
        config_type="relax",
        parent_id=material_id,
        label_source="mptrj",
        energy_scale="mp",
    )
    kept: dict[str, Any] = {"mp_id": material_id}
    for key in MPTRJ_INFO_KEYS:
        if key in info:
            kept[key] = info[key]
    if mptrj_frame_id is not None:
        kept["mptrj_frame_id"] = str(mptrj_frame_id)
    if raw_stress is not None:
        kept["mptrj_stress_raw_voigt6"] = [float(x) for x in raw_stress]
    return frame.model_copy(update={"stress": None, "info": clean_info(kept)})


def _voigt6(stress: Any) -> list[float]:
    from ase.stress import full_3x3_to_voigt_6_stress

    arr = np.asarray(stress, dtype=float)
    if arr.shape in ((3, 3), (9,)):
        arr = full_3x3_to_voigt_6_stress(arr.reshape(3, 3))
    return [float(x) for x in arr.ravel()]


def _copy_with_calc(atoms: Atoms) -> Atoms:
    from ase.calculators.singlepoint import SinglePointCalculator

    results = dict(atoms.calc.results) if atoms.calc is not None else {}
    out = atoms.copy()
    out.calc = SinglePointCalculator(out, **results)
    return out


def frames_from_member(zf: zipfile.ZipFile, name: str) -> list[Frame]:
    mp_id = member_mp_id(name)
    return [frame_from_mptrj_atoms(a, mp_id) for a in parse_member(zf, name)]


def extract_family(
    zip_path: str | Path,
    elements: Iterable[str | int] = FAMILY_ELEMENTS,
    out: str | Path | None = None,
    *,
    min_elements: int = 2,
    stats: dict[str, Any] | None = None,
    log: Log = None,
) -> list[Frame]:
    """Stream the archive and keep every frame whose element set is a subset of ``elements``.

    ``min_elements=2`` skips elemental members (counted in ``stats``). Frames are ordered by
    mp id, then archive order. When ``out`` is given the frames are also written there.
    """
    family = normalize_elements(elements)
    frames: list[Frame] = []
    per_set: dict[str, int] = {}
    scan: dict[str, Any] = {}
    with zipfile.ZipFile(Path(zip_path)) as zf:
        for name in iter_family_members(zf, family, min_elements=min_elements, stats=scan):
            member_frames = [
                f for f in frames_from_member(zf, name) if element_set(f.numbers) <= family
            ]
            for f in member_frames:
                key = element_key(element_set(f.numbers))
                per_set[key] = per_set.get(key, 0) + 1
            frames.extend(member_frames)
            if log is not None:
                log(member=name, frames=len(member_frames))
    if stats is not None:
        stats.update(scan)
        stats.update(frames=len(frames), per_element_set=dict(sorted(per_set.items())))
    if out is not None:
        write_frames(frames, out)
    return frames


def b20_subset(frames: Iterable[Frame], symprec: float = DEFAULT_SYMPREC) -> list[Frame]:
    """Frames that are 8-atom TM-X cells with space group 198 (P2_1 3)."""
    return [f for f in frames if is_b20_cell(frame_to_atoms(f), symprec)]


def b20_frames(
    zip_path: str | Path, *, symprec: float = DEFAULT_SYMPREC, stats: dict[str, Any] | None = None
) -> list[Frame]:
    """The B20 subset of the family extraction (see :func:`b20_subset`)."""
    return b20_subset(extract_family(zip_path, FAMILY_ELEMENTS, None, stats=stats), symprec)


def mp_ids(frames: Iterable[Frame]) -> dict[str, int]:
    """Frame counts per mp id."""
    counts: dict[str, int] = {}
    for f in frames:
        counts[f.parent_id] = counts.get(f.parent_id, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "MPTRJ_INFO_KEYS",
    "b20_frames",
    "b20_subset",
    "extract_family",
    "frame_from_mptrj_atoms",
    "frames_from_member",
    "iter_family_members",
    "member_mp_id",
    "mp_ids",
    "parse_member",
    "screen_member",
]
