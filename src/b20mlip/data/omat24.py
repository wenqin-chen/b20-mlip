"""OMat24 (CC-BY-4.0) ASE-LMDB shards: stream, filter to the Mn/Fe/Co/Si/Ge family, cap forces.

``fairchem-core`` cannot be installed next to ``mace-torch==0.3.16`` (e3nn pin), so the
``.aselmdb`` shards are read natively: an LMDB file (``subdir=False``) whose integer keys map
to zlib-compressed, orjson-encoded ``ase.db`` rows (``numbers``, ``positions``, ``cell``,
``pbc``, ``energy``, ``forces``, ``stress`` and a ``data`` dict with ``sid``, ``calc_id``,
``task_type``, ``parent_id``, ``prototype_label``, ``energy_corrected_mp2020``, ...); the
non-integer key ``nextid`` (and any ``metadata``/``deleted_ids``) is skipped.

Unit conventions (measured on the 1M subsplit; see docs/DATA_CARD.md): energies in eV,
forces in eV/Å and ``stress`` as the ASE-convention Voigt-6 ``[xx, yy, zz, yz, xz, xy]`` in
eV/Å^3 exactly as ``ase.db`` stored it (no sign flip, no unit conversion is applied here).

Frame conventions produced here (binding): ``label_source="omat24"``,
``energy_scale="omat24"``, ``config_type="omat24"``, ``group_id="<compound>/omat24/<parent_id>"``
with ``parent_id`` = OMat24's ``data["parent_id"]`` (the Alexandria parent, e.g.
``agm002241788_ABC2_3_spg225``; the ``sid`` when absent), ``info["omat24_sid"]`` etc. The
training-time rule R4 (``config_energy_weight=0``) is *not* applied here.
"""

from __future__ import annotations

import tarfile
import tempfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from b20mlip.data._common import (
    B20_NATOMS,
    B20_SPACEGROUP,
    DEFAULT_SYMPREC,
    FAMILY_ELEMENTS,
    clean_info,
    compound_name,
    element_key,
    element_set,
    normalize_elements,
    spacegroup_number,
)
from b20mlip.io import frame_id_for, frame_to_atoms, write_frames
from b20mlip.models import Frame

try:  # optional extra [omat]
    import lmdb
except ImportError:  # pragma: no cover - exercised only when the extra is missing
    lmdb = None
try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

SHARD_SUFFIX = ".aselmdb"
Log = Callable[..., None] | None
OMAT24_DATA_KEYS: tuple[str, ...] = (
    "sid",
    "calc_id",
    "task_type",
    "parent_id",
    "prototype_label",
    "parent_prototype_label",
    "composition_reduced",
    "energy_corrected_mp2020",
)


def missing_dependencies() -> list[str]:
    return [name for name, mod in (("lmdb", lmdb), ("orjson", orjson)) if mod is None]


def _require() -> None:
    missing = missing_dependencies()
    if missing:
        raise ImportError(
            f"reading OMat24 ASE-LMDB shards needs {', '.join(missing)} (install the [omat] extra)"
        )


def find_shards(root: str | Path) -> list[Path]:
    """All ``*.aselmdb`` files below ``root``, sorted by relative path."""
    base = Path(root)
    if base.is_file() and base.suffix == SHARD_SUFFIX:
        return [base]
    return sorted(p for p in base.rglob(f"*{SHARD_SUFFIX}") if p.is_file())


def shard_label(path: Path, root: Path | None) -> str:
    """``.../omat24_1M/test/rattled-300/test.aselmdb`` -> ``test/rattled-300``."""
    p = Path(path)
    try:
        rel = p.relative_to(root) if root is not None else p
    except ValueError:
        rel = p
    return str(rel.parent).replace("\\", "/") if str(rel.parent) != "." else p.stem


def decode_ase_json(obj: Any) -> Any:
    """Undo ase.db's JSON conventions: ``{"__ndarray__": [shape, dtype, flat]}`` -> list.

    Some OMat24 shards store ``numbers``/``positions``/``cell``/``forces``/``stress`` this way,
    others as plain nested lists; both decode to plain Python lists here.
    """
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            shape, dtype, data = obj["__ndarray__"]
            return np.asarray(data, dtype=dtype).reshape(shape).tolist()
        if "__complex_ndarray__" in obj:  # pragma: no cover - not used by OMat24
            re_, im_ = obj["__complex_ndarray__"]
            return (np.asarray(re_) + 1j * np.asarray(im_)).tolist()
        return {k: decode_ase_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_ase_json(v) for v in obj]
    return obj


def decode_row(value: bytes) -> dict[str, Any]:
    _require()
    try:
        raw = zlib.decompress(value)
    except zlib.error:  # rows written without compression
        raw = value
    row: dict[str, Any] = decode_ase_json(orjson.loads(raw))
    return row


def iter_shard_rows(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(key, row)`` for every integer key of one shard, in key order."""
    _require()
    env = lmdb.open(
        str(path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=1
    )
    try:
        with env.begin() as txn:
            for key, value in txn.cursor():
                if not key.isdigit():
                    continue
                yield int(key), decode_row(value)
    finally:
        env.close()


def iter_source(source: str | Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """``(shard label, key, row)`` from a directory of shards or an OMat24 ``.tar.gz``.

    A tarball is never fully extracted: each shard is copied to a temporary file, read and
    deleted before the next one.
    """
    src = Path(source)
    if src.is_dir() or src.suffix == SHARD_SUFFIX:
        root = src if src.is_dir() else src.parent
        for shard in find_shards(src):
            label = shard_label(shard, root)
            for key, row in iter_shard_rows(shard):
                yield label, key, row
        return
    if not src.is_file():
        raise FileNotFoundError(f"OMat24 source not found: {src}")
    with tarfile.open(src, "r:*") as tar, tempfile.TemporaryDirectory() as tmp:
        for member in tar:
            if not member.isfile() or not member.name.endswith(SHARD_SUFFIX):
                continue
            fh = tar.extractfile(member)
            if fh is None:  # pragma: no cover - defensive
                continue
            local = Path(tmp) / Path(member.name).name
            _copy_stream(fh, local)
            parts = Path(member.name).parts
            label = "/".join(parts[-3:-1]) if len(parts) >= 3 else Path(member.name).stem
            for key, row in iter_shard_rows(local):
                yield label, key, row
            local.unlink(missing_ok=True)


def _copy_stream(fh: Any, dest: Path, chunk_size: int = 1 << 20) -> None:
    with open(dest, "wb") as out:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)


def row_to_frame(row: dict[str, Any], *, shard: str = "", key: int | None = None) -> Frame:
    """One ase.db row -> :class:`Frame` (conventions in the module docstring)."""
    numbers = [int(z) for z in row["numbers"]]
    positions = np.asarray(row["positions"], dtype=float).tolist()
    cell = np.asarray(row["cell"], dtype=float).tolist()
    pbc_raw = row.get("pbc", [True, True, True])
    pbc = tuple(bool(b) for b in pbc_raw)
    data = dict(row.get("data") or {})
    sid = str(data.get("sid", row.get("unique_id", "")))
    parent = str(data.get("parent_id") or sid.split("_", 1)[0] or sid)
    compound = compound_name(numbers)
    forces = row.get("forces")
    stress = row.get("stress")
    stress_list: list[float] | None = None
    if stress is not None:
        arr = np.asarray(stress, dtype=float)
        if arr.shape == (3, 3) or arr.shape == (9,):
            from ase.stress import full_3x3_to_voigt_6_stress

            arr = full_3x3_to_voigt_6_stress(arr.reshape(3, 3))
        stress_list = [float(x) for x in arr.ravel()]
    info: dict[str, Any] = {"omat24_shard": shard}
    if key is not None:
        info["omat24_row"] = int(key)
    if row.get("unique_id"):
        info["omat24_unique_id"] = str(row["unique_id"])
    for k in OMAT24_DATA_KEYS:
        if k in data:
            info[f"omat24_{k}" if not k.startswith("energy") else k] = data[k]
    return Frame(
        frame_id=frame_id_for(numbers, positions, cell),
        group_id=f"{compound}/omat24/{parent}",
        compound=compound,
        config_type="omat24",
        parent_id=parent,
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=(pbc[0], pbc[1], pbc[2]),
        energy=None if row.get("energy") is None else float(row["energy"]),
        forces=None if forces is None else np.asarray(forces, dtype=float).tolist(),
        stress=stress_list,
        label_source="omat24",
        energy_scale="omat24",
        info=clean_info(info),
    )


def max_force_norm(frame: Frame) -> float | None:
    if frame.forces is None:
        return None
    arr = np.asarray(frame.forces, dtype=float)
    return float(np.linalg.norm(arr, axis=1).max()) if arr.size else 0.0


def stream_filter(
    source: str | Path,
    elements: Iterable[str | int] = FAMILY_ELEMENTS,
    out: str | Path | None = None,
    force_cap: float = 15.0,
    *,
    min_elements: int = 2,
    symprec: float = DEFAULT_SYMPREC,
    log: Log = None,
) -> tuple[list[Frame], dict[str, Any]]:
    """Keep frames whose element set is a subset of ``elements`` and max ||F|| <= ``force_cap``.

    Returns the frames and a counts dict: ``scanned``, ``in_family``, ``kept``,
    ``dropped_force_cap``, ``dropped_min_elements``, ``per_element_set``, ``per_shard``,
    ``per_calc_id``, ``n_b20_spg198`` (kept frames with spglib space group 198 at ``symprec``)
    and ``n_spg198_8atom``.
    """
    family = normalize_elements(elements)
    frames: list[Frame] = []
    counts: dict[str, Any] = {
        "scanned": 0,
        "in_family": 0,
        "kept": 0,
        "dropped_force_cap": 0,
        "dropped_min_elements": 0,
        "force_cap_eVA": float(force_cap),
        "symprec": float(symprec),
        "min_elements": int(min_elements),
        "elements": sorted(family),
    }
    per_set: dict[str, int] = {}
    per_shard: dict[str, int] = {}
    per_calc: dict[str, int] = {}
    n_198 = 0
    n_198_8 = 0
    family_z = {z for z in range(1, 119) if element_set([z]) <= family}
    for shard, key, row in iter_source(source):
        counts["scanned"] += 1
        zs = {int(z) for z in row["numbers"]}
        if not zs <= family_z:
            continue
        if len(zs) < min_elements:
            counts["dropped_min_elements"] += 1
            continue
        counts["in_family"] += 1
        frame = row_to_frame(row, shard=shard, key=key)
        fmax = max_force_norm(frame)
        if fmax is not None and fmax > force_cap:
            counts["dropped_force_cap"] += 1
            continue
        frames.append(frame)
        key_set = element_key(element_set(frame.numbers))
        per_set[key_set] = per_set.get(key_set, 0) + 1
        per_shard[shard] = per_shard.get(shard, 0) + 1
        calc_id = str(frame.info.get("omat24_calc_id", ""))
        per_calc[calc_id] = per_calc.get(calc_id, 0) + 1
        spg = spacegroup_number(frame_to_atoms(frame), symprec)
        if spg == B20_SPACEGROUP:
            n_198 += 1
            if len(frame.numbers) == B20_NATOMS:
                n_198_8 += 1
        if log is not None and counts["in_family"] % 100 == 0:
            log(scanned=counts["scanned"], kept=len(frames))
    counts["kept"] = len(frames)
    counts["per_element_set"] = dict(sorted(per_set.items()))
    counts["per_shard"] = dict(sorted(per_shard.items()))
    counts["per_calc_id"] = dict(sorted(per_calc.items()))
    counts["n_b20_spg198"] = n_198
    counts["n_spg198_8atom"] = n_198_8
    if out is not None:
        write_frames(frames, out)
    return frames, counts


def frames_to_rows(frames: Iterable[Frame]) -> list[dict[str, Any]]:
    """Inverse of :func:`row_to_frame` for building test shards (ase.db row layout)."""
    rows: list[dict[str, Any]] = []
    for f in frames:
        atoms = frame_to_atoms(f)
        rows.append(
            {
                "numbers": [int(z) for z in atoms.numbers],
                "positions": atoms.get_positions().tolist(),
                "cell": atoms.cell[:].tolist(),
                "pbc": [bool(b) for b in atoms.pbc],
                "unique_id": f.frame_id,
                "calculator": "unknown",
                "calculator_parameters": {},
                "energy": f.energy,
                "forces": f.forces,
                "stress": f.stress,
                "data": {
                    "sid": f.info.get("omat24_sid", f"{f.parent_id}_0_test_{f.frame_id[:6]}"),
                    "calc_id": f.info.get("omat24_calc_id", "rattled-300"),
                    "task_type": f.info.get("omat24_task_type", "Static"),
                    "parent_id": f.parent_id,
                    "prototype_label": f.info.get("omat24_prototype_label", ""),
                    "composition_reduced": f.info.get("omat24_composition_reduced", ""),
                },
            }
        )
    return rows


def write_shard(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write rows as an ``.aselmdb`` shard (zlib + orjson, integer keys from 1); for tests."""
    _require()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(p), subdir=False, map_size=1 << 30)
    try:
        with env.begin(write=True) as txn:
            n = 0
            for i, row in enumerate(rows, start=1):
                txn.put(str(i).encode(), zlib.compress(orjson.dumps(row)))
                n = i
            txn.put(b"nextid", zlib.compress(orjson.dumps(n + 1)))
    finally:
        env.close()
    return p


__all__ = [
    "OMAT24_DATA_KEYS",
    "SHARD_SUFFIX",
    "decode_ase_json",
    "decode_row",
    "find_shards",
    "frames_to_rows",
    "iter_shard_rows",
    "iter_source",
    "max_force_norm",
    "missing_dependencies",
    "row_to_frame",
    "shard_label",
    "stream_filter",
    "write_shard",
]
