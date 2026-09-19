"""Reference B20 cells for ``dft converge`` / ``dft phonons`` (the relaxed parent structures).

The default source is the local MPtrj extract ``data/raw/mptrj/b20_mptrj.extxyz`` (60 frames of
mp-871 FeSi, mp-7577 CoSi, mp-1431 MnSi, mp-21255 FeGe and the non-cubic mp-22510 FeGe polymorph,
which is excluded here); ``--structure`` points at any extxyz instead. The "relaxed" cell is the
lowest-energy frame of the compound when energies are present, else the last frame. Labels from
the source (MP energies/forces) are stripped: the returned frame is a geometry, not a datum.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ase.data import atomic_numbers

from b20mlip.config import Settings
from b20mlip.io import read_frames
from b20mlip.models import Frame

B20_MP_IDS: dict[str, str] = {
    "FeSi": "mp-871",
    "CoSi": "mp-7577",
    "MnSi": "mp-1431",
    "FeGe": "mp-21255",
}
DEFAULT_MPTRJ_RELPATH = Path("raw") / "mptrj" / "b20_mptrj.extxyz"
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


def elements_of(compounds: list[str] | tuple[str, ...] | str) -> list[str]:
    """Unique element symbols of compound strings, in order of first appearance."""
    if isinstance(compounds, str):
        compounds = [compounds]
    out: list[str] = []
    for compound in compounds:
        for symbol in _ELEMENT_RE.findall(compound):
            if symbol not in atomic_numbers:
                raise ValueError(f"unknown element {symbol!r} in compound {compound!r}")
            if symbol not in out:
                out.append(symbol)
    return out


def strip_labels(frame: Frame) -> Frame:
    """Drop energies/forces/stress/magmoms and the label provenance of a frame (geometry only)."""
    return frame.model_copy(
        update={
            "energy": None,
            "forces": None,
            "stress": None,
            "magmoms": None,
            "total_magnetization": None,
            "label_source": "none",
            "energy_scale": "none",
            "weights": {},
        }
    )


def _matches(frame: Frame, compound: str) -> bool:
    mp_id = frame.info.get("material_id")
    if compound in B20_MP_IDS and mp_id:
        return str(mp_id) == B20_MP_IDS[compound]
    return frame.compound == compound


def default_structure_path(cfg: Settings) -> Path:
    return Path(cfg.paths.data_dir) / DEFAULT_MPTRJ_RELPATH


def reference_frame(cfg: Settings, compound: str, structure: str | Path | None = None) -> Frame:
    """The relaxed reference cell of ``compound`` from ``structure`` or the MPtrj B20 extract."""
    path = Path(structure) if structure is not None else default_structure_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"no structure file {path}; pass --structure or run `b20mlip data pull` first"
        )
    candidates = [f for f in read_frames(path) if _matches(f, compound)]
    if not candidates:
        raise ValueError(f"no frame of compound {compound!r} in {path}")
    if any(f.energy is not None for f in candidates):
        best = min(candidates, key=lambda f: f.energy if f.energy is not None else math.inf)
    else:
        best = candidates[-1]
    ref = strip_labels(best)
    return ref.model_copy(
        update={
            "compound": compound,
            "config_type": "relax",
            "parent_id": ref.frame_id,
            "group_id": f"{compound}/relax/{ref.frame_id}",
            "info": {**ref.info, "reference_source": str(path)},
        }
    )


__all__ = [
    "B20_MP_IDS",
    "default_structure_path",
    "elements_of",
    "reference_frame",
    "strip_labels",
]
