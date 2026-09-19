"""WBM (CC-BY-4.0) seeded natural-prevalence sample for the discovery tier (T4b).

Inputs are the matbench-discovery files: ``wbm-summary.csv.gz`` (256,963 rows; columns
``material_id, formula, n_sites, ..., e_form_per_atom_mp2020_corrected,
e_above_hull_mp2020_corrected_ppd_mp, ...``) and ``wbm-initial-atoms.extxyz.zip`` (one
member ``<material_id>.extxyz`` per structure, so structures are fetched by name without
loading the archive).

Protocol (binding, recorded in the sample JSON): ``ids = sorted(material_id)``;
``perm = numpy.random.default_rng(seed).permutation(len(ids))``; ``sample = ids[perm[:n]]``.
Stability truth is ``e_above_hull_mp2020_corrected_ppd_mp <= 0`` (the MP2020-corrected
energy above the MP patched phase diagram hull, the matbench-discovery convention); the
formation energy is ``e_form_per_atom_mp2020_corrected``. Prevalence = stable fraction.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read as ase_read

from b20mlip.data._common import FAMILY_ELEMENTS, clean_info, compound_name, normalize_elements
from b20mlip.io import frame_from_atoms
from b20mlip.models import Frame
from b20mlip.provenance import sha256_file

HULL_COL = "e_above_hull_mp2020_corrected_ppd_mp"
E_FORM_COL = "e_form_per_atom_mp2020_corrected"
ID_COL = "material_id"
FORMULA_COL = "formula"
NSITES_COL = "n_sites"
SUMMARY_COLUMNS: tuple[str, ...] = (ID_COL, FORMULA_COL, NSITES_COL, E_FORM_COL, HULL_COL)
SAMPLE_SCHEMA = 1
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def formula_elements(formula: str) -> frozenset[str]:
    """``"Fe4 Si4"`` -> ``{"Fe", "Si"}`` (WBM formulas are space-separated element counts)."""
    return frozenset(m.group(1) for m in _ELEMENT_RE.finditer(str(formula)))


def read_summary(summary_csv: str | Path) -> pd.DataFrame:
    """The columns this module needs, indexed by ``material_id``."""
    df = pd.read_csv(summary_csv, usecols=list(SUMMARY_COLUMNS))
    missing = [c for c in SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{summary_csv}: missing columns {missing}")
    return df.set_index(ID_COL)


def _stable(series: pd.Series) -> pd.Series:
    return series.notna() & (series <= 0.0)


def _truth(df: pd.DataFrame, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mid in ids:
        row = df.loc[mid]
        e_hull = row[HULL_COL]
        e_form = row[E_FORM_COL]
        out[mid] = {
            "formula": str(row[FORMULA_COL]),
            "n_sites": int(row[NSITES_COL]),
            "e_form_per_atom": None if pd.isna(e_form) else float(e_form),
            "e_above_hull": None if pd.isna(e_hull) else float(e_hull),
            "stable": bool(not pd.isna(e_hull) and e_hull <= 0.0),
        }
    return out


def sample(
    summary_csv: str | Path,
    atoms_zip: str | Path,
    n: int,
    seed: int,
    family_elements: Iterable[str | int] = FAMILY_ELEMENTS,
    *,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Seeded natural-prevalence sample of ``n`` WBM ids plus the in-family stratum.

    Returns (and optionally writes) a JSON-able dict with ``ids``, ``prevalence`` (stable
    fraction of the sample), ``population_prevalence``, ``in_family_ids``,
    ``in_family_stable_fraction``, per-id ``truth`` and ``checksums`` of both source files.
    """
    family = normalize_elements(family_elements)
    df = read_summary(summary_csv)
    ids = sorted(str(i) for i in df.index)
    if n > len(ids):
        raise ValueError(f"n={n} exceeds the {len(ids)} WBM entries")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    chosen = [ids[i] for i in perm[:n]]
    stable_all = _stable(df[HULL_COL])
    in_family = [mid for mid in ids if formula_elements(df.at[mid, FORMULA_COL]) <= family]
    n_family_stable = int(stable_all.loc[in_family].sum()) if in_family else 0
    sample_stable = int(stable_all.loc[chosen].sum())
    result: dict[str, Any] = {
        "schema": SAMPLE_SCHEMA,
        "n": int(n),
        "seed": int(seed),
        "protocol": "ids=sorted(material_id); numpy.random.default_rng(seed).permutation; first n",
        "hull_column": HULL_COL,
        "e_form_column": E_FORM_COL,
        "stability_threshold": 0.0,
        "population_n": len(ids),
        "population_stable": int(stable_all.sum()),
        "population_prevalence": float(stable_all.mean()),
        "population_missing_hull": int(df[HULL_COL].isna().sum()),
        "ids": chosen,
        "n_stable": sample_stable,
        "prevalence": sample_stable / n if n else 0.0,
        "family_elements": sorted(family),
        "in_family_ids": in_family,
        "in_family_n": len(in_family),
        "in_family_stable": n_family_stable,
        "in_family_stable_fraction": (n_family_stable / len(in_family)) if in_family else 0.0,
        "truth": _truth(df, sorted(set(chosen) | set(in_family))),
        "checksums": {
            "summary_csv": {
                "path": str(summary_csv),
                "bytes": Path(summary_csv).stat().st_size,
                "sha256": sha256_file(summary_csv),
            },
            "atoms_zip": {
                "path": str(atoms_zip),
                "bytes": Path(atoms_zip).stat().st_size,
                "sha256": sha256_file(atoms_zip),
            },
        },
    }
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def load_sample(path: str | Path) -> dict[str, Any]:
    """Read a sample JSON written by :func:`sample` and check its required keys."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("ids", "prevalence", "in_family_ids", "truth", "checksums", "seed", "n")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path}: not a WBM sample file (missing {missing})")
    if len(data["ids"]) != data["n"]:
        raise ValueError(f"{path}: {len(data['ids'])} ids but n={data['n']}")
    return data


def atoms_for_ids(atoms_zip: str | Path, ids: Iterable[str]) -> dict[str, Atoms]:
    """Initial structures for ``ids`` read member-by-member (never the whole archive)."""
    out: dict[str, Atoms] = {}
    with zipfile.ZipFile(Path(atoms_zip)) as zf:
        for mid in ids:
            text = zf.read(f"{mid}.extxyz").decode("utf-8")
            atoms = ase_read(io.StringIO(text), index=0, format="extxyz")
            atoms.info.pop("material_id", None)
            out[mid] = atoms
    return out


def frames_for_ids(atoms_zip: str | Path, ids: Iterable[str]) -> list[Frame]:
    """:func:`atoms_for_ids` as frames (``config_type="wbm"``, group ``<compound>/wbm/<id>``)."""
    frames: list[Frame] = []
    for mid, atoms in atoms_for_ids(atoms_zip, ids).items():
        compound = compound_name(atoms)
        atoms.info = {}
        frame = frame_from_atoms(
            atoms, group_id=f"{compound}/wbm/{mid}", compound=compound,
            config_type="wbm", parent_id=mid,
        )  # fmt: skip
        frames.append(frame.model_copy(update={"info": clean_info({"wbm_id": mid})}))
    return frames


__all__ = [
    "E_FORM_COL",
    "HULL_COL",
    "SUMMARY_COLUMNS",
    "atoms_for_ids",
    "formula_elements",
    "frames_for_ids",
    "load_sample",
    "read_summary",
    "sample",
]
