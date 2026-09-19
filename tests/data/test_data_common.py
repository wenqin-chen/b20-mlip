"""Element helpers, compound naming and spglib checks shared by the data modules."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from helpers_data import b20_cell

from b20mlip.data import _common as c


def test_normalize_and_element_sets() -> None:
    assert c.normalize_elements(["Fe", 14, " Ge "]) == frozenset({"Fe", "Si", "Ge"})
    with pytest.raises(ValueError, match="unknown element"):
        c.normalize_elements(["Xx"])
    assert c.element_set([26, 26, 14]) == frozenset({"Fe", "Si"})
    assert c.element_key({"Si", "Fe", "Co"}) == "Co-Fe-Si"
    assert c.n_transition_metals([26, 27, 25, 14, 32]) == 3


def test_compound_names_metal_first() -> None:
    assert c.compound_name(Atoms("Ge4Fe4")) == "FeGe"
    assert c.compound_name([14, 14, 27, 27]) == "CoSi"
    assert c.compound_name(Atoms("Fe1Co2Si1")) == "Co2FeSi"
    assert c.compound_name(Atoms("Mn2Ge2")) == "MnGe"


def test_spacegroup_and_b20_checks() -> None:
    fesi = b20_cell("FeSi")
    assert c.spacegroup_number(fesi) == 198 and c.is_b20_cell(fesi)
    assert c.is_b20_cell(fesi.repeat((2, 1, 1))) is False  # 16 atoms
    sheared = fesi.copy()
    F = np.eye(3)
    F[0, 1] = 0.08
    sheared.set_cell(sheared.cell[:] @ F.T, scale_atoms=True)
    assert c.spacegroup_number(sheared) != 198 and not c.is_b20_cell(sheared)
    nacl = Atoms("Na4Cl4", positions=fesi.get_positions(), cell=fesi.cell[:], pbc=True)
    assert not c.is_b20_cell(nacl)  # right symmetry, wrong chemistry
    fe = Atoms("Fe8", positions=fesi.get_positions(), cell=fesi.cell[:], pbc=True)
    assert not c.is_b20_cell(fe)


def test_clean_info_drops_empty_and_converts_numpy() -> None:
    out = c.clean_info({"a": None, "b": "", "c": np.float64(1.5), "d": np.arange(2), "e": "x"})
    assert out == {"c": 1.5, "d": [0, 1], "e": "x"}
    assert isinstance(out["c"], float)
