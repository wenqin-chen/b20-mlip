from pathlib import Path

import pytest

from b20mlip.config import Settings

MPA0 = Path(__file__).resolve().parents[1] / "models" / "foundation" / "mace-mpa-0-medium.model"


@pytest.mark.skipif(not MPA0.is_file(), reason="foundation weights not present")
def test_zero_shot_foundation_loads_on_head_default() -> None:
    from b20mlip.evaluate.errors import make_calculator as eval_calc
    from b20mlip.md.ase_md import make_calculator as md_calc

    c1 = eval_calc(MPA0, "Default")
    c2 = md_calc(MPA0, Settings(), "Default")
    assert (
        getattr(c1, "head", "default") == "default" and getattr(c2, "head", "default") == "default"
    )
