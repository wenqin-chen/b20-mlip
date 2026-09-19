"""Data tier (CONTRACTS.md section 1, row 5).

Modules: pull, mptrj, omat24, optimade, wbm, sample, filters, split (+ cli).

Every stage function has the signature ``run(cfg, ctx, **kw)`` and writes its outputs under
the paths it is given (registered with ``ctx.add_output``) plus small JSON summaries under
``ctx.out_dir``. Frame conventions shared with the other tiers are documented in
``docs/DATA_CARD.md`` and in each module's docstring.
"""

from __future__ import annotations

from b20mlip.data._common import (
    B20_SPACEGROUP,
    FAMILY_ELEMENTS,
    TM_ELEMENTS,
    X_ELEMENTS,
    compound_name,
    element_set,
    is_b20_cell,
    spacegroup_number,
)

__all__ = [
    "B20_SPACEGROUP",
    "FAMILY_ELEMENTS",
    "TM_ELEMENTS",
    "X_ELEMENTS",
    "compound_name",
    "element_set",
    "is_b20_cell",
    "spacegroup_number",
]
