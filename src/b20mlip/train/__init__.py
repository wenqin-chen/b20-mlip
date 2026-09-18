"""Training tier (CONTRACTS.md row 7): ``finetune`` (mace_run_train argv + stage) and ``export``.

Nothing here imports torch or mace at module level: ``b20mlip.cli`` imports ``b20mlip.train.cli``
at start-up, and every MACE call happens in a subprocess (``mace_run_train``,
``mace_create_lammps_model``) or behind a lazy import.
"""

from __future__ import annotations

__all__ = ["cli", "export", "finetune"]
