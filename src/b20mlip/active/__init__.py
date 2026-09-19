"""Active-learning tier (CONTRACTS.md row 12): committee σ_F frame selection.

``committee.sigma_f`` / ``committee.select`` / ``committee.run`` and the ``b20mlip active
select`` command (``cli.register``). One round: committee σ_F over candidate frames ->
<= 100 frames -> QE labels (dft tier) -> retrain (train tier).
"""

from __future__ import annotations

__all__ = ["committee"]
