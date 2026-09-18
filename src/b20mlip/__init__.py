"""b20-mlip: fine-tune MACE-MPA-0 on spin-polarised QE frames of B20 skyrmion hosts.

See SPEC.md (goals, science) and CONTRACTS.md (binding names, signatures, schemas).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("b20-mlip")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
