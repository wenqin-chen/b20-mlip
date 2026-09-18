"""Report tier (CONTRACTS.md row 14): ``numbers.json``, README rendering and the honesty audit.

* :mod:`b20mlip.report.numbers` — harvest ``NumberRef`` entries from run manifests.
* :mod:`b20mlip.report.build` — render ``templates/README.md.j2`` and write ``README.md``.
* :mod:`b20mlip.report.audit` — honesty gates A1–A11 (CONTRACTS.md section 8).
* :mod:`b20mlip.report.markers` — the marker grammar shared by the writer and the auditor.
"""

from __future__ import annotations

__all__ = ["audit", "build", "markers", "numbers"]
