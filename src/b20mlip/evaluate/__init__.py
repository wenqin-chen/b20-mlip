"""Evaluate tier (CONTRACTS.md row 9): error tables, bootstrap CIs, vendored discovery metrics,
the labelled WBM sample, phonon comparison and elastic constants.

Modules: ``errors``, ``bootstrap``, ``mbd_vendored`` (verbatim from janosh/matbench-discovery
at a pinned commit; ``matbench_discovery`` itself is never imported), ``discovery``,
``phonon_compare``, ``elastic`` and ``cli`` (the ``b20mlip eval ...`` commands).

Every stage function has the signature ``run(cfg, ctx, **kw)`` and writes ``numbers.json``
in its run directory (flat dotted keys plus ``<key>@meta`` with reference, head, n, seed,
ci95, e0_source, energy_scale, model_label and tier) for the report tier to harvest.
"""

from __future__ import annotations

__all__ = ["bootstrap", "discovery", "elastic", "errors", "mbd_vendored", "phonon_compare"]
