"""Cluster hand-off (SPEC.md section 8): ``b20mlip cluster bootstrap | sync | status``.

* :mod:`b20mlip.cluster.remote` — the ``COMMANDS`` table, output parsers and the ``Transport``
  built on :class:`~b20mlip.executors.SlurmExecutor` (same runner, same ControlMaster socket);
* :mod:`b20mlip.cluster.discover` — ``bootstrap(cfg, ctx, *, runner)``: discover account,
  partitions, QOS, scratch and modules into ``configs/cluster/<alias>.yaml``, sync the repo,
  check QE, optionally submit the LAMMPS build;
* :mod:`b20mlip.cluster.sync` — ``pull(cfg, ctx, *, runner, jobs)``: sacct + rsync of markers,
  logs and results (markers are the only source of truth for "done");
* :mod:`b20mlip.cluster.status` — ``show(cfg, *, runner)``: squeue plus local marker counts.

Nothing here opens a socket on its own: every ssh/rsync call goes through the injectable runner.
"""

from __future__ import annotations

from b20mlip.cluster.remote import COMMANDS, MFA_INSTRUCTION, Transport, render_command

__all__ = ["COMMANDS", "MFA_INSTRUCTION", "Transport", "render_command"]
