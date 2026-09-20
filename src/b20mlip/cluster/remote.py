"""Remote plumbing shared by ``cluster bootstrap | sync | status``.

* :data:`COMMANDS` is the single table of every command this tier runs — remote shell strings
  and the two local transport commands (``ssh -O check``, ``rsync``) — as ``str.format``
  templates. Tests render and assert them; bootstrap/sync manifests record the rendered strings.
* :class:`Transport` reuses :class:`~b20mlip.executors.SlurmExecutor`: the same injectable
  runner, the same ControlMaster socket, the same ``rc == 255 -> ClusterUnreachable`` rule and
  the same template rendering/submission. Remote commands are wrapped in ``bash -lc`` so that
  Lmod (``module``) is initialised — a plain ``ssh host cmd`` shell has no ``module`` function.
* The parsers turn ``hyakalloc``, ``sacctmgr -P``, ``sinfo``, ``module avail``/``spider``,
  ``sacct -P`` and ``squeue`` output into plain dataclasses and tolerate junk lines.

Verified facts this module encodes (hyak.uw.edu docs and UWrc/tillicum-onboarding, 2026-09-17):
``hyakalloc`` prints a box table ``Account | Partition | CPUs | Memory | GPUs`` with
TOTAL/USED/FREE rows and is a Klone tool that may be absent on Tillicum; Tillicum's documented
partitions are ``gpu-h200`` and ``gpu-h200-mig`` (every job needs >= 1 GPU); scratch is
``/gpfs/scrubbed/<user>`` (60-day purge) or ``/gpfs/projects/<group>``; modules are Lmod with a
compiler hierarchy (``cuda`` only visible after ``gcc``), so ``module spider`` is authoritative.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from b20mlip.config import Settings
from b20mlip.executors import (
    TERMINAL_STATES,
    ClusterUnreachable,
    CommandResult,
    Runner,
    SlurmExecutor,
    subprocess_runner,
)

MFA_INSTRUCTION = "run `ssh -fN {alias}` (MFA) then retry"
REPO_DIRNAME = "b20-mlip"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
MICROMAMBA_URL = (
    "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
)
QE_CONDA_SPEC = "qe=7.5"
LAMMPS_REPO = "https://github.com/ACEsuit/lammps"
LAMMPS_BRANCH = "mace"
MODULE_QUERIES: tuple[str, ...] = (
    "quantum-espresso",
    "espresso",
    "qe",
    "lammps",
    "cuda",
    "gcc",
    "cmake",
    "openmpi",
)
# rsync excludes for the repo push (SPEC: .venv, data/raw, runs, models, .git) plus caches and
# the large local-only trees; and for pulling results back (QE wavefunctions, build trees).
# Anchored to the transfer root ("/dft" = <repo>/dft only): an unanchored "dft" also matched
# src/b20mlip/dft and the package was missing on the cluster (2026-09-20).
REPO_EXCLUDES: tuple[str, ...] = (
    "/.venv",
    "/data/raw",
    "/data/omat24",
    "/runs",
    "/models",
    "/dft",
    "/.git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".coverage",
)
RESULT_EXCLUDES: tuple[str, ...] = (
    "*.save",
    "*.wfc*",
    "*.mix*",
    "*.igk*",
    "*.hub",
    "build",
    "lammps-src",
    "libtorch*",
    "*.zip",
    ".venv",
)

# Every command, local or remote. Braces are str.format placeholders; shell braces are doubled.
# Remote entries run as ``bash -lc '<rendered>'`` over the ControlMaster socket.
COMMANDS: dict[str, str] = {
    # -- local transport (argv templates; rendered, then shlex.split) --
    "socket_check": "ssh -S {control_path} -O check {alias}",
    "rsync_push": "rsync -az {excludes} -e {transport} {src} {dst}",
    "rsync_pull": "rsync -az --prune-empty-dirs {excludes} -e {transport} {src} {dst}",
    # -- bootstrap: discovery --
    "whoami": "whoami",
    "hyakalloc": "hyakalloc",
    "sacctmgr_assoc": (
        'sacctmgr show assoc user="$USER" format=account,partition,qos,defaultqos -P'
    ),
    "sacctmgr_user": 'sacctmgr show user "$USER" format=user,defaultaccount -P',
    "sinfo": 'sinfo -o "%P %a %l %D %G"',
    "scratch_env": 'echo "${{SCRATCH:-}}"',
    "scratch_probe": (
        'test -d "{parent}" && d="{path}" && mkdir -p "$d/b20-mlip" && test -w "$d/b20-mlip" '
        '&& echo "$d"'
    ),
    "module_grep": 'module avail 2>&1 | grep -i -E "quantum|espresso|qe|lammps|cuda"',
    "module_spider": 'for m in {names}; do echo "### $m"; module spider "$m" 2>&1; done',
    "qe_which": "module load {modules} && which pw.x",
    "module_load_check": (
        "module purge >/dev/null 2>&1; module load {modules} 2>&1 && echo B20_MODULES_OK"
    ),
    "qe_probe": "test -x {scratch}/qe/bin/pw.x && echo {scratch}/qe/bin/pw.x",
    # -- bootstrap: repo, uv, QE fallback, LAMMPS --
    "remote_layout": "bash {repo}/scripts/tillicum/remote_bootstrap.sh {scratch}",
    "which_uv": 'command -v uv || test -x "$HOME/.local/bin/uv"',
    "uv_install": "curl -LsSf {url} | sh",
    "uv_sync": "bash {repo}/scripts/tillicum/uv_install.sh {scratch} {repo}",
    "micromamba_qe": "bash {repo}/scripts/tillicum/micromamba_qe.sh {scratch}",
    "lammps_probe": "test -x {prefix}/bin/lmp && cat {prefix}/bin/build.json",
    "lammps_fetch": (
        "B20_PREFIX={prefix} B20_LIBTORCH_URL={url} B20_LAMMPS_SHA={sha} "
        "bash {repo}/scripts/tillicum/build_lammps.sh fetch"
    ),
    # -- sync / status --
    "list_jobs": "ls -1 {jobs_dir}",
    "sacct": "sacct -j {ids} -X -P -o JobID,State,ExitCode,Elapsed,NodeList",
    "squeue": 'squeue -u "$USER" -o "%i %j %T %M %P %R"',
}
LOCAL_COMMANDS: frozenset[str] = frozenset({"socket_check", "rsync_push", "rsync_pull"})


def render_command(key: str, **fmt: Any) -> str:
    """Render ``COMMANDS[key]`` (always through ``str.format`` so doubled braces collapse)."""
    try:
        return COMMANDS[key].format(**fmt)
    except KeyError as exc:  # unknown key or a missing placeholder
        raise KeyError(f"COMMANDS[{key!r}]: {exc}") from exc


def q(value: str) -> str:
    """``shlex.quote`` alias for rendering paths into remote commands."""
    return shlex.quote(value)


# --- transport ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandRecord:
    key: str
    command: str
    returncode: int
    remote: bool


class Transport:
    """Every ssh/rsync call of one stage, recorded; built on :class:`SlurmExecutor`."""

    def __init__(
        self,
        cfg: Settings,
        runner: Runner | None = None,
        *,
        template_dir: str | Path | None = None,
        staging_root: str | Path | None = None,
        records: list[CommandRecord] | None = None,
    ) -> None:
        self.cfg = cfg
        self.executor = SlurmExecutor(
            cfg,
            runner=runner or subprocess_runner,
            template_dir=template_dir,
            staging_root=staging_root,
        )
        self.records: list[CommandRecord] = records if records is not None else []

    # -- facts -------------------------------------------------------------------------------

    @property
    def alias(self) -> str:
        return self.cfg.cluster.alias

    @property
    def control_path(self) -> str:
        return self.executor.control_path

    @property
    def mfa_instruction(self) -> str:
        return MFA_INSTRUCTION.format(alias=self.alias)

    @property
    def used(self) -> set[str]:
        return {r.key for r in self.records}

    def with_settings(self, cfg: Settings) -> Transport:
        """Same runner, templates, staging and record list; new (e.g. discovered) settings."""
        return Transport(
            cfg,
            self.executor.runner,
            template_dir=self.executor.template_dir,
            staging_root=self.executor.staging_root,
            records=self.records,
        )

    def _record(self, key: str, command: str, res: CommandResult, *, remote: bool) -> None:
        self.records.append(CommandRecord(key, command, res.returncode, remote))

    def records_json(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]

    # -- commands ----------------------------------------------------------------------------

    def check_socket(self) -> None:
        """``ssh -O check`` on the ControlMaster socket; dead -> :class:`ClusterUnreachable`."""
        cmd = render_command("socket_check", control_path=q(self.control_path), alias=self.alias)
        res = self.executor.runner(shlex.split(cmd))
        self._record("socket_check", cmd, res, remote=False)
        if not res.ok:
            raise ClusterUnreachable(
                f"no live ControlMaster socket at {self.control_path} for {self.alias!r} "
                f"(rc {res.returncode}): {self.mfa_instruction}"
            )

    def run(self, key: str, **fmt: Any) -> CommandResult:
        """Run ``COMMANDS[key]`` remotely in a login shell; never raises on a non-zero exit."""
        cmd = render_command(key, **fmt)
        try:
            res = self.executor._ssh(f"bash -lc {q(cmd)}")
        except ClusterUnreachable as exc:
            self.records.append(CommandRecord(key, cmd, 255, True))
            raise ClusterUnreachable(f"{exc}; {self.mfa_instruction}") from exc
        self._record(key, cmd, res, remote=True)
        return res

    def rsync(self, key: str, src: str, dst: str, *, excludes: Iterable[str] = ()) -> CommandResult:
        """``rsync -az`` over the same ControlMaster transport as the executor."""
        cmd = render_command(
            key,
            excludes=" ".join(f"--exclude={q(e)}" for e in excludes),
            transport=q(self.executor.ssh_transport),
            src=q(src),
            dst=q(dst),
        )
        res = self.executor.runner(shlex.split(cmd))
        self._record(key, cmd, res, remote=False)
        if res.returncode == 255:
            raise ClusterUnreachable(
                f"rsync transport to {self.alias!r} failed: {self.mfa_instruction}"
            )
        if not res.ok:
            raise RuntimeError(f"{key} failed (rc {res.returncode}): {res.stderr.strip()}")
        return res


# --- parsers: allocations ------------------------------------------------------------------------


def _is_int(text: str) -> bool:
    return re.fullmatch(r"\d+", text.strip()) is not None


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass(frozen=True)
class AllocRow:
    """One TOTAL row of the ``hyakalloc`` table."""

    account: str
    partition: str
    cpus: int
    memory: str
    gpus: int


def parse_hyakalloc(text: str) -> list[AllocRow]:
    """Parse the ``hyakalloc`` box table (``│`` or ``|`` separated).

    Only rows carrying an account and a partition are allocations (the TOTAL row of each block
    or a plain one-row-per-partition table); USED/FREE rows, the ``Usage`` (``0/40``) tables
    and the ``Checkpoint Resources`` block have no account cell and are skipped.
    """
    rows: list[AllocRow] = []
    for line in text.splitlines():
        if "│" not in line and "|" not in line:
            continue
        cells = [c.strip() for c in re.split(r"[│|]", line)]
        cells = [c for c in cells if c]
        if len(cells) < 5 or cells[0].lower() == "account":
            continue
        account, partition, cpus, memory, gpus = cells[:5]
        if not _is_int(cpus) or not _is_int(gpus):
            continue
        if len(cells) >= 6 and cells[5].upper() != "TOTAL":
            continue
        rows.append(AllocRow(account, partition, int(cpus), memory, int(gpus)))
    return rows


@dataclass(frozen=True)
class AssocRow:
    """One ``sacctmgr show assoc ... -P`` row."""

    account: str
    partition: str
    qos: tuple[str, ...]
    default_qos: str


def _pipe_fields(line: str) -> list[str]:
    parts = line.rstrip("\n").split("|")
    if len(parts) > 1 and parts[-1] == "":
        parts.pop()  # ``-P`` terminates every row with a pipe
    return [p.strip() for p in parts]


def parse_sacctmgr_assoc(text: str) -> list[AssocRow]:
    rows: list[AssocRow] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = _pipe_fields(line)
        if parts[0].lower() in ("account", "cluster"):
            continue
        parts += [""] * (4 - len(parts))
        qos = tuple(x.strip() for x in parts[2].split(",") if x.strip())
        rows.append(AssocRow(parts[0], parts[1], qos, parts[3]))
    return rows


def parse_sacctmgr_user(text: str) -> str | None:
    """Default account from ``sacctmgr show user <u> format=user,defaultaccount -P``."""
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = _pipe_fields(line)
        if parts[0].lower() == "user":
            continue
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return None


# --- parsers: partitions -------------------------------------------------------------------------


@dataclass
class PartitionInfo:
    """Aggregated ``sinfo -o "%P %a %l %D %G"`` lines of one partition."""

    name: str
    default: bool = False
    avail: str = ""
    timelimit: str = ""
    nodes: int = 0
    gres: list[str] = field(default_factory=list)

    @property
    def gpu(self) -> bool:
        return any("gpu" in g.lower() or "h200" in g.lower() for g in self.gres)

    @property
    def up(self) -> bool:
        return self.avail.lower() == "up"


def parse_sinfo(text: str) -> dict[str, PartitionInfo]:
    parts: dict[str, PartitionInfo] = {}
    for line in text.splitlines():
        fields = line.split(None, 4)
        if len(fields) < 4 or fields[0].upper() == "PARTITION":
            continue
        raw_name, avail, timelimit, nodes = fields[:4]
        gres = fields[4].strip() if len(fields) > 4 else ""
        name = raw_name.rstrip("*")
        info = parts.setdefault(name, PartitionInfo(name))
        info.default = info.default or raw_name.endswith("*")
        info.avail = info.avail or avail
        info.timelimit = info.timelimit or timelimit
        info.nodes += int(nodes) if _is_int(nodes) else 0
        if gres and gres.lower() != "(null)" and gres not in info.gres:
            info.gres.append(gres)
    return parts


# --- parsers: modules ----------------------------------------------------------------------------

_MODULE_TOKEN = re.compile(r"(?<![\w/.+-])([A-Za-z][\w+.-]*/[0-9][\w+.-]*)")


def version_key(name: str) -> tuple[tuple[int, Any], ...]:
    """Sort key for ``name/1.2.3``-style module names (numeric parts compare numerically)."""
    version = name.split("/", 1)[1] if "/" in name else name
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"[._-]", version))


def parse_module_grep(text: str) -> tuple[list[str], list[str]]:
    """``module avail | grep``: ``(module names, names marked (D) default)``."""
    names: list[str] = []
    defaults: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("-"):
            continue
        for match in _MODULE_TOKEN.finditer(line):
            token = match.group(1)
            names.append(token)
            tail = line[match.end() :].lstrip()
            if tail.startswith("(D)"):
                defaults.append(token)
    return unique(names), unique(defaults)


@dataclass
class ModuleQuery:
    """Result of ``module spider <name>``: versions found and their prerequisites."""

    name: str
    versions: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.versions)

    @property
    def newest(self) -> str | None:
        return max(self.versions, key=version_key) if self.versions else None


def parse_module_spider(text: str) -> dict[str, ModuleQuery]:
    """Split the ``### <name>`` sections written by ``COMMANDS["module_spider"]``."""
    queries: dict[str, ModuleQuery] = {}
    current: ModuleQuery | None = None
    prereq_mode = False
    for line in text.splitlines():
        header = re.match(r"^### (\S+)\s*$", line)
        if header:
            current = ModuleQuery(header.group(1))
            queries[current.name] = current
            prereq_mode = False
            continue
        if current is None:
            continue
        if "before the" in line and ("module" in line or "available to load" in line):
            prereq_mode = True  # the sentence may be wrapped over two lines
            continue
        if prereq_mode and "available to load" in line:
            continue
        tokens = [
            t
            for t in (m.group(1) for m in _MODULE_TOKEN.finditer(line))
            if t.split("/", 1)[0].lower() == current.name.lower()
            or t.lower().startswith(current.name.lower())
        ]
        if prereq_mode:
            if line.strip() == "" and current.prerequisites:
                prereq_mode = False
            for m in _MODULE_TOKEN.finditer(line):
                if m.group(1) not in current.prerequisites:
                    current.prerequisites.append(m.group(1))
            continue
        for token in tokens:
            if token not in current.versions:
                current.versions.append(token)
    return queries


# --- parsers: accounting -------------------------------------------------------------------------


@dataclass(frozen=True)
class SacctJob:
    """One ``sacct -X -P -o JobID,State,ExitCode,Elapsed,NodeList`` row."""

    job_id: str
    state: str
    exit_code: str
    elapsed: str
    nodelist: str

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def parse_sacct_jobs(text: str) -> list[SacctJob]:
    rows: list[SacctJob] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = _pipe_fields(line)
        if parts[0].lower() == "jobid" or "." in parts[0]:
            continue
        parts += [""] * (5 - len(parts))
        state = parts[1].split()[0].upper() if parts[1] else "UNKNOWN"
        rows.append(SacctJob(parts[0], state, parts[2], parts[3], parts[4]))
    return rows


def nodelist_count(nodelist: str) -> int:
    """Number of hosts in a SLURM hostlist (``g[001-003],g010`` -> 4; ``None assigned`` -> 0)."""
    text = nodelist.strip()
    if not text or text.lower().startswith("none"):
        return 0
    total = 0
    for item in re.split(r",(?![^\[]*\])", text):
        m = re.search(r"\[([^\]]*)\]", item)
        if not m:
            total += 1
            continue
        for rng in m.group(1).split(","):
            lo, _, hi = rng.partition("-")
            if hi and _is_int(lo) and _is_int(hi):
                total += int(hi) - int(lo) + 1
            else:
                total += 1
    return total


@dataclass(frozen=True)
class QueueRow:
    """One ``squeue -o "%i %j %T %M %P %R"`` row."""

    job_id: str
    name: str
    state: str
    time: str
    partition: str
    reason: str


def parse_squeue(text: str) -> list[QueueRow]:
    rows: list[QueueRow] = []
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 3 or parts[0].upper() == "JOBID":
            continue
        parts += [""] * (6 - len(parts))
        rows.append(QueueRow(*parts[:6]))
    return rows


__all__ = [
    "COMMANDS",
    "LAMMPS_BRANCH",
    "LAMMPS_REPO",
    "LOCAL_COMMANDS",
    "MFA_INSTRUCTION",
    "MICROMAMBA_URL",
    "MODULE_QUERIES",
    "QE_CONDA_SPEC",
    "REPO_DIRNAME",
    "REPO_EXCLUDES",
    "RESULT_EXCLUDES",
    "UV_INSTALL_URL",
    "AllocRow",
    "AssocRow",
    "CommandRecord",
    "ModuleQuery",
    "PartitionInfo",
    "QueueRow",
    "SacctJob",
    "Transport",
    "nodelist_count",
    "parse_hyakalloc",
    "parse_module_grep",
    "parse_module_spider",
    "parse_sacct_jobs",
    "parse_sacctmgr_assoc",
    "parse_sacctmgr_user",
    "parse_sinfo",
    "parse_squeue",
    "q",
    "render_command",
    "unique",
    "version_key",
]
