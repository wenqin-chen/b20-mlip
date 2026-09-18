"""Offline fixtures for the cluster tier: a FakeRunner scripted with realistic Hyak outputs.

The outputs follow the formats documented at hyak.uw.edu (the ``hyakalloc`` box table with
TOTAL/USED/FREE rows, Lmod ``module avail``/``spider`` text, Tillicum's ``gpu-h200`` and
``gpu-h200-mig`` partitions with an H200 GRES line) and the standard SLURM ``-P`` / ``-o``
formats. Two presets: ``simple`` (Klone-like: hyakalloc available, one CPU + one GPU
partition, a QE module) and ``tillicum`` (no hyakalloc, two GPU partitions and no CPU-only
partition, no QE module). Nothing here opens a socket; every ssh/rsync argv is recorded.
"""

from __future__ import annotations

import re
import shlex
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from b20mlip.config import Settings
from b20mlip.executors import CommandResult

USER = "wenqin98"
SCRATCH = f"/gpfs/scrubbed/{USER}"
BUILD_SHA = "4d222cb3ee2a6b14083c778968497bf9e0efc4b4"

HYAKALLOC_SIMPLE = f"""\
Account resources available to user: {USER}

╭─────────┬─────────────┬──────┬────────┬──────┬───────╮
│ Account │   Partition │ CPUs │ Memory │ GPUs │       │
├─────────┼─────────────┼──────┼────────┼──────┼───────┤
│     b20 │     compute │  320 │  1400G │    0 │ TOTAL │
│         │             │   16 │    64G │    0 │ USED  │
│         │             │  304 │  1336G │    0 │ FREE  │
│     b20 │    gpu-h200 │   64 │  1600G │    8 │ TOTAL │
│         │             │    0 │     0G │    0 │ USED  │
│         │             │   64 │  1600G │    8 │ FREE  │
╰─────────┴─────────────┴──────┴────────┴──────┴───────╯

Checkpoint Resources
╭───────┬──────┬──────╮
│       │ CPUs │ GPUs │
├───────┼──────┼──────┤
│ Idle: │  160 │   24 │
╰───────┴──────┴──────╯
"""
ASSOC_SIMPLE = "Account|Partition|QOS|Def QOS|\nb20||normal,debug,interactive|normal|\n"
ASSOC_TWO_ACCOUNTS = (
    "Account|Partition|QOS|Def QOS|\n"
    "b20||normal,debug|normal|\n"
    "stf||normal,debug,interactive|normal|\n"
)
USER_SIMPLE = f"User|Def Acct|\n{USER}|b20|\n"
SINFO_SIMPLE = """\
PARTITION AVAIL TIMELIMIT NODES GRES
compute* up 7-00:00:00 100 (null)
compute* up 7-00:00:00 12 (null)
gpu-h200 up 2-00:00:00 4 gpu:h200:8
ckpt-all up 4:00:00 200 (null)
broken down 1:00:00 2 (null)
"""
SINFO_TILLICUM = """\
PARTITION AVAIL TIMELIMIT NODES GRES
gpu-h200* up 1-00:00:00 20 gpu:h200:8
gpu-h200-mig up 1-00:00:00 4 gpu:h200_1g.18gb:56
"""
MODULE_GREP_SIMPLE = """\
   quantum-espresso/7.2          quantum-espresso/7.3.1 (D)
   cuda/12.4.0                   cuda/12.9.1            (D)    cuda/13.0.0
"""
SPIDER_COMMON = """\
### lammps

Lmod has detected the following error: Unable to find: "lammps".

### cuda

------------------------------------------------------------------------------------------------
  cuda:
------------------------------------------------------------------------------------------------
    Description:
      NVIDIA CUDA Toolkit for GPU-accelerated computing.

     Versions:
        cuda/12.4.0
        cuda/12.9.1
        cuda/13.0.0

------------------------------------------------------------------------------------------------
  For detailed information about a specific "cuda" package (including how to load the modules)
  use the module's full name.
  For example:

     $ module spider cuda/13.0.0
------------------------------------------------------------------------------------------------

### gcc

------------------------------------------------------------------------------------------------
  gcc:
------------------------------------------------------------------------------------------------
     Versions:
        gcc/11.5.0
        gcc/13.4.0

### cmake

------------------------------------------------------------------------------------------------
  cmake: cmake/3.31.8
------------------------------------------------------------------------------------------------

    You will need to load all module(s) on any one of the lines below before the "cmake/3.31.8"
    module is available to load.

      gcc/13.4.0

    Help:
      Adds CMake 3.31.8 to your environment.

### openmpi

------------------------------------------------------------------------------------------------
  openmpi: openmpi/5.0.8
------------------------------------------------------------------------------------------------

    You will need to load all module(s) on any one of the lines below before the
    "openmpi/5.0.8" module is available to load.

      gcc/13.4.0
"""
SPIDER_QE = """\
### quantum-espresso

------------------------------------------------------------------------------------------------
  quantum-espresso:
------------------------------------------------------------------------------------------------
     Versions:
        quantum-espresso/7.2
        quantum-espresso/7.3.1

### espresso

Lmod has detected the following error: Unable to find: "espresso".

### qe

Lmod has detected the following error: Unable to find: "qe".

"""
SPIDER_NO_QE = """\
### quantum-espresso

Lmod has detected the following error: Unable to find: "quantum-espresso".

### espresso

Lmod has detected the following error: Unable to find: "espresso".

### qe

Lmod has detected the following error: Unable to find: "qe".

"""
QE_PATH = "/sw/contrib/quantum-espresso/7.3.1/bin/pw.x"
SQUEUE = """\
JOBID NAME STATE TIME PARTITION NODELIST(REASON)
4242 build_lammps RUNNING 12:34 gpu-h200 g001
4243_[1-40] qe_r0 PENDING 0:00 gpu-h200-mig (Priority)
4243_41 qe_r0 RUNNING 3:05 gpu-h200-mig g004
"""
SACCT_DONE = (
    "JobID|State|ExitCode|Elapsed|NodeList|\n"
    "4243_1|COMPLETED|0:0|00:03:10|g004|\n"
    "4243_2|COMPLETED|0:0|00:02:59|g004|\n"
    "4243_3|TIMEOUT|0:1|02:00:01|g[005-006]|\n"
    "4243_3.batch|TIMEOUT|0:1|02:00:01|g[005-006]|\n"
)
SACCT_RUNNING = (
    "JobID|State|ExitCode|Elapsed|NodeList|\n"
    "4243_1|COMPLETED|0:0|00:03:10|g004|\n"
    "4243_[2-3]|PENDING|0:0|00:00:00|None assigned|\n"
)
BUILD_JSON = (
    '{"lammps_repo": "https://github.com/ACEsuit/lammps", "branch": "mace", '
    f'"sha": "{BUILD_SHA}", "cuda": true, "fallback": null, "libtorch": "2.14.0+cu126", '
    f'"log": "{SCRATCH}/b20-mlip/jobs/build_lammps/logs/build.log"}}\n'
)


@dataclass
class Scenario:
    """Knobs of the fake cluster (mutable: the fake flips some of them as side effects)."""

    socket_ok: bool = True
    ssh_rc: int = 0
    die_after_ssh: int | None = None  # simulate socket death after N remote commands
    hyakalloc_available: bool = True
    hyakalloc: str = HYAKALLOC_SIMPLE
    assoc: str = ASSOC_SIMPLE
    user: str = USER_SIMPLE
    sinfo: str = SINFO_SIMPLE
    scratch_env: str = ""
    writable: tuple[str, ...] = (SCRATCH,)
    module_grep: str = MODULE_GREP_SIMPLE
    module_spider: str = SPIDER_QE + SPIDER_COMMON
    qe_module_ok: bool = True
    qe_installed: bool = False
    uv_present: bool = True
    lmp_built: bool = False
    build_json: str = BUILD_JSON
    sacct: list[str] = field(default_factory=list)
    squeue: str = SQUEUE
    jobs: list[str] = field(default_factory=list)
    remote_user: str = USER
    rsync_rc: int = 0
    sbatch_id: str = "4242"


def simple_scenario(**overrides: object) -> Scenario:
    return Scenario(**overrides)  # type: ignore[arg-type]


def tillicum_scenario(**overrides: object) -> Scenario:
    base = dict(
        hyakalloc_available=False,
        sinfo=SINFO_TILLICUM,
        module_grep="",
        module_spider=SPIDER_NO_QE + SPIDER_COMMON,
        qe_module_ok=False,
    )
    base.update(overrides)
    return Scenario(**base)  # type: ignore[arg-type]


def unwrap(remote: str) -> str:
    """``bash -lc '<cmd>'`` -> ``<cmd>`` (other remote strings pass through)."""
    if remote.startswith("bash -lc "):
        return shlex.split(remote)[2]
    return remote


class FakeRunner:
    """Answers every ssh/rsync argv of the cluster tier without touching a socket.

    ``remote_root`` simulates ``<scratch>/b20-mlip`` on the cluster: pulls copy files out of it
    into the local destination, pushes copy staging dirs into it, ``ls -1 <jobs>`` lists it.
    """

    def __init__(self, scenario: Scenario | None = None, remote_root: Path | None = None) -> None:
        self.s = scenario or Scenario()
        self.remote_root = remote_root
        self.calls: list[list[str]] = []
        self.remote_cmds: list[str] = []
        self.ssh_count = 0
        self._sacct: Iterator[str] = iter(self.s.sacct)

    # -- helpers -----------------------------------------------------------------------------

    def remote_commands(self) -> list[str]:
        return list(self.remote_cmds)

    def rsyncs(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "rsync"]

    def _remote_path(self, spec: str) -> Path:
        assert self.remote_root is not None, "remote_root needed for this rsync"
        path = spec.split(":", 1)[1].rstrip("/")
        marker = "/b20-mlip/"
        assert marker in path + "/", path
        rel = (path + "/").split(marker, 1)[1].rstrip("/")
        return self.remote_root / rel if rel else self.remote_root

    # -- dispatch ----------------------------------------------------------------------------

    def __call__(self, argv: list[str]) -> CommandResult:
        self.calls.append(list(argv))
        tup = tuple(argv)
        if argv[0] == "ssh" and "-O" in argv:
            if self.s.socket_ok:
                return CommandResult(tup, 0, "Master running (pid=4711)\n", "")
            return CommandResult(
                tup, 255, "", "Control socket connect(cm): No such file or directory"
            )
        if argv[0] == "rsync":
            return self._rsync(argv, tup)
        if argv[0] == "ssh":
            self.ssh_count += 1
            if self.s.ssh_rc:
                return CommandResult(tup, self.s.ssh_rc, "", "Connection closed")
            if self.s.die_after_ssh is not None and self.ssh_count > self.s.die_after_ssh:
                return CommandResult(tup, 255, "", "Connection closed by remote host")
            cmd = unwrap(argv[-1])
            self.remote_cmds.append(cmd)
            return self._respond(cmd, tup)
        raise AssertionError(f"unexpected local command {argv}")

    def _rsync(self, argv: list[str], tup: tuple[str, ...]) -> CommandResult:
        if self.s.rsync_rc:
            return CommandResult(tup, self.s.rsync_rc, "", "rsync error")
        src, dst = argv[-2], argv[-1]
        if src.startswith("tillicum:"):  # pull
            remote = self._remote_path(src)
            if remote.is_dir():
                shutil.copytree(remote, Path(dst), dirs_exist_ok=True)
            return CommandResult(tup, 0, "", "")
        if dst.startswith("tillicum:") and self.remote_root is not None:  # push
            remote = self._remote_path(dst)
            local = Path(src)
            if local.is_dir() and "jobs/" in dst:  # staging dirs only; never the whole repo
                shutil.copytree(local, remote, dirs_exist_ok=True)
        return CommandResult(tup, 0, "", "")

    def _respond(self, cmd: str, tup: tuple[str, ...]) -> CommandResult:
        s = self.s

        def ok(out: str = "") -> CommandResult:
            return CommandResult(tup, 0, out, "")

        def fail(rc: int = 1, err: str = "") -> CommandResult:
            return CommandResult(tup, rc, "", err)

        if cmd == "whoami":
            return ok(s.remote_user + "\n")
        if cmd == "hyakalloc":
            if not s.hyakalloc_available:
                return fail(127, "bash: hyakalloc: command not found")
            return ok(s.hyakalloc)
        if cmd.startswith("sacctmgr show assoc"):
            return ok(s.assoc)
        if cmd.startswith("sacctmgr show user"):
            return ok(s.user)
        if cmd.startswith("sinfo"):
            return ok(s.sinfo)
        if cmd.startswith('echo "${SCRATCH:-}"'):
            return ok(s.scratch_env + "\n")
        if cmd.startswith("test -d "):
            m = re.match(r'test -d "([^"]+)" && d="([^"]+)"', cmd)
            assert m, cmd
            resolved = m.group(2).replace("$USER", s.remote_user)
            if resolved in s.writable:
                return ok(resolved + "\n")
            return fail(1)
        if cmd.startswith("module avail"):
            return ok(s.module_grep) if s.module_grep else fail(1)
        if cmd.startswith("for m in"):
            return ok(s.module_spider)
        if cmd.startswith("module load") and "which pw.x" in cmd:
            return ok(QE_PATH + "\n") if s.qe_module_ok else fail(1, "pw.x not found")
        if cmd.startswith("test -x") and "/qe/bin/pw.x" in cmd:
            path = cmd.split("echo ", 1)[1].strip()
            return ok(path + "\n") if s.qe_installed else fail(1)
        if "remote_bootstrap.sh" in cmd:
            return ok("layout ok\n")
        if cmd.startswith("command -v uv"):
            return ok("/gpfs/home/x/.local/bin/uv\n") if s.uv_present else fail(1)
        if cmd.startswith("curl -LsSf") and "astral.sh/uv/install.sh" in cmd:
            s.uv_present = True
            return ok("installing to /gpfs/home/x/.local/bin\n")
        if "uv_install.sh" in cmd:
            return ok("uv 0.8.11 (2026-08-01)\nResolved 120 packages\nuv sync ok\n")
        if "micromamba_qe.sh" in cmd:
            s.qe_installed = True
            return ok("qe: installed\n")
        if cmd.startswith("test -x") and "/bin/lmp" in cmd:
            return ok(s.build_json) if s.lmp_built else fail(1)
        if "build_lammps.sh fetch" in cmd:
            return ok(f"libtorch: 2.14.0+cu126\nlammps: checkout {BUILD_SHA}\n")
        if cmd.startswith("mkdir -p"):
            return ok()
        if "sbatch --parsable" in cmd:
            return ok(f"{s.sbatch_id}\n")
        if cmd.startswith("sacct"):
            return ok(next(self._sacct, ""))
        if cmd.startswith("squeue"):
            return ok(s.squeue)
        if cmd.startswith("ls -1"):
            if self.remote_root is not None and (self.remote_root / "jobs").is_dir():
                names = sorted(p.name for p in (self.remote_root / "jobs").iterdir() if p.is_dir())
                return ok("".join(f"{n}\n" for n in names))
            return ok("".join(f"{n}\n" for n in s.jobs))
        raise AssertionError(f"unscripted remote command: {cmd}")


# --- fixtures -----------------------------------------------------------------------------------


@pytest.fixture
def cluster_cfg(settings: Settings, tmp_path: Path) -> Settings:
    """Settings with tmp paths, the tillicum alias and a tmp control socket path."""
    return settings.model_copy(
        update={
            "cluster": settings.cluster.model_copy(
                update={"alias": "tillicum", "control_path": str(tmp_path / "cm.sock")}
            )
        }
    )


@pytest.fixture
def discovered_cfg(cluster_cfg: Settings) -> Settings:
    """As after a bootstrap: scratch and partitions known."""
    return cluster_cfg.model_copy(
        update={
            "cluster": cluster_cfg.cluster.model_copy(
                update={
                    "account": "b20",
                    "partition_cpu": "compute",
                    "partition_gpu": "gpu-h200",
                    "qos": "normal",
                    "scratch": SCRATCH,
                    "modules": ["gcc/13.4.0", "cuda/12.9.1"],
                }
            )
        }
    )


@pytest.fixture
def yaml_path(tmp_path: Path, repo: Path) -> Path:
    """A copy of the repo's placeholder overlay to write into."""
    dst = tmp_path / "configs" / "cluster" / "tillicum.yaml"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(repo / "configs" / "cluster" / "tillicum.yaml", dst)
    return dst


@pytest.fixture
def remote_root(tmp_path: Path) -> Path:
    """Simulated ``<scratch>/b20-mlip`` on the cluster."""
    root = tmp_path / "remote" / "b20-mlip"
    (root / "jobs").mkdir(parents=True)
    return root
