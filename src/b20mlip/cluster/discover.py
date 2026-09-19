"""``b20mlip cluster bootstrap`` (SPEC.md section 8): discover the cluster into the YAML overlay.

Steps, each recorded under ``extras["steps"][<name>]`` of the ``cluster.bootstrap`` manifest,
individually skippable (``skip=``) and resumed from ``bootstrap_state.json`` in the run
directory when ``ctx.resume`` is set (``socket`` and ``config`` always re-run):

``socket``      ``ssh -O check`` on the ControlMaster socket; dead -> :class:`ClusterUnreachable`
                carrying the exact instruction ``run `ssh -fN tillicum` (MFA) then retry``.
``accounts``    ``hyakalloc`` (parsed) plus ``sacctmgr show assoc`` / ``show user``: account
                (unique, else SLURM's DefaultAccount, else ``null``) and QOS (DefaultQOS, else
                the only QOS, else ``null``).
``partitions``  ``sinfo -o "%P %a %l %D %G"``: ``partition_gpu`` = the *unique* partition whose
                GRES mentions gpu/h200, ``partition_cpu`` = the unique one without; two
                candidates -> ``null`` plus the raw listing, never a guess.
``scratch``     ``$SCRATCH``, ``/gpfs/projects/<account>/$USER``, ``/gpfs/scrubbed/$USER``,
                ``/gscratch/<account>/$USER``, ``/gscratch/scrubbed/$USER``: the first candidate
                whose parent exists and where ``mkdir -p <it>/b20-mlip`` is writable.
``modules``     ``module avail | grep`` plus ``module spider`` for QE / LAMMPS / cuda / gcc /
                cmake / openmpi (Lmod hierarchies hide ``cuda`` until ``gcc`` is loaded).
``config``      write ``configs/cluster/<alias>.yaml`` (comments kept, unknowns ``null``).
``repo``        rsync the checkout to ``<scratch>/b20-mlip``, create the remote layout, install
                ``uv`` with the official installer when missing, ``uv sync --frozen``.
``qe``          ``module load <found> && which pw.x``; else the micromamba plan is recorded and
                executed only with ``cluster.install_qe`` / ``--install-qe``.
``lammps``      ``<scratch>/b20-mlip/bin/lmp`` present -> ``lammps_cmd``; otherwise, only with
                ``--build-lammps``, submit ``templates/slurm/build_lammps.sbatch.j2``.

Every remote command comes from :data:`b20mlip.cluster.remote.COMMANDS`; the manifest records
the rendered strings and the raw listings so a wrong parser is visible at the first login.
"""

from __future__ import annotations

import json
import posixpath
import re
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from b20mlip.cluster.configfile import write_cluster_yaml
from b20mlip.cluster.remote import (
    COMMANDS,
    MICROMAMBA_URL,
    MODULE_QUERIES,
    QE_CONDA_SPEC,
    REPO_DIRNAME,
    REPO_EXCLUDES,
    UV_INSTALL_URL,
    ModuleQuery,
    PartitionInfo,
    Transport,
    parse_hyakalloc,
    parse_module_grep,
    parse_module_spider,
    parse_sacctmgr_assoc,
    parse_sacctmgr_user,
    parse_sinfo,
    q,
    render_command,
    unique,
    version_key,
)
from b20mlip.config import Settings, repo_root
from b20mlip.executors import ClusterUnreachable, JobSpec, Runner
from b20mlip.models import SlurmInfo, StageResult
from b20mlip.provenance import RunContext

STEPS: tuple[str, ...] = (
    "socket",
    "accounts",
    "partitions",
    "scratch",
    "modules",
    "config",
    "repo",
    "qe",
    "lammps",
)
ALWAYS_RERUN: frozenset[str] = frozenset({"socket", "config"})
STATE_NAME = "bootstrap_state.json"
DISCOVERY_NAME = "discovery.json"
BUILD_JOB_NAME = "build_lammps"
BUILD_TEMPLATE = "build_lammps"
BUILD_SCRIPT = 'bash "$B20_REPO/scripts/tillicum/build_lammps.sh" build'
DEFAULT_KOKKOS_ARCH = "HOPPER90"  # NVIDIA H200 = Hopper sm_90 -> CMake keyword Kokkos_ARCH_HOPPER90
BUILD_RESOURCES: dict[str, Any] = {"time": "04:00:00", "cpus_per_task": 8, "mem": "64G"}
# placeholders used to render every COMMANDS entry for a --dry-run plan
PLAN_PLACEHOLDERS: dict[str, str] = {
    key: f"<{key}>"
    for key in (
        "control_path",
        "alias",
        "excludes",
        "transport",
        "src",
        "dst",
        "parent",
        "path",
        "names",
        "modules",
        "scratch",
        "repo",
        "url",
        "prefix",
        "sha",
        "jobs_dir",
        "ids",
    )
}


def micromamba_plan(scratch: str) -> list[str]:
    """The QE fallback, exactly as ``scripts/tillicum/micromamba_qe.sh`` runs it."""
    return [
        f"curl -Ls {MICROMAMBA_URL} -o {scratch}/bin/micromamba "
        f"&& chmod +x {scratch}/bin/micromamba",
        f"MAMBA_ROOT_PREFIX={scratch}/.micromamba {scratch}/bin/micromamba create -y "
        f"-p {scratch}/qe -c conda-forge {QE_CONDA_SPEC}",
    ]


def cuda_major(libtorch_url: str) -> str | None:
    """``.../cu126/...`` -> ``"12"`` (the CUDA major a libtorch build needs)."""
    m = re.search(r"/cu(\d{2})(\d)/", libtorch_url)
    return m.group(1) if m else None


MODULES_OK = "B20_MODULES_OK"
_LMOD_SUGGESTION_RE = re.compile(r"^\s*module load ((?:[\w.+-]+/[\w.+-]+\s*)+)$", re.MULTILINE)


def lmod_suggestions(text: str) -> list[list[str]]:
    """Module combinations Lmod prints after a hierarchy error
    (``Or load any one of these options: module load gcc/13.4.0 cuda/12.8.2 openmpi/5.0.10``)."""
    return [line.split() for line in _LMOD_SUGGESTION_RE.findall(text)]


@dataclass
class Discovered:
    """Everything bootstrap found; ``config_values`` is what the YAML overlay receives."""

    remote_user: str | None = None
    hyakalloc_available: bool | None = None
    accounts: list[str] = field(default_factory=list)
    account: str | None = None
    account_source: str = ""
    qos: str | None = None
    qos_all: list[str] = field(default_factory=list)
    qos_source: str = ""
    allowed_partitions: list[str] = field(default_factory=list)
    partitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    partition_cpu: str | None = None
    partition_gpu: str | None = None
    partition_candidates: dict[str, list[str]] = field(
        default_factory=lambda: {"cpu": [], "gpu": []}
    )
    partition_default: str | None = None
    partition_notes: list[str] = field(default_factory=list)
    scratch: str | None = None
    scratch_probes: list[dict[str, Any]] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    modules_source: str = ""
    module_versions: dict[str, list[str]] = field(default_factory=dict)
    module_defaults: list[str] = field(default_factory=list)
    module_prerequisites: dict[str, list[str]] = field(default_factory=dict)
    qe_module: str | None = None
    lammps_module: str | None = None
    qe_cmd: str | None = None
    qe_status: str = "unknown"
    qe_plan: list[str] = field(default_factory=list)
    micromamba_env: str | None = None
    repo_remote: str | None = None
    uv_present: bool | None = None
    uv_installed_now: bool = False
    uv_version: str | None = None
    lammps_cmd: str | None = None
    lammps_status: str = "unknown"
    lammps_job_id: str | None = None
    lammps_workdir: str | None = None
    lammps_log: str | None = None
    lammps_build: dict[str, Any] | None = None
    raw: dict[str, str] = field(default_factory=dict)

    def config_values(self, cfg: Settings) -> dict[str, Any]:
        return {
            "alias": cfg.cluster.alias,
            "control_path": cfg.cluster.control_path,
            "account": self.account,
            "partition_cpu": self.partition_cpu,
            "partition_gpu": self.partition_gpu,
            "qos": self.qos,
            "scratch": self.scratch,
            "modules": list(self.modules),
            "qe_cmd": self.qe_cmd,
            "lammps_cmd": self.lammps_cmd,
            "micromamba_env": self.micromamba_env,
        }

    def settings(self, cfg: Settings) -> Settings:
        """``cfg`` with the discovered cluster values applied (for submissions)."""
        values = self.config_values(cfg)
        return cfg.model_copy(update={"cluster": cfg.cluster.model_copy(update=values)})


class Bootstrap:
    """State of one ``cluster bootstrap`` run; :func:`bootstrap` is the stage entry point."""

    def __init__(
        self,
        cfg: Settings,
        ctx: RunContext,
        transport: Transport,
        *,
        build_lammps: bool,
        install_qe: bool,
        partition: str | None,
        yaml_path: Path,
        skip: Iterable[str],
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.t = transport
        self.build_lammps = build_lammps
        self.install_qe = install_qe
        self.partition = partition
        self.yaml_path = yaml_path
        self.skip = set(skip)
        unknown = self.skip - set(STEPS)
        if unknown:
            raise ValueError(f"unknown bootstrap step(s) {sorted(unknown)}; known: {STEPS}")
        self.found = Discovered()
        self.steps: dict[str, dict[str, Any]] = {}

    # -- state ---------------------------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.ctx.out_dir / STATE_NAME

    def load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.found = Discovered(**state["found"])
            self.steps = dict(state.get("steps", {}))
        except (ValueError, TypeError, KeyError):
            self.found, self.steps = Discovered(), {}

    def save_state(self) -> None:
        payload = {"steps": self.steps, "found": asdict(self.found)}
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def log_extras(self) -> None:
        self.ctx.log(
            steps=dict(self.steps),
            discovery={k: v for k, v in asdict(self.found).items() if k != "raw"},
            raw=dict(self.found.raw),
            commands=self.t.records_json(),
            yaml_path=str(self.yaml_path),
            mfa_instruction=self.t.mfa_instruction,
        )

    # -- driver --------------------------------------------------------------------------------

    def run(self) -> StageResult:
        if self.ctx.resume:
            self.load_state()
        for name in STEPS:
            if name in self.skip:
                self.steps[name] = {"status": "skipped", "reason": "skipped by request"}
            elif (
                name not in ALWAYS_RERUN
                and self.steps.get(name, {}).get("status") in ("ok", "resumed")
                and self.ctx.resume
            ):
                self.steps[name] = {**self.steps[name], "status": "resumed"}
            else:
                self.run_step(name)
            self.save_state()
        return self.finish()

    def run_step(self, name: str) -> None:
        fn = getattr(self, f"step_{name}")
        self.steps.pop(name, None)
        try:
            fn()
        except ClusterUnreachable as exc:
            self.steps[name] = {
                "status": "failed",
                "error": str(exc),
                "instruction": self.t.mfa_instruction,
            }
            self.save_state()
            self.log_extras()
            raise
        except Exception as exc:  # recorded; later steps still run so one login reports all
            self.steps[name] = {"status": "failed", "error": repr(exc)}
            return
        self.steps.setdefault(name, {"status": "ok"})

    def finish(self) -> StageResult:
        if "config" not in self.skip:  # qe_cmd / lammps_cmd / micromamba_env are found late
            self.run_step("config")
            self.save_state()
        f = self.found
        discovery_path = self.ctx.out_dir / DISCOVERY_NAME
        discovery_path.write_text(
            json.dumps(
                {"found": asdict(f), "steps": self.steps, "commands": self.t.records_json()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.ctx.add_output(discovery_path, "json")
        self.log_extras()
        bad = [n for n, s in self.steps.items() if s.get("status") in ("failed", "blocked")]
        status: Any = "partial" if bad else "ok"
        summary: dict[str, float | int | str] = {
            "account": f.account or "null",
            "partition_cpu": f.partition_cpu or "null",
            "partition_gpu": f.partition_gpu or "null",
            "qos": f.qos or "null",
            "scratch": f.scratch or "null",
            "qe": f.qe_status,
            "lammps": f.lammps_status,
            "steps_failed": len(bad),
            "yaml": str(self.yaml_path),
        }
        if bad:
            summary["failed_steps"] = ",".join(bad)
        return StageResult(
            stage="cluster.bootstrap",
            run_id=self.ctx.run_id,
            manifest_path=str(self.ctx.manifest_path),
            status=status,
            outputs=list(self.ctx.outputs),
            summary=summary,
        )

    # -- helpers -------------------------------------------------------------------------------

    def _choose(
        self, name: str, configured: str | None, candidates: list[str], known: Iterable[str]
    ) -> str | None:
        known_set = set(known)
        if configured and configured in known_set:
            self.found.partition_notes.append(f"{name}: kept {configured!r} from the config")
            return configured
        if configured:
            self.found.partition_notes.append(
                f"{name}: configured {configured!r} is not a partition sinfo lists; ignored"
            )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            self.found.partition_notes.append(f"{name}: no candidate partition; left null")
        else:
            self.found.partition_notes.append(
                f"{name}: ambiguous {candidates}; left null — set it explicitly "
                f"(`--set cluster.{name}=NAME` or edit the YAML)"
            )
        return None

    @staticmethod
    def _pick_module(
        names: Iterable[str],
        avail: list[str],
        defaults: list[str],
        queries: dict[str, ModuleQuery],
        *,
        major: str | None = None,
    ) -> str | None:
        """Default-marked ``(D)`` version if seen, else the newest (optionally of one major)."""
        for name in names:
            versions = list(queries.get(name, ModuleQuery(name)).versions)
            versions += [m for m in avail if m.lower().startswith(name.lower() + "/")]
            versions = unique(versions)
            if major is not None:
                versions = [v for v in versions if v.split("/", 1)[1].startswith(major + ".")]
            if not versions:
                continue
            marked = [v for v in versions if v in defaults]
            return marked[0] if marked else max(versions, key=version_key)
        return None

    # -- steps ---------------------------------------------------------------------------------

    def step_socket(self) -> None:
        self.t.check_socket()
        res = self.t.run("whoami")
        self.found.raw["whoami"] = res.stdout + res.stderr
        if not res.ok:
            raise RuntimeError(f"login shell on {self.t.alias!r} failed: {res.stderr.strip()}")
        self.found.remote_user = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else None
        self.steps["socket"] = {"status": "ok", "remote_user": self.found.remote_user}

    def step_accounts(self) -> None:
        f, cl = self.found, self.cfg.cluster
        res = self.t.run("hyakalloc")
        f.raw["hyakalloc"] = res.stdout + res.stderr
        f.hyakalloc_available = res.ok
        alloc = parse_hyakalloc(res.stdout) if res.ok else []
        assoc_res = self.t.run("sacctmgr_assoc")
        f.raw["sacctmgr_assoc"] = assoc_res.stdout + assoc_res.stderr
        assoc = parse_sacctmgr_assoc(assoc_res.stdout) if assoc_res.ok else []
        user_res = self.t.run("sacctmgr_user")
        f.raw["sacctmgr_user"] = user_res.stdout + user_res.stderr
        default_account = parse_sacctmgr_user(user_res.stdout) if user_res.ok else None

        f.accounts = unique([r.account for r in alloc] + [r.account for r in assoc])
        if cl.account and (cl.account in f.accounts or not f.accounts):
            f.account, f.account_source = cl.account, "config (kept)"
        elif len(f.accounts) == 1:
            f.account = f.accounts[0]
            f.account_source = "hyakalloc" if alloc else "sacctmgr assoc"
        elif default_account and default_account in f.accounts:
            f.account, f.account_source = default_account, "sacctmgr DefaultAccount"
        elif f.accounts:
            f.account, f.account_source = None, f"ambiguous: {', '.join(f.accounts)}"
        else:
            f.account, f.account_source = None, "none found (hyakalloc/sacctmgr empty)"

        rows = [r for r in assoc if f.account is None or r.account == f.account]
        f.qos_all = unique(x for r in rows for x in r.qos)
        default_qos = unique(r.default_qos for r in rows)
        if cl.qos and (cl.qos in f.qos_all or not f.qos_all):
            f.qos, f.qos_source = cl.qos, "config (kept)"
        elif len(default_qos) == 1:
            f.qos, f.qos_source = default_qos[0], "sacctmgr DefaultQOS"
        elif len(f.qos_all) == 1:
            f.qos, f.qos_source = f.qos_all[0], "only QOS of the association"
        elif f.qos_all:
            f.qos, f.qos_source = None, f"ambiguous: {', '.join(f.qos_all)}"
        else:
            f.qos, f.qos_source = None, "none reported"
        f.allowed_partitions = unique(
            [r.partition for r in alloc] + [r.partition for r in assoc if r.partition]
        )
        self.steps["accounts"] = {
            "status": "ok",
            "account": f.account,
            "account_source": f.account_source,
            "qos": f.qos,
            "qos_source": f.qos_source,
            "hyakalloc_available": f.hyakalloc_available,
        }

    def step_partitions(self) -> None:
        f, cl = self.found, self.cfg.cluster
        res = self.t.run("sinfo")
        f.raw["sinfo"] = res.stdout + res.stderr
        parts: dict[str, PartitionInfo] = parse_sinfo(res.stdout) if res.ok else {}
        f.partitions = {
            n: {
                "default": p.default,
                "avail": p.avail,
                "timelimit": p.timelimit,
                "nodes": p.nodes,
                "gres": list(p.gres),
                "gpu": p.gpu,
            }
            for n, p in parts.items()
        }
        f.partition_default = next((n for n, p in parts.items() if p.default), None)
        up = {n: p for n, p in parts.items() if p.up}
        cand = {n: p for n, p in up.items() if n in f.allowed_partitions} or up
        gpu = [n for n, p in cand.items() if p.gpu]
        cpu = [n for n, p in cand.items() if not p.gpu]
        f.partition_candidates = {"cpu": cpu, "gpu": gpu}
        f.partition_notes = []
        f.partition_gpu = self._choose("partition_gpu", cl.partition_gpu, gpu, parts)
        f.partition_cpu = self._choose("partition_cpu", cl.partition_cpu, cpu, parts)
        if parts and not cpu:
            f.partition_notes.append(
                "every partition carries GPU GRES: there is no CPU-only partition, so CPU work "
                "(QE arrays) must target a GPU partition explicitly (cluster.partition_cpu)"
            )
        if not parts:
            f.partition_notes.append("sinfo returned nothing parsable; partitions left null")
        self.steps["partitions"] = {
            "status": "ok",
            "partition_cpu": f.partition_cpu,
            "partition_gpu": f.partition_gpu,
            "candidates": f.partition_candidates,
            "default": f.partition_default,
            "notes": list(f.partition_notes),
        }

    def step_scratch(self) -> None:
        f, cl = self.found, self.cfg.cluster
        env = self.t.run("scratch_env")
        f.raw["scratch_env"] = env.stdout
        scratch_env = env.stdout.strip().splitlines()[-1].strip() if env.stdout.strip() else ""
        accounts = [f.account] if f.account else list(f.accounts)
        candidates: list[tuple[str, str, str]] = []
        if cl.scratch:
            candidates.append(("config", posixpath.dirname(cl.scratch.rstrip("/")), cl.scratch))
        if scratch_env:
            candidates.append(("$SCRATCH", posixpath.dirname(scratch_env.rstrip("/")), scratch_env))
        for acct in accounts:
            candidates.append(
                ("/gpfs/projects", f"/gpfs/projects/{acct}", f"/gpfs/projects/{acct}/$USER")
            )
        candidates.append(("/gpfs/scrubbed", "/gpfs/scrubbed", "/gpfs/scrubbed/$USER"))
        for acct in accounts:
            candidates.append(("/gscratch", f"/gscratch/{acct}", f"/gscratch/{acct}/$USER"))
        candidates.append(("/gscratch/scrubbed", "/gscratch/scrubbed", "/gscratch/scrubbed/$USER"))

        f.scratch_probes = []
        f.scratch = None
        seen: set[str] = set()
        for source, parent, path in candidates:
            if path in seen:
                continue
            seen.add(path)
            res = self.t.run("scratch_probe", parent=parent, path=path)
            resolved = res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else ""
            ok = res.ok and bool(resolved)
            f.scratch_probes.append(
                {"source": source, "candidate": path, "ok": ok, "resolved": resolved or None}
            )
            if ok:
                f.scratch = resolved.rstrip("/")
                break
        self.steps["scratch"] = {
            "status": "ok" if f.scratch else "failed",
            "scratch": f.scratch,
            "probes": list(f.scratch_probes),
        }
        if not f.scratch:
            self.steps["scratch"]["error"] = "no writable scratch candidate; set cluster.scratch"

    def step_modules(self) -> None:
        f, cl = self.found, self.cfg.cluster
        grep = self.t.run("module_grep")
        f.raw["module_grep"] = grep.stdout + grep.stderr
        avail, defaults = parse_module_grep(grep.stdout)
        spider = self.t.run("module_spider", names=" ".join(MODULE_QUERIES))
        f.raw["module_spider"] = spider.stdout + spider.stderr
        queries = parse_module_spider(spider.stdout)
        f.module_versions = {
            n: sorted(qr.versions, key=version_key) for n, qr in queries.items() if qr.versions
        }
        for m in avail:
            f.module_versions.setdefault(m.split("/", 1)[0], [])
            if m not in f.module_versions[m.split("/", 1)[0]]:
                f.module_versions[m.split("/", 1)[0]].append(m)
        f.module_defaults = defaults
        f.module_prerequisites = {
            n: list(qr.prerequisites) for n, qr in queries.items() if qr.prerequisites
        }
        pick = self._pick_module
        f.qe_module = pick(("quantum-espresso", "espresso", "qe"), avail, defaults, queries)
        f.lammps_module = pick(("lammps",), avail, defaults, queries)
        gcc = pick(("gcc",), avail, defaults, queries)
        major = cuda_major(cl.libtorch_cuda_url)
        cuda = pick(("cuda",), avail, defaults, queries, major=major) or pick(
            ("cuda",), avail, defaults, queries
        )
        cmake = pick(("cmake",), avail, defaults, queries)
        openmpi = pick(("openmpi",), avail, defaults, queries)
        discovered = [m for m in (gcc, cuda, cmake, openmpi, f.qe_module) if m]
        if cl.modules:
            f.modules, f.modules_source = list(cl.modules), "config (kept)"
        else:
            f.modules = discovered
            f.modules_source = (
                "module spider: gcc, cuda (major of libtorch_cuda_url), cmake, openmpi, QE; "
                "(D) default if shown, else newest"
            )
        load_check = self._verify_module_set(f.modules)
        self.steps["modules"] = {
            "status": "ok" if load_check["ok"] else "failed",
            "modules": list(f.modules),
            "qe_module": f.qe_module,
            "lammps_module": f.lammps_module,
            "versions": dict(f.module_versions),
            "load_check": load_check,
        }

    def _verify_module_set(self, modules: list[str]) -> dict[str, Any]:
        """``module load`` the chosen set on the login node; on an Lmod hierarchy error adopt the
        first combination Lmod suggests that still contains every family we asked for."""
        f = self.found
        if not modules:
            return {"ok": True, "checked": [], "note": "no modules configured"}
        res = self.t.run("module_load_check", modules=" ".join(modules))
        out = res.stdout + res.stderr
        f.raw["module_load_check"] = out
        if MODULES_OK in out:
            return {"ok": True, "checked": list(modules)}
        families = [m.split("/", 1)[0] for m in modules]
        for suggestion in lmod_suggestions(out):
            merged = list(suggestion)
            for m in modules:  # keep the families Lmod's line does not mention (e.g. cmake)
                if m.split("/", 1)[0] not in {x.split("/", 1)[0] for x in merged}:
                    merged.append(m)
            merged = [m for m in merged if m.split("/", 1)[0] in families]
            res2 = self.t.run("module_load_check", modules=" ".join(merged))
            out2 = res2.stdout + res2.stderr
            f.raw["module_load_check_retry"] = out2
            if MODULES_OK in out2:
                f.modules = merged
                f.modules_source += (
                    f"; hierarchy fixed via Lmod suggestion ({' '.join(suggestion)})"
                )
                return {"ok": True, "checked": merged, "replaced": list(modules)}
        return {
            "ok": False,
            "checked": list(modules),
            "error": "module set does not load together; see raw.module_load_check",
        }

    def step_config(self) -> None:
        f = self.found
        values = f.config_values(self.cfg)
        comments: dict[str, str] = {}
        if f.account is None and f.account_source:
            comments["account"] = f"unresolved ({f.account_source}); set explicitly"
        if f.qos is None and f.qos_source:
            comments["qos"] = f"unresolved ({f.qos_source})"
        for key in ("partition_cpu", "partition_gpu"):
            kind = key.split("_")[1]
            if getattr(f, key) is None:
                cands = f.partition_candidates.get(kind, [])
                comments[key] = (
                    f"ambiguous, candidates: {', '.join(cands)} — set explicitly"
                    if cands
                    else f"no {kind.upper()} partition seen in sinfo"
                )
        if f.scratch is None and f.scratch_probes:
            comments["scratch"] = "no writable candidate; set explicitly"
        if f.qe_cmd is None and f.qe_status != "unknown":
            comments["qe_cmd"] = f.qe_status
        if f.lammps_cmd is None and f.lammps_status != "unknown":
            comments["lammps_cmd"] = f.lammps_status
        write_cluster_yaml(
            self.yaml_path,
            values,
            comments=comments,
            provenance=f"by `b20mlip cluster bootstrap` run {self.ctx.run_id}",
        )
        copy = self.ctx.out_dir / self.yaml_path.name
        shutil.copyfile(self.yaml_path, copy)
        self.ctx.add_output(self.yaml_path, "yaml")
        self.ctx.add_output(copy, "yaml")
        self.steps["config"] = {"status": "ok", "path": str(self.yaml_path), "values": values}

    def step_repo(self) -> None:
        f = self.found
        if not f.scratch:
            self.steps["repo"] = {"status": "blocked", "reason": "scratch unknown"}
            return
        remote_repo = f"{f.scratch}/{REPO_DIRNAME}"
        f.repo_remote = remote_repo
        local = repo_root()
        self.t.rsync(
            "rsync_push",
            f"{local}/",
            f"{self.t.alias}:{remote_repo}/",
            excludes=REPO_EXCLUDES,
        )
        layout = self.t.run("remote_layout", repo=q(remote_repo), scratch=q(f.scratch))
        if not layout.ok:
            raise RuntimeError(f"remote_bootstrap.sh failed: {layout.stderr.strip()}")
        which = self.t.run("which_uv")
        f.uv_present = which.ok
        if not which.ok:
            install = self.t.run("uv_install", url=UV_INSTALL_URL)
            if not install.ok:
                raise RuntimeError(f"uv installer failed: {install.stderr.strip()}")
            f.uv_installed_now = True
        sync = self.t.run("uv_sync", scratch=q(f.scratch), repo=q(remote_repo))
        f.raw["uv_sync"] = (sync.stdout + sync.stderr)[-4000:]
        if not sync.ok:
            raise RuntimeError(f"uv sync --frozen failed on {self.t.alias!r}: {sync.stderr[-800:]}")
        versions = [ln for ln in sync.stdout.splitlines() if re.match(r"^uv \d", ln)]
        f.uv_version = versions[0].strip() if versions else None
        self.steps["repo"] = {
            "status": "ok",
            "remote": remote_repo,
            "excludes": list(REPO_EXCLUDES),
            "uv_present": f.uv_present,
            "uv_installed_now": f.uv_installed_now,
            "uv_install_command": render_command("uv_install", url=UV_INSTALL_URL),
            "uv_version": f.uv_version,
        }

    def step_qe(self) -> None:
        f = self.found
        if f.qe_module:
            mods = f.modules if f.qe_module in f.modules else [*f.modules, f.qe_module]
            res = self.t.run("qe_which", modules=" ".join(mods))
            f.raw["qe_which"] = res.stdout + res.stderr
            path = res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else ""
            if res.ok and path:
                f.qe_cmd, f.qe_status = path, "module"
                self.steps["qe"] = {"status": "ok", "qe": "module", "qe_cmd": path}
                return
        if not f.scratch:
            f.qe_status = "needs micromamba (scratch unknown)"
            self.steps["qe"] = {"status": "blocked", "reason": f.qe_status}
            return
        probe = self.t.run("qe_probe", scratch=q(f.scratch))
        if probe.ok and probe.stdout.strip():
            f.qe_cmd = probe.stdout.strip().splitlines()[-1].strip()
            f.micromamba_env = f"{f.scratch}/qe"
            f.qe_status = "micromamba"
            self.steps["qe"] = {"status": "ok", "qe": "micromamba (present)", "qe_cmd": f.qe_cmd}
            return
        f.qe_plan = micromamba_plan(f.scratch)
        if not self.install_qe:
            f.qe_status = "needs micromamba"
            self.steps["qe"] = {
                "status": "ok",
                "qe": f.qe_status,
                "plan": list(f.qe_plan),
                "hint": "re-run with --install-qe (or --set cluster.install_qe=true) to execute",
            }
            return
        if not f.repo_remote:
            self.steps["qe"] = {"status": "blocked", "reason": "repo not synced (helper script)"}
            return
        res = self.t.run("micromamba_qe", repo=q(f.repo_remote), scratch=q(f.scratch))
        f.raw["micromamba_qe"] = (res.stdout + res.stderr)[-4000:]
        if not res.ok:
            raise RuntimeError(f"micromamba QE install failed: {res.stderr[-800:]}")
        f.qe_cmd = f"{f.scratch}/qe/bin/pw.x"
        f.micromamba_env = f"{f.scratch}/qe"
        f.qe_status = "micromamba"
        self.steps["qe"] = {
            "status": "ok",
            "qe": "micromamba (installed now)",
            "qe_cmd": f.qe_cmd,
            "plan": list(f.qe_plan),
        }

    def step_lammps(self) -> None:
        f, cl = self.found, self.cfg.cluster
        if not f.scratch:
            self.steps["lammps"] = {"status": "blocked", "reason": "scratch unknown"}
            return
        prefix = f"{f.scratch}/{REPO_DIRNAME}"
        probe = self.t.run("lammps_probe", prefix=q(prefix))
        if probe.ok:
            try:
                f.lammps_build = json.loads(probe.stdout)
            except ValueError:
                f.lammps_build = {"raw": probe.stdout}
            f.lammps_cmd, f.lammps_status = f"{prefix}/bin/lmp", "built"
            self.steps["lammps"] = {"status": "ok", "lammps": "built", "build": f.lammps_build}
            return
        if not self.build_lammps:
            f.lammps_status = "not built (pass --build-lammps)"
            self.steps["lammps"] = {"status": "ok", "lammps": f.lammps_status}
            return
        partition = self.partition or f.partition_gpu or f.partition_cpu
        if not partition:
            f.lammps_status = "blocked: no partition known"
            self.steps["lammps"] = {
                "status": "blocked",
                "reason": "partition_gpu and partition_cpu are null: pass --partition NAME",
            }
            return
        if not f.repo_remote:
            self.steps["lammps"] = {"status": "blocked", "reason": "repo not synced"}
            return
        gpu_partition = bool(f.partitions.get(partition, {}).get("gpu"))
        cuda = gpu_partition
        sha = cl.lammps_sha or ""
        url = cl.libtorch_cuda_url if cuda else cl.libtorch_cpu_url
        fetch = self.t.run(
            "lammps_fetch", prefix=q(prefix), url=q(url), sha=q(sha), repo=q(f.repo_remote)
        )
        f.raw["lammps_fetch"] = (fetch.stdout + fetch.stderr)[-4000:]
        if not fetch.ok:
            raise RuntimeError(f"login-node fetch of lammps/libtorch failed: {fetch.stderr[-800:]}")
        resources: dict[str, Any] = {
            "template": BUILD_TEMPLATE,
            "partition": partition,
            "cuda": cuda,
            "gpus": 1 if gpu_partition else 0,
            "libtorch_url": url,
            "lammps_sha": sha,
            "kokkos_arch": DEFAULT_KOKKOS_ARCH,
            "prefix": prefix,
            "build_modules": [m for m in f.modules if m != f.qe_module],
            **BUILD_RESOURCES,
        }
        if f.account:
            resources["account"] = f.account
        if f.qos:
            resources["qos"] = f.qos
        spec = JobSpec(
            name=BUILD_JOB_NAME, script=BUILD_SCRIPT, units=["build"], resources=resources
        )
        executor = self.t.with_settings(f.settings(self.cfg)).executor
        handle = executor.submit(spec)
        f.lammps_job_id = handle.job_ids[0]
        f.lammps_workdir = handle.workdir
        f.lammps_log = f"{handle.workdir}/logs/build.log"
        f.lammps_status = f"submitted (job {f.lammps_job_id})"
        self.ctx.slurm = SlurmInfo(
            job_ids=list(handle.job_ids),
            account=f.account or "",
            partition=partition,
            nodes=1,
            wall=str(BUILD_RESOURCES["time"]),
            units_done=0,
            units_failed=0,
        )
        sbatch = Path(executor.staging_root) / BUILD_JOB_NAME / "job.sbatch"
        if sbatch.is_file():
            self.ctx.add_output(sbatch, "other")
        self.steps["lammps"] = {
            "status": "ok",
            "lammps": "submitted",
            "job_id": f.lammps_job_id,
            "workdir": handle.workdir,
            "partition": partition,
            "cuda": cuda,
            "libtorch_url": url,
            "lammps_sha": sha or "branch head (recorded in build.json)",
            "log": f.lammps_log,
            "build_json": f"{prefix}/bin/build.json",
            "hint": "re-run `b20mlip cluster bootstrap` after the job finishes to fill lammps_cmd",
        }


def plan_commands() -> list[dict[str, str]]:
    """Every COMMANDS entry rendered with ``<placeholder>`` values (for ``--dry-run``)."""
    return [{"key": k, "command": render_command(k, **PLAN_PLACEHOLDERS)} for k in COMMANDS]


def bootstrap(
    cfg: Settings,
    ctx: RunContext,
    *,
    runner: Runner | None = None,
    build_lammps: bool = False,
    install_qe: bool | None = None,
    skip: Iterable[str] = (),
    partition: str | None = None,
    yaml_path: str | Path | None = None,
    template_dir: str | Path | None = None,
) -> StageResult:
    """Stage entry point (``run_stage("cluster.bootstrap", cfg, bootstrap, runner=...)``).

    ``install_qe=None`` defers to ``cfg.cluster.install_qe``. ``yaml_path`` defaults to
    ``configs/cluster/<alias>.yaml`` in this checkout. A dead socket raises
    :class:`ClusterUnreachable` (``run_stage`` then writes ``status="failed"`` with the MFA
    instruction in ``extras``).
    """
    transport = Transport(
        cfg,
        runner,
        template_dir=template_dir,
        staging_root=Path(cfg.paths.runs_dir) / "slurm",
    )
    path = (
        Path(yaml_path)
        if yaml_path is not None
        else repo_root() / "configs" / "cluster" / f"{cfg.cluster.alias}.yaml"
    )
    if ctx.dry_run:
        ctx.log(plan=plan_commands(), steps=list(STEPS), yaml_path=str(path))
        return StageResult(
            stage="cluster.bootstrap",
            run_id=ctx.run_id,
            manifest_path=str(ctx.manifest_path),
            status="partial",
            outputs=[],
            summary={"planned_commands": len(COMMANDS), "steps": ",".join(STEPS)},
        )
    job = Bootstrap(
        cfg,
        ctx,
        transport,
        build_lammps=build_lammps,
        install_qe=cfg.cluster.install_qe if install_qe is None else install_qe,
        partition=partition,
        yaml_path=path,
        skip=skip,
    )
    return job.run()


__all__ = [
    "ALWAYS_RERUN",
    "BUILD_JOB_NAME",
    "BUILD_RESOURCES",
    "BUILD_SCRIPT",
    "BUILD_TEMPLATE",
    "DEFAULT_KOKKOS_ARCH",
    "DISCOVERY_NAME",
    "STATE_NAME",
    "STEPS",
    "Bootstrap",
    "Discovered",
    "bootstrap",
    "cuda_major",
    "micromamba_plan",
    "plan_commands",
]
