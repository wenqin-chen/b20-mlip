"""COMMANDS rendering, output parsers, the Transport and the build_lammps template/scripts."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from b20mlip.cluster import remote
from b20mlip.cluster.remote import (
    COMMANDS,
    LOCAL_COMMANDS,
    Transport,
    nodelist_count,
    parse_hyakalloc,
    parse_module_grep,
    parse_module_spider,
    parse_sacct_jobs,
    parse_sacctmgr_assoc,
    parse_sacctmgr_user,
    parse_sinfo,
    parse_squeue,
    render_command,
    version_key,
)
from b20mlip.config import Settings
from b20mlip.executors import ClusterUnreachable, JobSpec, SlurmExecutor

from .conftest import (
    ASSOC_TWO_ACCOUNTS,
    BUILD_SHA,
    HYAKALLOC_SIMPLE,
    MODULE_GREP_SIMPLE,
    SACCT_DONE,
    SINFO_SIMPLE,
    SINFO_TILLICUM,
    SPIDER_COMMON,
    SPIDER_NO_QE,
    SPIDER_QE,
    SQUEUE,
    USER_SIMPLE,
    FakeRunner,
    Scenario,
)


def test_render_command_collapses_braces_and_rejects_unknown_keys() -> None:
    assert render_command("sinfo") == 'sinfo -o "%P %a %l %D %G"'
    assert render_command("scratch_env") == 'echo "${SCRATCH:-}"'
    assert render_command("squeue") == 'squeue -u "$USER" -o "%i %j %T %M %P %R"'
    assert (
        render_command("sacct", ids="1,2")
        == "sacct -j 1,2 -X -P -o JobID,State,ExitCode,Elapsed,NodeList"
    )
    assert render_command("uv_install", url="U") == "curl -LsSf U | sh"
    with pytest.raises(KeyError, match="nosuch"):
        render_command("nosuch")
    with pytest.raises(KeyError, match="sacct"):
        render_command("sacct")  # missing placeholder
    assert LOCAL_COMMANDS <= set(COMMANDS)
    for key, template in COMMANDS.items():
        rendered = render_command(key, **{k: "x" for k in _placeholders(template)})
        assert "{{" not in rendered and "}}" not in rendered, key
        if _placeholders(template):
            assert "x" in rendered and rendered != template, key


def _placeholders(template: str) -> set[str]:
    import string

    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def test_parse_hyakalloc_total_rows_only() -> None:
    rows = parse_hyakalloc(HYAKALLOC_SIMPLE)
    assert [(r.account, r.partition, r.cpus, r.memory, r.gpus) for r in rows] == [
        ("b20", "compute", 320, "1400G", 0),
        ("b20", "gpu-h200", 64, "1600G", 8),
    ]
    assert parse_hyakalloc("bash: hyakalloc: command not found") == []
    plain = "| Account | Partition | CPUs | Memory | GPUs |\n| stf | ckpt | 40 | 175G | 2 |\n"
    assert parse_hyakalloc(plain)[0].partition == "ckpt"


def test_parse_sacctmgr() -> None:
    rows = parse_sacctmgr_assoc(ASSOC_TWO_ACCOUNTS)
    assert [r.account for r in rows] == ["b20", "stf"]
    assert rows[0].qos == ("normal", "debug") and rows[0].default_qos == "normal"
    assert rows[1].partition == "" and rows[1].qos[-1] == "interactive"
    assert parse_sacctmgr_assoc("Cluster|Account|\n") == []
    assert parse_sacctmgr_user(USER_SIMPLE) == "b20"
    assert parse_sacctmgr_user("User|Def Acct|\nx||\n") is None
    assert parse_sacctmgr_user("") is None


def test_parse_sinfo_aggregates_partitions_and_flags_gpus() -> None:
    parts = parse_sinfo(SINFO_SIMPLE)
    assert list(parts) == ["compute", "gpu-h200", "ckpt-all", "broken"]
    assert parts["compute"].default and parts["compute"].nodes == 112
    assert parts["compute"].gres == [] and not parts["compute"].gpu and parts["compute"].up
    assert parts["gpu-h200"].gres == ["gpu:h200:8"] and parts["gpu-h200"].gpu
    assert not parts["broken"].up
    till = parse_sinfo(SINFO_TILLICUM)
    assert till["gpu-h200"].default and till["gpu-h200-mig"].gpu
    assert parse_sinfo("PARTITION AVAIL TIMELIMIT NODES GRES\n") == {}


def test_parse_modules() -> None:
    names, defaults = parse_module_grep(MODULE_GREP_SIMPLE)
    assert names == [
        "quantum-espresso/7.2",
        "quantum-espresso/7.3.1",
        "cuda/12.4.0",
        "cuda/12.9.1",
        "cuda/13.0.0",
    ]
    assert defaults == ["quantum-espresso/7.3.1", "cuda/12.9.1"]
    assert parse_module_grep("-------- /gpfs/software/modulefiles/Core --------\n") == ([], [])

    queries = parse_module_spider(SPIDER_QE + SPIDER_COMMON)
    assert queries["quantum-espresso"].versions == [
        "quantum-espresso/7.2",
        "quantum-espresso/7.3.1",
    ]
    assert queries["quantum-espresso"].newest == "quantum-espresso/7.3.1"
    assert not queries["espresso"].found and not queries["qe"].found and not queries["lammps"].found
    assert queries["cuda"].versions == ["cuda/12.4.0", "cuda/12.9.1", "cuda/13.0.0"]
    assert queries["gcc"].newest == "gcc/13.4.0"
    assert queries["cmake"].versions == ["cmake/3.31.8"]
    assert queries["cmake"].prerequisites == ["gcc/13.4.0"]
    assert queries["openmpi"].prerequisites == ["gcc/13.4.0"]
    assert not parse_module_spider(SPIDER_NO_QE)["quantum-espresso"].found
    assert parse_module_spider("") == {}
    assert version_key("cuda/12.9.1") < version_key("cuda/12.10.0")
    assert max(["x/1.10", "x/1.9", "x/1.9-rc"], key=version_key) == "x/1.10"


def test_parse_sacct_squeue_nodelist() -> None:
    rows = parse_sacct_jobs(SACCT_DONE)
    assert [r.job_id for r in rows] == ["4243_1", "4243_2", "4243_3"]
    assert [r.state for r in rows] == ["COMPLETED", "COMPLETED", "TIMEOUT"]
    assert all(r.terminal for r in rows) and rows[2].exit_code == "0:1"
    assert parse_sacct_jobs("1|CANCELLED by 42|0:0|00:00:05|g1|\n")[0].state == "CANCELLED"
    assert parse_sacct_jobs("") == []
    assert nodelist_count("g[005-006]") == 2 and nodelist_count("g[001-003],g010") == 4
    assert nodelist_count("None assigned") == 0 and nodelist_count("") == 0
    assert nodelist_count("n[1,3,5-6]") == 4 and nodelist_count("a,b") == 2
    queue = parse_squeue(SQUEUE)
    assert len(queue) == 3 and queue[0].job_id == "4242" and queue[0].reason == "g001"
    assert queue[1].job_id == "4243_[1-40]" and queue[1].state == "PENDING"
    assert queue[1].partition == "gpu-h200-mig" and queue[1].reason == "(Priority)"
    assert parse_squeue("JOBID NAME STATE TIME PARTITION NODELIST(REASON)\n") == []


def test_transport_records_and_maps_255_to_unreachable(cluster_cfg: Settings) -> None:
    fake = FakeRunner(Scenario())
    t = Transport(cluster_cfg, fake)
    t.check_socket()
    assert fake.calls[0] == [
        "ssh",
        "-S",
        cluster_cfg.cluster.control_path,
        "-O",
        "check",
        "tillicum",
    ]
    res = t.run("whoami")
    assert res.ok and res.stdout.strip() == "wenqin98"
    assert fake.calls[-1][-1] == "bash -lc whoami"
    assert fake.calls[-1][:5] == [
        "ssh",
        "-S",
        cluster_cfg.cluster.control_path,
        "-o",
        "BatchMode=yes",
    ]
    assert t.used == {"socket_check", "whoami"}
    assert t.records_json()[-1] == {
        "key": "whoami",
        "command": "whoami",
        "returncode": 0,
        "remote": True,
    }
    t.rsync("rsync_push", "/src/", "tillicum:/dst/", excludes=(".git", "a b"))
    argv = fake.calls[-1]
    assert argv[:2] == ["rsync", "-az"] and "--exclude=.git" in argv and "--exclude=a b" in argv
    assert argv[argv.index("-e") + 1] == t.executor.ssh_transport
    assert argv[-2:] == ["/src/", "tillicum:/dst/"]

    dead = Transport(cluster_cfg, FakeRunner(Scenario(socket_ok=False)))
    with pytest.raises(ClusterUnreachable, match=r"run `ssh -fN tillicum` \(MFA\) then retry"):
        dead.check_socket()
    gone = Transport(cluster_cfg, FakeRunner(Scenario(ssh_rc=255)))
    with pytest.raises(ClusterUnreachable, match="MFA"):
        gone.run("whoami")
    assert gone.records[-1].returncode == 255
    with pytest.raises(ClusterUnreachable):
        Transport(cluster_cfg, FakeRunner(Scenario(rsync_rc=255))).rsync(
            "rsync_pull", "tillicum:/a/", "/b/"
        )
    with pytest.raises(RuntimeError, match="rsync_pull failed"):
        Transport(cluster_cfg, FakeRunner(Scenario(rsync_rc=23))).rsync(
            "rsync_pull", "tillicum:/a/", "/b/"
        )
    assert t.mfa_instruction == remote.MFA_INSTRUCTION.format(alias="tillicum")
    other = t.with_settings(cluster_cfg.model_copy(update={"compute": cluster_cfg.compute}))
    assert other.records is t.records and other.executor.runner is fake


def _build_spec(**resources: object) -> JobSpec:
    base: dict[str, object] = {
        "template": "build_lammps",
        "partition": "gpu-h200",
        "cuda": True,
        "gpus": 1,
        "libtorch_url": "https://example.invalid/libtorch-cu126.zip",
        "lammps_sha": BUILD_SHA,
        "kokkos_arch": "HOPPER90",
        "prefix": "/gpfs/scrubbed/wenqin98/b20-mlip",
        "build_modules": ["gcc/13.4.0", "cuda/12.9.1", "cmake/3.31.8"],
        "qos": "normal",
        "account": "b20",
    }
    base.update(resources)
    return JobSpec(
        name="build_lammps",
        script='bash "$B20_REPO/scripts/tillicum/build_lammps.sh" build',
        units=["build"],
        resources=base,
        env={"B20_EXTRA": "1"},
    )


def test_build_lammps_template_renders_cuda_and_cpu_variants(discovered_cfg: Settings) -> None:
    ex = SlurmExecutor(discovered_cfg, runner=FakeRunner())
    text = ex.render(_build_spec())
    assert text.startswith("#!/bin/bash\n#SBATCH --job-name=build_lammps\n")
    for line in (
        "#SBATCH --account=b20",
        "#SBATCH --partition=gpu-h200",
        "#SBATCH --qos=normal",
        "#SBATCH --gpus=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --time=04:00:00",
        "module load gcc/13.4.0\nmodule load cuda/12.9.1\nmodule load cmake/3.31.8\n",
        "export B20_REPO=/gpfs/scrubbed/wenqin98/b20-mlip",
        "export B20_PREFIX=/gpfs/scrubbed/wenqin98/b20-mlip",
        "export B20_LIBTORCH_URL=https://example.invalid/libtorch-cu126.zip",
        f"export B20_LAMMPS_SHA={BUILD_SHA}",
        "export B20_CUDA=1",
        "export B20_KOKKOS_ARCH=HOPPER90",
        'export B20_EXTRA="1"',
        'export B20_UNIT=$(sed -n 1p "$B20_WORKDIR/units.txt")',
        'bash "$B20_WORKDIR/script.sh"',
        '"$B20_WORKDIR/units/$slug.done"',
        '"$B20_WORKDIR/units/$slug.failed"',
        "set -euo pipefail",
    ):
        assert line in text, line
    assert "srun" not in text and "#SBATCH --array" not in text

    cpu = ex.render(_build_spec(cuda=False, gpus=0, partition="compute", qos=None, account=None))
    assert "--gpus" not in cpu and "--qos" not in cpu and "--account" not in cpu
    assert "export B20_CUDA=0" in cpu and "#SBATCH --partition=compute" in cpu
    minimal = JobSpec(
        name="build_lammps", script="true", units=["build"], resources={"template": "build_lammps"}
    )
    text = ex.render(minimal)  # every knob has a default from the cluster config
    assert "module load gcc/13.4.0\nmodule load cuda/12.9.1\n" in text
    assert f"export B20_LAMMPS_SHA={discovered_cfg.cluster.lammps_sha}" in text
    assert f"export B20_LIBTORCH_URL={discovered_cfg.cluster.libtorch_cpu_url}" in text
    assert "#SBATCH --partition=compute" in text and "#SBATCH --qos=normal" in text


@pytest.mark.parametrize(
    "name", ["remote_bootstrap.sh", "uv_install.sh", "micromamba_qe.sh", "build_lammps.sh"]
)
def test_tillicum_helper_scripts(repo: Path, name: str) -> None:
    path = repo / "scripts" / "tillicum" / name
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    assert path.stat().st_mode & 0o111, "must be executable"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_build_script_flags_and_fallbacks(repo: Path) -> None:
    text = (repo / "scripts" / "tillicum" / "build_lammps.sh").read_text()
    for flag in (
        "PKG_ML-MACE=ON",
        "PKG_KOKKOS=ON",
        "Kokkos_ENABLE_CUDA=ON",
        "Kokkos_ARCH_${B20_KOKKOS_ARCH}=ON",
        "nvcc_wrapper",
        'CMAKE_PREFIX_PATH="$libtorch"',
        "https://github.com/ACEsuit/lammps",
        'LAMMPS_BRANCH="mace"',
        'git -C "$src" rev-parse HEAD',
        "build.json",
        "make install",
        'FALLBACK="cpu"',
    ):
        assert flag in text, flag
    mm = (repo / "scripts" / "tillicum" / "micromamba_qe.sh").read_text()
    assert (
        "qe=7.5" in mm and "micromamba-releases/releases/latest/download/micromamba-linux-64" in mm
    )
    uv = (repo / "scripts" / "tillicum" / "uv_install.sh").read_text()
    assert "https://astral.sh/uv/install.sh" in uv and "uv sync --frozen" in uv
    layout = (repo / "scripts" / "tillicum" / "remote_bootstrap.sh").read_text()
    assert "env.sh" in layout and "UV_CACHE_DIR" in layout


def test_remote_bootstrap_script_runs_locally(repo: Path, tmp_path: Path) -> None:
    """The layout helper is plain bash: run it against a temp 'scratch' (no network)."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    res = subprocess.run(
        ["bash", str(repo / "scripts" / "tillicum" / "remote_bootstrap.sh"), str(scratch)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert (scratch / "b20-mlip" / "jobs").is_dir() and (scratch / "bin").is_dir()
    env = (scratch / "b20-mlip" / "env.sh").read_text()
    assert f'export B20_SCRATCH="{scratch}"' in env and "MAMBA_ROOT_PREFIX" in env
    assert shlex.split(env.splitlines()[1])[1].startswith("B20_SCRATCH=")
