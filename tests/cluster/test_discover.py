"""cluster bootstrap: discovery into the YAML overlay, ambiguity -> null, MFA failure path."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from b20mlip.cluster import discover
from b20mlip.cluster.configfile import parse_cluster_yaml, write_cluster_yaml, yaml_scalar
from b20mlip.cluster.discover import STEPS, Discovered, bootstrap, cuda_major, micromamba_plan
from b20mlip.cluster.remote import COMMANDS
from b20mlip.config import Settings, read_yaml
from b20mlip.executors import ClusterUnreachable, JobHandle
from b20mlip.provenance import read_manifest, run_stage

from .conftest import (
    ASSOC_TWO_ACCOUNTS,
    BUILD_SHA,
    QE_PATH,
    SCRATCH,
    SINFO_TILLICUM,
    FakeRunner,
    Scenario,
    simple_scenario,
    tillicum_scenario,
)

MODULES = ["gcc/13.4.0", "cuda/12.9.1", "cmake/3.31.8", "openmpi/5.0.8", "quantum-espresso/7.3.1"]


def run_bootstrap(cfg: Settings, fake: FakeRunner, yaml_path: Path, **kw: object):
    return run_stage("cluster.bootstrap", cfg, bootstrap, runner=fake, yaml_path=yaml_path, **kw)


def test_bootstrap_writes_the_expected_yaml(cluster_cfg: Settings, yaml_path: Path) -> None:
    fake = FakeRunner(simple_scenario())
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    assert result.status == "ok", result.summary
    assert read_yaml(yaml_path) == {
        "cluster": {
            "alias": "tillicum",
            "control_path": cluster_cfg.cluster.control_path,
            "account": "b20",
            "partition_cpu": "compute",
            "partition_gpu": "gpu-h200",
            "qos": "normal",
            "scratch": SCRATCH,
            "modules": MODULES,
            "qe_cmd": QE_PATH,
            "lammps_cmd": None,
            "micromamba_env": None,
        }
    }
    text = yaml_path.read_text()
    assert text.startswith("# Tillicum (UW Hyak) overlay") and "# last discovered:" in text
    assert "# discovered: hyakalloc / sacctmgr" in text  # per-key comments survive
    assert "lammps_cmd: null" in text and "pass --build-lammps" in text

    manifest = read_manifest(result.manifest_path)
    assert manifest.status == "ok" and manifest.stage == "cluster.bootstrap"
    steps = manifest.extras["steps"]
    assert set(steps) == set(STEPS) and all(s["status"] == "ok" for s in steps.values())
    assert steps["accounts"]["account_source"] == "hyakalloc"
    assert steps["accounts"]["qos_source"] == "sacctmgr DefaultQOS"
    assert steps["partitions"]["candidates"] == {"cpu": ["compute"], "gpu": ["gpu-h200"]}
    assert steps["partitions"]["default"] == "compute"
    assert steps["scratch"]["probes"][0]["candidate"] == "/gpfs/projects/b20/$USER"
    assert steps["scratch"]["probes"][0]["ok"] is False
    assert steps["scratch"]["probes"][1]["resolved"] == SCRATCH
    assert steps["modules"]["qe_module"] == "quantum-espresso/7.3.1"
    assert steps["repo"]["uv_present"] is True and steps["repo"]["uv_installed_now"] is False
    assert steps["repo"]["uv_version"].startswith("uv 0.8")
    assert steps["qe"] == {"status": "ok", "qe": "module", "qe_cmd": QE_PATH}
    assert steps["lammps"]["lammps"].startswith("not built")
    raw = manifest.extras["raw"]
    assert raw["sinfo"].startswith("PARTITION") and "hyakalloc" in raw and "module_spider" in raw
    keys = [c["key"] for c in manifest.extras["commands"]]
    assert keys[:2] == ["socket_check", "whoami"] and "rsync_push" in keys and "uv_sync" in keys
    assert "uv_install" not in keys and "micromamba_qe" not in keys and "lammps_fetch" not in keys
    assert manifest.extras["mfa_instruction"] == "run `ssh -fN tillicum` (MFA) then retry"
    assert manifest.extras["discovery"]["remote_user"] == "wenqin98"
    assert manifest.extras["discovery"]["modules_source"].startswith("module spider")
    names = {Path(a.path).name for a in manifest.outputs}
    assert names == {"tillicum.yaml", "discovery.json"}
    assert any(a.path == str(yaml_path) for a in manifest.outputs)
    push = next(c for c in fake.calls if c[0] == "rsync")
    assert "--exclude=.venv" in push and "--exclude=data/raw" in push and "--exclude=.git" in push
    assert push[-1] == f"tillicum:{SCRATCH}/b20-mlip/"
    assert result.summary["qe"] == "module" and result.summary["account"] == "b20"
    assert (Path(result.manifest_path).parent / "bootstrap_state.json").is_file()
    assert (Path(result.manifest_path).parent / "discovery.json").is_file()
    remote = fake.remote_commands()
    assert remote[0] == "whoami" and remote[1] == "hyakalloc"
    assert 'sinfo -o "%P %a %l %D %G"' in remote
    assert any(
        c.startswith("module load gcc/13.4.0 cuda/12.9.1") and c.endswith("which pw.x")
        for c in remote
    )


def test_two_gpu_partitions_and_no_cpu_partition_stay_null(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    fake = FakeRunner(tillicum_scenario())
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    assert result.status == "ok", result.summary
    data = read_yaml(yaml_path)["cluster"]
    assert data["partition_gpu"] is None and data["partition_cpu"] is None
    assert data["account"] == "b20" and data["qos"] == "normal" and data["scratch"] == SCRATCH
    assert data["modules"] == MODULES[:-1] and data["qe_cmd"] is None
    text = yaml_path.read_text()
    assert "partition_gpu: null" in text
    assert "ambiguous, candidates: gpu-h200, gpu-h200-mig" in text
    assert "no CPU partition seen in sinfo" in text
    assert "qe_cmd: null" in text and "needs micromamba" in text
    manifest = read_manifest(result.manifest_path)
    steps = manifest.extras["steps"]
    assert steps["accounts"]["hyakalloc_available"] is False
    assert steps["accounts"]["account_source"] == "sacctmgr assoc"
    assert steps["partitions"]["candidates"] == {"cpu": [], "gpu": ["gpu-h200", "gpu-h200-mig"]}
    assert steps["partitions"]["default"] == "gpu-h200"
    notes = "\n".join(steps["partitions"]["notes"])
    assert "ambiguous ['gpu-h200', 'gpu-h200-mig']" in notes and "no CPU-only partition" in notes
    assert manifest.extras["raw"]["sinfo"] == SINFO_TILLICUM
    assert manifest.extras["raw"]["hyakalloc"].endswith("command not found")
    assert steps["qe"]["qe"] == "needs micromamba"
    assert steps["qe"]["plan"] == micromamba_plan(SCRATCH)
    assert "qe=7.5" in steps["qe"]["plan"][1] and "micromamba-linux-64" in steps["qe"]["plan"][0]
    assert result.summary["partition_gpu"] == "null" and result.summary["qe"] == "needs micromamba"
    assert not any(c[0] == "rsync" and "jobs" in c[-1] for c in fake.calls)  # nothing submitted


def test_dead_socket_raises_and_manifest_carries_the_mfa_instruction(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    fake = FakeRunner(Scenario(socket_ok=False))
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    assert result.status == "failed"
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["mfa_instruction"] == "run `ssh -fN tillicum` (MFA) then retry"
    assert "ClusterUnreachable" in manifest.extras["error"]
    assert "run `ssh -fN tillicum` (MFA) then retry" in manifest.extras["error"]
    assert manifest.extras["steps"]["socket"]["status"] == "failed"
    assert manifest.extras["steps"]["socket"]["instruction"] == manifest.extras["mfa_instruction"]
    assert [c[:5] for c in fake.calls] == [
        ["ssh", "-S", cluster_cfg.cluster.control_path, "-O", "check"]
    ]
    assert fake.remote_commands() == []
    assert read_yaml(yaml_path)["cluster"]["scratch"] is None  # nothing written
    # the direct call raises (run_stage is what turns it into status="failed")
    from b20mlip.provenance import RunContext

    ctx = RunContext("cluster.bootstrap", cluster_cfg)
    with pytest.raises(ClusterUnreachable, match="ssh -fN tillicum"):
        bootstrap(
            cluster_cfg, ctx, runner=FakeRunner(Scenario(socket_ok=False)), yaml_path=yaml_path
        )
    # socket alive but the login shell dies (rc 255) mid-way: same outcome, later steps untouched
    gone = FakeRunner(Scenario(die_after_ssh=2))
    result = run_bootstrap(cluster_cfg, gone, yaml_path)
    assert result.status == "failed"
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert steps["socket"]["status"] == "ok" and steps["accounts"]["status"] == "failed"
    assert "partitions" not in steps


def test_install_qe_uv_installer_and_lammps_submission(
    cluster_cfg: Settings, yaml_path: Path, remote_root: Path
) -> None:
    fake = FakeRunner(tillicum_scenario(uv_present=False), remote_root=remote_root)
    result = run_bootstrap(
        cluster_cfg, fake, yaml_path, build_lammps=True, install_qe=True, partition="gpu-h200"
    )
    assert result.status == "ok", result.summary
    data = read_yaml(yaml_path)["cluster"]
    assert data["qe_cmd"] == f"{SCRATCH}/qe/bin/pw.x" and data["micromamba_env"] == f"{SCRATCH}/qe"
    assert data["lammps_cmd"] is None and "submitted (job 4242)" in yaml_path.read_text()
    manifest = read_manifest(result.manifest_path)
    steps = manifest.extras["steps"]
    assert steps["repo"]["uv_present"] is False and steps["repo"]["uv_installed_now"] is True
    assert steps["repo"]["uv_install_command"] == "curl -LsSf https://astral.sh/uv/install.sh | sh"
    assert steps["qe"]["qe"] == "micromamba (installed now)"
    lammps = steps["lammps"]
    assert (
        lammps["job_id"] == "4242" and lammps["cuda"] is True and lammps["partition"] == "gpu-h200"
    )
    assert lammps["libtorch_url"] == cluster_cfg.cluster.libtorch_cuda_url
    assert lammps["lammps_sha"] == BUILD_SHA
    assert lammps["workdir"] == f"{SCRATCH}/b20-mlip/jobs/build_lammps"
    assert lammps["log"] == f"{SCRATCH}/b20-mlip/jobs/build_lammps/logs/build.log"
    assert lammps["build_json"] == f"{SCRATCH}/b20-mlip/bin/build.json"
    assert manifest.slurm is not None and manifest.slurm.job_ids == ["4242"]
    assert manifest.slurm.partition == "gpu-h200" and manifest.slurm.units_done == 0
    keys = [c["key"] for c in manifest.extras["commands"]]
    for key in (
        "which_uv",
        "uv_install",
        "uv_sync",
        "qe_probe",
        "micromamba_qe",
        "lammps_probe",
        "lammps_fetch",
    ):
        assert key in keys, key
    remote = fake.remote_commands()
    fetch = next(c for c in remote if "build_lammps.sh fetch" in c)
    assert fetch.startswith(f"B20_PREFIX={SCRATCH}/b20-mlip B20_LIBTORCH_URL=")
    assert f"B20_LAMMPS_SHA={BUILD_SHA}" in fetch and "cu126" in fetch
    assert f"cd {SCRATCH}/b20-mlip/jobs/build_lammps && sbatch --parsable job.sbatch" in remote
    staged = remote_root / "jobs" / "build_lammps"
    sbatch = (staged / "job.sbatch").read_text()
    assert "#SBATCH --partition=gpu-h200" in sbatch and "#SBATCH --qos=normal" in sbatch
    assert "#SBATCH --gpus=1" in sbatch and "#SBATCH --account=b20" in sbatch
    assert (
        "export B20_CUDA=1" in sbatch
        and "module load gcc/13.4.0\nmodule load cuda/12.9.1" in sbatch
    )
    assert "quantum-espresso" not in sbatch
    assert (staged / "units.txt").read_text() == "build\n"
    assert 'build_lammps.sh" build' in (staged / "script.sh").read_text()
    local_staging = Path(cluster_cfg.paths.runs_dir) / "slurm" / "build_lammps"
    handle = JobHandle.model_validate_json((local_staging / "handle.json").read_text())
    assert handle.job_ids == ["4242"]
    assert any(Path(a.path).name == "job.sbatch" for a in manifest.outputs)
    assert (
        result.summary["lammps"] == "submitted (job 4242)" and result.summary["qe"] == "micromamba"
    )


def test_lammps_already_built_fills_lammps_cmd_without_submitting(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    fake = FakeRunner(simple_scenario(lmp_built=True, qe_installed=True))
    result = run_bootstrap(cluster_cfg, fake, yaml_path, build_lammps=True)
    assert result.status == "ok"
    data = read_yaml(yaml_path)["cluster"]
    assert data["lammps_cmd"] == f"{SCRATCH}/b20-mlip/bin/lmp"
    assert data["qe_cmd"] == QE_PATH  # the module wins over an installed micromamba env
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert steps["lammps"]["build"]["sha"] == BUILD_SHA and steps["lammps"]["lammps"] == "built"
    assert not any("sbatch" in c for c in fake.remote_commands())
    assert result.summary["lammps"] == "built"


def test_build_requires_a_partition_and_blocks_honestly(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    fake = FakeRunner(tillicum_scenario())
    result = run_bootstrap(cluster_cfg, fake, yaml_path, build_lammps=True)
    assert result.status == "partial" and result.summary["failed_steps"] == "lammps"
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert steps["lammps"]["status"] == "blocked" and "--partition" in steps["lammps"]["reason"]
    assert not any("sbatch" in c for c in fake.remote_commands())


def test_configured_values_are_kept_when_the_cluster_confirms_them(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    cfg = cluster_cfg.model_copy(
        update={
            "cluster": cluster_cfg.cluster.model_copy(
                update={
                    "partition_gpu": "gpu-h200-mig",
                    "account": "stf",
                    "modules": ["gcc/11.5.0"],
                }
            )
        }
    )
    fake = FakeRunner(tillicum_scenario(assoc=ASSOC_TWO_ACCOUNTS))
    result = run_bootstrap(cfg, fake, yaml_path)
    assert result.status == "ok"
    data = read_yaml(yaml_path)["cluster"]
    assert data["partition_gpu"] == "gpu-h200-mig" and data["account"] == "stf"
    assert data["modules"] == ["gcc/11.5.0"] and data["qos"] == "normal"
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert steps["accounts"]["account_source"] == "config (kept)"
    assert "kept 'gpu-h200-mig' from the config" in "\n".join(steps["partitions"]["notes"])
    # several accounts, none configured: SLURM's own default account is used, else null
    fake = FakeRunner(tillicum_scenario(assoc=ASSOC_TWO_ACCOUNTS))
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert steps["accounts"]["account"] == "b20"
    assert steps["accounts"]["account_source"] == "sacctmgr DefaultAccount"
    fake = FakeRunner(
        tillicum_scenario(assoc=ASSOC_TWO_ACCOUNTS, user="User|Def Acct|\nwenqin98||\n")
    )
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert (
        steps["accounts"]["account"] is None
        and steps["accounts"]["account_source"] == "ambiguous: b20, stf"
    )
    assert "unresolved (ambiguous: b20, stf)" in yaml_path.read_text()
    # a configured partition sinfo does not know is ignored, not trusted
    cfg = cluster_cfg.model_copy(
        update={"cluster": cluster_cfg.cluster.model_copy(update={"partition_cpu": "ghost"})}
    )
    result = run_bootstrap(cfg, FakeRunner(simple_scenario()), yaml_path)
    assert read_yaml(yaml_path)["cluster"]["partition_cpu"] == "compute"
    notes = read_manifest(result.manifest_path).extras["steps"]["partitions"]["notes"]
    assert any("'ghost' is not a partition sinfo lists" in n for n in notes)


def test_scratch_env_and_no_writable_scratch(cluster_cfg: Settings, yaml_path: Path) -> None:
    env_scratch = "/gscratch/scrubbed/wenqin98"
    fake = FakeRunner(simple_scenario(scratch_env=env_scratch, writable=(env_scratch,)))
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    assert result.status == "ok"
    assert read_yaml(yaml_path)["cluster"]["scratch"] == env_scratch
    probes = read_manifest(result.manifest_path).extras["steps"]["scratch"]["probes"]
    assert probes[0] == {
        "source": "$SCRATCH",
        "candidate": env_scratch,
        "ok": True,
        "resolved": env_scratch,
    }

    fake = FakeRunner(simple_scenario(writable=()))
    result = run_bootstrap(cluster_cfg, fake, yaml_path)
    assert result.status == "partial"
    assert sorted(result.summary["failed_steps"].split(",")) == [
        "lammps",
        "qe",
        "repo",
        "scratch",
    ] or ("scratch" in result.summary["failed_steps"])
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert steps["scratch"]["status"] == "failed" and steps["repo"]["status"] == "blocked"
    assert steps["qe"]["status"] == "ok"  # the QE module needs no scratch
    assert steps["lammps"]["status"] == "blocked"
    candidates = [p["candidate"] for p in steps["scratch"]["probes"]]
    assert candidates == [
        "/gpfs/projects/b20/$USER",
        "/gpfs/scrubbed/$USER",
        "/gscratch/b20/$USER",
        "/gscratch/scrubbed/$USER",
    ]
    text = yaml_path.read_text()
    assert "scratch: null" in text and "no writable candidate" in text


def test_resume_skips_finished_steps_and_skip_is_honoured(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    broken = FakeRunner(simple_scenario(rsync_rc=23))
    first = run_bootstrap(cluster_cfg, broken, yaml_path)
    assert first.status == "partial" and first.summary["failed_steps"] == "repo"
    steps = read_manifest(first.manifest_path).extras["steps"]
    assert steps["repo"]["status"] == "failed" and "rsync_push failed" in steps["repo"]["error"]
    assert steps["qe"]["status"] == "ok" and steps["lammps"]["status"] == "ok"

    fixed = FakeRunner(simple_scenario())
    second = run_bootstrap(cluster_cfg, fixed, yaml_path, resume=True)
    assert second.run_id == first.run_id and second.status == "ok"
    steps = read_manifest(second.manifest_path).extras["steps"]
    assert steps["accounts"]["status"] == "resumed" and steps["modules"]["status"] == "resumed"
    assert steps["socket"]["status"] == "ok" and steps["repo"]["status"] == "ok"
    remote = fixed.remote_commands()
    assert "hyakalloc" not in remote and not any(c.startswith("sinfo") for c in remote)
    assert any("uv_install.sh" in c for c in remote)
    assert read_yaml(yaml_path)["cluster"]["account"] == "b20"  # restored from the state file

    skipped = FakeRunner(simple_scenario())
    result = run_bootstrap(cluster_cfg, skipped, yaml_path, skip=("repo", "qe", "lammps"))
    assert result.status == "ok"
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert {steps[s]["status"] for s in ("repo", "qe", "lammps")} == {"skipped"}
    assert not any(c[0] == "rsync" for c in skipped.calls)

    bad = run_bootstrap(cluster_cfg, FakeRunner(), yaml_path, skip=("nosuch",))
    assert bad.status == "failed" and "unknown bootstrap step" in bad.summary["error"]


def test_dry_run_plans_every_command_without_running_any(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    fake = FakeRunner()
    result = run_bootstrap(cluster_cfg, fake, yaml_path, dry_run=True)
    assert result.status == "partial" and result.outputs == []
    assert result.summary["planned_commands"] == len(COMMANDS)
    plan = read_manifest(result.manifest_path).extras["plan"]
    assert {p["key"] for p in plan} == set(COMMANDS)
    assert fake.calls == [] and read_yaml(yaml_path)["cluster"]["scratch"] is None


def test_partial_failures_keep_going(cluster_cfg: Settings, yaml_path: Path) -> None:
    class UvFails(FakeRunner):
        def _respond(self, cmd, tup):  # type: ignore[no-untyped-def]
            if "uv_install.sh" in cmd:
                from b20mlip.executors import CommandResult

                return CommandResult(tup, 1, "", "error: failed to build torch")
            return super()._respond(cmd, tup)

    result = run_bootstrap(cluster_cfg, UvFails(simple_scenario()), yaml_path)
    assert result.status == "partial" and result.summary["failed_steps"] == "repo"
    steps = read_manifest(result.manifest_path).extras["steps"]
    assert "uv sync --frozen failed" in steps["repo"]["error"]
    assert steps["qe"]["status"] == "ok" and steps["lammps"]["status"] == "ok"
    assert read_yaml(yaml_path)["cluster"]["qe_cmd"] == QE_PATH


def test_helpers() -> None:
    assert cuda_major("https://download.pytorch.org/libtorch/cu126/x.zip") == "12"
    assert cuda_major("https://download.pytorch.org/libtorch/cu130/x.zip") == "13"
    assert cuda_major("https://download.pytorch.org/libtorch/cpu/x.zip") is None
    plan = micromamba_plan("/s")
    assert plan[0].startswith("curl -Ls https://github.com/mamba-org/micromamba-releases")
    assert plan[1].endswith("create -y -p /s/qe -c conda-forge qe=7.5")
    found = Discovered(account="a", scratch="/s", modules=["m"])
    cfg = Settings.model_validate({})
    assert (
        found.settings(cfg).cluster.scratch == "/s" and found.settings(cfg).cluster.account == "a"
    )
    assert found.config_values(cfg)["alias"] == "tillicum"
    assert discover.plan_commands()[0]["key"] == "socket_check"
    assert yaml_scalar(None) == "null" and yaml_scalar([]) == "[]" and yaml_scalar(True) == "true"
    assert yaml_scalar(["a/1", "b"]) == "[a/1, b]" and yaml_scalar("x: y") == "'x: y'"


def test_write_cluster_yaml_from_scratch_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "new.yaml"
    write_cluster_yaml(path, {"alias": "klone", "scratch": "/gscratch/x"})
    data = read_yaml(path)
    assert data["cluster"]["alias"] == "klone" and data["cluster"]["scratch"] == "/gscratch/x"
    header, values, comments, trailing = parse_cluster_yaml(path.read_text())
    assert header[0].startswith("# Cluster overlay") and any(
        h.startswith("# last discovered:") for h in header
    )
    assert values["scratch"] == "/gscratch/x" and trailing == []
    # a second write keeps other keys, replaces the provenance line, keeps comments and sections
    path.write_text(path.read_text() + "\ncompute:\n  threads: 2\n")
    write_cluster_yaml(
        path, {"account": "acct"}, comments={"account": "from test"}, provenance="run X"
    )
    text = path.read_text()
    assert text.count("# last discovered:") == 1 and "run X" in text
    assert "account: acct" in text and "# from test" in text and "scratch: /gscratch/x" in text
    assert read_yaml(path)["compute"] == {"threads": 2}
    # an invalid overlay never replaces the file
    with pytest.raises(ValidationError):
        write_cluster_yaml(path, {"bogus_key": 1})
    assert "bogus_key" not in path.read_text() and not path.with_suffix(".yaml.tmp").exists()


LMOD_HIERARCHY_ERROR = """Lmod has detected the following error: These module(s) or extension(s)
exist but cannot be loaded as requested: "openmpi/5.0.10"
   Try: "module spider openmpi/5.0.10" to see how to load the module(s).
   Or load any one of these options:
      module load gcc/13.4.0 cuda/12.8.2 openmpi/5.0.10
"""


def test_module_set_is_verified_and_lmod_suggestion_adopted(
    cluster_cfg: Settings, yaml_path: Path
) -> None:
    """Tillicum 2026-09-18: the newest cuda (12.9.1) does not sit under openmpi/5.0.10 in the
    Lmod hierarchy; the load check must catch it and adopt Lmod's suggested combination."""
    from b20mlip.cluster.discover import lmod_suggestions

    assert lmod_suggestions(LMOD_HIERARCHY_ERROR) == [
        ["gcc/13.4.0", "cuda/12.8.2", "openmpi/5.0.10"]
    ]
    configured = ["gcc/13.4.0", "cuda/12.9.1", "cmake/3.31.8", "openmpi/5.0.10"]
    cfg = cluster_cfg.model_copy(
        update={"cluster": cluster_cfg.cluster.model_copy(update={"modules": configured})}
    )
    fake = FakeRunner(
        tillicum_scenario(module_load_fail={" ".join(configured): LMOD_HIERARCHY_ERROR})
    )
    result = run_bootstrap(cfg, fake, yaml_path)
    assert result.status == "ok", result.summary
    step = read_manifest(result.manifest_path).extras["steps"]["modules"]
    assert step["status"] == "ok" and step["load_check"]["ok"] is True
    assert step["load_check"]["replaced"] == configured
    assert read_yaml(yaml_path)["cluster"]["modules"] == [
        "gcc/13.4.0",
        "cuda/12.8.2",
        "openmpi/5.0.10",
        "cmake/3.31.8",
    ]


def test_module_set_that_never_loads_fails_the_step(cluster_cfg: Settings, yaml_path: Path) -> None:
    configured = ["gcc/13.4.0", "cuda/12.9.1"]
    cfg = cluster_cfg.model_copy(
        update={"cluster": cluster_cfg.cluster.model_copy(update={"modules": configured})}
    )
    fake = FakeRunner(tillicum_scenario(module_load_fail={" ".join(configured): "Lmod error\n"}))
    result = run_bootstrap(cfg, fake, yaml_path)
    step = read_manifest(result.manifest_path).extras["steps"]["modules"]
    assert step["status"] == "failed" and step["load_check"]["ok"] is False


def test_skipped_steps_keep_configured_values(discovered_cfg: Settings, yaml_path: Path) -> None:
    """Regression (Tillicum 2026-09-18): `--skip qe` blanked qe_cmd to null in the overlay."""
    cfg = discovered_cfg.model_copy(
        update={
            "cluster": discovered_cfg.cluster.model_copy(
                update={"qe_cmd": "/scratch/qe/bin/pw.x", "micromamba_env": "/scratch/qe"}
            )
        }
    )
    fake = FakeRunner(tillicum_scenario())
    result = run_bootstrap(cfg, fake, yaml_path, skip=("qe", "lammps", "repo"))
    assert result.status == "ok", result.summary
    data = read_yaml(yaml_path)["cluster"]
    assert data["qe_cmd"] == "/scratch/qe/bin/pw.x"
    assert data["micromamba_env"] == "/scratch/qe"
    assert data["scratch"] == SCRATCH


def test_overlay_rewrite_keeps_nested_blocks_and_other_sections(tmp_path: Path) -> None:
    """`resources:` (a nested mapping) and a trailing `dft:` section survive a bootstrap rewrite."""
    path = tmp_path / "tillicum.yaml"
    path.write_text(
        "# header\n"
        "cluster:\n"
        "  alias: tillicum\n"
        "  scratch: /scratch/u   # discovered\n"
        "  resources:  # per template\n"
        "    # 8-atom units\n"
        "    qe_array: {ntasks: 2, gpus: 1, mem: 30G}\n"
        "    qe_phonons: {ntasks: 8, gpus: 4}\n"
        "  lammps_cmd: null\n"
        "\n"
        "dft:\n"
        "  pseudo_dir: /scratch/pseudos\n",
        encoding="utf-8",
    )
    write_cluster_yaml(path, {"lammps_cmd": "/scratch/bin/lmp"}, provenance="test")
    text = path.read_text(encoding="utf-8")
    nested = "\n".join(
        [
            "  resources:  # per template",
            "    # 8-atom units",
            "    qe_array: {ntasks: 2, gpus: 1, mem: 30G}",
            "",
        ]
    )
    assert nested in text
    assert "dft:\n  pseudo_dir: /scratch/pseudos" in text
    data = read_yaml(path)
    assert data["cluster"]["resources"] == {
        "qe_array": {"ntasks": 2, "gpus": 1, "mem": "30G"},
        "qe_phonons": {"ntasks": 8, "gpus": 4},
    }
    assert data["cluster"]["lammps_cmd"] == "/scratch/bin/lmp"
    assert data["dft"]["pseudo_dir"] == "/scratch/pseudos"
