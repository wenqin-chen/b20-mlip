"""qe_array / qe_phonons sbatch templates implement the templates/slurm/README.md unit protocol."""

from __future__ import annotations

from pathlib import Path

import pytest

from b20mlip.config import Settings
from b20mlip.dft import qe
from b20mlip.executors import CommandResult, JobSpec, SlurmExecutor

PROTOCOL_LINES = (
    'export B20_UNIT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$B20_WORKDIR/units.txt")',
    "export B20_UNIT_INDEX=$SLURM_ARRAY_TASK_ID",
    "slug=$(printf '%s' \"$B20_UNIT\" | tr -c 'A-Za-z0-9._-' '_')",
    '[ -f "$B20_WORKDIR/units/$slug.done" ] && { echo "unit $B20_UNIT already done"; exit 0; }',
    'rm -f "$B20_WORKDIR/units/$slug.failed"',
    'if bash "$B20_WORKDIR/script.sh" > "$B20_WORKDIR/logs/$slug.log" 2>&1; then',
    '> "$B20_WORKDIR/units/$slug.done"',
    '> "$B20_WORKDIR/units/$slug.failed"',
    "exit $rc",
    "set -euo pipefail",
)


def _runner(argv: list[str]) -> CommandResult:  # never called: rendering only
    raise AssertionError(f"unexpected command {argv}")


@pytest.fixture
def executor(slurm_settings: Settings, repo: Path) -> SlurmExecutor:
    return SlurmExecutor(slurm_settings, runner=_runner, template_dir=repo / "templates" / "slurm")


def _spec(slurm_settings: Settings, tmp_path: Path, template: str, **resources) -> JobSpec:
    return qe.job_spec(
        tmp_path / "r0", ["MnSi_ec40_k0.35", "a1b2c3d4e5f60718", "FeSi_disp000"], slurm_settings,
        name="qe-r0-deadbeef", template=template, resources=resources,
        units_root="/gscratch/b20/b20-mlip/dft/r0",
    )  # fmt: skip


@pytest.mark.parametrize(
    ("template", "ntasks", "time", "mem", "throttle"),
    [("qe_array", 8, "00:30:00", "16G", 16), ("qe_phonons", 32, "04:00:00", "64G", 4)],
)
def test_templates_implement_unit_protocol(
    executor: SlurmExecutor, slurm_settings: Settings, tmp_path: Path,
    template: str, ntasks: int, time: str, mem: str, throttle: int,
) -> None:  # fmt: skip
    text = executor.render(_spec(slurm_settings, tmp_path, template))
    assert text.startswith("#!/bin/bash\n#SBATCH --job-name=qe-r0-deadbeef\n")
    assert "#SBATCH --account=acct\n#SBATCH --partition=cpu-part\n#SBATCH --nodes=1\n" in text
    assert f"#SBATCH --ntasks={ntasks}\n#SBATCH --cpus-per-task=1\n#SBATCH --gpus=1\n" in text
    assert "--qos" not in text  # cluster.qos unset
    assert f"#SBATCH --mem={mem}\n#SBATCH --time={time}\n" in text
    assert f"#SBATCH --array=1-3%{throttle}\n" in text
    assert (
        "#SBATCH --output=/gscratch/b20/b20-mlip/jobs/qe-r0-deadbeef/logs/slurm-%A_%a.out" in text
    )
    assert "module load gcc\nmodule load openmpi\nmodule load quantum-espresso/7.3\n" in text
    assert "export OMP_NUM_THREADS=1\n" in text
    assert "export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS" in text
    assert 'export B20_UNITS_ROOT="/gscratch/b20/b20-mlip/dft/r0"\n' in text
    assert 'export QE_CMD="mpirun -np 8 pw.x -nk 2"\n' in text and text.count("export QE_CMD=") == 1
    assert (
        "export B20_WORKDIR=/gscratch/b20/b20-mlip/jobs/qe-r0-deadbeef B20_JOB_NAME=qe-r0-deadbeef"
        in text
    )
    for line in PROTOCOL_LINES:
        assert line in text, line
    assert "srun" not in text  # pw.x is launched by the unit script through $QE_CMD
    assert text.endswith("fi\n")
    for line in text.splitlines():
        assert "{{" not in line and "{%" not in line and "{#" not in line


def test_template_overrides_and_qe_cmd_fallback(
    executor: SlurmExecutor, slurm_settings: Settings, tmp_path: Path, repo: Path
) -> None:
    text = executor.render(
        _spec(slurm_settings, tmp_path, "qe_array", ntasks=16, cpus_per_task=2, mem="32G",
              time="01:00:00", max_parallel=50, partition="big")
    )  # fmt: skip
    assert (
        "#SBATCH --ntasks=16\n#SBATCH --cpus-per-task=2\n#SBATCH --gpus=1\n#SBATCH --mem=32G\n"
        "#SBATCH --time=01:00:00"
    ) in text
    assert "#SBATCH --array=1-3%50" in text and "#SBATCH --partition=big" in text
    assert "export OMP_NUM_THREADS=2\n" in text and "#SBATCH --gpus=1\n" in text
    gpu2 = executor.render(_spec(slurm_settings, tmp_path, "qe_array", gpus=2))
    assert "#SBATCH --gpus=2\n" in gpu2 and "#SBATCH --gpus=1" not in gpu2
    # no qe_cmd in the spec env: the template falls back to cluster.qe_cmd
    spec = _spec(slurm_settings, tmp_path, "qe_phonons").model_copy(
        update={"env": {"B20_UNITS_ROOT": "/gscratch/b20/b20-mlip/dft/ph", "B20_KEEP_TMP": "1"}}
    )
    text = executor.render(spec)
    assert 'export QE_CMD="mpirun -np 8 pw.x -nk 2"' in text and 'export B20_KEEP_TMP="1"' in text
    no_cmd = slurm_settings.model_copy(
        update={"cluster": slurm_settings.cluster.model_copy(update={"qe_cmd": None})}
    )
    ex2 = SlurmExecutor(no_cmd, runner=_runner, template_dir=repo / "templates" / "slurm")
    assert "QE_CMD" not in ex2.render(spec)
    staged = ex2.render(qe.job_spec(tmp_path / "r0", ["u"], no_cmd))
    assert "export B20_UNITS_ROOT=" in staged and "#SBATCH --array=1-1%16" in staged


@pytest.mark.parametrize("template", ["qe_array", "qe_phonons"])
def test_null_account_partition_and_qos(
    slurm_settings: Settings, repo: Path, tmp_path: Path, template: str
) -> None:
    """Tillicum: --account/--partition are omitted when unset (never "None"), --qos when set."""
    bare = slurm_settings.model_copy(
        update={
            "cluster": slurm_settings.cluster.model_copy(
                update={"account": None, "partition_cpu": None, "qos": None}
            )
        }
    )
    ex = SlurmExecutor(bare, runner=_runner, template_dir=repo / "templates" / "slurm")
    text = ex.render(_spec(bare, tmp_path, template))
    assert "--account" not in text and "--partition" not in text and "--qos" not in text
    assert "None" not in text
    assert text.startswith("#!/bin/bash\n#SBATCH --job-name=qe-r0-deadbeef\n#SBATCH --nodes=1\n")
    assert "#SBATCH --gpus=1\n" in text
    with_qos = slurm_settings.model_copy(
        update={"cluster": slurm_settings.cluster.model_copy(update={"qos": "normal"})}
    )
    ex2 = SlurmExecutor(with_qos, runner=_runner, template_dir=repo / "templates" / "slurm")
    text = ex2.render(_spec(with_qos, tmp_path, template, partition="gpu-h200-mig"))
    assert (
        "#SBATCH --account=acct\n#SBATCH --partition=gpu-h200-mig\n#SBATCH --qos=normal\n"
        "#SBATCH --nodes=1\n"
    ) in text
    assert "#SBATCH --gpus=1\n" in text and "#SBATCH --cpus-per-task=1\n" in text


@pytest.mark.parametrize("template", ["qe_array", "qe_phonons"])
def test_templates_skip_site_modules_for_micromamba_qe(
    slurm_settings: Settings, tmp_path: Path, template: str
) -> None:
    """Tillicum 2026-09-19: the self-contained micromamba QE must not get the site's gcc/cuda/
    openmpi modules loaded on top of it (foreign mpirun on PATH, MPI env) — CoSi SCFs diverged."""
    cfg = slurm_settings.model_copy(
        update={
            "cluster": slurm_settings.cluster.model_copy(
                update={"micromamba_env": "/gscratch/b20/qe", "qe_cmd": "/gscratch/b20/qe/bin/pw.x"}
            )
        }
    )
    executor = SlurmExecutor(cfg, runner=lambda argv: None)  # type: ignore[arg-type]
    text = executor.render(_spec(cfg, tmp_path, template))
    assert "module load" not in text
    assert 'export QE_CMD="/gscratch/b20/qe/bin/pw.x"\n' in text


@pytest.mark.parametrize("template", ["qe_array", "qe_phonons"])
def test_rendered_qe_job_is_valid_bash(
    slurm_settings: Settings, tmp_path: Path, template: str
) -> None:
    """Regression (Tillicum 2026-09-19): a whitespace-stripping Jinja comment glued a comment onto
    `set -euo pipefail` and every array task died in 3 s. Rendered jobs must pass `bash -n`."""
    import shutil
    import subprocess

    cfg = slurm_settings.model_copy(
        update={"cluster": slurm_settings.cluster.model_copy(update={"micromamba_env": "/g/qe"})}
    )
    for c in (slurm_settings, cfg):
        text = SlurmExecutor(c, runner=lambda argv: None).render(_spec(c, tmp_path, template))  # type: ignore[arg-type]
        assert "set -euo pipefail\n" in text
        path = tmp_path / f"{template}.sh"
        path.write_text(text, encoding="utf-8")
        bash = shutil.which("bash")
        assert bash is not None
        res = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stderr
