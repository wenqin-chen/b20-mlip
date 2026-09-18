"""templates/slurm/train_replay.sbatch.j2 renders a valid single-unit GPU job."""

from __future__ import annotations

import subprocess
from pathlib import Path

from b20mlip.config import Settings
from b20mlip.executors import CommandResult, JobSpec, SlurmExecutor
from b20mlip.train import finetune as ft


def _noop(argv: list[str]) -> CommandResult:
    return CommandResult(tuple(argv), 0, "", "")


def _spec(argv: list[str] | None) -> JobSpec:
    resources: dict[str, object] = {"template": "train_replay", "time": "08:00:00"}
    if argv is not None:
        resources["argv"] = ft.join_argv(argv)
    return JobSpec(
        name="train_replay_x",
        script="uv run --no-sync mace_run_train --name b20_replay",
        units=["20260918T000000-abcdef-0"],
        resources=resources,
        env={"WANDB_MODE": "offline"},
    )


def test_train_replay_template_renders(cluster_cfg: Settings, tmp_path: Path) -> None:
    ex = SlurmExecutor(cluster_cfg, runner=_noop, staging_root=tmp_path / "staging")
    argv = ft.build_args(
        cluster_cfg, "replay", {"train": "data/train.extxyz", "valid": "data/valid.extxyz"},
        0, ".", energy_scale="qe", resume=True, foundation="foundation.model",
        e0s_file="E0s_qe.json",
    )  # fmt: skip
    text = ex.render(_spec(argv))
    assert text.startswith("#!/bin/bash\n#SBATCH --job-name=train_replay_x\n")
    assert "#SBATCH --account=acct" in text
    assert "#SBATCH --partition=gpu-part" in text  # GPU partition by default
    assert "#SBATCH --nodes=1" in text and "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gpus=1" in text and "#SBATCH --time=08:00:00" in text
    assert "#SBATCH --cpus-per-task=6" in text and "#SBATCH --mem=64G" in text
    assert "#SBATCH --output=/gscratch/b20/b20-mlip/jobs/train_replay_x/logs/slurm-%j.out" in text
    assert "module load gcc\nmodule load cuda\n" in text
    assert "export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6" in text
    assert "export WANDB_MODE=offline" in text
    assert "export B20_WORKDIR=/gscratch/b20/b20-mlip/jobs/train_replay_x" in text
    assert 'export B20_UNIT=$(sed -n "1p" "$B20_WORKDIR/units.txt")' in text
    assert 'cd "$B20_WORKDIR"' in text
    assert "uv run --no-sync mace_run_train --name b20_replay --seed 0 --work_dir ." in text
    assert "--multiheads_finetuning True --pt_train_file mp --num_samples_pt 10000" in text
    assert "--E0s E0s_qe.json --restart_latest" in text
    assert "'{\"Default\":1.0}'" in text  # shell-quoted argv
    assert "$slug.done" in text and "$slug.failed" in text and "srun" not in text
    assert 'units/$slug.done" ] && { echo "unit $B20_UNIT already done"; exit 0; }' in text
    script = tmp_path / "job.sbatch"
    script.write_text(text, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_train_replay_template_fallbacks(cluster_cfg: Settings, tmp_path: Path) -> None:
    ex = SlurmExecutor(cluster_cfg, runner=_noop, staging_root=tmp_path / "staging")
    text = ex.render(_spec(None))
    assert 'run_unit() { bash "$B20_WORKDIR/script.sh"; }' in text
    assert "mace_run_train" not in text.split("run_unit()")[1].split("\n")[0]
    explicit = _spec(None).model_copy(
        update={"resources": {"template": "train_replay", "partition": "h200", "gpus": 2}}
    )
    text = ex.render(explicit)
    assert "#SBATCH --partition=h200" in text and "#SBATCH --gpus=2" in text
    no_gpu = cluster_cfg.model_copy(
        update={"cluster": cluster_cfg.cluster.model_copy(update={"partition_gpu": None})}
    )
    text = SlurmExecutor(no_gpu, runner=_noop, staging_root=tmp_path / "s2").render(_spec(None))
    assert "#SBATCH --partition=cpu-part" in text  # last resort: the executor's partition
