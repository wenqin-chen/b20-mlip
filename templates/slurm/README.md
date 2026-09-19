# templates/slurm

Jinja2 sbatch templates rendered by `b20mlip.executors.SlurmExecutor`. Planned files
(SPEC.md section 8; added by the dft / train / md / cluster tiers):

- `qe_array.sbatch.j2`, `qe_phonons.sbatch.j2` — QE arrays, one unit per frame / displacement
- `train_replay.sbatch.j2` — multihead replay fine-tune on the GPU partition
- `lammps.sbatch.j2`, `build_lammps.sbatch.j2` — MD and the ACEsuit/lammps `mace` build
- `generic.sbatch.j2` — the executor's default when `resources["template"]` is unset

`qe_array.sbatch.j2` and `qe_phonons.sbatch.j2` exist (dft tier; rendered in
`tests/dft/test_dft_templates.py`, the unit script they call is `b20mlip.dft.qe.UNIT_SCRIPT`,
which needs `QE_CMD` and `B20_UNITS_ROOT` in the job environment). `train_replay.sbatch.j2`
exists (train tier; GPU partition, one task, a single unit whose command is
`uv run --no-sync mace_run_train {{ resources.argv }}` run from the work directory, so a
resubmission resumes through MACE's `--restart_latest`; rendered in
`tests/train/test_train_template.py`). `build_lammps.sbatch.j2` exists (cluster tier: a one-unit ACEsuit/lammps `mace` build with
libtorch and Kokkos+CUDA on a GPU partition, submitted by `b20mlip cluster bootstrap --build-lammps`;
rendered in `tests/cluster/test_remote.py`, see `docs/CLUSTER.md`). The md templates do not exist yet.

## Rendering context

`SlurmExecutor.render(spec)` renders `<template>.sbatch.j2` (`spec.resources["template"]`,
default `generic`) with `StrictUndefined` and these variables:

| variable | value |
|---|---|
| `job_name`, `script`, `units`, `n_units` | from the `JobSpec` |
| `resources`, `env` | free-form dicts from the `JobSpec` (`time`, `mem`, `ntasks`, `gpus`, `max_parallel`, ...) |
| `account`, `partition` | `resources` override, else `cluster.account` / `cluster.partition_cpu` |
| `cluster` | `Settings.cluster` as a dict (`modules`, `qe_cmd`, `lammps_cmd`, `scratch`, ...) |
| `threads` | `Settings.compute.threads` (pinned BLAS threads) |
| `workdir` | `<cluster.scratch>/b20-mlip/jobs/<job_name>` (staged there by rsync) |
| `script_name`, `units_name` | `script.sh`, `units.txt` (written next to the sbatch file) |

## Unit protocol (binding)

Every template must implement the per-unit resume protocol the `LocalExecutor` uses, so a job
can be resubmitted after a timeout and only the missing units run:

```bash
{#- example generic.sbatch.j2 -#}
#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --account={{ account }}
#SBATCH --partition={{ partition }}
#SBATCH --nodes=1
#SBATCH --ntasks={{ resources.ntasks | default(1) }}
#SBATCH --cpus-per-task={{ resources.cpus_per_task | default(threads) }}
#SBATCH --mem={{ resources.mem | default('16G') }}
#SBATCH --time={{ resources.time | default('01:00:00') }}
#SBATCH --array=1-{{ n_units }}%{{ resources.max_parallel | default(8) }}
#SBATCH --output={{ workdir }}/logs/slurm-%A_%a.out
set -euo pipefail
{% for m in cluster.modules %}module load {{ m }}
{% endfor %}
export OMP_NUM_THREADS={{ threads }} MKL_NUM_THREADS={{ threads }} OPENBLAS_NUM_THREADS={{ threads }}
{% for k, v in env.items() %}export {{ k }}={{ v }}
{% endfor %}
export B20_WORKDIR={{ workdir }} B20_JOB_NAME={{ job_name }}
export B20_UNIT_INDEX=$SLURM_ARRAY_TASK_ID
export B20_UNIT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$B20_WORKDIR/{{ units_name }}")
slug=$(printf '%s' "$B20_UNIT" | tr -c 'A-Za-z0-9._-' '_')
[ -f "$B20_WORKDIR/units/$slug.done" ] && { echo "unit $B20_UNIT already done"; exit 0; }
rm -f "$B20_WORKDIR/units/$slug.failed"
if bash "$B20_WORKDIR/{{ script_name }}" > "$B20_WORKDIR/logs/$slug.log" 2>&1; then
  printf '{"unit": "%s", "state": "done", "returncode": 0}\n' "$B20_UNIT" > "$B20_WORKDIR/units/$slug.done"
else
  rc=$?
  printf '{"unit": "%s", "state": "failed", "returncode": %d}\n' "$B20_UNIT" "$rc" > "$B20_WORKDIR/units/$slug.failed"
  exit $rc
fi
```

Rules: no `srun` for single-task units; pin BLAS threads; markers are the only source of truth
for resume; a dead ssh socket is never treated as a finished job (`cluster sync` re-polls sacct).
