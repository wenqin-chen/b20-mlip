# Cluster hand-off: Tillicum (UW Hyak) runbook

`b20mlip cluster bootstrap | sync | status` (SPEC.md section 8) move the QE, replay-training and
LAMMPS stages to the Tillicum GPU cluster without ever needing a login inside a script: **you**
open the SSH ControlMaster socket once with MFA, the tool does everything else through that socket
and records what it found. The cluster is a pay-as-you-go GPU system (H200 nodes, Rocky 9, SLURM,
Lmod); every discovered fact goes into `configs/cluster/tillicum.yaml`, every command and its raw
output into the run manifest. Nothing is guessed: a value the cluster did not confirm stays `null`.

## 1. First login (MFA)

`~/.ssh/config` already has the alias (`Host tillicum`, `ControlMaster auto`,
`ControlPath ~/.ssh/tillicum-cm`, `ControlPersist 4h`). Tillicum requires password + Duo and does
not allow ssh keys on the login node, so the socket must be opened interactively:

```bash
ssh -fN tillicum            # UW NetID password, then the Duo push/passcode; returns at once
ssh -O check tillicum       # "Master running (pid=...)" = the socket is alive (4 h idle timeout)
uv run b20mlip cluster bootstrap                      # discovery + repo sync + uv sync (cheap)
uv run b20mlip cluster bootstrap --build-lammps       # ... and submit the LAMMPS build job
uv run b20mlip cluster bootstrap --install-qe         # ... and install QE via micromamba if needed
```

If the socket is dead every cluster command fails fast with `status="failed"` and the exact
instruction in the manifest and on stderr: ``run `ssh -fN tillicum` (MFA) then retry``. Nothing
is retried silently and nothing is ever marked done because the socket disappeared.

## 2. What `cluster bootstrap` discovers (and how)

Steps run in this order; each is recorded under `extras["steps"]` of
`runs/cluster.bootstrap/<run_id>/manifest.json`, can be skipped with `--skip <name>` (repeatable)
and is resumed from `bootstrap_state.json` when the global `--resume` flag is given (`socket`
and `config` always re-run). Every remote command is wrapped in `bash -lc` so that Lmod's
`module` function exists (a plain `ssh host cmd` shell has none).

| step | commands | rule |
|---|---|---|
| `socket` | `ssh -S ~/.ssh/tillicum-cm -O check tillicum`, `whoami` | dead socket -> `ClusterUnreachable` + MFA instruction |
| `accounts` | `hyakalloc`, `sacctmgr show assoc user=$USER format=account,partition,qos,defaultqos -P`, `sacctmgr show user $USER format=user,defaultaccount -P` | one account -> `account`; several -> SLURM's `DefaultAccount`; none/unclear -> `null`. `qos` = the association's `DefaultQOS`, else the only QOS, else `null`. `hyakalloc` is a Klone tool and may be absent on Tillicum (rc 127 is recorded, not an error) |
| `partitions` | `sinfo -o "%P %a %l %D %G"` | `partition_gpu` = the *unique* `up` partition whose GRES mentions `gpu`/`h200`; `partition_cpu` = the unique one without GPU GRES. Two candidates -> `null` plus the candidate list in the YAML comment and the manifest. Partitions listed by `hyakalloc`/`sacctmgr` restrict the candidates; a value already in the YAML is kept when `sinfo` knows it |
| `scratch` | `echo $SCRATCH`, then per candidate `test -d <parent> && mkdir -p <cand>/b20-mlip && test -w ...` | first writable of: the configured value, `$SCRATCH`, `/gpfs/projects/<account>/$USER`, `/gpfs/scrubbed/$USER`, `/gscratch/<account>/$USER`, `/gscratch/scrubbed/$USER`. Every probe is recorded |
| `modules` | `module avail 2>&1 \| grep -i -E "quantum\|espresso\|qe\|lammps\|cuda"` and `module spider <name>` for quantum-espresso, espresso, qe, lammps, cuda, gcc, cmake, openmpi | `modules` = `[gcc, cuda, cmake, openmpi]` (+ the QE module) — the `(D)` default when `module avail` shows one, else the newest version; `cuda` restricted to the CUDA major of `cluster.libtorch_cuda_url` (cu126 -> 12.x). Tillicum's Lmod hierarchy hides `cuda`/`cmake`/`openmpi` until `gcc` is loaded, which is why `module spider` is authoritative |
| `config` | — | writes `configs/cluster/tillicum.yaml` (values only; comments survive; unknowns `null`) and a copy into the run directory. Runs again at the end so `qe_cmd`/`lammps_cmd` land in the file |
| `repo` | `rsync -az --exclude=... <checkout>/ tillicum:<scratch>/b20-mlip/`, `scripts/tillicum/remote_bootstrap.sh`, `command -v uv`, `curl -LsSf https://astral.sh/uv/install.sh \| sh` (only when uv is missing), `scripts/tillicum/uv_install.sh` (`uv sync --frozen`) | excludes `.venv data/raw data/omat24 runs models dft .git` and caches. uv's cache and managed Python live in `<scratch>` because home directories are 10 GB |
| `qe` | `module load <modules> && which pw.x`; else `test -x <scratch>/qe/bin/pw.x`; else the micromamba plan | `qe_cmd` = the absolute `pw.x` path. No module -> the plan (`micromamba create -y -p <scratch>/qe -c conda-forge qe=7.5`) is written to the manifest and **executed only** with `--install-qe` / `--set cluster.install_qe=true` |
| `lammps` | `test -x <scratch>/b20-mlip/bin/lmp && cat .../build.json`; with `--build-lammps`: `scripts/tillicum/build_lammps.sh fetch` on the login node, then `sbatch` of `templates/slurm/build_lammps.sbatch.j2` | an existing `lmp` fills `lammps_cmd` (SHA from `build.json`). Otherwise nothing is submitted unless `--build-lammps`; the build needs a partition (`partition_gpu`, else `partition_cpu`, else `--partition NAME`) and is a Kokkos+CUDA build on a GPU partition, a CPU/OpenMP build elsewhere |

Exit code: 0 only when every executed step is `ok` (`status="ok"`). A blocked or failed step
(no writable scratch, `uv sync` failure, `--build-lammps` without a known partition) gives
`status="partial"`, exit 1, and the failing step names in `summary.failed_steps`; the other steps
still run so one login reports everything. `--dry-run` writes a `partial` manifest listing every
command template and touches nothing.

### Parser caveat — validated at the first login

The `hyakalloc`, `sacctmgr -P`, `sinfo`, Lmod and `sacct`/`squeue` parsers were written against
the formats documented at hyak.uw.edu and in `UWrc/tillicum-onboarding` (2026-09-17), not against
a live session. If a real listing differs, the affected field simply stays `null` and the raw text
is in `extras["raw"]` of the bootstrap manifest (and in `discovery.json`): read it, set the value
by hand (`--set cluster.partition_gpu=gpu-h200` or edit the YAML) and, if it is a parser bug,
paste the raw text into `tests/cluster/conftest.py` as a new fixture.

### Facts verified from the public docs (2026-09-17)

- Login `tillicum.hyak.uw.edu`, password + Duo, no ssh keys; login node `tillicum-login01`,
  compute nodes `g001`–`g024` (8 × H200 141 GB, 64 cores, 200 GB RAM + 8 CPUs per GPU).
- Documented partitions `gpu-h200` (full GPU) and `gpu-h200-mig` (MIG slice, 0.143 billing
  multiplier); QOS `normal` (24 h, 16 GPUs), `debug`, `interactive` (8 h, 2 GPUs), `long`, `wide`,
  `urgent`; `--gpus=N` / `--gres=gpu:N`; **every job must request at least one GPU** — there is
  no CPU-only partition, so bootstrap will report `partition_cpu: null` with a note and the QE
  arrays must target a GPU partition explicitly (the MIG partition is the cheap one).
- Storage: `/gpfs/home/<NetID>` (10 GB), `/gpfs/projects/<group>` (1 TB, backed up),
  `/gpfs/scrubbed/<dir>` (up to 100 TB, purged after 60 days of inactivity; conda environments
  there can lose files). `/gscratch/<group>` is the Klone layout; both are probed.
- Lmod with a compiler hierarchy: `gcc/13.4.0 (D)`, `gcc/11.5.0`; after gcc: `cuda/12.4.0`,
  `cuda/12.9.1 (D)`, `cuda/13.0.0`, `openmpi/5.0.8`, `cmake/3.31.8`; `conda/Miniforge3-…`.
- libtorch 2.14.0 (the torch pinned in `uv.lock`) exists for `cpu`, `cu126` and `cu130`
  (`cluster.libtorch_cpu_url` / `cluster.libtorch_cuda_url`); the ACEsuit/lammps `mace` branch
  head on 2026-09-17 is `4d222cb3ee2a6b14083c778968497bf9e0efc4b4` (`cluster.lammps_sha`, `null`
  = branch head; the SHA actually built is always recorded in `build.json`).

## 3. `cluster status`

```bash
uv run b20mlip cluster status          # squeue -u $USER -o "%i %j %T %M %P %R" + local marker counts
uv run b20mlip cluster status --json
```

Prints the queue (JOBID, NAME, STATE, TIME, PARTITION, NODELIST/REASON) and, for every synced job
under `runs/slurm/<job>/`, the unit counts from `units.txt` and the `units/*.done|*.failed` markers.
No manifest is written; a dead socket exits 1 with the MFA instruction.

## 4. `cluster sync` (idempotent)

```bash
uv run b20mlip cluster sync                 # every directory under <scratch>/b20-mlip/jobs/
uv run b20mlip cluster sync --jobs qe_r0    # only these
```

For each job: `sacct -j <ids> -X -P -o JobID,State,ExitCode,Elapsed,NodeList` (ids from the local
`runs/slurm/<job>/handle.json` written at submission), then
`rsync -az --prune-empty-dirs --exclude=... tillicum:<scratch>/b20-mlip/jobs/<job>/ runs/slurm/<job>/`
(QE wavefunction/mixing files, build trees, libtorch and zips are excluded). Running it again only
transfers what changed. **Units are done only when `units/<unit>.done` came back**; a unit whose
sacct row says `COMPLETED` but has no marker is `pending` (and flagged `crashed_without_marker`
when every job row is terminal — resubmit it with the stage's `--resume`). A socket that dies
mid-way raises `ClusterUnreachable`: the manifest is `failed`, no `sync.json` is written and no
unit changes state. Status: `ok` when no unit is pending, `partial` (exit 1) otherwise — failed
units are results, not sync errors, and are reported in `extras["jobs"][<job>]`.

## 5. Fallbacks (SPEC.md section 11)

- **No QE module** -> `qe_status: needs micromamba`, the two commands of the plan in the manifest;
  `--install-qe` runs `scripts/tillicum/micromamba_qe.sh` (static micromamba into `<scratch>/bin`,
  `conda-forge qe=7.5` into `<scratch>/qe`; idempotent) and sets `qe_cmd: <scratch>/qe/bin/pw.x`,
  `micromamba_env: <scratch>/qe`. If the scratch is `/gpfs/scrubbed`, re-run bootstrap after a
  long pause (the 60-day purge can remove untouched files of the environment).
- **LAMMPS build fails** -> `build_lammps.sh` retries a failed Kokkos+CUDA build once as a
  CPU/OpenMP build (`"fallback": "cpu"` in `build.json`); if that fails too the unit is marked
  `units/build.failed`, `lammps_cmd` stays `null` and the MD stays ASE-only ("LAMMPS" leaves the
  bullet; the parity gate cannot pass without it). The log is
  `<scratch>/b20-mlip/jobs/build_lammps/logs/build.log` (pulled by `cluster sync`).
- **Two GPU partitions / no CPU partition** (the Tillicum case) -> `partition_gpu: null`,
  `partition_cpu: null`, candidates in the YAML comment; choose with `--set cluster.partition_gpu=…`
  or edit the YAML, then re-run (`--build-lammps --partition gpu-h200` works without editing).
- **uv missing on the cluster** -> installed with the official script into `~/.local/bin` (40 MB);
  caches go to `<scratch>/.uv-cache`, managed Pythons to `<scratch>/.uv-python`.
- **torch on the cluster**: `uv.lock` pins the CPU torch wheel on Linux (pyproject
  `[tool.uv.sources]`), so `uv sync --frozen` gives a CPU venv; GPU replay training needs a CUDA
  torch on top (train tier: e.g. `uv pip install --python <scratch>/b20-mlip/.venv torch --index-url
  https://download.pytorch.org/whl/cu126` inside the job) — recorded here, not done by bootstrap.

## 6. What is written where

Local:

- `configs/cluster/tillicum.yaml` — the overlay (`account`, `partition_cpu`, `partition_gpu`,
  `qos`, `scratch`, `modules`, `qe_cmd`, `lammps_cmd`, `micromamba_env`); a
  `# last discovered: <utc> by … run <run_id>` line; comments survive rewrites.
- `runs/cluster.bootstrap/<run_id>/` — `manifest.json` (steps, discovery, raw listings, every
  rendered command with its return code, `slurm` block of the build submission), `discovery.json`,
  `bootstrap_state.json` (resume), `tillicum.yaml` (copy).
- `runs/slurm/<job>/` — staging of every SLURM submission (`job.sbatch`, `script.sh`, `units.txt`,
  `spec.json`, `handle.json`) and, after `cluster sync`, `units/`, `logs/` and the results.
- `runs/cluster.sync/<run_id>/` — `manifest.json` (per-job `units_done/failed/pending`), `sync.json`.

Remote (`<scratch>` = the discovered directory, e.g. `/gpfs/scrubbed/<NetID>`):

- `<scratch>/b20-mlip/` — the rsynced checkout with its own `.venv` (`uv sync --frozen`),
  `env.sh` (source it in ad-hoc scripts), `jobs/<job>/` (one directory per submission: the
  binding unit protocol of `templates/slurm/README.md`), `bin/lmp` + `bin/build.json` after the
  LAMMPS build, `build/` (sources, libtorch, build tree), `logs/`.
- `<scratch>/qe/` (micromamba QE), `<scratch>/bin/micromamba`, `<scratch>/.uv-cache`,
  `<scratch>/.uv-python`, `<scratch>/.micromamba`.

## 7. For the dft / train / md tiers

- Submit through `SlurmExecutor` with `--config configs/cluster/tillicum.yaml`; the job directory
  is `<scratch>/b20-mlip/jobs/<job name>/`, results come back to `runs/slurm/<job name>/` via
  `cluster sync`, and `units/<slug>.done|failed` are the only completion signal (slug = unit id
  with anything outside `A-Za-z0-9._-` replaced by `_`; keep unit ids slug-safe so Python and the
  sbatch `tr` agree).
- On Tillicum add `#SBATCH --qos={{ cluster.qos }}` and `--gpus=1` (8 CPUs and 200 GB per GPU)
  to CPU-style templates: CPU-only jobs are rejected. `{{ account }}` may be `None` when the
  account is ambiguous — guard the `#SBATCH --account` line.
- `cluster.modules` is the discovered toolchain (`gcc, cuda, cmake, openmpi` + the QE module);
  `cluster.qe_cmd` is an absolute `pw.x` path (module or micromamba); `cluster.lammps_cmd` is
  `<scratch>/b20-mlip/bin/lmp` once built (`pair_style mace no_domain_decomposition` needs the
  `-lammps.pt` export). Re-run `cluster bootstrap` after the build job finishes to fill it.
- Bootstrap is idempotent: re-running it re-discovers, re-syncs the code (rsync) and re-runs
  `uv sync --frozen`; it never deletes remote files.
