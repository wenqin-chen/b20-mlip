"""lammps: input rendering (snapshot), golden-log parsing (exact), the sbatch template, and the
md.lammps stage end to end through the LocalExecutor with the fake ``lmp``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from b20mlip.config import Settings
from b20mlip.executors import CommandResult, JobHandle, JobSpec, LocalExecutor, SlurmExecutor
from b20mlip.md import lammps
from b20mlip.models import MDResult, SlurmInfo
from b20mlip.provenance import read_manifest, run_stage, sha256_file

GOLDEN_LX = [
    10.52, 10.52002839, 10.52011392, 10.52025768, 10.52046146, 10.52072776, 10.52105976,
    10.52146129, 10.52193682, 10.52249142, 10.52313078, 10.52386119, 10.52468943, 10.5256226,
    10.5266685, 10.52783505,
]  # fmt: skip
GOLDEN_TOTENG = [
    -2.294582654, -2.294585562, -2.294594297, -2.294608017, -2.294625382, -2.294644633,
    -2.294663711, -2.294680394, -2.294692463, -2.294357798, -2.29401587, -2.293494553,
    -2.293303085, -2.29174473, -2.291199084, -2.290648707,
]  # fmt: skip
GOLDEN_STEPS = list(range(0, 31, 2))

EXPECTED_HEADER = (
    "# b20mlip-md engine=lammps compound=MnSi ensemble=npt T=300.0 timestep_fs=2.0 "
    "equil_steps=5000 steps=50000 reps=4 natoms_per_cell=8 natoms=512 head=Default "
    "model_label=B2 model_sha256=abc123 seed=7\n"
)
EXPECTED_NPT_INPUT = (
    EXPECTED_HEADER
    + """\
# rendered from templates/lammps/in.mace.j2 (b20mlip.md.lammps.render_input); ML-MACE syntax per
# https://mace-docs.readthedocs.io/en/latest/guide/lammps.html
units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p
read_data       data.lmp

pair_style      mace no_domain_decomposition
pair_coeff      * * b20_replay.model-lammps.pt Si Mn

timestep        0.002
velocity        all create 300.0 7 mom yes dist gaussian
fix             1 all npt temp 300.0 300.0 $(100*dt) iso 0 0 $(1000*dt)

thermo_style    custom step temp pe ke etotal press vol lx
thermo_modify   format float %.10g
thermo          10
# equilibration (5000 steps = cfg.md.equil_ps); excluded from every average by parse_thermo
run             5000
dump            1 all custom 100 dump.lammpstrj id type x y z vx vy vz
dump_modify     1 sort id
# production
run             50000
write_data      final.data
"""
)


# --- rendering ------------------------------------------------------------------------------------


def test_render_input_snapshot() -> None:
    cfg = Settings.model_validate({})
    text = lammps.render_input(
        cfg, "b20_replay.model-lammps.pt", ["Si", "Mn"], 300.0, 100.0, 512, ensemble="npt",
        natoms_per_cell=8, seed=7, compound="MnSi", head="Default", model_label="B2",
        model_sha256="abc123",
    )  # fmt: skip
    assert text == EXPECTED_NPT_INPUT
    nvt = lammps.render_input(cfg, "m-lammps.pt", ["Fe", "Si"], 500.0, 1.0, 64, ensemble="nvt")
    assert "fix             1 all nvt temp 500.0 500.0 $(100*dt)\n" in nvt and "iso" not in nvt
    assert "reps=2 natoms_per_cell=8 natoms=64" in nvt and "seed=12345" in nvt  # seed 0 -> 12345
    nve = lammps.render_input(
        cfg, "m-lammps.pt", ["Fe", "Ge"], 100.0, 0.02, 8, ensemble="nve", equil_ps=0.0
    )
    assert "fix             1 all nve\n" in nve and "equilibration" not in nve
    assert "run             10\n" in nve and nve.count("run ") == 1
    for line in nve.splitlines():
        assert "{{" not in line and "{%" not in line
    with pytest.raises(ValueError):
        lammps.render_input(cfg, "m", [], 300.0, 1.0, 64)
    with pytest.raises(ValueError):
        lammps.render_input(cfg, "m", ["Fe"], 300.0, 1.0, 64, ensemble="nph")


def test_write_data_and_elements(tmp_path: Path, b20_atoms) -> None:  # type: ignore[no-untyped-def]
    path = lammps.write_data(b20_atoms("MnSi"), tmp_path / "d" / "data.lmp", ["Si", "Mn"])
    text = path.read_text()
    assert "8 atoms" in text and "2 atom types" in text and "# Si" in text and "# Mn" in text
    assert "Atoms # atomic" in text


def test_lammps_command_unit_script_and_local_check(
    md_settings: Settings, fake_lmp_cmd: str
) -> None:
    with pytest.raises(FileNotFoundError, match="lammps_cmd not found"):
        lammps.lammps_command(md_settings, gpu=False)
    cfg = md_settings.model_copy(
        update={"cluster": md_settings.cluster.model_copy(update={"lammps_cmd": "lmp"})}
    )
    assert lammps.lammps_command(cfg, gpu=False) == "lmp"
    assert lammps.lammps_command(cfg, gpu=True) == f"lmp {lammps.KOKKOS_FLAGS}"
    assert lammps.KOKKOS_FLAGS == "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    assert lammps.unit_script("lmp") == 'cd "$B20_WORKDIR"\nlmp -in in.mace -log log.lammps\n'
    with pytest.raises(FileNotFoundError, match="lammps_cmd not found"):
        lammps.check_local_command("lmp_mace_definitely_absent -in x")
    assert lammps.check_local_command(fake_lmp_cmd).endswith("python") or True
    assert lammps.check_local_command("bash -c true") == "bash"


# --- parsing --------------------------------------------------------------------------------------


def test_parse_thermo_golden_exact(golden_log: Path) -> None:
    text = golden_log.read_text()
    header = lammps.parse_header(text)
    assert header == {
        "engine": "lammps", "compound": "Ar", "ensemble": "npt", "T": 100.0, "timestep_fs": 2.0,
        "equil_steps": 10, "steps": 20, "reps": 2, "natoms_per_cell": 4, "natoms": 32,
        "head": "Default", "model_label": "golden-lj", "model_sha256": "0" * 64, "seed": 12345,
    }  # fmt: skip
    blocks = lammps.parse_thermo_blocks(text)
    assert len(blocks) == 2 and [b["rows"].shape for b in blocks] == [(6, 8), (11, 8)]
    assert blocks[0]["columns"] == [
        "step",
        "temp",
        "poteng",
        "kineng",
        "toteng",
        "press",
        "volume",
        "lx",
    ]
    assert blocks[0]["loop"] == {"seconds": 0.000134834, "procs": 1, "steps": 10, "natoms": 32}
    assert blocks[1]["loop"]["steps"] == 20 and blocks[1]["loop"]["natoms"] == 32
    table = lammps.thermo_table(blocks)
    assert table["step"].tolist() == GOLDEN_STEPS  # the repeated step 10 is dropped
    assert table["lx"].tolist() == GOLDEN_LX and table["toteng"].tolist() == GOLDEN_TOTENG
    assert table["temp"][0] == 100.0 and table["poteng"][0] == -2.695289103
    assert lammps.parse_timestep_ps(text) == 0.002

    res = lammps.parse_thermo(golden_log, run_id="r0", traj_sha256="abc")
    lx = np.array(GOLDEN_LX)
    prod = lx[5:] / 2  # t >= equil (10 steps = 0.02 ps): steps 10..30 -> 11 rows
    assert isinstance(res, MDResult) and res.engine == "lammps" and res.ensemble == "npt"
    assert (
        res.natoms == 32
        and res.steps == 30
        and res.timestep_fs == 2.0
        and res.temperature_K == 100.0
    )
    assert res.compound == "Ar" and res.head == "Default" and res.model_sha256 == "0" * 64
    assert res.a_mean_A == pytest.approx(prod.mean(), abs=1e-12)
    assert res.a_mean_A == pytest.approx(5.261794754545455)
    assert res.a_std_A == pytest.approx(prod.std(ddof=1), abs=1e-12)
    assert res.pressure_GPa == 0.0 and res.drift_meV_atom_ps is None and res.alpha_per_K is None
    assert res.run_id == "r0" and res.traj_sha256 == "abc" and res.vdos_path is None

    info = lammps.analyze_thermo(text)
    assert info["n_rows"] == 16 and info["n_production"] == 11 and info["n_runs"] == 2
    assert info["natoms_log"] == 32 and info["equil_ps"] == 0.02 and info["reps"] == 2
    assert info["T_mean_K"] == pytest.approx(85.76621407090909)
    assert info["a_ci95"] is not None and info["production_window"].startswith("production")
    assert info["loop_seconds"] == pytest.approx(0.000134834 + 0.000411375)
    assert info["model_label"] == "golden-lj"

    # explicit overrides win over the header: NVE drift from etotal over the production window
    nve = lammps.parse_thermo(golden_log, ensemble="nve", equil_ps=0.0, compound="X", reps=1)
    t = np.array(GOLDEN_STEPS) * 0.002
    slope = np.polyfit(t, np.array(GOLDEN_TOTENG) / 32 * 1000, 1)[0]
    assert nve.drift_meV_atom_ps == pytest.approx(slope, rel=1e-9)
    assert nve.drift_meV_atom_ps == pytest.approx(1.8162721966909487)
    assert nve.a_mean_A is None and nve.compound == "X" and nve.pressure_GPa is None
    nve_prod = lammps.analyze_thermo(text, ensemble="nve")
    assert nve_prod["drift_meV_atom_ps"] == pytest.approx(3.287438281250447)
    assert nve_prod["drift_ci95"] is not None


def test_parse_thermo_without_header_and_error_paths(golden_log: Path, tmp_path: Path) -> None:
    text = "\n".join(
        ln for ln in golden_log.read_text().splitlines() if not ln.startswith("# b20mlip")
    )
    assert lammps.parse_header(text) == {}
    with pytest.raises(ValueError, match="ensemble"):
        lammps.analyze_thermo(text)
    info = lammps.analyze_thermo(text, ensemble="npt")  # timestep from the echoed command
    assert info["timestep_fs"] == 2.0 and info["equil_ps"] == 0.0 and info["reps"] == 1
    assert info["a_mean_A"] == pytest.approx(np.mean(GOLDEN_LX))
    no_ts = "\n".join(ln for ln in text.splitlines() if not ln.startswith("timestep"))
    with pytest.raises(ValueError, match="time step"):
        lammps.analyze_thermo(no_ts, ensemble="npt")
    assert lammps.analyze_thermo(no_ts, ensemble="npt", timestep_fs=1.0)["steps"] == 30
    with pytest.raises(ValueError, match="no thermo block"):
        lammps.thermo_table([])
    with pytest.raises(ValueError, match="no thermo block"):
        lammps.analyze_thermo("LAMMPS (22 Jul 2025)\nTotal wall time: 0:00:00\n", ensemble="nve")
    # a crashed run: a block without its Loop time line still parses; natoms from the header
    crashed = text.split("Loop time")[0]
    blocks = lammps.parse_thermo_blocks(crashed)
    assert len(blocks) == 1 and blocks[0]["loop"] is None and blocks[0]["rows"].shape == (6, 8)
    with pytest.raises(ValueError, match="atom count"):
        lammps.analyze_thermo(crashed, ensemble="nve")
    assert lammps.analyze_thermo(crashed, ensemble="nve", natoms=32)["steps"] == 10
    # warnings inside a block are skipped; changed columns are refused; volume-only NPT logs work
    noisy = (
        "   Step Temp PotEng KinEng TotEng Press Volume\n"
        "0 100 -2.0 0.4 -1.6 1.0 1000.0\n"
        "WARNING: something (src/x.cpp:1)\n"
        "10 90 -2.1 0.3 -1.8 2.0 1030.301\n"
        "Loop time of 1.0 on 1 procs for 10 steps with 8 atoms\n"
    )
    b = lammps.parse_thermo_blocks(noisy)
    assert b[0]["rows"].shape == (2, 7)
    vol_only = lammps.analyze_thermo(noisy, ensemble="npt", timestep_fs=1.0, equil_ps=0.0)
    assert vol_only["a_mean_A"] == pytest.approx((10.0 + 10.1) / 2, abs=1e-9)
    with pytest.raises(ValueError, match="columns changed"):
        lammps.thermo_table(
            b + [{"columns": ["step", "temp"], "rows": np.zeros((1, 2)), "loop": None}]
        )
    with pytest.raises(ValueError, match="lx or volume"):
        lammps.analyze_thermo(
            "   Step Temp\n0 1\n10 2\nLoop time of 1 on 1 procs for 10 steps with 8 atoms\n",
            ensemble="npt", timestep_fs=1.0,
        )  # fmt: skip
    with pytest.raises(ValueError, match="Step column"):
        lammps.thermo_table([{"columns": ["temp"], "rows": np.zeros((1, 1)), "loop": None}])


def test_read_dump_and_dump_analysis(
    tmp_path: Path, md_settings: Settings, argon_cell: Atoms
) -> None:
    from b20mlip.provenance import RunContext

    os.chdir(tmp_path)
    import subprocess
    import sys

    from .conftest import FAKE_LMP

    lammps.write_data(argon_cell.repeat(2), tmp_path / "data.lmp", ["Ar"])
    (tmp_path / "in.mace").write_text("# b20mlip-md engine=lammps\nrun 0\n")
    subprocess.run(
        [sys.executable, str(FAKE_LMP), "-in", "in.mace", "-log", "log.lammps"], check=True
    )
    images = lammps.read_dump(tmp_path / "dump.lammpstrj", ["Ar"])
    assert len(images) == 6 and len(images[0]) == 32 and images[0].get_chemical_symbols()[0] == "Ar"
    assert images[-1].info["timestep"] == 50 and np.abs(images[0].get_velocities()).max() > 0
    ctx = RunContext("md.lammps", md_settings)
    vdos_path, rdf_path, n = lammps.dump_analysis(
        tmp_path / "dump.lammpstrj", ["Ar"], md_settings, ctx, "npt_100K", equil_steps=10
    )
    assert n == 6 and vdos_path and rdf_path
    vd = json.loads(Path(vdos_path).read_text())
    assert vd["n_frames"] == 5 and vd["dt_fs"] == 20.0 and vd["source"] == "dump.lammpstrj"
    rd = json.loads(Path(rdf_path).read_text())
    assert rd["n_frames"] == 5 and rd["rmax_A"] < 5.26
    # too few production frames -> no VDOS, RDF from what is there
    vdos_path, rdf_path, n = lammps.dump_analysis(
        tmp_path / "dump.lammpstrj", ["Ar"], md_settings, ctx, "x", equil_steps=40
    )
    assert vdos_path is None and rdf_path is not None


# --- the sbatch template -------------------------------------------------------------------------


def _runner(argv: list[str]) -> CommandResult:  # rendering only
    raise AssertionError(f"unexpected command {argv}")


PROTOCOL_LINES = (
    'export B20_UNIT=$(sed -n 1p "$B20_WORKDIR/units.txt")',
    "slug=$(printf '%s' \"$B20_UNIT\" | tr -c 'A-Za-z0-9._-' '_')",
    '[ -f "$B20_WORKDIR/units/$slug.done" ] && { echo "unit $B20_UNIT already done"; exit 0; }',
    'rm -f "$B20_WORKDIR/units/$slug.failed"',
    'if run_unit > "$B20_WORKDIR/logs/$slug.log" 2>&1; then',
    '> "$B20_WORKDIR/units/$slug.done"',
    '> "$B20_WORKDIR/units/$slug.failed"',
    "exit $rc",
    "set -euo pipefail",
)


def test_lammps_sbatch_template(slurm_settings: Settings, repo: Path) -> None:
    ex = SlurmExecutor(slurm_settings, runner=_runner, template_dir=repo / "templates" / "slurm")
    spec = JobSpec(
        name="md-lammps-MnSi-r1", script=lammps.unit_script("lmp"), units=["r1"],
        resources={"template": "lammps", "input": "in.mace", "gpu": True, "gpus": 1,
                   "partition": "gpu-h200", "lammps_cmd": "/gscratch/b20/b20-mlip/bin/lmp"},
        env={"LAMMPS_CMD": "/gscratch/b20/b20-mlip/bin/lmp -k on g 1 -sf kk"},
    )  # fmt: skip
    text = ex.render(spec)
    assert text.startswith(
        "#!/bin/bash\n#SBATCH --job-name=md-lammps-MnSi-r1\n#SBATCH --account=acct\n"
    )
    assert "#SBATCH --partition=gpu-h200\n" in text and "#SBATCH --gpus=1\n" in text
    assert "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=6\n" in text
    assert "#SBATCH --mem=32G\n#SBATCH --time=04:00:00\n" in text
    assert (
        "#SBATCH --output=/gscratch/b20/b20-mlip/jobs/md-lammps-MnSi-r1/logs/slurm-%j.out" in text
    )
    assert "module load gcc\nmodule load cuda/12.6\n" in text
    assert "export OMP_NUM_THREADS=6\n" in text
    assert 'export LAMMPS_CMD="/gscratch/b20/b20-mlip/bin/lmp -k on g 1 -sf kk"\n' in text
    assert (
        "run_unit() { /gscratch/b20/b20-mlip/bin/lmp -k on g 1 -sf kk -pk kokkos newton on "
        "neigh half -in in.mace -log log.lammps; }\n"
    ) in text
    assert 'cd "$B20_WORKDIR"\n' in text and "srun" not in text
    for line in PROTOCOL_LINES:
        assert line in text, line
    for line in text.splitlines():
        assert "{{" not in line and "{%" not in line and "{#" not in line
    # CPU variant, cluster defaults, parity mode (no input -> script.sh with $LAMMPS_CMD)
    cpu = ex.render(
        spec.model_copy(
            update={"resources": {"template": "lammps", "gpu": False, "time": "00:30:00"}}
        )
    )
    assert "#SBATCH --partition=gpu-h200\n" in cpu  # cluster.partition_gpu is the default
    assert "--gpus" not in cpu and "#SBATCH --time=00:30:00\n" in cpu
    assert 'export LAMMPS_CMD="/gscratch/b20/b20-mlip/bin/lmp"\n' in cpu and "kokkos" not in cpu
    assert 'run_unit() { bash "$B20_WORKDIR/script.sh"; }\n' in cpu
    plain_cpu = ex.render(
        spec.model_copy(
            update={"resources": {"template": "lammps", "gpu": False, "input": "in.mace"}}
        )
    )
    assert (
        "run_unit() { /gscratch/b20/b20-mlip/bin/lmp -in in.mace -log log.lammps; }\n" in plain_cpu
    )
    no_cmd = slurm_settings.model_copy(
        update={
            "cluster": slurm_settings.cluster.model_copy(
                update={"lammps_cmd": None, "qos": "gpu-qos"}
            )
        }
    )
    ex2 = SlurmExecutor(no_cmd, runner=_runner, template_dir=repo / "templates" / "slurm")
    fallback = ex2.render(spec.model_copy(update={"resources": {"template": "lammps"}}))
    assert 'export LAMMPS_CMD="lmp -k on g 1' in fallback and "#SBATCH --qos=gpu-qos\n" in fallback


# --- the stage ------------------------------------------------------------------------------------


def _run(cfg: Settings, **kw):  # type: ignore[no-untyped-def]
    return run_stage("md.lammps", cfg, lammps.run, executor=LocalExecutor(cfg), seed=0, **kw)


def test_run_end_to_end_with_the_fake_lammps(
    lammps_settings: Settings,
    tiny_mace,
    argon_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = lammps_settings
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_LMP_CALLS", str(calls))
    result = _run(
        cfg, model=tiny_mace.model_path, compound="Ar", T=100.0, ps=0.04, natoms=32,
        structure=argon_file,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    s = result.summary
    assert (
        s["engine"] == "lammps"
        and s["ensemble"] == "npt"
        and s["natoms"] == 32
        and s["steps"] == 30
    )
    # cfg.md.equil_ps is 0 in the fast fixture: every one of the 16 golden rows is production
    assert s["a_mean_A"] == pytest.approx(np.mean(GOLDEN_LX) / 2) and s["model_label"] == "tiny_b20"
    assert 70.0 < s["T_mean_K"] < 100.0 and s["n_production"] == 16
    assert s["log"] == "job/log.lammps" and "a_exp_A" not in s
    run_dir = Path(result.manifest_path).parent
    manifest = read_manifest(result.manifest_path)
    names = {Path(a.path).name for a in manifest.outputs}
    assert {
        "log.lammps", "dump.lammpstrj", "in.mace", "data.lmp", "numbers.json", "handle.json",
        "md_result_npt_100K.json", "vdos_npt_100K.json", "rdf_npt_100K.json",
    } <= names  # fmt: skip
    assert {Path(a.path).name for a in manifest.inputs} >= {
        tiny_mace.model_path.name,
        tiny_mace.lammps_path.name,
    }
    ex = manifest.extras
    assert (
        ex["engine"] == "lammps"
        and ex["timestep_fs"] == 2.0
        and ex["thermostat"].startswith("fix npt")
    )
    assert ex["gpu"] is False and ex["lammps_command"] == cfg.cluster.lammps_cmd
    assert ex["pair_style"] == "mace no_domain_decomposition" and ex["reps"] == 2
    assert ex["job_ids"][0].startswith("local:md-lammps-Ar-") and ex["stats"]["n_runs"] == 2
    assert ex["stats"]["n_dump_frames"] == 6 and ex["model_sha256"] == tiny_mace.sha256
    job_dir = Path(ex["job_dir"])
    assert job_dir == LocalExecutor(cfg).workdir(f"md-lammps-Ar-{result.run_id}")
    assert (job_dir / tiny_mace.lammps_path.name).is_file() and (job_dir / "md_job.json").is_file()
    job_meta = json.loads((job_dir / "md_job.json").read_text())
    assert job_meta["elements"] == ["Ar"] and job_meta["lammps_sha256"] == tiny_mace.lammps_sha256
    assert calls.read_text().split() == [str(job_dir), "in.mace"]
    in_text = (run_dir / "job" / "in.mace").read_text()
    assert f"pair_coeff      * * {tiny_mace.lammps_path.name} Ar" in in_text
    assert (
        "compound=Ar ensemble=npt T=100.0 timestep_fs=2.0 equil_steps=0 steps=20 reps=2" in in_text
    )
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["script"].endswith("-in in.mace -log log.lammps\n") and spec["units"] == [
        result.run_id
    ]
    assert spec["resources"]["template"] == "lammps" and spec["resources"]["input"] == "in.mace"
    res = MDResult.model_validate_json((run_dir / "md_result_npt_100K.json").read_text())
    assert res.engine == "lammps" and res.traj_sha256 == sha256_file(
        run_dir / "job" / "dump.lammpstrj"
    )
    assert res.model_sha256 == tiny_mace.sha256 and res.run_id == result.run_id and res.natoms == 32
    assert res.vdos_path == str(run_dir / "vdos_npt_100K.json") and res.head == "Default"
    numbers = json.loads((run_dir / "numbers.json").read_text())
    keys = {k for k in numbers if not k.endswith("@meta")}
    assert keys == {
        "md.lammps.Ar.tiny_b20.npt.100.a_mean_A",
        "md.lammps.Ar.tiny_b20.npt.100.a_std_A",
    }
    meta = numbers["md.lammps.Ar.tiny_b20.npt.100.a_mean_A@meta"]
    assert meta["engine"] == "lammps" and meta["n"] == 16 and meta["ci95"] is not None
    assert meta["reference"]["code"] == "mace" and meta["reference"]["functional"] == "PBE"
    # a -lammps.pt as --model, NVE, a_exp override, 300 K alias keys
    nve = _run(
        cfg, model=tiny_mace.lammps_path, compound="Ar", T=300.0, ps=0.04, natoms=32,
        structure=argon_file, ensemble="nve", label="B0",
    )  # fmt: skip
    assert nve.status == "ok" and "drift_meV_atom_ps" in nve.summary
    numbers = json.loads((Path(nve.manifest_path).parent / "numbers.json").read_text())
    assert {"md.lammps.Ar.B0.nve.300.drift_meV_atom_ps", "md.lammps.Ar.drift_meV_atom_ps"} <= set(
        numbers
    )
    npt300 = _run(
        cfg, model=tiny_mace.model_path, compound="Ar", T=300.0, ps=0.04, natoms=32,
        structure=argon_file, a_exp_A=5.3,
    )  # fmt: skip
    assert npt300.status == "ok" and npt300.summary["a_exp_A"] == 5.3
    numbers = json.loads((Path(npt300.manifest_path).parent / "numbers.json").read_text())
    assert numbers["md.lammps.Ar.a_exp_A"] == 5.3 and "md.lammps.Ar.a_dev_pct" in numbers
    assert numbers["md.lammps.Ar.a_300K_A@meta"]["reference"]["code"] == "experiment"


def test_run_fails_cleanly_without_lammps(
    md_settings: Settings, tiny_mace, argon_file: Path
) -> None:  # type: ignore[no-untyped-def]
    unset = _run(
        md_settings,
        model=tiny_mace.model_path,
        compound="Ar",
        T=100.0,
        ps=0.01,
        natoms=32,
        structure=argon_file,
    )
    assert unset.status == "failed" and "lammps_cmd not found" in unset.summary["error"]
    absent = md_settings.model_copy(
        update={
            "cluster": md_settings.cluster.model_copy(
                update={"lammps_cmd": "lmp_mace_absent -sf kk"}
            )
        }
    )
    res = _run(
        absent,
        model=tiny_mace.model_path,
        compound="Ar",
        T=100.0,
        ps=0.01,
        natoms=32,
        structure=argon_file,
    )
    assert res.status == "failed" and "lammps_cmd not found" in res.summary["error"]
    assert read_manifest(res.manifest_path).status == "failed"
    no_export = _run(
        absent,
        model=argon_file.with_suffix(".model"),
        compound="Ar",
        T=100.0,
        ps=0.01,
        structure=argon_file,
    )
    assert no_export.status == "failed" and "b20mlip export" in no_export.summary["error"]
    no_exec = run_stage(
        "md.lammps", absent, lammps.run, executor=None, model=tiny_mace.model_path, compound="Ar",
        T=100.0, ps=0.01, structure=argon_file,
    )  # fmt: skip
    assert no_exec.status == "failed" and "no executor" in no_exec.summary["error"]


def test_run_dry_run_failed_unit_no_wait_and_resume(
    lammps_settings: Settings,
    tiny_mace,
    argon_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = lammps_settings
    dry = run_stage(
        "md.lammps", cfg, lammps.run, executor=LocalExecutor(cfg), dry_run=True,
        model=tiny_mace.model_path, compound="Ar", T=100.0, ps=0.04, natoms=32,
        structure=argon_file,
    )  # fmt: skip
    assert dry.status == "partial" and dry.summary["planned"] == 1 and dry.outputs == []
    job_dir = Path(dry.summary["job_dir"])
    assert (job_dir / "in.mace").is_file() and (job_dir / "data.lmp").is_file()
    assert not (job_dir / "log.lammps").exists()

    monkeypatch.setenv("FAKE_LMP_FAIL", "1")
    failed = _run(
        cfg,
        model=tiny_mace.model_path,
        compound="Ar",
        T=100.0,
        ps=0.04,
        natoms=32,
        structure=argon_file,
    )
    assert failed.status == "failed" and "failed" in failed.summary["error"]
    assert "forced failure" in failed.summary["error"]
    monkeypatch.delenv("FAKE_LMP_FAIL")

    class FakeSlurm:
        def __init__(self) -> None:
            self.specs: list[JobSpec] = []
            self.fetched: list[Path] = []

        def submit(self, spec: JobSpec) -> JobHandle:
            self.specs.append(spec)
            return JobHandle(job_ids=["4242"], workdir="/gscratch/b20/b20-mlip/jobs/x")

        def wait(self, handle: JobHandle, poll_s: float = 60) -> SlurmInfo:
            return SlurmInfo(job_ids=handle.job_ids, account="a", partition="gpu", nodes=1,
                             wall="00:10:00", units_done=1, units_failed=0)  # fmt: skip

        def fetch(self, handle: JobHandle, dest: Path) -> list:
            self.fetched.append(dest)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "log.lammps").write_text(
                (
                    Path(__file__).resolve().parents[1]
                    / "fixtures"
                    / "golden"
                    / "lammps_thermo.log"
                ).read_text()
            )
            return []

    ex = FakeSlurm()
    nowait = run_stage(
        "md.lammps", cfg, lammps.run, executor=ex, model=tiny_mace.model_path, compound="Ar",
        T=100.0, ps=0.04, natoms=32, structure=argon_file, wait=False, gpu=True,
    )  # fmt: skip
    assert (
        nowait.status == "partial"
        and nowait.summary["waited"] == 0
        and json.loads(nowait.summary["job_ids"]) == ["4242"]
    )
    assert ex.specs[0].resources["gpu"] is True and ex.specs[0].env["LAMMPS_CMD"].endswith(
        lammps.KOKKOS_FLAGS
    )
    assert (
        ex.specs[0].script.startswith('cd "$B20_WORKDIR"\n') and "-in in.mace" in ex.specs[0].script
    )
    assert (Path(nowait.manifest_path).parent / "handle.json").is_file()
    job_dir = Path(read_manifest(nowait.manifest_path).extras["job_dir"])
    assert job_dir == Path(nowait.manifest_path).parent / "job"  # unknown executor: run dir
    # waited: fetch brings the log back and accounting lands in the manifest
    waited = run_stage(
        "md.lammps", cfg, lammps.run, executor=ex, model=tiny_mace.model_path, compound="Ar",
        T=100.0, ps=0.04, natoms=32, structure=argon_file,
    )  # fmt: skip
    assert waited.status == "ok" and read_manifest(waited.manifest_path).slurm is not None
    assert waited.summary["a_mean_A"] == pytest.approx(np.mean(GOLDEN_LX) / 2)
    res = MDResult.model_validate_json(
        (Path(waited.manifest_path).parent / "md_result_npt_100K.json").read_text()
    )
    assert res.traj_sha256 == sha256_file(Path(waited.manifest_path).parent / "job" / "log.lammps")
    assert res.vdos_path is None  # no dump came back
    # resume: the partial run's directory already holds job/log.lammps -> no submission
    fetched = Path(nowait.manifest_path).parent / "job"
    ex.fetch(JobHandle(job_ids=["4242"], workdir="x"), fetched)
    n_specs = len(ex.specs)
    resumed = run_stage(
        "md.lammps", cfg, lammps.run, executor=ex, resume=True, model=tiny_mace.model_path,
        compound="Ar", T=100.0, ps=0.04, natoms=32, structure=argon_file, wait=False,
    )  # fmt: skip
    assert resumed.status == "ok" and resumed.run_id == nowait.run_id and len(ex.specs) == n_specs
    assert read_manifest(resumed.manifest_path).extras["resumed"] is True
