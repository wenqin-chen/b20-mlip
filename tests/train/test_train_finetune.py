"""build_args per variant (rules R1-R4), splits, results parsing, run() dry-run/SLURM/local."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from b20mlip.config import Settings
from b20mlip.executors import CommandResult, SlurmExecutor
from b20mlip.io import read_frames, write_frames
from b20mlip.models import CheckpointInfo, Frame
from b20mlip.provenance import read_manifest, run_stage, sha256_file
from b20mlip.train import finetune as ft

from .conftest import make_split, retag


def _opt(args: list[str], flag: str) -> str:
    assert flag in args, f"{flag} missing from {args}"
    return args[args.index(flag) + 1]


# --- build_args -----------------------------------------------------------------------------------


def test_build_args_naive_qe_uses_e0s_file_never_foundation(
    qe_cfg: Settings, fake_foundation: Path, split_paths: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "run"
    args = ft.build_args(qe_cfg, "naive", split_paths, 3, out, energy_scale="qe")
    assert _opt(args, "--E0s") == qe_cfg.train.e0s_file  # rule R1
    assert _opt(args, "--E0s") != "foundation" and "--foundation_model_elements" not in args
    assert _opt(args, "--foundation_model") == str(fake_foundation.resolve())
    assert _opt(args, "--multiheads_finetuning") == "False"
    assert "--pt_train_file" not in args and "--hidden_irreps" not in args
    assert "--restart_latest" not in args
    # common flags, exactly what b20mlip.io.write_frames emits
    assert _opt(args, "--energy_key") == "energy"
    assert _opt(args, "--forces_key") == "forces"
    assert _opt(args, "--stress_key") == "stress"
    assert _opt(args, "--config_type_weights") == '{"Default":1.0}'
    assert _opt(args, "--name") == "b20_naive" and _opt(args, "--seed") == "3"
    assert _opt(args, "--train_file") == str(split_paths["train"])
    assert _opt(args, "--valid_file") == str(split_paths["valid"])
    assert _opt(args, "--test_file") == str(split_paths["test"])
    for flag, sub in (
        ("--work_dir", ""),
        ("--log_dir", "logs"),
        ("--results_dir", "results"),
        ("--checkpoints_dir", "checkpoints"),
        ("--model_dir", "models"),
        ("--downloads_dir", "downloads"),
    ):
        assert _opt(args, flag) == str(out / sub if sub else out)
    assert _opt(args, "--loss") == "universal"
    assert _opt(args, "--energy_weight") == "1.0"
    assert _opt(args, "--forces_weight") == "100.0"
    assert _opt(args, "--stress_weight") == "1.0"
    assert _opt(args, "--default_dtype") == "float64" and _opt(args, "--device") == "cpu"
    assert _opt(args, "--batch_size") == "4" and _opt(args, "--valid_batch_size") == "4"
    assert _opt(args, "--max_num_epochs") == "1" and _opt(args, "--lr") == "0.0001"
    assert "--ema" in args and _opt(args, "--ema_decay") == "0.99"
    assert "--save_cpu" in args and _opt(args, "--plot") == "False"
    assert args[-8:] == list(qe_cfg.train.extra_args)  # cfg extras appended verbatim, last


def test_build_args_naive_mp_uses_foundation_e0s(
    qe_cfg: Settings, split_paths: dict[str, Path], tmp_path: Path
) -> None:
    args = ft.build_args(qe_cfg, "naive", split_paths, 0, tmp_path, energy_scale="mp")
    assert _opt(args, "--E0s") == "foundation"
    assert _opt(args, "--foundation_model_elements") == "True"


@pytest.mark.parametrize("scale", ["none", "omat24"])
def test_build_args_naive_rejects_other_scales(
    qe_cfg: Settings, split_paths: dict[str, Path], tmp_path: Path, scale: str
) -> None:
    with pytest.raises(ValueError, match="'qe' or 'mp'"):
        ft.build_args(qe_cfg, "naive", split_paths, 0, tmp_path, energy_scale=scale)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'qe' or 'mp'"):
        ft.e0_policy("replay", scale, qe_cfg)  # type: ignore[arg-type]


def test_build_args_replay(qe_cfg: Settings, split_paths: dict[str, Path], tmp_path: Path) -> None:
    cfg = qe_cfg.model_copy(update={"train": qe_cfg.train.model_copy(update={"loss": "stress"})})
    args = ft.build_args(cfg, "replay", split_paths, 1, tmp_path, energy_scale="qe")
    assert _opt(args, "--multiheads_finetuning") == "True"
    assert _opt(args, "--pt_train_file") == "mp"
    assert _opt(args, "--num_samples_pt") == "10000"
    assert _opt(args, "--subselect_pt") == "fps"
    assert _opt(args, "--force_mh_ft_lr") == "True"
    assert _opt(args, "--E0s") == cfg.train.e0s_file  # Default head on the QE scale (R2)
    assert _opt(args, "--loss") == "universal"  # what MACE enforces for multihead runs
    assert "--foundation_model" in args and "--hidden_irreps" not in args


def test_build_args_scratch(
    tiny_cfg: Settings, split_paths: dict[str, Path], tmp_path: Path
) -> None:
    # no foundation file exists in tiny_cfg's models_dir: scratch must not need one
    args = ft.build_args(tiny_cfg, "scratch", split_paths, 0, tmp_path, energy_scale="none")
    assert "--foundation_model" not in args and "--multiheads_finetuning" not in args
    assert _opt(args, "--hidden_irreps") == "8x0e" and _opt(args, "--r_max") == "4.0"
    assert _opt(args, "--E0s") == "average"
    assert ft.e0_policy("scratch", "qe", tiny_cfg) == ("average", "estimated")


def test_build_args_bootstrap(
    qe_cfg: Settings, split_paths: dict[str, Path], tmp_path: Path
) -> None:
    args = ft.build_args(qe_cfg, "bootstrap", split_paths, 0, tmp_path, energy_scale="omat24")
    assert _opt(args, "--energy_weight") == "0.0"  # rule R4: forces + stress only
    assert _opt(args, "--forces_weight") == "100.0" and _opt(args, "--stress_weight") == "1.0"
    assert _opt(args, "--E0s") == "foundation"
    assert _opt(args, "--multiheads_finetuning") == "False"
    assert _opt(args, "--foundation_model_elements") == "True"
    assert ft.e0_policy("bootstrap", "omat24", qe_cfg) == ("foundation", "foundation")


def test_build_args_options(qe_cfg: Settings, split_paths: dict[str, Path], tmp_path: Path) -> None:
    two = {k: v for k, v in split_paths.items() if k != "test"}
    args = ft.build_args(
        qe_cfg, "naive", two, 0, tmp_path, energy_scale="qe", name="x", lr=3e-4, epochs=7,
        resume=True, foundation="foundation.model", e0s_file="E0s_qe.json",
        extra_args=["--lr", "1.0"],
    )  # fmt: skip
    assert "--test_file" not in args and "--restart_latest" in args
    assert _opt(args, "--name") == "x" and _opt(args, "--max_num_epochs") == "7"
    assert _opt(args, "--foundation_model") == "foundation.model"
    assert _opt(args, "--E0s") == "E0s_qe.json"
    assert args[-2:] == ["--lr", "1.0"] and args.index("--lr") < len(args) - 2  # call extras last
    assert _opt(args, "--lr") == "0.0003"
    with pytest.raises(ValueError, match="'train' and 'valid'"):
        ft.build_args(qe_cfg, "naive", {"train": split_paths["train"]}, 0, tmp_path)
    with pytest.raises(ValueError, match="unknown variant"):
        ft.build_args(qe_cfg, "lora", split_paths, 0, tmp_path)
    cmd = ft.train_command(["--name", "a b"])
    assert cmd[:3] == [sys.executable, "-m", "mace.cli.run_train"] and cmd[-1] == "a b"
    assert ft.join_argv(["--name", "a b", "--E0s", '{"Default":1.0}']).startswith("--name 'a b'")


def test_e0_policy_table(qe_cfg: Settings) -> None:
    assert ft.e0_policy("naive", "qe", qe_cfg) == (qe_cfg.train.e0s_file, "E0s_qe.json")
    assert ft.e0_policy("naive", "mp", qe_cfg) == ("foundation", "foundation")
    assert ft.e0_policy("replay", "qe", qe_cfg) == (qe_cfg.train.e0s_file, "E0s_qe.json")
    assert ft.check_variant("scratch") == "scratch"


# --- frames, splits, results ----------------------------------------------------------------------


def test_frames_energy_scale(tiny_frames: list[Frame]) -> None:
    assert ft.frames_energy_scale(retag(tiny_frames, "none", "none")) == "none"
    assert ft.frames_energy_scale(tiny_frames) == "qe"  # 12 unlabelled + 3 qe frames
    mixed = retag(tiny_frames[:5], "qe", "qe") + retag(tiny_frames[5:], "mp", "mptrj")
    with pytest.raises(ValueError, match="mixed energy scales"):
        ft.frames_energy_scale(mixed)


def test_group_split_is_deterministic_and_group_safe(tiny_frames: list[Frame]) -> None:
    train, valid = ft.group_split(tiny_frames, 0)
    again_train, again_valid = ft.group_split(list(tiny_frames), 0)
    assert [f.frame_id for f in train] == [f.frame_id for f in again_train]
    assert [f.frame_id for f in valid] == [f.frame_id for f in again_valid]
    assert len(train) + len(valid) == len(tiny_frames) and 1 <= len(valid) < len(tiny_frames)
    assert {f.group_id for f in train}.isdisjoint({f.group_id for f in valid})
    other_valid = ft.group_split(tiny_frames, 1)[1]
    assert {f.frame_id for f in other_valid} != {f.frame_id for f in valid}
    # two derivatives of one parent share a group: never split across train/valid
    same = [f.model_copy(update={"group_id": "FeSi/relax/p"}) for f in tiny_frames]
    assert ft.group_split(same, 0)[1] == []


def test_select_split_frames(qe_frames: list[Frame]) -> None:
    split = make_split(qe_frames)
    parts = ft.select_split_frames(split, qe_frames)
    assert {k: len(v) for k, v in parts.items()} == {"train": 7, "valid": 4, "test": 4}
    assert [f.frame_id for f in parts["valid"]] == split.val
    with pytest.raises(ValueError, match="frame ids not in the frames file"):
        ft.select_split_frames(split, qe_frames[:3])
    with pytest.raises(ValueError, match="empty train or valid"):
        ft.select_split_frames(split.model_copy(update={"val": []}), qe_frames)


def test_with_energy_weight_zero(tiny_frames: list[Frame]) -> None:
    out = ft.with_energy_weight_zero(tiny_frames)
    assert all(f.weights["energy"] == 0.0 for f in out)
    assert tiny_frames[0].weights == {}  # inputs untouched


def test_parse_val_metrics(tmp_path: Path) -> None:
    path = tmp_path / "x_run-0_train.txt"
    lines = [
        {"loss": 0.1, "mae_e": 0.5, "rmse_e_per_atom": 0.02, "rmse_f": 0.03, "mode": "eval",
         "epoch": None, "head": "Default"},
        {"loss": 0.5, "time": 0.4, "mode": "opt", "epoch": 0},
        {"loss": 0.2, "rmse_f": 0.9, "mode": "eval", "epoch": 0, "head": "pt_head"},
        {"loss": 0.05, "rmse_e_per_atom": 0.01, "mae_e_per_atom": 0.008, "rmse_f": 0.04,
         "mae_f": 0.03, "rmse_stress": 0.002, "mae_stress": 0.001, "mode": "eval", "epoch": 0,
         "head": "Default"},
    ]  # fmt: skip
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n", encoding="utf-8")
    metrics = ft.parse_val_metrics(path)
    assert metrics == {
        "rmse_e_per_atom_meV": pytest.approx(10.0),
        "mae_e_per_atom_meV": pytest.approx(8.0),
        "rmse_f_meV_A": pytest.approx(40.0),
        "mae_f_meV_A": pytest.approx(30.0),
        "rmse_stress_meV_A3": pytest.approx(2.0),
        "mae_stress_meV_A3": pytest.approx(1.0),
        "loss": pytest.approx(0.05),
        "epoch": 0.0,
    }
    assert ft.parse_val_metrics(path, head="pt_head")["rmse_f_meV_A"] == pytest.approx(900.0)
    assert ft.parse_val_metrics(tmp_path / "missing.txt") == {}
    path.write_text('{"mode": "opt", "loss": 1}\n', encoding="utf-8")
    assert ft.parse_val_metrics(path) == {}


def test_locate_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no model"):
        ft.locate_model(tmp_path, "m")
    ckpt = tmp_path / "checkpoints" / "m_run-0.model"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"c")
    assert ft.locate_model(tmp_path, "m") == ckpt
    final = tmp_path / "models" / "m.model"
    final.parent.mkdir()
    final.write_bytes(b"m")
    assert ft.locate_model(tmp_path, "m") == final
    two = tmp_path / "models" / "m_stagetwo.model"
    two.write_bytes(b"s")
    assert ft.locate_model(tmp_path, "m") == two


def test_resolve_foundation_and_zero_shot(
    tiny_cfg: Settings, fake_foundation: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # no real MACE cache
    assert ft.mace_cache_dir() == tmp_path / "cache" / "mace"
    assert ft.cache_name(ft.FOUNDATION_URLS["medium-mpa-0"]) == "macempa0mediummodel"
    assert ft.resolve_foundation(tiny_cfg) == fake_foundation.resolve()
    assert ft.resolve_foundation(tiny_cfg, str(fake_foundation)) == fake_foundation.resolve()
    assert ft.foundation_candidates("medium-mpa-0", tiny_cfg)[0] == fake_foundation
    with pytest.raises(FileNotFoundError, match="2023-12-03-mace-128-L1_epoch-199.model"):
        ft.resolve_foundation(tiny_cfg, "medium")
    with pytest.raises(FileNotFoundError, match="not found locally"):
        ft.resolve_foundation(tiny_cfg, str(tmp_path / "nope.model"))

    info = ft.zero_shot_checkpoint(tiny_cfg, "mpa-0")
    assert info.variant == "zero_shot" and info.heads == ["Default"]
    assert info.energy_scale == "mp" and info.e0_source == "foundation"
    assert info.sha256 == info.foundation_sha256 == sha256_file(fake_foundation)
    assert info.foundation == "medium-mpa-0" and info.epochs == 0 and info.train_run_id is None
    with pytest.raises(FileNotFoundError):
        ft.zero_shot_checkpoint(tiny_cfg, "mp-0")
    with pytest.raises(ValueError, match="which must be"):
        ft.zero_shot_checkpoint(tiny_cfg, "omat")


# --- run() ----------------------------------------------------------------------------------------


def test_run_dry_run_with_split_finds_frames_by_hash(
    qe_cfg: Settings, qe_frames: list[Frame], tmp_path: Path
) -> None:
    frames_dir = Path(qe_cfg.paths.data_dir) / "frames"
    write_frames(qe_frames, frames_dir / "labelled_r0.extxyz")
    write_frames(qe_frames[:2], frames_dir / "other.extxyz")  # a decoy with a different hash
    split = make_split(read_frames(frames_dir / "labelled_r0.extxyz"))  # hashed as read
    split_path = tmp_path / "v1.json"
    split_path.write_text(split.model_dump_json(), encoding="utf-8")

    result = run_stage(
        "train", qe_cfg, ft.run, seed=0, dry_run=True, variant="naive", split=split_path
    )
    assert result.status == "partial" and result.outputs == []
    manifest = read_manifest(result.manifest_path)
    out_dir = Path(result.manifest_path).parent
    plan = json.loads((out_dir / "plan.json").read_text())
    assert plan["dry_run"] is True and plan["variant"] == "naive"
    assert plan["counts"] == {"train": 7, "valid": 4, "test": 4}
    assert _opt(plan["argv"], "--E0s") == qe_cfg.train.e0s_file
    assert _opt(plan["argv"], "--test_file") == str(out_dir / "data" / "test.extxyz")
    assert len(read_frames(out_dir / "data" / "train.extxyz")) == 7
    assert manifest.extras["e0_source"] == "E0s_qe.json"
    assert manifest.extras["energy_scale"] == "qe"
    assert manifest.extras["split_id"] == split.split_id
    assert manifest.extras["split_frames_sha256_match"] is True
    assert manifest.extras["foundation_sha256"] == sha256_file(
        Path(qe_cfg.paths.models_dir) / "foundation" / "mace-mpa-0-medium.model"
    )
    kinds = sorted((Path(a.path).name, a.kind) for a in manifest.inputs)
    assert ("labelled_r0.extxyz", "frames") in kinds and ("v1.json", "json") in kinds
    assert ("E0s_qe.json", "json") in kinds and ("mace-mpa-0-medium.model", "model") in kinds
    assert not list((out_dir / "logs").glob("*")) if (out_dir / "logs").is_dir() else True

    # a split whose frames are nowhere under data_dir
    orphan = split.model_copy(update={"frames_sha256": "0" * 64})
    orphan_path = tmp_path / "orphan.json"
    orphan_path.write_text(orphan.model_dump_json(), encoding="utf-8")
    failed = run_stage(
        "train", qe_cfg, ft.run, seed=0, dry_run=True, variant="naive", split=orphan_path
    )
    assert failed.status == "failed" and "no extxyz" in str(failed.summary["error"])
    with pytest.raises(FileNotFoundError, match="no extxyz"):
        ft.find_frames_for_split(orphan, tmp_path / "absent")


def test_run_requires_qe_e0s_and_some_input(
    qe_cfg: Settings, qe_frames_path: Path, tmp_path: Path
) -> None:
    no_e0s = qe_cfg.model_copy(
        update={"train": qe_cfg.train.model_copy(update={"e0s_file": str(tmp_path / "none.json")})}
    )
    result = run_stage(
        "train", no_e0s, ft.run, seed=0, dry_run=True, variant="naive", frames=qe_frames_path
    )
    assert result.status == "failed" and "rule R1" in str(result.summary["error"])
    result = run_stage("train", qe_cfg, ft.run, seed=0, dry_run=True, variant="naive")
    assert result.status == "failed" and "--split" in str(result.summary["error"])
    result = run_stage(
        "train", qe_cfg, ft.run, dry_run=True, variant="naive", frames=tmp_path / "missing.extxyz"
    )
    assert result.status == "failed" and "frames file not found" in str(result.summary["error"])


def test_run_bootstrap_dry_run_writes_energy_weight_zero(
    qe_cfg: Settings, omat_frames_path: Path
) -> None:
    result = run_stage(
        "train", qe_cfg, ft.run, seed=2, dry_run=True, variant="bootstrap", frames=omat_frames_path
    )
    assert result.status == "partial", result.summary
    out_dir = Path(result.manifest_path).parent
    plan = json.loads((out_dir / "plan.json").read_text())
    assert _opt(plan["argv"], "--energy_weight") == "0.0"
    assert _opt(plan["argv"], "--E0s") == "foundation"
    # rule R4: energy weight 0 keeps the checkpoint on the foundation ("mp") scale
    assert plan["energy_scale"] == "mp" and plan["e0_source"] == "foundation"
    assert "--test_file" not in plan["argv"]  # in-function split has no test set
    for part in ("train", "valid"):
        frames = read_frames(out_dir / "data" / f"{part}.extxyz")
        assert frames and all(f.weights == {"energy": 0.0} for f in frames)  # rule R4
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["n_train"] + manifest.extras["n_valid"] == 15
    assert manifest.extras["split_id"] is None and manifest.seed == 2


def test_run_rejects_mixed_scales(
    qe_cfg: Settings, tiny_frames: list[Frame], tmp_path: Path
) -> None:
    mixed = retag(tiny_frames[:8], "qe", "qe") + retag(tiny_frames[8:], "mp", "mptrj")
    path = tmp_path / "mixed.extxyz"
    write_frames(mixed, path)
    result = run_stage("train", qe_cfg, ft.run, dry_run=True, variant="naive", frames=path)
    assert result.status == "failed" and "mixed energy scales" in str(result.summary["error"])


class SlurmFake:
    """Answers ssh/rsync/sbatch/sacct offline; the pull rsync 'fetches' a finished job."""

    def __init__(self, model_src: Path, name: str) -> None:
        self.calls: list[list[str]] = []
        self.model_src = model_src
        self.name = name

    def __call__(self, argv: list[str]) -> CommandResult:
        self.calls.append(list(argv))
        tup = tuple(argv)
        if argv[0] == "rsync":
            src, dst = argv[-2], argv[-1]
            if src.startswith("tillicum:"):
                dest = Path(dst.rstrip("/"))
                (dest / "models").mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.model_src, dest / "models" / f"{self.name}.model")
                (dest / "results").mkdir(exist_ok=True)
                (dest / "results" / f"{self.name}_run-0_train.txt").write_text(
                    json.dumps({"mode": "eval", "epoch": 0, "rmse_f": 0.05, "head": "Default"})
                    + "\n"
                )
                (dest / "logs").mkdir(exist_ok=True)
                (dest / "logs" / "unit.log").write_text("Done\n")
            return CommandResult(tup, 0, "", "")
        remote = argv[-1]
        if "sbatch" in remote:
            return CommandResult(tup, 0, "777;cluster\n", "")
        if "sacct" in remote:
            return CommandResult(tup, 0, "777|COMPLETED|0:0|acct|gpu-part|1|00:10:00\n", "")
        return CommandResult(tup, 0, "", "")


def test_run_replay_on_slurm_stages_files_and_fetches(
    cluster_cfg: Settings, qe_frames_path: Path, tmp_path: Path, tiny_mace: object
) -> None:
    runner = SlurmFake(getattr(tiny_mace, "model_path"), "b20_replay")  # noqa: B009
    executor = SlurmExecutor(
        cluster_cfg, runner=runner, staging_root=tmp_path / "staging", sleep=lambda s: None
    )
    result = run_stage(
        "train", cluster_cfg, ft.run, seed=0, executor=executor, variant="replay",
        frames=qe_frames_path,
    )  # fmt: skip
    assert result.status == "ok", result.summary
    manifest = read_manifest(result.manifest_path)
    out_dir = Path(result.manifest_path).parent
    info = CheckpointInfo.model_validate_json((out_dir / "checkpoint.json").read_text())
    assert info.variant == "replay" and info.replay_samples == 10000
    assert info.train_run_id == manifest.run_id == result.run_id
    assert info.e0_source == "E0s_qe.json" and info.energy_scale == "qe"
    assert info.heads == ["Default"]  # what the (tiny) fetched model reports
    assert info.model_path == str(out_dir / "job" / "models" / "b20_replay.model")
    assert info.sha256 == sha256_file(info.model_path) == getattr(tiny_mace, "sha256")  # noqa: B009
    assert info.val_metrics == {"rmse_f_meV_A": pytest.approx(50.0), "epoch": 0.0}
    assert manifest.slurm is not None and manifest.slurm.job_ids == ["777"]
    assert manifest.slurm.units_done == 1 and manifest.slurm.partition == "gpu-part"

    staging = tmp_path / "staging" / f"train_replay_{result.run_id}"
    assert (staging / "data" / "train.extxyz").is_file()
    assert (staging / "data" / "valid.extxyz").is_file()
    assert (staging / "foundation.model").is_file() and (staging / "E0s_qe.json").is_file()
    sbatch = (staging / "job.sbatch").read_text()
    assert "#SBATCH --partition=gpu-part" in sbatch and "#SBATCH --gpus=1" in sbatch
    assert "uv run --no-sync mace_run_train --name b20_replay" in sbatch
    assert "--restart_latest" in sbatch and "--pt_train_file mp" in sbatch
    assert "--foundation_model foundation.model --multiheads_finetuning True" in sbatch
    assert "--E0s E0s_qe.json" in sbatch
    remote = manifest.extras["mace_argv_remote"]
    assert _opt(remote, "--train_file") == "data/train.extxyz"
    assert _opt(remote, "--work_dir") == "." and _opt(remote, "--model_dir") == "models"
    assert any("sbatch --parsable job.sbatch" in c[-1] for c in runner.calls if c[0] == "ssh")


def test_run_local_failure_reports_the_log(tiny_cfg: Settings, tiny_b20_path: Path) -> None:
    result = run_stage(
        "train", tiny_cfg, ft.run, seed=0, variant="scratch", frames=tiny_b20_path,
        extra_args=["--no_such_flag"],
    )  # fmt: skip
    assert result.status == "failed"
    error = str(result.summary["error"])
    assert "mace_run_train exited with" in error and "no_such_flag" in error
    out_dir = Path(result.manifest_path).parent
    assert (out_dir / "logs" / "mace_stdout.log").is_file()
    manifest = read_manifest(result.manifest_path)
    assert manifest.extras["mace_returncode"] != 0


REAL_FOUNDATION = (
    Path(__file__).resolve().parents[2] / "models" / "foundation" / "mace-mpa-0-medium.model"
)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_FOUNDATION.is_file(), reason="MACE-MPA-0 medium weights not present")
def test_naive_finetune_with_real_foundation(
    tiny_cfg: Settings, tiny_frames: list[Frame], tmp_path: Path
) -> None:
    """Opt-in (``-m slow``): one naive epoch on MP-scale frames with the real MPA-0 medium."""
    cfg = tiny_cfg.model_copy(
        update={
            "train": tiny_cfg.train.model_copy(
                update={"foundation": str(REAL_FOUNDATION), "extra_args": []}
            )
        }
    )
    path = tmp_path / "mp.extxyz"
    write_frames(retag(tiny_frames, "mp", "mptrj"), path)
    result = run_stage("train", cfg, ft.run, seed=0, variant="naive", frames=path)
    assert result.status == "ok", result.summary
    out_dir = Path(result.manifest_path).parent
    info = CheckpointInfo.model_validate_json((out_dir / "checkpoint.json").read_text())
    assert info.foundation_sha256 == ft.FOUNDATION_SHA256["medium-mpa-0"]
    assert info.e0_source == "foundation" and info.energy_scale == "mp"
    assert info.heads == ["Default"] and info.epochs == 1
