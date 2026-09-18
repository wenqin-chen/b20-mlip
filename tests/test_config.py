"""CONTRACTS.md section 4: merge order default.yaml -> YAMLs -> env (B20_, __) -> --set; sha256."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from b20mlip.config import (
    Settings,
    deep_merge,
    default_config_path,
    env_layer,
    load_config,
    parse_overrides,
    read_yaml,
    repo_root,
)


def test_repo_root_and_default_config_exist(repo: Path) -> None:
    assert repo_root() == repo
    assert default_config_path() == repo / "configs" / "default.yaml"
    assert default_config_path().is_file()


def test_default_yaml_equals_model_defaults() -> None:
    data = read_yaml(default_config_path())
    expected_sections = {
        "paths", "compute", "cluster", "data", "dft", "train", "eval", "md",
        "sampling", "agent", "report",
    }  # fmt: skip
    assert set(data) == expected_sections
    from_yaml = Settings.model_validate(data)
    from_defaults = Settings.model_validate({})
    assert from_yaml.model_dump() == from_defaults.model_dump()
    assert from_yaml.sha256() == from_defaults.sha256()


def test_spec_values_in_defaults() -> None:
    cfg = load_config([], [])
    assert cfg.compute.threads == 6 and cfg.compute.dtype == "float64"
    assert cfg.data.force_cap_eVA == 15.0 and cfg.data.wbm_sample_n == 1000
    assert cfg.dft.k_spacing_inv_A == 0.25 and cfg.dft.smearing == "mv"
    assert cfg.dft.thresholds.E_meV_atom == 1.0 and cfg.dft.thresholds.F_meV_A == 5.0
    assert cfg.train.foundation == "medium-mpa-0" and cfg.train.batch_size == 4
    assert cfg.train.replay.num_samples_pt == 10000 and cfg.train.replay.subselect_pt == "fps"
    assert cfg.eval.bootstrap_n == 2000 and cfg.eval.offset_residual_gate_meV == 20
    assert cfg.sampling.windows == 12 and cfg.md.timestep_fs == 2.0
    assert cfg.agent.backend == "mock" and cfg.agent.budget.max_calls == 25
    assert cfg.report.numbers_path == Path("reports/numbers.json")


def test_merge_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    y1 = tmp_path / "one.yaml"
    y1.write_text(yaml.safe_dump({"compute": {"threads": 4}, "cluster": {"account": "acct-a"}}))
    y2 = tmp_path / "two.yaml"
    y2.write_text(yaml.safe_dump({"compute": {"threads": 5}}))

    cfg = load_config([y1], [])
    assert cfg.compute.threads == 4 and cfg.cluster.account == "acct-a"
    cfg = load_config([y1, y2], [])
    assert cfg.compute.threads == 5 and cfg.cluster.account == "acct-a"  # y2 wins, y1 kept

    monkeypatch.setenv("B20_COMPUTE__THREADS", "7")
    cfg = load_config([y1, y2], [])
    assert cfg.compute.threads == 7  # env beats YAML
    cfg = load_config([y1, y2], ["compute.threads=9"])
    assert cfg.compute.threads == 9  # --set beats env
    assert cfg.compute.dtype == "float64"  # untouched defaults survive every layer


def test_env_prefix_and_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B20_DFT__THRESHOLDS__F_MEV_A", "3.5")
    monkeypatch.setenv("B20_CLUSTER__MODULES", '["gcc", "quantum-espresso"]')
    monkeypatch.setenv("b20_train__replay__num_samples_pt", "5000")  # case-insensitive
    monkeypatch.setenv("B20X_COMPUTE__THREADS", "1")  # wrong prefix: ignored
    layer = env_layer()
    assert layer["dft"]["thresholds"]["F_meV_A"] == "3.5"
    cfg = load_config([], [])
    assert cfg.dft.thresholds.F_meV_A == 3.5
    assert cfg.cluster.modules == ["gcc", "quantum-espresso"]
    assert cfg.train.replay.num_samples_pt == 5000
    assert cfg.compute.threads == 6
    assert Settings().train.replay.num_samples_pt == 5000  # plain Settings() reads env too


def test_unknown_keys_rejected_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        load_config([], ["compute.bogus=1"])
    with pytest.raises(ValidationError):
        load_config([], ["nosuchsection.x=1"])
    monkeypatch.setenv("B20_COMPUTE__BOGUS", "1")
    with pytest.raises(ValidationError):
        load_config([], [])


def test_parse_overrides() -> None:
    parsed = parse_overrides(
        ["a.b=1", "a.c=true", "d=[x, y]", "e=", "f='1'", "g=null", "h=1.5e-3", "a.b.z=2"]
    )
    assert parsed == {
        "a": {"b": {"z": 2}, "c": True},
        "d": ["x", "y"],
        "e": "",
        "f": "1",
        "g": None,
        "h": 1.5e-3,
    }
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["novalue"])
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["=1"])


def test_deep_merge_replaces_lists_and_scalars() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    out = deep_merge(base, {"a": {"y": [3]}, "b": {"nested": True}})
    assert out == {"a": {"x": 1, "y": [3]}, "b": {"nested": True}}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}  # inputs untouched


def test_sha256_is_stable_and_sensitive() -> None:
    a = load_config([], [])
    b = load_config([], [])
    assert a.sha256() == b.sha256() and len(a.sha256()) == 64
    assert Settings.model_validate({"md": {"taut": 100.0}}).sha256() == a.sha256()
    assert load_config([], ["compute.threads=1"]).sha256() != a.sha256()
    # key order of the input never matters
    x = Settings.model_validate({"compute": {"threads": 2, "device": "cpu"}})
    y = Settings.model_validate({"compute": {"device": "cpu", "threads": 2}})
    assert x.sha256() == y.sha256()


def test_yaml_round_trip_and_missing_default(tmp_path: Path) -> None:
    cfg = load_config([], ["cluster.scratch=/gscratch/x"])
    again = Settings.model_validate(yaml.safe_load(cfg.to_yaml()))
    assert again.sha256() == cfg.sha256()
    # a missing default.yaml falls back to the model defaults
    fallback = load_config([], [], default_path=tmp_path / "absent.yaml", use_env=False)
    assert fallback.sha256() == Settings.model_validate({}).sha256()
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        read_yaml(bad)
