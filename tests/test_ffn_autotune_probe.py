import pytest
import torch

from scripts.bench.ffn_autotune_probe import build_ffn, parse_args
from src.train.config import load_config


def test_probe_matches_hero_ffn_dimensions_and_parameter_precision():
    model = build_ffn(load_config(["configs/base.toml", "configs/hero.toml"]).model)
    assert model.up_proj.weight.shape == (6150, 1152)
    assert model.down_proj.weight.shape == (1152, 3075)
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_probe_defaults_to_recorded_token_counts_and_rejects_empty_work():
    args = ["--config", "configs/base.toml", "--out", "/tmp/ffn-probe-test.json"]
    assert parse_args(args).tokens == [26128, 21816]
    with pytest.raises(SystemExit):
        parse_args([*args, "--iterations", "0"])
