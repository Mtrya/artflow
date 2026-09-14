"""Tests for the TOML experiment configuration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.config import flatten, load_config

BASE = str(Path(__file__).parent.parent / "configs" / "base.toml")


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_base_config_loads_and_flattens():
    """The shipped recipe flattens to the names the training loop reads."""
    flat = flatten(load_config([BASE]))
    assert flat["hidden_size"] == 1152
    assert flat["single_stream_depth"] == 24
    assert flat["double_stream_depth"] == 0
    assert flat["muon_lr"] == 0.02
    assert flat["ema_decay"] == 0.999
    # Renamed keys: the config groups them for readability, the loop reads flat.
    assert flat["dataset_mix"] == ""
    assert flat["text_encoder_path"].endswith("Qwen3-0.6B")
    assert flat["text_encoder_exit_layer"] == 20
    assert flat["eval_loss_interval"] == 50
    assert flat["steady_state_skip_steps"] == 50


def test_later_config_overrides_earlier(tmp_path):
    base = _write(tmp_path / "base.toml", "[train]\nmax_steps = 10\n")
    over = _write(tmp_path / "over.toml", "[train]\nmax_steps = 20\n")
    assert load_config([base, over]).train.max_steps == 20
    assert load_config([over, base]).train.max_steps == 10


def test_partial_config_keeps_other_defaults(tmp_path):
    partial = _write(tmp_path / "partial.toml", "[train]\nmax_steps = 10\n")
    config = load_config([partial])
    assert config.model.hidden_size == 1152
    assert config.optim.muon_lr == 0.02


def test_unknown_section_is_rejected(tmp_path):
    bad = _write(tmp_path / "bad.toml", "[nope]\nx = 1\n")
    with pytest.raises(ValueError, match="unknown config section"):
        load_config([bad])


def test_unknown_key_is_rejected(tmp_path):
    bad = _write(tmp_path / "bad.toml", "[train]\nnot_a_key = 1\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config([bad])


def test_out_of_range_value_is_rejected(tmp_path):
    bad = _write(tmp_path / "bad.toml", "[data]\ncurriculum_start = 2.0\n")
    with pytest.raises(ValueError, match="curriculum_start"):
        load_config([bad])


def test_boolean_value_type_is_enforced(tmp_path):
    bad = _write(tmp_path / "bad.toml", "[model]\nqkv_bias = 1\n")
    with pytest.raises(ValueError, match="qkv_bias"):
        load_config([bad])


def test_hidden_size_must_divide_by_heads(tmp_path):
    bad = _write(tmp_path / "bad.toml", "[model]\nhidden_size = 1153\n")
    with pytest.raises(ValueError, match="divisible"):
        load_config([bad])


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_config([str(tmp_path / "nope.toml")])


@pytest.mark.parametrize("setting", ["ode_steps = 0", "ode_steps = true", "grid_steps = [-1]",
                                   "grid_steps = [true]", "grid_steps = 42"])
def test_invalid_grid_config(tmp_path, setting):
    path = _write(tmp_path / "grid.toml", "[eval]\n" + setting + "\n")
    with pytest.raises(ValueError):
        load_config([path])


def test_grid_config_flattened(tmp_path):
    path = _write(tmp_path / "grid.toml", "[eval]\node_steps = 32\ngrid_steps = [300000, 302000, 380000]\n")
    flat = flatten(load_config([BASE, path]))
    assert flat["ode_steps"] == 32
    assert flat["grid_steps"] == [300000, 302000, 380000]


def test_caption_loss_weights_are_validated_at_load(tmp_path):
    """The shipped recipe weights nothing, and an unusable curve fails like
    any other bad config value instead of being repaired on the way in."""
    flat = flatten(load_config([BASE]))
    assert flat["caption_loss_weight_curve"] == "none"
    assert flat["caption_loss_weight_reference"] == 128

    partial = _write(
        tmp_path / "partial.toml",
        '[train]\ncaption_loss_weight_curve = "log2"\n'
        "caption_loss_weight_reference = 64\n",
    )
    train = load_config([partial]).train
    assert train.caption_loss_weight_curve == "log2"
    assert train.caption_loss_weight_reference == 64

    typo = _write(
        tmp_path / "typo.toml",
        '[train]\ncaption_loss_weight_curve = "linear"\n')
    with pytest.raises(ValueError, match="must be one of"):
        load_config([typo])

    small = _write(
        tmp_path / "small.toml",
        '[train]\ncaption_loss_weight_curve = "log2"\n'
        "caption_loss_weight_reference = 1\n",
    )
    with pytest.raises(ValueError, match="reference must be >= 2"):
        load_config([small])
