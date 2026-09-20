import pytest
import torch
from dataclasses import asdict

from scripts.bench.ddp_compile_probe import make_inputs, parse_args, static_rope_buffers
from src.models.artflow import ArtFlow
from src.train.config import load_config


def test_probe_input_contract_uses_actual_hero_model(tmp_path):
    config = load_config(["configs/base.toml", "configs/hero.toml"])
    with torch.device("meta"):
        model = ArtFlow(**asdict(config.model))
    args = parse_args(["--config", "configs/base.toml", "--out", str(tmp_path),
                       "--micro", "2", "--latent", "4", "--text", "5", "7"])
    for length, (x, target, text, pooled, mask, t) in zip(args.text, make_inputs(model, args, "cpu")):
        assert x.shape == target.shape == (2, model.in_channels, 4, 4)
        assert text.shape == (2, length, model.txt_embedder.in_features)
        assert pooled.shape == (2, model.txt_pooled_proj.in_features)
        assert mask.shape == (2, length) and t.shape == (2,)
        assert not mask[0].any() and mask[1].all()


def test_probe_has_explicit_bounded_shapes_and_workaround(tmp_path):
    args = parse_args(["--config", "configs/base.toml", "--out", str(tmp_path),
                       "--disable-ddp-compile-split"])
    assert args.disable_ddp_compile_split
    assert args.steps == 12 and args.warmup == 4
    assert args.text == [128, 192]
    assert not args.gradient_bucket_views and not args.save_final_gradients
    assert not args.no_broadcast_buffers
    assert args.reference_gradients is None


def test_bucket_view_probe_can_require_matched_reference(tmp_path):
    args = parse_args(["--config", "configs/base.toml", "--out", str(tmp_path / "candidate"),
                       "--gradient-bucket-views", "--reference-gradients", str(tmp_path / "reference")])
    assert args.gradient_bucket_views
    assert args.reference_gradients == tmp_path / "reference"


@pytest.mark.parametrize("flag,value", [("--micro", "0"), ("--steps", "4"),
                                       ("--warmup", "1"), ("--accumulation", "0")])
def test_probe_rejects_unusable_windows(flag, value, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--config", "configs/base.toml", "--out", str(tmp_path), flag, value])


def test_trace_must_finish_within_window(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--config", "configs/base.toml", "--out", str(tmp_path),
                    "--steps", "5", "--trace"])


def test_static_rope_guard_covers_frozen_hero_and_rejects_expansion():
    config = load_config(["configs/base.toml", "configs/hero.toml"])
    with torch.device("meta"):
        model = ArtFlow(**asdict(config.model))
    buffers = static_rope_buffers(model, max_position=2048 + 64)
    assert len(buffers) == 25
    assert sum(v.numel() * v.element_size() for v in buffers.values()) == 18432000
    with pytest.raises(ValueError, match="expanding"):
        static_rope_buffers(model, max_position=2561)
    model.register_buffer("mutable_counter", torch.zeros(1))
    with pytest.raises(ValueError, match="only supports"):
        static_rope_buffers(model, max_position=2112)
