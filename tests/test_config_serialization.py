"""Topology round-trip for `ArtFlow.get_config()` / state-dict inference.

The hero recipe is 1 double-stream block followed by 24 single-stream blocks,
so the double-stream path is the one that actually ships. Both serialisation
paths used to disagree with it:

  * `get_config()` read `attn.qkv` off the first block, but the double-stream
    attention splits QKV into `qkv_img`/`qkv_txt` -- AttributeError;
  * both `get_config()` and `_infer_config_from_state_dict()` detected
    double-stream blocks by looking for `txt_mlp`, a name that exists in
    neither the module (`mlp_img`/`mlp_txt`) nor the state dict
    (`mlp_txt`/`mlp`), so the depth was always reported as 0.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.artflow import ArtFlow

TINY = dict(
    hidden_size=64,
    num_heads=4,
    single_stream_depth=2,
    mlp_ratio=2.0,
    conditioning_scheme="fused",
    txt_in_features=32,
    patch_size=2,
    in_channels=4,
)


def _build(double_stream_depth, **kwargs):
    return ArtFlow(**TINY, double_stream_depth=double_stream_depth, **kwargs)


def test_get_config_reports_double_stream_depth():
    for depth in (0, 1, 2):
        config = _build(depth).get_config()
        assert config["double_stream_depth"] == depth, config
        assert config["single_stream_depth"] == 2, config


def test_get_config_reads_qkv_bias_from_the_double_stream_split_projection():
    assert _build(1).get_config()["qkv_bias"] is True
    assert _build(1, qkv_bias=False).get_config()["qkv_bias"] is False


def test_state_dict_inference_recovers_the_double_stream_depth():
    for depth in (0, 1, 2):
        inferred = ArtFlow._infer_config_from_state_dict(_build(depth).state_dict())
        assert inferred["double_stream_depth"] == depth, inferred
        assert inferred["single_stream_depth"] == 2, inferred


def test_config_rebuild_preserves_every_parameter_key():
    model = _build(1)
    config = model.get_config()
    rebuilt = ArtFlow(
        **{
            **TINY,
            "double_stream_depth": config["double_stream_depth"],
            "single_stream_depth": config["single_stream_depth"],
        }
    )
    assert set(rebuilt.state_dict()) == set(model.state_dict())
    rebuilt.load_state_dict(model.state_dict())
