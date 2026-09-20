import pytest
import torch

from scripts.bench.attention_backend_probe import make_inputs, evaluate, errors
from torch.nn.attention import SDPBackend


def test_probe_matches_observed_layout_and_preserves_image_keys():
    inputs, bias, upstream = make_inputs(3, 17, 12, device="cpu", dtype=torch.float32)
    q, k, v = inputs
    assert q.stride() == k.stride() == (16 * 17 * 72, 17 * 72, 72, 1)
    assert v.stride() == (17 * 3 * 16 * 72, 72, 3 * 16 * 72, 1)
    assert upstream.stride() == (17 * 16 * 72, 72, 16 * 72, 1)
    assert bias.stride(0) == 24
    assert (bias[..., :12] == 0).all()
    assert torch.isneginf(bias[0, ..., 12:]).all()
    assert (bias[-1] == 0).all()
    result = evaluate(inputs, bias, upstream, SDPBackend.MATH)
    assert all(torch.isfinite(t).all() for t in result)
    assert all(errors(t, t)["relative_l2"] == 0 for t in result)
    # Masked keys must receive no K/V gradients, including text-dropped rows.
    assert (result[2][0, :, 12:] == 0).all()
    assert (result[3][0, :, 12:] == 0).all()


def test_probe_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        make_inputs(1, 12, 12, device="cpu")
