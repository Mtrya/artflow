"""Tests for the fixed eval-loss probe (determinism, shape grouping)."""

import torch
import pytest
from datasets import Dataset

from src.evaluation import eval_loss as eval_loss_module
from src.evaluation.eval_loss import EvalLossProbe
from src.models.artflow import ArtFlow


def _write_tiny_dataset(tmp_path):
    gen = torch.Generator().manual_seed(0)
    rows = {
        "latents": [],
        "captions": [],
        "resolution_bucket_id": [],
    }
    # Two shape groups: 32x32 and 32x16 latents (16ch)
    for i in range(8):
        h, w = (32, 32) if i % 2 == 0 else (32, 16)
        rows["latents"].append(torch.randn((16, h, w), generator=gen).numpy())
        rows["captions"].append([f"short caption {i}", f"a longer caption {i}"])
        rows["resolution_bucket_id"].append(1 if i % 2 == 0 else 2)
    ds = Dataset.from_dict(rows)
    path = tmp_path / "tiny_eval"
    ds.save_to_disk(str(path))
    return str(path)


def _fake_encode_text(texts, model, tokenizer, pooling, exit_layer=None):
    b = len(texts)
    emb = torch.zeros(b, 5, 1024)
    mask = torch.ones(b, 5, dtype=torch.long)
    pooled = torch.zeros(b, 1024) if pooling else None
    return emb, mask, pooled


@pytest.fixture()
def probe_env(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_loss_module, "encode_text", _fake_encode_text)
    path = _write_tiny_dataset(tmp_path)

    class _FakeEncoder:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    model = ArtFlow(
        hidden_size=64,
        num_heads=4,
        double_stream_depth=0,
        single_stream_depth=2,
        conditioning_scheme="pure",
        qkv_bias=False,
        ffn_type="gated",
    )
    vae_mean = torch.zeros(16, 1, 1)
    vae_std = torch.ones(16, 1, 1)
    kwargs = dict(
        dataset_path=path,
        text_encoder=_FakeEncoder(),
        tokenizer=None,
        pooling=False,
        exit_layer=None,
        vae_mean=vae_mean,
        vae_std=vae_std,
        num_samples=8,
        batch_size=4,
        device=torch.device("cpu"),
    )
    return model, kwargs


def test_probe_is_deterministic(probe_env):
    model, kwargs = probe_env
    probe_a = EvalLossProbe(**kwargs)
    probe_b = EvalLossProbe(**kwargs)
    ma = probe_a.evaluate(model)
    mb = probe_b.evaluate(model)
    assert ma.keys() == mb.keys()
    for k in ma:
        assert ma[k] == pytest.approx(mb[k], rel=0, abs=1e-7)
    # repeated evaluation on the same probe is also stable
    ma2 = probe_a.evaluate(model)
    assert ma["eval/loss"] == pytest.approx(ma2["eval/loss"], rel=0, abs=1e-7)


def test_probe_grouping_and_metrics_keys(probe_env):
    model, kwargs = probe_env
    probe = EvalLossProbe(**kwargs)
    assert len(probe.groups) == 2  # two latent shapes were grouped separately
    metrics = probe.evaluate(model)
    assert "eval/loss" in metrics
    assert set(k for k in metrics if k.startswith("eval/loss_t")) == {
        f"eval/loss_t{tag}" for tag in ["015", "040", "065", "090"]
    }
    assert all(v == v and v >= 0 for v in metrics.values())  # no NaN


def test_probe_restores_training_mode(probe_env):
    model, kwargs = probe_env
    model.train()
    probe = EvalLossProbe(**kwargs)
    probe.evaluate(model)
    assert model.training
