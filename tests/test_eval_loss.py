"""Evaluation reductions checked against known per-caption velocity errors."""

import numpy as np
import pytest
import torch
from datasets import Dataset

from src.dataset.length_metadata import RowLengthMetadata, sidecar_path
from src.evaluation import eval_loss


class KnownErrorVelocity(torch.nn.Module):
    def forward(self, z_t, t, txt=None, **kwargs):
        # The fixture's clean latent is zero: its exact velocity is -z_t/(1-t).
        # Each caption encodes a known additive prediction error.
        return -z_t / (1 - t[:, None, None, None]) + txt[:, :1, :1, None]


def encode_errors(captions, *args, **kwargs):
    errors = torch.tensor([float(caption) for caption in captions])[:, None, None]
    return errors, torch.ones(len(captions), 1, dtype=torch.long), None


@pytest.mark.parametrize('batch_size', [1, 4])
def test_loss_matches_analytic_sample_mean_across_bands_and_shapes(tmp_path, monkeypatch, batch_size):
    # Three short captions have squared error 4; four long ones have error 16.
    captions = [['2'], ['2', '4'], ['4'], ['2', '4'], ['4']]
    dataset_path = str(tmp_path / 'eval')
    Dataset.from_dict({
        'latents': [np.zeros((2, height, 4), dtype=np.float32) for height in (4, 4, 4, 2, 2)],
        'captions': captions,
        'resolution_bucket_id': [1, 1, 1, 2, 2],
    }).save_to_disk(dataset_path)
    RowLengthMetadata(
        resolution_ids=np.array([1, 1, 1, 2, 2]),
        caption_offsets=np.array([0, 1, 3, 4, 6, 7]),
        prompt_lengths=np.array([100, 100, 900, 900, 100, 900, 900]),
    ).save(sidecar_path(dataset_path))
    monkeypatch.setattr(eval_loss, 'encode_text', encode_errors)
    probe = eval_loss.EvalLossProbe(
        dataset_path=dataset_path, text_encoder=None, tokenizer=None,
        pooling=False, exit_layer=None, vae_mean=torch.zeros(2, 1, 1),
        vae_std=torch.ones(2, 1, 1), num_samples=8, batch_size=batch_size,
        device=torch.device('cpu'),
    )
    metrics = probe.evaluate(KnownErrorVelocity())
    assert metrics['eval/loss'] == pytest.approx(76 / 7, rel=1e-6)
    assert metrics['eval/loss/band_le128'] == pytest.approx(4, rel=1e-6)
    assert metrics['eval/loss/band_513_1024'] == pytest.approx(16, rel=1e-6)
    for tag in ('015', '040', '065', '090'):
        assert metrics[f'eval/loss_t{tag}'] == pytest.approx(76 / 7, rel=1e-6)
