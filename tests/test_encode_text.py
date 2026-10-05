"""Training text features checked against a complete Transformers forward pass."""

import pytest
import torch

from src.utils.encode_text import DROP_IDX, encode_text


def _assert_same_retained(fast_emb, fast_mask, ref_emb, ref_mask):
    """Fast and reference agree on every position the DiT can attend to.

    The fast path keeps the padded hidden values in the columns between a row's
    retained length and the batch width; the reference zeroes them. Both are
    masked out of attention, so only the valid positions and the mask itself
    have to match.
    """
    width = ref_emb.shape[1]
    assert fast_emb.shape[1] >= width
    assert torch.equal(fast_mask[:, :width], ref_mask)
    assert not fast_mask[:, width:].any()
    assert fast_emb[fast_mask.bool()].shape == ref_emb[ref_mask.bool()].shape
    assert torch.equal(fast_emb[fast_mask.bool()], ref_emb[ref_mask.bool()])


class _StubInputs:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class _StubTokenizer:
    padding_side = "right"

    def __init__(self, lengths):
        self.lengths = list(lengths)

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=None):
        assert len(prompts) == len(self.lengths)
        width = max(self.lengths)
        input_ids = torch.zeros(len(prompts), width, dtype=torch.long)
        attention_mask = torch.zeros(len(prompts), width, dtype=torch.long)
        for row, length in enumerate(self.lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return _StubInputs(input_ids, attention_mask)


@pytest.mark.parametrize("exit_layer", [1, 2, 4])
def test_training_features_match_full_forward_hidden_state(exit_layer):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from src.pretrain.train import encode_training_text
    from src.dataset.sampler import pad_text_to_hi

    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8,
    )).eval()
    tokenizer = _StubTokenizer([DROP_IDX + 12, DROP_IDX + 3, DROP_IDX - 1])
    captions = ["long", "short", ""]
    ref, ref_mask, ref_pool = encode_text(
        captions, model, tokenizer, pooling=True, exit_layer=exit_layer,
        exit_mode="full_forward_slice",
    )
    ref, ref_mask = pad_text_to_hi(ref, ref_mask, 16)
    actual, mask, pooled = encode_training_text(
        captions, model, tokenizer, exit_layer=exit_layer, bucket_hi=16,
    )
    _assert_same_retained(actual, mask, ref, ref_mask)
    torch.testing.assert_close(pooled, ref_pool, rtol=0, atol=0)
    assert not actual.requires_grad
