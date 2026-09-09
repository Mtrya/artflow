"""Equivalence tests for the throughput fast paths.

Every fast path must produce the reference result, not merely a close one:
these paths are enabled in training runs that are compared against a baseline
on eval/loss trajectories, so a fast path that silently changes the
conditioning features or the attention math would invalidate the comparison.
"""

import torch

from src.models.artflow import ArtFlow
from src.utils.encode_text import (
    DROP_IDX,
    MAX_SEQUENCE_LENGTH,
    RETAINED_MIN_LENGTH,
    _extract_masked_hidden,
    _retained_slice,
    _trim_sequence,
    encode_text,
)


def _reference_retained(hidden, mask):
    """The original gather/trim/pad repack, kept here as the oracle."""
    sequences = _extract_masked_hidden(hidden, mask)
    trimmed = [_trim_sequence(seq) for seq in sequences]
    max_seq_len = max((seq.size(0) for seq in trimmed), default=0) or 1
    embeddings, masks = [], []
    for seq in trimmed:
        pad_len = max_seq_len - seq.size(0)
        if pad_len > 0:
            seq = torch.cat([seq, seq.new_zeros((pad_len, seq.size(1)))], dim=0)
        embeddings.append(seq)
        row_mask = seq.new_zeros(max_seq_len, dtype=torch.long)
        row_mask[: seq.size(0) - pad_len] = 1
        masks.append(row_mask)
    return torch.stack(embeddings), torch.stack(masks)


def _random_right_padded_hidden(batch, seq_len, dim, seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq_len, dim, generator=generator)
    lengths = torch.randint(0, seq_len + 1, (batch,), generator=generator)
    lengths[0] = seq_len
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for row, length in enumerate(lengths.tolist()):
        mask[row, :length] = 1
    return hidden, mask, lengths


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


def test_retained_slice_matches_reference_with_full_rows():
    hidden, mask, _ = _random_right_padded_hidden(4, DROP_IDX + 32, 8, seed=1)
    fast_emb, fast_mask = _retained_slice(hidden, mask)
    ref_emb, ref_mask = _reference_retained(hidden, mask)

    _assert_same_retained(fast_emb, fast_mask, ref_emb, ref_mask)


def test_retained_slice_handles_empty_and_short_rows():
    """Rows at or below DROP_IDX keep the reference single zero token."""
    seq_len = DROP_IDX + 16
    hidden, mask, lengths = _random_right_padded_hidden(6, seq_len, 8, seed=2)
    mask[1, :] = 0  # fully empty row (dropped caption)
    mask[2, : DROP_IDX] = 1  # exactly DROP_IDX valid tokens
    mask[3, : DROP_IDX + 1] = 1  # a single retained token
    mask[0, :seq_len] = 1

    fast_emb, fast_mask = _retained_slice(hidden, mask)
    ref_emb, ref_mask = _reference_retained(hidden, mask)

    _assert_same_retained(fast_emb, fast_mask, ref_emb, ref_mask)
    assert fast_mask[1].sum() == 1
    assert not fast_emb[1].any()


def test_retained_slice_handles_all_empty_batch():
    hidden = torch.randn(2, DROP_IDX, 8)
    mask = torch.zeros(2, DROP_IDX, dtype=torch.long)
    fast_emb, fast_mask = _retained_slice(hidden, mask)
    assert fast_emb.shape == (2, RETAINED_MIN_LENGTH, 8)
    assert torch.equal(fast_mask, torch.ones(2, RETAINED_MIN_LENGTH, dtype=torch.long))
    assert not fast_emb.any()


def test_retained_slice_caps_at_max_sequence_length():
    seq_len = DROP_IDX + MAX_SEQUENCE_LENGTH + 40
    hidden, mask, _ = _random_right_padded_hidden(3, seq_len, 8, seed=3)
    fast_emb, fast_mask = _retained_slice(hidden, mask)
    assert fast_emb.shape[1] == MAX_SEQUENCE_LENGTH
    assert fast_mask.shape[1] == MAX_SEQUENCE_LENGTH


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


class _StubOutput:
    def __init__(self, hidden):
        self.hidden_states = [hidden, hidden]
        self.last_hidden_state = hidden


class _StubInner:
    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, input_ids, attention_mask, use_cache=False,
                 output_hidden_states=False):
        return _StubOutput(self.hidden)


class _StubModel:
    def __init__(self, hidden):
        self.hidden = hidden
        self.model = _StubInner(hidden)
        self.device = torch.device("cpu")
        self.dtype = hidden.dtype


def test_fast_encode_text_matches_reference():
    seq_len = DROP_IDX + 12
    hidden, mask, lengths = _random_right_padded_hidden(5, seq_len, 6, seed=4)
    hidden = hidden.to(torch.float32)
    model = _StubModel(hidden)
    tokenizer = _StubTokenizer(lengths.tolist())

    ref_emb, ref_mask, ref_pooled = encode_text(
        ["a"] * 5, model, tokenizer, pooling=True, exit_layer=None
    )
    fast_emb, fast_mask, fast_pooled = encode_text(
        ["a"] * 5, model, tokenizer, pooling=True, exit_layer=None, fast_slice=True
    )

    _assert_same_retained(fast_emb, fast_mask, ref_emb, ref_mask)
    assert torch.equal(fast_pooled, ref_pooled)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    model = ArtFlow(
        hidden_size=64,
        num_heads=4,
        double_stream_depth=0,
        single_stream_depth=3,
        mlp_ratio=2.0,
        conditioning_scheme="fused",
        qkv_bias=True,
        single_stream_modulation="layer",
        ffn_type="gated",
        rope_centered_grid=True,
    )
    model.eval()
    return model


def test_fast_attn_matches_reference_forward():
    model = _tiny_model()
    x = torch.randn(2, 16, 8, 8)
    t = torch.rand(2)
    txt = torch.randn(2, 20, 1024)
    txt_pooled = torch.randn(2, 1024)
    txt_mask = torch.ones(2, 20, dtype=torch.long)
    txt_mask[1, 11:] = 0

    with torch.no_grad():
        ref = model(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask)
        fast = model(
            x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask, fast_attn=True
        )

    assert torch.equal(ref, fast)


def test_fast_attn_matches_reference_without_text_mask():
    model = _tiny_model(seed=1)
    x = torch.randn(2, 16, 8, 8)
    t = torch.rand(2)
    txt = torch.randn(2, 20, 1024)
    txt_pooled = torch.randn(2, 1024)

    with torch.no_grad():
        ref = model(x, t, txt=txt, txt_pooled=txt_pooled)
        fast = model(x, t, txt=txt, txt_pooled=txt_pooled, fast_attn=True)

    assert torch.equal(ref, fast)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    model = ArtFlow(
        hidden_size=64,
        num_heads=4,
        double_stream_depth=0,
        single_stream_depth=3,
        mlp_ratio=2.0,
        conditioning_scheme="fused",
        qkv_bias=True,
        single_stream_modulation="layer",
        ffn_type="gated",
        rope_centered_grid=True,
    )
    model.eval()
    return model


def test_fast_attn_matches_reference_forward():
    model = _tiny_model()
    x = torch.randn(2, 16, 8, 8)
    t = torch.rand(2)
    txt = torch.randn(2, 20, 1024)
    txt_pooled = torch.randn(2, 1024)
    txt_mask = torch.ones(2, 20, dtype=torch.long)
    txt_mask[1, 11:] = 0

    with torch.no_grad():
        ref = model(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask)
        fast = model(
            x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask, fast_attn=True
        )

    assert torch.equal(ref, fast)


def test_fast_attn_matches_reference_without_text_mask():
    model = _tiny_model(seed=1)
    x = torch.randn(2, 16, 8, 8)
    t = torch.rand(2)
    txt = torch.randn(2, 20, 1024)
    txt_pooled = torch.randn(2, 1024)

    with torch.no_grad():
        ref = model(x, t, txt=txt, txt_pooled=txt_pooled)
        fast = model(x, t, txt=txt, txt_pooled=txt_pooled, fast_attn=True)

    assert torch.equal(ref, fast)
