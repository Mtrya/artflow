"""The offline caption-length pass and the training pass must agree.

A caption's retained length decides which training bucket its row lands in:
the sampler reads it from the dataset's length sidecar, while the trainer
recomputes the same token window when it encodes the caption.  A disagreement
never raises - the row is simply bucketed by a number the trainer never sees -
so the two paths are pinned together here.

The tokenizer is a character-indexing stub (one token per character), which
makes both paths offline and lets the expected window be derived by hand.
"""

import torch
import pytest

from src.dataset.length_metadata import _tokenize_prompt_lengths
from src.utils.encode_text import _build_prompt, _retained_slice
from src.utils.prompt_contract import DROP_IDX, MAX_SEQUENCE_LENGTH

CAP = MAX_SEQUENCE_LENGTH + DROP_IDX


class CharIndexTokenizer:
    """One token per character, honouring the call contract of both paths."""

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        cap = kwargs.get("max_length")
        rows = [list(range(len(text))) for text in texts]
        if cap is not None:
            rows = [row[:cap] for row in rows]
        if kwargs.get("return_tensors") == "pt":
            width = max(len(row) for row in rows) if kwargs.get("padding") else None
            if width is None:
                width = len(rows[0])
            ids, masks = [], []
            for row in rows:
                pad = width - len(row)
                ids.append(row + [0] * pad)
                masks.append([1] * len(row) + [0] * pad)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}


CAPTIONS = [
    "a short caption",
    "中文标题，描述图片内容。",
    "",
    "x",
    "s" * (CAP + 500),          # far past the sequence cap
    "s" * (CAP - len(_build_prompt(""))) if CAP > len(_build_prompt("")) else "s",
]


def training_retained_lengths(tokenizer, captions):
    """Retained window the trainer sees, via the real training helper."""
    prompts = [_build_prompt(caption) for caption in captions]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=CAP)
    hidden = torch.zeros(encoded["input_ids"].shape[0],
                         encoded["input_ids"].shape[1], 1, dtype=torch.float32)
    _, mask = _retained_slice(hidden, encoded["attention_mask"])
    return [int(row.sum()) for row in mask]


@pytest.mark.parametrize("group", [CAPTIONS[:3], CAPTIONS[3:]])
def test_offline_lengths_equal_training_lengths(group):
    tokenizer = CharIndexTokenizer()
    offline = list(_tokenize_prompt_lengths(tokenizer, [_build_prompt(c) for c in group]))
    assert offline == training_retained_lengths(tokenizer, group)


def test_lengths_never_exceed_the_retained_cap():
    tokenizer = CharIndexTokenizer()
    lengths = _tokenize_prompt_lengths(tokenizer, [_build_prompt(c) for c in CAPTIONS])
    assert max(lengths) == MAX_SEQUENCE_LENGTH
    assert min(lengths) >= 1
