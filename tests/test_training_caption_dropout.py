"""Classifier-free exposure must not depend on memory-driven bucket batches."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _select_captions(probability):
    # Execute the trainer's actual nested function without initializing DDP,
    # loading datasets, or maintaining a second implementation in this test.
    source = Path(__file__).resolve().parents[1] / "src/pretrain/train.py"
    function = next(node for node in ast.walk(ast.parse(source.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name == "select_captions")
    namespace = {"torch": torch, "args": SimpleNamespace(caption_dropout_prob=probability)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["select_captions"]


def test_dropout_exposure_is_invariant_to_microbatch_partition():
    select = _select_captions(0.1)
    captions = [f"caption {i}" for i in range(256)]
    partitions = []
    for batch_size in (1, 2, 8, 32):
        torch.manual_seed(42)
        selected, dropped = [], []
        for start in range(0, len(captions), batch_size):
            texts, mask = select({"captions": captions[start:start + batch_size]})
            selected.extend(texts)
            dropped.extend(mask)
        partitions.append((selected, dropped))
    assert all(result == partitions[0] for result in partitions)
    selected, dropped = partitions[0]
    assert 0 < sum(dropped) < len(captions)
    assert [text == "" for text in selected] == dropped


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_dropout_allows_all_dropped_or_all_retained(batch_size, probability):
    captions = ["a caption"] * batch_size
    selected, dropped = _select_captions(probability)({"captions": captions})
    assert selected == ([""] * batch_size if probability else captions)
    assert dropped == [bool(probability)] * batch_size
    assert captions == ["a caption"] * batch_size


def test_all_dropped_batch_encodes_valid_unconditional_features():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from src.pretrain.train import encode_training_text

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-0.6B tokenizer not cached locally")
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8,
    )).eval()
    selected, dropped = _select_captions(1.0)({"captions": ["one", "two"]})
    text, mask, pooled = encode_training_text(
        selected, model, tokenizer, exit_layer=1, bucket_hi=16,
    )
    assert dropped == [True, True]
    assert mask.sum(dim=1).tolist() == [5, 5]  # retained chat suffix, not all masked
    assert torch.isfinite(text).all() and torch.isfinite(pooled).all()
    # Identical rows may use different vectorized reduction lanes in Qwen.
    torch.testing.assert_close(text[0], text[1], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(pooled[0], pooled[1], rtol=1e-5, atol=1e-7)
