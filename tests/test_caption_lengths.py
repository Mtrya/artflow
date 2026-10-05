"""Stored caption lengths cross-checked against training and Hugging Face readers."""

import json

import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.dataset.length_metadata import build_from_dataset, ensure_sidecar
from src.utils.encode_text import _build_prompt, _retained_slice
from src.utils.prompt_contract import DROP_IDX, MAX_SEQUENCE_LENGTH


@pytest.fixture
def tokenizer(tmp_path):
    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, 'word': 2}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path / 'tokenizer')
    return tokenizer, str(tmp_path / 'tokenizer')


def training_lengths(tokenizer, captions):
    encoded = tokenizer([_build_prompt(c) for c in captions], return_tensors='pt',
                        padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH + DROP_IDX)
    hidden = torch.zeros(*encoded.input_ids.shape, 1)
    _, mask = _retained_slice(hidden, encoded.attention_mask)
    return mask.sum(dim=1).numpy()


def test_saved_lengths_match_training_in_dataset_row_order(tmp_path, tokenizer):
    encoder, tokenizer_path = tokenizer
    path = str(tmp_path / 'dataset')
    Dataset.from_dict({
        'captions': [['short caption', '中文标题，描述图片内容。'], [''], ['word ' * 2500], ['word ' * 90]],
        'resolution_bucket_id': [1, 2, 1, 2],
    }).save_to_disk(path, num_shards=2)
    # A valid HF dataset may list shards in a different order from their names.
    state_path = tmp_path / 'dataset' / 'state.json'
    state = json.loads(state_path.read_text())
    state['_data_files'].reverse()
    state_path.write_text(json.dumps(state))
    actual = build_from_dataset(path, tokenizer_path)
    reference = load_from_disk(path)
    np.testing.assert_array_equal(actual.resolution_ids, reference['resolution_bucket_id'])
    for index, row in enumerate(reference):
        np.testing.assert_array_equal(actual.prompt_lengths[actual.row_slice(index)],
                                      training_lengths(encoder, row['captions']))


def test_cached_lengths_follow_replaced_dataset(tmp_path, tokenizer):
    encoder, tokenizer_path = tokenizer
    path = str(tmp_path / 'dataset')
    def save(caption):
        Dataset.from_dict({'captions': [[caption]], 'resolution_bucket_id': [1]}).save_to_disk(path)
    save('short')
    original = ensure_sidecar(path, tokenizer_path)
    save('word ' * 500)
    refreshed = ensure_sidecar(path, tokenizer_path)
    np.testing.assert_array_equal(refreshed.prompt_lengths, training_lengths(encoder, ['word ' * 500]))
    np.testing.assert_array_equal(ensure_sidecar(path, tokenizer_path).prompt_lengths, refreshed.prompt_lengths)
    assert not np.array_equal(original.prompt_lengths, refreshed.prompt_lengths)
