"""What the frozen caption table keeps, measures, and drops.

The table is the last point where caption production can change: whatever it
writes is what the training data gets.  Retries must collapse to one caption
per image, a caption that repeats itself must not reach training, and a row the
provider refused must stay visible instead of silently vanishing.
"""

import json

from scripts.caption.consolidate import load
from scripts.caption.freeze_captions import freeze
from src.utils.prompt_contract import DROP_IDX, PROMPT_TEMPLATE, SYSTEM_PROMPT


class CharTokenizer:
    """One token per character: makes the expected lengths exact and offline."""

    def __call__(self, text, truncation=True, max_length=None, padding=False):
        ids = list(range(len(text)))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def record(image_id, text, **extra):
    row = {"image_id": image_id, "source": "d3_pexels", "model": "deepseek:deepseek-flash",
           "language": "en", "format": "prose", "band": "256-511", "domain": "photograph",
           "prompt_version": "cap-v6-en-prose", "text": text, "usage": {"total": 1},
           "request_key": "abc", "image_fingerprint": "def", "cost_usd": 0.001}
    row.update(extra)
    return row


def write(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")


def test_retries_collapse_to_one_caption_per_image(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(first, [record("img-1", "", error="timeout"),
                  record("img-2", "a caption about a street")])
    write(second, [record("img-1", "a caption about a harbour")])

    frozen, order = freeze(load([str(first), str(second)]), CharTokenizer())

    by_id = {row["image_id"]: row for row in frozen}
    assert order == ["img-1", "img-2"]
    assert by_id["img-1"]["text"] == "a caption about a harbour"
    assert by_id["img-1"]["accepted"] is True
    assert by_id["img-2"]["accepted"] is True


def test_keeps_the_text_and_drops_a_provider_refusal_and_repeated_text():
    repeated = "The harbour is quiet today. " * 2
    frozen, _ = freeze(
        {("img-1", "m"): record("img-1", "a short caption"),
         ("img-2", "m"): record("img-2", repeated),
         ("img-3", "m"): record("img-3", "", error="content_filter")},
        CharTokenizer(),
    )
    by_id = {row["image_id"]: row for row in frozen}
    assert by_id["img-1"]["accepted"] is True
    assert by_id["img-2"]["accepted"] is False
    assert by_id["img-2"]["reject_reasons"] == ["repeated sentences"]
    assert by_id["img-3"]["accepted"] is False
    assert by_id["img-3"]["frozen_reason"] == "no text"


def test_measures_retained_tokens_of_the_whole_prompt():
    text = "x" * 250
    frozen, _ = freeze({("img-1", "m"): record("img-1", text)}, CharTokenizer())
    prompt = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=text)
    assert frozen[0]["retained_tokens"] == len(prompt) - DROP_IDX
