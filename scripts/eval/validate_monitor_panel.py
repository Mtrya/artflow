#!/usr/bin/env python3
"""Verify the frozen panel's retained lengths with the training tokenizer.

Run from the repository root, for example:
python -m scripts.eval.validate_monitor_panel --tokenizer /path/to/Qwen3-0.6B
This is a CPU-only, offline check; it does not download models or generate images.
"""

import argparse
from collections import Counter

from transformers import AutoTokenizer

from src.dataset.length_metadata import _prompt_text, _tokenize_prompt_lengths
from src.evaluation.prompt_grid import load_prompt_suite


def validate(path, tokenizer_path):
    prompts = load_prompt_suite(path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    lengths = _tokenize_prompt_lengths(
        tokenizer, [_prompt_text(p["text"]) for p in prompts]
    )
    for prompt, length in zip(prompts, lengths):
        if length != prompt["retained_tokens"]:
            raise ValueError(f"{prompt['id']}: recorded {prompt['retained_tokens']}, actual {length}")
        lo, hi = (1, 255) if prompt["variant"] == "short" else map(
            int, prompt["length_band"].split("-")
        )
        if not lo <= length <= hi:
            raise ValueError(f"{prompt['id']}: {length} outside [{lo}, {hi}]")
    print(f"Verified {len(prompts)} prompts; retained lengths {min(lengths)}–{max(lengths)}")
    print(dict(Counter(p["length_band"] for p in prompts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="assets/eval/hero_monitor_v1.jsonl")
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    validate(args.panel, args.tokenizer)
