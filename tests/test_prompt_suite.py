"""Prompt identity and noise pairing are explicit inputs to evaluation."""

import json

import pytest

from src.evaluation.prompt_grid import load_prompt_suite, resolved_prompt_seed


@pytest.mark.parametrize("seed", [None, True, -1, 2**63, "42"])
def test_prompt_seed_is_required_and_validated(seed):
    prompt = {"id": "example"}
    if seed is not None:
        prompt["seed"] = seed
    with pytest.raises(ValueError, match="prompt seed"):
        resolved_prompt_seed(prompt)


def test_renaming_prompt_does_not_change_its_explicit_seed():
    assert resolved_prompt_seed({"id": "first", "seed": 37}) == 37
    assert resolved_prompt_seed({"id": "renamed", "seed": 37}) == 37


def test_scene_variants_require_matching_explicit_seeds(tmp_path):
    rows = [
        {
            "id": f"{lang}_{variant}",
            "scene_id": "scene",
            "lang": lang,
            "variant": variant,
            "seed": 17,
            "prompt": "fixture",
            "aspect": "3:4",
        }
        for lang in ("zh", "en")
        for variant in ("short", "long")
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert len(load_prompt_suite(path)) == 4
    rows[-1]["seed"] = 18
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="must share a seed"):
        load_prompt_suite(path)
