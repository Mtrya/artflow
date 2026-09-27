"""Reward processing checked against fixed scorer results and cached judge responses."""

import asyncio
import json


from src.dataset.caption_client import CaptionClient, ModelPricing
from src.posttrain.rewards import JUDGE_PROMPT_VERSION, AsyncRewardPipeline, LocalScorer, RewardEnsemble, ScoredSample, VLMJudge, parse_judge_scores


def test_parse_judge_scores_variants():
    text = '{"anatomy": 7, "adherence": 8, "naturalness": 6, "aesthetics": 9}'
    axes = parse_judge_scores(text)
    assert axes == {"anatomy": 7.0, "adherence": 8.0,
                    "naturalness": 6.0, "aesthetics": 9.0}
    # Clamped to [0, 10]; extra prose tolerated.
    text = 'Here: {"anatomy": 12, "adherence": 5, "naturalness": 5, "aesthetics": 5}.'
    assert parse_judge_scores(text)["anatomy"] == 10.0
    assert parse_judge_scores("no scores here") is None
    assert parse_judge_scores('{"anatomy": 5}') is None


def _sample(group_id: str, image: bytes, prompt: str = "p") -> ScoredSample:
    return ScoredSample(prompt=prompt, group_id=group_id, image_ref=image.decode(),
                        raw_scores={}, combined=0.0, image_bytes=image)


def _run_batch(pipe, samples):
    import httpx

    async def run():
        async with httpx.AsyncClient() as http:
            return await pipe.score_batch(http, samples)

    return asyncio.run(run())


def test_pipeline_scores_and_normalizes_groups():
    aesthetic = LocalScorer("aesthetic", lambda img, prompt: 1.0)
    hps = LocalScorer("hps", lambda img, prompt: 1.0 if img == b"a" else 0.0)
    pipe = AsyncRewardPipeline(
        ensemble=RewardEnsemble(weights={"aesthetic": 1.0, "hps": 1.0}),
        local_scorers=[aesthetic, hps])
    samples = [_sample("g1", b"a"), _sample("g1", b"b"),
               _sample("g2", b"b"), _sample("g2", b"b")]
    kept, stats = _run_batch(pipe, samples)

    assert len(kept) == 4 and stats["dropped"] == 0 and stats["groups"] == 2
    # g1: combined 1.0 (img a) vs 0.5 (img b); first group uses its own std
    # (0.25) as Z -> the pair saturates to 1.0 / 0.0.
    assert kept[0].combined == 1.0 and kept[1].combined == 0.5
    assert kept[0].r == 1.0 and kept[1].r == 0.0
    # g2: constant group carries no signal; Z comes from g1's history.
    assert kept[2].r == 0.5 and kept[3].r == 0.5
    assert stats["z"] == 0.0  # last appended group std is g2's


def test_pipeline_drops_only_when_optimized_scorer_fails():
    def flaky(img, prompt):
        if img == b"bad":
            raise RuntimeError("boom")
        return 0.5

    optimized = LocalScorer("aesthetic", flaky)
    held_out = LocalScorer("hps", lambda img, prompt: (_ for _ in ()).throw(
        RuntimeError("held-out down")))
    pipe = AsyncRewardPipeline(
        ensemble=RewardEnsemble(weights={"aesthetic": 1.0, "hps": 2.0},
                                held_out=["hps"]),
        local_scorers=[optimized, held_out])
    samples = [_sample("g1", b"ok"), _sample("g1", b"bad")]
    kept, stats = _run_batch(pipe, samples)

    # The held-out failure never drops; the optimized failure drops exactly
    # the bad sample.
    assert [s.image_ref for s in kept] == ["ok"]
    assert stats["dropped"] == 1
    assert kept[0].raw_scores == {"aesthetic": 0.5}
    assert kept[0].combined == 0.5 and kept[0].r == 0.5
    assert len(stats["scorer_errors"]) == 3  # held-out x2 + optimized x1


def test_pipeline_with_cached_judge_end_to_end(tmp_path):
    client = CaptionClient(cache_dir=tmp_path, model="deepseek-v4-pro",
                           pricing=ModelPricing(prompt=1.0, completion=2.0))
    judge = VLMJudge(client)
    prompt_text = judge.judge_prompt.format(prompt="a cat")
    settings = {"max_tokens": judge.max_tokens, "temperature": judge.temperature}
    key = client.request_key(b"img", prompt_text, JUDGE_PROMPT_VERSION, settings)
    record = {
        "request_key": key, "image_fingerprint": "x", "model": client.model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "text": '{"anatomy": 8, "adherence": 8, "naturalness": 8, "aesthetics": 8}',
        "usage": {}, "cost_usd": 0.0, "pricing": {}, "settings": settings,
        "latency_s": 0.0, "error": None, "transient": False,
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(record))

    pipe = AsyncRewardPipeline(
        ensemble=RewardEnsemble(weights={"judge": 1.0, "aesthetic": 1.0}),
        judge=judge,
        local_scorers=[LocalScorer("aesthetic", lambda img, prompt: 0.6)])
    kept, stats = _run_batch(pipe, [_sample("g1", b"img", prompt="a cat")])
    assert len(kept) == 1
    assert kept[0].raw_scores == {"judge": 0.8, "aesthetic": 0.6}
    assert kept[0].combined == 0.7 and kept[0].r == 0.5
    # Single-member group: own std 0 -> Z clamped at z_min.
    assert stats["z"] == 0.0
