import asyncio
import json

import pytest

from src.dataset.caption_client import CaptionClient, ModelPricing
from src.posttrain.joint import LambdaController, PromotionGate, TwoTimescale
from src.posttrain.rewards import (
    JUDGE_PROMPT_VERSION,
    AsyncRewardPipeline,
    LocalScorer,
    RewardEnsemble,
    ScoredSample,
    VLMJudge,
    group_normalize,
    parse_judge_scores,
)


def test_group_normalize_centers_and_clips():
    out = group_normalize([1.0, 2.0, 3.0], z=0.5)
    assert out[0] == 0.0 and out[2] == 1.0 and out[1] == 0.5
    # Constant group carries no signal.
    assert group_normalize([2.0, 2.0], z=1.0) == [0.5, 0.5]
    # Within-Z differences map linearly (group mean is 0.25 here).
    out = group_normalize([0.0, 0.5], z=2.0)
    assert abs(out[1] - 0.5625) < 1e-9
    with pytest.raises(ValueError):
        group_normalize([1.0], z=0.0)
    assert group_normalize([], z=1.0) == []


def test_ensemble_combine_weights_and_held_out():
    ens = RewardEnsemble(weights={"aesthetic": 1.0, "judge": 3.0, "hps": 2.0},
                         held_out=["hps"])
    score = ens.combine({"aesthetic": 0.4, "judge": 0.8, "hps": 0.0})
    assert abs(score - (0.4 + 3 * 0.8) / 4) < 1e-9
    assert ens.optimized_names() == ["aesthetic", "judge"]
    with pytest.raises(KeyError):
        ens.combine({"aesthetic": 0.5})
    with pytest.raises(ValueError):
        RewardEnsemble(weights={"hps": 1.0}, held_out=["hps"]).combine({"hps": 1.0})


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


def test_vlm_judge_reads_from_client_cache(tmp_path):
    client = CaptionClient(cache_dir=tmp_path, model="deepseek-v4-pro", pricing=ModelPricing(prompt=1.0, completion=2.0))
    judge = VLMJudge(client)
    image_bytes = b"fake-jpeg-bytes"
    prompt_text = judge.judge_prompt.format(prompt="a cat")
    settings = {"max_tokens": judge.max_tokens, "temperature": judge.temperature}
    key = client.request_key(image_bytes, prompt_text,
                             JUDGE_PROMPT_VERSION, settings)
    record = {
        "request_key": key, "image_fingerprint": "x", "model": client.model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "text": '{"anatomy": 8, "adherence": 8, "naturalness": 8, "aesthetics": 8}',
        "usage": {}, "cost_usd": 0.0, "pricing": {}, "settings": settings,
        "latency_s": 0.0, "error": None, "transient": False,
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(record))

    async def run():
        import httpx
        async with httpx.AsyncClient() as http:
            return await judge.score(http, image_bytes=image_bytes,
                                     prompt="a cat")

    assert asyncio.run(run()) == 0.8


def test_vlm_judge_unparseable_returns_none(tmp_path):
    client = CaptionClient(cache_dir=tmp_path, model="deepseek-v4-pro", pricing=ModelPricing(prompt=1.0, completion=2.0))
    judge = VLMJudge(client)
    image_bytes = b"other-bytes"
    prompt_text = judge.judge_prompt.format(prompt="a dog")
    settings = {"max_tokens": judge.max_tokens, "temperature": judge.temperature}
    key = client.request_key(image_bytes, prompt_text,
                             JUDGE_PROMPT_VERSION, settings)
    record = {
        "request_key": key, "image_fingerprint": "x", "model": client.model,
        "prompt_version": JUDGE_PROMPT_VERSION, "text": "I cannot score this.",
        "usage": {}, "cost_usd": 0.0, "pricing": {}, "settings": settings,
        "latency_s": 0.0, "error": None, "transient": False,
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(record))

    async def run():
        import httpx
        async with httpx.AsyncClient() as http:
            return await judge.score(http, image_bytes=image_bytes,
                                     prompt="a dog")

    assert asyncio.run(run()) is None


def test_two_timescale_ratio():
    ts = TwoTimescale(fake_per_generator=5)
    assert ts.generator_updates(10) == 2
    assert ts.generator_updates(9) == 1
    with pytest.raises(ValueError):
        TwoTimescale(fake_per_generator=0)


def test_promotion_gate_requires_a_streak():
    gate = PromotionGate(coherence_threshold=0.8, patience=3)
    assert not gate.observe(0.9)
    assert not gate.observe(0.85)
    gate.observe(0.1)  # resets the streak
    assert not gate.observe(0.95)
    assert not gate.observe(0.95)
    assert gate.observe(0.95)
    assert gate.promoted
    with pytest.raises(ValueError):
        gate.observe(1.5)


def test_promotion_is_a_one_time_latch():
    gate = PromotionGate(coherence_threshold=0.8, patience=2)
    assert not gate.observe(0.9)
    assert gate.observe(0.9)  # the transition itself
    # Later probes never re-emit the transition, and a failing probe must
    # not revoke promotion: the cold-start -> joint switch is one-way.
    assert not gate.observe(0.9)
    assert not gate.observe(0.0)
    assert gate.promoted


def test_lambda_controller_lowers_on_plateau_plus_kid_regression():
    ctl = LambdaController(value=1.0, factor=0.5, window=3,
                           plateau_tol=0.01, kid_regress_tol=0.05)
    # Improving reward: no change.
    for i in range(5):
        assert ctl.observe(0.5 + 0.1 * i, 1.0) == 1.0
    # Plateau must fill the whole window before it can trigger; the improving
    # tail keeps the gate closed while it is still inside the window.
    assert ctl.observe(1.0, 1.0) == 1.0
    assert ctl.observe(1.005, 1.0) == 1.0
    assert ctl.observe(1.002, 1.0) == 1.0
    # Now the last window+1 rewards span 0.005 <= tol and KID regressed.
    assert ctl.observe(1.003, 1.2) == 0.5
    # Floor respected.
    ctl2 = LambdaController(value=0.06, factor=0.5, floor=0.05, window=2,
                            plateau_tol=0.01, kid_regress_tol=0.0)
    ctl2.observe(1.0, 1.0)
    ctl2.observe(1.0, 1.0)
    assert ctl2.observe(1.0, 1.1) == 0.05


def test_lambda_controller_never_raises_automatically():
    ctl = LambdaController(value=0.5)
    for i in range(10):
        assert ctl.observe(0.1 * i, 1.0) == 0.5


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


def test_pipeline_rejects_scorer_without_weight():
    pipe_ok = AsyncRewardPipeline(
        ensemble=RewardEnsemble(weights={"aesthetic": 1.0}),
        local_scorers=[LocalScorer("aesthetic", lambda img, prompt: 1.0)])
    assert pipe_ok is not None
    with pytest.raises(ValueError):
        AsyncRewardPipeline(
            ensemble=RewardEnsemble(weights={"aesthetic": 1.0}),
            local_scorers=[LocalScorer("hps", lambda img, prompt: 1.0)])


def test_pipeline_requires_image_bytes():
    pipe = AsyncRewardPipeline(
        ensemble=RewardEnsemble(weights={"aesthetic": 1.0}),
        local_scorers=[LocalScorer("aesthetic", lambda img, prompt: 1.0)])
    sample = ScoredSample(prompt="p", group_id="g1", image_ref="x",
                          raw_scores={}, combined=0.0)
    with pytest.raises(ValueError, match="image_bytes"):
        _run_batch(pipe, [sample])
