"""The college SII provider reuses CaptionClient end to end.

The endpoint is OpenAI-compatible, so no bespoke client is needed: a Provider
registry entry (key from the environment) plus the ``extra`` body passthrough
covers everything, including the private thinking toggle.
"""

import asyncio
import json

import httpx
import pytest

from src.dataset.caption_client import (
    CaptionClient,
    ModelPricing,
    model_id,
    provider_for,
)
from src.posttrain.rewards import VLMJudge

FREE = ModelPricing(prompt=0.0, completion=0.0)

RUBRIC_REPLY = ('{"anatomy": 7, "adherence": 8, '
                '"naturalness": 6, "aesthetics": 9}')


class StubHttp:
    """Records request bodies and plays back a canned chat completion."""

    def __init__(self, content: str = RUBRIC_REPLY):
        self.bodies = []
        self._content = content

    async def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append({"url": url, "json": json, "headers": headers})
        payload = {
            "choices": [{"message": {"content": self._content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=payload,
                              request=httpx.Request("POST", url))


def _sii_client(tmp_path, **kwargs):
    return CaptionClient(cache_dir=tmp_path, model="sii:Qwen3.8-27B",
                         pricing=FREE, **kwargs)


def test_sii_provider_selection_and_key_env(tmp_path, monkeypatch):
    provider = provider_for("sii:Qwen3.8-27B")
    assert provider.name == "sii"
    assert provider.chat_url.endswith("/v1/chat/completions")
    assert model_id("sii:Qwen3.8-27B") == "Qwen3.8-27B"
    # The credential comes from the environment, never from the repo.
    monkeypatch.delenv("SII_VLM_API_KEY", raising=False)
    client = _sii_client(tmp_path)
    with pytest.raises(Exception):
        asyncio.run(client.generate(
            StubHttp(), image_bytes=b"img", prompt="p",
            prompt_version="v", max_tokens=8))


def test_generate_passes_extra_body_and_caches_by_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SII_VLM_API_KEY", "test-key")
    client = _sii_client(tmp_path)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}

    async def run():
        http = StubHttp()
        first = await client.generate(
            http, image_bytes=b"img", prompt="score this",
            prompt_version="v1", max_tokens=64, temperature=0.0, extra=extra)
        # Same extra: served from cache, no second POST.
        second = await client.generate(
            http, image_bytes=b"img", prompt="score this",
            prompt_version="v1", max_tokens=64, temperature=0.0, extra=extra)
        return http, first, second

    http, first, second = asyncio.run(run())
    body = http.bodies[0]["json"]
    assert body["model"] == "Qwen3.8-27B"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert http.bodies[0]["headers"]["Authorization"] == "Bearer test-key"
    assert first.error is None and not first.cached
    assert second.cached and len(http.bodies) == 1
    # A free endpoint records zero cost.
    assert first.cost_usd == 0.0

    # A different toggle must not reuse the cached response.
    async def run_other():
        http2 = StubHttp()
        other = await client.generate(
            http2, image_bytes=b"img", prompt="score this",
            prompt_version="v1", max_tokens=64, temperature=0.0,
            extra={"chat_template_kwargs": {"enable_thinking": True}})
        return http2, other

    http2, other = asyncio.run(run_other())
    assert not other.cached and len(http2.bodies) == 1


def test_vlm_judge_forwards_extra(tmp_path, monkeypatch):
    monkeypatch.setenv("SII_VLM_API_KEY", "test-key")
    client = _sii_client(tmp_path)
    judge = VLMJudge(client, extra={"chat_template_kwargs":
                                    {"enable_thinking": False}})
    http = StubHttp()

    async def run():
        async with httpx.AsyncClient():
            return await judge.score(http, image_bytes=b"img",
                                     prompt="a landscape")

    assert asyncio.run(run()) == pytest.approx((7 + 8 + 6 + 9) / 40)
    assert http.bodies[0]["json"]["chat_template_kwargs"] == {
        "enable_thinking": False}
    # And the toggle joins the cache key: default judge misses this entry.
    default_judge = VLMJudge(client)
    key = client.request_key(
        b"img", default_judge.judge_prompt.format(prompt="a landscape"),
        default_judge.prompt_version,
        {"max_tokens": default_judge.max_tokens,
         "temperature": default_judge.temperature})
    assert not (tmp_path / f"{key}.json").exists()
