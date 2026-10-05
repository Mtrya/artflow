"""A repaired endpoint is contacted again after a failed caption attempt."""

import asyncio
import json

import httpx
import pytest

from src.dataset.caption_client import CaptionClient, ModelPricing, Provider


@pytest.mark.parametrize("old_error", [None, "HTTP 401: expired"])
def test_only_successful_nonempty_responses_are_reused(
    tmp_path, monkeypatch, old_error
):
    monkeypatch.setenv("TEST_CAPTION_KEY", "test-key")
    attempts = []

    def serve(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(401, text="expired")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "A tree."}, "finish_reason": "stop"}
                ]
            },
        )

    async def exercise():
        captioner = CaptionClient(
            tmp_path,
            "test",
            pricing=ModelPricing(1, 1),
            max_retries=1,
            provider=Provider("test", "https://example.invalid", "TEST_CAPTION_KEY"),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:

            async def request():
                return await captioner.generate_text(
                    client, prompt="describe", prompt_version="v1", max_tokens=16
                )

            first = await request()
            assert first.error
            assert not list(tmp_path.glob("*.json"))
            if old_error:
                # Reproduce a failure file written by the old client.
                (tmp_path / f"{first.request_key}.json").write_text(
                    json.dumps(first.to_record())
                )
            second = await request()
            assert (
                second.text == "A tree." and second.error is None and not second.cached
            )
            third = await request()
            assert third.text == "A tree." and third.cached

    asyncio.run(exercise())
    assert len(attempts) == 2
