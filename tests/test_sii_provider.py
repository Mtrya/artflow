"""Caption response caching checked against independent HTTP response fixtures."""

import asyncio

import httpx

from src.dataset.caption_client import CaptionClient, ModelPricing


class Reply:
    def __init__(self, text):
        self.text = text

    async def post(self, url, **kwargs):
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'choices': [{'message': {'content': self.text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
        })


def test_cache_preserves_response_and_separates_generation_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('SII_VLM_API_KEY', 'test-key')
    client = CaptionClient(cache_dir=tmp_path, model='sii:Qwen3.8-27B',
                           pricing=ModelPricing(prompt=0., completion=0.))

    async def run():
        settings = dict(image_bytes=b'img', prompt='describe this', prompt_version='v1',
                        max_tokens=64, temperature=0.)
        first = await client.generate(Reply('original'), **settings,
                    extra={'chat_template_kwargs': {'enable_thinking': False}})
        cached = await client.generate(Reply('different server response'), **settings,
                    extra={'chat_template_kwargs': {'enable_thinking': False}})
        changed = await client.generate(Reply('new settings response'), **settings,
                    extra={'chat_template_kwargs': {'enable_thinking': True}})
        return first, cached, changed

    first, cached, changed = asyncio.run(run())
    assert first.text == cached.text == 'original'
    assert changed.text == 'new settings response'
    assert all(reply.error is None for reply in (first, cached, changed))
