"""ZenMux chat-completions client for caption generation and review.

Every request is cached on disk under a key derived from the exact inputs that
determine the response: the encoded image bytes, the model, the prompt text,
and every request parameter.  A changed prompt therefore cannot silently reuse
an old response, and re-running a batch costs nothing for rows already done.

The client records billed usage per call.  Cost is derived from the price list
ZenMux publishes on ``/models`` and the price actually used is stored next to
the response, so a later price change does not rewrite history.

Credentials come from the ``ZENMUX_API_KEY`` environment variable and are never
written to artifacts.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image

ZENMUX_ROOT = "https://zenmux.ai/api/v1"
OPENROUTER_ROOT = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible endpoint plus where its credential comes from."""

    name: str
    root: str
    key_env: str
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return f"{self.root}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.root}/models"


DEEPSEEK_ROOT = "https://api.deepseek.com"

PROVIDERS = {
    "zenmux": Provider(name="zenmux", root=ZENMUX_ROOT, key_env="ZENMUX_API_KEY"),
    "deepseek": Provider(name="deepseek", root=DEEPSEEK_ROOT, key_env="DEEPSEEK_API_KEY"),
    "openrouter": Provider(
        name="openrouter", root=OPENROUTER_ROOT, key_env="OPENROUTER_API_KEY",
        extra_headers={"HTTP-Referer": "https://github.com/kaupane/artflow",
                       "X-Title": "artflow caption enrichment"},
    ),
}


def provider_for(model: str) -> Provider:
    """Infer the provider from the model id.

    A ``<provider>:<model>`` prefix selects the provider explicitly, which is
    needed when the same model id exists on more than one provider.  The prefix
    is not part of the id sent to the API; use :func:`model_id` for that.
    """
    if ":" in model:
        prefix, remainder = model.split(":", 1)
        if prefix in PROVIDERS:
            return PROVIDERS[prefix]
    return PROVIDERS["zenmux"]


def model_id(model: str) -> str:
    """The model id to send to the provider, without any provider prefix."""
    provider = provider_for(model)
    prefix = f"{provider.name}:"
    return model[len(prefix):] if model.startswith(prefix) else model

# Downscale bound for the image sent to the model.  Museum scans are far
# larger than any VLM needs, and image tokens (and therefore cost) scale with
# pixels: a 1024px edge costs roughly 1k input tokens.
DEFAULT_MAX_EDGE = 1024
DEFAULT_JPEG_QUALITY = 88

# Status codes a retry can plausibly fix.  Everything else is an answer rather
# than a failure, and is cached so the same request is never paid for twice.
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
    """Seconds to wait before retrying, honouring ``Retry-After`` when present."""
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(max(float(header), 1.0), 120.0)
            except ValueError:
                pass
    # A rate limit needs more patience than a dropped connection: the provider
    # is asking for less traffic, and every worker backs off at the same time.
    base = 4.0 if response is not None and response.status_code == 429 else 2.0
    return min(base * 2 ** attempt, 90.0) + random.uniform(0.0, 1.5)


class CaptionAPIError(RuntimeError):
    """A request failed in a way that should be visible to the caller."""


@dataclass
class ModelPricing:
    """USD per million tokens, as published for one model."""

    prompt: float
    completion: float
    currency: str = "USD"

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.prompt + completion_tokens * self.completion) / 1e6


def fetch_pricing(model: str, provider: Optional[Provider] = None,
                  timeout: float = 60.0) -> ModelPricing:
    """Read the published price for ``model`` from the provider's model list.

    Two schemas are handled.  ZenMux lists ``pricings`` with a value per million
    tokens; OpenRouter lists ``pricing`` as strings in USD per token.  Tiered
    prices (different rates above a prompt-size threshold) are not modelled; the
    first entry is recorded so the caller can see what was assumed.
    """
    provider = provider or provider_for(model)
    static = STATIC_PRICING.get(model_id(model))
    if static is not None:
        return static
    response = httpx.get(provider.models_url, headers=_auth_headers(provider), timeout=timeout)
    response.raise_for_status()
    for entry in response.json()["data"]:
        if entry["id"] != model_id(model):
            continue
        price = _pricing_from_entry(entry)
        if price is None:
            raise CaptionAPIError(f"model {model} publishes no token price: "
                                  f"{entry.get('pricings') or entry.get('pricing')}")
        return price
    raise CaptionAPIError(f"model {model} not offered by the provider")


# Providers whose model list carries no prices, transcribed from their public
# pricing pages.  DeepSeek bills half rate outside 01:00-04:00 and 06:00-10:00
# UTC on weekdays, so the value here is the off-peak rate; calls made during
# peak hours cost twice this.  Recorded per response so an old artifact is not
# reinterpreted with a new price.
STATIC_PRICING = {
    "deepseek-v4-flash-vision-exp": ModelPricing(prompt=0.22, completion=0.66),
    # Same flash tier as the other v4 models; DeepSeek has not published a
    # separate line for this preview id.
    "deepseek-v4.1-flash-expires-on-0910": ModelPricing(prompt=0.22, completion=0.66),
    "deepseek-v4-flash": ModelPricing(prompt=0.22, completion=0.66),
    # Current API id for the flash tier (the /models endpoint now lists
    # deepseek-flash, not deepseek-v4-flash).
    "deepseek-flash": ModelPricing(prompt=0.21, completion=0.63),
    "deepseek-v4-pro": ModelPricing(prompt=0.66, completion=1.98),
}


def _pricing_from_entry(entry: Dict) -> Optional[ModelPricing]:
    per_million = entry.get("pricings")
    if per_million:
        prompt = _first_value(per_million.get("prompt"))
        completion = _first_value(per_million.get("completion"))
        if prompt is not None and completion is not None:
            return ModelPricing(prompt=prompt, completion=completion)
        return None
    per_token = entry.get("pricing")
    if per_token:
        try:
            return ModelPricing(prompt=float(per_token["prompt"]) * 1e6,
                                completion=float(per_token["completion"]) * 1e6)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _first_value(entries: Optional[List[Dict]]) -> Optional[float]:
    if not entries:
        return None
    return float(entries[0]["value"])


def _auth_headers(provider: Provider) -> Dict[str, str]:
    key = os.environ.get(provider.key_env)
    if not key:
        raise CaptionAPIError(f"{provider.key_env} is not set")
    return {"Authorization": f"Bearer {key}", **provider.extra_headers}


def encode_image(path: str | Path, max_edge: int = DEFAULT_MAX_EDGE,
                 quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Return the JPEG bytes actually sent to the model."""
    Image.MAX_IMAGE_PIXELS = None  # museum scans exceed PIL's decompression bomb limit
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image_fingerprint(image_bytes: bytes) -> str:
    """Content hash of the exact pixels sent, used as cache and provenance key."""
    return hashlib.sha256(image_bytes).hexdigest()[:32]


@dataclass
class Response:
    """One model response plus everything needed to audit its cost."""

    request_key: str
    image_fingerprint: str
    model: str
    prompt_version: str
    text: str
    usage: Dict[str, Any]
    cost_usd: float
    pricing: Dict[str, float]
    settings: Dict[str, Any]
    latency_s: float
    cached: bool = False
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    # A failure that a retry can plausibly fix (rate limit, gateway error,
    # dropped connection).  Such a response is never written to the cache, so
    # the next run sends the request again instead of replaying the failure.
    transient: bool = False

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens", 0) or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens", 0) or 0)

    @property
    def reasoning_tokens(self) -> int:
        details = self.usage.get("completion_tokens_details") or {}
        return int(details.get("reasoning_tokens", 0) or 0)

    def to_record(self) -> Dict[str, Any]:
        return {
            "request_key": self.request_key,
            "image_fingerprint": self.image_fingerprint,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "text": self.text,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "pricing": self.pricing,
            "settings": self.settings,
            "latency_s": self.latency_s,
            "cached": self.cached,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "transient": self.transient,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Response":
        return cls(**record)


class CaptionClient:
    """Async client with a per-request disk cache.

    ``cache_dir`` holds one JSON file per request key.  Failed requests are
    cached too, marked with their error, so a rerun does not pay for the same
    deterministic failure again; delete the file to retry.
    """

    def __init__(self, cache_dir: str | Path, model: str, pricing: Optional[ModelPricing] = None,
                 concurrency: int = 8, timeout: float = 600.0, max_retries: int = 6,
                 provider: Optional[Provider] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.provider = provider or provider_for(model)
        self.api_model = model_id(model)
        self.pricing = pricing or fetch_pricing(model, self.provider)
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)

    def pricing_record(self) -> Dict[str, float]:
        return {"prompt": self.pricing.prompt, "completion": self.pricing.completion,
                "currency": self.pricing.currency}

    def request_key(self, image_bytes: bytes, prompt: str, prompt_version: str,
                    settings: Dict[str, Any]) -> str:
        payload = {
            "provider": self.provider.name,
            "model": self.model,
            "image": image_fingerprint(image_bytes),
            "prompt": prompt,
            "prompt_version": prompt_version,
            "settings": settings,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]

    async def generate(self, client: httpx.AsyncClient, *, image_bytes: bytes, prompt: str,
                       prompt_version: str, max_tokens: int, temperature: float = 0.2,
                       extra: Optional[Dict[str, Any]] = None) -> Response:
        settings: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if extra:
            settings.update(extra)
        key = self.request_key(image_bytes, prompt, prompt_version, settings)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            record["cached"] = True
            return Response.from_record(record)

        body = {
            "model": self.api_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()}},
                ],
            }],
            **settings,
        }
        response = Response(
            request_key=key, image_fingerprint=image_fingerprint(image_bytes), model=self.model,
            prompt_version=prompt_version, text="", usage={}, cost_usd=0.0,
            pricing=self.pricing_record(), settings=settings, latency_s=0.0,
        )
        async with self._semaphore:
            for attempt in range(self.max_retries):
                start = time.time()
                try:
                    http_response = await client.post(self.provider.chat_url, json=body,
                                                      headers=_auth_headers(self.provider),
                                                      timeout=self.timeout)
                except (httpx.HTTPError, OSError) as exc:
                    response.error = f"{type(exc).__name__}: {exc}"
                    response.transient = True
                    await asyncio.sleep(_retry_delay(attempt, None))
                    continue
                response.latency_s = time.time() - start
                if http_response.status_code in TRANSIENT_STATUS:
                    response.error = f"HTTP {http_response.status_code}: {http_response.text[:200]}"
                    response.transient = True
                    await asyncio.sleep(_retry_delay(attempt, http_response))
                    continue
                if http_response.status_code != 200:
                    response.error = f"HTTP {http_response.status_code}: {http_response.text[:400]}"
                    break
                try:
                    data = http_response.json()
                except ValueError as exc:
                    response.error = f"bad JSON: {exc}"
                    break
                message = data["choices"][0]["message"]
                response.text = (message.get("content") or "").strip()
                response.usage = data.get("usage") or {}
                response.finish_reason = data["choices"][0].get("finish_reason")
                # A retry may have succeeded after an earlier attempt failed.
                # The stale error would otherwise be recorded next to a good
                # caption and read as a failed row.
                response.error = None
                response.transient = False
                # Providers that bill directly report the charge; prefer it over
                # the price list, which can lag a promotion or a price change.
                reported = response.usage.get("cost")
                response.cost_usd = float(reported) if reported is not None else self.pricing.cost(
                    response.prompt_tokens, response.completion_tokens)
                if not response.text:
                    response.error = "empty content"
                break
        if not response.transient:
            cache_path.write_text(json.dumps(response.to_record(), ensure_ascii=False),
                                  encoding="utf-8")
        return response


async def run_batch(client: CaptionClient, jobs: List[Dict[str, Any]],
                    progress: Optional[Any] = None) -> List[Response]:
    """Run ``jobs`` concurrently, where each job is a kwargs dict for ``generate``."""
    async with httpx.AsyncClient() as http:
        tasks = [client.generate(http, **job) for job in jobs]
        results = []
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
            if progress is not None:
                progress(results)
        return results


def summarise(responses: List[Response]) -> Dict[str, Any]:
    """Aggregate cost and usage over a batch, counting cached calls separately."""
    billed = [r for r in responses if not r.cached]
    return {
        "requests": len(responses),
        "billed_requests": len(billed),
        "errors": sum(1 for r in responses if r.error),
        "prompt_tokens": sum(r.prompt_tokens for r in billed),
        "completion_tokens": sum(r.completion_tokens for r in billed),
        "reasoning_tokens": sum(r.reasoning_tokens for r in billed),
        "cost_usd": sum(r.cost_usd for r in billed),
        "mean_latency_s": (sum(r.latency_s for r in billed) / len(billed)) if billed else 0.0,
    }
