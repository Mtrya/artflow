"""Reward machinery for joint DMD+RL post-training (DMDR route, Stage 6).

Three pieces:

- Group reward normalization (DiffusionNFT Algorithm 1): raw rewards are
  centered per prompt group and mapped to an optimality probability
  r in [0, 1] via a global scale Z.
- A weighted ensemble over heterogeneous scorers (local aesthetic/HPS-style
  scorers plus a VLM judge), so no single scorer's bias dominates — the
  primary defense against reward hacking at the reward-design level.
- The VLM judge: a thin rubric wrapper over the dataset captioning client,
  which already provides async batching, retries, a disk cache, and cost
  accounting. The judge inherits all of it, including the async pipeline
  (DiffusionNFT is natively off-policy, so reward latency is tolerable).

A held-out scorer must always be configured at the call site and never
optimized; it is the quantitative tripwire for hacking of the trained rewards.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..dataset.caption_client import CaptionClient, Response


def group_normalize(raw: List[float], z: float) -> List[float]:
    """Map a prompt group's raw rewards to optimality probabilities.

    r_norm = r_raw - mean(group); r = 0.5 + 0.5 * clip(r_norm / Z, -1, 1),
    with Z a global reward scale (e.g. the running std over recent groups).
    A constant group carries no preference signal and maps to all-0.5.
    """
    if not raw:
        return []
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    mean = sum(raw) / len(raw)
    out = []
    for value in raw:
        normed = (value - mean) / z
        out.append(0.5 + 0.5 * max(-1.0, min(1.0, normed)))
    return out


@dataclass
class ScoredSample:
    """One rollout with its rewards, before and after normalization."""

    prompt: str
    group_id: str
    image_ref: str            # path or buffer key for the generated image/latent
    raw_scores: Dict[str, float]
    combined: float
    r: Optional[float] = None  # optimality probability, set after group normalize


@dataclass
class RewardEnsemble:
    """Fixed-weight convex combination of named scorers.

    `weights` need not sum to one; they are normalized at combine time.
    `held_out` names scorers that may be computed for monitoring but are
    excluded from the optimized combination.
    """

    weights: Dict[str, float]
    held_out: List[str] = field(default_factory=list)

    def combine(self, scores: Dict[str, float]) -> float:
        total_w = 0.0
        total = 0.0
        for name, w in self.weights.items():
            if name in self.held_out:
                continue
            if name not in scores:
                raise KeyError(f"missing score '{name}'")
            total += w * scores[name]
            total_w += w
        if total_w <= 0:
            raise ValueError("no optimized scorer has positive weight")
        return total / total_w

    def optimized_names(self) -> List[str]:
        return [n for n in self.weights if n not in self.held_out]


JUDGE_PROMPT_VERSION = "judge-v1"

JUDGE_RUBRIC = """You are grading a generated image against a prompt.

Prompt: {prompt}

Score each axis from 0 to 10:
- anatomy: bodies, hands and faces are structurally correct (10 = flawless).
- adherence: the image depicts what the prompt asks for (10 = exact).
- naturalness: the image looks like a real photograph or authentic artwork,
  with NO plastic skin, over-saturation, over-smoothing, or HDR-ish artifacts
  (10 = fully natural; penalize the "over-optimized" look harshly).
- aesthetics: composition, lighting, color harmony (10 = excellent).

Reply with JSON only: {{"anatomy": int, "adherence": int, "naturalness": int,
"aesthetics": int}}"""

_SCORE_RE = re.compile(
    r'"(anatomy|adherence|naturalness|aesthetics)"\s*:\s*(\d+(?:\.\d+)?)')


def parse_judge_scores(text: str) -> Optional[Dict[str, float]]:
    """Extract the four rubric axes from a judge reply; None if unusable."""
    found = {m.group(1): float(m.group(2)) for m in _SCORE_RE.finditer(text)}
    axes = ("anatomy", "adherence", "naturalness", "aesthetics")
    if not all(axis in found for axis in axes):
        return None
    return {axis: min(10.0, max(0.0, found[axis])) for axis in axes}


class VLMJudge:
    """Rubric judge over the shared async captioning client.

    Scores are the mean of the rubric axes, rescaled to [0, 1]. Requests are
    disk-cached by the underlying client, so reruns and crashes do not pay
    twice; the cache also makes the judge testable without network access.
    """

    def __init__(self, client: CaptionClient, *, max_tokens: int = 256,
                 temperature: float = 0.0, judge_prompt: str = JUDGE_RUBRIC,
                 prompt_version: str = JUDGE_PROMPT_VERSION,
                 extra: Optional[Dict[str, Any]] = None):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.judge_prompt = judge_prompt
        self.prompt_version = prompt_version
        # Extra request-body fields passed through to the provider, e.g.
        # {"chat_template_kwargs": {"enable_thinking": False}} for the college
        # Qwen endpoint.  They join the cache key, so a toggle cannot silently
        # reuse a response scored under different settings.
        self.extra = extra

    async def score(self, http_client, *, image_bytes: bytes,
                    prompt: str) -> Optional[float]:
        prompt_text = self.judge_prompt.format(prompt=prompt)
        response: Response = await self.client.generate(
            http_client, image_bytes=image_bytes, prompt=prompt_text,
            prompt_version=self.prompt_version, max_tokens=self.max_tokens,
            temperature=self.temperature, extra=self.extra)
        if response.error or not response.text:
            return None
        axes = parse_judge_scores(response.text)
        if axes is None:
            return None
        return sum(axes.values()) / (10.0 * len(axes))

    def describe_cost(self) -> Dict[str, Any]:
        return {"model": self.client.model,
                "pricing": self.client.pricing_record()}
