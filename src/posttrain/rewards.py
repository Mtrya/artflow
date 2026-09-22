"""Reward machinery for joint DMD+RL post-training (DMDR route, Stage 6).

Four pieces:

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
- The async reward pipeline: fans a rollout batch out to the local scorers
  (threads) and the VLM judge (the client's async HTTP path) concurrently,
  assembles ScoredSamples, and applies ensemble + group normalization.
  Local CLIP-based scorers (LAION aesthetic, HPSv2) are provided with lazy
  weight loading so importing this module stays cheap.

A held-out scorer must always be configured at the call site and never
optimized; it is the quantitative tripwire for hacking of the trained rewards.
"""

import asyncio
import io
import json
import re
from collections import deque
from dataclasses import dataclass, field
from statistics import pstdev
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

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
    image_bytes: Optional[bytes] = None  # set by the rollout before scoring


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


@dataclass
class LocalScorer:
    """A named local scorer wrapping a plain callable.

    The callable takes (image_bytes, prompt) and returns a raw score.
    Anything else with `.name` and a matching `.score(image_bytes, prompt)`
    method (e.g. the CLIP wrappers below) plugs into the pipeline the same
    way.
    """

    name: str
    fn: Callable[[bytes, str], float]

    def score(self, image_bytes: bytes, prompt: str) -> float:
        return self.fn(image_bytes, prompt)


class ClipAestheticScorer:
    """Improved LAION aesthetic predictor: CLIP ViT-L/14 image embedding -> MLP.

    The checkpoint is the christophschuhmann/improved-aesthetic-predictor MLP
    (768 -> 1024 -> 128 -> 64 -> 16 -> 1, dropout between layers, inactive in
    eval mode). The head output (roughly 0-10) is divided by 10 into [0, 1].
    The prompt is ignored (the predictor is prompt-free). Requires
    `open_clip_torch`; weights load in __init__ so the object, once built,
    is thread-safe for concurrent CPU/GPU forwards.
    """

    name = "aesthetic"

    def __init__(self, head_path: str, *, device: str = "cpu",
                 clip_model: str = "ViT-L-14",
                 pretrained: str = "laion2b_s32b_b82k"):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "ClipAestheticScorer requires open_clip_torch "
                "(pip install open_clip_torch)") from exc
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained, device=device)
        model.eval()
        dim = model.visual.output_dim
        head = torch.nn.Sequential(
            torch.nn.Linear(dim, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
        state = torch.load(head_path, map_location=device)
        # The upstream checkpoint nests the stack under a `layers.` prefix
        # (it was an attribute of a LightningModule); strip it.
        state = {k.removeprefix("layers."): v for k, v in state.items()}
        head.load_state_dict(state)
        head.to(device).eval()
        self._torch = torch
        self._model = model
        self._head = head
        self._preprocess = preprocess
        self._device = device

    def score(self, image_bytes: bytes, prompt: str) -> float:
        from PIL import Image
        torch = self._torch
        image = self._preprocess(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        with torch.no_grad():
            features = self._model.encode_image(
                image.unsqueeze(0).to(self._device)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            return float(self._head(features).item()) / 10.0


class HPSV2Scorer:
    """Human Preference Score v2: fine-tuned OpenCLIP ViT-H/14 cosine reward.

    Returns the image-text cosine similarity (the reference implementation
    reports 100x this value; we keep the [0, 1]-ish raw cosine so ensemble
    weights stay comparable to the other scorers). `checkpoint` is the HPS
    v2.1 state dict loaded over the LAION-2B ViT-H/14 base. Requires
    `open_clip_torch`.
    """

    name = "hps"

    def __init__(self, checkpoint: str, *, device: str = "cpu",
                 clip_model: str = "ViT-H-14",
                 pretrained: str = "laion2b_s32b_b79k"):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "HPSV2Scorer requires open_clip_torch "
                "(pip install open_clip_torch)") from exc
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained, device=device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        model.eval()
        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(clip_model)
        self._device = device

    def score(self, image_bytes: bytes, prompt: str) -> float:
        from PIL import Image
        torch = self._torch
        image = self._preprocess(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        text = self._tokenizer([prompt])
        with torch.no_grad():
            img_f = self._model.encode_image(
                image.unsqueeze(0).to(self._device)).float()
            txt_f = self._model.encode_text(text.to(self._device)).float()
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            return float((img_f @ txt_f.T).item())


@dataclass
class AsyncRewardPipeline:
    """Score a rollout batch with the ensemble, then group-normalize.

    Scoring fans out per sample: the VLM judge goes through the caption
    client's async HTTP path (its own concurrency limit), local scorers run
    in threads (bounded by `max_local_workers`).  A sample missing any
    *optimized* scorer (judge returned None, local scorer raised) is dropped
    so every surviving group member is compared under identical weights;
    held-out scorers are best-effort and never cause a drop.  Z is a running
    mean of recent group population-stds, clamped below by `z_min`; a group's
    own std feeds the history only after that group is normalized (the first
    group normalizes against its own std).
    """

    ensemble: RewardEnsemble
    judge: Optional[VLMJudge] = None
    judge_name: str = "judge"
    local_scorers: Sequence[Any] = ()
    z_window: int = 64
    z_min: float = 1e-3
    max_local_workers: int = 4
    _z_history: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._z_history = deque(maxlen=self.z_window)
        names = [s.name for s in self.local_scorers]
        if self.judge is not None:
            names.append(self.judge_name)
        unknown = set(names) - set(self.ensemble.weights)
        if unknown:
            raise ValueError(f"scorers without ensemble weights: {sorted(unknown)}")

    async def _score_one(self, http_client, sample: ScoredSample,
                         local_sem: asyncio.Semaphore) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        async def run_local(scorer) -> None:
            async with local_sem:
                value = await asyncio.to_thread(
                    scorer.score, sample.image_bytes, sample.prompt)
            scores[scorer.name] = float(value)

        tasks = [run_local(s) for s in self.local_scorers]
        if self.judge is not None:
            async def run_judge() -> None:
                value = await self.judge.score(
                    http_client, image_bytes=sample.image_bytes,
                    prompt=sample.prompt)
                if value is not None:
                    scores[self.judge_name] = value
            tasks.append(run_judge())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self._last_errors.append(str(result))
        return scores

    async def score_batch(self, http_client,
                          samples: List[ScoredSample]
                          ) -> Tuple[List[ScoredSample], Dict[str, Any]]:
        """Score and normalize one rollout batch (one or more prompt groups)."""
        missing = [s.image_ref for s in samples if s.image_bytes is None]
        if missing:
            raise ValueError(f"samples without image_bytes: {missing[:5]}")
        self._last_errors: List[str] = []
        local_sem = asyncio.Semaphore(self.max_local_workers)
        per_sample = await asyncio.gather(
            *(self._score_one(http_client, s, local_sem) for s in samples))
        optimized = set(self.ensemble.optimized_names())

        kept: List[ScoredSample] = []
        dropped = 0
        groups: Dict[str, List[ScoredSample]] = {}
        order: List[str] = []
        for sample, scores in zip(samples, per_sample):
            sample.raw_scores = scores
            if not optimized.issubset(scores):
                dropped += 1
                continue
            sample.combined = self.ensemble.combine(scores)
            kept.append(sample)
            if sample.group_id not in groups:
                groups[sample.group_id] = []
                order.append(sample.group_id)
            groups[sample.group_id].append(sample)

        for group_id in order:
            members = groups[group_id]
            combined = [m.combined for m in members]
            std = pstdev(combined) if len(combined) > 1 else 0.0
            z = max(self.z_min,
                    sum(self._z_history) / len(self._z_history)
                    if self._z_history else std)
            for member, r in zip(members, group_normalize(combined, z)):
                member.r = r
            self._z_history.append(std)

        stats = {"scored": len(kept), "dropped": dropped,
                 "groups": len(order),
                 "z": (self._z_history[-1] if self._z_history else None),
                 "scorer_errors": list(self._last_errors)}
        return kept, stats
