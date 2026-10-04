# Ascend post-training preflight

Updated September 27, 2026. Stage 6 targets **Ascend** for the teacher, student,
fake-score network and training updates. The shared VLM judge is an API;
local scorer placement follows measured compatibility and throughput. The
method and sequence are in [redesign_plan.md](redesign_plan.md).

## Implemented components and remaining qualification

[`src/posttrain/`](../src/posttrain/) contains DMD, joint-loss, NFT and reward
components. The [prompt-pool builder](../scripts/posttrain/build_rollout_prompts.py)
reads precomputed captions from a complete run config and selected stage. Source
weights are sampling probabilities, matching `src/dataset/mix.py`. Sampling is
without replacement; exhausted sources redistribute their shortfall by the
remaining weights, and the command reports final source counts.

Complete the training launcher and orchestration, then qualify on the intended
Ascend allocation:

- Teacher/student/fake-score forward and backward, GAN head, 8-step rollout,
  VAE decoding, distributed updates and complete checkpoint/resume.
- Device/dtype assumptions, including the FP64 residual reduction currently
  used by `dmd_surrogate_loss`, against the qualified Ascend runtime.
- Real image bytes through local scorers and the API judge, including decoding,
  preprocessing, cache identity and group/sample alignment.
- Reward failures, backpressure and asynchronous completion; record sample
  drops and the effective group size used for normalization.
- End-to-end time and memory, separating rollout, teacher/fake-score compute,
  backward, synchronization, optimizer, local scoring and remote reward wait.

The CPU scorer checks below establish real-weight loading and outputs. Ascend
compatibility and reward throughput remain qualification work. Use one embedded
measurement pass, then the 640p pilot and 896p main line. Main-run and ablation
budgets are set from measured wall time/NPU-hours and judge demand.

## Judge contract

Use the registered SII provider through the existing caption client:

```python
from src.dataset.caption_client import CaptionClient, ModelPricing
from src.posttrain.rewards import VLMJudge

client = CaptionClient(
    cache_dir="reward_cache",
    model="sii:Qwen3.8-27B",
    pricing=ModelPricing(0.0, 0.0),
    concurrency=16,
)
judge = VLMJudge(
    client,
    extra={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Credentials come from `SII_VLM_API_KEY`; platform secret handling is in
`INSPIRE.md`. The judge uses temperature 0, maximum 256 output tokens and the
versioned four-axis rubric. Its scalar is the equal-weight mean of parsed
0–10 axes divided by 10. Request extras participate in the cache key.

`AsyncRewardPipeline` drops a sample when any optimized scorer is missing or
fails. Held-out scorer failures are best-effort and appear in error statistics.
The caller configures a held-out signal excluded from the optimized ensemble.
Group normalization uses a running scale history. Inspect actual surviving
group sizes and missing-score rates alongside reward statistics.

Cache records retain raw responses and errors. Replaying a cached permanent
failure requires explicit invalidation if a new attempt is warranted. Instrument
the client for usage, latency and retries when measuring cost; `VLMJudge.score`
returns only a score or `None`.

## Evidence selecting thinking off

The September 22 probe made 106 requests: 20 thinking-off, 20 thinking-on,
16 controls and 50 repeated thinking-off calls. It used ten early, often blurry
256p generated images and three real photos. There were zero transport failures
and one thinking-on truncation. This is evidence for the initial judge setting;
within-group ranking on the mature teacher remains to be measured.

| Observation | Result |
|---|---|
| Thinking off, two repeats | 9/10 image pairs identical; all differed by at most 0.025 in normalized score |
| Thinking off, five repeats | 4/10 images identical across repeats; 8/10 had range ≤2 summed rubric units; mean score SD 0.021, maximum 0.08 |
| Cross-image separation | Score SD approximately 0.21; larger than typical repeat noise in this small panel |
| Thinking on | Mean latency 33.3s versus 2.07s; 826 versus 33.5 output tokens; one truncation in 20 calls and inconsistent layout/anatomy scores |
| Controls | Real photos 0.85–0.90, gray 0, noise 0.075; a texture-like generated flower image reached 0.875 |

The texture result motivates heterogeneous rewards, a held-out scorer and
manual domain review. The panel has eight short Chinese, one long Chinese and
one short English prompt; it cannot establish general cross-language ranking
reliability. Repeat noise can be material for close candidate groups, so the
pilot must determine whether averaging repeated calls is useful.

## Local scorer weights and observed outputs

September 23 CPU checks loaded the real aesthetic and HPSv2 weights and scored
the same ten early generated cells plus three real Pexels images.

| Component | Weight / model identity |
|---|---|
| Aesthetic head | `camenduru/improved-aesthetic-predictor`, `sac+logos+ava1-l14-linearMSE.pth` |
| Aesthetic CLIP | ViT-L/14, `laion2b_s32b_b82k` |
| HPS | `xswu/HPSv2`, `HPS_v2.1_compressed.pt` |
| HPS CLIP | ViT-H/14, `laion2b_s32b_b79k` |

`ClipAestheticScorer` loads the 768→1024→128→64→16→1 MLP, strips the
checkpoint's `layers.` prefix and divides its prediction by 10.
`HPSV2Scorer` returns image/text cosine similarity. These implementations
require compatible Torch/torchvision/open_clip installations and cached weights.
Prepared paths and dependency setup belong in machine-local `INSPIRE.md`.

| Group | Aesthetic range / mean | HPS cosine range / mean |
|---|---|---|
| Generated, n=10 | 0.510–0.602 / 0.551 | 0.043–0.184 / 0.147 |
| Real, n=3 | 0.529–0.596 / 0.571 | 0.227–0.258 / 0.241 |

HPS separated these small groups; aesthetic scores overlapped substantially.
The aesthetic output is scaled rather than clamped to [0,1], and cosine scores
have their usual [-1,1] range. Chinese captions and the tiny panel limit broader
interpretation. Local scorer speed was not measured in this correctness check.

## Reward throughput and accounting

The September 22 thinking-off batch at concurrency 16 achieved approximately
**3.7 calls/s**, or 0.27s amortized per image. Mean individual latency was
2.07s. Sustained 768/1,152-image batches and contention with shared users need
measurement; keep concurrency at most 16 and client backoff during preflight.
The SII calls had approximately zero API fee in that setup. Record actual
provider/quota terms when choosing the production service.

```text
calls = iterations × prompts_per_iteration × G × repeats × (1 + retry_margin)
judge_seconds = calls / measured_calls_per_second
api_cost = calls × actual_price_per_call
```

The table uses one call per image, a 10% allowance and the historical 3.7
calls/s. Times are projections for the judge alone, excluding model work:

| Scenario | Base calls | With 10% allowance | Projected judge time |
|---|---:|---:|---:|
| One iteration, 48 prompts × G16 | 768 | 845 | 3.8 min |
| One iteration, 48 prompts × G24 | 1,152 | 1,267 | 5.7 min |
| 5,000-image pool | 5,000 | 5,500 | 24.8 min |
| G16/G24 comparison, 500 prompts at each G | 20,000 | 22,000 | 1.65 h |
| Two algorithm arms × 2,000 iterations × 48 × G16 | 3,072,000 | 3,379,200 | 253.7 h |
| Main run, J iterations at 48 × G16 | 768J | 844.8J | 228.3J seconds |

Repeat averaging multiplies calls and time by the repeat count. The algorithm
comparison is **3.38 million** calls including allowance; that scale makes
throughput a material design constraint even with zero per-call fees. Measure
an overlapped rollout/reward iteration before estimating total training time.

## Decisions for the pilot

Select the optimized reward weights, independent held-out signal, normalization
scale, G=16/24, prompts per iteration and any repeat averaging using
representative final-quality images. Record per-domain/language behavior and
reward-versus-KID/diversity trends. Align these choices with the method and
promotion rule in [the stage plan](redesign_plan.md).

Concept-coverage assessment used the concept benchmark, retired after the
480k round; the 600k final benchmark's form is still to be decided. Manual
grid review remains the guard against false capability gaps.
