# Post-training preflight

Stage 6 targets Ascend for the teacher, student, fake-score network and
training updates. The shared VLM judge is a remote API; local scorer placement
follows measured compatibility and throughput. The method and sequence are in
[redesign_plan.md](redesign_plan.md); method notes are in `literature/`.

This note holds the stack-independent evidence for the reward stack: judge
configuration and probe results, scorer identities and output ranges,
throughput accounting, and the pipeline requirements a rewrite must satisfy.
The reward/RL implementation itself is still to be written; the measurements
below constrain it.

## Qualification checklist

Qualify on the intended Ascend allocation before setting budgets:

- Teacher/student/fake-score forward and backward, GAN head, 8-step rollout,
  VAE decoding, distributed updates and complete checkpoint/resume.
- Device and dtype assumptions against the qualified Ascend runtime.
- Real image bytes through local scorers and the API judge, including decoding,
  preprocessing, cache identity and group/sample alignment.
- Reward failures, backpressure and asynchronous completion; record sample
  drops and the effective group size used for normalization.
- End-to-end time and memory, separating rollout, teacher/fake-score compute,
  backward, synchronization, optimizer, local scoring and remote reward wait.

Use one embedded measurement pass, then the 640p pilot and the 896p main line.
Main-run and ablation budgets are set from measured wall time/NPU-hours and
judge demand.

## Judge configuration

Use the registered SII provider through
[`src/dataset/caption_client.py`](../src/dataset/caption_client.py), which
already provides async batching, retries, a disk cache and cost accounting.
Credentials come from `SII_VLM_API_KEY`; platform secret handling is in
`INSPIRE.md`. Measured settings: model `sii:Qwen3.8-27B`, temperature 0,
maximum 256 output tokens, thinking disabled via
`chat_template_kwargs: {"enable_thinking": False}`, concurrency at most 16.
Request extras must participate in the cache key.

The probe instrument was this rubric; its scalar is the equal-weight mean of
the four parsed 0–10 axes, divided by 10:

```text
You are grading a generated image against a prompt.

Prompt: {prompt}

Score each axis from 0 to 10:
- anatomy: bodies, hands and faces are structurally correct (10 = flawless).
- adherence: the image depicts what the prompt asks for (10 = exact).
- naturalness: the image looks like a real photograph or authentic artwork,
  with NO plastic skin, over-saturation, over-smoothing, or HDR-ish artifacts
  (10 = fully natural; penalize the "over-optimized" look harshly).
- aesthetics: composition, lighting, color harmony (10 = excellent).

Reply with JSON only: {"anatomy": int, "adherence": int, "naturalness": int,
"aesthetics": int}
```

## Judge probe evidence

The judge probe made 106 requests: 20 thinking-off, 20 thinking-on, 16
controls and 50 repeated thinking-off calls. It used ten early, often blurry
256p generated images and three real photos. There were zero transport
failures and one thinking-on truncation. This is evidence for the initial
judge setting; within-group ranking on the mature teacher remains to be
measured.

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

CPU checks loaded the real aesthetic and HPSv2 weights and scored the same ten
early generated cells plus three real Pexels images.

| Component | Weight / model identity |
|---|---|
| Aesthetic head | `camenduru/improved-aesthetic-predictor`, `sac+logos+ava1-l14-linearMSE.pth` |
| Aesthetic CLIP | ViT-L/14, `laion2b_s32b_b82k` |
| HPS | `xswu/HPSv2`, `HPS_v2.1_compressed.pt` |
| HPS CLIP | ViT-H/14, `laion2b_s32b_b79k` |

Checkpoint-format facts the rewrite must reproduce: the aesthetic head is a
768→1024→128→64→16→1 MLP whose state dict carries a `layers.` prefix, and its
prediction is divided by 10; HPSv2 scoring is image/text cosine similarity.
Both require compatible Torch/torchvision/open_clip installations and cached
weights. Prepared paths and dependency setup belong in machine-local
`INSPIRE.md`.

| Group | Aesthetic range / mean | HPS cosine range / mean |
|---|---|---|
| Generated, n=10 | 0.510–0.602 / 0.551 | 0.043–0.184 / 0.147 |
| Real, n=3 | 0.529–0.596 / 0.571 | 0.227–0.258 / 0.241 |

HPS separated these small groups; aesthetic scores overlapped substantially.
The aesthetic output is scaled rather than clamped to [0,1], and cosine scores
have their usual [-1,1] range. Chinese captions and the tiny panel limit
broader interpretation. Local scorer speed was not measured in this
correctness check.

## Reward pipeline requirements

Whatever shape the rewrite takes, these properties are requirements, not
options:

- A held-out scorer is always configured at the call site and never
  optimized; it is the quantitative tripwire for hacking of the trained
  rewards.
- A sample is dropped when any optimized scorer is missing or fails.
  Held-out scorer failures are best-effort and appear in error statistics.
  Inspect actual surviving group sizes and missing-score rates alongside
  reward statistics.
- Group normalization maps each prompt group's raw rewards to optimality
  probabilities: subtract the group mean, divide by a running scale history,
  clip to [-1, 1], then map to [0, 1] around 0.5. A constant group carries
  no preference signal.
- Cache records retain raw responses and errors. Replaying a cached permanent
  failure requires explicit invalidation if a new attempt is warranted.
  Instrument the client for usage, latency and retries when measuring cost.

## Reward throughput and accounting

The thinking-off batch at concurrency 16 achieved approximately
**3.7 calls/s**, or 0.27s amortized per image. Mean individual latency was
2.07s. Sustained 768/1,152-image batches and contention with shared users
need measurement; keep concurrency at most 16 and client backoff during
preflight. The SII calls had approximately zero API fee in that setup. Record
actual provider/quota terms when choosing the production service.

```text
calls = iterations × prompts_per_iteration × G × repeats × (1 + retry_margin)
judge_seconds = calls / measured_calls_per_second
api_cost = calls × actual_price_per_call
```

The table uses one call per image, a 10% allowance and the measured 3.7
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

Select the optimized reward weights, independent held-out signal,
normalization scale, G=16/24, prompts per iteration and any repeat averaging
using representative final-quality images. Record per-domain/language behavior
and reward-versus-KID/diversity trends. Align these choices with the method
and promotion rule in [the stage plan](redesign_plan.md).

Concept-coverage assessment used the concept benchmark, retired after the
480k round; the 600k final benchmark's form is still to be decided. Manual
grid review remains the guard against false capability gaps.
