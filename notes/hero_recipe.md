# Ascend hero recipe

The complete runnable recipe is
[`configs/hero.toml`](../configs/hero.toml). It explicitly specifies every
training tunable for all three resolution stages and rejects missing or
unknown fields. Stage selection and checkpoint resume are operations.
Run state and recovery details live in [Ascend pretraining](archive/ascend_pretraining.md).

## Model and optimization

| Component | Selected setting |
|---|---|
| Architecture | `artflow-v2`, h1152, 16 heads, 1 double-stream + 24 single-stream blocks, FFN width 3072 (ratio 8/3), 532,496,992 parameters |
| Text | Frozen Qwen3-0.6B, online true early exit at layer 20, hidden size 1024 |
| Image latents | Qwen-Image VAE, 16 channels, factor 8; 2×2 latent patches |
| Conditioning | Fused pooled text/time, factor-1000 timestep features, branch RMSNorm before gates |
| Positions/modulation | Centered image RoPE and fixed text diagonal; independent double-stream and per-block shared single-stream modulation |
| Muon | Original per-chunk scaling, LR 0.02, weight decay 0.0015, momentum 0.95 |
| Auxiliary AdamW | Peak LR 1e-4, betas 0.9/0.95, epsilon 1e-8, ordinary decay 0.01 |
| Conditioning decay | 0.4 on `txt_pooled_proj.weight`, `c_mlp.0.weight`, `c_mlp.2.weight` |
| Gradient clipping | Global norm 1.0 |
| EMA | Bias-corrected, decay 0.9999, updated every optimizer step |

Muon chunks fused QKV/modulation matrices before orthogonalization. Its update
multiplier is `sqrt(max(1, rows/cols))` for each chunk. Conditioning matrices
belong to AdamW. The selected decay 0.4 was installed through a full-state
migration at 56k; [qualification and provenance](archive/ascend_pretraining.md) explain
that continuation.

## Schedule and sampling

One continuous 600k-update schedule uses 20k warmup and cosine decay. AdamW
starts at 1e-5, peaks at 1e-4 and reaches 5e-6. Stage endpoints are 450k, 570k
and 600k (256p / 640p / 896p, a 75:20:5 step split).

Rows are sampled from explicit weighted source pools; a caption is selected
within the row. Caption-length beta changes linearly from −1 to +1 over the
run. A 20% short-caption reserve uses a 256-token threshold. Caption dropout
is independently sampled at probability 0.1. Loss weighting uses log2 caption
length relative to 128 tokens. Logit-normal timestep parameters are 0 and 1;
resolution time shift is native. Text tokenization/encoding runs online;
precompute stores latents and cleaned caption text with sampling metadata.

All three bucket plans target 16×910B2C. Each contains 20 caption-length
buckets per aspect ratio, up to 2,048 tokens.

| Stage | Micro-batch per rank | Accumulation | Expected samples/update |
|---|---:|---:|---:|
| 256p | 8–72 | 1 | Depends on caption progress; about 950 in the early qualification |
| 640p | 5–12 | 3 | 528 planned; 532 measured |
| 896p | 3–6 | 4 | 373 planned; 375 measured |

The later-stage plans use the final mixtures below and their own caption
progress windows. Accumulation 3/4 keeps effective batches near the historical
targets of approximately 512/400; retaining 4/5 would increase training samples
and compute per optimizer update substantially. These are qualified practical
choices, not a claim of a globally optimal statistical batch size. Their
52 GiB planner ceiling includes model, encoder, optimizer, EMA, gradients and
DDP residency. The earlier 33/42 GiB DiT-only candidates failed before their
first update and are superseded. Full-workload results and limits are in
[the infrastructure record](infra_pretrain.md).

## Data additions for the later resolutions

These amendments address full-body people and underexposed world
content. The running 256p mixture stays fixed.

| Source | Current eligible rows / change | 640p weight | 896p weight |
|---|---|---:|---:|
| `d3-pexels` | +6,008 rows; pool 37,417 → 43,425 | 8.50 | 11.50 |
| `d2-museum` | User-selected source emphasis | 0.800000 | 0.825000 |
| `d4-extra` | D4 Pexels merged with megalith; 34,935 usable rows at 640p, 30,726 at 896p | 2.476274 | 3.467343 |

D4 Pexels contributes 29,574 rows. Megalith contributes 5,361/1,152 usable
rows at the respective resolutions. Scaling its former weights 0.38/0.13
by the eligible-row ratios preserves the unnormalized weight per pre-existing
row; source probabilities are normalized over the complete mixture. See
[dataset provenance](dataset_plan.md).

These are the working recipe's later-stage amendments. The active deployment
uses its pinned complete config. Strict resume compares the full recorded
recipe, including future stages, so applying amendments requires an explicit
stage-transition migration. Use
[`migrate_stage_recipe.py`](../scripts/pretrain/migrate_stage_recipe.py) on an
independent copy of the completed 450k checkpoint. It locks the completed
stage, model, optimizer, monitoring and global schedule; allows future-stage
data/bucket/accumulation amendments; verifies relocated current buckets and
prompts; and records old/new recipes plus hashes of every preserved state
artifact. Ordinary resume remains strict. Editing checkpoint metadata to
conceal a recipe mismatch would invalidate recovery evidence.

## Evaluation and telemetry

- Fixed loss probe every 500 updates, requesting 512 held-out cases per
  caption-length band with evaluation batch size 8. Record actual counts when
  eligibility or buckets reduce the requested set; compare matched panels/counts.
- Image grids every 2,500 updates and at the explicit stage `grid_steps`.
  [`hero_monitor_v1.jsonl`](../assets/eval/hero_monitor_v1.jsonl) contains
  12 scenes × Chinese/English × short/long captions = 48 images. Each scene
  shares its seed across its four variants. Review anatomy, architecture,
  style and layout as improved / unchanged / regressed / uncertain.
- Grid sampling: stored EMA, Euler 50, CFG 1, resolution time shift;
  solver times/state at least FP32, model forward and VAE decode BF16.
- End-of-global-training KID uses 2,000 generated images.
- Health telemetry every 250 updates, stability probes every 100, caption/source
  logging every 25. Preserve per-update samples, losses and gradient records.

Compare live and stored EMA loss together: smoothing can hide a live-model
regression. Detailed fixed-panel metric interpretation and review rules are
in [Ascend pretraining](archive/ascend_pretraining.md).

At the completed hero checkpoint, evaluate live weights first, then compare
stored EMA on loss, grids and KID before selecting the post-training teacher.
Post-hoc EMA is an option only if enough suitable checkpoint history was
retained; the rotating two-checkpoint policy cannot reconstruct arbitrary
historical decay profiles.

## Launch, checkpoints and resolution transitions

Set `paths.storage_root` in the complete run file to the prepared artifact
root. A relative root resolves against the config file; data/model/output/
bucket paths resolve against that root, and the prompt path against the
config file. Platform preparation is documented in `INSPIRE.md`.

```bash
python -m src.pretrain.train --config configs/hero.toml --stage 256p --check_config
python -m scripts.pretrain.launch --config configs/hero.toml --stage 256p --nproc_per_node 16
```

The launcher sets the qualified NPU expandable-segments allocator policy and
one OpenMP thread per worker before initialization. It holds a writer lock,
mirrors attempt output and terminates the worker group on NPU OOM.

Save every 2,000 updates and retain the latest two complete checkpoints per
stage. Checkpoints contain the complete run config, model metadata, active
bucket table, model/EMA, optimizers, schedulers, sampler and per-rank RNG state.
Recovery chooses the latest complete same-stage checkpoint; a stage's first
launch uses its predecessor's endpoint. `--resume` selects a complete checkpoint
explicitly. Same-stage recovery also checks bucket-table equality.

Preserve the complete 450k and 570k endpoints before transitioning. Qualify
the new bucket plan, accumulation, memory, monitoring and full-state recovery
while retaining global schedule progress. Any required recipe change uses a
tested migration with provenance. SwanLab records the complete actual recipe,
architecture version, selected stage and active bucket contents.

The [infrastructure record](infra_pretrain.md) contains matched throughput,
recovery and monitoring measurements. The current run supplies longer-term
stability evidence.
