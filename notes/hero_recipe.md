# Hero run recipe

Last updated: 2026-09-14. Current configuration for the pre-NFT hero run.
Launch is pending end-to-end validation, final cost measurement, and the
stage-stop implementation listed below. Model, optimizer, data policy and stage
split are fixed; remaining optimization concerns infrastructure and the resulting
training steps/exposure within the budget. There is no numerical capability
target or inference hardware/latency gate. Image panels monitor regressions and
document the trained model's strengths and limitations.

## Training configuration

| item | value |
|---|---|
| Model | 1152 hidden / 16 heads, 1 double-stream + 24 single-stream blocks; 532,706,812 parameters; `configs/ladder-ld533.toml` |
| Hardware | 8× RTX 4090, 48 GB per card |
| Budget | Target the full ~2,200 RTX 4090 GPU-hours, including training overhead |
| Resolution stages | 256p → 640p → 896p; no 1024p stage |
| Stage step split | 75 : 20 : 5 of the finalized total optimizer steps |
| Mean effective-batch targets | ≥640 / ≥512 / ≥400 samples per update at 256p / 640p / 896p |
| Gradient accumulation | 1 / 5 / 7 micro-batches per rank |
| Optimizer | Chunked Muon for eligible 2-D hidden weights; auxiliary AdamW for embeddings, conditioning/output layers, norms, biases and other parameters |
| Peak LR | Muon 0.02; auxiliary AdamW 3e-4 |
| LR schedule | 5,000-step warmup, then cosine to 5% of peak over the whole run; `min_learning_rate=1.5e-5` for auxiliary AdamW, proportional Muon floor 0.001 |
| Caption selection | Exact retained-length beta selector; linear beta −1→+1 across the whole run; 0.20 reserve for captions below 256 tokens |
| Caption loss weight | `max(1, log2(L/128))`; classifier-free dropped captions retain weight 1 |
| Caption cap | 2048 retained tokens for training and planning |
| EMA | Decay 0.9999, updated every optimizer step |
| Timestep shift | Noise→data convention: `t' = t / (s - (s-1)*t)`, `s=max(1,sqrt(image_tokens/256))` |
| DDP | Accumulation-boundary gradient synchronization; no per-micro allreduce |
| Activation checkpointing | Disabled; validate full-stream memory with the selected plans |
| Data | Fixed eligible pools and per-stage mixtures below |

### Dataset mix (user spec 2026-09-14)

Canonical form: per-resolution multipliers on **absolute row counts**,
then normalize — weight_i ∝ rows_i × mult_i. Shard weights are split in proportion to their row counts.

| dataset | 256p | 640p | 896p |
|---|---:|---:|---:|
| d1 (Chinese painting) | 2.0 | 2.0 | 2.0 |
| d2-wikiart | 1.2 | 1.5 | 1.5 |
| d2-museum | 1.2 | 1.5 | 1.5 |
| d3-human | 1.8 | 1.8 | 1.5 |
| d3-people | 1.8 | 1.8 | 1.5 |
| d3-pexels | 3.0 | 3.0 | 3.0 |
| d3-synth-v2 | 1.2 | 1.2 | 0 |
| d4-inat | 1.0 | 0 | 0 |
| d4-megalith | 1.2 | 1.2 | 1.2 |
| d4-pd12m | 1.0 | 0.9 | 0.8 |
| d4-vintage | 1.0 | 1.0 | 1.0 |
| d4-zimage | 1.0 | 1.2 | 0 |
| d4-relaion | 1.0 | 1.0 | 1.0 |

Resulting normalized weights (%) — these are what the training configs
and the bucket-plan boundary computation consume:

| dataset | 256p | 640p | 896p |
|---|---:|---:|---:|
| d1 | 9.01 | 10.82 | 17.36 |
| d2-wikiart | 12.66 | 12.32 | 11.45 |
| d2-museum | 0.60 | 0.78 | 0.52 |
| d3-human | 10.50 | 12.19 | 15.75 |
| d3-people | 10.14 | 11.76 | 15.14 |
| d3-pexels | 5.53 | 6.66 | 10.74 |
| d3-synth-v2 | 0.64 | 0.78 | 0 |
| d4-inat | 0.14 | 0 | 0 |
| d4-megalith | 0.33 | 0.38 | 0.13 |
| d4-pd12m | 8.00 | 8.43 | 11.17 |
| d4-vintage | 9.59 | 9.41 | 1.57 |
| d4-zimage | 2.23 | 3.22 | 0 |
| d4-relaion | 30.64 | 23.24 | 16.16 |

Selected training-pool sizes: 1,628,830 / 1,268,958 / 759,017 rows at
256p / 640p / 896p, excluding zero-weight sources and evaluation rows.
Compute per-source repetition from the final sample budgets and mixture
weights; it is not fixed before the run length and realized batch sizes are known.

## Bucket plans and batch targets

Current plan revision: `batch-targets-0914`. Remote directory:
`$W/bucket_plans/hero/batch-targets-0914/`, where
`W` is the shared-workspace root supplied through `ARTFLOW_ROOT`.
Published reports use `${ARTFLOW_ROOT}` in place of machine-specific absolute
paths; the plan JSONs and their hashes below are unchanged.

| stage | plan | generation report |
|---|---|---|
| 256p | [JSON](../bucket_plans/hero/batch-targets-0914/hero-256p-k20.json) | [report](../bucket_plans/hero/batch-targets-0914/hero-256p-k20.report.md) |
| 640p | [JSON](../bucket_plans/hero/batch-targets-0914/hero-640p-k20.json) | [report](../bucket_plans/hero/batch-targets-0914/hero-640p-k20.report.md) |
| 896p | [JSON](../bucket_plans/hero/batch-targets-0914/hero-896p-k20.json) | [report](../bucket_plans/hero/batch-targets-0914/hero-896p-k20.report.md) |

Each plan has 20 caption-length buckets for each of five aspect ratios.
Boundaries minimize padded compute under the calibrated time model.
Per-bucket memory limits are unchanged; the alignment search maximizes its
throughput proxy **subject to the mean-batch target**. An infeasible target
raises an error instead of silently producing an undersized plan.

| stage | planner VRAM budget | minimum mean micro-batch | predicted mean micro-batch | accum | predicted 8-rank mean batch | target |
|---|---:|---:|---:|---:|---:|---:|
| 256p | 33 GB | 80 | 81.534840 | 1 | **652.28** | 640 |
| 640p | 30 GB | 12.8 | 14.609115 | 5 | **584.36** | 512 |
| 896p | 30 GB | 7.142857 | 7.607902 | 7 | **426.04** | 400 |

These are modeled long-run means, not guaranteed sizes of individual
updates. Verify actual sample counts over representative sampler windows,
including late-stage caption distributions, before accepting the targets.

Planning uses `dataset_weight / row_count × caption_probability`, retaining
the actual aspect-ratio mass. Caption probabilities reuse the trainer's
selector and are averaged at eight midpoint positions over stage progress
[0,.75], [.75,.95], and [.95,1]. This assumes equal exposure per progress point;
actual variable-batch exposure must be checked. The mean emitted micro-batch is
`1 / sum(p_i / B_i)`, and the planner threshold is
`global_target / (8 * accumulation)`.

The time-alignment proxy charges the slowest bucket and does not model the
expected slowest rank over accumulated micro-batches. It is not measured DDP
throughput. End-to-end memory must include text encoding, optimizers, EMA, DDP,
and retained compile allocations; changing bucket shapes requires validation.

Generation entry points:

- `scripts/bench/gen_hero_bucket_plans.py`: shared stage targets, accumulation,
  policy intervals, mixtures, and memory budgets.
- `scripts/bench/gen_hero_configs.py`: generates configs pointing to this same
  plan directory; `HERO_PLAN_OUTDIR` can override it for an explicit new revision.
- `jobs/hero_stage.sh`: accumulation 1/5/7 and the selected optimizer settings.

Plans were generated on Inspire's existing CPU preparation notebook; no GPU
job was launched. Full commands, calibration fits and warnings are in the
reports. Source snapshot: `$W/bucket_plans/artflow-batch-targets-0914.tar.gz`,
SHA-256 `d1ca60d4d2ca7404366ca9a9e87ad28e72d4cea9650e80bc80385fdba54910c8`.
Calibration: `$W/bucket_plans/calib-533m/merged.json`.

## Training length, stage boundaries, and cost

Let `T` be the finalized total optimizer-step count. **400k is the
infrastructure benchmark target, not the final length or a ceiling.**
If validated throughput permits 420k within the budget, use 420k.

Each stage is a separate launch with a shared schedule horizon `T`.
Cumulative stopping/checkpoint boundaries are `floor(0.75*T)`,
`floor(0.95*T)`, and `T`; their differences are the per-stage step budgets.

| example T | 256p steps / endpoint | 640p steps / endpoint | 896p steps / endpoint |
|---|---:|---:|---:|
| 400k | 300k / 300k | 80k / 380k | 20k / 400k |
| 420k | 315k / 315k | 84k / 399k | 21k / 420k |

A separate `stop_at_step` must stop and checkpoint the invocation without
changing its LR/caption schedule horizon. The next stage can be launched
manually; restore weights, optimizer, scheduler and EMA, validate the predecessor
endpoint, reset resolution-specific sampler queues, and initialize caption
progress from the restored global step before drawing data. Caption progress is
initialized before data prefetch and uses the same whole-run formula as subsequent
updates; resolution changes do not restart the short-to-long curriculum. Same-stage
resume retains the saved sampler queues and replay batches.

This stopping control and boundary validation are **not yet implemented**.
The current launcher's `max_steps=T` alone does not enforce stage budgets.
Wall-time termination and a shorter per-stage schedule horizon are not substitutes.

For measured wall-seconds per optimizer step `t256, t640, t896` on eight ranks:

```text
weighted_step_seconds = 0.75*t256 + 0.20*t640 + 0.05*t896
training_GPU_hours = T * weighted_step_seconds * 8 / 3600
```

Add compilation/startup, evaluation, checkpointing, allocated idle time and
restart allowances not already included in those rates. Derive `T` from the
remaining training budget, account for integer stage boundaries, and check
the complete rounded schedule against 2,200 GPU-hours. Record sample budgets
from actual emitted batches rather than assuming identical samples per step.

The 400k benchmark permits 19.8 GPU-seconds per step **including overhead**,
equivalent to 2.475 wall-seconds on eight ranks. If that target is missed,
provide measured FLOP/MFU and bottleneck evidence for the infrastructure
decision. Current-plan end-to-end times and final `T` remain unmeasured;
there is no accepted wall-clock estimate yet.

## Remaining launch checks

1. Validate all three selected plans on eight ranks with accumulation 1/5/7:
   realized mean batch, caption exposure, long-caption tails, memory peaks,
   steady-state speed, and compilation/startup cost.
2. Finalize total steps and per-stage endpoints from those measurements, with
   overhead and bounded restart allowances inside the hero budget.
3. Implement and smoke-test stage stopping, endpoint checkpoints, and cross-stage
   continuation with continuous LR/caption progress.
4. Generate and verify resolved stage configs against the selected plan paths;
   pin code, data/metadata, configuration and artifact revisions. Do not change
   inputs of running jobs; deploy the selected revision for new validations.
5. Verify the operational checks below and resume behavior before the hero launch.

Operational checks: use the frozen 48-image bilingual short/long-prompt panel
in [stage4_eval_freeze.md](stage4_eval_freeze.md), covering full-body figures,
faces, hand–object interaction, styles/buildings, and layout. The hero config
selects `assets/eval/hero_monitor_v1.jsonl`: 12 scene grids with shared seeds,
Chinese/English columns and short/long rows, rendered with EMA, Euler-50,
bf16 and no CFG. Full prompts and sampling records accompany the grid PNGs.
Review it at the existing 10k-step image-evaluation cadence, at each stage endpoint,
and after the first 2k updates at a new resolution; render the incoming checkpoint
at that resolution as the transition baseline. Proceed only from a complete,
loadable endpoint checkpoint with the expected step and continuous training state.
Stop on nonfinite loss/gradients or corrupted state; pause for review if successive
panels show worsening artifacts or loss of previously demonstrated abilities.
Compare loss trends within a resolution, not raw losses across resolutions.
Preserve the last verified checkpoint; do not automatically extend training,
change the recipe, or roll back and restart without review. Charge these checks
and any approved recovery to the 2,200 GPU-hour ceiling. These are regression and
execution-safety checks, not capability qualification gates.

The final infrastructure pass remains bounded within Stage 4's remaining
450 GPU-hour experiment/profiling cap. No further broad sweep is required.

## Optional later adjustments

These are not launch requirements and require validation before adoption:

- Compile the frozen text-encoder path or pre-tokenize captions if profiling
  demonstrates meaningful remaining host overhead.
- Evaluate bf16 gradient communication, checking numerical behavior as well as speed.
- Consider rank-synchronized bucket order if exposed imbalance warrants changing
  batch composition.
- Revisit source-mixture weights at resolution boundaries only through an explicit
  recipe revision; the tables above define the current default.
