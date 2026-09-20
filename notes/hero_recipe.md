# Hero run recipe

Last updated: 2026-09-18. Current configuration for the pre-NFT hero run.
1024p is definitively dropped: the complete pre-NFT path is 256p → 640p → 896p
with a 75:20:5 optimizer-step split. All scientific recipe decisions are frozen.
The hero run is **launched and fixed on 4× RTX 4090** (user, 2026-09-18):
T = 480,000 optimizer steps at priority 1 (preemptible, auto-restart), no
GPU-hour budget; the H100/H200 variant and its 800-hour budget frame are
dropped. Stage launches chain with `jobs/hero_stage_4090.sh <stage> 480000 4`
(endpoints 360,000 / 456,000 / 480,000).
There is no numerical capability
target or inference hardware/latency gate. Image panels monitor regressions and
document the trained model's strengths and limitations.

## Training configuration

| item | value |
|---|---|
| Model | 1152 hidden / 16 heads, 1 double-stream + 24 single-stream blocks; 532,706,812 parameters; `configs/hero.toml` layered after `configs/base.toml` |
| Hardware | **Fixed: 4× RTX 4090** (user decision 2026-09-18, after the 4-rank job scheduled and passed validation). H100/H200 variant dropped along with its 800-GPU-hour budget frame; 896p accumulation-2 H200 plan preserved on GPFS if ever revisited |
| Budget | No GPU-hour budget on the 4090 path — fixed **480,000 optimizer steps** at priority 1 (preemptible idle-fill) until done |
| Resolution stages | 256p → 640p → 896p; 1024p definitively dropped, not a pending optional stage |
| Stage step split | 75 : 20 : 5 of the finalized total optimizer steps |
| Mean effective-batch targets | ≥640 / ≥512 / ≥400 samples per update at 256p / 640p / 896p |
| Gradient accumulation | 4×4090 final: **256p = 3 with the v93 four-rank plan** (`fallback4r-proposal-v93/256p-acc3`, the 8-rank 256p plan OOMs 48-GB cards at four ranks), **640p = 10, 896p = 14** on the standard `batch-targets-0914` plans — all preserving the frozen mean effective-batch targets |
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

The existing 4090 execution candidate in `jobs/hero_stage.sh` uses dynamic block
compilation with matmul autotuning at every resolution, disables DDP compiler
splitting, hoists double-stream RoPE, uses native varlen flash attention and real
RoPE, and compiles the square Newton–Schulz portion of Muon. Health snapshots
stay on GPU and periodic allocator cleanup is disabled in `configs/hero.toml`.
Compiler workers are capped at two per rank; optional CPU-wall/breakdown and
per-micro shape logging are off. FP32 parameter/gradient/EMA and communication
policies are unchanged. These settings are selected from lower-rank comparisons;
their H100/H200 acceptance and all-in costs are still pending. Do not use this
launcher as a finalized H100/H200 recipe until that work is complete.

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
  plan directory for an explicit candidate revision; it recalculates shard
  weights from row counts and is not the frozen launch path.
- `configs/hero/{256p,640p,896p}.toml.in` preserves the exact measured shard
  weights and artifact paths. `scripts/bench/render_hero_stage.py` substitutes
  only an explicit absolute artifact root, without reading or reweighting data.
- `jobs/hero_stage.sh`: renders those versioned inputs into a private temporary
  config, then applies the fixed policy and accumulation 1/5/7. It no longer
  depends on mutable external `$W/configs/hero-*.toml` files.

Plans were generated on Inspire's existing CPU preparation notebook; no GPU
job was launched. Full commands, calibration fits and warnings are in the
reports. Source snapshot: `$W/bucket_plans/artflow-batch-targets-0914.tar.gz`,
SHA-256 `d1ca60d4d2ca7404366ca9a9e87ad28e72d4cea9650e80bc80385fdba54910c8`.
Calibration: `$W/bucket_plans/calib-533m/merged.json`.

## Training length, stage boundaries, and cost

Let `T` be the finalized total optimizer-step count. **400k is the
infrastructure benchmark target, not the final length or a ceiling.**
If validated throughput permits 420k within the budget, report it as an option.
The infra pass supplies a measured cost curve and uncertainty; the user selects
the final production `T` afterward. Do not freeze the hero step count during
infra closeout.

Each stage is a separate launch with a shared schedule horizon `T`.
Cumulative stopping/checkpoint boundaries are `floor(0.75*T)`,
`floor(0.95*T)`, and `T`; their differences are the per-stage step budgets.

| example T | 256p steps / endpoint | 640p steps / endpoint | 896p steps / endpoint |
|---|---:|---:|---:|
| 400k | 300k / 300k | 80k / 380k | 20k / 400k |
| 420k | 315k / 315k | 84k / 399k | 21k / 420k |

`[train].stop_at_step` stops and checkpoints the invocation without changing
its LR/caption schedule horizon (`max_steps=T`). Zero disables the separate stop;
enabled endpoints must be integers within `[1,T]`. The next stage is launched
manually with `jobs/hero_stage.sh <resolution> <T>`. The launcher validates the
predecessor endpoint and resumes weights, optimizer, scheduler and EMA directly
from it, using `--reset_sampler` to start fresh resolution-specific queues without
copying or modifying the predecessor. Caption progress is
initialized before data prefetch and uses the same whole-run formula as subsequent
updates; resolution changes do not restart the short-to-long curriculum. Same-stage
resume retains the saved sampler queues and replay batches.

Resuming exactly at an endpoint performs no additional updates; a checkpoint
beyond the endpoint is rejected. Every endpoint is saved before image/loss
evaluation, even between regular checkpoint slots. Intermediate stops skip
end-of-global-training KID; stage panels still run at their scheduled endpoints.

Each checkpoint publishes `training_state.json` after all ranks finish saving.
It records the global step, T, scheduler count, EMA policy, rank count and file
sizes. Staged resume requires this completion record and matching T; missing or
truncated files are rejected, and scheduler steps must match the checkpoint step.
Older checkpoints without the record remain available for non-staged resume,
but cannot silently seed the hero stages. Full resume never silently reinitializes
a missing scheduler or enabled EMA. File-size checks do not replace actual
deserialization or the distributed smoke test still required by the infra pass.
Finalize T before starting the hero; do not change it between stages.

Retention is **keep all hero checkpoints** through training and post-run
verification; the trainer does not automatically prune them. Preserve all
three resolution endpoints and the last successfully loaded recovery point.
At a 2,000-step cadence, 400k steps produces about 200 checkpoints; the measured
single-rank checkpoint size of about 6.46 GB implies roughly 1.29 TB before rank
sidecars, incomplete writes and other run artifacts. The infra pass must replace
this estimate with eight-rank size measurements, count exact endpoints at final
T, and verify writable capacity with headroom. Shared-pool free space is not a
reserved project allowance. Incomplete checkpoints remain for inspection and
must not silently become resume inputs; cleanup requires explicit target review.

`jobs/hero_stage.sh` holds a non-blocking writer lock in the stage run directory
before selecting a checkpoint, preventing concurrent writers from duplicate
submissions. A leftover `.writer.lock` file is normal after exit; never delete it
to bypass a live holder. If the newest checkpoint is incomplete or inconsistent,
the launcher stops. Rollback is pre-authorized by the user (2026-09-19): after
confirming all writers are terminal, preserve the failed directory under a
`quarantine/` subdirectory (outside the launcher's top-level checkpoint glob),
explicitly validate the preceding checkpoint, relaunch from it without waiting
for approval, and report the rollback to the user afterwards. Do not change
the shared global T or advance the step to conceal lost work. Replayed updates,
startup and repeated evaluation must fit the recovery allowance. Target shared-
filesystem lock contention/release remains a final launch check.

For measured wall-seconds per optimizer step `t256, t640, t896` on eight ranks:

```text
weighted_step_seconds = 0.75*t256 + 0.20*t640 + 0.05*t896
training_GPU_hours = T * weighted_step_seconds * 8 / 3600
```

Add compilation/startup, evaluation, checkpointing, allocated idle time and
restart allowances not already included in those rates. Estimate feasible `T`
ranges from the remaining training budget, account for integer stage boundaries, and check
the complete rounded schedule against 800 H100/H200 GPU-hours. Record sample budgets
from actual emitted batches rather than assuming identical samples per step.

The 400k benchmark permits 7.2 GPU-seconds per step **including overhead**,
equivalent to 0.900 wall-seconds on eight ranks. If that target is missed,
provide measured FLOP/MFU and bottleneck evidence for the infrastructure
decision. Final-path eight-rank end-to-end times remain unmeasured;
there is no accepted wall-clock estimate yet. Production `T` remains unselected.

Use `python -m scripts.bench.cost_hero_run <measurements.json> --budget-gpu-hours 800` to cost the
final measured eight-rank path and estimate the largest integer `T` within the
budget under each stated cost scenario, without selecting production `T`.
The input schema is documented in that module: supply representative,
conservative per-stage update times, startup/compilation excess, whole-allocation
checkpoint/grid/loss-probe durations, final KID, allocated idle and bounded
recovery costs. These must be non-overlapping accounting categories. At 400k,
the frozen schedule has 200 checkpoints, 44 image-panel evaluations and 803
loss evaluations (800 periodic plus three stage-entry baselines), before any
recovery repeats. The calculator accounts for overlaps and exact integer
boundaries; its output is conditional accounting, not measured throughput or a
Stage 5 readiness certificate. No measured input file or final `T` is established
yet.

## Remaining infrastructure work and launch checks

All remaining work is infrastructure optimization, execution readiness, or a
quantity derived from that work. No additional resolution, model-size, corpus,
mixture, optimizer, curriculum, or capability-forecast experiment is required.

1. **Measure, optimize, then remeasure** all three resolutions on eight ranks.
   Start with the current plans and accumulation 1/5/7. Profile execution,
   memory/allocator behavior, communication/rank imbalance, and the input/text
   path; implement and test changes addressing the measured bottlenecks.
   Report before/after end-to-end rates, realized mean batch, caption exposure,
   long-caption tails, memory peaks, and compilation/startup cost. Re-screen
   bucket sizes/bounds and accumulation where justified, preserving the fixed
   training policies and documenting any change in actual samples per update.
2. Deliver the measured cost curve and budget-supported step range, including
   overhead and bounded restart allowances. Leave production T to the user;
   its cumulative stage endpoints are `floor(0.75*T)`, `floor(0.95*T)`, and `T`.
3. Run the distributed smoke test of stage stopping, endpoint checkpoints, and cross-stage
   continuation with continuous LR/caption progress.
4. The selected model/caption policy is now versioned in `configs/hero.toml`;
   the launcher no longer depends on external ladder configs. Verify fully resolved configs
   against the fixed recipe and selected plan paths. Pin code/dependencies,
   data/metadata, encoder/VAE and configuration/artifact revisions. Do not change
   inputs of running jobs; deploy the selected revision for new validations.
5. The coordinated nonfinite-loss/gradient guard is implemented before optimizer
   updates and tested with two CPU ranks; validate it on the final eight-GPU stack.
   Verify the monitoring/review and resume workflows, checkpoint
   retention, storage capacity and bounded recovery allowances before launch.

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
and any approved recovery to the 800 H100/H200 GPU-hour ceiling. These are regression and
execution-safety checks, not capability qualification gates.

The original Stage-4 450 GPU-hour cap is exhausted according to the job ledger.
The user approved **120 additional RTX 4090 GPU-hours** for this final pass
(96 on September 14 plus 24 on September 16), separate from the hero budget.
The user separately approved **16 H100/H200 GPU-hours for the migration pilot**
on September 17. Do not count H100/H200 hours as RTX 4090 hours or spend the
800-hour hero budget on validation.
See [infra_pass.md](infra_pass.md).
No further broad scientific sweep is required.

## Final-model evaluation policy (decided 2026-09-19)

Mid-training probe and grids intentionally stay on the EMA copy with
ema_decay=0.9999. Under the constant muon learning rate the EMA runs a
steady-state tracking lag of roughly the model's state from 7-10k steps
earlier, so mid-run EMA readings describe the live model, not its own quality;
they are trend references only (measured live-probe check at step 10k confirms
the training itself is healthy). No training change is made for this.

After the full run completes, evaluate final quality on **live weights
first**, then decide whether to reconstruct an EMA **post-hoc** (EDM2-style,
arXiv 2312.02696): the per-2000-step checkpoints allow reconstructing any
decay profile offline and comparing live vs post-hoc-EMA head to head on
probe loss, grids and KID before selecting the released weights. The cosine
tail is expected to close most of the EMA gap on its own, making the stored
run-EMA a strong candidate as well.

## Candidate optimizations for the infra pass

Choose from measured bottlenecks; these are candidates, not predetermined winners
or separate mandatory sweeps. Test correctness and end-to-end benefit before
adoption, and record why a tested candidate was accepted or rejected:

- Compile the frozen text-encoder path or pre-tokenize captions if profiling
  demonstrates meaningful remaining host overhead.
- Evaluate bf16 gradient communication, checking numerical behavior as well as speed.
- Consider rank-synchronized bucket order if exposed imbalance warrants changing
  batch composition.

The pass does not reopen source mixtures or add a 1024p stage. Any unexpected
need to change the scientific recipe is a separate redesign requiring approval.
