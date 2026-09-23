# Hero run recipe

Last updated: 2026-09-24. Current configuration for the pre-NFT hero run.
1024p is definitively dropped: the complete pre-NFT path is 256p → 640p → 896p
with a 75:20:5 optimizer-step split. All scientific recipe decisions are frozen.
The user authorized a **fresh 4×H200 hero after qualification** on 2026-09-24:
**600,000 steps, 20,000-step LR warmup, higher 256p D4 share, and
bias-corrected EMA**, with endpoints 450,000 / 570,000 / 600,000.
The four-H200 smoke retry is submitted after scheduler preemption;
production has **not yet launched**.
The earlier 4×4090 run (T=480k, endpoints 360k/456k/480k) is a historical
reference, not the initialization source. See [H200 qualification](h200_smoke_0923.md)
for current acceptance evidence and launch status. The repeatable H200 launcher
is `scripts/pretrain/hero_stage_h200.sh` with `configs/hero/h200.toml` layered
after the base hero policy.
There is no numerical capability
target or inference hardware/latency gate. Image panels monitor regressions and
document the trained model's strengths and limitations.

## Training configuration

| item | value |
|---|---|
| Model | 1152 hidden / 16 heads, 1 double-stream + 24 single-stream blocks; 532,706,812 parameters; `configs/hero.toml` layered after `configs/base.toml` |
| Hardware | **4×H200**, conditional on the current qualification passing; four-GPU quota is LOW/preemptible |
| Run length | **600,000 optimizer steps**, fresh initialization (user, 2026-09-24); historical 4090 run was 480k |
| Resolution stages | 256p → 640p → 896p; 1024p definitively dropped, not a pending optional stage |
| Stage step split | 75 : 20 : 5 of the finalized total optimizer steps |
| Mean effective-batch targets | ≥640 / ≥512 / ≥400 samples per update at 256p / 640p / 896p |
| Gradient accumulation | H200 256p candidate: **1**, tripling v93 micro-batches, pending throughput acceptance. **640p/896p are to be tuned separately** with larger micro-batches; 10/14 are conservative 4090 reference values, not required H200 settings. Higher-stage tuning does not block the 256p hero (user, 2026-09-24) |
| Optimizer | Chunked Muon for eligible 2-D hidden weights; auxiliary AdamW for embeddings, conditioning/output layers, norms, biases and other parameters |
| Peak LR | Muon 0.02; auxiliary AdamW 3e-4 |
| LR schedule | **20,000-step warmup**, then cosine to 5% of peak over the 600k-step horizon; `min_learning_rate=1.5e-5` for auxiliary AdamW, proportional Muon floor 0.001 |
| Caption selection | Exact retained-length beta selector; linear beta −1→+1 across the whole run; 0.20 reserve for captions below 256 tokens |
| Caption loss weight | `max(1, log2(L/128))`; classifier-free dropped captions retain weight 1 |
| Caption cap | 2048 retained tokens for training and planning |
| EMA | **Bias-corrected warmup**, `decay_t = min(0.9999, (1+t)/(10+t))`, updated every optimizer step; full resume retains EMA weights and global t |
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

### Dataset mix

The 2026-09-24 H200 amendment uses the higher-D4 256p mixture already in
`configs/hero/256p.toml.in`: normalized D4 share **55.4522%**, up from about
50.9% in the earlier NVIDIA snapshot. The 640p/896p mixtures are unchanged.
The tables below record the original 2026-09-14 reference mixture; the
versioned H200 input TOMLs and their hashes are authoritative for the new run.

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

## Ascend (910B) variant — 2026-09-22, signed off by user

Applies only if the hero run migrates to the 昇腾卡公共空间 (16×910B/node).
Everything not listed here is identical to the 4090 recipe above (model shape,
data mix, curriculum, optimizer, evaluation policy).

| item | 4090 value | Ascend value |
|---|---|---|
| total steps T | 480k (anchored) | **600k** |
| warmup | 5,000 | **20,000** |
| LR schedule | cosine to min 1.5e-5 | unchanged (T=600k horizon) |
| EMA | 0.9999 | 0.9999 with **bias-corrected warmup** (user, 2026-09-22): `decay_t = min(0.9999, (1+t)/(10+t))` — the ADM/EDM schedule, `ema_decay_warmup = true` in the launch override |
| bucket plan | batch-targets-0914 | **ascend-0922** (44GiB level, bs 8..72; the 48g variant was retired — see amendment below) |
| grad accumulation | 1/5/7 (256/640/896) | **1/4/5** (effective batch ≈1010/750/470 vs thresholds 640/512/400) |
| allocator | cuda native | **`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`**, no periodic cache_clear (amendment 2026-09-22 late) |
| compile | per-block inductor | none (`--no-compile`) — torchair decision 2026-09-23: **dead**, see below |
| attention | native varlen flash | `npu_fusion_attention` bool (B,1,S,S) drop-mask, explicit scale (0922b) |

Bucket-budget calibration (sweep3 linear model + two real 16-card runs, all
30/30 unless noted): 44GiB budget → OK; 54GiB → OOM at 59.5GiB active; 50GiB →
OK with **peak 58.9GiB / reserved 59.5GiB** (measured via the new NPU memory
telemetry). Real static (Muon+Adam+EMA) ≈ 12GiB vs the sweep's 3.3GiB (SGD).
~~Final choice 48GiB~~ — **superseded**, see amendment below.

**Amendment 2026-09-22 late (smoke3-8 + memdiag6):** the 48g plan died in
practice — smoke3/4 OOMed at ~step 55-250 with reserved pinned at the
~59.7GiB ceiling while active was only ~51.4GiB (fragmentation accumulates
with the number of distinct bucket shapes seen; `max_split_size_mb:256` did
not help). The run moved to the **ascend-0922 (44GiB-level) plan**. Periodic
`empty_cache` (cache_clear) did not prevent the creep either (smoke6) and was
dropped. **memdiag6 settled the mechanism**: with `expandable_segments:True`
reserved oscillates 51-60GiB with the shape mix but does **not** grow
monotonically (fixed-shape control run is perfectly flat: resv 51.11GiB,
seg 568, 100 steps), i.e. no leak — the earlier OOMs were peak-vs-ceiling,
and cache_clear only added release/realloc spikes. Final memory config:
`expandable_segments:True` alone (smoke7 two-factor + smoke8 single-factor
both green over 750 steps). bs ranges at 44g: 256p 8..72 / 640p 5..13 /
896p 4..6.

EMA note (user decision 2026-09-22): the Ascend run launches with the
bias-corrected warmup schedule directly. The form `min(d, (1+t)/(10+t))` is
the ADM/EDM codebase standard; it removes the initialization drag and keeps
the averaging window short through the 20k LR warmup (saturation at t≈90k).
What it does not change is the steady-state tracking lag at high constant LR
(the `ema_rel_distance` plateau seen on the 4090 run) — that is inherent to
any fixed decay and resolves in the cosine tail; if mid-run grids look too
stale we can still lower the decay itself at a later restart.

Wall-clock estimate (single 16-card node, measured bucket times): ~312h ≈ 13
days for 600k steps (~5000 NPU-hours). Multi-node would shorten this but the
昇腾 workspace offers only 16-card job quotas, so single-node it is.

**torchair decision 2026-09-23 (final: NOT used).** Six bench rounds cleared
two GE blockers (`aten.isneginf` → `eq(-inf)`; complex RoPE purged from the
compiled region — `apply_rotary_emb_realfreq` + real freq table materialized
at model level, bitwise-identical, kept in the tree as 0922m) but hit a third:
an empty `[0]` FX-graph output (likely the NPU fused-attention's empty aux
output at dropout=0, saved for backward) that GE materializes as `[0,0,0,0]`.
More decisive than the bug chain was the expected gain: hostprobe5's cpu-wall
breakdown shows `syncopt` (grad allreduce + optimizer) at ~850ms of the
~1.3s/step, with fwd+bwd host time only ~175ms — torchair could only compress
a slice of the latter, far below the user's >20% adoption threshold. The hero
launches **eager** with `FOREACH=1 FUSED_ADAM=1` (validated 300/300 steps in
hostprobe5: eval/loss 1.998→1.004, peak 52.7GiB). Retained risk: if someone
revives torchair later, `src/pretrain/train.py`'s torchair path already wires
`set_real_rope` + single-stream-only compilation.

**Launch record 2026-09-23 (UTC+8):** `ascend-hero2-256p`, code 0922l,
eager, `expandable_segments:True`, T=600k, fault-tolerance 200 retries.
Launched WITHOUT FOREACH/FUSED_ADAM: hostprobe5's levered run showed ~725
samples/s clean (eval-probe overhead removed) vs smoke8's ~890 un-levered —
the levers target optimizer dispatch, but `syncopt` is allreduce-dominated,
so they aimed at the wrong segment and may even cost throughput. A/B deferred
to a post-launch checkpoint restart. Launcher v2 adds an endpoint watchdog
(kill the wedged torchrun group 180s after "reached stage endpoint" —
hostprobe5 showed the tbe shutdown can hang a finished job on 16 idle cards).
(First launch `ascend-hero-256p` 00:20 carried the levers and was stopped
~00:50 before meaningful progress; renamed for uniqueness after duplicate
同名 job rows proliferated via fault-tolerance retries.)

**Launch record 2026-09-23 ~01:05 (UTC+8):** `ascend-hero3-256p`, code
**0922n** — hero2 crashed 4 min in with a DataLoader worker bus error:
`/dev/shm` on the ascend pod is only 64MB and the default `file_system`
tensor-sharing strategy fills it (`torch_*` spill files). 0922n sets
`mp.set_sharing_strategy("file_descriptor")` (memfd, bypasses /dev/shm)
via new CLI arg `--dataloader_sharing_strategy file_descriptor`, and the
launcher raises `ulimit -n` to 1048576. Same config otherwise: eager,
`expandable_segments:True`, T=600k, no FOREACH/FUSED_ADAM. Expected
throughput ~890 samples/s (smoke8 baseline); first 2000 steps are the
verification window.

256p data-mix tweak (user decision 2026-09-22, code bundle 0922f): with the
step budget enlarged to 600k, D4 (world) weights in the **256p stage only**
are raised by +0.2× each (inat 1.2, megalith 1.4, pd12m/vintage/zimage/
relaion 1.2), moving D4 share from 50.9% to 55.5% after renormalization;
D1–D3 multipliers unchanged. 640p/896p mixes and all bucket plans untouched.

Migration status (2026-09-22 morning): 256p (104GB) already uploaded to the
private HF bridge repos from inko-patrol at ~55MB/s aggregate (6 parallel
`upload_folder`); 640p (502GB) uploading, 896p (574GB) next. sj-side pull
runs as a 16-card job (`ascend-dl-256p`, snapshot_download max_workers=24 via
hf-mirror) into `…/ky26021/artflow/precomputed_dataset` on sj-ssd3; the 2-card
notebook route was abandoned — the cann image's JupyterTerminal never comes
up. Remaining open items: (1) verify sj-side download integrity (row counts
vs qb), (2) hero launch script with OOM watchdog + auto-resume, (3) user
sign-off on this variant.
