# Ascend pretraining infrastructure

Updated September 27, 2026. This record captures the Ascend execution path,
resolution-specific bucket plans and their measurements. The recipe is in
[hero_recipe.md](hero_recipe.md); current operation and recovery are in
[ascend_pretraining.md](archive/ascend_pretraining.md).

## Measured improvements

September 25 comparisons used 16×910B2C on matched allocations, the selected
variable-aspect/caption workload and updates 100–450. Rates exclude startup,
profiling and periodic evaluation/checkpoint work. They describe early-256p
caption lengths. Match actual sample identities and account for samples per
update when comparing changes to micro-batches/accumulation.

| Execution path | Samples/s | Seconds/update | Incremental throughput gain |
|---|---:|---:|---:|
| Baseline | 656.29 | 1.43628 | — |
| Native fused RMSNorm | 737.68 | 1.27782 | 12.40% |
| Larger micro-batches, accumulation 1 | 765.34 | 1.24012 | 3.75% |
| Aligned 3072-wide FFN | 843.29 | 1.12549 | 10.19% |
| Native SwiGLU | 878.91 | 1.07987 | 4.22% |

The combined gain is **33.92% throughput**, with **24.81% lower update time**.
FFN alignment reduces total parameters by approximately 0.05%. The final
SwiGLU comparison used 600 updates and matched 9,600 sample identities;
short-run loss/gradient distributions were comparable, parameter layout was
unchanged, and peak allocation fell 52.591 → 49.678 GiB.

The selected 256p bucket plan uses micro-batches 8–72 and accumulation 1.
Real-workload and padded 2,048-token tail probes peaked at 52.61 and 49.10 GiB
allocated. Native RMSNorm/SwiGLU and launch policies are settled mechanisms
in code. The later-resolution measurements are recorded below.

## Later-resolution plans — September 27

The final 640p/896p mixtures contain 1,304,540/794,599 eligible rows across
17/14 sources. All source caption sidecars passed dataset validation, and all
five latent aspect shapes were inventoried. No source weights were changed.

Calibration retained the FP32 model, EMA, frozen BF16 Qwen k20, initialized
Muon/AdamW states and gradients, plus a model-sized DDP-buffer reserve. The
52 GiB full-residency ceiling replaces the earlier DiT-only budgets; those
earlier candidates OOMed before their first update. Fitted memory residuals
were at most 0.60 GiB. Bounds use each stage's caption-progress window
(.75–.95 / .95–1.0), 20 buckets per aspect and the 2,048-token cap.

Each resolution compared time-balanced and memory-sized buckets on the same
16×910B2C allocation, with 64 native training updates per candidate and the
first 20 excluded. Source weights, initialization, optimizer and accumulation
were held fixed. Different batch sizes change sample grouping, so this is a
samples-per-second comparison, not a paired loss-quality experiment.

| Stage | Aligned samples/s | Selected memory-sized samples/s | Samples/update | Seconds/update | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| 640p, accumulation 3 | 169.78 | **173.47** | 531.66 | 3.0648 | 51.18 / 51.59 |
| 896p, accumulation 4 | 83.30 | **83.90** | 375.43 | 4.4747 | 51.74 / 52.67 |

Time alignment needed a measured gain of at least 3% with a positive lower
95% block-bootstrap bound to be retained. Its measured changes were −2.13%
(interval −3.04% to −1.20%) and −0.72% (−1.61% to +0.13%); neither qualifies.
These intervals describe variation within the short comparison phases, not
future-node reproducibility. Earlier 128-update aligned bring-ups measured
169.12/83.31 samples/s, consistent with the repeated baselines. Accumulation
reduces the significance of the slowest individual micro-batch, which the
planner's isolated time-alignment prediction does not model.

The selected ranges are 5–12 / 3–6 samples per rank. Harmonic emitted-batch
estimates are 11.002774 / 5.835881, giving 528.13 / 373.50 samples per update
with accumulation 3/4. A forced-shape check covered **all 280 distinct
aspect/length/batch combinations** across both candidates, three forward/backward
repeats each, with finite loss and gradient checks. Peak allocations were
51.57/52.03 GiB against 60.96 GiB device capacity. This complements the native
mixed-stream tests; it is not a long-run allocator or stability guarantee.

The throughput probes used the same mature 180k model weights with fresh
optimizers, fixed caption progress .85/.975, the normal 600k schedule, and
monitoring disabled. Rates exclude startup, checkpoint and periodic evaluation
costs. They qualify execution; actual 450k/570k model transfer and training
stability remain observations for the hero at its normal monitoring cadence.

Selected plan SHA-256 values:

- 640p: `71f9599966773ba54e20bac5daf8de8a52c4f60dc9c0305ec9c8dbc8c168b066`
- 896p: `9cc148738b14f40a6fa34b53843559db5173e936b8275dbe9a681493b3aed824`

Detailed calibration, comparison and tail evidence is archived locally under
`notes/archive/bucket_qualification_0927*`. Platform artifact locations are in
`INSPIRE.md`.

A bounded native curriculum then exercised **256p→640p→896p**, using
diagnostic endpoints 2/16 and stopping at 40 while retaining the 600k scheduler
horizon. The first checkpoint carried the deployed hero's legacy future-stage
fields; the migration tool independently copied it and amended six future
data/bucket/accumulation fields. The other 56 inventoried artifacts were
byte-identical. Both transitions and the 896p step-32 recovery verified exact
model, optimizers, schedulers, EMA and RNG restoration on **all 16 ranks**.
The replay's eight updates matched all **128 rank/update identity hashes**
(3,026 global samples); endpoint sampler and Python/NumPy/CPU-Torch/NPU RNG
states also matched on every rank. Maximum absolute paired loss difference
was 2.27e-5; post-update floating-point equality is not required.

Monitoring passed at both higher resolutions: batch-8 live/EMA loss probes,
health/stability telemetry, complete checkpoints, and the full 48-prompt,
50-ODE-step image panels. The bounded check requested 64 cases per caption
band and obtained 195/170 total cases; long bands were sparse, including zero
cases above 1,024 tokens. Steady live+EMA evaluation took 45.42/84.15s for
those reduced panels, **not the production 512-per-band request**. Checkpoint
saves took about 4.27/4.0s. Long-caption memory coverage comes from the forced
shapes above, not these held-out panels.

All diagnostic updates applied. Four early 640p updates clipped (maximum
pre-clip norm 1.2834), as did the first two 896p updates (maximum 1.8710);
later updates fell below 1, and the recovery segment had no clips. These
brief transfer transients are recorded rather than presented as spike-free
stability evidence. The mature-model/fresh-optimizer test is not the actual
450k/570k continuation and does not justify changing its optimizer recipe.

The prepared production recipe also passed read-only amendment preflight
against the real 184k checkpoint. The active 256p deployment is unchanged.
Future-stage activation follows the explicit 450k handoff in
[Ascend pretraining](archive/ascend_pretraining.md).

## Recovery and monitoring qualification

The 400→600 recovery test restored model, both optimizers, schedulers, EMA
and per-rank RNG exactly on all 16 ranks. The subsequent 3,200 sample identities
matched; floating-point training continuation was not bit-exact. Checkpoint
retention, complete saves at 400/600, the 48-prompt/12-grid panel and clean
shutdown were exercised.

Evaluation padding was reduced while preserving the panel. On the same
864 cases, loss-evaluation time changed 65.19s (batch 64 with padding) →
29.15s (trimmed batch 64) → 31.44s (trimmed configured batch 8), with maximum
metric difference 9.4e-6. Production uses the explicit batch size 8.

| Recurring work | Measured time | Cadence |
|---|---:|---:|
| EMA + live loss evaluation | 52.658s | 500 updates |
| Checkpoint | 4.203s | 2,000 updates |
| Image grids | 31.403s | 2,500 updates |

These costs raise the early-256p estimate from 1.07987 to about 1.200s/update.
At that workload, 450k updates project to about 150 hours plus one-time costs.
Actual duration depends on caption progression, sample grouping and system
conditions. Total curriculum time is
`450k × t256 + 120k × t640 + 30k × t896 + one-time costs`; higher-resolution
compute-only estimates are now approximately 102.2 hours for 640p and 37.3
hours for 896p using the measured rates above. Add their monitoring costs;
these estimates are not a wall-clock completion promise.

## Mechanisms and resolved failures

- Native Ascend attention uses a `[B,1,Sq,Sk]` boolean mask with **True meaning
  drop**, and an explicit `head_dim**-0.5` scale. The bring-up probe measured
  approximately 2.81 GiB extra saved tensors for generic float-mask SDPA versus
  0.06 GiB for the native path. Preserve mask and scale semantics in new callers.
- DataLoader uses spawn. The diagnosed shutdown hang was a subprocess errpipe
  inherited by forked loader workers during a CLOEXEC race. A teardown barrier
  precedes `end_training` so distributed workers finish coherently.
- The launcher fixes expandable allocator segments and one OpenMP thread per
  worker. HCCL telemetry avoids unsupported double-precision reductions.
- Read the training log before diagnosing a hang. An early alleged autocast
  deadlock was a slow evaluation pass; elapsed platform-log silence alone did
  not identify the failure.

## Optimization boundary

The final three-update trace attributed 84.56% to compute, 7.22% to exposed
communication and 8.22% to free gaps. Eliminating all non-compute time would
remove only 15.44%; reaching the original 2× target would require a further
compute improvement. Overlapping kernel sums do not add into wall time.

Measured candidates explain why the September 25 pass closed:

| Candidate | Evidence / disposition |
|---|---|
| Compiled attention | Converter/output handling and later device bounds failures in the tested stack; native attention remains selected. |
| Custom RoPE kernel | 1.167/2.304/2.140ms versus native 0.478/1.074/1.327ms at the tested shapes; rejected. |
| Batched `baddbmm` Muon | 40–59% slower in the measured probe and numerically different; retained the existing optimizer. |
| DDP buffer changes | No material workload gain; retained the qualified path. |

Further optimization starts from a measured bottleneck and changes one
variable at a time. Adopt only a material end-to-end gain on the real workload,
then qualify correctness, memory and recovery. The current hero supplies
longer stability evidence; a short throughput test cannot establish it.
Operator fusion, custom NPU kernels, layouts and communication scheduling are
all eligible when profiling identifies recoverable cost. Select representative
shapes and caption/aspect tails that resolve the decision; keep bounded profiling
separate from ordinary throughput measurement.
