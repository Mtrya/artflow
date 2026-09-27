# Ascend pretraining infrastructure

Updated September 27, 2026. This record captures the qualified 256p execution
path and the measurements behind it. The recipe is in
[hero_recipe.md](hero_recipe.md); current operation and recovery are in
[ascend_pretraining.md](ascend_pretraining.md).

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
in code. Later resolutions need their own measurements.

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
rates remain to be measured.

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
