# Ascend infrastructure qualification

The next pass targets the native `artflow-v2` architecture and the complete
[`configs/hero.toml`](../configs/hero.toml) recipe. Run it after the repository
cleanup checks pass. No hero has been launched by this cleanup.

## Existing evidence

The [short fresh stability check](archive/ascend_stability_stage1_0924.md) used
16×910B, eager blocks, NumPy DataLoader transport, foreach gradient/EMA
updates, batched Muon, standard AdamW, and optimizer-boundary DDP reduction.
It reported approximately 656.72 samples/s and 37.7 GiB peak memory with a
halved micro-batch plan and accumulation 2. Those are historical measurements,
not qualifications of the renamed architecture, revised Muon scaling, or
640p/896p candidates.

Keep the [Ascend bring-up findings](archive/ascend_probe_0921.md) for the actual
NPU failure mechanisms and software environment. The previous
[CUDA infrastructure record](archive/cuda_infra_pass_0924.md) remains useful
historical evidence; its command lines and implementation flags are retired.

## Next measurements

1. Exercise the exact consolidated model/optimizer and full-state recovery on
   Ascend. Check loss, gradient clipping, conditioning response metrics,
   EMA/live evaluation, sampler continuity, and checkpoint retention.
2. Measure the real step breakdown and long-caption/aspect memory tails at
   256p; qualify the bucket table and accumulation together at the intended
   global sample exposure. Change one execution lever per comparison.
3. Qualify 640p/896p plans and transitions using the same shared recipe. No
   inference-only benchmark or old CUDA bucket table is a production plan.
4. Preserve stage boundaries and global LR/caption progress on recovery.
   Report actual memory and wall time including monitoring/checkpoint overhead.

The launcher no longer installs dependencies, downloads mutable code bundles,
constructs TOML overrides, or reads environment hyperparameters. A short
experiment uses its own complete run file. Platform setup and credentials
remain separate from training settings.

The historical CUDA-only batch screen, transformer ceiling probe, backward
memory probe, old per-stage config generators, and completed normalization
probes were retired. Their measured conclusions remain in archived notes;
reproducing their commands requires the original source revision. Offline
bucket-boundary solving, plan resizing, exposure analysis and measurement
merging remain available. New Ascend measurements must use the current recipe
and launcher.

## Diagnostics

`--step_breakdown`, `--cpu_wall_profile`, `--log_shapes`, `--infra_metrics`,
`--trace_start`, `--trace_steps`, and `--record_identity` are operational
observability controls. They do not change training hyperparameters. NPU event
and memory accounting use the active device API. Prompt-grid evaluation
preserves the NPU RNG as well as the CPU RNG.

Bounded NPU traces use `torch_npu.profiler` and its trace handler, following the
[official 2.9.0 implementation](https://github.com/Ascend/pytorch/blob/v2.9.0/torch_npu/profiler/profiler.py).
The handler emits the NPU profiler's native trace/operator artifacts. Local
routing tests do not validate profiler operation on a live NPU; that check
belongs to this pass.
