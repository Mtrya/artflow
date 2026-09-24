# Ascend infrastructure optimization and qualification

The September 25 pass qualified native `artflow-v2` and the 256p stage of
[`configs/hero.toml`](../configs/hero.toml) on 16×910B2C. The subsequently launched hero is tracked in the
[maintenance handoff](pretrain_hero_handoff.md).

## Current result — September 25

Commit `72b287c` contains the qualified 256p plan (micro-batches 8–72,
accumulation one), FFN width 3072, native fused RMSNorm, evaluation corrections,
and recovery/shutdown fixes. The complete suite passed 795 tests, with four
NPU skips; device-specific checks and full-state recovery also passed live.

| Real training variant | Samples/s | Seconds/update | Incremental throughput gain |
|---|---:|---:|---:|
| Consolidated baseline | 656.29 | 1.43628 | reference |
| Native fused RMSNorm | 737.68 | 1.27782 | 12.40% |
| Doubled micro-batches, accumulation one | 765.34 | 1.24012 | 3.75% |
| Aligned FFN width 3072 | 843.29 | 1.12549 | 10.19% |
| Native fused SwiGLU | 878.91 | 1.07987 | 4.22% |

These use updates 100–450 on the early-caption workload. Total sample
throughput improved **33.92%**; time per update fell **24.81%**. Batch regrouping
slightly changes exposure per update, so these percentages have different
denominators. They exclude checkpoint/grid work and do not establish a
curriculum-wide speedup. Qualification includes all 16 ranks, the 2048-token
padded tail, complete-state/RNG recovery, retention, and corrected monitoring.

Native SwiGLU is qualified and adopted after a matched-node 600-update
comparison. The full local suite passes **798 tests, six NPU skips** (43.42 s);
all four CPU/NPU SwiGLU forward/gradient cases passed on the device. The
production launcher binds the qualified allocator and host-thread policies.
Final checkpoint-400 recovery passed exact state/RNG checks on all 16 ranks;
all 3,200 rank/update sample identities matched through update 600. Corrected
48-prompt monitoring, checkpoint retention and clean launcher shutdown passed.
The test allocation was stopped after verification. **This pass is closed**
under the agreed stopping rule; the 2× target was not reached. Short-run health
is qualified, not long-run or peak-LR stability. The hero launch is a separate
record in the maintenance handoff.

## Goal and stopping rule — agreed September 25, 2026

Target approximately **2× end-to-end speedup across the complete
256p → 640p → 896p curriculum**, relative to a measured baseline of the
consolidated Ascend implementation on the same hardware allocation. Preserve
the selected model, optimizer mathematics, data exposure, numerical health,
and recovery behavior. The working expectation is substantial untapped
performance; revise the target when real profiles and measurements justify
it, not because the easy optimizations are exhausted.
The user subsequently allowed small model-capacity/configuration adjustments
when a measured gain justifies them (starting with FFN width alignment).
Document those tradeoffs explicitly; do not present them as equivalent kernels.

There is **no fixed wall-clock or GPU/NPU-hour budget for this pass**.
Prioritize opportunities by their likely curriculum-wide savings and the
insight an experiment provides. Continue while profiling supports a credible,
material opportunity. Reassess and stop when remaining opportunities are small
or speculative, with the remaining bottlenecks and estimated headroom recorded.
Missing 2× requires an evidence-based explanation; reaching it does not by
itself end the pass or excuse correctness problems. This is a stopping rule,
not a requirement to prove global optimality.

**Deep low-level optimization is explicitly encouraged.** Investigate
operator fusion, tensor layouts, memory movement, launch overhead, and
communication scheduling where traces identify recoverable costs. Writing
our own NPU kernels is encouraged when it is the appropriate way to address
a measured bottleneck; it need not wait until every higher-level option has
been tried. Engineering depth alone is not a reason to reject an opportunity.
Adoption still requires material end-to-end benefit and numerical checks
appropriate to the affected forward, backward, or optimizer computation.

## Measurement and experiment policy

- **Use 256p as the main real-workload baseline and qualification target.**
  User clarification on September 25: 640p checks are optional when they add
  insight into a hotspot or resolution-sensitive change; neither 640p nor
  896p validation is a prerequisite for completing this pass. Do not run every
  candidate at every resolution. Select shapes that distinguish the mechanism
  and expand coverage when the result will change a decision. No exhaustive
  A/B matrix or separate long validation stage is required.
- Before each experiment, state the decision it can change and the evidence
  it should produce. Profile data, forward, backward, synchronization, and
  optimizer costs, including host/device overlap and rank imbalance. Change
  one execution lever per comparison, but collect multiple useful metrics
  from that run. Existing evidence and applicable official implementations
  should avoid redundant experiments. Microbenchmarks can reject a candidate
  or justify a training probe; they cannot establish production gains.
- Compare matched hardware, model, data/caption distributions, and effective
  batch/sample exposure. Micro-batch size and accumulation may be retuned
  together without silently changing the training recipe. Include long-caption
  and aspect-ratio tails and representative points in the caption schedule.
  Separate bounded profiling from ordinary throughput measurements so profiler
  synchronization overhead does not become the baseline.
- Report throughput and memory for measured stages, then project total time:
  `T = 450000*t256 + 120000*t640 + 30000*t896 + overhead`, where each `t` is
  representative training time per optimizer update. Overhead includes the
  configured evaluation, telemetry, checkpoints, startup/compilation, and
  transitions, counted once. The target is `T_baseline / T_final ≈ 2`, not
  an arithmetic average of stage speedups. Label projections and uncertainty;
  measuring the target does not require running the entire 600k-step curriculum.
  For unmeasured stages, expose the assumptions and sensitivity of the estimate;
  do not claim a measured curriculum-wide speedup from 256p results alone.
- Preserve training semantics. Ordinary floating-point differences need not
  be bitwise identical, but kernel/reduction/precision changes require checks
  against the reference and short numerical-health measurements proportional
  to their risk. Altering model capacity, optimizer rules, exposure, or
  monitoring coverage is a separate recipe decision, not an infra speedup.

## Temporary controls and final implementation

Optimization methods may be explicit tunables during investigation, in the
same complete run config and recorded in SwanLab with code/environment
identity. Do not introduce environment/CLI recipe overrides or config layers.
Once the execution recipe is settled, make the winning mechanisms native,
including measured stage/shape-dependent selection rules where needed.
Delete temporary switches, losing implementations, and one-off probes after
recording their evidence. Genuine training tunables, including bucket plans
and accumulation, remain explicit.

The pass closes with measured gains and remaining limits documented, a qualified
256p bucket plan, proportional numerical-health checks, tested full-state
recovery, and a consolidated production implementation. Use realistic shape
variation to check memory headroom and drift; a short run surviving a few
convenient buckets is insufficient evidence. Unmeasured later-stage plans and
transitions remain explicit follow-ups before those stages launch, without
blocking this pass or the 256p hero. The hero supplies longer stability evidence.

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

## Completed evidence

The [archived execution record](archive/ascend_infra_pass_0925.md) contains source/config
hashes, numerical checks, full-state recovery, kernel screens, raw artifact
locations, and the shutdown-race diagnosis. It is the experiment archive;
one-off probe scripts are retired. The selected implementation uses no Triton
runtime dependency and no compiler/converter monkey patches.

## Stopping assessment and projection

The roughly 2× target is not achieved. Matching half the original training
update time would require another 0.362 s/update, or 33.50% of the selected
recipe's measured time. Even eliminating every exposed communication/free
interval in the final trace would remove only 15.44% of that trace; real data
and synchronization cannot be eliminated. Substantial compute changes would
therefore still be required.

The measured candidates cover normalization, activation fusion, batch packing,
FFN alignment, DDP buffers, optimizer algebra, graph compilation and several
low-level rotary/normalization kernel mappings. The surviving implementation
uses native kernels with verified gains. No tested losing candidate has a
credible material end-to-end win. Further combined QK-normalization/rotary
fusion or attention-layout work could help, but the current custom mappings
are slower and compiled attention fails device execution. They are research
leads, not demonstrated unused speedups. Changing attention head width/count
would also change the architecture and needs its own quality evidence; it is
not implied by the nearly capacity-neutral FFN alignment decision. Final
recovery qualification passed, so this pass closes under the agreed stopping
rule. This does not claim a global optimum or exclude future kernel work.
Reopen on a concrete faster operator or a materially different workload/profile.

For the selected recipe, checkpoint barriers average 4.203 s, EMA/live loss
at update 500 costs 52.658 s, and the corrected 48-prompt panel costs about
31.403 s (checkpoint-record/panel-manifest mtimes; approximate). At configured
intervals these add 0.00210, 0.10532 and 0.01256 s/update. Thus **about
1.200 s/update** is an early-caption 256p projection including those recurring
costs, versus 1.07987 s for training. Initial setup/evaluation, stage transitions,
endpoint KID and caption drift are additional. Extrapolating 450k updates at
that early workload gives **150 hours**, not a measured full-stage duration.
Training-only savings against the original rate would be about 44.6 hours over
450k updates if both rates persisted.

The isolated old evaluation pass took 65.19 s; doubling it estimates 130.38 s
for EMA/live. Assuming that estimate and equal checkpoint/grid costs gives
about 1.712 s/update before versus 1.200 after (~1.43×). This is an overhead
projection, not a measured end-to-end A/B: the old pair/grid costs were not
measured together. Report the measured 1.339× sample throughput separately.

For the complete curriculum, use
`T = 450000*t256 + 120000*t640 + 30000*t896 + one-time overhead`.
Here each stage time includes its recurring monitoring/checkpoints. Only the
early 256p estimate is available. For example, with t256=1.20, later-stage
assumptions (t640,t896)=(2,4), (4,8), or (8,16) seconds give **250, 350, or
550 hours**, respectively, before one-time overhead and caption drift. These
are sensitivity examples, not later-stage forecasts. No measured full-
curriculum speedup or cost guarantee follows. Later bucket plans and stage
transitions must be qualified before those stages launch, without blocking
this pass or the 256p hero.
