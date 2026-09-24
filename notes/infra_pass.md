# Ascend infrastructure optimization and qualification

The next pass targets the native `artflow-v2` architecture and the complete
[`configs/hero.toml`](../configs/hero.toml) recipe. Run it after the repository
cleanup checks pass. No hero has been launched by this cleanup.

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

## Execution record — September 25

The first allocation, `ascend-infra-0925-a1`, used 16×910B with
128 CPU cores, 1024 GiB host memory, and 64 GiB shared memory. This allocation
is the comparison baseline; historical 64-core runs are not speedup evidence.
The prepared runtime reports torch 2.9.0+cu128, torch_npu 2.9.0, Accelerate
1.14.0, and Transformers 5.16.1 in the CANN 9.0.1 image.

Source snapshot: 69 files, archive SHA-256
`58d2e1f8083d06dae56b42e65cac91a16b205416cae795d9ac23e489aed0f404`.
The snapshot is isolated in `repo-infra-0925-a1`; its per-file manifest passed
remote verification. The complete `configs/infra-baseline.toml` in that
snapshot has SHA-256
`7321aa23ba4b22206d854ba37a4567f5d5ba304afd3c0a3e008b035857b1a258`.

First question: does the consolidated trainer execute the selected recipe
correctly, and where is its actual 256p runtime spent? The fresh 256p run stops
at 600 updates while retaining the 600k schedule horizon, 20k warmup, selected
optimizer values, and dataset mixture. It uses the earlier conservative
4–36 micro-batch table with accumulation 2 (plan SHA-256
`b009d87ecf17b5b764150a11b7827b52dbf89da4e5a2e3c75b0488184a0a19a6`).
Checkpoints every 200 updates exercise retention; no endpoint sample grid is
requested. These diagnostic deviations are explicit in the complete run file
and do not change the hero config. The preserved allocator setting is
`expandable_segments:True`; SwanLab is offline, with local records available.
Per-rank metrics and sample identities are enabled, plus a bounded NPU trace
at updates 61–63. Exclude tracing and its transient effects from throughput;
production checkpoint/grid overhead still needs its own accounting.

Preflight found all 256p dataset directories, only part of 640p, and no 896p
data on Ascend storage. Directory presence is not a completeness check.
Another agent owns the 640p download; do not duplicate or interfere with it.
The source inventory was read-only; no new data transfer was started here.
After the user's scope clarification, 896p staging is not part of this pass.

The first allocation failed before its first update: the consolidated trainer
passed `exit_mode="stop"` to the text encoder, whose accepted early-exit mode
is `stop_at_layer`. This was a cleanup regression, not an NPU arithmetic or
performance failure. The call is corrected and the training text-encoding
function is now covered with a small real Qwen model: retained features and
pooling match full-forward slicing, bucket padding is preserved, and the
unneeded transformer layers are verified not to execute. The full repository
suite passes: **782 tests**, including the new regression test. The failed job is
terminal; the corrected snapshot is submitted separately as
`ascend-infra-0925-a2` with a fresh output name.

### Consolidated 256p baseline and recovery test

The corrected snapshot has archive SHA-256
`72b7ecd5d339165ac1fe5fd8daeb5f47442ea087d19e1d4c7372d96ffb823987`;
its complete run config has SHA-256
`41496d141f0c2ff1ed54b35f7c657c48cbc6114da00dcc339a67879741b5701e`.
The actual device is 910B2C, with driver 25.0.rc1.1. The run reached update
500 without skipped updates. Updates **100–450** (351 updates, excluding
profiling and startup) measured **656.29 samples/s**, **1.43628 s/update**,
and **37.20 GiB** maximum rank-0 allocated memory. This is training-update
throughput, not a monitoring/checkpoint-inclusive rate or a later-caption
memory qualification. At update 500, loss was 1.10648, gradient norm 0.49656,
conditioning update RMS 0.00815 and conditioning response gain 0.66447.
The short window is still early in the configured 20k warmup; it does not
qualify peak-LR stability.

The bounded native trace (three updates) reports rank-0 computation 3.826 s,
non-overlapped communication 0.364 s, and free time 0.263 s within 4.454 s.
Communication totals 1.422 s but mostly overlaps computation. The dominant
kernel families include elementwise multiplication (515.5 ms), matrix
multiplication (501.9 ms), batched matrix multiplication (358.6 ms), addition
(180.8 ms), powers (162.5 ms), casts (137.3 ms), and reductions (136.7 ms).
These are kernel sums, not independently recoverable wall-time savings.
Large host `copy_` durations mostly wait for preceding queued device work:
even two-element copies take hundreds of milliseconds. They are not evidence
that transferring two numbers is the bandwidth bottleneck.
This device trace supersedes the archived host-time argument that compute
optimization has little opportunity because synchronization dominates; that
argument confused where the host waits with where the device spends its time.

First optimization candidate: fuse the model's 104 RMS normalizations per
micro-forward. The trace contains the corresponding 624 power operations
across six micro-forwards. PyTorch's BF16-input/FP32-gain RMSNorm expands into
FP32 arithmetic; a replacement must preserve that computation and final BF16
cast, including learned gains. Check the official NPU fused operation in FP32
against the reference forward and both gradients before a real training A/B.
The operator trace attributes 182.6 ms to RMSNorm forward, and another 249.7 ms
to its power/mean/reciprocal-square-root backward nodes, before counting their
multiplication gradients. This supports a targeted fusion probe. Microtiming
alone will not select the production implementation. References:
[PyTorch 2.9 RMSNorm implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/layer_norm.cpp)
and [Ascend fused RMSNorm API](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md).

For recovery qualification, the complete update-400 checkpoint was inspected:
both optimizer/scheduler states, EMA, model/config metadata, and all 16 ranks'
CPU/NPU RNG and sampler sidecars are present. Original metrics and per-rank
sample hashes were copied to `baseline-before-recovery` inside the run directory.
`ascend-infra-0925-a2` was deliberately stopped after update 500; terminal
status was verified before submitting `ascend-infra-0925-a3`. The latter uses
the same source/config pins, explicitly restores update 400 with exact
post-load state verification, and then runs the standalone RMSNorm probe after
training finishes. Exact post-load verification passed on all 16 ranks for
model, both optimizers, both schedulers, EMA and CPU/NPU RNG. All **1,600**
replayed rank/update records at updates 401–500 have the same sample/caption
hash and global sample count as the uninterrupted run. This verifies replay
identity, not bitwise equality of subsequent floating-point updates. The resumed
run completed all 200 updates through 600 and wrote a complete endpoint;
retention correctly left only checkpoints 400 and 600. Its shutdown then failed:
`accelerator.end_training()` destroys the process group, but the trainer called
`wait_for_everyone()` afterward. The first traceback identifies that final
barrier, not training or checkpoint writing. The barrier now precedes teardown;
clean termination was later verified by a5b and a6. The retained
container runs the RMSNorm probe separately because the failed launcher prevented
the originally chained probe from starting.

### RMSNorm candidate

The isolated 910B check passed 12 BF16-input/FP32-gain cases: widths 72 and
1152, contiguous and transposed sequence/head axes, and input scales
1e-4 / 1 / 1e4. Maximum relative L2 differences against PyTorch were
**1.26e-5 output**, **3.14e-5 input gradient**, and **3.03e-7 gain gradient**;
all results were finite. Forward/backward microtiming at 26×384 tokens was
1.082 → 0.319 ms for width 1152 and 1.255 → 0.527 ms for transposed width 72
(3.39× and 2.38× respectively). This justifies a training comparison, not
adoption by itself. Raw summary: `runs/infra-rms-probe-0925.json`.

The candidate uses the existing model module, preserves checkpoint keys and
FP32 learned gains, and leaves non-NPU execution with PyTorch. It adds no recipe
switch. The new normalization/block checks pass locally (16 passed; four NPU
cases skipped because this workstation has no Ascend device). The actual
candidate class also passed forward/input-gradient/gain-gradient checks on
910B for FP32 and BF16 at widths 72 and 1152; maximum relative L2 error was
5.84e-6. The full suite passes **786 tests**, with four NPU skips locally.
`ascend-infra-0925-a4` is running, with source archive SHA-256
`8f2d3e875f095c1a250da7f8a161a722b7f24e890828a2440345029a77265950`.
Its complete config SHA-256 is
`3329ece070bd59aab0f475f69250e97ef0c755fee3256afb9b4561ce03fce94f`.
The fresh run changes this normalization implementation only
within the training step; it also includes the endpoint teardown-order fix.
It retains the baseline seed, data, bucket table, accumulation and complete
numerical recipe. In the matched ordinary-training window, updates
100–450 (351 updates), baseline and candidate consumed the same global samples.
All 5,616 rank/update sample hashes match. Throughput was
**656.29 → 737.68 samples/s (+12.40%)**, update time **1.43628 → 1.27782 s**,
and rank-0 peak allocated memory **37.20 → 32.88 GiB**. The candidate's maximum
across all 16 ranks in this window was 33.34 GiB. Both allocations are
16×910B2C with the same driver/image/quota, on different nodes. The bounded
candidate trace reports 3.370 s computation, 0.342 s exposed communication and
0.298 s free time across 4.010 s. This agrees with the measured training gain;
the standalone 2.4–3.4× kernel gain is not the model gain. Endpoint health,
clean shutdown was subsequently fixed and verified below; the final
monitoring-inclusive estimate remains pending.

The next probe will double every 256p micro-batch (4–36 → 8–72) and reduce
accumulation 2 → 1, preserving the target sample exposure. This is a proposed
use of recovered memory, not a qualified production bucket table. The baseline
across 16 ranks sampled caption buckets through 1289 tokens but **never 2048**;
ordinary early-curriculum sampling is insufficient to qualify the longest
shapes. A targeted tail check remains required.

`ascend-infra-0925-a5` was submitted for that micro-batch comparison, with
source archive SHA-256
`4aacb5097303376f72efd1fbf02d6e39608ee0260aa8b8e70e4be79000999393`.
Its fresh complete config retains the 600k schedule horizon, 20k warmup and
600-update endpoint. The only new training-step lever is doubled micro-batches
with accumulation one; it also carries the separately measured evaluation
padding/batch-size fix below. Compare ordinary update throughput separately
from evaluation overhead, and compare actual data exposure rather than
assuming identical per-update sample counts under different batch grouping.

The remaining trace also shows repeated DDP buffer broadcasts. The registered
model buffers are fixed timestep/position tables; the trainer currently uses
DDP's default per-forward buffer synchronization. Investigate suppressing those
redundant broadcasts as a separate lever after the batch comparison, with an
explicit check that training does not mutate the buffers. HCCL duration includes
rank waiting, so the listed broadcast times are not directly additive savings.
The next candidate is `broadcast_buffers=False`, **not enabled in the trainer
or in submitted snapshots and not yet performance-qualified**.
Parameter initialization synchronization and gradient reduction remain enabled.
[PyTorch's implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/parallel/distributed.py)
also skips the initial buffer broadcast in this mode. A model-level test checks
that all registered buffers are the expected positional tables, identical
across independent seeds and unchanged by training with different shapes.

### Shutdown deadlock: inherited subprocess pipe

a4 completed all 600 updates, wrote checkpoints 400/600, and passed the final
barrier, but rank 0 hung inside SwanLab 0.10's `finish`. Live `py-spy` stacks
showed `Monitor.stop → Timer.join`, whose executor was blocked in
`subprocess.Popen._execute_child` reading its exec-error pipe. `/proc` inspection
proved the cause: the monitor read descriptor 134 (`pipe:[1734782724]`) still
had a write descriptor 135 open in persistent `pt_data_worker` PID 61341.
The `npu-smi` child had already exited. A concurrently forked DataLoader worker
inherited that transient CLOEXEC descriptor; fork without exec does not close
it. The monitor awaited EOF while shutdown kept the worker alive.

This is a process/descriptor-lifetime race, not failed training or an NPU
arithmetic fault. Do not hide it with a timeout or disable hardware monitoring.
DataLoader workers now explicitly use `spawn`, so exec closes unrelated
descriptors and workers do not inherit the initialized NPU runtime. The loader
regression test holds a transient pipe across real worker startup, requires EOF
while persistent workers remain alive, and also checks data and checkpoint-RNG
continuity. Live fresh-run and replay validation passed below. Evidence remains in
`runs/infra-a4-shutdown-{stack,child-stack,pipe}.txt`.
The local regression passes with spawned workers and deterministically fails
with `BlockingIOError` when the original fork context is substituted. This
tests the descriptor-lifetime mechanism, not just a flag value.

After preserving that evidence and the complete checkpoint, a4 was stopped to
release its allocation. The still-queued a5 was canceled before starting; its
replacement included the worker-start fix while retaining DDP's existing
buffer behavior, keeping the batch-size comparison isolated.
That replacement is `ascend-infra-0925-a5b`, source archive SHA-256
`2bbd3a3e58770c8eb32daaca753eaf303c2657c67ef456860c337e1beeca9660`,
with fresh output `ascend-infra-0925-batch2-256p` and otherwise the prepared
doubled-batch recipe. It ran on the same 910B2C node used for the original
baseline and completed 600 updates, including normal SwanLab shutdown and the
`[training-complete]` marker. This validates the spawned-worker fix in a live
run. Checkpoint retention left complete updates 400/600. The all-rank peak was
**52.61 GiB**; observed text buckets reached only 879, so long-tail memory
qualification remains necessary. The complete local suite passed **788 tests**,
with four NPU skips, before removing the rejected buffer-broadcast test; the old
multi-threaded-fork warnings in the loader tests are gone.

The doubled plan has SHA-256
`e66622da5e6b53240098fa97782f3a0529b929fecc3c188309a17e89ea6f3536`;
its complete config has SHA-256
`232a3721d65a9ee61a0939b4b197357b702861055653870676c506eaca4cb946`.
The ordinary comparison window (updates 100–450) measures **765.34 samples/s**,
**1.24012 s/update**, versus 737.68 samples/s for a4: a **3.75%** gain on top of
fused RMSNorm, **16.62%** over the original baseline. Sample counts differ by
0.69% because combining micro-batches changes bucket packing; this is not the
exact replay comparison
used for RMSNorm. The three-update profile reports 3.409 s computation,
0.147 s exposed communication and 0.216 s free time in a 3.772 s stage.
The gain so far comes mostly from less exposed overhead, not halved compute.

The buffer-broadcast comparison also verified recovery under spawned workers:
preserve uninterrupted per-rank evidence, restore update 400 with the same
complete config and exact state checks, then compare replayed sample identities
and steady timings after warmup. This provided both checks without another
independent fresh 600-update run.
After a5b reached platform status `job_succeeded`, a6 was submitted with source
archive SHA-256
`ed591b7b53ea22b8f962b15f5063c9fee5796a1d5ea0bf4fd0af6c849e81a2be`.
The comparison disabled DDP buffer broadcasts. A model test verified that every
registered buffer is a
fixed timestep/position table, identical across initialization seeds and
unchanged after training across shapes. Parameter initialization and gradient
collectives remained enabled. After replay, a separate 32-update diagnostic
started with forced 2,048-token padding and batch 8 across aspect ratios, using the real
trainer and the late caption preference. Its smaller sample exposure is for
memory qualification only, not a throughput or quality comparison.
The job then runs the isolated operator screens below on the same allocation.
The replay completed cleanly through update 600. Post-load verification passes
on all 16 ranks, including both optimizers/schedulers, EMA and CPU/NPU RNG;
all **3,200/3,200** replayed rank/update sample hashes and global sample counts
match the uninterrupted run. This also qualifies recovery with spawned workers.
Updates 450–600 measured **746.07 samples/s, 1.26159 s/update**, versus the
uninterrupted **753.30 samples/s, 1.24949 s/update**. The new trace contains zero
broadcast kernels, but that did not improve throughput (0.96% lower in this
window). **Reject the lever**: restore ordinary DDP buffer synchronization and
remove its candidate-only test. Collective duration included rank waiting;
removing it did not translate into recoverable end-to-end time. The code and
candidate-only test have been reverted locally. The tail diagnostic's complete
config SHA-256 is
`9865110fac418cede85fb46d23e695f703e8b23d0ec671c0490fce59ebd8829f`;
it still uses the a6 snapshot, so its DDP broadcasts remain disabled.

The tail diagnostic completed all 32 updates and shut down cleanly, saving a
complete update-32 checkpoint. All **512 rank/update records** use batch 8 and
2,048-token text padding. Latent shape coverage was 24×42 (53), 28×36 (170),
32×32 (87), 36×28 (115), and 42×24 (87). The all-rank peak was **49.10 GiB
allocated / 49.43 GiB reserved**. Together with the ordinary run's 52.61 GiB
maximum, this supports the doubled 256p plan with accumulation one for the
current eager/fused-RMS implementation. Recheck memory for subsequent kernel
changes that retain additional activations. This forces the DiT's padding
tail; it does not assert that every selected caption contains 2,048 actual
encoder tokens. Per-rank peaks outside rank 0 are cumulative, so their maxima
are valid capacity evidence but cannot be attributed to a particular update.

Further screening targets identified by the device trace:

- Feed-forward matrices have width `int(1152 * 2.67) = 3075`. Their forward
  and backward GEMMs dominate several kernel rows. Test padding computation
  to 3088 with exactly zero extra rows/columns, preserving parameter shapes
  and gradients; changing learned width would be an architecture change.
  **User clarification on September 25:** small model-configuration changes
  are allowed when worthwhile. Also measure a direct width of 3072 (three
  fewer FFN channels, about 0.1% of FFN capacity). Prefer the simpler aligned
  capacity if its real training gain warrants the tradeoff; report it as a
  capacity adjustment, not a mathematically equivalent execution change.
  Meta-device model counts are 532,766,716 versus 532,496,992 parameters:
  269,724 fewer, or about 0.05% of the full model.
- RoPE currently includes FP32 casts, strided complex views and layout copies.
  Test the [official interleaved rotary operator](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000141.html)
  for forward/gradient agreement, representative batch/head constraints and
  memory. Its backward saves inputs even when frequencies need no gradient,
  so a faster operator could still harm the bucket plan.
- Compiling only the feed-forward module avoids the known complex-RoPE and
  empty-attention-auxiliary GE failures. Test dynamic shapes and numerical
  agreement before considering a real-workload comparison; do not repeat the
  concluded whole-block experiment unchanged.

The isolated screens (`runs/infra-operator-probe-0925.{json,log}`) measured:

| FFN tokens | Width 3075 | Width 3072 | Speedup |
|---|---:|---:|---:|
| 19,228 | 10.95 ms | 8.05 ms | 1.36× |
| 19,720 | 11.39 ms | 8.41 ms | 1.35× |
| 18,432 | 10.80 ms | 7.68 ms | 1.41× |

These include forward/backward with BF16 inputs and FP32 weights under
autocast. They justify a real-training width comparison, not an overall gain
claim. Exact zero-padding to 3088 gave only 3–5% FFN improvement; forward and
parameter gradients matched, with input-gradient relative L2 below 4.3e-5.
The 3072 capacity candidate was selected for the a7 real-training comparison below.

Dynamic FFN-only compilation succeeded, including reuse at new batch/sequence
sizes after a 46 s initial compile. It was **19–23% slower** than eager at all
three measured shapes, so it will not enter training. Maximum observed relative
L2 was 0.00153 for outputs and 0.00345 for gradients. Raw evidence:
`runs/infra-ffn-compile-probe-0925.{json,log}`. Successful compilation is not
performance evidence; this result does not rule out fusion of larger regions
or custom pointwise kernels.

Stock fused interleaved RoPE matched the reference output/gradient at supported
shapes, but its backward rejects batch 72 × 16 heads: `B * N should smaller
than 1000` in `rope_interleaved_grad_tiling.cpp:391`. It also retains the full
input, increasing the measured operator peak by about 99 MiB at batch 8,
sequence 2304. The next diagnostic uses the inverse rotation for the input
gradient, saving only fixed frequencies; this could avoid both limitations.

That inverse-backward diagnostic matched output and input gradient exactly at
all three shapes, including batch 72, and retained only 0.15–1.27 MiB of
frequency data. It was 10.7% slower at B26/S384, about 1% faster at B72/S275,
and 1.33× faster at B8/S2304. This is not enough evidence for a useful gain on
the common early-caption workload; do not ship it alone. These exploratory
timings used the shell's default OMP setting, not the trainer's explicit one.
Raw evidence: `runs/infra-rope-backward-probe-0925.{json,log}`.

A separate manual attention backward omitted unused empty auxiliaries and
matched eager outputs and Q/K/V gradients exactly at B4/H16/S320/D72. Its
first harness attempt incorrectly unpacked four values from the installed
five-output gradient API; after correcting that, GE backward conversion failed
with `The length of ge_outputs must be equal to meta_outputs` at
`npu_fusion_attention_grad`. The full traceback is in
`runs/infra-attention-compile-probe-0925.{json,log}`. This is a new converter/API
compatibility issue to inspect, not proof that the historical empty-auxiliary
failure is solved or that whole-block compilation is fast.

After these results were preserved, the retained a6 allocation was released.
`ascend-infra-0925-a7` was submitted with source archive SHA-256
`3e8810c28e77da2e4ad0019c65553e9e5c5f2a204a18e695e28d005f82ca4b6d`.
Its fresh 600-update recipe changes FFN expansion to `8/3` (width 3072), keeping
the doubled plan, accumulation one, and ordinary DDP buffer behavior. An
explicit update-600 prompt grid exercises the real monitoring path after the
training comparison. The run completed 600 updates and its 48-prompt/12-grid endpoint, exited with
code zero, and retained complete checkpoints 400/600 while pruning 200.
It ran on 910B2C node 339 with the same allocation/runtime. Complete config
SHA-256: `c88337e80d111704c574abd70235c7540e2be26cd71bc3ded03e66d840704a9f`.

The full steady window (updates 100–450) measured **843.29 samples/s,
1.125486 s/update**, versus 765.34 samples/s and 1.240115 s/update for a5b.
Both processed exactly 333,138 samples; **9,600/9,600 rank/update identities
and global sample counts matched** across the complete 600-update runs.
Width alignment adds **10.19% throughput**, with **28.49% total improvement**
relative to a2's original 656.29 samples/s. The three-update kernel trace
supports the mechanism despite the different physical node: matrix-multiply
sums fell 603.3→453.8 ms, batched Muon GEMMs 358.3→245.6 ms, and addmm
210.4→174.0 ms, while attention/RMSNorm timings stayed nearly unchanged.
This is measured 256p training throughput, not a curriculum-wide speedup.

All-rank training peak was **52.591 GiB allocated**. At update 500, loss was
1.11781, gradient norm 0.50919, conditioning update RMS 0.008046 and response
gain 0.67126; EMA fixed-probe loss was 1.18195. EMA/live evaluation took 54.28 s.
At 600, loss was 1.03261, gradient norm 0.43073, conditioning update RMS
0.007846 and gain 0.74906. These short warmup checks show no material health
regression; they do not prove peak-LR or long-run stability.

**Adopted:** width 3072 (`mlp_ratio = 8/3`) and the qualified doubled 256p plan
with accumulation one. The installed plan is
`bucket_plans/ascend/hero-256p-k20.json`, SHA-256
`e66622da5e6b53240098fa97782f3a0529b929fecc3c188309a17e89ea6f3536`.
Later-stage plans remain unqualified. No hero was launched.
Original a7 evidence is preserved under
`runs/ascend-infra-0925-ffn3072-256p/before-monitor-replay/`; the subsequent
checkpoint-600 monitor replay must not be included in its throughput window.

### Monitoring precision correction

A concurrent sampling audit identified that Euler/Heun
rounded model timesteps to the BF16 latent dtype and accumulated solver state
in BF16. Source inspection confirms both paths; with factor-1000 features,
the timestep error is consequential even when sampling completes successfully.
The solver now promotes FP16/BF16 state and velocity arithmetic to FP32 and
always supplies FP32 times, while retaining explicitly requested FP64 state.
Model forwards still use autocast and sampling wrappers already cast to BF16
at VAE decode. Regression tests cover direct Euler/Heun steps, fifty small
increments, precision before Heun's velocity addition, and intermediate states.
The focused solver/grid/KID tests pass **55 tests**; the complete suite passes
**795 tests**, with four NPU skips.

A7's immutable training snapshot predates this correction. A separate source
snapshot, `repo-infra-0925-a7-monitor`, resumed its complete checkpoint 600
without performing another training update. All 16 ranks verified exact model,
optimizer, scheduler, EMA and CPU/NPU RNG restoration. The corrected real
48-prompt, 50-step monitoring path produced all 12 grids and exited cleanly.
Its manifest now distinguishes BF16 model precision from FP32 solver/time
precision. Source-manifest SHA-256:
`38c1a4e14addfbebd9a29217bba34743bb98eab678236cff03df2e9281648147`.
Raw launcher log: `runs/infra-a7-monitor-replay-0925.log`.
The fixed-loss probe already uses FP32 times. Other audit findings remain
outside this change. After the capacity/plan and manifest updates, the complete
suite remains green: 795 passed, four skipped (43.56 s; run with local loopback
access for distributed tests).

### Custom-kernel toolchain preparation

Triton-Ascend 3.2.1 (CPython 3.12, x86_64 wheel SHA-256
`60bebbd2c24ebd2fe8105af10f0224b338f0c439b6dc0bb93a19ec8eb121ca3c`)
was installed with base Triton 3.2.0 and pybind11 2.13.6 in an isolated
`infra-build/triton-3.2.1/pylibs` directory on shared storage. The training
environment is unchanged. Import, Ascend backend discovery, and initial custom-kernel execution pass. This follows the [3.2.1 release compatibility
matrix](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.1), which
includes Python 3.12 and CANN 9.0, rather than the stale Python 3.9–3.11 limit
in the installation guide. The initial tiled RMSNorm kernel runs on the installed torch/torch_npu 2.9.0
stack. Small width-72 cases matched output/input gradient exactly; gain-gradient
relative L2 was below 2e-7. A 16-row × padded-width-2048 tile failed compilation
with a concrete unified-buffer overflow (328 KiB requested versus 192 KiB).
Reduced tiles ran representative shapes but were slower than native fused
RMSNorm: the best B72/H16/S275/D72 tile took 1.155 versus 1.041 ms; B26/S384/C1152
took 0.418 versus 0.312 ms. Output/input-gradient relative L2 remained below
1.3e-5; gain gradients below 1.6e-6. Eight-row wide tiles still overflowed the
unified buffer. Reject this implementation; native fused RMSNorm remains the
production path. Raw evidence: `runs/infra-triton-rms-probe{,2}-0925.{json,log}`.
The toolchain works, but this mapping does not improve the measured operation.

### Evaluation padding and explicit batch size

Inspection found that every latent-shape batch carried the widest text padding
in the entire fixed probe. Training also created `EvalLossProbe` without passing
the explicit `eval.batch_size`: it used the class default 64 while the recipe
said 8. An isolated real-probe comparison on the trained update-600 model
measured each effect separately:

| Probe execution | Batch | Wall time | Max loss-metric difference |
|---|---:|---:|---:|
| Existing full-probe padding | 64 | 65.19 s | reference |
| Trim unused padding per batch | 64 | 29.15 s | 7.96e-6 |
| Trim; use configured batch size | 8 | 31.44 s | 9.40e-6 |

All three used exactly the same 864 samples, caption-band membership, paired
images, fixed noise and four timesteps. The full probe text width was 721;
trimming changes only masked positions. This is a **2.07× evaluation speedup**
with the actual configured batch size, not a 2.07× training/curriculum gain.
Raw summary: `runs/infra-eval-probe-0925.json`. The padding change and explicit
batch-size wiring are included in a5b and a6; they were absent from a4 so its
normalization comparison stays isolated. A5b's combined EMA/live evaluation at
update 500 took **57.48 s**, compared with 72.31 s for startup evaluation.
The retained
a3 allocation was released after the comparisons completed. Subsequent trainer
snapshots also record `perf/eval_seconds`, including the EMA and live passes,
to account for monitoring overhead in the final end-to-end estimate.
The complete local suite passes **787 tests**, with four NPU-only tests skipped;
the new padding regression checks a model with active attention and entirely
empty caption batches against the padded prediction.

## Next measurements

1. Exercise the exact consolidated model/optimizer and full-state recovery on
   Ascend. Check loss, gradient clipping, conditioning response metrics,
   EMA/live evaluation, sampler continuity, and checkpoint retention.
2. Measure the real step breakdown and long-caption/aspect memory tails at
   256p; qualify the bucket table and accumulation together at the intended
   global sample exposure. Change one execution lever per comparison.
3. Use optional 640p probes when they materially inform an optimization. Record
   what remains unqualified for later stages; an old CUDA bucket table is not
   a production plan. Higher-resolution validation does not gate this pass.
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

### Next pointwise screen

The aligned a7 trace still spends 292 ms in elementwise Mul, 33 ms in its
implicit slices, and 32 ms in SiLU backward over three updates. The gated FFN
already stores gate/linear projections contiguously, matching the
[official native SwiGLU operator](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001219.html).
An isolated forward/gradient and whole-FFN timing screen will decide whether
this fusion warrants a real-training comparison. It changes intermediate BF16
rounding, so numerical agreement is measured explicitly. No recipe change is
justified by the operator's existence alone.
