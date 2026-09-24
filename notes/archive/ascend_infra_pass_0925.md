# Ascend infra evidence — September 25, 2026

Closed experiment record. Current implementation, limits and stopping decision
are summarized in [the living infra note](../infra_pass.md). Raw run paths below
are relative to the Ascend ArtFlow storage root described in `INSPIRE.md`.

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
monitoring-inclusive accounting is provided in the final assessment.

The subsequent a5b probe doubled every 256p micro-batch (4–36 → 8–72) and reduced
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

The trace also motivated an isolated DDP buffer-broadcast comparison. The
registered model buffers are fixed timestep/position tables, and a temporary
model-level test confirmed seed independence and immutability during training.
The candidate disabled per-forward and initial buffer synchronization while
retaining parameter initialization synchronization and gradient reduction,
following [PyTorch's implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/parallel/distributed.py).
A6 subsequently found no end-to-end gain, so the flag and candidate-only test
were removed; ordinary DDP behavior remains selected. HCCL durations include
rank waiting and cannot be interpreted as directly additive savings.

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
sequence 2304. The subsequent diagnostic used the inverse rotation for the input
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
At that change point, the complete local suite passed **787 tests**, with four NPU-only tests skipped;
the new padding regression checks a model with active attention and entirely
empty caption batches against the padded prediction.

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

### Adopted native SwiGLU

The aligned a7 trace spent 292 ms in elementwise Mul, 33 ms in its implicit
slices, and 32 ms in SiLU backward over three updates. The gated FFN already
stores gate/linear projections contiguously, matching the
[official native SwiGLU operator](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001219.html).
The isolated screen passed FP32/BF16 at activation scales 1e-3, 1, 10 and 1000.
FP32 relative L2 stayed below 5.9e-8; BF16 output/input-gradient differences
stayed below 0.0030. Whole-FFN output/parameter-gradient differences were at
most 0.00363. Fusion removes intermediate BF16 rounding; it is not bitwise
identical. Forward/backward times were 3.691→3.070 ms (B26/S384), 8.506→6.589 ms
(B72/S275), and 7.676→5.998 ms (B8/S2304). Raw operator evidence:
`runs/infra-swiglu-probe-0925.{json,log}`.

A8 ran on a7's retained node 339, under platform job `ascend-infra-0925-a7`,
as `ascend-infra-0925-swiglu-256p`. It changed only SwiGLU in the training step;
the separately qualified solver correction affected endpoint monitoring.
Source: `repo-infra-0925-a8`, source-manifest SHA-256
`61f55af902a2ca0e7f83e38c1a8337df8d9c82ab78940797770858f28525cf7c`;
complete fresh 600-update config SHA-256
`adb114b6861c98df4d6801c166e031611e7f93b8ee2609f633fa6df4b9644440`.
Launcher log: `runs/infra-a8-launch-0925.log`.

Updates 100–450 measured **878.91 samples/s, 1.079868 s/update**, versus a7's
843.29 samples/s and 1.125486 s/update, on exactly 333,138 samples: **4.22%**
higher throughput. Across all 600 updates, **9,600/9,600 rank/update identities
and global sample counts matched**. All-rank peak fell from 52.591 to
**49.678 GiB allocated**. Checkpoints 400/600 were retained, 200 was pruned,
all 48 prompts/12 corrected grids completed, and the launcher exited cleanly.

At update 500, loss was 1.11761, gradient norm 0.55952, conditioning update
RMS 0.008517 and response gain 0.72605. EMA/live probe losses were
1.181881/1.140012, taking 52.66 s; a7's EMA loss was 1.181954. Over updates
400–600, median loss was 1.11775 versus 1.11789, and gradient norm median/p95
was 0.49768/0.58051 versus 0.50311/0.58671. At 600, conditioning update RMS
was 0.009412 and gain 0.82849. These checks show no material short-run health
regression; they do not prove long-run or peak-LR stability.

**Adopted:** native NPU SwiGLU, with the existing generic-device reference
path and unchanged parameter layout. No execution switch was added. The
complete local suite passes **797 tests, six skipped** (43.65 s). All four
CPU/NPU FP32/BF16 FFN regressions passed on the node. Pytest 8.4.2 is isolated
in `infra-build/pytest`, outside the training module path.

### Rejected fusion investigations

The attention gradient schema has five outputs, but its bundled GE converter
returns four. A process-local diagnostic supplied the unused, sink-absent
fifth result and passed conversion, then failed synchronization with error
507057 (`SUSPECT REMOTE ERROR`). Runtime logs localize the failure to an MTE
instruction accessing an out-of-range DDR address in the compiled attention
kernel. Fresh-process native attention subsequently passed on the same card;
hardware health was OK. No production or shared-package converter was changed.
Native attention remains selected. Raw evidence:
`runs/infra-attention-compile-probe{2,3}-0925.{json,log}` and
`runs/infra-attention-compile3-device-errors-0925.txt`.

The custom RoPE candidate targets 74.26 ms of multiplication across the three
single-stream shapes in a7's trace, before casts/layout operations. It uses
FP32 arithmetic, preserves head-major layout and saves only fixed frequencies
for inverse backward. Unlike the stock rotary gradient, it need not retain
inputs or impose its batch/head limit. The initial small-shape checks matched
outputs/gradients exactly, but a 2048-pair tile required 384 KiB of local buffer
where 192 KiB is available. The 512/1024-pair screen matched outputs/input gradients exactly, but took
57–116 ms versus 0.47–1.32 ms on real shapes. It is rejected as written.
Specializing shape/stride indices at compile time did not rescue it: real
shapes still took 54–112 ms, with exact output/gradient agreement. Thus dynamic
integer addressing alone does not explain the poor performance. The row-tiled variant reduced B26/S384 time to 1.41 ms, versus 0.47 ms
reference, but B72 with four rows per block exceeded the runtime's grid limit
(79,200 blocks requested, maximum 65,535). A 64-row persistent tile also exceeded local buffer capacity. The completed
48-block/16-row persistent screen matched outputs/gradients exactly but took
1.167/2.304/2.140 ms at the three real shapes, versus 0.478/1.074/1.327 ms
reference. It solved the grid-limit failure but did not beat the existing
implementation. Reject these standalone custom RoPE mappings; no Triton
runtime dependency is added to production. Hardware reports 24 cube/48 vector
cores. Raw tiled/persistent results are in
`runs/infra-triton-rope-{tiled,persistent,persistent2}-probe-0925.{json,log}`. Raw evidence:
`runs/infra-triton-rope-probe{,2,3}-0925.{json,log}`.

Muon pointwise work is another lead: its 150×1152×1152 stack contributes
32.51 ms of addition and 28.45 ms of scalar multiplication per three-update
trace, alongside GEMMs. [PyTorch 2.9 Muon](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/optim/_muon.py)
expresses Newton–Schulz updates with `addmm`; a batched `baddbmm` analogue merits
a numerical/timing screen. It was rejected: the three actual stack shapes
(150×1152×1152, 52×3072×1152, 26×1152×3072) took 85.52/61.60/28.84 ms versus
61.07/42.50/18.10 ms in the existing implementation, or 40–59% slower. Random
real-shape update relative L2 changed by 1.8–1.9%, with cosine above 0.9998;
a synthetic low-rank case differed much more. These synthetic numerical
differences do not diagnose training instability. No optimizer source,
coefficients, scaling or decay were changed. Raw evidence:
`runs/infra-muon-addmm-probe-0925.{json,log}`.

### Launch policy and intermediate accounting

The launcher now enforces `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
and `OMP_NUM_THREADS=1` before child processes initialize. Every measured
infra run already used these settings; this closes the manual-launch gap
without changing the measured execution recipe or adding a tunable. A real
child-process regression confirms that conflicting caller values cannot
silently replace the qualified policy.

A7 checkpoint barriers took 4.245, 4.177 and 4.311 s at updates 200, 400 and
600. Its endpoint prompt panel finished approximately 34.09 s after the
complete checkpoint record (artifact mtimes, not a precise profiler interval;
this predates the solver correction). At hero intervals, a7 checkpoint and
EMA/live-loss costs contribute about 0.0021 and 0.1086 s/update on the early
256p workload; the grid estimate adds about 0.0136 s/update. This gives roughly
1.250 s/update before startup/transition/KID costs and caption-schedule drift.
The living infra note recomputes these costs with the final recipe. Later-stage
timings remain unmeasured; no curriculum-wide 2× claim follows.

### Selected recipe recovery/profile replay

The source snapshot `repo-infra-0925-a8-recovery` adds the qualified launch
policy to a8 and resumes its complete checkpoint 400 with full state/RNG
verification, sample-identity recording and native profiling at updates
411–413. This reuses the retained a7 allocation; it is not a new hero run.
Original a8 logs, metrics, identities and grids are preserved under
`runs/ascend-infra-0925-swiglu-256p/before-recovery/`. Launcher log:
`runs/infra-a8-recovery-0925.log`. The replay completed through update 600,
matched all 3,200 rank/update sample identities and counts, generated all 48 prompts/12 corrected grids, retained
only checkpoints 400/600, and exited cleanly. The allocation was then stopped.
Machine-readable verification is in `infra/recovery-qualification.json` under
the run directory. It establishes the remaining step breakdown after SwiGLU.
Its source-manifest SHA-256 is
`93b57c94cf8fc2cc646267d0366e3cf8b1645531bf02d450a08c38232f517924`;
the complete config remains a8's `adb114b6861c98df4d6801c166e031611e7f93b8ee2609f633fa6df4b9644440`.
Exact model, both optimizers/schedulers, EMA and CPU/NPU RNG restoration passed
on all 16 ranks before the first resumed update.

The three-update native trace reports 2.88790 s computing, 0.24647 s exposed
communication and 0.28082 s free time: 84.56%, 7.22%, and 8.22% of 3.41518 s.
Another 1.05292 s of communication overlaps compute. These are profiled updates
411–413, not a replacement for the unprofiled 100–450 throughput comparison.
The earlier traces used different updates; kernel totals below identify costs,
not matched-shape speedup claims.

Remaining three-update kernel totals include dense MM 447.14 ms, Muon batched
MM 248.87 ms, AddMM 176.92 ms, multiplication 203.89 ms, casts 146.90 ms,
cast/copy transposes 273.82 ms, RMSNorm forward/backward 155.88 ms, LayerNorm
91.86 ms, and native attention forward/backward 227.72 ms. Attention also
requires 137.74 ms of transposes and 108.30 ms of padding/slicing. Kernel sums
are not an additive wall-time model: concurrency and enclosing operators
must not be counted twice.
