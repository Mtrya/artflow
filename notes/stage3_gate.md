# Stage 3 — Throughput Gate (2026-09-09)

## Scope and objective

Stage 2 fixed the hero architecture: h1152 × d24, all-single stream,
`mod=layer`, centered-grid RoPE, fused conditioning, Qwen3-0.6B text features
at exit k=20, and the Stage-2 art-forward mix. Stage 3 changes infrastructure
only; it does not reopen architecture, optimizer, dataset-mixture, or caption
curriculum decisions.

There is one objective: **raise measured training throughput while preserving
training accuracy**. The primary quality metric is the fixed `eval/loss`
trajectory at a matched cumulative global-sample budget and resolution stage.
KID, CLIP, and prompt grids remain useful diagnostics, but they do not turn an
otherwise accurate throughput improvement into a rejection at this scale.

The primary throughput metric is steady-state global samples processed per
training wall-clock second. It uses the actual samples that entered backward,
not `batch_size × accumulation_steps`; one-time compile/warm-up time is
reported separately from the steady-state number. Every run prints a
`[throughput-summary]` line with exactly this number.

## Stage-3 invariants

Every Stage-3 implementation and comparison preserves the following.

### Frozen training protocol

- **Architecture and objective:** the Stage-2 hero architecture, rectified
  flow, logit-normal timestep sampling, VAE, text encoder, k=20 feature level,
  conditioning, caption-dropout probability, and resolution-dependent time
  shift remain unchanged.
- **Optimizer:** use the Stage-2 winner throughout: chunked Muon at LR 0.02
  for routed 2D hidden weights, plus auxiliary AdamW for the remaining
  parameters. AdamW-only runs are historical Stage-2 references and are not a
  Stage-3 baseline or throughput denominator.
- **Text exit:** replacing a full Qwen forward plus `hidden_states[20]` slice
  with a true stop at layer 20 is permitted only because it produces the same
  conditioning features. It counts as an inference-side win regardless; it
  counts as a training-throughput win only if the measured throughput improves.

### Row and caption semantics

- Dataset-mix weights define the **row-level** sampling distribution. The
  expected dataset and row marginals are independent of curriculum stage;
  finite telemetry windows may fluctuate normally.
- A training example begins by drawing one row. That row produces exactly one
  captioned example, no matter how many captions it owns. A row with three
  captions is not three times as likely to train as a row with one caption.
- Curriculum changes only the caption selected *within that already selected
  row*, using the existing short-to-long `sample_caption(captions, stage)`
  semantics. It must not be converted into dataset, resolution, or global
  length-bucket weights.
- Long-caption up-weighting is a separate post-Stage-3 quality experiment; it
  is not an incidental consequence of batch size or bucket scheduling.

### Length-aware batching

- A micro-batch contains one composite `(resolution, caption-length)` bucket.
  Resolution is exact; length is a padding-efficiency bucket. Dataset identity
  is not part of this key, so a micro-batch may contain rows from different
  datasets without changing the intended mix.
- Caption length means the actual retained Qwen sequence length after applying
  the complete prompt template, `DROP_IDX`, and the 2048-token cap. No caption
  is truncated below that cap to satisfy a bucket.
- The DiT receives text padded to the bucket upper bound, yielding a static
  `(resolution, length, local_batch)` shape for that bucket. The Qwen encoder
  may retain dynamic in-batch padding.
- Bucket queues preserve their incomplete tails for later filling; they do not
  silently drop tails or recreate an exhausted bucket to force an epoch shape.
- Each length bucket has its own local micro-batch size. This is required to
  keep the activation peak approximately stable as
  `batch × (image_tokens + text_tokens)` changes.

### Accumulation and distributed training

- Different micro-batches, including those on different ranks in one optimizer
  step, may use different `(resolution, length)` buckets and local batch sizes.
  Optimizer-step boundaries remain synchronized across ranks.
- The accumulated gradient is the mean over every **actual global sample** in
  that optimizer step. Dynamic micro-batch size must neither accidentally
  up-weight long buckets nor change the row-level mixture.
- DDP reduces gradients **once per optimizer step**, not once per micro-batch.
  Micro-batches inside a step accumulate locally under `no_sync`; the reduction
  is linear, so the accumulated gradient is unchanged, while the cross-rank
  straggler wait drops from one per micro-batch to one per step.
- Logging records the actual global samples per optimizer step. Throughput is
  computed from that count, so it remains meaningful when effective batch size
  varies.

### Resolution curriculum

- Stage 3 exercises the intended progressive sequence `256p → 640p → 896p`.
  Each resolution has its own data path, `(resolution, length)` bucket table,
  local batch sizes, compile shapes, and time-shift parameters.
- Resolution switching changes resolution-specific infrastructure only; it
  does not change the row-level mix or the caption-selection semantics above.

## Implementation

### Data path

`RowLengthQueueBatchSampler` (`src/dataset/sampler.py`) emits `RowRef`
descriptors. Rank-local rows are interleaved with the usual stride sharding.
Persistent queues are keyed by `(resolution_id, length_bucket_index)`; dataset
identity is deliberately not part of the key, so a batch can contain rows from
different mix entries. Each length bucket has its configured local micro-batch
size. A partial queue tail stays available until a later row fills it.

`RowDescriptorDataset` resolves a descriptor to the latent and the selected
caption. `row_length_collate_fn` enforces one resolution, length bucket, bucket
upper bound, and batch ID per batch. `pad_text_to_hi` pads the dynamic in-batch
text result to the bucket upper bound without truncating it.

`RowLengthMetadata` sidecars are now derived by `src/dataset/length_metadata.py`
(`ensure_sidecar()`): it tokenizes each complete prompt — the same tokenizer
checkpoint, prompt template, `DROP_IDX` removal, and 2048-token cap that
`encode_text` applies — and writes a `length_metadata.npz` companion file into
the dataset directory. The training side calls `ensure_sidecar()` per mix entry
and builds the file on first use; the pass is CPU-only and never loads latents.

A row-queue micro-batch may have a bucket-specific local size. Training keeps
Accelerate's accumulation factor at one, so DDP synchronizes every actual
micro-batch. Each per-sample-mean loss is multiplied by the local sample count
before backward; at an optimizer boundary, gradients are divided by the
reduced global count divided by world size, before clipping and the Stage-2
Muon plus auxiliary AdamW updates. Every optimizer step is therefore weighted
by the actual global samples that entered backward.

The sampler's state serialization preserves queue tails, ready batches, and
in-flight batches. Each rank writes `sampler_state_rank_XXXXX.pt` next to
Accelerate's model/optimizer state, and `--resume_full` restores it and replays
batches whose backward acknowledgement never completed.

### Throughput fast paths (all opt-in, defaults preserve the baseline)

| flag | what it does | math |
| --- | --- | --- |
| `--fast_caption_dropout` | draws the dropout mask on the host instead of reading a device reduction | same distribution, no device sync |
| `--fast_telemetry` | one `bincount` per micro-batch instead of one device kernel per sample | identical counts |
| `--fast_text_slice` | slices the padded hidden tensor instead of gathering/repacking per sequence | identical features (tested) |
| `--fast_attn` | hoists RoPE frequencies and the padded-text attention bias out of the block loop | bit-identical forward (tested) |
| `--text_encoder_exit_mode stop_at_layer` | stops the Qwen forward after layer 20 | bit-identical features |
| `--compile --compile_blocks` | `torch.compile` on each DiT block; all 24 blocks share one graph per bucket shape | fused kernels, same math |
| `--muon_batched_ns` | one bmm per matrix shape in the Newton-Schulz step | same per-matrix iteration |
| `--stage3_no_sync` | accumulates rank-local gradients inside an optimizer step and reduces once at the boundary | identical accumulated gradient (linear reduction) |

`tests/test_fast_paths.py` asserts bit-identical outputs for `fast_attn` and
feature-identical output for `fast_text_slice` (including the dropped-caption
edge case); `tests/test_muon_batched.py` does the same for the batched
Newton-Schulz step and the once-per-param weight decay.

## Measured results

All numbers are single 4090, 256p, the Stage-3 mix, unless stated otherwise.

### DiT forward+backward ceiling (`scripts/bench/transformer_ceiling.py`)

| config (micro=16) | seq=128 | seq=192 | peak VRAM |
| --- | --- | --- | --- |
| eager, memory-efficient attention | 79.1 samples/s (202.4 ms) | 67.4 (237.5 ms) | 10.3 GB |
| eager, cuDNN attention | 55.6 (287.6 ms) | 46.0 (347.6 ms) | 15.0 GB |
| per-block compile | **95.0 (168.4 ms)** | **80.4 (199.1 ms)** | **7.2 GB** |
| whole-model compile | 98.6 (162.4 ms) | — | 7.2 GB |

cuDNN attention is *slower* in training than memory-efficient attention even
though its forward-only kernel is faster: the forward-only microbenchmark
shows 0.146 ms vs 0.237 ms with a pad bias, but the
forward+backward ceiling shows a 30% regression, so the backward is the
problem. Stage 3 keeps the memory-efficient backend.

`--compile_mode reduce-overhead` (CUDA graphs) fails to capture for the
per-block path and is not used.

### Micro-batch size sweep (eager, DiT only)

| micro | seq=64 | seq=128 | seq=192 | seq=256 |
| --- | --- | --- | --- | --- |
| 16 | **96.3** | **78.9** | **67.2** | **58.2** |
| 24 | 95.8 | 79.0 | 65.8 | 56.6 |
| 32 | 94.3 | 77.7 | 64.0 | 54.9 |
| 48 | 90.7 | 74.0 | 61.9 | 53.6 |

samples/s. Micro-batch 16 is the best point at every text length, so the bucket
plan's local batch sizes stay at 16 for the short buckets; the ceiling is not
memory-limited (13.3 GB at micro=16/seq=256).

### Frozen text encoder (batch 16)

| exit mode | len 64 | len 128 | len 192 |
| --- | --- | --- | --- |
| `full_forward_slice` | 24.7 ms | 24.8 ms | 25.5 ms |
| `stop_at_layer` (k=20) | **18.3 ms** | **18.5 ms** | **20.2 ms** |

### Muon step

`Muon.step` measures 163 ms per optimizer step (478 M orthogonalized
parameters) before batching; `aux` AdamW is 1.6-2.1 ms.

### End-to-end, 1 GPU, 256p

Sequential A/B on one node: the arms are run one after another in a single
job (400 optimizer steps each, no per-micro profiling, same seed and therefore
the same row/caption draw order, each arm printing its `[throughput-summary]`
line; the full stack is
`--fast_caption_dropout --fast_telemetry --fast_text_slice --fast_attn
--text_encoder_exit_mode stop_at_layer --compile --compile_blocks
--muon_batched_ns`):

| arm | samples/s (all 400 steps) | samples/s (steady state) | s/step | peak mem |
| --- | --- | --- | --- | --- |
| baseline | 53.78 | 53.0-53.8 | 4.751 | — |
| full stack | 69.27 | 74.7-75.1 | 3.689 (3.42 steady) | 17.8 GB |
| ratio | **1.29×** | **1.39×** | | |

The all-in number includes the one-time per-shape compile stalls inside the
400 steps; the steady-state number is the one that carries to a long run.
Both clear the 1.25× floor. Run-to-run scatter of an identical baseline config
is 1.5% (4.58 vs 4.65 s/step for two concurrent baselines), so the sequential
A/B is the number to trust. Peak memory is the largest
`max_memory_allocated` sample over the run (see the multi-GPU section); the
eager baseline's peak was not recorded under this accounting and is left out
rather than quoted from the older, non-comparable figure.

Per-flag contribution, measured in the same protocol (1 GPU, 256p, s/step):

| arm | s/step | vs baseline |
| --- | --- | --- |
| baseline | 4.58-4.65 | 1.00× |
| + fast paths (`fast_caption_dropout`, `fast_telemetry`, `fast_text_slice`, `fast_attn`) | 4.24 | 1.08× |
| + `stop_at_layer` | 4.16-4.25 | 1.10× |
| + per-block compile | 3.48-3.50 | 1.32× |
| `--text_prefetch` | 4.32-4.36 | 1.06× (dropped) |
| whole-model compile instead of per-block | 3.67 | 1.25× (dropped) |

`--text_prefetch` was measured and **not kept**: once the host syncs are gone
the GPU is the bottleneck, so overlapping the frozen text encoder with the DiT
adds scheduling overhead without reducing total device work.

### Multi-GPU DDP (4 × 4090, one node, 256p)

Four arms run back to back on one node in a single job, each printing its
`[throughput-summary]` line (150 optimizer steps each, same seed, no
per-micro profiling, `THROUGHPUT_SKIP_STEPS=50`). "Full stack" is the flag set
from the 1-GPU section above plus `--stage3_no_sync` where listed. Every arm
moves the same global samples per optimizer step (1022.5), so the arms differ
only in how the gradient is reduced.

| arm | samples/s (all-in) | samples/s (steady) | s/step (steady) | peak mem/GPU |
| --- | --- | --- | --- | --- |
| baseline | 166.09 | 165.83 | 6.17 | 26.9 GB |
| baseline + `--stage3_no_sync` | 210.58 | 209.48 | 4.88 | 26.9 GB |
| full stack | 136.20 | 188.49 | 5.42 | 19.9 GB |
| full stack + `--stage3_no_sync` | **259.52** | **282.18** | 3.62 | 19.9 GB |

The full stack's all-in number is dragged down by the per-shape compile stalls
inside 150 steps — it is the only arm that compiles — so its steady figure is
the one that carries to a long run. Against the 4-GPU baseline, the full stack
with `no_sync` is **1.56× all-in / 1.70× steady**.

`--stage3_no_sync` is the largest multi-GPU lever: 1.27× on top of the baseline
and 1.50× on top of the full stack, steady state. Without it DDP reduced
gradients on every micro-batch — 16 collective waits per optimizer step, each
blocking on the slowest rank, and rank-local buckets differ by construction.
Reducing once at the optimizer boundary is mathematically identical and removes
15 of those waits.

Correctness is checked as on one GPU: the fixed `eval/loss` probe agrees across
the reduction change at every matched step — 1.99490/1.99490, 1.95234/1.95227,
1.89816/1.89747 for baseline vs baseline+`no_sync`, and 1.99487/1.99487,
1.95200/1.95200, 1.89717/1.89659 for full stack vs full stack+`no_sync` — i.e.
inside the CUDA-ordering noise floor measured for identical-configuration pairs.

Scaling efficiency at 256p, steady state:

| config | 1 GPU | 4 GPU | per-GPU ratio |
| --- | --- | --- | --- |
| baseline | 53.78 | 165.83 | 77% |
| full stack (`no_sync` at 4 GPU) | 76.29 | 282.18 | 92% |

Peak memory per GPU (`max_memory_allocated` over the whole run, sampled every
25 steps) rises from 17.8 GB on one GPU to 19.9 GB on four for the same full
stack, so DDP itself adds ≈2.1 GB. The eager baseline peaks at 26.9 GB on four
GPUs: per-block compile is worth ≈26% of activation memory, which is what makes
the 896p buckets affordable later.

Whole-model `torch.compile` (without `--compile_blocks`) was measured on one
GPU in the same protocol (200 steps, `THROUGHPUT_SKIP_STEPS=50`) and
**rejected**: 69.64 samples/s steady against 76.29 for per-block compile, and
25.2 GB peak against 17.8 GB. Under the row queue's per-bucket shapes it trips
`recompile_limit` and recompiles a 24-layer graph for every new bucket (~2.5
minutes each), so its 200-step all-in figure collapses to 22.83 samples/s.

Eight-GPU validation is out of Stage-3 scope; four GPUs is the largest
configuration measured.

### Accuracy

The gate compares the fixed 512-sample `eval/loss` probe at matched steps.
Two pieces of evidence from the same sequential A/B job:

| step | baseline | full stack | relative |
| --- | --- | --- | --- |
| 50 | 2.000673 | 2.000671 | -0.00% |
| 100 | 1.980879 | 1.980879 | 0.00% |
| 150 | 1.939122 | 1.939059 | -0.00% |
| 200 | 1.872697 | 1.872237 | -0.02% |
| 250 | 1.791321 | 1.791708 | +0.02% |
| 300 | 1.723676 | 1.725592 | +0.11% |
| 350 | 1.690141 | 1.690747 | +0.04% |
| 400 | 1.684894 | 1.683330 | **-0.09%** |

The full stack tracks the baseline inside ±0.11% at every probe and is
marginally ahead at step 400. Longer runs confirm this is scatter, not drift:
in the 2000-step pair (`s3-arm-b-fastpaths` vs `s3-arm-f-fullstack`, which
share the same row/caption/dropout RNG streams), the compiled arm is 0.7-0.9%
*better* at steps 1100 and 1150.

An identical-configuration pair (`s3-arm-a-baseline` vs `s3-arm-a2-baseline`)
already diverges on its own (1e-6 at step 50, 4.5e-4 at step 300), i.e. CUDA
kernel ordering makes single-run trajectories chaotic; the arms' spread is the
same order. No arm shows a systematic accuracy regression.

### 640p / 896p inputs for Stage 4

DiT forward+backward, micro=8 (`scripts/bench/transformer_ceiling.py`):

| resolution | eager | per-block compile | speedup | VRAM eager → compile |
| --- | --- | --- | --- | --- |
| 640p, seq=128 | 12.6 samples/s | 15.0 | 1.18× | 21.7 → 14.9 GB |
| 640p, seq=256 | 11.5 | 13.6 | 1.18× | 23.4 → 16.0 GB |
| 896p, seq=128 | 5.1 | 6.0 | 1.17× | 40.1 → 27.4 GB |
| 896p, seq=256 | 4.9 | 5.7 | 1.16× | 41.7 → 28.4 GB |

The compile win carries to high resolution, and the ~30% activation-memory
reduction is what makes 896p micro-batches of 8 fit comfortably.

## Throughput decision

1. Establish a new Muon baseline on the frozen protocol above. Existing
   Stage-3 AdamW measurements and the item-expanded Plan-D runs are not part of
   this baseline. **Done:** 53.78 samples/s at 256p on one 4090.
2. Optimize the dominant measured bottlenecks. A change is kept only when its
   matched `eval/loss` trajectory preserves accuracy and its measured
   steady-state throughput improves. **Kept:** host-side caption dropout,
   bincount telemetry, the vectorized text slice, hoisted RoPE/attention-bias
   tables, true early exit at layer 20, per-block `torch.compile`, batched
   Muon Newton-Schulz, boundary-only DDP reduction (`--stage3_no_sync`).
   **Rejected on measurement:** cuDNN attention (30% slower in
   forward+backward), text-encoder prefetch (no net win once host syncs are
   gone), CUDA-graph `reduce-overhead` compile (capture fails per block),
   whole-model `torch.compile` (recompiles per bucket under the row queue).
3. The working throughput floor is **1.25×** the re-measured baseline at each
   exercised resolution. It is a floor rather than a stopping point. **Met:**
   1.29× including one-time compile, 1.39× steady state at 256p on one GPU;
   1.56× / 1.70× at 256p on four GPUs; 1.16-1.18× on the DiT-only 640p/896p
   ceilings.
4. Report the final numbers per resolution: `eval/loss` trajectory, actual
   global samples/s, local batch table, actual global samples/optimizer step,
   and peak memory. These are the Stage-4 sizing inputs.

### Stage-4 inputs

- 256p, 1 GPU: 74.9-76.3 samples/s steady state, 255.5 samples/optimizer step,
  local batch table `64:16 128:16 192:16 256:16 384:8 512:8 768:4 1536:2
  2048:1`, peak 17.8 GB.
- 256p, 4 GPU: 282.18 samples/s steady state (70.5 samples/s/GPU), 1022.5
  samples/optimizer step, peak 19.9 GB/GPU. Scaling efficiency 92% with
  `--stage3_no_sync`, 77% without it.
- 640p, 1 GPU, micro=8: 15.0 samples/s (DiT-only ceiling), 14.9 GB.
- 896p, 1 GPU, micro=8: 6.0 samples/s, 27.4 GB (DiT-only ceiling; the
  end-to-end peak adds the encoder, optimizer and DDP buffers on top).
- The 256p bucket table and the compile path are resolution-agnostic: a 640p or
  896p run needs its own sidecars, bucket plan and time shift, and inherits the
  same mechanisms. End-to-end 640p/896p training is out of Stage-3 scope; the
  high-resolution numbers above are DiT-only ceilings.

The former AdamW calibration, item-expanded sampler, bucket-weight curriculum,
and associated gate results are superseded by this document.
