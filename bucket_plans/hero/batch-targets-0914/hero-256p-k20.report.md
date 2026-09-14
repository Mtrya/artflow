# Bucket plan: ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-256p-k20.json

Produced by `scripts/bench/plan_buckets.py`.  Bounds minimise padded
compute under the fitted time model; batch sizes are solved from a
measured memory model, not scanned.  Validate the plan with a
mixed-stream training run before relying on it: an isolated measurement
bounds a shape, it does not certify a plan whose many shapes share one
allocator.

```
python scripts/bench/plan_buckets.py --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d1@256p:9.010000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d2-wikiart@256p:12.660000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d2-museum@256p:0.600000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-human@256p:10.500000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-people@256p:10.140000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-pexels@256p:5.530000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-synth-v2@256p:0.640000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-inat@256p:0.140000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-megalith@256p:0.330000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-pd12m@256p:8.000000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-vintage@256p:9.590000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-zimage@256p:2.230000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion@256p:30.640000 --calibration ${ARTFLOW_ROOT}/bucket_plans/calib-533m/merged.json --image-tokens '{"1": 256, "2": 252, "3": 252, "4": 252, "5": 252}' --buckets 20 --vram-budget-gb 33.0 --min-batch 2 --max-batch 256 --min-mean-batch 80.0 --length-cap 2048 --caption-policy beta --caption-beta-start -1 --caption-beta-end 1 --caption-short-reserve 0.20 --caption-short-threshold 256 --caption-schedule linear --progress-start 0.0 --progress-end 0.75 --progress-grid 8 --out ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-256p-k20.json --report ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-256p-k20.report.md
```

## Inputs

| item | value |
| --- | --- |
| datasets | 13 |
| buckets per resolution | 20 |
| caption length cap | 2048 (the prompt contract allows 2048) |
| VRAM budget | 33.00 GB peak allocated, as the calibration measures it |
| batch range | [2, 256] |
| time alignment | on, kept above 3% predicted gain |
| alignment weights | count |
| minimum mean emitted micro-batch | 80 |
| image tokens | `{"1": 256, "2": 252, "3": 252, "4": 252, "5": 252}` |
| calibration | ${ARTFLOW_ROOT}/bucket_plans/calib-533m/merged.json (29 points) |

- `${ARTFLOW_ROOT}/precomputed_dataset/d1@256p` weight 0.0901, 91416 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d2-wikiart@256p` weight 0.1266, 214041 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d2-museum@256p` weight 0.0060, 10065 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-human@256p` weight 0.1050, 118393 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-people@256p` weight 0.1014, 114345 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-pexels@256p` weight 0.0553, 37417 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-synth-v2@256p` weight 0.0064, 10893 rows, resolutions: res1, res2, res3, res4
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-inat@256p` weight 0.0014, 2823 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-megalith@256p` weight 0.0033, 5504 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-pd12m@256p` weight 0.0800, 162307 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-vintage@256p` weight 0.0959, 194625 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-zimage@256p` weight 0.0223, 45228 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion@256p` weight 0.3064, 621773 rows, resolutions: res1, res2, res3, res4, res5

## Calibration

Image-token counts [252, 256, 1600, 1610, 3136], text lengths [128, 512, 1024, 2048], micro-batches [2, 4, 8, 16, 32, 64]; sequence lengths 384..5184.

## Memory model

Peak allocated memory is fitted as `m0 + B * m1 * L`, `L` being the padded
sequence length (image tokens + caption bound) — the form flash /
memory-efficient attention implies.  A fit below R^2 0.95 is refitted with a
`+ B * m2 * L^2` term.

`pooled`: peak = 1.07803 + 0.00105 * x   (R^2 1.0000, worst point off by 0.04 GB, 29 points, x = batch * L)
`img tokens 256`: peak = 1.09427 + 0.00105 * x   (R^2 1.0000, worst point off by 0.01 GB, 9 points, x = batch * L)
`img tokens 1600`: peak = 1.06846 + 0.00105 * x   (R^2 1.0000, worst point off by 0.01 GB, 5 points, x = batch * L)
`img tokens 3136`: peak = 1.06520 + 0.00105 * x   (R^2 1.0000, worst point off by 0.02 GB, 11 points, x = batch * L)

R^2 alone is a weak check on a sweep with a wide `batch * L` range: the
largest point dominates the total variance, so a model that is a few GB off
in the middle still scores above 0.99.  Read the worst-point residual next to
it, against the budget the sizes have to fit in.


## Time model

Micro-batch time is fitted as `t0 + B * (t1 * L + t2 * L^2)`.  The quadratic
term is mandatory: attention time grows with the square of the padded
sequence, and a plan sized without it mispredicts its long buckets.

`pooled`: t = 2.09255 + batch * (0.02597 * L + 8.43197e-06 * L^2)   (R^2 0.9998, worst point off by 15.4 ms, 29 points)
`img tokens 256`: t = -2.59707 + batch * (0.02555 * L + 8.85825e-06 * L^2)   (R^2 0.9999, worst point off by 7.3 ms, 9 points)
`img tokens 1600`: t = -6.27513 + batch * (0.02540 * L + 8.84856e-06 * L^2)   (R^2 1.0000, worst point off by 2.1 ms, 5 points)
`img tokens 3136`: t = 16.25734 + batch * (0.02493 * L + 8.52031e-06 * L^2)   (R^2 0.9999, worst point off by 10.5 ms, 11 points)


## Buckets

### resolution 1 (image tokens 256, 517098 captions)

Bounds minimise padded compute under the shape fit (padding overhead 2.9% of useful compute); sizes come from the shape fit for memory and the shape fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 19 | 1.95% | 110 | 110 | 32.98 **(extrapolated)** | 844.0 |
| 1 | 31 | 1.67% | 105 | 105 | 32.86 **(extrapolated)** | 844.0 |
| 2 | 42 | 1.22% | 101 | 100 | 32.51 **(extrapolated)** | 837.5 |
| 3 | 59 | 0.90% | 96 | 94 | 32.31 **(extrapolated)** | 836.6 |
| 4 | 76 | 1.50% | 91 | 89 | 32.24 | 839.3 |
| 5 | 86 | 1.62% | 88 | 86 | 32.10 | 838.0 |
| 6 | 95 | 1.50% | 86 | 84 | 32.18 | 842.4 |
| 7 | 106 | 1.48% | 83 | 81 | 32.01 | 840.6 |
| 8 | 118 | 1.05% | 80 | 78 | 31.85 | 839.4 |
| 9 | 135 | 0.79% | 77 | 74 | 31.60 | 836.9 |
| 10 | 157 | 0.61% | 73 | 70 | 31.57 | 841.8 |
| 11 | 182 | 0.56% | 69 | 65 | 31.11 | 835.3 |
| 12 | 212 | 0.38% | 64 | 61 | 31.19 | 845.2 |
| 13 | 261 | 0.19% | 58 | 54 | 30.53 | 838.6 |
| 14 | 364 | 0.14% | 48 | 44 | 29.85 | 844.2 |
| 15 | 499 | 0.09% | 40 | 34 | 28.16 | 825.0 |
| 16 | 647 | 0.07% | 33 | 27 | 26.80 | 815.4 |
| 17 | 825 | 0.05% | 27 | 22 | 26.17 | 832.8 |
| 18 | 1138 | 0.02% | 21 | 16 | 24.61 | 842.7 |
| 19 | 2048 | 0.00% | 13 | 8 | 20.53 | 844.5 |

### resolution 2 (image tokens 252, 344601 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 3.7% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 21 | 1.45% | 110 | 109 | 32.44 **(extrapolated)** | 843.3 |
| 1 | 39 | 1.09% | 104 | 101 | 32.06 **(extrapolated)** | 837.4 |
| 2 | 55 | 0.90% | 98 | 96 | 32.14 **(extrapolated)** | 843.7 |
| 3 | 73 | 1.02% | 93 | 90 | 31.91 | 841.8 |
| 4 | 88 | 1.18% | 89 | 86 | 31.90 | 845.2 |
| 5 | 103 | 1.11% | 85 | 82 | 31.76 | 845.1 |
| 6 | 120 | 0.88% | 81 | 77 | 31.27 | 835.7 |
| 7 | 141 | 0.74% | 77 | 73 | 31.32 | 842.1 |
| 8 | 163 | 0.79% | 72 | 68 | 30.82 | 833.6 |
| 9 | 185 | 0.73% | 69 | 65 | 31.02 | 844.4 |
| 10 | 210 | 0.50% | 65 | 61 | 30.78 | 843.7 |
| 11 | 252 | 0.24% | 60 | 55 | 30.30 | 839.7 |
| 12 | 350 | 0.16% | 50 | 45 | 29.63 | 843.1 |
| 13 | 434 | 0.12% | 44 | 38 | 28.56 | 829.8 |
| 14 | 547 | 0.11% | 37 | 32 | 28.03 | 838.3 |
| 15 | 666 | 0.08% | 32 | 27 | 27.21 | 837.6 |
| 16 | 800 | 0.06% | 28 | 23 | 26.58 | 845.0 |
| 17 | 991 | 0.04% | 24 | 18 | 24.66 | 817.6 |
| 18 | 1289 | 0.01% | 19 | 14 | 23.82 | 842.6 |
| 19 | 2048 | 0.00% | 13 | 8 | 20.47 | 836.7 |

### resolution 3 (image tokens 252, 453102 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 3.4% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 23 | 3.14% | 110 | 108 | 32.38 **(extrapolated)** | 842.2 |
| 1 | 33 | 2.08% | 106 | 104 | 32.32 **(extrapolated)** | 843.0 |
| 2 | 44 | 1.47% | 102 | 100 | 32.28 **(extrapolated)** | 844.6 |
| 3 | 60 | 1.15% | 97 | 94 | 31.99 **(extrapolated)** | 840.8 |
| 4 | 76 | 1.49% | 92 | 89 | 31.85 | 840.9 |
| 5 | 88 | 1.67% | 89 | 86 | 31.90 | 845.2 |
| 6 | 101 | 1.59% | 85 | 82 | 31.59 | 839.9 |
| 7 | 116 | 1.14% | 82 | 78 | 31.33 | 836.5 |
| 8 | 137 | 0.75% | 77 | 74 | 31.42 | 844.0 |
| 9 | 168 | 0.66% | 72 | 68 | 31.18 | 844.9 |
| 10 | 201 | 0.54% | 66 | 62 | 30.68 | 838.7 |
| 11 | 246 | 0.29% | 60 | 56 | 30.47 | 843.4 |
| 12 | 327 | 0.26% | 52 | 47 | 29.76 | 841.6 |
| 13 | 391 | 0.21% | 47 | 41 | 28.87 | 829.6 |
| 14 | 469 | 0.16% | 42 | 36 | 28.44 | 833.9 |
| 15 | 565 | 0.14% | 37 | 31 | 27.78 | 834.2 |
| 16 | 685 | 0.12% | 32 | 26 | 26.76 | 827.2 |
| 17 | 858 | 0.06% | 27 | 21 | 25.65 | 825.6 |
| 18 | 1179 | 0.03% | 21 | 15 | 23.70 | 818.5 |
| 19 | 2048 | 0.00% | 13 | 8 | 20.47 | 836.7 |

### resolution 4 (image tokens 252, 1053285 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 3.4% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 18 | 4.41% | 112 | 110 | 32.38 **(extrapolated)** | 840.9 |
| 1 | 28 | 3.21% | 108 | 106 | 32.36 **(extrapolated)** | 842.9 |
| 2 | 43 | 3.62% | 102 | 100 | 32.17 **(extrapolated)** | 841.5 |
| 3 | 57 | 2.99% | 98 | 95 | 32.02 **(extrapolated)** | 840.8 |
| 4 | 71 | 3.08% | 93 | 91 | 32.06 | 845.4 |
| 5 | 85 | 3.37% | 89 | 86 | 31.63 | 837.0 |
| 6 | 98 | 2.95% | 86 | 83 | 31.70 | 842.2 |
| 7 | 113 | 2.53% | 82 | 79 | 31.47 | 839.6 |
| 8 | 131 | 1.77% | 79 | 75 | 31.36 | 840.8 |
| 9 | 154 | 1.41% | 74 | 70 | 31.04 | 837.4 |
| 10 | 179 | 1.34% | 70 | 66 | 31.06 | 844.1 |
| 11 | 208 | 0.99% | 65 | 61 | 30.66 | 839.6 |
| 12 | 256 | 0.57% | 59 | 54 | 29.99 | 831.9 |
| 13 | 358 | 0.42% | 49 | 44 | 29.37 | 837.1 |
| 14 | 450 | 0.30% | 43 | 37 | 28.46 | 830.3 |
| 15 | 567 | 0.25% | 36 | 31 | 27.84 | 836.7 |
| 16 | 703 | 0.19% | 31 | 25 | 26.24 | 814.3 |
| 17 | 879 | 0.11% | 26 | 21 | 26.11 | 845.3 |
| 18 | 1165 | 0.05% | 21 | 15 | 23.48 | 808.0 |
| 19 | 2048 | 0.01% | 13 | 8 | 20.47 | 836.7 |

### resolution 5 (image tokens 252, 790504 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 2.9% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 18 | 2.80% | 112 | 110 | 32.38 **(extrapolated)** | 840.9 |
| 1 | 28 | 2.00% | 108 | 106 | 32.36 **(extrapolated)** | 842.9 |
| 2 | 38 | 2.22% | 104 | 102 | 32.26 **(extrapolated)** | 842.5 |
| 3 | 49 | 1.79% | 100 | 98 | 32.17 **(extrapolated)** | 842.9 |
| 4 | 63 | 1.38% | 96 | 93 | 31.96 **(extrapolated)** | 840.6 |
| 5 | 77 | 1.84% | 92 | 89 | 31.94 | 843.7 |
| 6 | 88 | 2.32% | 89 | 86 | 31.90 | 845.2 |
| 7 | 98 | 2.15% | 86 | 83 | 31.70 | 842.2 |
| 8 | 109 | 1.90% | 83 | 80 | 31.52 | 839.9 |
| 9 | 122 | 1.29% | 80 | 77 | 31.43 | 840.7 |
| 10 | 141 | 0.76% | 77 | 73 | 31.32 | 842.1 |
| 11 | 171 | 0.69% | 71 | 67 | 30.95 | 839.1 |
| 12 | 202 | 0.48% | 66 | 62 | 30.75 | 840.8 |
| 13 | 253 | 0.25% | 59 | 55 | 30.36 | 841.6 |
| 14 | 374 | 0.24% | 48 | 43 | 29.45 | 843.2 |
| 15 | 508 | 0.16% | 39 | 34 | 28.32 | 838.7 |
| 16 | 652 | 0.12% | 33 | 27 | 26.81 | 821.9 |
| 17 | 834 | 0.07% | 27 | 22 | 26.26 | 841.3 |
| 18 | 1113 | 0.03% | 22 | 16 | 24.10 | 820.6 |
| 19 | 2048 | 0.00% | 13 | 8 | 20.47 | 836.7 |

## Time alignment

Predicted throughput in samples per millisecond of the step's slowest
micro-batch: 0.0622 for the memory solution, 0.0964 as planned (+55.11%).

alignment at 845.4 ms buys +55.11% predicted throughput over the memory solution

The step model assumes one micro-batch per rank per optimizer step, so
every step pays the slowest runnable bucket, and each bucket is weighted
by its share of the draws.  With gradient accumulation that gain shrinks
towards the slowest-rank premium: a rank's micro-batches are a sum over
draws, and one rare bucket inside sixteen of them costs far less than the
whole step.  Read the number as an upper bound on what balancing buys,
and use the measured-candidate scan in
`scripts/bench/merge_screen_results.py` once a screen exists.

## Limits

- The models are fitted on DiT forward+backward points.  A training step also
  carries the frozen text encoder, the optimizer and EMA state, DDP buffers,
  and the per-shape allocations a mixed stream retains, so the budget has to
  cover whatever the calibration did not measure.  The fitted `m0` is the
  fixed part the sweep saw: several GB for a training run's weights, optimizer
  state and kernels, much less for a DiT-only sweep, which spends the
  difference out of the same budget.
- Peak memory is a property of the whole plan, not of a single shape; the
  numbers above are per-shape predictions.  A bucket marked **over budget**
  does not fit even at the smallest allowed batch, so the plan is not usable
  at that shape until the batch floor or the budget changes.
- Caption distribution: CaptionPolicy(kind='beta', beta_start=-1.0, beta_end=1.0, schedule='linear', early_at=0.5, short_reserve=0.2, short_threshold=256); progress [0.0, 0.75], 8 midpoints (stationary uses full-run 64-point average).
  Stage-progress averaging assumes equal exposure per progress point;
  changing emitted batch sizes can make actual exposure differ.
- Predicted queue-emitted mean micro-batch: 81.534840
  samples, using 1 / sum(p_i / B_i), not sum(p_i * B_i). This is a
  long-run mean, not a per-update floor; validate the actual sampler stream.
- The alignment model costs a step by its slowest micro-batch and ignores
  queue tails, compilation and synchronization, so treat its gain as a
  direction rather than a measurement.

## Warnings

- res1 len<=19: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res1 len<=31: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res1 len<=42: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res1 len<=59: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res2 len<=21: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res2 len<=39: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res2 len<=55: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res3 len<=23: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res3 len<=33: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res3 len<=44: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res3 len<=60: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res4 len<=18: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res4 len<=28: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res4 len<=43: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res4 len<=57: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res5 len<=18: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res5 len<=28: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res5 len<=38: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res5 len<=49: the padded sequence is outside the calibrated range; measure this shape before trusting its size
- res5 len<=63: the padded sequence is outside the calibrated range; measure this shape before trusting its size
