# Bucket plan: ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-640p-k20.json

Produced by `scripts/bench/plan_buckets.py`.  Bounds minimise padded
compute under the fitted time model; batch sizes are solved from a
measured memory model, not scanned.  Validate the plan with a
mixed-stream training run before relying on it: an isolated measurement
bounds a shape, it does not certify a plan whose many shapes share one
allocator.

```
python scripts/bench/plan_buckets.py --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d1@640p:10.820000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d2-wikiart@640p:12.320000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d2-museum@640p:0.780000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-human@640p:12.190000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-people-a@640p:5.876955 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-people-b@640p:5.883045 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-pexels@640p:6.660000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-synth-v2@640p:0.780000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-megalith@640p:0.380000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-pd12m@640p:8.430000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-vintage@640p:9.410000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-zimage@640p:3.220000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p0@640p:4.649294 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p1@640p:4.634332 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p2@640p:4.651669 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p3@640p:4.650601 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p4@640p:4.654104 --calibration ${ARTFLOW_ROOT}/bucket_plans/calib-533m/merged.json --image-tokens '{"1": 1600, "2": 1590, "3": 1590, "4": 1610, "5": 1610}' --buckets 20 --vram-budget-gb 30.0 --min-batch 2 --max-batch 256 --min-mean-batch 12.8 --length-cap 2048 --caption-policy beta --caption-beta-start -1 --caption-beta-end 1 --caption-short-reserve 0.20 --caption-short-threshold 256 --caption-schedule linear --progress-start 0.75 --progress-end 0.95 --progress-grid 8 --out ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-640p-k20.json --report ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-640p-k20.report.md
```

## Inputs

| item | value |
| --- | --- |
| datasets | 17 |
| buckets per resolution | 20 |
| caption length cap | 2048 (the prompt contract allows 2048) |
| VRAM budget | 30.00 GB peak allocated, as the calibration measures it |
| batch range | [2, 256] |
| time alignment | on, kept above 3% predicted gain |
| alignment weights | count |
| minimum mean emitted micro-batch | 12.8 |
| image tokens | `{"1": 1600, "2": 1590, "3": 1590, "4": 1610, "5": 1610}` |
| calibration | ${ARTFLOW_ROOT}/bucket_plans/calib-533m/merged.json (29 points) |

- `${ARTFLOW_ROOT}/precomputed_dataset/d1@640p` weight 0.1082, 91151 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d2-wikiart@640p` weight 0.1232, 138346 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d2-museum@640p` weight 0.0078, 8727 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-human@640p` weight 0.1219, 114100 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-people-a@640p` weight 0.0588, 55009 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-people-b@640p` weight 0.0588, 55066 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-pexels@640p` weight 0.0666, 37417 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-synth-v2@640p` weight 0.0078, 10893 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-megalith@640p` weight 0.0038, 5361 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-pd12m@640p` weight 0.0843, 157683 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-vintage@640p` weight 0.0941, 158531 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-zimage@640p` weight 0.0322, 45248 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p0@640p` weight 0.0465, 78307 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p1@640p` weight 0.0463, 78055 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p2@640p` weight 0.0465, 78347 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p3@640p` weight 0.0465, 78329 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p4@640p` weight 0.0465, 78388 rows, resolutions: res1, res2, res3, res4, res5

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

### resolution 1 (image tokens 1600, 357616 captions)

Bounds minimise padded compute under the shape fit (padding overhead 0.8% of useful compute); sizes come from the shape fit for memory and the shape fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 21 | 0.97% | 16 | 16 | 28.41 | 1024.6 |
| 1 | 34 | 1.59% | 16 | 16 | 28.63 | 1035.8 |
| 2 | 46 | 1.10% | 16 | 16 | 28.83 | 1046.3 |
| 3 | 63 | 0.92% | 16 | 15 | 27.36 | 994.4 |
| 4 | 77 | 1.56% | 16 | 15 | 27.58 | 1006.0 |
| 5 | 88 | 1.94% | 16 | 15 | 27.76 | 1015.1 |
| 6 | 99 | 1.73% | 16 | 15 | 27.93 | 1024.2 |
| 7 | 111 | 1.33% | 16 | 15 | 28.12 | 1034.2 |
| 8 | 125 | 0.92% | 15 | 15 | 28.34 | 1045.9 |
| 9 | 144 | 0.78% | 15 | 14 | 26.80 | 990.7 |
| 10 | 165 | 0.70% | 15 | 14 | 27.11 | 1007.3 |
| 11 | 187 | 0.60% | 15 | 14 | 27.44 | 1024.8 |
| 12 | 215 | 0.43% | 15 | 14 | 27.85 | 1047.3 |
| 13 | 263 | 0.21% | 14 | 13 | 26.60 | 1008.2 |
| 14 | 400 | 0.11% | 13 | 12 | 26.37 | 1028.1 |
| 15 | 533 | 0.08% | 12 | 11 | 25.80 | 1032.6 |
| 16 | 687 | 0.08% | 12 | 10 | 25.18 | 1037.5 |
| 17 | 864 | 0.05% | 11 | 9 | 24.44 | 1040.5 |
| 18 | 1165 | 0.03% | 9 | 7 | 21.47 | 958.9 |
| 19 | 2048 | 0.00% | 7 | 5 | 20.29 | 1045.8 |

### resolution 2 (image tokens 1590, 226447 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 1.0% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 23 | 0.86% | 17 | 16 | 28.28 | 1023.3 |
| 1 | 43 | 1.03% | 16 | 16 | 28.62 | 1040.3 |
| 2 | 59 | 0.77% | 16 | 16 | 28.89 | 1054.1 |
| 3 | 75 | 0.93% | 16 | 15 | 27.40 | 1001.3 |
| 4 | 88 | 1.02% | 16 | 15 | 27.61 | 1011.8 |
| 5 | 102 | 1.02% | 16 | 15 | 27.83 | 1023.2 |
| 6 | 117 | 0.78% | 16 | 15 | 28.07 | 1035.5 |
| 7 | 136 | 0.60% | 15 | 15 | 28.37 | 1051.2 |
| 8 | 159 | 0.55% | 15 | 14 | 26.89 | 999.0 |
| 9 | 183 | 0.54% | 15 | 14 | 27.24 | 1017.7 |
| 10 | 209 | 0.40% | 15 | 14 | 27.63 | 1038.1 |
| 11 | 252 | 0.22% | 14 | 13 | 26.32 | 995.8 |
| 12 | 348 | 0.12% | 14 | 12 | 25.59 | 986.0 |
| 13 | 434 | 0.10% | 13 | 12 | 26.68 | 1047.3 |
| 14 | 547 | 0.11% | 12 | 11 | 25.86 | 1036.1 |
| 15 | 671 | 0.09% | 12 | 10 | 24.91 | 1020.3 |
| 16 | 817 | 0.07% | 11 | 9 | 23.91 | 1004.3 |
| 17 | 1003 | 0.04% | 10 | 8 | 22.94 | 994.3 |
| 18 | 1289 | 0.02% | 9 | 7 | 22.32 | 1014.6 |
| 19 | 2048 | 0.00% | 7 | 5 | 20.25 | 1032.4 |

### resolution 3 (image tokens 1590, 172212 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 1.1% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 0.52% | 16 | 16 | 28.32 | 1025.0 |
| 1 | 43 | 0.77% | 16 | 16 | 28.62 | 1040.3 |
| 2 | 59 | 0.52% | 16 | 16 | 28.89 | 1054.1 |
| 3 | 80 | 0.78% | 16 | 15 | 27.48 | 1005.3 |
| 4 | 94 | 0.93% | 16 | 15 | 27.70 | 1016.7 |
| 5 | 108 | 0.80% | 16 | 15 | 27.93 | 1028.1 |
| 6 | 123 | 0.52% | 16 | 15 | 28.16 | 1040.5 |
| 7 | 146 | 0.39% | 15 | 14 | 26.70 | 989.0 |
| 8 | 171 | 0.46% | 15 | 14 | 27.07 | 1008.4 |
| 9 | 193 | 0.39% | 15 | 14 | 27.39 | 1025.6 |
| 10 | 219 | 0.29% | 15 | 14 | 27.77 | 1046.0 |
| 11 | 262 | 0.15% | 14 | 13 | 26.46 | 1003.2 |
| 12 | 365 | 0.13% | 14 | 12 | 25.81 | 998.0 |
| 13 | 456 | 0.11% | 13 | 11 | 24.80 | 974.8 |
| 14 | 565 | 0.10% | 12 | 11 | 26.07 | 1048.4 |
| 15 | 685 | 0.08% | 12 | 10 | 25.06 | 1029.3 |
| 16 | 845 | 0.06% | 11 | 9 | 24.18 | 1021.1 |
| 17 | 1048 | 0.04% | 10 | 8 | 23.32 | 1019.5 |
| 18 | 1323 | 0.01% | 9 | 7 | 22.57 | 1032.4 |
| 19 | 2048 | 0.00% | 7 | 5 | 20.25 | 1032.4 |

### resolution 4 (image tokens 1610, 794171 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 1.0% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 23 | 3.33% | 16 | 16 | 28.62 | 1040.3 |
| 1 | 40 | 3.75% | 16 | 16 | 28.91 | 1054.9 |
| 2 | 53 | 3.25% | 16 | 15 | 27.37 | 999.6 |
| 3 | 68 | 3.22% | 16 | 15 | 27.61 | 1011.8 |
| 4 | 83 | 3.49% | 16 | 15 | 27.85 | 1024.0 |
| 5 | 97 | 3.13% | 16 | 15 | 28.07 | 1035.5 |
| 6 | 113 | 2.71% | 15 | 15 | 28.32 | 1048.7 |
| 7 | 133 | 2.02% | 15 | 14 | 26.80 | 994.4 |
| 8 | 157 | 1.78% | 15 | 14 | 27.15 | 1013.0 |
| 9 | 181 | 1.70% | 15 | 14 | 27.51 | 1031.9 |
| 10 | 210 | 1.26% | 15 | 14 | 27.94 | 1054.8 |
| 11 | 257 | 0.61% | 14 | 13 | 26.66 | 1014.4 |
| 12 | 368 | 0.59% | 13 | 12 | 26.10 | 1014.3 |
| 13 | 450 | 0.50% | 13 | 11 | 24.96 | 984.1 |
| 14 | 544 | 0.43% | 12 | 11 | 26.05 | 1047.7 |
| 15 | 650 | 0.35% | 12 | 10 | 24.90 | 1019.6 |
| 16 | 780 | 0.25% | 11 | 9 | 23.75 | 994.1 |
| 17 | 977 | 0.16% | 10 | 8 | 22.89 | 991.0 |
| 18 | 1282 | 0.06% | 9 | 7 | 22.42 | 1021.4 |
| 19 | 2048 | 0.01% | 7 | 5 | 20.36 | 1041.2 |

### resolution 5 (image tokens 1610, 853005 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 0.8% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 23 | 3.53% | 16 | 16 | 28.62 | 1040.3 |
| 1 | 34 | 4.00% | 16 | 16 | 28.81 | 1049.8 |
| 2 | 45 | 3.38% | 16 | 15 | 27.25 | 993.2 |
| 3 | 60 | 2.79% | 16 | 15 | 27.48 | 1005.3 |
| 4 | 76 | 3.66% | 16 | 15 | 27.74 | 1018.3 |
| 5 | 88 | 4.31% | 16 | 15 | 27.93 | 1028.1 |
| 6 | 99 | 3.82% | 16 | 15 | 28.10 | 1037.2 |
| 7 | 110 | 2.89% | 15 | 15 | 28.27 | 1046.2 |
| 8 | 124 | 2.09% | 15 | 14 | 26.67 | 987.4 |
| 9 | 145 | 1.38% | 15 | 14 | 26.98 | 1003.7 |
| 10 | 177 | 1.15% | 15 | 14 | 27.45 | 1028.7 |
| 11 | 219 | 0.65% | 15 | 13 | 26.14 | 986.2 |
| 12 | 327 | 0.53% | 14 | 12 | 25.58 | 985.3 |
| 13 | 390 | 0.50% | 13 | 12 | 26.38 | 1030.0 |
| 14 | 466 | 0.42% | 13 | 11 | 25.15 | 994.8 |
| 15 | 553 | 0.35% | 12 | 11 | 26.16 | 1053.9 |
| 16 | 656 | 0.27% | 12 | 10 | 24.96 | 1023.5 |
| 17 | 814 | 0.16% | 11 | 9 | 24.07 | 1014.5 |
| 18 | 1113 | 0.07% | 10 | 7 | 21.17 | 934.7 |
| 19 | 2048 | 0.01% | 7 | 5 | 20.36 | 1041.2 |

## Time alignment

Predicted throughput in samples per millisecond of the step's slowest
micro-batch: 0.0105 for the memory solution, 0.0138 as planned (+31.58%).

alignment at 1054.9 ms buys +31.58% predicted throughput over the memory solution

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
- Caption distribution: CaptionPolicy(kind='beta', beta_start=-1.0, beta_end=1.0, schedule='linear', early_at=0.5, short_reserve=0.2, short_threshold=256); progress [0.75, 0.95], 8 midpoints (stationary uses full-run 64-point average).
  Stage-progress averaging assumes equal exposure per progress point;
  changing emitted batch sizes can make actual exposure differ.
- Predicted queue-emitted mean micro-batch: 14.609115
  samples, using 1 / sum(p_i / B_i), not sum(p_i * B_i). This is a
  long-run mean, not a per-update floor; validate the actual sampler stream.
- The alignment model costs a step by its slowest micro-batch and ignores
  queue tails, compilation and synchronization, so treat its gain as a
  direction rather than a measurement.
