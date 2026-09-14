# Bucket plan: ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-896p-k20.json

Produced by `scripts/bench/plan_buckets.py`.  Bounds minimise padded
compute under the fitted time model; batch sizes are solved from a
measured memory model, not scanned.  Validate the plan with a
mixed-stream training run before relying on it: an isolated measurement
bounds a shape, it does not certify a plan whose many shapes share one
allocator.

```
python scripts/bench/plan_buckets.py --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d1@896p:17.360000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d2-wikiart@896p:11.450000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d2-museum@896p:0.520000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-human@896p:15.750000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-people@896p:15.140000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d3-pexels@896p:10.740000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-megalith@896p:0.130000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-pd12m@896p:11.170000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-vintage@896p:1.570000 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p0@896p:3.239290 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p1@896p:3.226566 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p2@896p:3.225705 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p3@896p:3.239195 --dataset ${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p4@896p:3.229245 --calibration ${ARTFLOW_ROOT}/bucket_plans/calib-533m/merged.json --image-tokens '{"1": 3136, "2": 3108, "3": 3108, "4": 3072, "5": 3072}' --buckets 20 --vram-budget-gb 30.0 --min-batch 2 --max-batch 256 --min-mean-batch 7.142857142857143 --length-cap 2048 --caption-policy beta --caption-beta-start -1 --caption-beta-end 1 --caption-short-reserve 0.20 --caption-short-threshold 256 --caption-schedule linear --progress-start 0.95 --progress-end 1.0 --progress-grid 8 --out ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-896p-k20.json --report ${ARTFLOW_ROOT}/bucket_plans/hero/batch-targets-0914/hero-896p-k20.report.md
```

## Inputs

| item | value |
| --- | --- |
| datasets | 14 |
| buckets per resolution | 20 |
| caption length cap | 2048 (the prompt contract allows 2048) |
| VRAM budget | 30.00 GB peak allocated, as the calibration measures it |
| batch range | [2, 256] |
| time alignment | on, kept above 3% predicted gain |
| alignment weights | count |
| minimum mean emitted micro-batch | 7.14286 |
| image tokens | `{"1": 3136, "2": 3108, "3": 3108, "4": 3072, "5": 3072}` |
| calibration | ${ARTFLOW_ROOT}/bucket_plans/calib-533m/merged.json (29 points) |

- `${ARTFLOW_ROOT}/precomputed_dataset/d1@896p` weight 0.1736, 90695 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d2-wikiart@896p` weight 0.1145, 79763 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d2-museum@896p` weight 0.0052, 3643 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-human@896p` weight 0.1575, 109740 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-people@896p` weight 0.1514, 105450 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d3-pexels@896p` weight 0.1074, 37417 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-megalith@896p` weight 0.0013, 1152 rows, resolutions: res1, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-pd12m@896p` weight 0.1117, 145832 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-vintage@896p` weight 0.0157, 16416 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p0@896p` weight 0.0324, 33858 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p1@896p` weight 0.0323, 33725 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p2@896p` weight 0.0323, 33716 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p3@896p` weight 0.0324, 33857 rows, resolutions: res1, res2, res3, res4, res5
- `${ARTFLOW_ROOT}/precomputed_dataset/d4-relaion-p4@896p` weight 0.0323, 33753 rows, resolutions: res1, res2, res3, res4, res5

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

### resolution 1 (image tokens 3136, 218780 captions)

Bounds minimise padded compute under the shape fit (padding overhead 0.5% of useful compute); sizes come from the shape fit for memory and the shape fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 29 | 1.68% | 8 | 8 | 27.75 | 1330.2 |
| 1 | 39 | 1.50% | 8 | 8 | 27.84 | 1336.5 |
| 2 | 52 | 0.93% | 8 | 8 | 27.95 | 1344.7 |
| 3 | 70 | 1.32% | 8 | 8 | 28.10 | 1356.2 |
| 4 | 80 | 1.62% | 8 | 8 | 28.18 | 1362.5 |
| 5 | 89 | 1.72% | 8 | 8 | 28.26 | 1368.3 |
| 6 | 99 | 1.55% | 8 | 8 | 28.34 | 1374.7 |
| 7 | 110 | 1.15% | 8 | 7 | 25.02 | 1211.1 |
| 8 | 124 | 0.82% | 8 | 7 | 25.12 | 1218.9 |
| 9 | 144 | 0.59% | 8 | 7 | 25.27 | 1230.2 |
| 10 | 172 | 0.53% | 8 | 7 | 25.47 | 1246.1 |
| 11 | 204 | 0.41% | 8 | 7 | 25.71 | 1264.4 |
| 12 | 251 | 0.17% | 8 | 7 | 26.06 | 1291.4 |
| 13 | 400 | 0.15% | 7 | 6 | 23.43 | 1184.3 |
| 14 | 489 | 0.11% | 7 | 6 | 23.99 | 1230.2 |
| 15 | 593 | 0.10% | 7 | 6 | 24.65 | 1284.8 |
| 16 | 716 | 0.09% | 7 | 6 | 25.43 | 1350.9 |
| 17 | 893 | 0.06% | 6 | 5 | 22.30 | 1209.9 |
| 18 | 1152 | 0.03% | 6 | 5 | 23.66 | 1334.0 |
| 19 | 2048 | 0.01% | 5 | 3 | 17.46 | 1090.8 |

### resolution 2 (image tokens 3108, 131072 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 0.7% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 0.66% | 8 | 8 | 27.50 | 1315.1 |
| 1 | 44 | 0.91% | 8 | 8 | 27.66 | 1327.1 |
| 2 | 61 | 0.79% | 8 | 8 | 27.80 | 1337.8 |
| 3 | 78 | 0.99% | 8 | 8 | 27.95 | 1348.7 |
| 4 | 92 | 1.04% | 8 | 8 | 28.06 | 1357.6 |
| 5 | 106 | 0.86% | 8 | 8 | 28.18 | 1366.6 |
| 6 | 123 | 0.68% | 8 | 7 | 24.92 | 1205.6 |
| 7 | 148 | 0.59% | 8 | 7 | 25.10 | 1219.7 |
| 8 | 174 | 0.56% | 8 | 7 | 25.29 | 1234.4 |
| 9 | 199 | 0.43% | 8 | 7 | 25.48 | 1248.7 |
| 10 | 238 | 0.22% | 8 | 7 | 25.77 | 1271.1 |
| 11 | 355 | 0.15% | 7 | 7 | 26.63 | 1339.4 |
| 12 | 419 | 0.13% | 7 | 6 | 23.38 | 1181.0 |
| 13 | 510 | 0.15% | 7 | 6 | 23.96 | 1228.0 |
| 14 | 604 | 0.13% | 7 | 6 | 24.56 | 1277.5 |
| 15 | 709 | 0.09% | 7 | 6 | 25.22 | 1333.9 |
| 16 | 820 | 0.07% | 6 | 5 | 21.78 | 1162.6 |
| 17 | 986 | 0.06% | 6 | 5 | 22.66 | 1240.3 |
| 18 | 1289 | 0.03% | 6 | 4 | 19.62 | 1110.9 |
| 19 | 2048 | 0.01% | 5 | 3 | 17.38 | 1076.2 |

### resolution 3 (image tokens 3108, 77581 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 0.8% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 27 | 0.47% | 8 | 8 | 27.51 | 1316.3 |
| 1 | 45 | 0.74% | 8 | 8 | 27.67 | 1327.7 |
| 2 | 64 | 0.50% | 8 | 8 | 27.83 | 1339.7 |
| 3 | 84 | 0.68% | 8 | 8 | 28.00 | 1352.5 |
| 4 | 101 | 0.53% | 8 | 8 | 28.14 | 1363.4 |
| 5 | 123 | 0.34% | 8 | 7 | 24.92 | 1205.6 |
| 6 | 157 | 0.40% | 8 | 7 | 25.17 | 1224.8 |
| 7 | 182 | 0.42% | 8 | 7 | 25.35 | 1239.0 |
| 8 | 207 | 0.33% | 8 | 7 | 25.54 | 1253.3 |
| 9 | 241 | 0.14% | 8 | 7 | 25.79 | 1272.8 |
| 10 | 344 | 0.13% | 7 | 7 | 26.55 | 1332.9 |
| 11 | 408 | 0.14% | 7 | 7 | 27.02 | 1370.9 |
| 12 | 470 | 0.11% | 7 | 6 | 23.71 | 1207.2 |
| 13 | 546 | 0.12% | 7 | 6 | 24.19 | 1246.9 |
| 14 | 636 | 0.10% | 7 | 6 | 24.76 | 1294.6 |
| 15 | 748 | 0.08% | 7 | 6 | 25.47 | 1355.1 |
| 16 | 872 | 0.06% | 6 | 5 | 22.05 | 1186.7 |
| 17 | 1048 | 0.04% | 6 | 5 | 22.98 | 1269.9 |
| 18 | 1323 | 0.02% | 6 | 4 | 19.76 | 1124.5 |
| 19 | 2048 | 0.00% | 5 | 3 | 17.38 | 1076.2 |

### resolution 4 (image tokens 3072, 527151 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 0.7% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 27 | 3.21% | 8 | 8 | 27.21 | 1293.7 |
| 1 | 45 | 4.73% | 8 | 8 | 27.36 | 1305.0 |
| 2 | 60 | 3.87% | 8 | 8 | 27.49 | 1314.4 |
| 3 | 76 | 3.90% | 8 | 8 | 27.62 | 1324.5 |
| 4 | 92 | 3.90% | 8 | 8 | 27.76 | 1334.7 |
| 5 | 109 | 3.31% | 8 | 8 | 27.90 | 1345.5 |
| 6 | 130 | 2.49% | 8 | 8 | 28.08 | 1358.9 |
| 7 | 154 | 2.25% | 8 | 8 | 28.28 | 1374.3 |
| 8 | 178 | 2.37% | 8 | 7 | 25.06 | 1216.3 |
| 9 | 203 | 1.79% | 8 | 7 | 25.24 | 1230.5 |
| 10 | 242 | 0.89% | 8 | 7 | 25.53 | 1252.7 |
| 11 | 358 | 0.86% | 7 | 7 | 26.39 | 1320.0 |
| 12 | 432 | 0.83% | 7 | 7 | 26.93 | 1363.7 |
| 13 | 513 | 0.70% | 7 | 6 | 23.75 | 1210.9 |
| 14 | 599 | 0.57% | 7 | 6 | 24.30 | 1255.8 |
| 15 | 704 | 0.46% | 7 | 6 | 24.96 | 1311.7 |
| 16 | 838 | 0.34% | 7 | 5 | 21.69 | 1154.3 |
| 17 | 1024 | 0.20% | 6 | 5 | 22.67 | 1241.2 |
| 18 | 1310 | 0.08% | 6 | 4 | 19.55 | 1104.9 |
| 19 | 2048 | 0.02% | 5 | 3 | 17.27 | 1064.1 |

### resolution 5 (image tokens 3072, 442779 captions)

Bounds minimise padded compute under the pooled fit (padding overhead 0.6% of useful compute); sizes come from the pooled fit for memory and the pooled fit for time.

| bucket | max_length | draw share | batch (memory) | batch (plan) | predicted GB | predicted ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 3.72% | 8 | 8 | 27.19 | 1292.4 |
| 1 | 36 | 4.86% | 8 | 8 | 27.29 | 1299.3 |
| 2 | 48 | 3.60% | 8 | 8 | 27.39 | 1306.9 |
| 3 | 63 | 2.99% | 8 | 8 | 27.51 | 1316.3 |
| 4 | 78 | 3.71% | 8 | 8 | 27.64 | 1325.8 |
| 5 | 90 | 3.39% | 8 | 8 | 27.74 | 1333.4 |
| 6 | 104 | 2.71% | 8 | 8 | 27.86 | 1342.3 |
| 7 | 123 | 2.04% | 8 | 8 | 28.02 | 1354.4 |
| 8 | 149 | 1.68% | 8 | 8 | 28.24 | 1371.1 |
| 9 | 180 | 1.46% | 8 | 7 | 25.07 | 1217.4 |
| 10 | 219 | 0.81% | 8 | 7 | 25.36 | 1239.6 |
| 11 | 327 | 0.81% | 8 | 7 | 26.16 | 1301.8 |
| 12 | 379 | 0.75% | 7 | 7 | 26.54 | 1332.3 |
| 13 | 448 | 0.70% | 7 | 7 | 27.05 | 1373.2 |
| 14 | 518 | 0.54% | 7 | 6 | 23.78 | 1213.5 |
| 15 | 594 | 0.44% | 7 | 6 | 24.26 | 1253.2 |
| 16 | 687 | 0.32% | 7 | 6 | 24.85 | 1302.6 |
| 17 | 836 | 0.20% | 7 | 5 | 21.68 | 1153.4 |
| 18 | 1113 | 0.09% | 6 | 5 | 23.14 | 1283.9 |
| 19 | 2048 | 0.02% | 5 | 3 | 17.27 | 1064.1 |

## Time alignment

Predicted throughput in samples per millisecond of the step's slowest
micro-batch: 0.0044 for the memory solution, 0.0055 as planned (+26.88%).

alignment at 1374.7 ms buys +26.88% predicted throughput over the memory solution

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
- Caption distribution: CaptionPolicy(kind='beta', beta_start=-1.0, beta_end=1.0, schedule='linear', early_at=0.5, short_reserve=0.2, short_threshold=256); progress [0.95, 1.0], 8 midpoints (stationary uses full-run 64-point average).
  Stage-progress averaging assumes equal exposure per progress point;
  changing emitted batch sizes can make actual exposure differ.
- Predicted queue-emitted mean micro-batch: 7.607902
  samples, using 1 / sum(p_i / B_i), not sum(p_i * B_i). This is a
  long-run mean, not a per-update floor; validate the actual sampler stream.
- The alignment model costs a step by its slowest micro-batch and ignores
  queue tails, compilation and synchronization, so treat its gain as a
  direction rather than a measurement.
