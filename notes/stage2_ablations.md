# Stage 2 — Ablation Runs @256p: configs, protocol, records

Working file for stage-2 execution and experiment records. The *why* lives in
`redesign_plan.md` (design-dimension ledger + Stage 2 section); this file is the
*how* and the running results. Fill the scoreboard and per-arm records as arms finish.

- Date opened: 2026-09-04
- Budget: ≤400 4090-h nominal (trough-hour semantics per redesign_plan; see §6)

## 1. Fixed protocol (every arm, no exceptions)

| Knob | Value |
|---|---|
| Data mix | stage-2 mix (§4) |
| Steps | screen 8K / confirm 16K / LR-probe 3K |
| Effective batch | 128 (Inspire 8×4090: micro 16 × 8; Andromeda: micro 8 × accum 16) |
| Seed | 42 |
| Conditioning | `fused` (pooled text + t in AdaLN) |
| Objective | rectified flow, logit-normal(0,1), shift=1 @256p |
| Caption | curriculum 0.0→1.0 over arm steps, dropout 0.1 |
| AdamW LR | 3e-4, linear_cosine, warmup 500 (screen) / 1000 (confirm), min 0.5e-4, wd default 0.01 |
| EMA | 0.999 (screen) / 0.9995 (confirm), eval always on EMA |
| Precision | bf16 end-to-end |
| Text encoder | Qwen3-0.6B frozen, k=28 unless the arm says otherwise |

## 2. Telemetry & eval (rebuilt 2026-09-04)

Old infra problems (why we rebuild): eval subset rotated with step
(`pipeline.py` `end_idx = current_step % ...`) → not comparable across steps/arms;
sample noise unseeded → grids not comparable; FID@200 samples meaningless;
aspect squash to 299×299; CLIP weights downloaded at eval time (offline = hang)
and English CLIP is useless on zh prompts.

Rebuilt three-piece eval, all **fixed and deterministic**:

1. **`eval/loss` probe** (primary comparator, new `src/evaluation/eval_loss.py`):
   fixed 512 held-out samples from `light-eval@256p` (fixed indices), deterministic
   caption (field idx 1, fallback 0), no dropout, fixed t grid {0.15, 0.4, 0.65, 0.9},
   fixed noise per sample. Forward-only, every 250 steps. Logs `eval/loss` +
   `eval/loss_t{015,040,065,090}`.
2. **Fixed sample grids** (rewritten sample path): prompt suite
   `data/eval/prompts_v1.jsonl` (24 prompts, tagged style/anatomy/zh/aspect),
   per-prompt fixed seed (hash of prompt id) → identical noise across arms and steps.
   50-step Euler ODE, aspect per prompt's `aspect` field mapped to nearest bucket.
   Logged as swanlab images with prompt captions, every 1000 (screen) / 2000 (confirm)
   steps.
3. **End-of-arm KID** (torchmetrics, reuse `metrics.py::calculate_kid`): 2,000 fakes
   from fixed light-eval captions (fixed seeds) vs the full 2,371 real, subset_size 100,
   log mean±std. Preprocess: aspect-preserving short-side-299 resize + center crop.
   **FID dropped** at this sample size (biased); CLIP dropped for screens (zh prompts,
   weak EN CLIP); heavy eval at stage 4/5 revisits both with ≥10K samples.

Training-side logging additions: `train/grad_norm` (from clip return),
`train/samples_per_sec`, `train/step_time_s` — the throughput table fills itself.

SwanLab conventions:

- Project **`artflow-stage2`** (new `--swanlab_project` arg, default `artflow`).
- Run names `s2-<axis>-<variant>`, e.g. `s2-mod-layer`, `s2-stream-hybrid`,
  `s2-opt-muon-lr02`. Full argparse config is auto-logged by `init_trackers`.
- Compare arms inside the project dashboard; per-arm verdict goes into §7 records.
- Andromeda: verify swanlab login works; fallback `SWANLAB_MODE=offline` + sync later.

## 3. Platform rules

- Fair (deciding) arms: Inspire 4090 only. Screens may co-run 2 arms × 4 GPUs when
  the queue favors partial nodes; GPU-h cost is identical.
- Andromeda (4060 Ti): 2.1 RoPE gate + 2.3a exit qualitative only. Its numbers inform,
  never decide.
- All Inspire arms scheduled into night troughs; Andromeda avoids late night.

## 4. Stage-2 data mix

New mix (domain targets D1 .15 / D2 .20 / D3 .15 / D4 .50, within-domain ∝ source size;
rationale: hero intent is art-forward without repeating the old run's 10%-art failure):

```
$W/precomputed_dataset/d1@256p:0.150
$W/precomputed_dataset/d2-wikiart@256p:0.191
$W/precomputed_dataset/d2-museum@256p:0.009
$W/precomputed_dataset/d3-human@256p:0.076
$W/precomputed_dataset/d3-people@256p:0.074
$W/precomputed_dataset/d4-vintage@256p:0.094
$W/precomputed_dataset/d4-zimage@256p:0.022
$W/precomputed_dataset/d4-megalith@256p:0.003
$W/precomputed_dataset/d4-inat@256p:0.001
$W/precomputed_dataset/d4-pd12m@256p:0.079
$W/precomputed_dataset/d4-relaion@256p:0.301
```

2.4 old-recipe arm (world .8 / art .1 / portrait .1, within-group ∝ size):
d4_{vintage .151, zimage .035, megalith .004, inat .002, pd12m .126, relaion .482},
d1 .029, d2_wikiart .068, d2_museum .003, d3_human .051, d3_people .049.

## 5. Arm table

Params measured by instantiation (fused conditioning; 2026-09-04). GPU-h estimates
assume 8 samples/s/GPU pre-stage-3 (GPU-h = steps×128/(8×3600)); **recalibrate after
the first measured arm**. Eval overhead ~10% included in the right column.

| Arm | Config | Params | Steps | Platform | Depends on | Est. GPU-h |
|---|---|---|---:|---|---|---:|
| 2.1 rope-smoke | wiring smoke, 300 steps, Monet mini set (not an arm) | 459.0M | 0.3K | Andromeda | code prep | free |
| 2.1 rope-A/B | old vs new RoPE, h1024 d24 all-single mod=none (neutral shape, runs before 2.2a), 6K steps, then 256p→640p transfer sampling | 459.0M | 2×6K | Inspire | rope-smoke | ~2×24 |

2.1 moved from Andromeda to Inspire (2026-09-04): 6K steps at batch 128 on the 4060 Ti
would take >24 h/arm plus nightly stops — too slow for a gate. Two 4090s in parallel
settle it overnight; budget headroom (400 h) covers it.
| 2.2a mod-none | h1024 d24 all-single, mod=none | 459.0M | 8K | Inspire | 2.1 | ~32 |
| 2.2a mod-layer | same shape, mod=layer | 383.4M | 8K | Inspire | 2.1 | ~32 |
| 2.2b wide | h1152 d24, @2.2a-winner | 484.9M | 16K | Inspire | 2.2a | ~63 |
| 2.2b deep | h1024 d30, @2.2a-winner | 478.0M | 16K | Inspire | 2.2a | ~63 |
| 2.2c big | h1152 d33 (~664M), @2.2a-winner | 664.3M | 16K | Inspire | 2.2a | ~63 |
| 2.2c small | h1024 d25 (~399M), @2.2a-winner | 399.2M | ~26.6K (fvcore-calibrated iso-FLOP) | Inspire | 2.2a | ~63 |
| 2.2d all-single | h1024 d30 | 478.0M | 8K screen | Inspire | 2.2a | ~32 |
| 2.2d hybrid | h1024, 8 double + 14 single (mod=layer on both) | 477.9M | 8K screen | Inspire | 2.2a | ~32 |
| 2.2d all-double | h1024, 15 double | 478.0M | 8K screen | Inspire | 2.2a | ~32 |
| 2.2d confirm | 2.2d winner | — | 16K | Inspire | 2.2d screen | ~63 |
| 2.3a exit k∈{8,16,28} | h1024 d24 mod-layer, 4K each, eval-loss separation | 383.4M | 3×4K | Inspire (relocated from Andromeda 2026-09-05, see note) | 2.1 | ~9 |

2.3a relocation note (2026-09-05): Andromeda holds only the 459-sample Monet
mini set; farming 3 arms out over daytime windows (plus a mix download) would
take ~2 weeks for a qualitative screen. §3 already lets screens co-run on
Inspire; all three k arms share platform/mix/protocol so the comparison stays
internally fair. ~3 GPU-h/arm in the night trough is negligible. Pre-registered
shape (mod-layer) is kept — 2.3a was designed to run parallel to 2.2a and the
k-axis interacts weakly with modulation; caveat noted in its verdict.
| 2.3b exit confirm | winner-config @ best k (only if 2.3a ambiguous) | — | 8K | Inspire | 2.2, 2.3a | ~32 |
| 2.4 mix-old | §4 old recipe @ h1024 d24 2.2a-winner mod | — | 8K | Inspire | 2.2a | ~32 |
| 2.4 mix-new | §4 new mix, same shape | — | 8K | Inspire | 2.2a | ~32 |
| 2.5 muon-LR probe | 2.2-winner arch, Muon LR ∈ {0.01, 0.02, 0.04} | — | 3×3K | Inspire | 2.2b/d | ~38 |
| 2.5 muon confirm | Muon @ best probed LR vs AdamW (reuse the 2.2-winner's 16K run as baseline) | — | 16K | Inspire | probe | ~63 |

Fallback shapes if `mod=none` wins 2.2a (iso-param re-matched): 2.2b h1152 d20 (484.8M)
vs h1024 d25 (477.9M); 2.2c/2.2d re-derived the same way before launch.

**Shape table, pre-derived by instantiation on 2026-09-05 (param_count.py;
targets: 2.2b ≈ 485/478M, 2.2c ≈ 664/399M iso-FLOP, 2.2d ≈ 478M across
schedules). Winner's modulation applies to every block in every arm:**

| Arm | mod=layer wins (scenario A) | params | mod=none wins (scenario B) | params |
|---|---|---|---|---|
| 2.2b wide | h1152 d24 | 484.87M | h1152 d20 | 484.84M |
| 2.2b deep | h1024 d30 | 477.96M | h1024 d25 | 477.92M |
| 2.2c big (16K) | h1152 d33 | 664.26M | h1152 d27 | 652.26M |
| 2.2c small (iso-FLOP steps) | h1024 d25, 26.6K | 399.20M | h1024 d21, 25.9K | 402.31M |
| 2.2d all-single (8K screen) | h1024 d30 (=2.2b deep; reuse its run) | 477.96M | h1024 d25 (=2.2b deep; reuse) | 477.92M |
| 2.2d hybrid (8K screen) | d2x8+s14, mod=layer both | 477.96M | d2x8+s9, mod=none both | 477.92M |
| 2.2d all-double (8K screen) | d2x15, mod=layer both | 477.96M | d2x13, mod=none both | 496.82M (+4%, no integer layer lands nearer; documented tolerance) |
| 2.4 mix arms | h1024 d24 mod=winner, 8K (2.2a shapes: layer 383.4M / none 459.0M) | | | |

Note: h1024 d25 at mod=layer (399.2M, 2.2c small) and at mod=none (477.9M,
2.2b deep/2.2d all-single) are different configs — don't conflate. h1040 with
16 heads is invalid (head_dim 65 odd); heads stay 16 everywhere (head_dim
h/16 = 64/72/…, always even → RoPE-valid).

**Baseline reuse decisions (2026-09-05, paper trail for the scoreboard):**
- 2.3a k=28 arm: s2-exit-k28 is NOT launched — the s2-mod-layer run (identical
  config: h1024 d24 mod=layer, same mix/protocol, exit at last hidden state =
  k=28 behavior) provides the k=28 trajectory at every step up to 8000. 2.3a
  compares k=8/k=16 (new 4K arms) against s2-mod-layer's 4K/8K eval-loss marks.
- 2.2d all-single screen: NOT launched separately — s2-deep (h1024 d30
  mod=layer, 16K) IS the all-single schedule at the 2.2d target shape; its 8K
  trajectory serves as the screen, and if deep also wins 2.2b its full 16K run
  doubles as the 2.2d confirm. Only s2-hybrid (d2x8+s14) and s2-double (d2x15)
  are new 8K screen arms.
- 2.3a relocated from Andromeda to Inspire (see §5 note): screens may co-run on
  Inspire; three k arms share platform/mix/protocol.

Total (with all-double screen, without 2.3b): ≈ **570 GPU-h**; drops to ~505 if
all-double is cut, ~475 if screens go to 6K. Over the 400 nominal — sanctioned by
trough-hours semantics; revisit after the first arm recalibrates throughput.

**Recalibrated 2026-09-04 (s2-rope-old measured):** 1.85 s/it @ batch 128, 459M,
256p on one 4090 ≈ **69 samples/s/GPU** — the 8 samples/s planning assumption was
~8.7× too pessimistic (it came from 4060 Ti-scale smoke numbers). 6K steps ≈ 3.1 h
train + probe/grid/ckpt overhead ≈ 3.5–4 h/arm. Recalibrated ledger: 2.1 ≈ 2×3.6,
2.2a ≈ 2×4.5, 2.2b ≈ 2×9.6, 2.2c ≈ 2×13, 2.2d ≈ 3×4.8+9.6, 2.4 ≈ 2×4.5,
2.5 ≈ 3×2+10 → total ≈ **110 GPU-h**, comfortably inside the 400 h budget.

## 6. Execution order

1. **Code prep** (§8) + Andromeda smoke → 2.1 gate (Inspire, 2 arms overnight).
2. 2.2a (Inspire night 1; 2 arms, ~8h wall) ∥ 2.3a on Andromeda.
3. 2.2b + 2.4 pair (night 2-3); 2.2d screens + 2.2c (night 3-4).
4. 2.2d confirm; then 2.5 probes → 2.5 confirm at the final arch (night 5-6).
5. Decision memo → redesign_plan.md; hero recipe seeds stage 4.

## 7. Records

### Smoke (Andromeda, 2026-09-04) — PASS

- 300 steps on the 459-sample Monet mini set, batch 4×accum2 @2.15 it/s.
- Verified: fused conditioning, eval-loss probe (fixed 64 samples), prompt grids
  @100/200/300, checkpoints @100/200/300, end KID (0.657±0.009, meaningless value,
  pipeline OK), full resume from ckpt-100 → 150 (step/scheduler/RNG/EMA restored,
  swanlab run `9m7ofplb` continued via runtime.json).
- Fixes found by smoke: `eval_loss.py` needed an explicit bf16 autocast context
  (training gets it from `accelerator.accumulate`, eval does not); torch-fidelity
  inception weights can't be downloaded on Andromeda (GitHub blocked) — pre-staged
  to `torch_home/hub/checkpoints` and TORCH_HOME pinned by the arm scripts.

### Scoreboard (fill as arms finish)

| Arm | eval/loss@end | KID (end) | samples/s/GPU | verdict |
|---|---|---|---|---|
| s2-rope-old | 0.95207 | 0.02247±0.00504 | ~56 (micro 8 post-restart) | lose (ladder tie → prior) |
| s2-rope-new | 0.95230 | 0.02294±0.00509 | ~55 (micro 8 post-restart) | **WINNER** (ladder tie → Qwen-Image prior) |
| s2-mod-none | 0.94405 | 0.01949±0.00499 | 55.2 | lose (narrow; tie-break also → layer) |
| s2-mod-layer | 0.94373 | 0.01901±0.00363 | 55.5 | **WINNER of 2.2a** → all later arms mod=layer |

256p probe is a statistical tie (Δ0.0002, 0.025%; per-t grid also identical:
t015 0.8935/0.8938, t040 0.8045/0.8047, t065 0.9735/0.9739, t090 1.13676/1.13676).
KID ties within std.

**640p transfer check (job `s2-transfer-r2`, 2026-09-04): TIE, non-decisive.**
24 fixed prompts × 5 buckets at 2.5× latent scale (640×640 / 832×480 / 480×832 /
720×560 / 560×720, snapped to even latents), 50-step Euler ODE off the final EMA
ckpts. Both arms collapse identically into structureless high-frequency texture in
every bucket — palettes track the prompts, global structure is gone. No
differential signal. Sanity: the same ckpts' 256p step-6000 grids are coherent, so
the stack is fine and this is genuine length-extrapolation failure (6.25× tokens).
Consequences: (a) zero-shot 2.5× transfer fails regardless of RoPE variant →
progressive resolution training (256p→640p→896p) is mandatory, and neither RoPE
buys extrapolation at this scale; (b) 640p is too far out to discriminate — the
tie-break descends to 480p/384p/320p (ladder below). Grids:
`$W/runs/stage2/transfer/s2-rope-{new,old}-640p/transfer_bucket*.png`.
(Bug fixed en route: scaled latent dims must stay divisible by patch 2 — odd 105
broke patchify in r1; `transfer_check.py` now snaps to even.)

**480p→384p→320p ladder (job `s2-transfer-ladder`, 2026-09-05): TIE all the way →
new RoPE wins on the Qwen-Image adoption prior (user-amended rule).**
Same 24 prompts × 5 buckets at 1.875×/1.5×/1.25×, identical per-prompt seeds:

- 480p (1.875×): both arms still fully collapsed (brush texture only) — tie.
- 384p (1.5×): both arms partially structured; same panels hold up, same panels
  degrade, no consistent differential — tie.
- 320p (1.25×): both arms coherent and panel-by-panel identical in quality (same
  seeds → same compositions, same minor artifacts) — tie.

RoPE choice is therefore not decidable by 6K-step 256p training dynamics or by
near-range extrapolation; per the amended rule the Qwen-Image production prior
decides: **all subsequent stage-2 arms use centered-grid RoPE**
(`--rope_centered_grid`). Ladder grids:
`$W/runs/stage2/transfer/s2-rope-{new,old}-{480,384,320}p/transfer_bucket*.png`.

### Launched arms

#### s2-rope-new — 2026-09-04 — Inspire 4090 (job `s2-rope-new`, restarted as `s2-rope-new-r1`)
- config: h1024/d24 all-single, mod=none, fused, qkv_bias, gated FFN,
  centered-grid RoPE; AdamW 3e-4 cosine (min 0.5e-4, warmup 500), batch 16×8,
  6000 steps, EMA 0.999, seed 42; §4 stage-2 mix; exit layer 28 (default).
- swanlab: artflow-stage2 / s2-rope-new (run id `6u5hzgqp`)
- results: eval/loss@end 0.95230; KID 0.02294±0.00509; finished 6000/6000 at
  15:01 UTC.
- **OOM at ~step 1300** (2026-09-04 10:13; rope-old followed at step 1417, 10:27):
  root cause found by code inspection + memory math — `F.scaled_dot_product_attention`
  was called with a bool key-padding mask, which forces PyTorch off flash onto the
  math backend, materializing B·H·S² attention weights **per layer, saved for
  backward**. At seq ≈ 2K (long-caption batches: text up to 1024 tokens + 1024 image
  tokens) that is ~2 GiB×24 layers ≈ 45 GiB — matches both OOM reports exactly
  (45.1/45.3 GiB allocated). This was also the old run's unexplained "memory spike"
  that forced the batch-size cuts. Fix (`dit_blocks.py::sdpa_with_pad_mask`): convert
  the bool mask to an additive bias in q.dtype and pin the memory-efficient kernel
  (never materializes S² scores); mask=None still takes flash. Numerical parity vs an
  fp64 reference is pinned in `tests/test_sdpa_mask.py` (incl. a CUDA regression test
  asserting linear memory at seq 4K — passes on Andromeda). train.py now also logs
  `train/mem_peak_gb`, `train/mem_alloc_gb`, `train/txt_seq_len` every step so any
  future spike shows up in swanlab correlated with batch composition. Both arms
  restarted from ckpt-1000 with the fix (jobs `s2-rope-new-r2`, `s2-rope-old-r1`).

  Quantitative confirmation (`s2-mem-probe-r1`, fixed code, real mix, micro 16,
  150 steps): run peak **21.52 GiB**; post-step steady state 6.34 GiB; per-step
  peak tracks `txt_len` linearly (top-10 steps all txt_len 233–274 → 20.5–21.5 GiB);
  checkpoint-save adds **+0.00 GiB** (never a suspect). Worst-case extrapolation
  (txt_len 1024, seq 2048) ≈ 30 GiB — micro 16 is safe again post-fix; micro 8
  remains the common.sh default for headroom on larger 2.2 arms. Report:
  `$W/runs/stage2/mem_profile/mem_report.json` + `mem_snapshot.pickle`.

  Metric-artifact notes for reading the 2.1 sps curves (explained, fixed for
  later arms): (a) `train/samples_per_sec` was computed from wall-clock step
  time *including* the eval probe (every 100 steps, ~14 s) and grid/ckpt
  (every 1000) — each probe injected one ~16 s "step" into the EMA, producing
  the periodic cliff→slow-climb sawtooth (~65→63); fixed by resetting the
  clock after eval blocks. (b) The big cliff at rope-new step ~1415 is the
  restart seam: three processes (orig micro-16, r1 micro-8, r2) log into one
  swanlab run, each resuming from step 1000, so series overlap; r1/r2 at
  micro 8 run ~56 samples/s vs the original ~63. Not a training event.
- verdict: **WINNER of 2.1** — 256p tie, 640p tie, 480p/384p/320p ladder tie →
  user-amended final tie-break (Qwen-Image adoption prior) picks centered-grid RoPE.
  All subsequent arms run with `--rope_centered_grid`.

#### s2-rope-old — 2026-09-04 — Inspire 4090 (jobs `s2-rope-old`, `s2-rope-old-r1`)
- config: identical to s2-rope-new except legacy corner-anchored RoPE.
- swanlab: artflow-stage2 / s2-rope-old (run id `4f5jcjvy`)
- throughput: 1.85 s/it @ batch 128 micro 16 (69 samples/s) pre-fix;
  2.2–2.3 s/it at micro 8 post-restart.
- results: eval/loss@end 0.95207; KID 0.02247±0.00504.
- notes: OOM at step 1417 (masked-SDPA quadratic memory), resumed from
  ckpt-1000 with the fix; finished 6000/6000 at 14:47 UTC.
- verdict: lose — 256p tie, 640p tie, 480p/384p/320p ladder tie → final tie-break
  (Qwen-Image prior) favors centered-grid RoPE. Legacy RoPE retired from stage 2.

#### s2-mod-none — 2026-09-05 — Inspire 4090 (jobs `s2-mod-none`)
- config: h1024/d24 all-single, mod=none, fused, qkv_bias, gated FFN,
  centered-grid RoPE (2.1 winner); AdamW 3e-4 cosine (min 0.5e-4, warmup 500),
  batch 8×accum16 (=128) on 1 GPU, 8000 steps, EMA 0.999, seed 42;
  §4 stage-2 mix; exit layer 28 (default).
- swanlab: artflow-stage2 / s2-mod-none (run id `ppn097xnxxwpq8ivfjini`)
- results: eval/loss@end 0.94405; per-t @end t015 0.88692 / t040 0.79557 /
  t065 0.96172 / t090 1.13198; KID 0.01949±0.00499; 55.2 samples/s; peak mem
  18.1 GiB; grad_norm decay 0.56→0.10. Finished 8000/8000 06:30 local.
- verdict: lose — see the 2.2a decision note.

#### s2-mod-layer — 2026-09-05 — Inspire 4090 (jobs `s2-mod-layer`)
- config: identical to s2-mod-none except single-stream modulation = layer
  (shared per-layer mod MLP); params 383,443,168 as pre-registered.
- swanlab: artflow-stage2 / s2-mod-layer (run id `ql0c6u3viwa29pbimnasl`)
- results: eval/loss@end 0.94373; per-t @end t015 0.88673 / t040 0.79460 /
  t065 0.96124 / t090 1.13236; KID 0.01901±0.00363; 55.5 samples/s; peak mem
  16.6 GiB. Finished 8000/8000 06:25 local.
- verdict: **WINNER of 2.2a**.

**2.2a decision (2026-09-05)**: eval/loss curve (per-1000 table in swanlab):
layer leads early (Δ-0.016 @1K, -0.006 @2K — faster convergence, plausible:
shared mod MLP gets 24× gradient signal), configs cross ~3.5-5K (none ahead
≤+0.0003), then layer re-takes the lead at 7-8K (-0.0002/-0.0003, ≈ the 2.1
noise floor of ±0.00025). Per-t: t040 (most informative mid-noise regime) is
layer's — consistently better 5/5 probes from 3K on, growing -0.0002 →
-0.0010 (~0.12%, above floor); t015/t065 tie late; t090 (high-noise tail) is
none's by +0.0004 (< floor). KID agrees in direction (0.0190 vs 0.0195,
within std but same sign). Layer also +0.6% samples/s and -8% peak mem (16.6
vs 18.1 GiB). Not a slam dunk on the primary axis alone — but directionally
consistent, t040 persistent, KID agreeing, and the pre-registered tie-break
(literature prior mod→layer, PixArt/DiT-Air) points the same way → **mod=layer
locked for every subsequent arm**. Scenario A shapes in the §5 table apply.
Grid eyeball note: step-8000 grids are on swanlab for a human look (in-session
image tooling was unavailable at verdict time; numerical proxy — both KIDs
~0.019, inter-arm pixel diff ≈ 4% mean, no collapse signal — is consistent
with two converged models).

2.2a gate protocol: 256p eval/loss curve separation (noise floor from 2.1:
Δ <0.025% at 6K was noise), KID end must agree in direction, grids get a human
look. Tie → literature prior mod→layer (PixArt/DiT-Air). Winner's modulation
feeds 2.2b/2.2c/2.2d param re-matching (mod=none fallback shapes pre-derived in
§5) and 2.4. — **RESOLVED: layer wins (2026-09-05).**

### Per-arm record template

```
### s2-<axis>-<variant> — <date> — <platform>
- config: (h/d/mod/stream/exit/optimizer/LR/steps/batch/EMA)
- swanlab: <run url or name>
- eval/loss curve: <value @25/50/75/100% steps>
- KID end: <mean±std> ; grids: <link/step>
- throughput: <samples/s/GPU>, step time <s>, GPU-h actual <n>
- notes: <anomalies, spikes, restarts>
- verdict: <win/lose/tie vs sibling arm, one line>
```

### Decision rules (pre-registered)

- Primary: fixed `eval/loss` at matched steps; separation must exceed run noise
  (judge on the curve, not the last point).
- Confirm: end-of-arm KID direction must agree; grids get a human look for artifacts.
- Tie → pick the cheaper/simpler arm, then the literature prior
  (mod→layer per PixArt/DiT-Air; stream→single-heavy hybrid per DiT-Air/FLUX;
  exit→k=28 fallback).
- **2.1 tie-break ladder (user-amended 2026-09-05)**: old/new RoPE tie on
  compute complexity (identical rotary application; only grid-index construction
  differs, ~56 vs ~55 samples/s measured = noise). Descend 480p → 384p → 320p
  transfer grids until one arm shows clearly fewer artifacts; if still tied at
  320p, **new RoPE wins on the Qwen-Image adoption prior**.
- 2.5 Muon wins only if it beats AdamW on eval/loss at matched steps **and** its
  step-time overhead stays <15%.

## 8. Code-prep checklist (done 2026-09-04; 129 tests + smoke pass)

- [x] `src/evaluation/eval_loss.py`: fixed-set eval-loss probe + train.py wiring
      (`--eval_loss_interval`, logs every 250 steps).
- [x] Sample-grid path rewrite: prompts_v1.jsonl + per-prompt fixed seeds +
      aspect→bucket mapping; replace the rotating-subset generation in
      `run_evaluation_light` for stage-2 runs (keep old path callable).
- [x] `metrics.py`: aspect-preserving 299 resize + center crop; KID subset_size 100;
      inception weights pre-downloaded to a local cache dir (no runtime download);
      FID/CLIP removed from the light path.
- [x] train.py: `--swanlab_project` (default `artflow`), `--text_encoder_exit_layer`,
      `--optimizer {adamw,muon}`, `--muon_lr`, `train/grad_norm` +
      `train/samples_per_sec` logging, EMA/scheduler args verified per §1.
- [x] `src/utils/encode_text.py`: `exit_layer` support via `output_hidden_states`
      (k indexes `hidden_states[k]`, k=28 = last layer = current behavior).
- [x] `src/train/muon.py`: Muon with Newton–Schulz on **chunked** fused matrices
      (qkv → q/k/v; modulation 6×dim → per-piece), Moonlight scaling
      (update ≈ 0.2·√max(m,n) RMS-matched), weight decay, AdamW for embeddings /
      patch conv / final_layer / norms / biases / t & c MLPs. DDP-safe (NS after
      grad allreduce, deterministic).
- [x] RoPE centered-grid option in `MSRoPE` (`--rope_centered_grid`), keeping the
      old 0-based grid selectable for the 2.1 A/B; text stays on the fixed diagonal.
- [x] `scripts/stage2/<run_name>.sh` per arm, generated from one template.
- [x] Prep smokes: 100-step run on Andromeda (all new flags on), swanlab offline
      fallback verified, inception cache seeded on both platforms.
