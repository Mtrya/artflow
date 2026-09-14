# Stage 4 — Scaling experiment and evaluation design

Status: design proposal, 2026-09-12, revised same day per user directives.
Refines [stage4_plan.md](stage4_plan.md)
§5–§7 into an executable ladder, an evaluation protocol, and a fitting
procedure, from a fresh literature pass (every claim below was verified by
fetching the source; the two survey reports are in the session record).
Run-specific observations below are recorded with their experiment identifiers.
GPU-hour envelopes are estimates, not the spend ledger. The selected hero
configuration and remaining launch work are maintained in [hero_recipe.md](hero_recipe.md).

**User directives (2026-09-12), applied throughout:** keep the stage lean
and reach the hero run sooner; where a literature prior exists for an
axis, adopt it instead of re-measuring (repeated-data tolerance is the
headline case, §1.2); the plan's envelopes may relax. Two clarifications
from the same discussion:

- *Decoupling data amount from training amount.* Three knobs, not two:
  N (model size), S (sample draws consumed — equal to step count under
  the shared bucket plan), and U (unique eligible images). Epochs = S/U
  is the coupling between them. Arm B varies S at fixed U (training
  amount); a corpus arm would vary U at fixed S (data amount) — and is
  descoped because the repetition prior covers our regime.
- *Are the imported α/β valid for a DiT?* They come from DiT-family image
  models in the first place (Abra's flow-matching T2I transformers,
  Liang et al.'s DiTs; Chimera's image models agree at α≈0.315). What
  does not transfer is the intercepts (E, A, B), which we fit on our own
  corpus. The pinned exponents are additionally tested, not trusted: the
  XL rung is invisible to the fit, which predicts its loss before it
  trains.

## 1. What the literature decides, and what it cannot

### 1.1 Imported with confidence

- **Form of the law.** `L(N,D) = E + A·N^-α + B·D^-β` fits flow/diffusion
  transformers well. Abra (60M–2B flow-matching T2I,
  [2608.17286](https://arxiv.org/html/2608.17286)) and Chimera
  ([2607.28611](https://arxiv.org/html/2607.28611)) independently find
  α ≈ 0.315 and β ≈ 0.34–0.39 for images, i.e. near-balanced allocation
  exponents (a ≈ 0.51, b ≈ 0.49) and a compute-optimal tokens-per-parameter
  ratio that is nearly constant in N.
- **Compute-optimal TPP ≈ 200 image tokens/param at 512px; 165 at 256px;
  ~250 at 768px** (Abra §4.6; the resolution series assumes a=b=0.5).
  Evaluated at our hero budget, Abra's `N_opt = 0.0301·C^0.4951` puts the
  optimum at ≈350–430M for a 1,600–2,200 GPU-h hero — our 485M base sits
  slightly above it, on the undertrained side, which Abra shows is the
  dangerous side (2× overtraining costs <0.5% loss; undertraining is
  catastrophic). The ladder's job is to confirm or refute this direction
  on our corpus, not to rediscover the exponent.
- **Fitting procedure.** Fit the EMA-model loss (Abra: per-step batch noise
  is 4–7× the reducible improvement over 10k steps; Liang et al.'s
  EMA-of-training-loss is noisier). Interpolate each run's checkpoints into
  a continuous loss-vs-tokens curve with PCHIP. Pin α, β to the imported
  values and fit only E, A, B — a 5-parameter fit is not identifiable on
  the points we can afford. Do **not** use iso-FLOP parabolas for
  allocation numbers: [2603.22339](https://arxiv.org/html/2603.22339v1)
  shows Approach-2 parabolas are biased on asymmetric surfaces and
  off-centre grids, exactly our regime. Uncertainty: bootstrap over model
  configurations + leave-one-model-out, per Chimera (their image fit:
  R²=0.993, 0.22% LOMO error).
- **Validation by prediction.** Hold out the ladder's top rung from the
  fit, predict it, then compare (Liang, video-DiT, and Abra's
  drop-the-largest check all do this).
- **LR transfer across the ladder.** Our Muon already folds the Moonlight
  rule into the update (`scale = 0.2·√max(m,n)`, `src/train/muon.py:147`),
  i.e. update RMS is matched to AdamW's ~0.2 by construction. Per
  [Moonlight](https://arxiv.org/html/2502.16982v1) and
  [Kimi K2](https://arxiv.org/html/2507.20534v1), with this normalization
  the tuned LR transfers **unchanged** across widths. Video-DiT's fitted
  η_opt ∝ N^-0.19 (image) is a ~15% effect over our range — within noise
  of the fit. Decision: **muon_lr = 0.02 and aux AdamW lr = 3e-4 constant
  across all rungs**, no per-size retune; a single short LR sanity run at
  the smallest size lives in the calibration package as insurance.
  Constant head dim (72) and a near-constant width/depth ratio across
  rungs keep the family muP-clean (DiT-μP,
  [2505.15270](https://arxiv.org/html/2505.15270v2); Liang et al. show
  mixing aspect ratios obscures the trend).
- **Batch size.** No diffusion critical-batch-size measurement exists
  (Abra says so explicitly); LLM evidence says CBS scales with data
  duration, not N ([2410.21676](https://arxiv.org/html/2410.21676v4)).
  Decision: the shared bucket plan fixes per-bucket micro-batches for all
  rungs; effective batch is tuned only via `gradient_accumulation_steps`
  (stage4_plan §7.2). Fixed plan + fixed accumulation ⇒ fixed samples per
  optimizer step across sizes, so **fixed sample exposure = fixed step
  count** — the ladder is a fixed-steps ladder, which is what a
  step-time-bound 4090 budget actually pays for (Abra's own choice).

### 1.2 Not decidable from the literature — measured locally or flagged

- **Repeated data: covered by prior, watched in-ladder (user directive
  2026-09-12).** The controlled literature stops at single-digit epochs
  for image diffusion (Abra's largest is 7.7), but the production record
  is much stronger: DiT-XL/2 (675M) trained 7M steps × batch 256 on
  ImageNet's 1.28M images — **~1,400 epochs** — with FID improving
  monotonically ([DiT](https://arxiv.org/pdf/2212.09748v1)); its
  follow-ups train hundreds of epochs as a matter of course. Masked
  diffusion *language* models tolerate ~100–500 epochs
  ([2507.15857](https://arxiv.org/html/2507.15857v7)). The LLM
  data-constrained literature does find superlinear repetition damage
  ([2605.01640](https://arxiv.org/html/2605.01640v1)), but at our hero's
  ~100–200 epochs over a deduplicated 1.5M-image corpus we sit far inside
  the demonstrated-safe region. **Decision: no dedicated corpus-size arm;
  U is not a scanned axis.** The watch is free: arm B's S-rung extension
  reaches 42 epochs on the full pool, and a deviation of its loss from
  the fitted curve there is the repetition warning sign. Memorization
  spot-checks join the hero-run monitoring rather than a Stage-4 arm.
  **2026-09-13 user directive: the dataset axis is dropped from Stage 4
  entirely** — the corpus is fixed at the current ~1.5M images; the
  scaling runs answer only the three-way tradeoff among model size,
  training steps and batch size.
- **Resolution curricula.** No scaling law for progressive resolutions
  exists. Production recipes (PixArt-α, Seaweed, Z-Image, LLaDA-Image)
  consistently spend **~60–75% of steps at the lowest resolution**, but
  every one confounds resolution with data changes. Abra's per-resolution
  TPP table plus a local continuation-vs-scratch control (§4, arm C) is
  what we get.
- **Loss ≠ capability allocation.** Abra: FID/KID and CLIPScore/CMMD
  demand opposite allocations; Chimera: treat distributional/preference
  metrics as post-training metrics. We fit the loss surface for the
  *budget allocation*. Capability is observed through the monitoring panel (§5.3),
  not forecast as a launch gate.
- **Our corpus is art-heavy and bilingual**; irreducible loss E is a
  property of the data distribution, so we import exponents and ratios,
  never absolute loss values.
- **Muon in an image-diffusion scaling law is unpublished.** We keep the
  family fixed (Stage 2's decision) and accept the intercept shift; the
  exponents are optimizer-insensitive in the LLM evidence Moonlight
  provides.

## 2. Budget arithmetic (corrected for the measured Stage-3 stack)

Measured on one 4090 with the current bucketed stack: ~80 samples/s at
485M/256p steady state (Stage-3 gate: 73; Stage-3.5 arms: ~85–89), and
~13 samples/s at 640p (probe-640p). Planning rates, scaling throughput as
485/N and assuming the text-encoder/VAE overhead share stays put:

| quantity | value |
|---|---|
| samples/GPU-h, 485M @256p | ~290,000 |
| image tokens/GPU-h, 485M @256p | ~7.4e7 |
| DiT training FLOPs/GPU-h, 485M @256p (6·N·seq, seq≈400) | ~2.5e17 |
| 450 GPU-h probe budget | ~2× the pre-Stage-3 estimate |

Consequence: the ladder below costs ≈45% less than a naive Stage-2-era
estimate; the 450-hour cap buys a real (if compact) surface fit.

## 3. The ladder

All rungs: 256p, fixed steps **15,000** (≈1.25e7 sample draws with the
shared plan, TPP quoted on image tokens), same mix (run-256p.toml with the
d3-synth-v2 swap, §6), beta-ramp caption policy + log2 loss weighting,
EMA 0.999, warmup 500, seed 42. Rungs (head_dim 72 throughout):

| rung | shape (h×d, heads) | N | ratio h/d | TPP | est. GPU-h |
|---|---|---|---|---|---|
| S | 720×12, 10 | 97M | 60 | 33 | ~9 |
| M | 1008×16, 14 | 249M | 63 | 13 | ~22 |
| L | 1152×24, 16 | 485M | 48 | 6.6 | ~43 |
| XL | 1440×20, 20 | 632M | 72 | 4.2 | ~56 |

- **Arm A (size axis):** S, M, L at fixed steps. ~74 GPU-h. Answers
  "loss vs N at fixed exposure."
- **Arm B (data axis):** S at 5× exposure (75,000 steps, TPP 165 -
  exactly the imported 256px optimum, so the smallest rung brackets the
  literature's optimum from both sides) and M at 3× (45,000 steps, TPP
  39). ~110 GPU-h. With arm A that is 5 (N,D) points for the
  3-parameter fit (E,A,B), spanning 0.9 decades in N and 0.7 in D.
- **Holdout:** XL is fitted-blind, predicted, then trained and compared.
  It doubles as the direct test of "would a bigger hero pay off." ~68
  GPU-h, charged to the held-out-validation package.
- **Aspect-ratio note:** L's ratio (48) sits off the 60–63 line. If the
  fit shows L as an outlier, the fallback reading uses S/M/XL (60/63/60)
  and treats L as a same-shape reference to every Stage-2/3.5 result we
  already own. Widths were chosen so this ambiguity costs nothing.
- S sits near the floor where Abra excluded its 60M as out-of-regime;
  the fit must survive leave-one-rung-out, and S is the first rung to
  drop if it misbehaves.

### 3.1 LR sanity check (resolved 2026-09-12)

All ladder rungs inherit Stage 2's Muon LR of 0.02. A 3,000-step S-rung
control at 0.01 (`s4-lr010-0912` vs `s4-s-15k-0912`, identical otherwise)
compared on the fixed eval-loss probe at shared steps:

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| lr 0.01 | 1.552 | 1.514 | 1.116 | 0.997 | 0.966 | 0.947 |
| lr 0.02 | 1.722 | 1.975 | 1.605 | 1.088 | 0.980 | 0.951 |

Early-step differences swing wildly (up to −30% at step 1500), showing
how noisy single-probe comparisons are in the first fifth of a rung; by
step 3000 the gap is 0.4%, far inside that noise and under the 1%
alarm threshold. **0.02 stands**; no rung is re-launched. Caveat recorded:
3,000 steps cannot exclude a late-training LR effect; arm B's 75k S-rung
at 0.02 gives the long-horizon reference if the question resurfaces.

**Transient gradient spikes (observed 2026-09-13, user-flagged).** Every
rung shows occasional grad_norm excursions to 50-230 against a 0.2-0.6
median. Two were strong enough to kick train/loss back to its step-1
value before recovering within a few hundred steps (l-15k at step ~5000,
loss 2.058 with grad_norm 230; m-45k shows repeated 80-150 spikes at
2427/5723/9268/11122 without a sampled loss excursion). No run entered
an unrecoverable plateau — unlike the XL-at-0.02 failure, which did.
Hero-recipe implication (user's read, agreed): consider shaving the peak
Muon LR slightly or lengthening the warmup ramp; the hero run's smoother
early LR schedule already mitigates part of the risk, so this is a hedge,
not a blocker. Decide when the hero recipe is frozen, with the bs-arm
results in hand (larger batches may also tolerate the spike rate
differently).


### 3.2 Batch-size arms (user directive 2026-09-13)

Batch size becomes a scanned axis, at 256p only, S-rung size, fixed total
samples (702 × 15000 = 10.53M), LR held at 0.02:

| run | accum (per rank × 4 GPUs) | samples/step | steps | warmup |
|---|---|---|---|---|
| `s4-s-bs2x-0913` | 8 × 4 = 32 | ~1404 | 7500 | 250 |
| `s4-s-bs4x-0913` | 16 × 4 = 64 | ~2809 | 3750 | 125 |

**Incident (2026-09-13):** the first launch of both bs arms ran the
wrong model size. `ladder-s-bs2x.toml`/`ladder-s-bs4x.toml` had no
`[model]` section, so they inherited the base config's L shape
(1152/16/24, ~485M params) instead of the S rung (720/10/12). Both jobs
(`s4-s-bs2x-0913e`, `s4-s-bs4x-0913e`) were killed and relaunched from
scratch on 4 GPUs with explicit `[model]` sections
(`s4-s-bs2x-0913f`/`s4-s-bs4x-0913f`). The SwanLab runs reuse the same
run names, so their curves contain a wrong-size prefix segment — when
comparing, only use data logged after the 4-GPU relaunch (the size
mismatch is also visible as the jump in `train/grad_norm` scale and in
the logged model-param count).

Warmup is scaled to keep warmup *samples* constant (500 × 702). The
reference is the ladder's own S-rung at 15k (accum 16, ~702/step). Read:
loss-vs-samples-seen at 1×/2×/4× batch gives the local curvature of the
batch axis for the hero allocation. Caveat: LR is not re-tuned per batch
size (Muon LR-vs-batch scaling is unpublished for image diffusion), so a
large-batch arm that loses may be LR-limited rather than batch-limited;
if bs4x underperforms clearly, one follow-up at raised LR is the cheap
disambiguation.

Cross-resolution effective-batch constraints for the hero recipe
(user's heuristic, 2026-09-13): 640p runs keep ≥ 80% of the 256p
effective batch, 896p ≥ 75%. At 256p the ladder runs ~702 samples/step
(accum 16 × measured mean micro-batch 43.89).

After the scaling tradeoff is resolved there is **one limited final
infra-optimization pass** before the hero run (user directive
2026-09-13): re-screen the bucket plan at the final model size, tune the
accumulation settings against the 80%/75% floors, minor config/code
tweaks — then the hero recipe is frozen.

## 4. Adjacent arms (inside the plan's existing packages)

Per the user's 2026-09-12 directive (relax the plan, use literature priors
where they exist, reach the hero run sooner), the two data-package arms
from the draft are descoped:

- ~~Corpus-size contrast~~ — covered by the repeated-data prior (§1.2);
  the in-ladder watch replaces it.
- ~~Mixture contrast~~ — no strong prior either way, but the Stage-2/3.5
  art-forward mix is already validated end-to-end and the mixture is
  re-adjustable between hero stages without re-running Stage 4. Kept as
  an option only if ladder results look anomalous.

What remains:

- **Resolution transfer** (resolution package): one experiment, per the
  user's 2026-09-13 directive (supersedes the earlier two-arm draft).
  Continue arm-A's M-rung 15k checkpoint at 640p for the same image-token
  count (2e6 samples ≈ 3100 steps), then evaluate at both 640p and 896p;
  the 896p read is zero-shot transfer (eval-loss probe + sample panel,
  no 896p training). `gradient_accumulation_steps` is raised 16 → 56 so
  the effective batch (~646 samples/step) stays above the 80%-of-256p
  floor instead of collapsing with 640p's smaller micro-batches. No
  640p-from-scratch control: the question is how much of a 256p budget
  transfers upward, not the continuation ordering. The 640p loss is
  compared only within 640p (per-resolution curves; losses never compared
  across resolutions). Checkpoint: M-rung 15k endpoint, confirmed by the
  user on 2026-09-13. Runs on 4 GPUs with per-rank accumulation 14
  (56 total, same effective batch as the single-card draft).

  Launch-record caveat (2026-09-13): the first attempt reused the
  crash-resume launcher (`s4_resume_rung.sh`, which forces
  `--resume_full`), so the restored global_step 15000 already exceeded
  max_steps 3100 and the loop never ran. The transfer must use plain
  `--resume` — weights+optimizer carried, schedule and global_step
  restart (`jobs/transfer_640p_4gpu.sh`). Same class of config bug as
  §3.2's missing `[model]` section bit here too: the transfer config
  relied on a `ladder-m.toml` layer the resume launcher never applied,
  and now spells out the M shape itself.
- **LR sanity** (calibration package): one 3,000-step S-rung at
  muon_lr 0.01 vs 0.02, pick by eval-loss probe; the ladder then holds
  the winner constant. ~4 GPU-h.

## 5. Evaluation protocol

### 5.1 Eval-loss probe (the scaling target)

- EMA weights for all reported/fitted values; raw weights as a diagnostic
  series only (SD3 and Abra both evaluate EMA).
- Fixed paired draws: the probe's (image, caption, timestep, noise)
  tuples are frozen for the whole stage, so all checkpoints of all rungs
  are evaluated on identical draws. Timesteps sampled from the training
  sampler (logit-normal(0,1), shift=1), not a uniform grid.
- Cadence: every 250 steps, stored per run — 60 points per 15k rung, far
  above the 12–24 needed for PCHIP interpolation.
- Per-length-band values are diagnostics; the fitted scalar is the
  probe mean. Never compared across resolutions.

### 5.2 KID

- n = 2,000 generated samples against the fixed reference set, reported
  as mean ± std over 40×512 subset draws (the KID paper's own block
  protocol; unbiased MMD, and at n=2k their curve is flat). Differences
  under 2σ are ties. FID is not reported at this n (model-dependent bias;
  >20k images needed — [2401.09603](https://arxiv.org/html/2401.09603v1)).
- One fixed guidance for ladder comparisons; a small CFG sweep is
  reserved for the finalist, because optimal guidance falls with size and
  training (Abra).

### 5.3 Capability monitoring

Use the small bilingual short/long-prompt panel in [hero_recipe.md](hero_recipe.md)
with fixed seeds and sampling settings. Record both joint image success and
diagnostic atoms (anatomy, contact, style, layout); compare checkpoints directly.
The execution agent performs VLM-first review and surfaces uncertain or systematic
failures to the user. A confident judge output alone is not ground truth.

There is no per-atom scaling fit, numerical capability target, or formal
qualification suite. [stage4_eval_freeze.md](stage4_eval_freeze.md) describes the
small monitoring panel. Its purpose is to detect regressions and document observed
abilities, not to certify or forecast a final capability level.

### 5.4 Resolution time-shift (fixed 2026-09-13)

The shift transform had its direction inverted relative to SD3's intent.
This repo's flow path is `z_t = (1-t)·noise + t·data` (t=0 noise, t=1
data); SD3 uses the opposite convention, and its Eq. 23 push toward t=1
means *toward noise* there. The old code applied SD3's formula verbatim,
so at 640p/896p training inputs were made *cleaner* instead of noisier.
Fixed to `t' = t / (s − (s−1)·t)` (SD3 Eq. 23 applied to 1−t), verified
against the paper's derivation.

The resolution schedule is also no longer a heuristic: the old log-linear
1→3 interpolation between 256 and 4096 tokens is replaced by SD3 Eq. 23's
own law, `shift = sqrt(n_tokens / 256)` (pixel counts are proportional to
token counts at fixed VAE/patch). Consequences: 256p is unchanged
(shift 1.0, so all running ladder rungs are unaffected); 640p square goes
2.32→2.5, 896p square 2.81→3.5. SD3's human-preference study found
little quality difference among shifts above 1.5 and used 3.0 at 1024²
where the formula gives 4.0, so the formula values sit inside their
accepted range.

Training, eval loss and the sampling schedule all share
`shift_timesteps`, so they stay internally consistent. Caveat: eval
losses at 640p/896p measured before this fix and after it are *not*
comparable — different objective, different timestep distribution.

## 6. Bucket-plan status (the "one usable plan" decision)

- **256p: `bucket_plans/256p-k10-13src.screened.json`** — boundaries from
  the corpus length distribution, micro-batches screened on the 485M
  model, validated in the arms and the full-mix run. Safe for every
  rung ≤ 485M. This is the Stage-4 early/mid-phase plan at 256p.
- **640p: `bucket_plans/640p-k10.full.sized.json`** — same boundaries
  method, analytic sizes, full-mix validation passed. Same coverage.
- **XL is 632M (h1440×d20)**, confirmed by the user on 2026-09-12. The
  sparse probe (12 isolated points at 256p, buckets 0/5 on two aspects;
  bucket 9 could not be isolated — its draw rate is too low to fill a
  timing batch) refit the memory model to **const = 12.29 GB, slope =
  0.00104 GB/sample/token** (max residual 0.2 GB), versus 9.5 / 0.00101
  at 485M: the per-token slope is essentially architecture-independent at
  fixed head_dim, the constant scales with parameter count. Every bucket
  of the existing 256p plan predicts ≤ 40.2 GB isolated reserved at 632M,
  so the plan is reused unchanged, conditional on the mixed-stream
  validation (`diag-fullmix-632m-0912`) — the retained-state surcharge is
  the empirical term and is what that run signs off.
- **XL training instability at the ladder LR (measured 2026-09-12).** The
  first XL attempt at muon_lr 0.02 destabilized: grad_norm crept up through
  warmup (max 12.9 over steps 350–500, vs ≤4.6 for M/L) and spiked to
  56–137 right after warmup ended; train/loss jumped from ~1.00 onto a
  ~1.1–1.3 plateau and stayed there while M/L sat at ~0.95. The recipe's
  LR transfer boundary therefore lies between 485M and 632M — itself a
  load-bearing fact for hero sizing. XL was restarted as
  `s4-xl-15k-lr015-0912` at **muon_lr 0.015 on 4 ranks** (accumulation 4
  per rank, keeping the ladder's ~830 samples/optimizer step; the gradient
  is the exact global weighted mean at any world size). Read the rung as
  "632M at its feasible LR", not as a same-recipe ladder point. New inline
  health telemetry (update/weight RMS ratio per optimizer group, attention
  QK gain, EMA-live distance; `[telemetry] health_interval`) watches the
  restart; grad_norm gave the early warning this time.
- 896p: boundaries from the 896p sidecars once precompute finishes, sizes
  from the same constants (256p→640p transfer was exact on the 485M
  model), one validation run. Mid-stage, not now.
- Final hero plan: re-solved at Stage-4 exit for the chosen model and
  per-stage mixes, per the pipeline. The ladder plans above are
  deliberately *not* re-optimized mid-stage: fixed bucket semantics are
  what make rungs comparable.

## 7. Budget against the 450-hour cap

| package | plan envelope | this design |
|---|---:|---:|
| Calibration/profiling (incl. LR sanity, judge calibration) | 35 | ~39 |
| Size × exposure ladder (arms A+B) | 120 | ~184 |
| Corpus + mixture contrasts | 80 | 0 (descoped, §4) |
| Resolution continuation | 90 | ~44 |
| Held-out validation (XL) + finalist checks | 85 | ~68 |
| Reserve | 40 | ~40 |
| **total** | **450** | **~375** |

The ~375-hour envelope is not measured spend and cannot establish the remaining
infra allowance. Use the actual job ledger against the separate 450-hour cap.
Descoped experiments do not authorize new sweeps or transfers to the hero budget.

## 8. Open items before the first rung launches

1. **d3-synth-v2 swap.** The v2 rescue set exists at all three
   resolutions (10,893 rows at 256p vs v1's 20,000 — 45% was filtered as
   junk) with length sidecars, but run-256p.toml / run-640p.toml still
   point at `d3-synth`. Swap the mix entries before the ladder starts
   (the v1 rows stay on disk for the already-finished arms).
2. **XL go/no-go** — resolved: XL is h1440×d20 (632M), user decision
   2026-09-12; probe constants refitted (§6).
3. **Evaluation-suite freeze** (prompt manifests, rubrics, seeds) is the
   calibration package's deliverable and gates arm-A reads.
4. **The 896p precompute is still running** under cron watch; the 640p
   transfer arm does not depend on it, the 896p retention checks do.

## 9. Hero direction (user decision 2026-09-13, late)

The hero model is **LD-533M**: the L shape (1152/16, 24 single-stream
blocks) plus one double-stream block (`double_stream_depth=1`),
532.7M params — the user wants the hero under 600M and sized for a
~2,200 GPU-hour run. Rationale captured from the discussion: at probe
scale the eval-loss/KID gaps between rungs are too small to see
capability cliffs (the 15k-step sample grids show S<M<XL face quality
clearly while losses sit within 0.4%), so marginal-loss-per-GPU-hour
arguments underweight size; the user prefers spending the relaxed budget
on a bigger model trained long rather than a smaller model at the
loss-knee.

A short validation rung (`s4-ld533-15k-0913`, 4 ranks, lr 0.02) probes
only two things and is stopped at ~2k steps: (a) stability of the
double-block shape at the ladder LR (the XL@0.02 failure pattern would
show within ~1k steps), and (b) per-step cost + per-bucket memory peaks
to calibrate the hero bucket plan and the 2,200-hour budget split across
the 256p/640p/896p stages. It is deliberately not a full 15k ladder
point — the shape decision is already made.

The executable hero decisions are maintained in [hero_recipe.md](hero_recipe.md):
533M, eight ranks, 256p→640p→896p with a 75:20:5 step split, accumulation 1/5/7,
mean effective-batch targets 640/512/400, Muon peak LR 0.02, 5,000-step warmup,
and whole-run cosine decay to 5% of peak. Caption progress remains continuous
across resolution changes. The eligible pools and mixtures differ by resolution;
use the recipe's counts, not a shared 1.5M-row assumption. The bounded infra pass
validates plans and end-to-end rates, then sets total steps inside 2,200 GPU-hours
including overhead. It does not assume zero input stalls or exact budget exhaustion.
