# Muon weight growth and normalization probes

> Archived record. Current pretraining decisions and status are in
> [the Ascend plan](../ascend_pretraining_0924.md); old launch commands and
> configuration switches require the source revision used for that run.

2026-09-24. Third link in the chain, after
[H200 spike root-cause](h200_spike_root_cause_0924.md) (the triggering *update*)
and [timestep-factor note](timestep_factor_0924.md) (factor 1000 is insufficient
on its own). Weight growth is measured; its causal role in creating downstream
sensitivity is a hypothesis. See the [stability assessment](ascend_stability_review_0924.md)
for remaining risks and proposed actions after the completed branch-norm probes.

## 1. Measurement: Muon-routed weights inflate, AdamW-routed ones do not

Read straight from the H200 hero's three retained checkpoints
(`runs/h200-hero-256p/checkpoint_step_*/model.safetensors`, live weights, via
`inko-patrol`: `/inspire/qb-ilm/.../ky26021/artflow/venv-precompute-gpu/bin/python`).
Per-tensor RMS = `sqrt(mean(w^2))`, averaged over the matrices in each group.

| group | step 12000 | 14000 | 16000 |
|---|---:|---:|---:|
| **mean over Muon-routed 2D weights** | **0.471** | **0.542** | **0.635** |
| mean over AdamW-routed weights | 0.368 | 0.377 | 0.403 |
| `c_mlp.0` (AdamW) | 0.0239 | 0.0243 | 0.0246 |
| `c_mlp.2` (AdamW) | 0.0286 | 0.0285 | 0.0285 |
| `txt_pooled_proj` (AdamW) | 0.0276 | 0.0272 | 0.0269 |
| `x_embedder` (AdamW) | 0.0703 | 0.0694 | 0.0685 |
| `blocks.12.attn.proj` (Muon) | 0.580 | 0.687 | 0.951 |
| `blocks.24.attn.proj` (Muon) | 0.319 | 0.368 | 0.686 |
| `blocks.12.attn.qkv` (Muon) | 0.544 | 0.636 | 0.763 |

Two facts:

- **The conditioning head's weights are frozen in scale** (`c_mlp`, `txt_pooled_proj`
  sit at ~0.027, i.e. at their Xavier init; `xavier_uniform_` on fan 1152 gives
  RMS 0.0295). So the spike is *not* caused by the conditioning weights growing.
- **Every Muon-routed matrix keeps growing**, and the growth is fastest exactly
  where the activations blow up: `blocks.*.attn.proj` — the attention-output
  projection that writes into the residual stream — grows 1.6× (block 12) to
  2.2× (block 24) per 4000 steps, while `qkv` and the MLP matrices grow ~1.25×.
  At step 12000 the Muon mean is **16× its init** and still climbing ~+15 % / 2000 steps.

## 2. Why: Muon's update magnitude is decoupled from both the gradient and the weight

`src/pretrain/muon.py:165-177` scales the orthogonalized update by
`0.2 * sqrt(max(m, n))`, so the update has RMS ≈ 0.2 per element *by
construction* (Moonlight's Adam-RMS matching). At `muon_lr = 0.02` that is

    |ΔW|_rms per step = 0.2 · 0.02 = 4e-3

against an initial weight RMS of 0.0295 — **13 % of the whole weight, every step**,
largely independent of gradient magnitude. Five NS iterations give an
approximate update RMS; actual updates also depend on momentum, rank and the
LR schedule. Decoupled weight decay (`p.mul_(1 - lr*wd)`) and the alignment of
each update with the weights determine norm growth together. There is no
general LR-independent equilibrium `0.2/wd`; see the corrected derivation in §8.

Moonlight's constant 0.2 matches typical AdamW update RMS, not weight RMS.
It does not assume weights have RMS 1 or establish a 30× error for Xavier
initialization. Comparing LR conventions is still important: at width 1152,
our square-matrix update is 6.79× the Keller reference at the same scalar LR.
The current review documents this comparison and the relevant DiT priors.

## 3. Proposed connection to the captured spike

The residual stream has **no scale control**: `dit_blocks.py` uses pre-norm only
(`x = x + gate·out`, `:972`/`:983`), with no sandwich norm on the branch output
and no residual scaling; qk-norm exists, branch norm does not. So each block's
contribution to the residual stream is proportional to its weight norms, and the
measured activation scale follows the weights:

| step | blocks 0-11 | blocks 12-22 | ratio |
|---|---:|---:|---:|
| 2000 | 110.1 | 331.8 | 1.00 |
| 4000 | 718.3 | 1596 | 4.81 |
| 8000 | 3632 | 8372 | 25.2 |
| 12000 | 1.35e4 | 4.76e5 | **1433.6** |

`model_output` stays at 5.0–5.4 throughout. Its roughly constant RMS does not
establish invariant predictions or loss, or show that gradients cannot oppose
the internal growth.

Two observations relevant to the proposed mechanism:

1. The conditioning path's inputs are structurally thin. Masked mean pooling
   (`utils/encode_text.py:270-274`, mathematically correct) averages over the
   retained slice; a `1/sqrt(L)` variance argument would require assumptions
   about token independence that have not been established here. Measured
   centered-ratio 0.766 at `x_embedder` → 0.584 at `txt_embedder` → **0.0975 at
   `txt_pooled_proj`** → 0.064 through `c_mlp` → 0.035-0.085 at all 26
   `modulation` heads. `c` is a shared constant carrying ~6 % per-sample signal.
2. The captured spike has high sensitivity to `c`. An AdamW conditioning
   update produces a shared shift of 0.206, versus cross-sample spread 0.196.
   A shared shift preserves pairwise differences; it does not erase
   conditioning. Its effect depends on the downstream network. The replay
   shows that this particular displacement triggers the captured spike.

That is the fusion of this note with the causal ablation in
`h200_spike_root_cause_0924.md` (applying *only* `c_mlp.2`'s update turns
grad_norm 0.292 into 55.76). It also explains the cross-platform results: the
spike is a property of (weights × batch), so a 4090 reproduces H200's spike at
cosine 0.9998, and fp32 Newton-Schulz shifts *when* the knife edge is crossed
without removing it.

**Status of the claim — a possible contributor (user correction,
2026-09-24).** The 4×4090 hero run's *live* loss curve was recorded, and on it
53.5 k steps at lr 0.02 / warmup 5000 / `muon_wd` 0.01 produced no tier-2/3
event; the EMA contamination only makes that run's checkpoint unusable, it does
not weaken the observation. An earlier version of this section argued the
4090's evidence was void — that was wrong. Note also that `grad_spike_skip` is a
local, uncommitted addition to `configs/hero.toml` ("Ascend stack insurance"),
so the 4090 control ran with the guard *off* and still stayed clean.

Weight growth and downstream sensitivity are plausible contributors. The
measurements do not establish that weight growth is necessary for spikes,
that curvature is determined by activation RMS, or that every platform event
shares the captured mechanism. The same captured weights and batch reproduce
the H200 spike on 4090; this establishes that an H200-specific arithmetic fault
is not required. Short fp32-NS outcomes do not establish a durable cure.

## 4. Intervention status after the probes

| Change | Evidence and limit |
|---|---|
| Branch-output RMSNorm | Improves measured residual scale and evaluation loss. Learned gates and norm gains remain; no general conditioning-Jacobian bound follows. Leading hero candidate. |
| Conditioning-input LayerNorm | Reduces negative preactivations but does not restore hidden cross-caption variation. Combined loss is slightly better; a durable advantage is not established. Leave off in the proposed first recipe. |
| Stronger Muon decay | Plausible, with decay already 0.01. The old predicted norm ceiling was incorrect (§8). Inspect actual radial updates before selecting a stronger value. |
| Muon step calibration | Compare effective updates under the actual shape convention. Multiplying by weight RMS would be a different optimizer; Keller's shape scaling is not weight-relative scaling. |

The [pre-experiment review](ascend_stability_review_0924.md) gives the proposed combined
recipe and one short check that can inform several decisions.

## 5. Side finding: the LR-curve shape in the swanlab panel is the *old* bug

`train.py:953-959` documents a fixed defect: `AcceleratedScheduler.step()`
advances the inner scheduler `num_processes` times, compressing the schedule by
the rank count — "warmup 500 ending at step 125 on 4 ranks, cosine hitting its
floor at max_steps/4", with the floor landing exactly on
`min_learning_rate·lr_ratio` (0.0033 for Muon, 5e-5 for the aux AdamW). That is
precisely the panel shape seen for `hero-val-256p-hostfix`.

The current code does **not** have it: schedulers are built bare
(`train.py:707-730`), never passed to `accelerator.prepare` (`:960`), and stepped
once per accumulation boundary (`:1738-1741`, gated on
`optimizer_step_boundary = (micro_count+1) % gradient_accumulation_steps == 0`,
`:1459-1461`). So a 4-rank run with `lr_warmup_steps = 500` peaks at 500, not
125. A curve that still peaks at `warmup/n_ranks` means the running code came
from a bundle older than this fix.

One real consequence that remains, and was observed live on hero7: on
`grad_spike_skip` the optimizer step is skipped (`:1707-1709`) while the
scheduler still advances and `global_step` increments (`:1738-1741`). At a 100 %
skip rate (hero7 from step 10600) the LR schedule burns down while the weights
are frozen.

## 6. The ablation flags and the four-arm probe

Two switches were added to `ModelConfig` (both default off = the shipped
recipe), plumbed through `train.py` into `ArtFlow`:

- **`branch_norm`** — an affine `RMSNorm` on each block's attention and MLP
  branch output, applied before the adaLN gate
  (`dit_blocks.py`, `SingleStreamDiTBlock` and `DoubleStreamDiTBlock`).
- **`cond_norm`** — an affine `LayerNorm` on the conditioning MLP's input
  (`artflow.py`). LayerNorm and not RMSNorm: RMSNorm is a pure rescaling, so it
  leaves c's direction and therefore its centered ratio untouched, while the
  measured pathology is a *DC offset* (the sinusoid's slow dimensions sit at
  cos ~ 1, so the embedding carries a large sample-independent component that
  lands `c_mlp.0` in SiLU's saturated tail). Only mean subtraction removes that.

Cost: `+59,904` parameters for `branch_norm` (0.011 %) and `+4,608` for
`cond_norm` (0.001 %) on the 532.7 M hero model. Both are initialisation
transparent — with adaLN-zero every gate is 0, so the eval-mode output is
bit-identical to the baseline at init, at production shape and in the unit
tests (`tests/test_conditioning_flags.py`).

`scripts/ascend/branch_norm_probe.sh` runs all four arms — `off`, `branch`,
`cond`, `both` — sequentially on 16 Ascend ranks, 2000 steps of 256p, then one
`layer_probe.py --weight-stats` pass per arm. Two deliberate deviations from the
hero recipe, so that a 2000-step window is informative rather than empty:
`lr_warmup_steps` 200 instead of 20000 (at the hero's warmup the LR is still at
10 % of peak at step 2000), and `max_steps` 200000 with `stop_at_step` 2000
(`max_steps` is the *LR horizon*, so the LR stays at peak instead of decaying
away). Everything else — optimiser, batch, bucket plan, spike-skip — is the
hero's. What the arms are read for: per-block activation scale (the ×2/2000-step
residual growth), the `c`/modulation centered ratios, per-module weight RMS, and
grad-norm/spike-skip counts.

## 7. Where the Muon step size is controlled

All of it is in this repository; no lower-level code is involved. PyTorch has no
Muon in our pinned stack (`torch==2.9.0` in the Ascend environment) and
`src/pretrain/muon.py` is a hand-written `torch.optim.Optimizer` subclass.

1. **Direction** — `_zeropower_via_newtonschulz5` (`muon.py:44-65`): quintic
   Newton-Schulz, coefficients `(3.4445, -4.7750, 2.0315)`, 5 steps, bf16
   (`_NS_DTYPE`; `MUON_NS_FP32=1` promotes it). Magnitude-invariant.
2. **Magnitude** — `muon.py:165-177`, the whole story:
   - unchunked: `scale = 0.2 * math.sqrt(max(m, n))`
   - chunked (`_chunk_hint`: fused qkv → 3, modulation → 6, gated up_proj → 2):
     `scale = 0.2 * math.sqrt(max(rows, n))`
   - applied as `dest.add_(updated, alpha=-lr * scale)`.
   The `0.2` is Moonlight's "match AdamW's update RMS"; Keller Jordan's original
   is `sqrt(max(1, rows/cols))`, which for a 1152² matrix is 1 — i.e. a per-step
   displacement RMS of `lr/sqrt(1152) = 0.03·lr`, **6.7× smaller** than what we
   run. A step that is *relative* to the weight would multiply `scale` by
   `|W|_rms`.
3. **Chunking changes the effective LR** for fused tensors: a `[3456, 1152]`
   fused qkv gets `0.2·sqrt(3456) = 11.76` whole but `0.2·sqrt(1152) = 6.79`
   chunked, a 1.73× difference. It is a side effect of `_chunk_hint`, not an
   independent knob.
4. **`lr` itself** — `muon_lr` in the TOML configs, driven by
   `build_linear_cosine_scheduler` and stepped once per optimizer step
   (`train.py:1738-1741`); the accumulation boundary is
   `(micro_count+1) % gradient_accumulation_steps == 0` (`:1459-1461`). It is
   *not* gated on `grad_spike_skip`, so a skipped step still advances the LR.
5. Our local `torch 2.13` venv does ship an official `torch.optim.Muon`, whose
   `adjust_lr_fn` is exactly this choice: `None`/`"original"` →
   `sqrt(max(1, A/B))`, `"match_rms_adamw"` → `0.2·sqrt(max(A, B))`, anything
   else → 1.0 — and whose default `weight_decay` is 0.1, not 0.01. It has no
   chunking, so it is not a drop-in for `CMuon`; it is only a reference point.

## 8. What `muon_wd` actually does (corrected September 24)

For the additive optimizer step `D` before decay, let `a = 1 - lr*wd`.
The exact identity is

    W_next = a*W + D
    ||W_next||² = a²||W||² + 2*a*<W,D> + ||D||²

The old equilibrium table implicitly assumed persistent outward alignment.
It does not apply generally. With independent zero-mean updates of RMS
`0.2*lr`, fixed LR, and vanishing expected cross term, the illustrative
stationary RMS is `0.2*lr / sqrt(1-a²)`, approximately
`0.2*sqrt(lr/(2*wd))`. Real momentum updates need not satisfy those assumptions.
Measure the cross term rather than predicting a ceiling from `wd` alone.

The measured weight RMS 0.471 at 12k remains valid. Comparing it with
`0.004*sqrt(12000)` ignores warmup, decay and correlated updates, so their
numerical proximity does not establish a random walk. A 5000-step decay time
constant at peak LR also does not imply negligible decay across 20k warmup;
the relevant quantity is accumulated `sum(lr_t*wd)`.

Our decay is already 0.01. Raising it may help, but neither a required value
6.7 nor a predicted equilibrium of 20 follows from these measurements.

## 9. Probe result: arm `off` (baseline), 2000 steps of 256p at full LR

`ascend-bnprobe3-256p`, arm `off`, 16×910B, 1.29 s/it, steady 721 samples/s,
peak 51 GiB, 46 min. Live weights, probe batch = the fixed 4-row selection,
`t = 0.5` (`layer_probe.py --weight-stats`; 112 K `probe-off-2000.json`).

**Weights.** Muon-routed blocks 0.553 RMS against the 0.0295 Xavier init —
**18.7× init, after 2000 steps** — while the AdamW-routed modules sit at 0.0346
(1.2× init). Flat across depth (`blk00` 0.370, `blk06` 0.582, `blk12` 0.594,
`blk18` 0.549). The 26 blocks' aggregates are dominated by their 2D matrices
(~19.9 M of ~20.0 M parameters per block), so a block's `rms` is the Muon
weight scale to within 0.2 % and the norm weights at 1.0 do not contaminate it.

**Conditioning path.** `centered_ratio` = cross-caption RMS / RMS:

| tensor | rms | ratio | frac < −5 |
|---|---:|---:|---:|
| `x_embedder` | 0.407 | 0.759 | — |
| `txt_embedder` | 4.08 | 0.707 | 0.105 |
| `t_embedder` | 0.707 | 0.000 | — |
| `txt_pooled_proj` | 0.920 | 0.289 | — |
| `c_mlp.0` | 7.35 | **0.0246** | **0.553** |
| `c_mlp.1` (SiLU hidden) | 0.183 | **0.0093** | — |
| `c_mlp.2` (= c) | 0.143 | **0.0120** | — |
| 26 × `modulation` | 0.776 | **0.0147** | — |
| `model_output` | 1.09 | 0.739 | — |

Three things this pins down:

- **Many preactivations are in SiLU's negative tail at step 2000**: 55 % are
  below −5. SiLU approaches zero from below in that tail; its minimum near
  −0.278 occurs around input −1.278, not below −5. The negative-tail fraction
  alone does not establish loss of useful conditioning.
- **`t_embedder.ratio` is exactly 0** because the probe holds `t` fixed: the
  ratio measures cross-*caption* spread, not t-sensitivity. Read it as
  "per-sample signal", never as "does this depend on t".
- **Relative cross-caption variation is small**: 1.2 % for `c`, 1.5 % for
  modulation. These ratios are not update safety margins. A shared displacement
  preserves pairwise differences; loss response to the actual displacement is
  the relevant functional check.

**Residual stream.** Per-block activation RMS ramps 594 (block 0) → 5500
(blocks 16-18) → 5700 (block 24): the same monotone "the trunk grows its own
scale" profile as the 20 k-warmup run, reached in 1/10th of the steps.

Predictions for the three treatment arms, stated before reading them:

| arm | signature that would count as working |
|---|---|
| `branch` | substantially reduced activation ramp; this was the intended signature, not a guaranteed bound on depth accumulation or gates |
| `cond` | `c_mlp.0.frac_lt_neg5` → ~0, and `c_mlp.1/2` + `modulation` ratios well above 0.009-0.015 |
| `both` | both of the above; activation ramp flat *and* conditioning margin restored |

## 10. Probe incident: `branch_norm` OOMs the stock micro-batch

`ascend-bnprobe3-256p` arm `branch` died 20 minutes in with

    RuntimeError: NPU out of memory. Tried to allocate 88.00 MiB
    (NPU 0; 60.96 GiB total capacity; 59.26 GiB in use)

against arm `off`'s measured peak of **51.0 GiB allocated / 51.4 GiB reserved**
(throughput summary, 16×910B 64 GB). Sandwich norm keeps two more per-block
tensors alive for the backward pass (`x_attn` and its normalised copy, times
attn and MLP, times 24 single-stream blocks), which is enough to push a recipe
that already sits at 80 % of the card over the line. **That is itself a
constraint on fix (1) for the hero recipe**: the 640p/896p stages run at larger
micro-batches and the headroom was already the tightest part of the Ascend
configuration.

The follow-up pass (`ascend-bnprobe4-256p`) therefore halves every bucket's
`batch_size` and sets `gradient_accumulation_steps = 2`, which keeps samples per
optimizer step unchanged (`PLAN_FACTOR=2 ACCUM=2`). It re-runs `off` as well, so
`off@accum2` is an explicit control for whether the micro-batch split moves any
of the measured quantities — every norm in the model is per-token, the loss is a
per-sample weighted mean, and the gradient is a sum over micro-batches divided
by the same total sample count, so the dynamics should be equivalent up to data
order and floating-point summation order, and this control checks that claim.
Arm `cond` finished at the stock micro-batch (its LayerNorm touches only
`[B, 1152]`, so its memory cost is nil), giving a second accum=1 reference.

Two infrastructure notes from the same incident, both now fixed in
`branch_norm_probe.sh`:

- **torchrun wedges in the tbe multiprocess cleanup after the run prints its
  stage endpoint** (hostprobe5, 2026-09-22). Arm `off` sat 8 minutes past
  "Training finished" with rank 0 spinning at 99 % CPU, so the probe never
  started and the arm loop never advanced. The launcher now has an
  `endpoint_watchdog` that kills the ranks 180 s after a finished arm's log goes
  quiet.
- **`torchrun` does not exist in these containers** — torch is installed with
  `pip --target`, which puts no console scripts on PATH — so the first
  submission died with rc=127 for every arm. Use
  `python3 -m torch.distributed.run`.
- **The teardown watchdog must be scoped to its own arm.** An arm that exits
  *cleanly* leaves its watchdog looping; 180 s later that stale subshell sees a
  log containing "Training finished" plus whatever training processes are alive
  by then — the *next* arm's — and kills them. `off-a2` exited cleanly at
  08:24:04 and its watchdog killed `branch-a2` at 08:27:06, 2.5 minutes after
  that arm started. The watchdog now breaks unless
  `$LOGDIR/.current-arm` still names its own run.

## 11. Probe result: arm `cond` — the predicted signature is FALSIFIED

`ascend-bnprobe3-256p` arm `cond` (LayerNorm on the conditioning MLP's input,
stock micro-batch, accum 1) finished 2000 steps at 723 samples/s. Probe JSON
`probe-cond-2000.json`; same probe rows as `off`
(`row_idx 86778, caption_idx 0`, 4 captions, `bucket_hi 86`), so the comparison
is like-for-like.

| | `off` | `cond` | reading |
|---|---:|---:|---|
| `eval/loss` @2000 | 0.92240 | 0.92441 | indistinguishable |
| Muon block weight RMS | 0.5535 | 0.548 | unchanged |
| AdamW module weight RMS | 0.03456 | 0.03475 | unchanged |
| `c_mlp.0.frac_lt_neg5` | 0.553 | **0.066** | ✅ de-saturated as designed |
| `c_mlp.0.ratio` | 0.0246 | 0.0123 | ✗ 2× **worse** |
| `c_mlp.1.ratio` | 0.0093 | 0.0041 | ✗ 2× worse |
| `c_mlp.2.ratio` | 0.0120 | 0.0084 | ✗ worse |
| `modulation.ratio` | 0.0147 | 0.0071 | ✗ 2× worse |
| `txt_pooled_proj` act RMS | 0.920 | **6.854** | ✗ response to the shared direction 7.4× |
| `txt_pooled_proj` centered RMS | 0.266 | 0.284 | unchanged |
| per-block activation RMS | 594 → 5500 | **946 → 3.15e5** | ✗ 57× larger, flat across depth |
| `model_output` RMS | 1.095 | 1.086 | unchanged |

The prediction stated in §9 (saturation → 0 **and** all ratios well above
0.009-0.015) is half right and half wrong: the SiLU saturation was removed
exactly as designed, and every conditioning ratio got **worse**, not better. The
internal activation scale also grew 57× and flattened across depth.

**Mechanism.** The decisive pair is `txt_pooled_proj`'s weights against its
output: the tensor's element-wise weight RMS moves by 4.9 % (0.028433 →
0.029827, bias 0.002991 → 0.005552) between the two arms, while its response to
the *shared* part of the input grows 7.4× (0.92 → 6.85) with the per-sample part
unchanged (0.266 → 0.284). So the conditioning path's gain on the shared
direction is a nearly free direction — AdamW walks along it coherently and a few
percent of element-wise change buys most of a decade of DC gain. LayerNorm on the
input does not constrain that: mean subtraction removes the all-ones direction,
and the shared component of `cat([t_emb, txt_pooled_emb])` is not the all-ones
direction. Worse, by de-saturating `c_mlp.0` it *restores* the conditioning MLP's
ability to amplify whatever shared component reaches it — which is why the
downstream ratios fall.

So the SiLU saturation was a symptom of the DC-dominated input, not the binding
constraint. **Verdict: do not ship `cond_norm`.** Fix (2) as implemented is
falsified; the remaining candidate is fix (1) `branch_norm`, whose a2 re-run is
`ascend-bnprobe6-256p` after the watchdog incident above killed the first one.

## 12. Experiment ledger

Every arm is 2000 optimizer steps of 256p on 16×910B (64 GB), lr peak 0.02 /
3e-4 with a 200-step warmup, `max_steps` 200000 as the LR horizon and
`stop_at_step` 2000, checkpoint interval 1000, grid evaluation off. `arm` names
below are the launcher's; the run directory is `bnprobe-<arm><suffix>`.

| job | arms | accum | bucket plan | log dir | code dir | status |
|---|---|---|---|---|---|---|
| `ascend-bnprobe3-256p` | `off` `branch` `cond` `both` | 1 | stock `ascend-0922` k20 | `$W/logs/bnprobe` | `repo-bnprobe` | terminal, `failures=4` |
| `ascend-bnprobe4-256p` | — | — | — | — | — | no-op (submit-quoting bug), deleted from consideration |
| `ascend-bnprobe5-256p` | `off-a2` `branch-a2` `both-a2` | 2 | halved | `$W/logs/bnprobe-a2` | `repo-bnprobe-a2` | terminal, `failures=1` |
| `ascend-bnprobe6-256p` | `branch-a2b` | 2 | halved | `$W/logs/bnprobe-a2` | `repo-bnprobe-a2b` | terminal, `failures=1` |

`$W = /inspire/sj-ssd3/project/cq-scientific-cooperation-zone/ky26021/artflow`.
The `failures=4` in bnprobe3 is 2 genuine OOMs (`branch`, `both` — sandwich norm
does not fit the stock micro-batch) plus 2 arms whose teardown wedge was killed
by hand.

Both `failures=1` counts are bookkeeping, not science:

- **bnprobe5 / `branch-a2`**: killed 2.5 min after it started, by the *previous*
  arm's watchdog. That job's repo copy (`repo-bnprobe-a2`) predates the
  `.current-arm` guard, so `off-a2`'s finished watchdog saw a quiet log, found
  the newly started `branch-a2` processes, and killed them. This is why the arm
  was rerun as `branch-a2b` out of `repo-bnprobe-a2b`, whose copy carries the
  guard — and why `branch-a2`'s probe JSON must not be used.
- **bnprobe6 / `branch-a2b`**: `rc=137` from the intentional teardown kill
  (the arm itself printed `Training finished`, its `eval-loss@2000` and its
  `throughput-summary`), then `PROBE_OK bnprobe-branch-a2b`.

Probe artifacts (`layer_probe.py --weight-stats --num-rows 4 --t 0.5`, live
weights from `model.safetensors`):

| arm | probe JSON | bytes |
|---|---|---|
| `off` | `$W/logs/bnprobe/probe-off-2000.json` | 46 663 |
| `cond` | `$W/logs/bnprobe/probe-cond-2000.json` | 46 594 |
| `off-a2` | `$W/logs/bnprobe-a2/probe-bnprobe-off-a2-2000.json` | 46 866 |
| `branch-a2b` | `$W/logs/bnprobe-a2/probe-bnprobe-branch-a2b-2000.json` | 46 976 |
| `both-a2` | `$W/logs/bnprobe-a2/probe-bnprobe-both-a2-2000.json` | 46 927 |

Measured, step 2000. `branch-a2b` and `both-a2` are read against `off-a2`
only (rule 1 below); the blank throughput cells for `cond` were never recorded
separately from `off`.

| | `off` | `off-a2` | `cond` | `branch-a2b` | `both-a2` |
|---|---:|---:|---:|---:|---:|
| `eval/loss` | 0.92240 | 0.92154 | 0.92441 | 0.91259 | 0.91149 |
| samples/s (steady) | 722.6 | 715.1 | 723.2 | 653.7 | 659.7 |
| samples per optimizer step | 934.7 | 939.7 | 934.7 | 939.7 | 939.7 |
| train wall, 2000 steps | — | 2634 s | — | 2880 s | 2849 s |
| peak memory, alloc / reserved | 51.0 / 51.4 GiB | **33.4 / 33.8 GiB** | 51.0 / 51.4 GiB | 37.7 / 38.0 GiB | 37.7 / 38.0 GiB |
| Muon block weight RMS | 0.5535 | 0.5558 | 0.5480 | 0.5853 | 0.5851 |
| AdamW module weight RMS | 0.03456 | 0.03457 | 0.03475 | 0.03520 | 0.03575 |
| `txt_pooled_proj` RMS / ratio | 0.920 / 0.289 | 0.907 / 0.291 | 6.854 / 0.0415 | 1.794 / 0.163 | 7.330 / 0.0389 |
| `c_mlp.0` RMS / ratio / < −5 | 7.35 / 0.0246 / 0.553 | 7.32 / 0.0242 / 0.555 | 2.86 / 0.0123 / 0.066 | 8.19 / 0.0289 / 0.666 | 2.48 / 0.0145 / 0.0525 |
| `c_mlp.1` ratio | 0.00932 | 0.00922 | 0.00406 | 0.01562 | 0.00515 |
| `c_mlp.2` ratio | 0.01197 | 0.01428 | 0.00845 | 0.01236 | 0.00582 |
| `modulation` RMS / ratio | 0.776 / 0.0147 | 0.735 / 0.0156 | 0.859 / 0.0071 | 12.74 / 0.00767 | 2.943 / 0.00316 |
| `model_output` RMS | 1.095 | 1.085 | 1.086 | 1.087 | 1.077 |
| per-block activation RMS | 594 → 5500 → 5741 | 637 → 8697 → 12001 | 946 → 3.15e5 → 2.57e5 | 23.3 → 34.0 → 73.6 | 5.51 → 7.45 → 18.3 |

(the per-block row reads block 00 → block 02 → block 24; §13 carries the full
series for the three a2 arms and its interpretation.)

### The accum=2 / half-micro-batch control does NOT reproduce the internal scale

This is the ledger's most important negative result, and it constrains how any
future arm comparison may be read.

`off-a2` matches `off` on everything one would normally call the state of the
run: loss within 0.001, every Muon and AdamW weight RMS within 2 % (0.5558 vs
0.5535, and `blk00/06/12/18/24` = 0.379/0.582/0.601/0.563/0.499 against
0.370/0.582/0.594/0.549/0.500), and the whole conditioning path within 1 %
(`c_mlp.0` ratio 0.02423 vs 0.02456, `c_mlp.1` 0.00922 vs 0.00932, `modulation`
0.0156 vs 0.0147). **And yet the per-block activation RMS differs by up to 5.2×**
(block 2: 8697 vs 1683; block 24: 12001 vs 5741), with a *step* at block 2 instead
of the baseline's smooth ramp.

So the internal activation scale is not a function of the weights in any usable
sense: a 1-2 % difference in weights — here caused only by splitting each
optimizer step's batch into two micro-batches, which by construction consumes the
same samples, has no batch statistics anywhere, and divides by the same total
sample weight — moves it by a factor of a few. That is the same
flat-direction/high-sensitivity character the spike mechanism needs, seen from
the other side.

Consequences, recorded as rules for the remaining readings:

1. **Only accumulation-matched comparisons are admissible.** `branch-a2b` and
   `both-a2` must be read against `off-a2`, not against `off`; `cond` may be read
   against `off` (both accum 1).
2. A single-run value of the per-block scale is not a property of the recipe, so
   `branch_norm`'s signature must be judged on its *shape* (flat versus ramped /
   stepped across depth), not on matching `off`'s numbers.
3. Loss and weight-norm metrics are the robust ones; they moved by <2 % under the
   same perturbation, which is why the cond falsification rests on the ratio and
   shared-component columns rather than on the loss.

### Memory: the halved micro-batch is also the cheapest headroom lever found so far

The same configuration that broke the internal-scale comparison is a clean
memory result, and a useful one for the hero recipe:

| | stock micro-batch, accum 1 | half micro-batch, accum 2 |
|---|---:|---:|
| samples per optimizer step | 934.7 | 939.7 |
| peak allocated / reserved | 51.0 / 51.4 GiB | **33.4 / 33.8 GiB** |
| steady samples/s | 722.6 | 715.1 (−1 %) |
| step time | 1.29 s/it | 1.45 s/it |

17.6 GiB freed for a 1 % throughput cost. `branch_norm` needs about 8 GiB more
than the baseline (it OOMed at 59.3 GiB against 51.4 GiB reserved on a 64 GB
card), so this configuration is what makes fix (1) fit at all. Also recorded:
`cond_norm` costs *nothing* in memory — `cond`'s peak is 51.0 / 51.4 GiB,
identical to `off`, as expected from a LayerNorm on a `[B, 1152]` tensor.

## 13. Probe result: arm `branch-a2b` — the runaway becomes a saturating ramp

**Verdict: `branch_norm` passes the shape criterion.** Recommended for the hero
recipe (which needs the halved bucket plan plus accum 2 at 256p); `cond_norm`
stays out, per §11 and the extra argument in §13.4. Cost is a measured **−8.6 %
step throughput** and **+4.3 GiB** peak memory, both below.

Both jobs are terminal; see §12 for why each reports `failures=1`.

### 13.1 Read the curves, not the endpoints

`eval/loss` is emitted every 500 steps, so all three a2 arms can be compared as
trajectories (same evaluator, same 512-sample pool):

| step | `off-a2` | `branch-a2b` | `both-a2` |
|---:|---:|---:|---:|
| 0 | 1.99821 | 1.99821 | 1.99821 |
| 500 | 0.96855 | 0.96054 | 0.95667 |
| 1000 | 0.94101 | 0.93279 | 0.93007 |
| 1500 | 0.92925 | 0.92057 | 0.91873 |
| 2000 | 0.92154 | 0.91259 | 0.91149 |

`branch-a2b`'s advantage over `off-a2` is present from the first evaluation and
is stable rather than late-blooming: 0.0080 / 0.0082 / 0.0087 / 0.0090 absolute
(0.83 % → 0.98 % relative). For scale, the run-to-run control error is
`off` vs `off-a2` = 0.0009 at step 2000, so the effect is ~10× the control.

At step 2000 the gain is spread over every timestep, not concentrated at one:
`t015` −0.0097, `t040` −0.0096, `t065` −0.0110, `t090` −0.0055.

`both-a2` is a further 0.0011 better at step 2000, but that margin is *shrinking*
(0.0039 → 0.0027 → 0.0018 → 0.0011), i.e. `cond_norm` contributes a small early
advantage that is being absorbed. It is not enough to overturn §11, and §13.4
gives a second, sharper reason to leave it out.

### 13.2 The per-block activation series (the shape criterion)

Activation RMS per block, live weights, step 2000, `t=0.5`, 4 rows:

| block | `off-a2` | `branch-a2b` | `both-a2` |
|---|---:|---:|---:|
| 00 | 636.9 | 23.30 | 5.508 |
| 01 | 1265 | 30.10 | 7.188 |
| 02 | 8697 | 34.03 | 7.448 |
| 03 | 8783 | 38.87 | 7.658 |
| 04 | 9028 | 46.97 | 8.054 |
| 05 | 8954 | 51.92 | 8.384 |
| 06 | 9014 | 55.42 | 8.695 |
| 07 | 9204 | 59.73 | 9.363 |
| 08 | 9535 | 63.48 | 10.05 |
| 09 | 1.017e4 | 66.01 | 10.72 |
| 10 | 1.059e4 | 68.11 | 11.53 |
| 11 | 1.083e4 | 71.03 | 12.27 |
| 12 | 1.098e4 | 72.80 | 13.09 |
| 13 | 1.110e4 | 75.80 | 14.07 |
| 14 | 1.120e4 | 77.23 | 15.08 |
| 15 | 1.131e4 | 83.13 | 15.85 |
| 16 | 1.141e4 | 86.53 | 16.87 |
| 17 | 1.147e4 | 87.99 | 17.58 |
| 18 | 1.148e4 | 89.07 | 17.45 |
| 19 | 1.149e4 | 87.73 | 17.42 |
| 20 | 1.149e4 | 84.75 | 17.36 |
| 21 | 1.145e4 | 75.26 | 17.02 |
| 22 | 1.140e4 | 65.82 | 17.12 |
| 23 | 1.140e4 | 61.61 | 16.95 |
| 24 | 1.201e4 | 73.58 | 18.29 |

Reading, against the criterion set in §12 rule 2 (shape, not values):

- **`off-a2` has a step.** 637 → 1265 → **8697** is a 6.9× jump inside block 02,
  followed by a plateau that drifts up 1.4× over the remaining 22 blocks. Total
  entry-to-peak 18.8×.
- **`branch-a2b` has no step and does not compound.** 23.3 → 34.0 by block 02
  (1.46×), then a monotone rise whose step ratio decays every block
  (1.29, 1.13, 1.14, 1.21, ≤1.08 everywhere after) and goes *below* 1 past
  block 18, peaking at 89.1 and closing at 73.6. Entry-to-peak 3.8×.
- **`both-a2`** has the same shape at an even smaller scale: 5.51 → 18.29, 3.3×.

The observed depth profile has a much smaller ramp and turns over near the end.
This supports the intended intervention. It does not establish a uniform bound
over future weights, inputs or depth, nor prove a random-walk accumulation law.

### 13.3 What `branch_norm` does *not* fix

- **Weight inflation continues.** Muon block weight RMS 0.5558 → 0.5853 (+5.3 %),
  and the sampled blocks are all higher (`blk24` 0.499 → 0.540, `blk06` 0.582 →
  0.607). Sandwich norm decouples weight scale from the residual stream; it does
  not restrain the weights themselves. `muon_wd` (§7/§8) remains the independent
  lever and `branch_norm` is **not** a substitute for it.
- **`c_mlp.0` SiLU saturation gets worse**: 0.555 → 0.666 fraction below −5.
  The combined arm lowers that fraction to 0.0525, but also has smaller hidden
  cross-caption variation/RMS. The branch-only arm improves that latter ratio
  from 0.00922 to 0.01562 despite its higher negative-tail fraction.
- `txt_pooled_proj`'s cross-sample ratio drops (0.291 → 0.163) while its RMS
  doubles, so the product is roughly unchanged (0.264 → 0.293).

### 13.4 Conditioning amplitudes and their limits

Modulation RMS × centered ratio measures absolute cross-caption variation on
this probe. It does not measure usable signal, a Jacobian, or update sensitivity:

| | `off-a2` | `branch-a2b` | `both-a2` |
|---|---:|---:|---:|
| modulation RMS × ratio | 0.7352 × 0.01555 = **0.0114** | 12.74 × 0.00767 = **0.0977** | 2.943 × 0.00316 = **0.0093** |
| vs last-block residual RMS | 0.7352 / 12001 = 6.1e-5 | 12.74 / 73.58 = **0.173** | 2.943 / 18.29 = 0.161 |

`branch_norm` raises the measured cross-caption modulation amplitude 8.5×.
The modulation/residual amplitude quotient also changes substantially, but it
is not a measurement of conditioning influence. Adding `cond_norm` reduces
the amplitudes while improving evaluation loss by 0.0011; these measurements
alone do not establish that its conditioning is functionally weaker.

### 13.5 Cost

| | `off-a2` | `branch-a2b` | `both-a2` |
|---|---:|---:|---:|
| steady samples/s | 715.05 | 653.68 (−8.6 %) | 659.70 (−7.7 %) |
| train wall, 2000 steps | 2634 s | 2880 s | 2849 s |
| peak memory, alloc / reserved | 33.4 / 33.8 GiB | 37.7 / 38.0 GiB | 37.7 / 38.0 GiB |

`cond_norm` is genuinely free (identical 37.7 / 38.0 GiB and within 1 % of
`branch-a2b`'s throughput); the +4.3 GiB and the throughput are the price of
`branch_norm` alone.

**The 8.6 % is flagged, not accepted.** These runs use `--no-compile`, and an
RMSNorm on a branch output is exactly the kind of op that fuses into the
preceding linear's epilogue. So part of the 8.6 % may be launch overhead rather
than work, and it is worth one measurement before the hero launch if throughput
is the binding constraint: it is ~190 of the 2200 GPU-hours, or ~8.6 % of a
480k-step budget. Until that is measured, treat −8.6 % as the honest number.

### 13.6 What this probe cannot say

2000 steps at 256p. The 4090 hero run's first spikes appeared between 8k and 26k
steps, and hero5/6/7's onsets were 8.5k–26k. **This probe cannot show that spikes
are gone.** It shows the amplification path documented in §1–§3 is gone at 2000
steps, which is the strongest evidence obtainable at this horizon. The spike-skip
guard and the tier policy in [hero_spike_policy.md](hero_spike_policy.md) stay in
place regardless, and the decisive test remains a longer run.

### 13.7 Recommendation

1. Add `branch_norm` to the hero recipe. It is the only lever tested that
   directly removes the runaway, it is neutral-to-positive on loss from the
   first evaluation, and its memory cost is already paid by the half-micro-batch
   configuration that §12 found independently useful.
2. Do not add `cond_norm`.
3. Keep `muon_wd` in the recipe: `branch_norm` does not restrain weight growth.
4. Measure `branch_norm` with compilation enabled before launch; if most of the
   8.6 % is launch overhead, the remaining objection to shipping it disappears.
