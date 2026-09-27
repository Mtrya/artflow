# Ascend pretraining

Updated September 27, 2026. The hero trains on 16×910B2C using
[the complete recipe](hero_recipe.md). The current decision is to continue
conditioning-matrix AdamW decay 0.4 and review persistent internal scale growth
alongside loss, gradients and functional response.

## Latest recorded observation

The last recorded patrol is **September 27, 19:15 CST, approximately step
181k**. Full-history curves (0–181k) are plotted under
`notes/assets/hero_patrol_20260927_1910/`; internal readings come from
`stability.jsonl` step 181,000 and `training_metrics.jsonl` step 180,319 on
the pod.

| Signal | Recorded value |
|---|---|
| Throughput | ~805 samples/s median over the last 400 updates, stable |
| Last 400 updates | Median loss 0.8047; median pre-clip gradient 0.196, maximum 0.619; zero clips/skips |
| Evaluation | EMA 0.83535, live approximately 0.8457; both flat within noise over 178.5k–181k (EMA was 0.83561 at 165.5k) |
| EMA relative distance | 0.2633, still declining |
| Conditioning / hidden RMS | 107.2 / 91.8 |
| Maximum residual / gate RMS | 22,134 / 3,341 (both at block 23–24 tail) |
| Block 24 output / maximum branch norm gain | 22,134 / 1.785 |
| Absolute conditioning time spread | 107.2 × 0.0608 = 6.51 |
| Conditioning update RMS | Median 0.268 over >170k, p90 0.34, maximum 0.41 — the envelope keeps rising |
| Caption-borne conditioning fraction | `condition_caption_ratio` **0.0069**, below the 0.008 (≈2 bf16 levels) triage line set September 25; `hidden_caption_ratio` 0.0085 |
| Negative tail fraction | 0.488, oscillating around ~0.5 since ~130k |

Growth per 1k updates, refit on 160k→180k (previous window 145k→165.5k in
parentheses): residual +0.71% (0.90%), gate +0.75% (0.85%), conditioning
+0.83% (0.94%), hidden +1.15% (1.43%). All four log-slope estimates
**decelerated**; growth persists but is no longer accelerating on any tracked
carrier.

The 178k→180k matrix-RMS changes per 1k were +2.08% (`c_mlp.0`), −2.47%
(`c_mlp.2`) and +1.54% (pooled projection) — larger magnitudes than the
162k→164k window (+0.33/−0.14/+0.05), but the sampled-weight decomposition at
181k still nets negative (`norm_sq_change_mean` −4.4e-5), so the decay-0.4
balance holds on average and single 2k windows remain noisy.

Per the September 25 triage rule, `condition_caption_ratio` crossing ~0.008
makes the **text-ablation functional eval the next indicated measurement**.
No recipe change is implied; the question is whether the shrinking
caption-borne signal is still functionally used. Verdict this patrol: **no
intervention**; run the text-ablation eval when convenient (it needs the
pod), keep watching update-RMS envelope and caption ratios.

Patrol cadence is every six hours, using full-history curves, full-resolution
local scalars, internal per-layer metrics and a recent complete checkpoint
pair. Record the observation time and step. Preserve a complete checkpoint
before an intervention that could invalidate comparison or recovery.

## Deployment and recovery identity

- Job: `ascend-hero-256p-wd04-56k-r2`.
- Pinned snapshot: `repo-hero-wd04-56k`, training source revision `136d9c5`;
  all 69 runtime source files were verified before submission.
- Stage directory, relative to `ARTFLOW_ROOT`:
  `runs/ascend-hero-wd04-56k/ascend-hero-256p`.
- Cloud experiment: [ascend-hero-256p / i9r8wnah](https://swanlab.cn/@mtrya/artflow/runs/i9r8wnah).
  Preserve `runtime.json` with `swanlab_run_id: i9r8wnah` on recovery.
- Qualified image, allocation, storage-root value and secret locations are in
  machine-local `INSPIRE.md`. Recover with the same qualified image and 16 ranks.

| Deployment artifact | SHA-256 |
|---|---|
| Source manifest | `3af6057c6cb3401514b7518e0ee1cc0532dd9446dd29b61410c1cfd59f32355a` |
| Platform wrapper | `019526d06e8caa4df65d39e7d7668051725ea6e320f65a30f3fa0228ed7142d8` |
| 256p bucket plan | `e66622da5e6b53240098fa97782f3a0529b929fecc3c188309a17e89ea6f3536` |
| 56k migration record | `0f14cbdf4e338b4db38da867027c38c19c1632582c467f33ef771c8cd90c26b6` |

Use the checkpoint's recorded complete recipe and verify the pinned files
against the manifest. The working recipe has September 27 later-stage data
and batching amendments; compare it with deployed/checkpoint state before applying them.
Strict resume checks the entire recipe, including future stages.

Determine whether the prior attempt is running, preempted, dead or blocked in
I/O. Stop a still-live failed attempt before creating a replacement; the
launcher also enforces a single-writer lock. In the replacement allocation:

```bash
cd "$ARTFLOW_ROOT/repo-hero-wd04-56k"
bash scripts/ascend/pretrain.sh \
  --config configs/hero.toml --stage 256p --nproc_per_node 16 \
  --verify_resume_state
```

This discovers the newest complete same-stage checkpoint and verifies restored
state. The flag makes a missing checkpoint fail rather than begin a fresh run.
Use `--resume /absolute/path/to/checkpoint_step_XXXXXX` for a reviewed earlier
complete checkpoint. Weights-only loading loses optimizer/scheduler/EMA,
sampler and per-rank RNG state.

The original 56k state is an independent protected copy at
`keep/ascend-hero-256p/checkpoint_step_056000`. Its explicit migration splits
the AdamW conditioning group while preserving model/EMA/Muon/sampler/RNG,
parameter moments and counters. Scheduler group lists and recipe bookkeeping
are adjusted by [`migrate_conditioning_decay.py`](../scripts/pretrain/migrate_conditioning_decay.py).
`resume_migration_056000.json` and `transition_056000.json` preserve provenance
outside rotating checkpoint retention. Ordinary recovery from this phase
uses the already-migrated state. A protected 50k checkpoint also belongs to
the earlier recipe and needs its historical source or an explicit migration.

The production continuation restarted at 56k after a separate 500-update
qualification; all 16 ranks restored model, optimizers, schedulers, EMA and
RNG exactly. The original cloud series already contained loss through 57,094
and stability through 57,000. Thus cloud points in that overlap still represent
the preceding recipe. Use the continuation's local JSONL for 56,001–57,094
and preserve the actual global step when comparing phases.

## Prepared 450k resolution transition

The September 27 working recipe selects accumulation **3/4** for 640p/896p
with micro-batches **5–12 / 3–6** on 16×910B2C. The source mixtures are the
user's settled values. Throughput, tail coverage and qualification limits are
in [the infrastructure record](infra_pretrain.md). The current 256p job keeps
its pinned recipe; do not restart it to install future-stage settings.

`repo-pretrain-later-0927` is the prepared snapshot under `ARTFLOW_ROOT`.
Its complete runtime config and all three bucket files have their own manifest
in `INSPIRE.md`. A read-only amendment preflight passed against the real
184k checkpoint: exactly six future-stage fields differ (datasets, bucket
plan and accumulation for each later stage), plus verified operational file
relocations. This did not modify or resume the hero checkpoint.

The separate native qualification completed both stage transitions and an
896p replay. All 16 ranks restored model/optimizers/schedulers/EMA/RNG exactly;
the replay matched 128 rank/update sample-identity hashes and all endpoint
RNG/sampler states. Both higher-resolution monitoring paths and 48-image panels
completed. Brief early clipping peaked at 1.28/1.87 with no skipped updates;
this supports infrastructure readiness, not a guarantee about the future
hero transfer. Detailed conditions are in the infrastructure record.

At the completed 450k endpoint, preserve the original checkpoint in the
finished 256p stage directory. From the prepared snapshot, create an independent
amended copy, then explicitly select it for the first 640p launch:

```bash
cd "$ARTFLOW_ROOT/repo-pretrain-later-0927"
python -m scripts.pretrain.migrate_stage_recipe \
  --source "$ARTFLOW_ROOT/runs/ascend-hero-wd04-56k/ascend-hero-256p/checkpoint_step_450000" \
  --destination "$ARTFLOW_ROOT/keep/hero-recipe-0927/checkpoint_step_450000" \
  --config configs/hero.toml \
  --reason "Install measured later-resolution buckets, accumulation and settled data mixtures"
bash scripts/ascend/pretrain.sh \
  --config configs/hero.toml --stage 640p --nproc_per_node 16 \
  --resume "$ARTFLOW_ROOT/keep/hero-recipe-0927/checkpoint_step_450000" \
  --verify_resume_state
```

The migration refuses an existing destination and records hashes plus both
recipes. It preserves model, both optimizers/schedulers, EMA, sampler and
per-rank RNG files byte for byte. Stage transitions reset the sampler for the
new data/shape stream while retaining global optimizer/scheduler/EMA progress.
Verify all 16 restore reports and the 600k scheduler horizon at launch.
Review the initial and 452k 640p grids and live/EMA loss, gradients and stability
telemetry; short infrastructure checks cannot prove the actual 450k transfer
will remain stable.

Later 640p recovery can use ordinary same-stage discovery. Preserve its
complete 570k endpoint; 896p can use ordinary predecessor discovery with the
same prepared config, because both later-stage recipes are already recorded.
The launch commands belong in separately scheduled allocations after each
predecessor has completed; this preparation has not launched a later hero stage.

## Runtime records

| Record in the stage directory | Use |
|---|---|
| `training.log` | Actual stdout/stderr and first traceback; normalize carriage returns when reading progress-bar output. |
| `training_metrics.jsonl` | Per-update loss, gradient, throughput and durable evaluation scalars; caption/source summaries every 25 updates. |
| `stability.jsonl` | Detailed response, per-layer and sampled-weight records every 100 updates. |
| `stability_inputs.pt` | Fixed inputs and panel identity for comparable probes after recovery. |
| `runtime.json` | Cloud identity and last saved progress; recovery requires a complete checkpoint. |
| `samples/` | Bilingual image grids and generation manifests. |
| `elastic/` | Worker error files and launcher diagnostics. |
| `checkpoint_step_*/training_state.json` | Completion record after all-rank saves; require a complete checkpoint. |

Read the first traceback and real training log when diagnosing a crash.
Platform output can lag; HCCL teardown messages may describe consequences.
For a suspected platform failure, reproduce the same code on another node/card
before changing the model. Check startup/evaluation/checkpoint timing against
[measured workload costs](infra_pretrain.md).

## Stability evidence behind the current decision

The following results explain the selected decay and ongoing watch duties.
Each diagnostic applies to its measured checkpoints, panels and update window.

| Experiment | Finding and implication |
|---|---|
| September 24 H200 spike reconstruction | Captured step 12,829 reproduced in FP32 on H200 and 4090 (gradient 41.709637 / 41.709521), establishing a model-side mechanism for that event. An exact reconstruction at step 12,172 isolated the last conditioning matrix update acting on a large shared hidden feature: its update alone gave gradient 55.8181; removing that shared displacement reduced it to 0.247787. This forensic intervention uses the implicated batch and is not a deployable guard. |
| Branch normalization, matched 2k | Loss improved 0.92154 → 0.91259 while block-24 RMS fell 12,001 → 73.58. Weight growth and negative SiLU tails still increased, so those statistics alone did not diagnose damage. |
| Short Ascend stability check, September 24 | With the preceding RMS-matched Muon recipe, all 2k updates applied, no post-warmup clipping, final-500 maximum gradient 0.40657. EMA/live loss 0.85689331 / 0.86698182. Current original-scaling Muon has separate infrastructure qualification. |
| 44k→46k conditioning attribution | New matrices with old biases reproduced 99.87% of conditioning growth; bias-only changes reproduced 0.12%. Matrix RMS alone missed changes in directional response. |
| 48k functional panel | 160 caption cases / 158 images / five times; 99 held-out cases plus 61 training long-caption supplements. The reconstructed next update improved held-out loss 0.0513%. FP32 conditioning changes had confidence intervals crossing zero. Swapping pooled/token/both text paths raised loss 1.03% / 10.42% / 12.14%. Both paths were useful on this panel. |
| 56k→56.5k native decay-0.4 qualification | Conditioning/gate/residual RMS fell 1.39% / 1.30% / 2.31%; maximum gradient 0.393383, no clips. EMA/live loss 0.84713080 / 0.85551629 versus control 0.84714676 / 0.85555276. Some pre-normalization layers still grew. This supports the selected continuation, with longer evidence supplied by the hero. |
| 84k functional check | Removing time variation raised held-out loss 40.65%; boosting it or restoring the 66k component also hurt. A fixed gate RMS cost 0.664% loss; a 1.05× reference RMS ceiling cost 0.082% on insertion and touched 96.16% of forwards. |
| 92k→93k paired ceiling trial | 16 ranks, matched 16k sample identities and endpoint RNG. Residual RMS fell 11.77% relative to control; train loss changed +0.0036%, held-out live +0.0626% (inconclusive), EMA +0.0438%, long-caption supplement +0.4922%. Several raw-gate/pre-branch paths still grew. A gradient outlier 2.591105 was not reproduced on a drifted replay, leaving its cause unresolved. The ceiling remains unadopted. |

The decay choice followed four matched 48k→49k continuations. Fitted late
slopes, percent per 1k updates:

| Arm | Conditioning RMS | Gate RMS | Residual RMS | Pre-branch maximum |
|---|---:|---:|---:|---:|
| Control | +6.73 | +7.45 | +7.04 | +12.20 |
| Conditioning decay 0.2 | +3.20 | +4.42 | +4.68 | +7.20 |
| Conditioning-output normalization | 0 | −3.56 | −4.40 | −6.73 |
| Conditioning decay 0.5 | −2.55 | −0.49 | −0.35 | +1.17 |

Decay 0.5 controlled measured late gate/residual slopes with a smaller
transition than output normalization; its broad-panel loss improvement was
inconclusive. The user selected the intermediate 0.4 dose and the 56k input.

A CPU conditioning probe at 112k, using the same inputs as the protected 66k
checkpoint, helps explain subsequent scale growth:

| Quantity | 66k | 112k |
|---|---:|---:|
| Pooled projection RMS | 6.44 | 7.53 |
| Text half of first preactivation RMS | 20.86 | 39.34 |
| Time half of first preactivation RMS | 3.62 | 4.09 |
| Hidden RMS | 19.84 | 39.82 |
| Effective final-matrix gain `RMS(W2 h)/RMS(h)` | 2.35 | 1.56 |
| Conditioning RMS | 46.68 | 62.14 |
| Absolute time-varying RMS | 5.52 | 4.82 |

Crossing old hidden/new W2 gave conditioning RMS 33.66; new hidden/old W2 gave
85.02. Zeroing biases changed conditioning by 0.605% at 112k (0.393% at 84k).
This panel attributes most measured growth to upstream activations and matrix
directions. Repeat the attribution if the behavior changes; it does not settle
all later checkpoints or caption distributions.

## Reading the monitoring signals

Start with rolling per-update loss/gradient distributions, the last few
stability probes and nearby live/EMA evaluations. Record LR, caption lengths,
source mix and bucket workload. Downsampled cloud points are unsuitable for
estimating gradient medians or rare-event frequencies.

The fixed panel holds at most four rank-zero rows at three shifted times.
`t010/t050/t090` refer to base times 0.1/0.5/0.9 before resolution shift.
Compare matching `panel_id`, row counts and text lengths. Its short captions
cover only a small part of the curriculum. Activations/features/gradients are
measured before the sampled update; weight deltas and response tests describe
that update. Per-update records catch events between 100-step probes.

`train/grad_norm` is pre-clip; values above 1 mean clipping. Muon's momentum
orthogonalization means raw-gradient clipping does not bound parameter-update
norm. `update_applied` should be 1 and `consecutive_skips` 0. Nonfinite loss or
gradients stop all ranks before update. Finite-spike skipping and automatic
rollback are absent, so finite deterioration requires operator review.

The response test holds the **updated trunk** fixed while comparing old,
new and twice-displaced conditioning:

| Signal | Interpretation |
|---|---|
| `loss_after - loss_before` | Net loss effect of the whole sampled update. |
| `conditioning_loss_delta` | `loss_after - loss_held_condition`; positive means the conditioning move hurts loss with the updated trunk. |
| `conditioning_prediction_change_rms` | Absolute prediction effect of that conditioning displacement; combine with actual loss changes. |
| `conditioning_gain` | Prediction-change RMS / conditioning-update RMS, a directional sensitivity for this update. Tiny displacements and BF16 rounding can make it noisy. |
| `second_difference` | `L(c+2Δc) - 2L(c+Δc) + L(c)`, measured on the updated trunk; depends on step size and numerical precision. |
| `c2_shared_shift_rms` / `c2_bias_shift_rms` | Last conditioning matrix update applied to the old mean hidden feature / actual bias displacement. |

Prediction-change RMS terms are not additive attributions. These sensitivities
have no calibrated universal failure threshold. If conditioning motion is
harmful, inspect its AdamW-routed matrices and trunk interaction before
choosing an optimizer intervention.

Branch ratios are RMS(actual gated addition) / RMS(incoming residual).
Read per-block inputs, outputs and additions alongside maxima; the maximum's
layer can change. Branch RMSNorm precedes the learned gate, so learned gains,
gates and accumulated additions can amplify the stream.
`branch_norm_gain_max` is the largest absolute affine element.

The SiLU negative-tail fraction counts panel preactivations below −5.
Caption/time ratios divide centered variation by feature RMS; multiply the
ratio by RMS to recover absolute spread. BF16-floor watch labels are heuristics
rather than measured failure bounds. Functional text/time ablations establish
whether variation remains useful.

For a sampled tensor, write `W_new = a W_old + U`, with
`a = 1 - actual_LR × weight_decay`. The squared-norm change decomposes into:

```text
radial = 2a <W_old, U> / ||W_old||²
energy = ||U||² / ||W_old||²
decay  = a² - 1
norm_sq_change ≈ radial + energy + decay
```

The implementation clamps near-zero denominators. Whole-optimizer update
ratios and averages over selected tensors have different coverage. Muon decay
uses its base LR; the update additionally uses the chunk's shape multiplier.
`attn_qk_gain_*` summarizes signed learned Q/K norm weights; attention logits
or entropy require a direct probe. EMA distance is `||EMA-live||/||live||`;
interpret it together with both losses and the EMA schedule.

## Review and intervention

Continue while learning and functional response remain healthy. Investigate
persistent scale growth even when loss improves: inspect individual layers,
absolute variation, checkpoint pairs and a short test that distinguishes
plausible mechanisms. A repeated harmful response with a sustained loss or
gradient shift strengthens the case for intervention.

Stop and preserve evidence for nonfinites or sustained, rapidly worsening
finite loss with deteriorating gradients/response. Save the first traceback,
pre/post-onset logs, panel files, source/config and complete checkpoints before
retention removes a healthy predecessor. Diagnose before restarting, and use
a checkpoint shown healthy before onset. Change one mechanism at a time with
explicit provenance and tested state migration.

Metric definitions are implemented in
[`stability.py`](../src/pretrain/stability.py),
[`health.py`](../src/pretrain/health.py) and
[`train.py`](../src/pretrain/train.py).
