# Ascend pretraining hero maintenance handoff — September 25, 2026

This is the fresh post-infra hero authorized by the user on September 25.
Use [the complete recipe](hero_recipe.md) and [infra evidence](infra_pass.md).
No H200/4090 checkpoint is involved.
The hero is the longer stability validation; it is not evidence that stability
at peak learning rate has already been established.

## Launch identity

- Platform job: `ascend-hero-256p-0925-r3`, workspace `昇腾卡公共空间`, project
  `SCION-自动化科研`, compute group `910B资源`.
- One instance: 16×910B2C, 128 CPU cores, 1024 GiB RAM, 64 GiB shared memory;
  qualified image `cann:9.0.1-910b-ubuntu22.04-py3.12`. Request priority 10 in
  this workspace; verify the assigned level is HIGH. Priority 4 mapped to
  NORMAL in the first attempt. Automatic fault retries are disabled.
- Runtime source is an immutable snapshot, `repo-hero-0925`, under the Ascend
  ArtFlow storage root. The exact root is in gitignored `INSPIRE.md`; below,
  `ARTFLOW_ROOT` means that root. Training source revision: `8b4257d`.
- Complete deployed config: `repo-hero-0925/configs/hero.toml`. Its only
  differences from the checked-in recipe are storage/output/256p-plan paths.
  It does not inherit or merge any other config.
- Config SHA-256: `cfb8a8f2ed2492ddbbf74213cc452f52d7d49e54e475beef4054cc395478bbd0`.
- Source-manifest SHA-256: `893090154d136d7517f67a929a0fff7d7f3d629f410d593c6bd7ef72074a8ed3`.
  The manifest covers the platform wrapper as deployed in addition to the
  committed training source. Wrapper SHA-256:
  `019526d06e8caa4df65d39e7d7668051725ea6e320f65a30f3fa0228ed7142d8`.
  Validate files against the manifest before a restart.
- 256p plan SHA-256: `e66622da5e6b53240098fa97782f3a0529b929fecc3c188309a17e89ea6f3536`.
- Output root: `runs/ascend-hero-0925`; current stage directory:
  `runs/ascend-hero-0925/ascend-hero-256p`. This separate directory prevents
  accidental resume of the older Ascend hero with the same experiment name.
- SwanLab: [ascend-hero-256p](https://swanlab.cn/@mtrya/artflow/runs/i9r8wnah),
  run ID `i9r8wnah` (account `mtrya`, project `artflow`). Ascend uses direct cloud
  access; no relay is required or started. The user-supplied login is installed
  at the documented
  platform secret path with mode 600. Verified cloud identity and training
  progress are recorded below.

The first job, `ascend-hero-256p-0925`, stopped in its deployment preflight
because a nested Python newline was quoted incorrectly. It ran **zero training
updates** and was released. The corrected bootstrap passes Python and shell
syntax checks. Attempt r2 then passed the real NPU all-empty-caption
forward/backward check and started offline while authentication was being
supplied. It was stopped at global step zero, before its first checkpoint;
its stage directory is
preserved under `runs/ascend-hero-0925/attempts/offline-start-0925`. The user then
provided the standard local SwanLab login. Attempt r3 starts fresh online with
that login in the designated platform secret file. These startup retries do
not resume a model from the old hero or from the infra probes.

## Frozen run contract

The schedule is 600k total updates with 20k warmup. This job executes 256p to
450k, then exits; it does not silently launch later resolutions. The later
endpoints are 570k and 600k. The 256p bucket plan uses micro-batches 8–72 and
accumulation one on 16 ranks. Preserve the world size on recovery.

Architecture is `artflow-v2`, h1152, 16 heads, 1 double-stream + 24
single-stream blocks, FFN width 3072, 532,496,992 parameters. Branch RMSNorm,
x1000 timestep features, no conditioning-input normalization, native NPU
RMSNorm/SwiGLU and fixed execution policies are built into code.

Muon uses original scaling, LR 0.02, decay 0.0015, momentum 0.95. Auxiliary
AdamW uses LR 1e-4, decay 0.01 and betas 0.9/0.95; clip norm is 1.0.
Bias-corrected EMA uses decay 0.9999. Dataset mixtures, caption progression,
dropout 0.1 and every numerical setting are explicit in the complete config.
Do not add environment hyperparameters or recipe CLI overrides.

Pre-launch corrections: independent caption dropout now permits all-dropped
micro-batches; final-layer Qwen exits include final normalization. Hero k20/28
features are unchanged. Euler/Heun monitoring already uses FP32 time/state.
Standalone generation and the unused decode helper remain outside this run's
execution paths. Local suite: **806 passed, six NPU skips**, 43.65 seconds.

## Observe the actual run

```bash
inspire job status ascend-hero-256p-0925-r3 --workspace 昇腾卡公共空间
inspire job logs ascend-hero-256p-0925-r3 --workspace 昇腾卡公共空间 --tail 80
inspire job events ascend-hero-256p-0925-r3 --workspace 昇腾卡公共空间 --tail 15
inspire job shell ascend-hero-256p-0925-r3 --workspace 昇腾卡公共空间
```

Inside the job, inspect the current stage directory:

- `training.log`: real stdout/stderr and the first traceback. Progress bars
  use carriage returns; read with `.replace('\r', '\n').splitlines()` when
  printing a short tail. Platform logs alone can lag or omit buffered output.
- `training_metrics.jsonl`: durable training and evaluation scalars, independent
  of cloud availability. Loss, gradient and throughput records every update; caption/source summaries
  every 25, stability every 100, model/optimizer health every 250.
- `stability.jsonl` and `stability_inputs.pt`: detailed fixed-panel response and
  sampled-weight records. Keep the panel on resume so trends remain comparable.
- `runtime.json`: checkpoint-time global step and SwanLab run ID; it first
  appears at the first save, not the first training update. It is **not a
  recoverable checkpoint** by itself; use live metrics for current progress.
- `samples/`: periodic 48-prompt grids and their generation manifests.
- `elastic/`: launcher/worker diagnostics; consult any error files alongside
  the first traceback. Do not infer a cause from HCCL teardown messages.
- `checkpoint_step_*/training_state.json`: completion record written after
  all-rank state saves. Do not resume from a directory merely because it exists.

The selected infra workload measured ~879 samples/s and ~49.7 GiB allocated.
Startup includes model/data loading, initial evaluation and spawning 128
loader workers; its first update can take ~90 seconds. Steady measurements
exclude startup and should be compared at similar caption lengths/aspects.
Evaluation every 500 updates costs roughly 53 seconds for EMA/live combined;
grids every 2500 take roughly 31 seconds. Checkpoints every 2000 take ~4 seconds.
These are measured references, not hang timeouts or later-resolution forecasts.

## How to interpret and react on stability metrics

These are diagnosis tools, not a validated predictor that a loss spike must
happen. Training and held-out loss establish whether learning is deteriorating;
the fixed-panel interventions help locate the mechanism. Weight growth, SiLU
tail fraction and response gain alone do not establish damage. In the previous
branch-normalization comparison, loss and residual amplification improved even
though weight RMS and the SiLU negative-tail fraction increased; see the
[growth evidence](archive/muon_weight_growth_0924.md) and
[stability review](archive/ascend_stability_review_0924.md).

### Compare the right observations

- Start with rolling 100-update training loss/gradient summaries, then the last
  five stability observations and nearby EMA/live evaluations. Compare median,
  upper tail and persistence, not just the latest point. These windows are
  practical triage aids, not calibrated alarm thresholds.
- Record LR, caption lengths, source mixture and bucket workload alongside the
  interval. Warmup changes update sizes; caption progression changes difficulty.
  Establish a baseline from the healthy early run and revise it as warmup
  advances toward 20k. Do not transplant absolute cutoffs from old runs with
  different LR conventions, architecture, panels or accumulation.
- The stability panel is at most four retained rank-zero examples, evaluated
  at three resolution-shifted times. `t010`, `t050`, `t090` name the **unshifted**
  time inputs, not three bins of all training examples. Compare the same
  `panel_id`; check `panel/rows_per_timestep` and `panel/text_tokens`. This hero's
  initial panel has short text, so a healthy panel cannot clear long captions.
- Block activations, conditioning features and sampled gradients are measured
  **before** the sampled optimizer update; weight deltas and response tests
  describe that update. Stability is sampled every 100 updates, so a transient
  event between probes can be missed. Use per-update training logs too.
- SwanLab publishes summaries. Read `stability.jsonl` for the implicated block
  or tensor; a maximum can switch between layers. Keys below omit the
  `stability/` prefix unless they start with `train/`, `eval/` or `health/`.

### Loss, gradients and actual updates

`train/loss` is the global weighted training objective. `train/grad_norm` is
the **pre-clip** norm; values above the configured 1.0 mean clipping occurred,
not that an update was skipped. Compute the fraction above 1.0 in a window.
Some early clipping is ordinary; a sudden persistent rise after a quiet period,
together with worsening loss, warrants investigation. Muon orthogonalizes a
momentum-derived direction: clipping the raw gradient does **not** cap its
parameter update norm at 1.0. Check actual update ratios as well.

`eval/loss` uses EMA and `eval_live/loss` uses live weights. Inspect their
timestep and caption-band losses, including `samples` and `shortfall` counts.
A band with three examples is weak evidence; an empty band provides none.
One harder training batch is different from deterioration on fixed inputs.
Live loss worsening while EMA stays healthy can expose recent damage that
smoothing conceals; a persistent rise in both is stronger evidence of damage.
Do not wait for the next 500-update evaluation if current training is already
clearly diverging.

`train/update_applied` should remain 1 and `train/consecutive_skips` 0. There is
**no finite-spike skipping or automatic rollback**. A nonfinite loss/gradient
causes all ranks to stop before the optimizer update. This guard does not prove
that a large finite update is safe, nor does it check every internal activation.

### Conditioning displacement and its effect

`conditioning/update_rms` measures the actual change in conditioning output
on the fixed panel. `shared_update_rms` is the part shared across panel rows
and times; `centered_update_rms` is the remainder. A growing displacement is
concerning when its **effect** grows harmfully, not merely because LR is rising.

The response test runs the updated trunk with either the old conditioning
(`loss_held_condition`), the new conditioning (`loss_after`), or twice the
actual conditioning displacement (`loss_twice_condition_update`). For each
time, read:

| Metric | Meaning and response |
|---|---|
| `response/t*/loss_after - loss_before` | Net loss change from the entire sampled update. Repeated increases on the same panel, corroborated by training/evaluation loss, merit investigation. One unfavorable SGD update is not a failure. |
| `response/t*/conditioning_loss_delta` | `loss_after - loss_held_condition`. Positive means the conditioning move hurts MSE **with the updated trunk held fixed**. If holding conditioning restores the loss while the real update worsens it, prioritize the conditioning path. If the held loss also deteriorates, inspect trunk updates and their interaction with conditioning. |
| `response/conditioning_prediction_change_rms` | Absolute prediction change caused by replacing old conditioning with new conditioning in the updated trunk. Compare with `prediction_change_rms` from the full update and with loss changes. These RMS values do not form an additive attribution. |
| `response/conditioning_gain` | Conditioning-induced prediction change divided by `conditioning/update_rms`. It is sensitivity along this particular update, not a Jacobian bound. There is no universal safe value below 1; tiny displacements and BF16 rounding can make the ratio noisy. Act on large absolute effects and harmful loss changes, not this ratio alone. |
| `response/t*/second_difference` | `L(c_old + 2Δc) - 2L(c_old + Δc) + L(c_old)`, all on the updated trunk. Positive values indicate curvature along this displacement. A growing positive value with a harmful actual step supports an overshoot hypothesis; it does not prove the next update will spike. Its scale also grows with displacement size. |

If harmful conditioning motion persists, inspect `c2_shared_shift_rms`
(`ΔW` of the last conditioning linear layer applied to the **old mean hidden
feature**) and `c2_bias_shift_rms` (actual bias change). This distinguishes a
matrix update acting like a shared shift from a bias update. Neither alone
accounts for every term in the total conditioning change.

The conditioning MLP and pooled-text projection are routed to **auxiliary
AdamW**, not Muon. A large conditioning effect is not sufficient evidence to
reduce Muon LR or increase Muon decay. Use the measured source to choose one
short replay/probe, with an explicit decision it will resolve; preserve the
original recipe and evidence before changing anything.

### Branch amplification and conditioning features

`summary/residual_rms_max`, `attention_ratio_max`, `mlp_ratio_max` and
`gate_rms_max` summarize the image residual stream and gated branch additions.
The branch ratios are RMS(actual residual addition) / RMS(incoming residual).
Read the corresponding `blocks/NN/` values to find where amplification starts.
Branch RMSNorm controls the branch before gating; learned gains, gates and
repeated residual addition can still amplify the signal. A rising gate or
residual norm becomes actionable when branch contributions, response effects
and loss/gradients worsen together. A large ratio can also come from a small
denominator; inspect input/output RMS before diagnosing an exploding branch.

`summary/branch_norm_gain_max` is the maximum **absolute learned affine
element** across branch RMSNorms, not the RMS of their outputs. Use it to
distinguish growing normalization gains from growing gates; there is no fixed
cutoff at 1.

`conditioning/negative_tail_fraction` is the fraction of first-linear
preactivations below −5 on this panel, a proxy for the SiLU negative tail.
It is not a count of permanently dead neurons across the dataset. The
`hidden_*` features are SiLU outputs; `condition_*` features are the final
conditioning outputs. Their `caption_ratio` and `time_ratio` measure centered
variation divided by feature RMS. A falling ratio can mean larger shared
features rather than disappearance of absolute variation. Inspect RMS too
(ratio × RMS recovers the centered RMS). Escalate sustained loss of variation
with poor conditional/timestep behavior, not a tail fraction alone. Do not
reintroduce conditioning-input normalization solely to improve this statistic.

### Weight growth, attention gains and EMA

`health/update_weight_ratio_muon` and `_aux` are whole-optimizer
`||W_new - W_old|| / ||W_old||` ratios. `sampled_weights/<role>/*_mean` instead
averages diagnostics from selected tensors; it is not a whole-model estimate.
The selected conditioning matrices and beginning/middle/end block projections
can miss an unsampled failure. Inspect individual `weights/<name>/` records.
Tiny initial weights can produce very large relative updates without a large
absolute displacement.

For one sampled tensor, write `W_new = a W_old + U`, where
`a = 1 - actual_LR × weight_decay`. The reported fractional squared-norm change
decomposes as:

```text
radial = 2a <W_old, U> / ||W_old||²
energy = ||U||² / ||W_old||²
decay  = a² - 1
norm_sq_change ≈ radial + energy + decay
```

The implementation clamps the denominator for near-zero weights. `energy` is
nonnegative; `radial` can grow or shrink weights. This measures where growth
comes from without assuming a random walk. Positive growth with improving loss
and controlled responses is a watch item, not a reason to stop. Persistent
growth accompanied by increasing functional amplification justifies a targeted
optimizer investigation: distinguish outward radial motion from step energy
and examine the responsible tensors. Do not infer that a larger decay is the
right fix from weight RMS alone. Current Muon peak shrinkage is
`0.02 × 0.0015 = 0.00003` per update; warmup reduces it. Decay uses the base LR,
while Muon's update also uses each chunk's shape multiplier.

`health/attn_qk_gain_max` and `_mean` summarize learned Q/K normalization gains
(signed values, not absolute maxima). They are **not measured attention logits
or entropy**. If attention branch growth accompanies worsening response/loss,
inspect the responsible layer and use a focused logits/entropy probe if that
would distinguish causes; gain drift alone does not prove attention collapse.

`health/ema_rel_distance` is `||EMA - live|| / ||live||`. Warmup and the
bias-corrected EMA schedule naturally change it. A sudden separation plus
worsening live loss deserves investigation; distance alone is not failure.
Healthy EMA loss does not authorize continuing a collapsing live model.

### What the maintainer should do

1. **Continue and watch** when loss is improving or locally noisy, gradients
   remain in their recent range, and no repeated harmful fixed-panel response
   appears. Record unusual norm/tail/gain trends with their context. Do not
   stop for one high batch loss, one clipped update, or one positive panel delta.
2. **Investigate while it still learns** when a change persists and has
   corroboration: for example, three successive stability probes show worsening
   response effects alongside a sustained loss/gradient shift. Three probes is
   a practical review trigger, not a validated predictor or a requirement to
   wait during rapid deterioration. Save the interval, inspect per-time and
   per-layer details, check workload changes, and identify a test that separates
   plausible mechanisms. Prefer a short replay of the implicated checkpoint
   over a new long validation or an indiscriminate sweep.
3. **Stop and preserve evidence** for nonfinites or sustained, rapidly growing
   finite loss with gradient/functional-response deterioration. Do not wait for
   NaNs, a scheduled grid, or an arbitrary number of probes. Capture the first
   traceback, logs before and after onset, panel files, exact config/source,
   and complete checkpoints before retention removes a potentially healthy
   predecessor. A newly saved checkpoint can already contain damaged state.
4. **Diagnose before restarting.** Separate a numerical failure from OOM,
   storage/HCCL or node failure using the actual traceback. For a suspected
   platform issue, reproduce the same code on another node/card before changing
   model code. For numerical damage, choose the latest checkpoint shown healthy
   *before onset*, not automatically the newest directory. A restart from the
   same damaged state is not a fix. Change one mechanism per diagnostic
   iteration and record what it showed. Use the strict recovery procedure below;
   any deliberate recipe change needs explicit provenance and a tested state
   transition, not a bypass of config validation.

These are manual maintenance rules; no new automatic thresholds, job stopper,
or recipe switches are installed. Definitions are grounded in
[`stability.py`](../src/pretrain/stability.py),
[`health.py`](../src/pretrain/health.py), and the update order in
[`train.py`](../src/pretrain/train.py).

## Restart and recovery

First determine whether the old job is running, dead, preempted or blocked in
I/O. Stop a still-live failed attempt before creating a replacement. The launcher
also enforces a single-writer lock. Never run two writers in the same stage dir.

In a replacement allocation with the **same image and qualified resources**,
set the storage-root environment as documented in `INSPIRE.md`, then execute:

```bash
cd "$ARTFLOW_ROOT/repo-hero-0925"
bash scripts/ascend/pretrain.sh \
  --config configs/hero.toml --stage 256p --nproc_per_node 16 \
  --verify_resume_state
```

The launcher finds the newest complete same-stage checkpoint. The verification
flag makes a missing checkpoint fail instead of silently starting a new run.
An explicit `--resume /absolute/path/to/checkpoint_step_XXXXXX` can select a
reviewed earlier checkpoint. Preserve the full recorded config and bucket plan;
weights-only recovery loses optimizer, scheduler, EMA, RNG and sampler state.
The trainer reuses the stored SwanLab ID when resuming a checkpoint.

Keep the latest two complete checkpoints; deletion happens only after a new
complete save. The first recovery checkpoint is at step 2000. Before it exists,
a failure has no resumable training state: retain the failed attempt's logs and
resolve its cause before explicitly starting fresh. Do not lower the checkpoint
interval or silently overwrite the output to make a failed launch look healthy.

The deployment bootstrap is for the initial fresh launch. The r3 bootstrap
also performs the one-time archival of the stopped step-zero offline attempt. **Do not replay that bootstrap for recovery**; use the pinned
platform wrapper above. It loads prepared dependencies, fixes platform cache
locations, uses the designated platform SwanLab secret if present, and delegates
to the strict launcher. It downloads no mutable code or packages.

## Upcoming checks and stage boundary

The first periodic EMA/live evaluation (500) passed. Next check the first
complete recovery checkpoint (2000), first periodic grid (2500), and warmup
endpoint (20000).
The preceding 600-update infra checks establish early execution health only.
The hero supplies later stability evidence; do not add a separate long-validation
run or exhaustively sweep tuning candidates alongside it.

640p data preparation belongs to the other agent. Neither its download progress
nor 896p staging is a reason to modify this 256p run. The later-stage bucket
plans do not yet exist; accumulations 4/5 are unqualified candidates. Install
and qualify each later plan and its transition before that stage launches,
preserving global LR/caption progress and the predecessor's complete endpoint.
Do not switch to old CUDA plans or automatically enter an unqualified stage.
If qualification requires changing accumulation or another recorded recipe
value, the current strict resume check will reject that changed config. Resolve
that as an explicit, tested stage-transition migration; do not edit checkpoint
metadata or bypass validation to conceal the change.

## Verified launch observations

- Actual platform state: running, HIGH/35, 16 NPUs; final attempt started
  September 25 at approximately 04:45 CST.
- Real Qwen k20 empty prompt retained five tokens; full 532,496,992-parameter
  DiT forward/backward produced finite loss/gradients. Record:
  `runs/ascend-hero-0925/caption-preflight.json`.
- SwanLab direct online initialization succeeded as run `i9r8wnah`; no relay.
- At **05:01 CST / update 572**, all 572 updates had applied with finite
  recorded loss/gradient norm. Latest loss **1.07022**, pre-clip gradient norm
  **0.46621**, smoothed throughput **882.47 samples/s**, rank-zero reported
  peak **48.31 GiB**; no clipping in the last 100 updates. These are
  early-warmup observations, not a new matched
  throughput benchmark or a long-run stability claim.
- The update-500 fixed panel (`a99a7a1db1f86b03`) recorded conditioning update
  RMS **0.00799**, response gain **0.66720**, induced prediction change RMS
  **0.00533**, maximum gate RMS **0.60407**, maximum residual RMS **4.78404**,
  and maximum branch norm gain **1.00194**. The t050 conditioning loss effect
  was **−0.0000162**. Gain had risen from **0.14282** at update 100 while
  training and evaluation loss improved: interpret the joint behavior, not a
  gain threshold.
- Cloud API readback independently confirmed the RUNNING identity and training
  metrics through update 138, including loss **1.63767** and **894.43 samples/s**.
- No recovery checkpoint exists yet; the first is due at update 2000. Initial
  EMA/live loss was **1.99821**, matching the qualified initialization. At
  update 500, EMA/live loss improved to **1.18204 / 1.13985**, with all four
  timestep losses lower for both copies. Evaluation took **52.63 seconds**.
- End-of-stage Inception weights are now cached under the platform's
  `TORCH_HOME/hub/checkpoints`: 95,628,359 bytes, SHA-256
  `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
  Official weights passed hash checking and CPU state loading (566 entries).
  Preparation log: `runs/ascend-hero-0925/kid-cache-prepare.log`. This removes
  the deferred download; it is not a new end-to-end KID qualification.
