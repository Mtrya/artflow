# Hero run loss-spike triage policy (ascend-hero-256p-v2)

> Archived record. Current pretraining decisions and status are in
> [the Ascend plan](../ascend_pretraining_0924.md); old launch commands and
> configuration switches require the source revision used for that run.

**STATUS (2026-09-24): SUSPENDED — root-cause investigation in progress,
user-directed. No automatic rollback / stop / lr change on hero6.** The tier
rules below are kept as reference for the future fixed hero run and will be
revised by the probe verdict.

## Key facts established by the investigation (2026-09-23/24)

- **Upstream driver found (2026-09-24 afternoon):**
  [Muon weight-norm inflation](muon_weight_growth_0924.md). Muon-routed weight
  RMS grows without bound during warmup (16× init at step 12k, +15 %/2000
  steps, equilibrium `0.2/muon_wd` = 680× init), the residual stream has no
  branch normalisation so it follows, and the near-constant conditioning vector
  `c` is then destroyed by any single AdamW update to the conditioning head.
  That note also carries the weight-RMS tables and four ranked fixes.
  Companion: [timestep-factor note](timestep_factor_0924.md) (lead tested and
  rejected — `time_factor=1000` does not move `c`'s cross-sample spread).
- **Loss returning to range means NOTHING about quality.** A DiT reaches
  0.8–0.9 train loss within ~1k steps and then spends tens of thousands of
  steps refining to 0.75–0.85. After the 26.6k tier-3 event, train/loss fell
  back to ~0.85 within ~100 steps — but the step-27500 EMA sample grids are
  pure noise texture across all 12 scenes (step-25000 grids are intact:
  coherent ink-wash figures, clean portrait faces). Loss 0.85 at 27500
  corresponded to noise-level generation. Quality signals are grids and
  eval/loss trend, never train/loss level.
- **Spikes are not an *lr* problem — but they are a recipe problem.** The
  4×4090 hero (lr 0.02 + 5k warmup, 53.5k steps) had every >50 grad-norm event
  inside the first 5k warmup steps; the Ascend run shows >50 events persist and
  intensify after warmup: 13793:79 / 14879:132 / 26359:270 / 26616:174 /
  27538:56 / 29165:86 (~1 event per 1–2.5k steps lately). Lowering lr
  0.016→0.012 only changed which steps events land on. That is expected once the
  driver is known: Muon's per-step displacement is `0.2·lr`, so lowering lr only
  slows the random walk that inflates weight norms, i.e. it delays the crossover
  without removing it (see `muon_weight_growth_0924.md`).
- **Not poisoned data.** Sampler replay of the spike windows (from
  checkpoint sampler states) shows no dominant bucket, no mix drift, and a
  retained-length distribution identical to baseline.
- Suspect: Ascend stack numerics (bf16 kernels / Muon Newton-Schulz).
  ~~Verdict pending the spike-batch precision probe~~ → probe v3 verdict in.
- **Probe v3 verdict (2026-09-24 01:15, `ascend-spikeprobe3-256p`): the spike
  batches are innocent; all precision legs are clean.** Replaying the exact
  micro-batches of spike steps 26359/26616/26617/27538 (all 16 ranks, both
  accumulation seeds) from ckpt-26000 gives grad_norm 0.63–1.09 — versus
  55–270 measured live at those steps. Leg A (text bf16 + DiT autocast bf16,
  = training config), B1 (DiT fp32), B2 (full fp32) agree with each other and
  with a CPU fp32 forward reference (loss 0.78–0.82); text-encoder activation
  absmax ~110 is identical in bf16 and fp32. Reading: NOT the data batches,
  NOT the NPU bf16 DiT kernels, NOT text-encoder bf16. The spike is a
  training-time dynamical event — it needs the live trajectory state
  (weights + optimizer), not just the batch. Results:
  `sj-ssd3:.../logs/spike_probe_results.jsonl` (648 rows) +
  `spike_probe_console.log` (PROBE_ALL_DONE).
- **Next experiment: trajectory replay (R1/R2).** Resume the full training
  state (model + Muon momentum + AdamW + EMA + sampler + RNG, all present in
  ckpt-26000) under the exact hero6 config and see whether the spikes
  reproduce at the same steps. R1 = bit-identical config (CODE_VER 0923b,
  MUON_LR 0.012, ADAM_LR 1e-4, RESUME_PIN 26000, run dir
  `runs/ascend-nsreplay-r1`, swanlab run `ascend-nsreplay-r1`). R2 = same but
  Newton-Schulz promoted to fp32 (env `MUON_NS_FP32=1`, code bundle
  CODE_VER=0924a, `src/pretrain/muon.py` `_NS_DTYPE`).
- **R1 result (2026-09-24 02:10), with the user's correction applied.** The
  bit-identical replay did NOT spike at 26359 (grad 0.43 vs live 270, loss
  0.86 vs live 1.53) or 26616-17 (grad 0.59/0.65 vs live 174) — but produced
  its OWN tier-1 grad spikes at 26149 (29.4) and 26190 (25.4) within the
  first 200 steps, no loss damage. **What this does NOT prove** (user,
  09-24): nondeterminism. A ckpt resume only approximates live's 26000-step
  state (live came 14000→26000 continuously); near a critical regime, tiny
  state gaps amplify over hundreds of steps and move spike steps around.
  **What it DOES prove**: spikes need no special 12k-step trajectory history
  — a plain resumed state on Ascend spontaneously spikes within hundreds of
  steps. The Ascend run sits near a critical regime as a persistent
  property; spike occurrence is common, severity varies (R1: 25-29 tier-1;
  live: 55-270 with occasional tier-3). The 4×4090 control (same model /
  data / recipe, lr 0.02 > Ascend's 0.012, zero post-warmup spikes in 53.5k
  steps) still says the regime is platform-specific. Open question:
  transient stack glitches vs NPU-numerics-driven edge-of-stability. Next
  discriminator: per-parameter spike structure (single dominant tensor →
  glitch; broad → dynamics) — dump hook added to the hero-candidate code
  (`grad_spikes.jsonl` in the run dir, CODE_VER=0924b). Note the grad spike
  is measured pre-clip and pre-optimizer, so Muon NS cannot be the direct
  source; R2 (NS fp32) still runs as a frequency check.
- **Fix track (needed regardless of root cause): spike-skip.** `optim
  .grad_spike_skip` (default 0 = off; hero.toml sets 10.0 — normal grad
  0.3-1.3, glitch spikes 25-270). On a boundary step whose pre-clip grad
  norm exceeds the threshold, the optimizer step AND the EMA update are
  skipped (momentum buffers untouched, poisoned grads dropped by zero_grad,
  scheduler/step counter still advance). Logs `train/grad_spike_skip` and
  `train/grad_spike_skips_total`. Rationale: one skipped step per few
  thousand costs nothing; one absorbed glitch poisons Muon momentum for
  ~20 effective steps and the EMA for ~10k steps (the 27500-grid lesson).
  Code: `src/pretrain/train.py` + `src/pretrain/config.py`, bundle
  CODE_VER=0924b.
- **hero6 will never be the hero run** (user, 2026-09-24). It keeps running
  only as a dynamics data source until the probe verdict; the real hero run
  restarts from scratch with spike-fixed scripts.
- **Per-parameter spike structure (hero7, 2026-09-24 ~06:00; first 217 rows
  of `grad_spikes.jsonl`) — the discriminator called for above.** The top-3
  tensors are **217/217 identical**: `c_mlp.2.weight`,
  `txt_pooled_proj.weight`, `c_mlp.0.weight`. The rest of the top-10 is a
  fixed set too (`c_mlp.2.bias`, `x_embedder.weight`, `txt_embedder.weight`,
  `blocks.N.modulation*.{weight,bias}`). So the spike has a **fixed dominant
  path** — neither a random single-tensor glitch nor a broad
  dynamics-driven excursion. The top-10 accounts for only roughly a third of
  the pre-clip norm; the remainder is diffuse across many tensors.
- **Routing correction that kills the fp32-NS hypothesis.**
  `c_mlp` / `txt_pooled_proj` / `x_embedder` / `txt_embedder` / `final_layer`
  are listed in `adam_name_patterns` (`src/pretrain/muon.py:246`) and are
  routed to **AdamW — they never touch Newton-Schulz**. Only
  `blocks.*.modulation*.weight` among that top-10 is a Muon tensor. Since the
  dominant spiking tensors bypass NS entirely, `MUON_NS_FP32=1` is very
  unlikely to be the fix. R2's 441-step zero-spike window is also too short to
  count as evidence: hero7 spiked once at 8561, stayed quiet ~2000 steps, then
  went persistent at 10400 — **the spikes are long-intermittent**, and a short
  window proves little either way.
- **`grad_spikes.jsonl` caveat (self-inflicted).** `train.py:1673` computes the
  per-parameter norms *after* `clip_grad_norm_` (`train.py:1652`), so the
  recorded `norm` values are globally rescaled (`max_grad_norm = 1.0`) and
  must **not** be compared across records — apparent stability like
  "0.945 → 0.855" is a clip artifact. The *ordering* is unaffected by a
  uniform rescale, so the fixed-dominant-set finding above still holds.
- **spike-skip has a failure mode: persistent spikes.** hero7 went from zero
  skips (step 8136) to `grad_spike_skips_total` 3172 at step 13705 and 6159 at
  16692 — each increment exactly equals the step increment, i.e. **100 % of
  steps skipped, weights frozen**. Independent evidence: `eval/loss` identical
  to four decimals at 15500/16000/16500 (0.8815), while `grad_norm` median
  climbs 62 → 87 → 80. The mechanism only protects while spikes are
  occasional; once they become the steady state, the protection *is* a halt.
- **lr-vs-onset correlation (suggestive, not proof)**: hero5/hero6
  (`ADAM_LR 1e-4`) onset ~13793 / 26358; hero7 (`ADAM_LR 3e-4`,
  `MUON_LR 0.02`) onset 8563. 8563 × 3.08 ≈ 26358 hints that update scale sets
  the onset step, but both lrs changed together and hero5→hero6 drifts 1.9×
  at constant lr.

## Classification (reference, for the future fixed run)

Baseline = median of `train/loss` over the 200 steps before the spike.

- **Spike**: `loss > 1.5 × baseline`.
- **Tier 1 (self-healing)**: back within 1.1× baseline in ≤50 steps.
  Action: record only (step, peak, recovery length). No intervention.
- **Tier 2 (slow recovery)**: decreasing, but still >1.15× baseline at
  spike+500 steps.
  Action: roll back to the newest checkpoint before the spike (quarantine
  later checkpoints with the launcher RESUME_PIN mechanism), resume at the
  same lr.
- **Tier 3 (plateau / escalation / divergence)**: no downward trend at
  spike+200 steps, or a higher spike follows, or loss stays >1.3× baseline
  for 200+ steps.
  Action: roll back as above **and** resume with `MUON_LR` × 0.75.

## Guards

- A spike younger than 500 steps at scan time is "pending" — recheck next
  patrol, never roll back early.
- At most 2 automatic rollbacks per run. A third qualifying event, or a
  spike recurring within ±500 steps of the previous one after a rollback,
  means a systematic cause: **stop the job and page the user**.
- Infra crashes (entire-restart) are separate: handled by job
  fault-tolerance + the crash template in the patrol cron.

## Notes

- Checkpoints every 2000 steps; rollback restores optimizer, EMA
  (bias-corrected) and sampler state together.
- swanlab `train/grad_norm` is pre-clip; a grad spike alongside a tier-1
  loss spike is expected and not by itself actionable.
- EMA (decay 0.9999) needs ~10k steps to flush a weight displacement; after
  a tier-2/3 event expect eval/loss and grids to stay degraded long after
  train/loss "recovers" (see 27500 grids: total destruction while train
  loss read 0.85).

## Event log

- 2026-09-23 ~18:00: user-directed manual rollback (outside the automatic
  tiers). hero5 stopped at ~step 15000 after repeated spikes + grad 79.2
  @13793; resumed from checkpoint_step_014000 with MUON_LR 0.016→0.012,
  ADAM_LR 3e-4→1e-4 (hero6).
- 2026-09-23 ~23:00 (hero6, lr 0.012): tier-1 @26358 (loss 1.531, grad 270),
  recovery within ~2 steps. Recorded.
- 2026-09-24 ~00:00: tier-3 @26616–26617 (grad 174, loss 1.99→slow decay).
  EMA eval/loss 0.867@26500 → 1.80@27500; step-27500 grids pure noise.
  NO action taken (policy suspended); investigation instead.
- 2026-09-24 00:10: further grad events @27538 (56) and @29165 (86, loss
  2.03 @29169, fast loss recovery). Event cadence ~1 per 1–2.5k steps.
  User: hero6 demoted to dynamics source; real hero run restarts fresh
  with fixed scripts.
- 2026-09-24 00:36–01:15: probe v3 (`ascend-spikeprobe3-256p`) ran to
  PROBE_ALL_DONE after v1 (PYTHONPATH clobbered CANN tbe) and v2 (fp32 leg
  backward OOM on seq² plain attention → chunked backward). Verdict: all
  legs clean → training-dynamics event, not batch/kernel/text-encoder.
- 2026-09-24 01:32: trajectory replay R1 (`ascend-nsreplay-r1`) launched:
  full-state resume from ckpt-26000, exact hero6 config (CODE_VER 0923b),
  window 26000→~27200 covers the 26359/26616 live spikes. R2 (NS fp32,
  CODE_VER 0924a with `MUON_NS_FP32=1` switch added to `src/pretrain/muon.py`)
  prepared; code bundle uploaded to the HF bridge.
- 2026-09-24 01:16: dl-640p-r7 relaunched (HF→sj-ssd3, resuming ~55%).
- 2026-09-24 ~02:30: hero6 crashed at step 35000 — root cause: the 0924a
  code bundle (built ad-hoc for R2) omitted `assets/`, and R2's launcher
  re-staged the SHARED `$W/repo-ascend` at 02:22, so hero6's eval at 35000
  found `assets/eval/hero_monitor_v1.jsonl` gone. User: do not restart
  hero6; its checkpoints deleted (103G freed), samples kept.
- 2026-09-24 02:35: R2 final read (441 steps): max grad_norm 3.26, zero
  spikes in the window where R1 spiked twice — fp32 NS suppresses spikes
  but costs ~100 samples/s. User decision: keep bf16 NS, rely on
  spike-skip, restart hero at recipe lr 0.02/3e-4. R2 stopped.
- 2026-09-24 02:36: **real hero `ascend-hero7-256p` launched from scratch**
  (RUN_NAME=ascend-hero-256p-v3, CODE_VER=0924b with assets restored,
  MUON_LR=0.02, ADAM_LR=3e-4, warmup 20k, T=600000, spike-skip=10.0).
- 2026-09-24 ~03:25: hero7 first read — step 344, train/loss 2.02→1.22,
  grad_norm max 1.34, `grad_spike_skips_total` 0, steady ~1.27 s/it. The
  4090 reference run (`runs/hero-256p`, job `hero-256p-4g-v2`, stopped at
  ckpt-52000) had its last checkpoint deleted (6.1G; `samples/` kept), so
  the 4090 control's weights are no longer available for re-inspection —
  its grid outputs remain the only retained artifact. Patrol collapsed to a
  single hourly cron (hero7 health + 640p download); the 6-hourly cron and
  its 4090/H200 tracking were cancelled by user decision.
- 2026-09-24 03:43: hero7 passed its last commissioning check — step-2500
  eval and grids both produced (`grid_step_002500_*.png`, 6 files), which is
  the exact step hero6 died on with the missing-`assets` bundle. 0924b fix
  confirmed in production.
- 2026-09-24 ~06:00: **hero7 spike-skip saturates.** First skipped step 8563;
  by 10400 the rate is 32 %, from 10600 it is 100 %. `grad_spike_skips_total`
  tracks step count exactly (3172 @13705, 6159 @16692) — optimizer frozen,
  scheduler still advancing. `eval/loss` flat at 0.8815 across
  15500/16000/16500; `train/loss` drifting 0.85 → ~1.03 (damage happened in
  the 10400–10600 partial window). Per-parameter dump analysed (217 rows):
  fixed dominant tensor set, AdamW-routed → fp32 NS de-recommended, R2's
  short window discounted. **Awaiting user decision**: layer-wise activation
  comparison (ckpt-8000 healthy vs ckpt-10600 blown, Ascend vs 4090) to
  separate model state from platform numerics, or abandon the Ascend
  pretraining line. hero7 left running (idle, no output) pending that call.
- 2026-09-24 ~09:45: **the platform hypothesis is dead, and the layer probe
  found the real signature.** (a) Another agent's `spike-replay-4090-0924`
  replayed the H200 spike batch (step 12829) on a 4090 and reproduced
  grad_norm **41.76** against H200's captured 41.21, cosine 0.9998, TF32 on
  and off alike; `per_parameter` reported `c_mlp.2.weight` 39.76 and
  `txt_pooled_proj.weight` 9.62 — **the same top-2 tensors as hero7's 217
  Ascend dumps**. So the spike is a cross-platform deterministic product of
  (weights × batch), not an Ascend numerics artefact. That also retires the
  plan to ship the checkpoints to a 4090 for comparison: the weights are the
  cause, so any platform reproduces it. H200's own `ns-replay-0924` rejects
  fp32 NS ("it also produces severe gradient spikes and EMA texture
  collapse"). (b) The layer probe (`scripts/diag/layer_probe.py`, one fixed
  4-row batch, t=0.5, same batch_rows for both runs) on hero7
  ckpt-8000 vs ckpt-12000 gives the signature: inputs and conditioning are
  clean (`x_embedder` 1.01, `txt_embedder` 1.83, `c_mlp` **1.12**,
  `txt_pooled_proj` **0.76**, `t_embedder` 1.00, every one of the 26
  `blocks.N.modulation` outputs 0.98–1.36) and `model_output` is essentially
  unchanged (5.375 → 5.25), **but the single-stream image stream is amplified
  38–81× in blocks 12–22**, stepping up 10× at block 12 (11:5.1 → 12:77.1),
  then decaying at 23–24 (13.7–16.7). `blocks.*.txt` stays at 4.2–6.3. **The
  conditioning-path tensors are innocent** — my earlier "conditioning head
  weights are broken" reading was wrong; what changes is the internal scale of
  the trunk from block 12 onward. This is an internal reparameterisation:
  output preserved, activations ~80× larger, which is exactly what makes the
  backward gradients blow up 40–270 while `train/loss` still reads ~0.9.
  Caveat: single batch, single t — the 80× gap cannot be noise, but whether
  "steps up at block 12" is robust wants a multi-batch/multi-t check, and a
  ckpt 2000→18000 sweep would date the onset of the drift.
- 2026-09-24 ~09:40: **Ascend-side tooling gotcha, second occurrence.** The
  first probe job "succeeded" in 48 s having done nothing: the launcher set
  `PYTHONPATH=$W/pylibs` **without** appending `${PYTHONPATH:+:$PYTHONPATH}`,
  clobbering the CANN-injected path, so the first op needing tbe compilation
  died with `Failed to import Python module ModuleNotFoundError: No module
  named 'tbe'` → `tbe_op_adapter` init failure → `GEInitialize failed`. This is
  the same trap probe v1 hit. Additionally the probe call sat inside an
  `if/else`, which `set -e` exempts, so a hard failure still exited 0 and the
  job reported success. Both fixed; `ulimit -n` and TF32 were ruled out by a
  three-arm minimal NPU test that passed. hero7 stopped on user instruction
  (16 cards released); `ascend-dl-640p-r8` died silently at 74 % and was not
  restarted, since the value of that data depends on whether the Ascend
  pretraining line survives.
- 2026-09-24 08:00-09:40: **fix-ablation probe** (`branch_norm` = sandwich
  RMSNorm on every block branch, `cond_norm` = LayerNorm on the conditioning
  MLP's input; both implemented as `ModelConfig` flags, init-transparent, +0.012 %
  parameters, unit tests in `tests/test_conditioning_flags.py`). Four arms of
  2000 steps at full LR on 16×910B, launcher
  `scripts/ascend/branch_norm_probe.sh`, analysis
  `scripts/diag/summarize_layer_probe.py`. Results and the full ledger are in
  `muon_weight_growth_0924.md` §9-§12; the two headlines so far:
  **`cond_norm` is falsified** (it de-saturates `c_mlp.0` exactly as designed but
  every conditioning ratio gets 2× *worse* and the module's response to the
  shared input direction grows 7.4× off a 4.9 % weight change), and the
  accum=2/half-micro-batch control shows the **internal activation scale moves
  2-5× between runs whose weights, conditioning ratios and loss agree to 1-2 %**,
  so only accumulation-matched arm comparisons are admissible. Two infra traps
  hit and fixed here: `branch_norm` OOMs the stock micro-batch (51.4 GiB
  baseline peak on a 64 GB card), and the teardown watchdog must be scoped to the
  arm it watches or it kills the next one.
- 2026-09-24 09:20-09:50: **fix-ablation probe closed.** All four arms measured;
  full analysis in `muon_weight_growth_0924.md` §13. `branch_norm` **passes the
  shape criterion** and is recommended for the hero recipe: across depth the
  activation RMS goes from a 637 → 1265 → 8697 step plus an 18.8× total runaway
  (`off-a2`) to a bounded 23.3 → 89.1 → 73.6 rise with no step and a decaying
  block-to-block ratio (`branch-a2b`), with `eval/loss` 0.92154 → 0.91259 at step
  2000 and the gain present from the first 500-step evaluation and spread over
  every timestep. Costs, both measured: **−8.6 % steady throughput** (715.1 →
  653.7 samples/s, `--no-compile`) and **+4.3 GiB** peak (33.4 → 37.7 GiB) on the
  halved plan. Two things it does not do: Muon block weight RMS still rises
  (0.556 → 0.585), so `muon_wd` stays in the recipe as the independent lever, and
  `c_mlp.0` SiLU saturation gets worse (0.555 → 0.666), which only `cond_norm`
  fixes — at the price of the worst conditioning ratios of the three arms, and of
  throwing away `branch_norm`'s 8.5× modulation-amplitude gain (12.74 → 2.943
  RMS), so **`cond_norm` stays out**. Decisive caveat recorded with the result:
  at 2000 steps this probe cannot show that spikes are gone (the 4090 hero's
  onsets were 8.5k–26k); it shows the §1-§3 amplification mechanism is gone. The
  spike-skip guard and tier policy stay in force, and the 8.6 % throughput cost
  is flagged for one compile-on measurement before launch rather than accepted.
  Both probe jobs are terminal; each reports `failures=1` for bookkeeping only
  (a watchdog kill, see §12).
