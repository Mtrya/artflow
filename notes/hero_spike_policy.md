# Hero run loss-spike triage policy (ascend-hero-256p-v2)

Unattended spike response for the hourly patrol cron. Baseline = median of
`train/loss` over the 200 steps before the spike.

## Classification

- **Spike**: `loss > 1.5 × baseline`.
- **Tier 1 (self-healing)**: back within 1.1× baseline in ≤50 steps.
  Action: record only (step, peak, recovery length). No intervention.
- **Tier 2 (slow recovery)**: decreasing, but still >1.15× baseline at
  spike+500 steps.
  Action: roll back to the newest checkpoint before the spike (quarantine
  later checkpoints with the launcher RESUME_PIN mechanism), resume at the
  same lr. Slow re-convergence wastes more compute than the ≤2000-step
  rollback plus patrol lag.
- **Tier 3 (plateau / escalation / divergence)**: no downward trend at
  spike+200 steps, or a higher spike follows, or loss stays >1.3× baseline
  for 200+ steps.
  Action: roll back as above **and** resume with `MUON_LR` × 0.75 (the
  launcher override + base_lrs config-authoritative patch make the reduced
  base lr survive resume).

## Guards

- A spike younger than 500 steps at scan time is "pending" — recheck next
  patrol, never roll back early.
- At most 2 automatic rollbacks per run. A third qualifying event, or a
  spike recurring within ±500 steps of the previous one after a rollback,
  means a systematic cause (toxic data shard, config): **stop the job and
  page the user**, no further automatic action.
- Infra crashes (entire-restart) are separate: handled by job
  fault-tolerance + the crash template in the patrol cron.

## Notes

- Checkpoints every 2000 steps; rollback restores optimizer, EMA
  (bias-corrected) and sampler state together, so the post-rollback run is
  a clean continuation, not a mixed state.
- swanlab `train/grad_norm` is pre-clip; a grad spike alongside a tier-1
  loss spike is expected and not by itself actionable.

## Event log

- 2026-09-23 ~18:00: user-directed manual rollback (outside the automatic
  tiers). After repeated loss spikes and a grad_norm 79.2 single-step event
  @13793, hero5 was stopped at ~step 15000 and the run was resumed from
  checkpoint_step_014000 with MUON_LR 0.016 -> 0.012 and ADAM_LR 3e-4 ->
  1e-4 (job ascend-hero6-256p). Automatic-rollback counter resets for the
  new segment; stability timer restarts at the hero6 training start.
