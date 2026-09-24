# Ascend pretraining direction — 2026-09-24

The user decided to pivot pretraining from native CUDA to **pure Ascend
pretraining**. Subsequent pretraining debugging, architecture/configuration
experiments, performance optimization and production qualification target the
Ascend stack in the already authorized 昇腾卡公共空间. This supersedes the
four-H200 production direction in earlier September 24 records. It is a
pretraining decision; inference/deployment hardware is outside its scope.

The rationale is the cross-card evidence in the
[H200 investigation](h200_spike_root_cause_0924.md): the captured spike
reproduces in full fp32 on both H200 and RTX 4090, and a captured AdamW
conditioning-weight update causally triggers it. An Ascend-specific arithmetic
fault is not required for that failure. This does not prove that every
Ascend event has the same cause or that numerical execution can never change
the training trajectory.

## Experiment policy (user correction, September 24)

Current sequence, explicitly clarified by the user:

1. Establish stability through shorter experiments and internal-metric monitoring.
2. Clean the repository, consolidate configuration into the chosen architecture
   and sensible defaults, and remove stale documentation.
3. Run an infrastructure pass on that model and configuration.
4. Launch the hero.

We are in **Stage 1**. A successful short check does not authorize bypassing
Stages 2–3 or imply an immediate hero launch. See the
[Stage-1 experiment and telemetry contract](ascend_stability_stage1.md).
Optimize for useful decisions per unit of time and compute; do not require a
clean A/B for every candidate.

- Use relevant papers and official implementations as substantive priors.
  When multiple applicable sources support a choice, do not spend a long run
  merely re-proving it. Check that the implementation and assumptions match.
- Run a short experiment only when its possible outcomes would materially
  change the recipe or the diagnosis. State that decision and the expected
  observations before spending compute; extract multiple useful measurements
  from the same run rather than launching redundant probes.
- Use internal metrics together with the causal evidence and observed learning
  to choose a configuration under uncertainty. Neither universal spike
  predictors nor proof of complete stability are prerequisites to launch.
  A smaller proxy metric alone is not the objective.
- Adopt well-supported changes together when that is the efficient decision;
  clean attribution is a tool, not a mandatory experiment matrix.
- Use successive short experiments to establish a credible stable recipe,
  then complete cleanup and the infrastructure pass before the hero. The hero
  still supplies longer-term evidence; there is no additional long-validation
  run between the infrastructure pass and hero.
- Preserve useful progress and capture enough evidence at failure to make the
  next decision. Monitor loss/spikes, actual optimizer updates and skips,
  live/EMA quality, and selected mechanism-related metrics.

This supersedes both the assistant's earlier long-validation matrix and its
later suggestion to go directly from one replay to the hero. When an experiment
is running and no useful independent work remains, monitor sleep is authorized.

## Current evidence and work

- H200 production remains stopped; its snapshots, measurements and captures
  are references, not the current production destination. This decision does
  not validate a replacement hero recipe or a restart from damaged weights.
- The user reports that the fresh `tf1000-256p` experiment did not resolve the
  issue. On September 24 at 15:03 CST, platform reads confirmed
  `ascend-tf1000-256p` is stopped and its startup log confirms the factor 1000
  patch. Platform stdout contains no completed training metrics, so exact
  failure steps and severity are not independently established by this read.
  The local [timestep note](timestep_factor_0924.md) still records early-run
  status. Treat factor 1000 as **insufficient on its own**, not as proof that
  embedding bandwidth has no effect on the trajectory.
- The completed Ascend growth/normalization probes are recorded in
  [the Muon note](muon_weight_growth_0924.md). Its final accumulation-matched
  2k-step comparison reports baseline/branch evaluation loss 0.92154/0.91259
  and last-block activation RMS 12001/73.58; weight RMS and the fraction of
  conditioning preactivations below -5 are higher with branch normalization.
  These results support `branch_norm` as the leading hero candidate, without
  establishing that any single internal metric predicts future spikes.
  The user has handed further work to this agent; the previous experiment
  agent is finished. No additional long-validation requirement applies.
- The subsequent [stability review](ascend_stability_review_0924.md) separates
  remaining functional risks from raw norm/saturation statistics, checks Muon
  LR conventions against primary sources, and informed the
  [instrumented Stage-1 experiments](ascend_stability_stage1.md). Two 1000-update
  continuations have completed: calibrated rates reduce sampled conditioning
  update effects and improve live loss, with slightly worse EMA loss. A fresh
  2000-step combined-recipe check is running. These are stability experiments,
  not a hero launch or authorization to skip cleanup and the infrastructure pass.

## Time resolution and normalization address different properties

For a sinusoidal pair `e_k(t) = [sin(k*f*t), cos(k*f*t)]`, squared norm is
one for every factor k, while the local derivative norm is `abs(k*f)`.
Increasing k therefore gives more rapidly varying time features without
reducing their amplitude or constraining later learned weights. It does not
guarantee globally greater separation for every pair of times, nor that the
conditioning MLP will preserve the variation.

The proven H200 mechanism concerns a different quantity: an optimizer update
to nearly constant hidden features creates a shared conditioning shift, which
a sensitive downstream network amplifies. Richer time features can coexist
with that failure. Missing normalization is a plausible intervention target,
but its location and mechanism need to be tested:

- Conditioning-input normalization controls the input distribution; it does
  not guarantee that later SiLU hidden features retain sample variation.
- Normalization of conditioning features/output changes scale and update
  geometry, but does not generally remove a sample-independent vector or
  prevent direction changes.
- Branch-output normalization reduces sensitivity to branch amplitude;
  learned gates, modulation weights and affine norm gains can still grow.

The current `cond_norm` probe normalizes the conditioning MLP's **input**;
the `branch_norm` probe normalizes branch outputs **before** learned gates.
Neither is yet a validated cure. Read the experiments for hidden-feature
variation/saturation, shared conditioning displacement per update, modulation
and gate scales, sensitivity to conditioning perturbations, and sustained
loss/EMA/sample quality. Short probes can identify candidates; preventing one
spike or lowering activation maxima alone does not establish long-run health.
