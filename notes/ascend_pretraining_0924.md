# Ascend pretraining direction — 2026-09-24

The user decided to pivot pretraining from native CUDA to **pure Ascend
pretraining**. Subsequent pretraining debugging, architecture/configuration
experiments, performance optimization and production qualification target the
Ascend stack in the already authorized 昇腾卡公共空间. This supersedes the
four-H200 production direction in earlier September 24 records. It is a
pretraining decision; inference/deployment hardware is outside its scope.

Branch policy clarified by the user on September 24: **`main` itself pivots
to Ascend-flavored pretraining**. Do not maintain a separate Ascend pretraining
branch. The temporary `ascend-pretraining` branch was fast-forwarded into
`main` and removed; its completed work and the existing working-tree edits
were preserved.

The rationale is the cross-card evidence in the
[H200 investigation](archive/h200_spike_root_cause_0924.md): the captured spike
reproduces in full fp32 on both H200 and RTX 4090, and a captured AdamW
conditioning-weight update causally triggers it. An Ascend-specific arithmetic
fault is not required for that failure. This does not prove that every
Ascend event has the same cause or that numerical execution can never change
the training trajectory.

## Experiment policy (user correction, September 24)

Current sequence, explicitly clarified by the user:

1. Establish stability through shorter experiments and internal-metric monitoring.
2. Clean the repository, consolidate configuration into the chosen architecture
   with explicit run settings, and remove stale documentation.
3. Run an infrastructure pass on that model and configuration.
4. Launch the hero.

**Stage 1 closed at 20:39 CST on September 24. Repository/config/docs cleanup
is complete as of September 25; infrastructure qualification is next.** The completed short checks do not authorize bypassing cleanup and
the infrastructure pass. See the
[archived Stage-1 evidence](archive/ascend_stability_stage1_0924.md).
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
  The local [timestep note](archive/timestep_factor_0924.md) still records early-run
  status. Treat factor 1000 as **insufficient on its own**, not as proof that
  embedding bandwidth has no effect on the trajectory.
- The completed Ascend growth/normalization probes are recorded in
  [the Muon note](archive/muon_weight_growth_0924.md). Its final accumulation-matched
  2k-step comparison reports baseline/branch evaluation loss 0.92154/0.91259
  and last-block activation RMS 12001/73.58; weight RMS and the fraction of
  conditioning preactivations below -5 are higher with branch normalization.
  These results support `branch_norm` as the leading hero candidate, without
  establishing that any single internal metric predicts future spikes.
  The user has handed further work to this agent; the previous experiment
  agent is finished. No additional long-validation requirement applies.
- The subsequent [stability review](archive/ascend_stability_review_0924.md) separates
  remaining functional risks from raw norm/saturation statistics, checks Muon
  LR conventions against primary sources, and informed the
  [instrumented Stage-1 experiments](archive/ascend_stability_stage1_0924.md). Two 1000-update
  continuations have completed: calibrated rates reduce sampled conditioning
  update effects and improve live loss, with slightly worse EMA loss. A fresh
  2000-step combined-recipe check has now completed successfully. These are stability experiments,
  not a hero launch or authorization to skip cleanup and the infrastructure pass.

## Selected model and optimizer for cleanup

- Branch normalization on; conditioning-input normalization off; timestep factor 1000.
- Muon **original scaling, LR 0.02**, auxiliary AdamW LR 1e-4, Muon decay 0.0015;
  gradient clip 1.0. The user requested the scaling change on September 24.
- h1152, 16 heads, 1 double-stream + 24 single-stream blocks: **532,766,716 parameters**.
  Double-stream modulation none, single-stream modulation layer, as actually tested.
- Bias-corrected EMA retained. Fresh hero remains 600k steps with 20k warmup;
  the short check's 200-step warmup is not the production schedule.

Muon now uses `sqrt(max(1, rows / cols))` on each independently orthogonalized
chunk, matching PyTorch's `original` convention. Square and wide chunks use
multiplier 1; tall chunks retain the aspect-ratio correction. LR 0.02 is the
base LR in this convention. See the
[official PyTorch Muon documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html)
and [implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/_muon.py).

This supersedes the accepted `match_rms_adamw` candidate at LR 0.003. The
completed stability experiments used that previous convention; their archived
measurements remain unchanged. The change makes the LR convention explicit
and follows the original reference, without asserting a new optimal LR.

Concrete conversion for a 1152×1152 matrix, holding the orthogonalized update
`Q` fixed and omitting weight decay:

| Convention | Base LR | Shape multiplier | Update |
|---|---:|---:|---|
| Original, selected | 0.02 | `sqrt(max(1, 1152/1152)) = 1` | `-0.02000 * Q` |
| RMS matching, historical baseline | 0.02 | `0.2 * sqrt(1152) = 6.7882` | `-0.13576 * Q` |
| RMS matching, stability candidate | 0.003 | `0.2 * sqrt(1152) = 6.7882` | `-0.02036 * Q` |

Thus 6.79× compares the first two rows at the **same numerical LR**. The
new coefficient is close to the tested candidate for this shape, but this is
not an exact whole-optimizer conversion. Other input widths change by different
ratios. The user accepted Muon decay **0.0015**, preserving the tested peak
shrinkage: `0.02 * 0.0015 = 0.003 * 0.01 = 0.00003` per update. This preserves
decay throughout a schedule with the same multiplicative LR factors. Auxiliary
AdamW decay is a separate setting and remains 0.01.
The implementation changes existing checkpoint continuation behavior too;
use the source snapshot recorded for an experiment to reproduce its old
trajectory. The next infrastructure pass will exercise the new convention.
The continuation experiments also changed auxiliary AdamW LR, so their
conditioning improvements cannot be attributed to Muon LR alone.

The fresh check of the previous RMS-matching candidate applied all 2000 updates,
with no post-warmup clipping. Its last
500 steps have maximum gradient norm 0.40657. Final EMA/live loss is
**0.85689331/0.86698182**, versus **0.85988654/0.87153464** for the earlier
branch-norm checkpoint at the same training age and on the same evaluation panel.
The largest sampled conditioning loss effect over the fresh run is 0.00300884.
Raw weight growth and large shared conditioning shifts still exist; they are
not sufficient failure criteria. These results support moving to cleanup,
not a claim that long-run spike prevention is proven. Both jobs have succeeded
and released their allocations. No new hero was launched by this stage.

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

## Configuration contract — clarified September 24–25

The cleanup is a reduction in actual choices and configuration sources, not
merely a relocation of model construction or dashboard filtering.

- One complete run file contains shared capacity/optimizer/training settings
  and explicit per-stage datasets, bucket plans, accumulation, and endpoints.
  Missing fields are errors; there is no base config, layered fallback, or
  environment/CLI hyperparameter override.
- Settled architecture and execution choices become native code, with old
  switches and superseded implementations removed. No `pretrain/model.py`.
- Capacity, optimizer/EMA numbers, data weights, and the numeric parameters of
  caption selection/loss weighting remain tunables. SwanLab displays all of
  them, all stages, the architecture identifier, and the active bucket table.
- Breaking obsolete configs, model variants and checkpoints is accepted.
  Historical source revisions preserve historical experiments.
- The interview established these choices one question at a time. Do not
  restart it or use asynchronous questions for dependent decisions.

The selected production candidate remains original Muon LR 0.02, decay 0.0015,
AdamW LR 1e-4/decay 0.01. The actual-update ratio versus the tested RMS-matched
0.003 candidate is about 0.982 for most matrices and 0.601 for the FFN down
projection. The latter is a meaningful change, to be exercised in the infra
pass. The 1000-update continuation tradeoff was 0.46% worse EMA loss and 1.51%
better live loss for the conservative rates, with smaller sampled conditioning
response. This gives no reliable estimate of distance to a global optimum.
Do not add a low-information optimizer sweep or a separate long-validation
stage before infrastructure qualification.


## Cleanup verification — September 25

The single-file schema, native architecture, trainer, checkpoint path and
launcher are implemented on `main`, without a new pretraining model wrapper.
Legacy pretraining overlays/launchers and completed probe scripts are removed;
closed records moved to `notes/archive/`. README work remains deferred to
Stage 7; the current recipe and launch instructions live in `notes/`.

Validation: 781 tests passed, including strict missing-field/type checks,
full-checkpoint writing and retention, stage horizons, launcher locking,
NPU RNG verification and profiler routing. A saved nonzero forward/backward
reference for the selected architecture matches exactly on CPU after switch
removal; the full model remains 532,766,716 parameters. Config validation also
runs through the real entry point without loading models or datasets.

These local checks do not qualify runtime performance or memory on Ascend.
The [infra pass](infra_pass.md) must supply the actual three bucket artifacts,
confirm their accumulation values and exercise full recovery on the target
runtime. No new compute job or hero was launched by cleanup.
