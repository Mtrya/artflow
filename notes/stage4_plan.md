# Stage 4 — Capability forecasting and the complete pre-NFT hero recipe

Status: design agreed through the user interview on 2026-09-11. The capability,
budget, and serving requirements below are agreed constraints. Experimental
allocations and statistical implementation choices marked provisional are design
proposals to finalize inside Stage 4, not measurements or promises of success.

This plan refines [the roadmap](redesign_plan.md). It does not launch experiments,
authorize additional spending, or declare Stage 3.5 complete. Stage 4 execution
uses the accepted outputs of [Stage 3.5](stage3_5_plan.md), including amendments
and measured outcomes in [its execution record](stage3_5_pilot.md).

## 1. Outcome and stage boundary

**Stage 4 must deliver the complete, executable pre-NFT hero-run recipe, so
Stage 5 can proceed directly. There is no planned Stage 4.5.**

The recipe covers training from scratch through **256p → 640p → 896p**, with
**1024p optional**, including model size, corpus, stage-dependent data mixtures,
caption policy, effective batch, optimization, resolution transitions, training
duration, evaluation, inference settings, and operational preparation. Stage 5
executes and verifies this recipe; it is not another recipe-discovery phase.
Later DiffusionNFT training belongs to Stage 6 and is excluded.

This is a conditional handoff, not a requirement to produce a positive verdict:

- If the evidence supports the targets within the budget and serving constraints,
  close Stage 4 with the recipe and a go recommendation.
- If evidence is insufficient, use the remaining Stage-4 budget for the most
  informative validation. If still unresolved, report that Stage 4 cannot close
  for hero execution; do not disguise missing work as a routine Stage 4.5.
- If the design is infeasible, present a redesign decision. For example, a
  credible estimate that the core targets need **4,000 RTX 4090 hours**, exceeding
  the **2,200-hour** hero ceiling, requires explicit agreement on a changed
  budget, architecture/data strategy, or capability scope. It does not authorize
  a 4,000-hour run or a weaker target by default.

The same rule applies if required quality cannot fit the serving memory/latency
envelope. Redesign is an exception prompted by evidence, not unfinished planning.

## 2. Agreed base-model capabilities

The project prioritizes ordinary useful painting generation, not an exhaustive
human-anatomy or spatial-reasoning benchmark.

| Capability | Required before NFT | Outside the core gate |
|---|---|---|
| Full-body figures | Ordinary standing, sitting, and everyday movement; coherent proportions, limb count, joints, and connections | Extreme poses, difficult foreshortening, complex multi-person contact |
| Hand–object interaction | Everyday interaction with common objects; plausible hands, contact, grip, and object geometry | Intricate finger choreography, unusual tools, highly occluded multi-hand interactions |
| Faces | Plausible facial structure, feature placement, and ordinary views/expressions at an assessable scale | Identity matching, biometric accuracy, photographic detail in an abstract style |
| Architecture | Recognizable common building types, coherent overall perspective/structure, plausible roofs, doors, and windows | Exact landmark reconstruction, blueprint precision, strict window counts |
| Painting style | Recognizable requested medium, marks, palette, and degree of abstraction in the six styles below | Exact named-artist imitation or photorealism as a product requirement |
| Basic layout | Two or three main entities and a few simple relations involving common objects | Dense arrangements, exact coordinates, long chains of spatial constraints |

The six styles are:

- Chinese painting: **ink wash, light-color ink painting, gongbi**.
- Western painting: **representational oil painting, watercolor, impressionistic
  painting**.

Anatomy and architecture must work in both painting families. Judge detail and
perspective according to the requested style: sparse ink need not show realistic
skin, and Chinese painting need not adopt photographic single-point perspective.
Abstraction is not permission for unintended extra limbs or impossible joins.
Photographic prompts may diagnose transfer but cannot substitute for painting
performance.

Layout covers viewer-left/right/center, above/below, beside, in front of/behind,
and foreground/middle-ground/background. An ordinary example is a person on the
left, a pavilion on the right, and mountains in the background. The base model
must already show this basic arrangement ability; NFT can refine its precision.

### 2.1 What NFT may improve

NFT may improve reliability within demonstrated capabilities, finer prompt
adherence, spatial precision, aesthetic preference, and stylistic polish. It is
not the justification for omitting the base-model gates. In particular, do not
launch on the assumption that NFT will create anatomy competence that the base
model almost never exhibits.

This is a project risk boundary, not a theorem that preference training cannot
improve anatomy. [BACON](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Kas_We_Still_See_Broken_Limbs_Towards_Anatomical_Realism_in_GenAI_WACVW_2026_paper.html)
reports anatomical improvements from preference learning and sample selection;
its results do not guarantee rescue for this model or these painting domains.
[DiffusionNFT](https://arxiv.org/abs/2509.16117) supplies a post-training method,
not an ArtFlow-specific pretraining capability forecast.

## 3. Evaluation contract

### 3.1 Headline target: 80% first-attempt success

The target is **80% joint whole-image success in each agreed evaluation category**,
using ordinary user-written prompts, **without API prompt polishing**. Chinese
and English must meet the gates **independently**; a pooled bilingual score is
not sufficient.

A success requires the requested capability, recognizable requested style, and
the relevant objects, action, and framing to be present together. Score anatomy
subcategories separately; good faces cannot average away weak hand interaction.
For capability tests in painting, report category × language × painting-family
cells with balanced coverage of the three styles in each family. Also publish
per-style results and a dedicated style-recognition breakdown. Freeze the exact
gate table before comparative results are inspected, not after seeing weak cells.

No retries, best-of-N selection, repair, prompt rewriting, or favorable seed
selection enter the headline score. Multiple preassigned seeds estimate the
distribution of independent first attempts; they are not chances to select one
successful image. Best-of-k can be a separate diagnostic of occasional correct
generation, never a replacement for pass@1 or a guarantee of NFT gains.

Cropping a requested full-body figure, hiding a required hand, or omitting the
interaction object is a failure, not a way to avoid anatomy judgment. Natural
occlusion is allowed where compatible with the prompt. An unassessable required
feature does not count as a pass. Record missing-content, structural, style,
layout, and reviewer-uncertainty labels separately for diagnosis.

### 3.2 Freeze development and confirmation suites

Stage 4 must deliver versioned prompt manifests, explicit rubrics, fixed seed
lists, and image-level result records. Include:

- Matched Chinese/English situations, checked for natural wording and equivalent
  difficulty; do not treat translations as independent scenario diversity.
- Ordinary short/medium prompts for the core user task, plus a separate
  long-prompt adherence panel and difficult stress cases. Freeze length bands
  and aggregation rules; do not replace ordinary prompts with elaborate
  benchmark-specific instructions to raise the headline score.
- Diverse common people, poses, objects, building types, and compositions,
  with suitable framing and feature visibility. Include variable aspect ratios.
- A development suite for recipe selection and a held-out confirmation suite.
  Evaluation images must be separated from training rows and near-duplicates.
  Once confirmation is used for tuning, it is no longer an untouched test.

The exact prompt counts, seeds per prompt, uncertainty coverage level, and
multiple-gate reporting procedure were **not numerically settled in the
interview**. Finalize them during evaluator calibration, within the evaluation
budget, and before recipe comparisons. Plan sample counts around the precision
needed to distinguish a near-threshold result; a few attractive grids cannot
establish 80%. These choices must not remain open at Stage-4 exit.

### 3.3 VLM-first review with calibrated uncertainty

The execution agent acts as the VLM reviewer. Calibrate on clear positive and
negative examples, borderline cases, and style-specific abstractions. Review the
full image and relevant crops; crops must not hide scene-level failures. Blind
the reviewer to recipe identity where practical, pin its rubric and model/version,
and retain decisions with reasons and uncertainty flags.

Escalate consequential ambiguity and systematic disagreements to the user,
including a sample of apparently confident passes. Do not plan a large manual
labeling project by default, but do not treat reviewer confidence as accuracy.
Record inter-review consistency and calibration errors by category/style.
Pose/mesh detectors and image-text scores are auxiliary diagnostics, not ground
truth for painted anatomy. An editing benchmark, [MPIE-Bench](https://arxiv.org/abs/2607.27616),
reports optimistic VLM checklist scores despite structural failures; its task
differs from ours, but motivates explicitly testing judge reliability.

KID, fixed-distribution flow loss, and caption adherence components support
diagnosis. None alone establishes anatomical competence or substitutes for the
joint success target. API-billed review needs a separate cost estimate and
approval; Stage 3.5's caption API allocation is not an unclaimed Stage-4 budget.

## 4. Serving contract and training–inference tradeoff

Agreed target: **RTX 4060 Ti 16 GB, one 896p image, batch size 1; approximately
10 seconds preferred and 20 seconds maximum**, excluding optional API prompt
polishing. Base quality and latency must be assessed under the **same inference
settings**. An 80% result at 50 sampling steps cannot qualify a different,
faster configuration without evaluating that configuration's quality.

Stage 4 selects and records solver, steps and actual model evaluations, guidance,
EMA checkpoint policy, precision, attention/compile settings, and residency or
CPU offload. The current local pipeline supports component-wise offload of the
text encoder, DiT, and VAE, but this is not measured proof of meeting the target.

Measure finalists on the actual 4060 Ti, including text encoding, denoising,
decoding, and per-request transfers. Record CPU/RAM, software versions, peak
allocated/reserved VRAM, host memory, and memory headroom. Test representative
and longest supported prompts and all declared service aspect ratios; distinguish
caption-generation target lengths from the actual retained-token context cap.

Proposed timing protocol: report a resident service's request latency separately
from initial model loading/compilation, and disclose both cold and warm results.
Freeze the timing boundary and supported shape/prompt envelope before declaring
compliance. Report individual runs and median/tail/maximum, not just a favorable
average; a mean below 20 seconds does not establish the maximum requirement.
If a supported case exceeds the limit, it needs a remedy or an explicitly agreed
service-contract change, not omission from the benchmark.

Do not estimate 4060 Ti serving latency by scaling 4090 training throughput.
Sampling correctness and consistency between evaluation and serving paths are
Stage-4 readiness checks. Necessary in-scope fixes and regression tests belong
inside Stage 4; discovering them must not silently defer validation to Stage 5.

Prefer measured candidates that satisfy both quality and serving cost. A smaller
model trained longer is a candidate, not a guaranteed winner. If quantization or
distillation is essential, explicitly scope, budget, implement, and validate it
within the recipe decision, or request redesign; do not leave a hidden pre-NFT
optimization stage. Optional 1024p has its own quality/memory/latency qualification
and cannot invalidate the required 896p deliverable.

## 5. What scaling literature can and cannot decide

Diffusion scaling literature supports controlled compute/model/data experiments,
but supplies no universal coefficients for this corpus, optimizer, conditioning
stack, or anatomy rubric. [Scaling Laws for Diffusion Transformers](https://arxiv.org/html/2410.08184)
demonstrates empirical compute-optimal scaling and prediction beyond small-run
budgets. It motivates the method, not a guarantee of our extrapolation.

[Abra: Scaling Diffusion Image Training](https://arxiv.org/html/2608.17286v1)
studies flow-matching transformers and finds approximately 200 image tokens per
parameter compute-optimal in its setting. This counts consumed image tokens,
not unique images or text tokens, and excludes the frozen text encoder from
model size. Its architecture/data/optimizer differ from ArtFlow. Different
generative metrics favor different allocations; its appendix also finds the
additive parametric loss surface poorly identifiable. Consequently, use its
findings as priors and check local iso-compute curves rather than inserting our
parameter count into its rule to declare a hero recipe.

Use explicit quantities:

- `N`: trainable DiT parameters; report frozen encoder/VAE size separately.
- `U`: unique eligible training images, also broken down by source/stratum.
- `S`: actual image–caption draws consumed, including repeats.
- `T_img`: consumed image tokens, summed over actual shapes/resolutions.
- `w(r, t)`: data-mixture weights at resolution `r` and training progress `t`.
- `H`: RTX 4090 GPU-hours, with training-only and allocated-time totals separated.

A candidate loss model such as

\[
L(N,S \mid U,r,w,\text{recipe})
=L_\infty+A N^{-\alpha}+B S^{-\beta}
\]

is conditional, not an independently validated law for all these axes. Fit it
only when observations identify its parameters and held-out tests support it.
Otherwise use simpler local curves and iso-compute comparisons with uncertainty.
Do not fit every mixture, resolution, data-size, and optimization interaction
from a small grid. Raw losses under different resolution/time-shift definitions
are not automatically comparable.

[Scaling Laws for Optimal Data Mixtures](https://arxiv.org/html/2507.09404)
models the dependence on model size, training exposure, and mixture in language,
native multimodal, and vision settings. Its useful lesson here is to test mixture
and scale together against fixed target domains, not transfer its coefficients
to a DiT or optimize thirteen unconstrained source weights at every phase.

### 5.1 Forecast capability directly

Measure category-specific success trajectories, not just a loss-to-anatomy
conversion. Research on [direct downstream-metric scaling](https://arxiv.org/abs/2512.08894)
supports investigating such fits, while [work on prediction failures](https://arxiv.org/abs/2406.04391)
shows why predictable loss need not yield equally predictable accuracy. Both
concern language models, not an established scaling law for painted anatomy.

For a fixed recipe family, an illustrative candidate is failure rate

\[
e_c(H)=e_{\infty,c}+a_c H^{-b_c},\qquad p_c(H)=1-e_c(H).
\]

If the fit is credible and `0.20 > e_inf,c`, its implied target budget is

\[
H_{80,c}=\left(\frac{a_c}{0.20-e_{\infty,c}}\right)^{1/b_c}.
\]

This is a proposed local model, not a promised anatomy law. A weakly identified
floor, flat/near-zero success, or a target far outside validated scale makes
inversion unreliable. Report an unresolved requirement rather than a fabricated
finite budget. Compare simple alternative fits and sensitivity to dropping runs;
retain bounded probabilities and uncertainty rather than selecting a convenient
curve. A multi-resolution schedule needs its own validation, not an `H`-only
fit pooling unrelated recipes.

Separate these diagnoses:

- **Compute-responsive:** observed capability improves with longer/larger runs;
  estimate a budget interval conditional on the tested recipe.
- **Recipe-limited:** evidence points to data coverage, quality, resolution, or
  conditioning shortcomings; more hours alone are not the justified remedy.
- **Unresolved:** experiments cannot distinguish the two; identify a bridging
  experiment and its cost.

Hold out a larger and/or longer run from initial fitting and predict it before
observing its capability scores. Include resolution-continuation checks: 256p
figures cannot establish 896p hand/face detail. Account for prompt clustering,
training-seed variability, reviewer error, and extrapolation/model uncertainty;
an interval on observed sample accuracy alone is not a hero prediction interval.

**Launch gate:** the conservative lower end of the validated forecast must clear
80% for every required gate, including Chinese and English independently, at the
chosen serving settings. Freeze the uncertainty method and coverage level before
comparison. Forecast validation must include checking interval calibration and
systematic prediction error, not merely fitting the held-out point afterwards.
The 450-hour cap does not guarantee that such a forecast will be possible.
Stage 4 forecasts; Stage 5 must still verify actual achievement.

## 6. Corpus, exposure, and stage-dependent mixture design

Stage 3.5 selects captions within an already selected row and experiments with
length-related weighting. Stage 4 additionally decides the row-level mixture
across phases. Keep these distinct controls explicit:

`domain/source → quality and capability strata → row → caption → loss weight`

Strata should describe properties relevant to training: grounded caption length,
image quality, useful anatomy/interaction/layout coverage, language availability,
and eligible resolution. Do not revive an original-versus-enriched provenance
split as a quality metric; the Stage-3.5 execution record explicitly dropped it.
Keep license-restricted sources separately identifiable.

For source/stratum `d`, let `q_d(t)` be the probability that its selected caption
has at least 256 retained tokens. Then actual long-caption draw exposure is

\[
q_{\rm long}(t)=\sum_d w_d(t)q_d(t).
\]

If only 20% of row draws can supply a long caption and those choose one 60% of
the time, global exposure is 12%, not 60%. Increasing exposure changes repetition
and diversity unless more eligible rows are added. Length-based loss weighting
changes weighted loss mass, not the number of distinct images or long-caption
draws; it is not a measured gradient-norm contribution.

The September-11 Stage-3.5 simulation reported approximately 6–9% long-caption
draws during candidate ramps and about 12–13% for several endpoint policies.
These are conditional simulated exposures, not training gains or fixed hero
targets. Refresh them from the accepted corpus and final Stage-3.5 policy.

Use a low-dimensional search:

1. Audit unique eligible rows, quality/capability coverage, caption distributions,
   and expected repeats at each resolution. Large source size does not establish
   either quality or usefulness; prioritize visible full bodies and actual
   hand–object contact, not just more portraits.
2. Screen baseline, long-caption-capable-row boost, quality/capability boost, and
   their combination at matched GPU budgets. Bound source/domain drift and
   repetition; keep the accepted within-row and loss-weight policy fixed initially.
3. Compare the best stationary mixture with a restrained staged alternative,
   such as broad coverage earlier and more high-quality/detailed examples later.
   Match cumulative exposures where possible to isolate ordering from total dose.
   Poor-quality data early is a hypothesis, not a required curriculum.
4. Check the promising mixture at another model size and through higher-resolution
   continuation. Resolutions change which rows are eligible, so audit the realized
   rather than nominal mixture again after each transition.

Compare on a fixed evaluation distribution. Do not select a mixture because its
own training loss is lower or because dropping costly long text raises samples/s.
Keep short-prompt, language, style, and domain guardrails. Per-source multipliers,
eligible-row definitions, schedule breakpoints, and normalization semantics must
be executable rather than prose such as “favor high quality later.”

## 7. Experimental program and budget

**Stage-4 experiment/profiling cap: 450 RTX 4090 GPU-hours**, separate from
Stage 3.5's 75-hour cap and the hero working range of **1,600–2,200 hours**.
Do not borrow between these budgets or use older off-peak flexibility as standing
authorization to exceed the current range.

Provisional allocation, to refine before jobs launch:

| Work package | RTX 4090 GPU-hours |
|---|---:|
| Evaluator calibration, end-to-end profiling, accumulation and topology checks | 35 |
| Model-size × training-exposure ladder | 120 |
| Independent corpus-size and mixture/curriculum contrasts | 80 |
| Resolution continuation, high-resolution detail and retention checks | 90 |
| Held-out larger/longer validation and finalist confirmation | 85 |
| Failure, restart, and uncertainty reserve | 40 |
| Total | 450 |

These are planning envelopes, not known runtimes or a promise that all possible
contrasts fit. Charge generation/evaluation, compilation, checkpointing, allocated
idle time, and failed jobs to their package. Keep CPU/API/storage and actual
4060 Ti device-hours in a separate explicit ledger; they are not assumed free
or converted by an informal speed ratio. Full-corpus high-resolution precompute
needs its own approved allocation, with no omission from total project cost.

### 7.1 Sequence and comparison controls

1. **Freeze inputs and the evaluation contract.** Accept Stage-3.5 results, pin
   code/data/length metadata and caption/loss policy, calibrate the reviewer, and
   define confirmation holdouts, confidence procedure, and stopping rules.
2. **Measure real cost.** Refresh Stage-3 end-to-end measurements for candidate
   sizes, actual caption mixtures, aspect ratios, and resolutions. The corrected
   [Stage-3 gate](stage3_gate.md) closed on one GPU; old multi-GPU timing and
   high-resolution DiT-only ceilings do not establish hero throughput.
3. **Run a compact size/exposure ladder.** Include the 485M baseline and a small
   number of structurally compatible sizes within the ≤0.7B scope, with overlapping
   GPU budgets and multiple saved evaluation points. A smaller proxy is allowed
   for Stage-4 scaling, but must demonstrate relevance to larger candidates.
   Initialize independent size/corpus runs from scratch; do not assume an existing
   “real” checkpoint. Compare effective batches through accumulation, not a full
   resweep of per-bucket micro-batches.
4. **Separate data amount from training amount.** Use nested, stratified unique
   corpus subsets at matched actual draws `S` to measure the effect of `U`.
   Fixed optimizer steps alone do not match exposure when bucket batches differ.
   Also report quality at matched GPU-hours, the primary resource comparison.
5. **Test the small mixture/curriculum set in §6.** Reuse a common prefix checkpoint
   for controlled continuation branches when useful; record shared ancestry and
   charge shared-prefix compute once. Branches are not independent training seeds.
   Do not import Stage 3.5's all-from-scratch restriction into a resolution-transfer
   experiment that specifically requires continuation.
6. **Validate the resolution path.** Compare 256p→640p continuation with a limited
   640p-from-scratch control, then test 640p→896p continuation. Measure anatomy
   detail, style/layout retention, cost, and source availability. Test 1024p only
   if a bounded candidate allocation leaves the required 896p result credible.
7. **Confirm and forecast.** Protect held-out larger/longer tests and at least a
   finalist training-seed check where affordable; if seed variance cannot be
   estimated adequately, state the limitation in the launch forecast. Measure
   the selected inference settings on the 4060 Ti, fit conditional capability
   curves, and estimate the complete hero cost with uncertainty.

Prune clearly dominated candidates rather than building a full Cartesian grid.
Do not spend the confirmation reserve on more speculative screens. If early
anatomy results are too weak to extrapolate, favor a genuinely informative longer
or higher-resolution bridge over additional tiny runs.

### 7.2 Bucket and optimizer boundaries

Reuse Stage 3.5's distribution-aware boundary optimization and measured batch-size
screening. Freeze a measured plan per tested model/resolution/distribution regime;
refresh it when model size, high-resolution shape, or mixture materially changes.
Such safety/performance validation is part of Stage 4, not a new bucket research
stage. Compare quality at equal GPU-hours when execution plans differ, and verify
that the plans preserve the intended sampling/loss semantics.

Tune effective batch with `gradient_accumulation_steps`; record actual samples per
update and actual exposures because variable micro-batches make nominal products
misleading. Retain the established optimizer family: chunked Muon plus auxiliary
AdamW, with Muon LR 0.02 as the starting point. Finalize LR/warmup/decay, stage
restart or continuation behavior, EMA timescale, clipping, caption dropout,
resolution time shift, and loss normalization. Reopening architecture/objective
or optimizer-family choices requires a stated reason and scope decision, not an
unbounded sweep hidden inside “complete recipe.”

## 8. Derive the whole hero allocation

For stage `r`, with measured global throughput `v_r` in actual samples/second,
`G_r` GPUs, and planned actual sample draws `S_r`:

\[
H_{\rm train,r}=\frac{S_r}{v_r}\frac{G_r}{3600}.
\]

Sum across 256p, 640p, 896p, and any enabled 1024p stage, then add startup,
evaluation, checkpoint, expected restart, and allocated-idle allowances without
double-counting costs already in measured rates. Queue waiting is wall-clock
delay, not allocated GPU-hours. Record both and the assumptions about preemption.
Measure the chosen multi-GPU topology or use a tested smaller topology; unvalidated
eight-GPU efficiency cannot be the reason the budget appears to fit.

Decide stage allocations from capability gain per cost, high-resolution detail,
retention, and uncertainty, not the roadmap's earlier guessed step percentages.
Compute minimum, recommended, and conservative-cost scenarios within the working
range, with a single selected executable schedule and bounded contingencies.
Report which capability controls the required budget and what extra budget would
buy, when the evidence supports an estimate. If not, give the cost of resolving
uncertainty rather than pretending to know the final required hours.

Predefine transition checks, allowed bounded extensions, and rollback/stop rules
inside the total ceiling. Automatic adaptation is permitted only with explicit
metrics, thresholds, maximum spend, and resulting next configuration. “Tune after
seeing the hero” is not a complete recipe. Unexpected scientific failures during
Stage 5 may still trigger redesign; completeness does not imply infallibility.

## 9. Required Stage-4 handoff

Stage 4 produces `notes/hero_recipe.md` and the executable artifacts it references.
This design document is not that result. The handoff must contain:

| Deliverable | Required content |
|---|---|
| Capability contract and evidence | Frozen bilingual prompt/seed manifests, rubrics, gate table, reviewer calibration, raw scores/images, uncertainty and held-out forecast validation |
| Selected model | Exact config/parameter count, text encoder and exit layer, VAE, initialization, code/dependency revisions; no unresolved size choice |
| Data recipe | Immutable dataset/metadata manifests, unique eligible counts, quality/capability strata, licenses, dedup/eval separation, exact per-stage mixture and caption/loss-weight schedules |
| Full resolution schedule | 256p, 640p, 896p and an explicit enable/disable decision or bounded trigger for 1024p; aspect buckets, sample budgets, update counts, transitions and retention checks |
| Optimization recipe | Bucket plans, accumulation and actual effective batch, optimizer groups, LR/warmup/decay, EMA, time shifts, dropout, clipping and normalization |
| Serving recipe | Same configuration used for capability qualification; measured 4060 Ti memory/latency, solver/steps/guidance/precision/offload, tested prompt and shape envelope |
| Operations | Runnable configs and launch/resume commands, tested checkpoint state restoration, storage/data paths, precompute workflow, evaluation/checkpoint cadence, failure handling and spend limits |
| Cost and decision report | Training and overhead ledger, preparation/API/storage costs, topology and wall-clock assumptions, uncertainty, alternatives rejected, and explicit go/no-go/redesign verdict |

Before a go handoff, required implementation and relevant tests/smokes must be
complete, including mixture schedules, telemetry, evaluation, and all enabled
resolution transitions. Production 896p/optional 1024p precompute may execute
as a specified Stage-5 preparation task, but its data policy, commands, tested
small-scale path, storage, cost allocation, and validation must already be settled.
It cannot conceal another design or feasibility study.

The hero run itself should not need a fresh model-size choice, mixture sweep,
inference optimization project, evaluation definition, or undecided resolution
allocation. Preparation and monitoring are normal execution, not Stage 4.5.

### Exit checklist

- [ ] Stage-3.5 outputs accepted and exact experiment inputs frozen.
- [ ] All required bilingual capability gates and statistical procedures frozen.
- [ ] Held-out validation supports conservative ≥80% hero forecasts for every gate.
- [ ] The same quality-qualified serving settings fit 16 GB and the 20-second ceiling.
- [ ] The complete selected pre-NFT schedule fits the agreed ≤2,200-hour hero budget,
      with overhead, bounded contingencies, and separately approved preparation costs.
- [ ] All recipe decisions, executable configs, operational paths, and relevant
      validation are complete; measured and extrapolated numbers are distinguished.
- [ ] `notes/hero_recipe.md` and supporting artifacts are versioned and reviewed.
- [ ] An explicit go verdict allows Stage 5 to proceed directly; otherwise identify
      the unresolved evidence or the redesign decision requiring user agreement.

An infeasibility report is a useful Stage-4 result, but it is **not** successful
closure for launching the hero. Budget exhaustion, an appealing loss curve, or
optimism about NFT cannot replace these exit conditions.
