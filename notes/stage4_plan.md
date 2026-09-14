# Stage 4 — Complete pre-NFT hero recipe and infrastructure validation

Status: updated 2026-09-14. Model size, training budget, optimizer, caption policy,
mixtures, and resolution split are fixed in [hero_recipe.md](hero_recipe.md).
Remaining work is infrastructure optimization and execution readiness. There is
no numerical capability target, capability forecast gate, or serving-performance gate.

This plan refines [the roadmap](redesign_plan.md). It does not launch experiments,
authorize additional spending, or declare Stage 3.5 complete. Stage 4 execution
uses the accepted outputs of [Stage 3.5](stage3_5_plan.md), including amendments
and measured outcomes in [its execution record](stage3_5_pilot.md).

## 1. Outcome and stage boundary

**Stage 4 must deliver the complete, executable pre-NFT hero-run recipe, so
Stage 5 can proceed directly. There is no planned Stage 4.5.**

The recipe covers training from scratch through **256p → 640p → 896p**, with
**no 1024p stage**, including model size, corpus, stage-dependent data mixtures,
caption policy, effective batch, optimization, resolution transitions, training
duration, evaluation, inference settings, and operational preparation. Stage 5
executes and verifies this recipe; it is not another recipe-discovery phase.
Later DiffusionNFT training belongs to Stage 6 and is excluded.

This is a conditional handoff, not a requirement to produce a positive verdict:

- If the selected recipe is executable, stable in relevant checks, and fits the
  training budget, close Stage 4 with a go recommendation and explicit capability
  uncertainty. Report the trained model's observed abilities rather than certifying
  a predetermined success rate.
- Unresolved correctness or cost feasibility blocks launch. A small run's inability
  to predict eventual anatomy accuracy does not itself require more proxy runs.
- If the design is infeasible, present a redesign decision. For example, a
  measured cost that cannot fit the selected execution plan inside the
  **2,200-hour** hero ceiling requires revising total steps within the recipe or
  an explicit redesign decision. It does not authorize extra spending.

Redesign is an exception prompted by evidence, not unfinished planning.

## 2. Qualitative interests for monitoring

The project prioritizes ordinary useful painting generation, not an exhaustive
human-anatomy or spatial-reasoning benchmark.

| Area | What to inspect | Outside the ordinary-use focus |
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

Inspect anatomy and architecture in both painting families. Judge detail and
perspective according to the requested style: sparse ink need not show realistic
skin, and Chinese painting need not adopt photographic single-point perspective.
Abstraction is not permission for unintended extra limbs or impossible joins.
Photographic prompts may diagnose transfer but cannot substitute for painting
performance.

Layout covers viewer-left/right/center, above/below, beside, in front of/behind,
and foreground/middle-ground/background. An ordinary example is a person on the
left, a pavilion on the right, and mountains in the background. The base model
is monitored for this basic arrangement ability; later NFT may refine its precision.

### 2.1 What NFT may improve

NFT may improve reliability within demonstrated capabilities, finer prompt
adherence, spatial precision, aesthetic preference, and stylistic polish. It is
not a reason to hide limitations of the trained base model. Assess its usefulness
from observed outputs before choosing any later post-training work.

This is a project risk boundary, not a theorem that preference training cannot
improve anatomy. [BACON](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Kas_We_Still_See_Broken_Limbs_Towards_Anatomical_Realism_in_GenAI_WACVW_2026_paper.html)
reports anatomical improvements from preference learning and sample selection;
its results do not guarantee rescue for this model or these painting domains.
[DiffusionNFT](https://arxiv.org/abs/2509.16117) supplies a post-training method,
not an ArtFlow-specific pretraining capability forecast.

## 3. Qualitative monitoring

### 3.1 Fixed comparisons, not capability gates

Use the small bilingual short/long-prompt panel in
[stage4_eval_freeze.md](stage4_eval_freeze.md) and the operational cadence in
[hero_recipe.md](hero_recipe.md). Freeze prompts, seeds, and sampling settings;
retain all designated outputs and reviewer notes. Report Chinese and English
observations separately where useful, without numerical pass/fail thresholds.

Look for recognizable subjects and styles, coherent full-body/face/hand geometry,
everyday object contact, and basic spatial layout. Do not hide a required hand or
crop a requested full body to present an apparent improvement. Interpret natural
occlusion and abstraction according to the prompt and style.

### 3.2 VLM-first review

The execution agent reviews full images and relevant crops and surfaces uncertain
or systematic failures to the user. Record reasons and uncertainty, and do not
equate VLM confidence with correctness. No large human-labeling or judge-calibration
quota is required for this monitoring workflow.

Flow loss, KID, and image panels diagnose learning and implementation regressions.
None is a forecast or a certification of final capability. Compare losses within
a resolution. Pause/review and nonfinite/corruption stops follow the hero recipe;
they are safety checks, not a disguised quality threshold. API-billed review
requires its own approved allocation.

## 4. Reproducible inference for evaluation

Stage 4 records solver, sampling steps and actual model evaluations, guidance,
EMA checkpoint policy, precision, attention/compile settings, and any CPU offload
used for its evaluation panels. Keep those settings fixed within a comparison;
a quality result does not qualify a different inference configuration without
re-evaluation. Sampling correctness and consistency between training, evaluation,
and inference remain readiness checks.

There is no prescribed inference device, VRAM ceiling, or latency gate for the
hero run. Training quality and the training budget determine the recipe.
Deployment optimization, quantization, and distillation are optional later work,
not prerequisites for Stage-4 closure or the hero launch.

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

### 5.1 Capability uncertainty and monitoring

Measure category-specific success trajectories, not just a loss-to-anatomy
conversion. Research on [direct downstream-metric scaling](https://arxiv.org/abs/2512.08894)
supports investigating such fits, while [work on prediction failures](https://arxiv.org/abs/2406.04391)
shows why predictable loss need not yield equally predictable accuracy. Both
concern language models, not an established scaling law for painted anatomy.

Do not invert a poorly identified small-run accuracy curve into a claimed budget
for correct anatomy. Small runs support stability/cost checks and qualitative
comparison, not a reliable guarantee of the hero's final success rate.

Separate these diagnoses:

- **Compute-responsive:** observed capability improves with longer/larger runs;
  estimate a budget interval conditional on the tested recipe.
- **Recipe-limited:** evidence points to data coverage, quality, resolution, or
  conditioning shortcomings; more hours alone are not the justified remedy.
- **Unresolved:** experiments cannot distinguish the two; identify a bridging
  experiment and its cost.

Use the fixed panel to check resolution continuation and monitor hero checkpoints:
256p figures alone do not establish 896p hand/face detail. No dedicated per-atom
scaling fit or larger/longer capability-confirmation run is required for Stage-4
closure. There is no final numerical capability target. Report observed limitations
and follow the recipe's pause/review rules if the hero develops clear regressions.

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

The per-stage row multipliers, normalized mixtures, and eligible pools are fixed
in [hero_recipe.md](hero_recipe.md). Verify actual exposure and per-source
repetition with the final plans; do not reopen corpus-size or mixture sweeps as
routine launch prerequisites. A source's size alone does not establish its quality
or usefulness. A material mixture change requires an explicit recipe revision.

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

Set the infra-pass allowance from the **unspent** portion of the 450-hour cap;
the remaining amount requires the actual job ledger, not subtraction of estimated
package runtimes. Do not refill descoped experiment packages with new sweeps.
Charge generation/evaluation, compilation, checkpointing, allocated idle time,
and failed jobs to the ledger. Keep CPU/API/storage and actual
4060 Ti device-hours in a separate explicit ledger; they are not assumed free
or converted by an informal speed ratio. Full-corpus high-resolution precompute
needs its own approved allocation, with no omission from total project cost.

### 7.1 Sequence and comparison controls

1. **Freeze the selected inputs.** Pin code/data/length metadata, configurations,
   bucket plans, caption/loss policy, and the small bilingual monitoring panel.
2. **Validate continuation controls.** The local implementation supports exact
   stage stopping, pre-evaluation endpoint checkpoints, predecessor/T validation,
   and full-state resume. Smoke-test these paths on the distributed stack. Keep
   the shared LR/caption horizon; initialize caption progress before prefetch.
3. **Run the bounded infrastructure pass (§7.3).** Measure the selected 533M model
   on eight ranks at all three resolutions, including actual caption mixtures,
   aspect ratios, memory, exposure, and startup/steady-state costs. The corrected
   [Stage-3 gate](stage3_gate.md) closed on one GPU; neither old multi-GPU timing
   nor DiT-only ceilings establish hero throughput.
4. **Cost and hand off.** Derive final total steps, 75:20:5 stage endpoints, sample
   budgets, and bounded overhead/recovery allowances. Verify cross-resolution
   continuation and the monitoring workflow. No new architecture, corpus, mixture,
   or per-atom capability-forecast sweep is required.

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
AdamW, with the fixed LR/warmup/decay, continuous stage schedules, EMA, clipping,
caption dropout, time shift, and loss normalization in the hero recipe. Reopening architecture/objective
or optimizer-family choices requires a stated reason and scope decision, not an
unbounded sweep hidden inside “complete recipe.”

### 7.3 Final hero infrastructure pass (agreed 2026-09-13)

This is a final Stage-4 exit task, **not a separate Stage 4.5 or an open-ended
optimization stage**. Its input is the selected scientific recipe; its output
is the final executable, costed hero recipe that Stage 5 can launch directly.
Before profiling, set a fixed GPU-hour allowance from the remaining profiling
envelope or reserve **inside the 450-hour cap**, and record it in the ledger.
Stop at that allowance or when plausible remaining hero-run savings no longer
justify further profiling. It does not authorize additional spending or a new
architecture/optimizer search.

The user reports roughly 3.1× four-GPU speedup and about 60% “GPU Time Spent
Accessing Memory” in SwanLab. Treat these as diagnostic leads, not established
bottlenecks: 3.1× implies 77.5% scaling efficiency only for comparable workloads
and steady-state measurement boundaries, and memory-access time alone does not
establish bandwidth saturation. Confirm the metric definition and use traces
to distinguish exposed communication, rank imbalance, synchronization, input
stalls, and memory-heavy compute.

The bounded pass must:

- Benchmark the chosen model at every enabled hero resolution, on the intended
  GPU count/topology, with representative caption lengths, mixtures, and aspect
  ratios. Separate compilation/startup from steady state and report global
  samples/second, samples/update, GPU-hours, and wall-clock time.
- Profile a small number of representative steady-state steps, target the
  measured bottlenecks with low-risk changes, and verify gains end to end.
  Re-screen micro-batch sizes and, where justified, refresh bucket boundaries
  using the existing Stage-3.5 pipeline; no new bucket-planning research is needed.
- Preserve training semantics: check sampling/exposure and loss normalization
  after bucket changes, adjust accumulation where needed, and verify actual
  effective batches. If a change materially alters effective batch, optimizer
  cadence, or caption curriculum, treat it as a recipe revision needing bounded
  validation, not as a transparent speed optimization.
- Recompute the cost of the selected **sample exposure first**, including
  checkpoint/evaluation, startup, and restart allowances. Derive optimizer-step
  counts from the final execution plan; fixed step counts do not preserve dose
  when bucket batches change. Any budget-driven change to training dose or
  resolution allocation must be explicit in the revised recipe, not a silent
  shortening of training.

Pin the final code/config revisions, bucket plans, accumulation, and topology;
retain before/after measurements and relevant correctness/resume smokes. Refresh
the cost scenarios in §8 from these measurements, distinguishing measured rates
from extrapolation. There is no required 3.5× or 4× scaling gate: the goal is a
reliable, affordable hero run, not an idealized speedup number.

## 8. Derive the whole hero allocation

For stage `r`, with measured global throughput `v_r` in actual samples/second,
`G_r` GPUs, and planned actual sample draws `S_r`:

\[
H_{\rm train,r}=\frac{S_r}{v_r}\frac{G_r}{3600}.
\]

Sum across 256p, 640p, and 896p, then add startup,
evaluation, checkpoint, expected restart, and allocated-idle allowances without
double-counting costs already in measured rates. Queue waiting is wall-clock
delay, not allocated GPU-hours. Record both and the assumptions about preemption.
Measure the chosen multi-GPU topology or use a tested smaller topology; unvalidated
eight-GPU efficiency cannot be the reason the budget appears to fit.

The selected stage split is 75:20:5 of total optimizer steps. Derive the total
step count from measured end-to-end rates and overhead inside the 2,200-hour cap;
400k is a benchmark target, not a ceiling. Record realized sample exposure and
bounded contingencies with the executable schedule in [hero_recipe.md](hero_recipe.md).
Report how infra changes affect measured training cost and feasible steps/exposure;
do not translate the extra steps into an unsupported capability prediction.

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
| Qualitative monitoring | Fixed bilingual prompts/seeds and sampling settings, review notes/images, regression rules and uncertainty; no numerical capability gate |
| Selected model | Exact config/parameter count, text encoder and exit layer, VAE, initialization, code/dependency revisions; no unresolved size choice |
| Data recipe | Immutable dataset/metadata manifests, unique eligible counts, quality/capability strata, licenses, dedup/eval separation, exact per-stage mixture and caption/loss-weight schedules |
| Full resolution schedule | 256p, 640p, 896p with a 75:20:5 optimizer-step split; no 1024p; aspect buckets, measured sample budgets, final update counts, transitions and retention checks |
| Optimization recipe | Bucket plans, accumulation and actual effective batch, optimizer groups, LR/warmup/decay, EMA, time shifts, dropout, clipping and normalization |
| Final infrastructure validation | §7.3 profiling allowance and actual spend, representative traces, before/after end-to-end rates per enabled resolution/topology, pinned execution settings, and sampling/loss-normalization validation |
| Evaluation inference recipe | Reproducible solver/steps/guidance/precision/offload settings and tested prompt/shape envelope; no deployment hardware or latency gate |
| Operations | Runnable configs and launch/resume commands, tested checkpoint state restoration, storage/data paths, precompute workflow, evaluation/checkpoint cadence, failure handling and spend limits |
| Cost and decision report | Training and overhead ledger, preparation/API/storage costs, topology and wall-clock assumptions, uncertainty, alternatives rejected, and explicit go/no-go/redesign verdict |

Before a go handoff, required implementation and relevant tests/smokes must be
complete, including mixture schedules, telemetry, evaluation, and all enabled
resolution transitions. Any remaining production 896p preparation may execute
as a specified Stage-5 preparation task, but its data policy, commands, tested
small-scale path, storage, cost allocation, and validation must already be settled.
It cannot conceal another design or feasibility study.

The hero run itself should not need a fresh model-size choice, mixture sweep,
inference optimization project, evaluation definition, or undecided resolution
allocation. Preparation and monitoring are normal execution, not Stage 4.5.

### Exit checklist

- [ ] Stage-3.5 outputs accepted and exact experiment inputs frozen.
- [x] The 48-image bilingual monitoring panel, qualitative review rubric, shared
      seeds and operational checks are frozen in `stage4_eval_freeze.md`.
- [ ] Qualitative observations and limitations are recorded without a capability pass/fail target.
- [ ] Evaluation sampling settings are recorded and the sampling path passes correctness checks.
- [ ] The complete selected pre-NFT schedule fits the agreed ≤2,200-hour hero budget,
      with overhead, bounded contingencies, and separately approved preparation costs.
- [ ] The bounded final infrastructure pass is complete; its measured rates,
      final bucket/accumulation settings, and explicit exposure/update-count
      revisions are incorporated into the hero recipe and wall-clock estimates.
- [ ] All recipe decisions, executable configs, operational paths, and relevant
      validation are complete; measured and extrapolated numbers are distinguished.
- [ ] `notes/hero_recipe.md` and supporting artifacts are versioned and reviewed.
- [ ] An explicit go verdict allows Stage 5 to proceed directly; otherwise identify
      the unresolved evidence or the redesign decision requiring user agreement.

An infeasibility report is a useful Stage-4 result, but it is **not** successful
closure for launching the hero. Budget exhaustion, an appealing loss curve, or
optimism about NFT cannot replace these exit conditions.
