# Stage 3.5 — Caption enrichment, selection policy, and bucket planning

Status: design agreed on 2026-09-09. Caption production, review, selection and
publication were executed and are recorded in [the pilot record](stage3_5_pilot.md);
the caption-policy training comparison (§5, §6) and the 640p precompute (§9) have
not been run. Sections below are kept as written at design time, so quantities in
§2 and §3.4 are forecasts, not results.

This document specifies the design for Stage 3.5 of [the redesign plan](redesign_plan.md).
It does not authorize launching jobs, spending API credit, publishing a dataset,
or entering Stage 4. Numeric settings explicitly marked provisional must be
finalized from the pilot or recorded before training results are inspected.

This interview refines the roadmap's earlier comparison protocol: equal GPU-hours
is the primary caption-policy comparison; matched actual samples are secondary.
The budgets below replace its previously unspecified experimental/API estimates,
but do not assign a budget to full-dataset precompute.

## 1. Purpose and settled decisions

The objective is to improve useful long-prompt conditioning while making its
training cost manageable. Longer text is not itself evidence of better data or
a better model.

- Allocate caption-enrichment effort and production API spending **70% to broad
  long-prompt competence and 30% to richer Chinese-art understanding**.
- Broad competence includes both natural descriptions and descriptions containing
  many distinct visual requirements. Prioritize 256–1024 retained tokens;
  1024–2048 is a supported but uncommon tail with dedicated tests.
- Enrich existing rows, mostly with one new caption per selected row. Preserve
  existing captions and the dataset/row sampling marginals; do not duplicate rows
  to increase long-caption exposure.
- Within broad enrichment, target Chinese/English at **50:50**; within specialized
  Chinese-art enrichment, target **80:20**. Do not require paired translations on
  every row.
- Cover natural prose and structured descriptions. The provisional pilot split
  is **50:50**, distributed across domains, languages, and lengths.
- Captions contain visible content and verified metadata, not speculative
  art-historical essays. Richness means grounded specificity.
- Use VLM-first review by the execution agent. Escalate consequential uncertainty
  to the user rather than scheduling a large manual-review workload.
- All training runs, including screening, start **from scratch** on the current
  **1152-wide, 24-layer** architecture. Small scale means short runs, not a smaller
  model or continuation from an existing checkpoint.
- Compare caption policies by quality at **equal GPU-hours**, not samples/s or
  equal optimizer-step counts. Prompt adherence is primary; visual quality and
  short-prompt performance are guardrails.
- A small short-prompt regression is acceptable for a clear long-prompt gain.
  Its allowable size must be specified before training results are available.
- Include a small number of caption-selection and curriculum comparisons, not
  a large sweep of curriculum schedules or legacy constants.
- For bucket planning, hold the input sample distribution fixed and minimize
  execution cost. A faster plan must not achieve its speed by avoiding long text.

The general usefulness of detailed captions is a working prior, not the main
hypothesis to re-establish here. The unresolved experimental question is which
selection policy and schedule make the enriched data useful within our budget.

## 2. Budgets and accounting

### API budget: approximately $100

| Purpose | Provisional allocation |
|---|---:|
| Captioner, prompting, and reviewer-calibration pilot | $5 |
| Production caption generation, including its repair/escalation costs | $75 |
| Evaluation captions and API-billed quality checks | $10 |
| Contingency for uncertain costs and overruns | $10 |
| Total | $100 |

The production allocation is initially **$52.50 broad / $22.50 Chinese-art
specialization**. This is not a rigid cheap-model/expensive-model split.
Model routing follows difficulty and review results.

Execution-agent review is not assumed free or unlimited. Record whether its
inference is charged against this API budget or supplied separately by the agent
environment. API-billed review must be included in the ledger; separately supplied
review still needs its usage/capacity recorded. Resolve this before production.

Every forecast must include failed paid attempts and repair/escalation costs.
Do not count an accepted caption as costing only its final successful call.
Reserve capacity for in-flight requests and delayed billing when enforcing the
budget. Do not launch an unbounded retry or repair loop.

### GPU budget: 75 RTX 4090 GPU-hours

This cap is **for experiments and profiling only**. It excludes full-dataset VAE
precompute, including 640p precompute, which requires a separate estimate and
budget decision. The 640p item below is a training/profiling spot check, not a
full-dataset encoding allocation.

| Work | Provisional GPU-hours |
|---|---:|
| Bucket cost profiling and candidate validation | 8 |
| Five short training screens, approximately 4 hours each | 20 |
| Two finalists × two seeds × approximately 8 hours each | 32 |
| 640p training performance and memory checks | 5 |
| Contingency | 10 |
| Total | 75 |

These are spending ceilings, not measured runtime predictions. Four GPUs running
for one hour consume four GPU-hours. Count allocated GPU time spent on startup,
compilation, evaluation, checkpointing, and failed jobs as well as training.
Log training-only time separately. Do not spend confirmation automatically if
screening or data quality makes it uninformative. Report inconclusive results as
inconclusive instead of expanding the budget silently.

## 3. Caption production design

### 3.1 Model routing and reasoning effort

| Role | User-selected candidate |
|---|---|
| Cheap, broad generation | Gemini 3.5 Flash-Lite |
| Medium difficulty and repairs | Gemini 3.8 Flash or GLM 5.3 Flash |
| Hardest Chinese-art cases | Kimi K3 |
| Image-grounded reviewer | Execution agent |

These are the intended model choices, not a claim that all provider identifiers,
prices, modalities, or request parameters have been verified. Verify them before
the pilot. Compare the middle-tier candidates on the same images. Hard-case
escalation should follow demonstrated difficulty, not token length alone.

Use the **lowest supported reasoning effort** for ordinary captioning; disable
reasoning when the model/provider supports it. Parameter names and the meaning of
"low" differ, so record the actual request settings and billed reasoning usage.
Higher effort is an escalation option only if the pilot shows a useful gain in
acceptance or grounding per dollar. Do not run a broad effort-level sweep.

### 3.2 Which rows receive captions

The broad portion spans the existing training mix, including Chinese painting.
The specialized portion additionally targets Chinese-art detail. For example,
with a 15% Chinese-painting weight in the broad mix, proportional broad allocation
would produce 70% × 15% + 30% = **40.5%** Chinese-painting production spending.
This is an illustration, not a fixed row-count quota or a new training mix.

Select rows using:

1. **Coverage deficits:** prioritize rows lacking an already-good caption in the
   desired length range; do not regenerate useful existing text merely to meet a
   production count.
2. **Training exposure:** account for dataset sampling weight and row count, not
   only corpus size.
3. **Diversity:** stratify by source, subject, composition, aspect ratio, language,
   and existing caption coverage. Avoid repeated spending on near-duplicate views.
4. **Visual information:** reserve the longest descriptions for images that
   support them. Include compositional relationships and interactions, not only
   object inventories or artistic adjectives.
5. **Training-view agreement:** inspect the artwork crop actually used for training.
   Do not describe details outside that crop. Record the image/crop identity.

Use stable row identities and preserve train/eval separation at the artwork or
near-duplicate-group level, not merely the file level.

Report training-weighted availability at threshold L:

```text
C(>=L) = sum over datasets d of:
         normalized_mix_weight[d]
         * fraction_of_eligible_rows[d]_with_an_accepted_caption_at_least_L
```

This is an upper bound on the fraction of row draws that can supply such text,
before caption dropout, ignoring finite queue effects. Stronger within-row bias
cannot manufacture coverage on rows without long captions. For illustration,
30K enriched rows out of 1.5M uniformly sampled rows cover only 2% of draws;
actual mixture weights change that arithmetic. Report incremental coverage from
new captions separately from coverage already present.

### 3.3 Lengths, languages, and caption structure

Measure lengths using the training Qwen tokenizer and exact prompt-retention
contract: template, dropped prefix, and 2048-token retained cap. These are not
API billing tokens, word counts, or heuristic language-dependent estimates.
Also inspect untruncated lengths so truncation is not mistaken for compliance.

| Retained tokens | Provisional share of accepted additions | Illustration at 30K additions |
|---|---:|---:|
| 256–511 | 50% | 15,000 |
| 512–1023 | 30% | 9,000 |
| 1024–1535 | 15% | 4,500 |
| 1536–2048 | 5% | 1,500 |

These are soft production targets, not desired training exposure or bucket
boundaries. The pilot may revise them. **30K is only an earlier planning
illustration, not a promised or locked production count.**

Natural captions should read as coherent descriptions. Structured captions should
group grounded facts about subjects, attributes, placement, relationships, and
rendering style; they are not imagined image-editing commands. Cross both formats
with languages and lengths so format is not confounded with length or domain.

Allow precise visual terminology for brushwork, ink texture, color, composition,
and spatial relationships. Artist, title, or period may be included when supplied
by trusted metadata, not guessed from style. Exclude invented provenance,
biography, symbolism, or artistic intent. Keep uncertainty in review records,
not padded training prose. Do not inflate lengths with repetition or pasted OCR.

If an image cannot support the requested detail, accept an appropriate shorter
description or reassign the quota. Never invent content to fill a length band.

### 3.4 Pilot and production-count forecast

Start with a stratified pilot; **200–400 unique images** is an initial proposal,
subject to the $5 cap and the cost of paired model calls. Include both languages,
both formats, the full length range, ordinary broad examples, and difficult
Chinese-art cases. It is not necessary to send every image to every model.

Review every pilot output. Measure, by model/provider, difficulty, language,
format, length target, and reasoning setting:

- Billed image/text input, output, and reasoning usage; total actual charges.
- Grounding acceptance and achieved retained length.
- Useful additional detail, repetition, unsupported claims, and length failures.
- Retry, repair, and escalation rates and costs.
- Cost per accepted caption and per unique enriched image.
- Latency, rate limits, and sustainable concurrency.

For production stratum s, estimate:

```text
c[s] = all generation / failed-attempt / repair / escalation charges in s
       divided by accepted captions in s

sum_s planned_accepted_count[s] * c[s] <= remaining_production_budget
```

If the routing mixture is fixed, a coarse total is remaining production budget
divided by weighted cost per accepted caption. Use conservative, central, and
optimistic forecasts; uncertainty in acceptance and escalation matters as much
as advertised token prices. Do not equate Qwen retained tokens with billed output.

Before production, revise the row count, length counts, middle-tier choice, and
routing rules using these measurements. Keep the overall budget and 70:30
objective allocation fixed unless the user changes them. Report the resulting
training-weighted coverage, not only how many API requests can be purchased.

### 3.5 Review and provenance

The execution agent reviews the actual image alongside the caption. Where
practical, hide generator identity and randomize comparison order. Identify
specific claims as supported, contradicted, or uncertain; avoid deciding from
fluency or self-reported confidence alone.

- **Pilot:** image-grounded review of every output.
- **Production:** structural checks on every output, a stratified random
  image-grounded audit, and review of all flagged cases.
- **High risk:** review every 1536–2048-token caption and every difficult
  Chinese-art escalation.
- **Systematic failure:** expand the audit or pause the affected stratum.
- **Human escalation:** present the disputed claim, relevant image/region, and a
  focused question for consequential uncertainty or reviewer disagreement.

Set the production random-audit rate after measuring reviewer capacity and pilot
error rates. Maintain random review even if confidence flags look reliable.
Test the reviewer using deliberately incorrect attributes, counts, and spatial
relationships. A reviewer unable to detect these controls cannot be the sole
acceptance mechanism. Preserve uncertain cases rather than forcing a verdict.

Cache raw responses and record stable row ID, image/crop fingerprint, model and
provider/version when available, prompt version, generation settings, language,
format, token lengths, billed cost, review outcome, and repairs. Cache identity
must include generation inputs/settings so a changed prompt cannot silently
reuse an old response. No credentials belong in artifacts.

## 4. Caption selection and curriculum

### 4.1 Preserve row semantics and sample-normalized loss

The current sampler chooses dataset, then row, then one caption within that row.
Keep this ordering. Length queues remain an execution mechanism, not a source of
new sampling weights. Existing sample-count loss normalization stays unchanged:
variable-size micro-batches must not receive equal total weight regardless of
their sample count.

The proposed bias changes caption selection, not long-bucket loss weights. Do
not importance-correct away the intended change in caption distribution.

### 4.2 Reference and simplified replacement

The reference selector in `src/dataset/captions.py` uses heuristic lengths,
preference strength 2.0, minimum/maximum probabilities 0.15/0.80, and a 1e-6
score floor. Clipping is followed by renormalization, so those bounds need not
hold for the final probabilities. Keep this reference unchanged in its arm.

Instead of sweeping all these constants or stacking another weighting layer on
top, test an interpretable replacement. For caption c in selected row r:

```text
q_beta(c | r, u) = L[c]^beta(u) / sum_j L[j]^beta(u)
```

L is the exact retained length, at least one under the existing contract; u is
normalized training progress. Implement stably as a softmax over beta × log(L).
Before a protective reserve, beta=0 is uniform, negative beta prefers shorter
captions, and positive beta prefers longer ones. At beta=1, twice the retained
length gives twice the selection probability.

When the row has short captions, define S as captions with **L < 256** and use:

```text
p(c | r, u) = epsilon * Uniform(S)[c] + (1 - epsilon) * q_beta(c | r, u)
```

If S is empty, use q_beta alone. This reserves probability for the short-caption
group, not for every individual caption. It guarantees at least epsilon short
exposure within eligible rows; the remaining component can also choose short
captions. Rows with only short captions still produce only short captions.

**epsilon=0.20 is the proposed initial fixed setting**, not another sweep axis.
Unlike a per-caption minimum, this reserve is not diluted by adding more long
variants. Log actual exposure: the reserve does not imply a global 20:80 split.

The reference-to-replacement comparison changes both the length measurement and
the probability formula/protection. Treat it as a policy-package comparison, not
as isolated causal evidence about each component.

### 4.3 Small curriculum comparison

Separate the selector from its schedule. A provisional simple baseline is a
linear beta ramp from -1 to +1; an upward shift such as +0.5 is a candidate added
long bias. These numeric endpoints are starting proposals to inspect against
actual metadata and freeze before GPU screening, not already measured winners.

Compare the selected replacement policy under:

1. **Linear short-to-long:** the reference schedule for that policy.
2. **Stationary, exposure-matched:** use each row's average caption probabilities
   under the reference schedule throughout training, including the reserve.
   Average probabilities, not beta, because the mapping is nonlinear.
3. **Earlier transition:** reach the final preference halfway through the run,
   then hold it. This deliberately changes both ordering and total exposure.

The stationary control aims to isolate ordering. Its average must be weighted by
the expected **sample-draw exposure** over progress, not blindly by optimizer
steps or wall time: throughput and samples/update can vary with length. Use
metadata simulation and timing estimates, then validate emitted sample exposure.
Finite queues, randomness, and a fixed compute cutoff prevent assuming exact
matching. If realized exposure differs materially, qualify the ordering claim.

Before launch, pin the progress clock and horizon. The current trainer uses
optimizer-step progress; equal GPU-time experiments do not automatically align
that clock. Record the schedule/LR horizon calibration and use a consistent rule
across arms. Do not change the clock or endpoints after inspecting quality.

## 5. Training experiment matrix

All arms use the same versioned **enriched** dataset, row-level mixture, current
architecture, frozen text encoder, and evaluation version. We are not reserving
a full training arm to compare original versus enriched data. Consequently this
matrix does not estimate enrichment's standalone causal effect.

| Screen | Selection | Schedule | Purpose |
|---|---|---|---|
| A | Existing selector | Existing linear short-to-long mapping | Reference recipe |
| B | Simplified selector + short reserve | Linear beta ramp | Replacement policy |
| C | Simplified selector + short reserve | Upward-shifted linear ramp | Added long preference |
| D | Selected B/C policy | Stationary, exposure-matched | Value of ordering |
| E | Selected B/C policy | Earlier transition, then hold | Longer exposure to final preference |

Run A/B/C first; D/E branch from the selected B/C candidate. This is a sequential
screen, not a full factorial design or a claim to find a global optimum. Stop
unpromising branches rather than spending their allocation automatically.

Confirmation compares the existing reference policy against the selected
challenger, from fresh initialization with **two paired seeds**. Each pair uses
identical initial weights. Record row/caption/dropout/noise RNG handling; separate
streams are preferable so changing caption selection need not accidentally alter
the dataset draw sequence. GPU nondeterminism remains a limitation.

Use equal GPU-time allowances and quality curves against cumulative GPU-hours.
Report allocated end-to-end cost and training-only time separately, with identical
evaluation conventions. Include matched-actual-sample comparisons as a secondary
view where curves overlap; equal steps are not equal data or equal compute.

Bucket plans and accumulation also influence actual samples/update. Start with a
common memory-safe plan for screening where practical. Do not silently give only
the challenger a better execution plan. If per-policy plans are optimized, give
both finalists the same profiling procedure and label the result a recipe-level
comparison. Record effective sample batch distributions and any accumulation
calibration; do not turn this into Stage 4's batch-size sweep or claim a purely
caption-only effect when update batch sizes differ substantially.

## 6. Evaluation and acceptance

### 6.1 What counts as improvement

Prompt adherence is primary. Keep separate broad-competence and Chinese-art
scores; a **70:30** aggregate may summarize them if scales are comparable, but
must not hide either result. Visual quality is a separate guardrail, not a term
that can be arbitrarily traded against adherence in a blended score.

Inspect subjects, attributes, counts, spatial relationships, composition, medium,
and requested style. Separately assess coherence, anatomy where relevant,
rendering artifacts, and artistic quality. Attractive but noncompliant images
and compliant but visibly degraded images should both be identifiable.

Permit a small short-prompt regression only with a clear long-prompt benefit.
Before training, record the short-prompt tolerance, visual-quality tolerance,
minimum useful adherence gain, uncertainty method, and decision rule. Calibrate
the rubric and its noise in the review pilot; measurement noise alone must not
dictate what degradation is substantively acceptable. Do not equate a
non-significant difference with proof of non-regression.

### 6.2 Fixed evaluation suites

- Preserve the old short-prompt regression suite and its data version.
- Add broad and Chinese-art held-out suites covering both languages and formats.
- Give 256–1024-token performance primary emphasis, with dedicated but lower-weight
  1024–2048 tests. Final suite counts/weights are fixed before training.
- Include paired short/detailed descriptions of the same held-out images and
  prose/structured variants expressing the same facts.
- Place selected requirements early and late, and move their positions in paired
  tests. A model that ignores later paragraphs should not appear competent merely
  because the opening subject is correct.
- Evaluate raw short prompts. Deployment prompt polishing is a potential later
  mitigation, not an excuse to conceal regression; any polished-prompt track is
  separate and identical across compared models.

Use fixed generation settings and seeds across arms. The reviewer extracts
checkable requirements from the prompt, inspects the generated image, and records
per-requirement outcomes. Hide arm identity and randomize presentation. Escalate
uncertainty and audit reviewer behavior with known-error controls. Repeated
judgments by the same model are not independent training replications.

Distinguish details visible in the source crop from details still resolvable at
256p or 640p. Evaluate at the stated output resolution and do not generalize a
256p failure on tiny details into a conclusion about high-resolution conditioning.
The 640p performance checks alone are not a 640p caption-quality experiment.

Flow-matching eval loss is supporting evidence, stratified by caption length,
domain, language, and timestep where sample counts permit. Use fixed image,
caption, noise, and timestep assignments and aggregate by actual sample count.
The current `src/evaluation/eval_loss.py` chooses a fixed caption position and
averages batch-level losses; it must not be assumed to automatically evaluate
newly appended captions or supply the required sample-weighted strata.

Report uncertainty at the appropriate unit: shared artwork/prompt variants are
correlated, and training-seed variability is distinct from judge variability.
Two seeds support a modest confirmation, not a strong claim about long-run hero
training. If neither challenger benefit nor guardrail compliance is established,
retain the reference or record an inconclusive result.

## 7. Caption-length telemetry

The current `train/txt_seq_len` is the last micro-batch's padded text width on the
logging rank, not the distribution of captions actually consumed. Add:

| Category | Measurements |
|---|---|
| Selected text | Count, mean, retained-length histogram, histogram-derived p50/p90/p99 |
| Exposure | Shares below 256 and in 256–511, 512–1023, 1024–1535, 1536–2048 |
| Actual conditioning | Caption-dropout rate; long-caption exposure after dropout, with denominators explicit |
| Execution | Padded versus retained tokens, padding fraction, samples and micro-batches per bucket |
| Policy | Progress clock/value, curriculum position, beta or legacy strength, short reserve |
| Coverage | Original versus added captions; domain/language/format breakdowns when metadata supports them |
| Repetition | Unique enriched rows seen and repeat exposure, using bounded/periodic accounting if necessary |

Log both windowed and cumulative exposure. Aggregate counts/sums across ranks and
weight by samples, not micro-batches or averages of rank means. Derive global
percentiles from the merged histogram, not averaged rank percentiles. Separate
selected-text lengths from post-dropout conditioning and padded execution shape.

Use existing metadata and compact counters; do not tokenize again or add a GPU
synchronization per micro-batch. Reduce at the existing logging cadence, including
single-dataset runs. A post-dropout metric should reuse the actual dropout draw,
not consume a second RNG sample. Measure instrumentation overhead in profiling.

## 8. Reusable bucket planner

### 8.1 Contract and objective

The planner accepts model configuration, hardware/memory budget, resolution/aspect
shapes, software/precision/attention/compile settings, dataset length metadata,
row weights, caption policy, desired bucket count K, and expected run duration.
Output per-resolution length boundaries and micro-batch sizes with measured
evidence and limitations. K is initially an input, not another large GPU sweep.

The optimizer may change execution grouping, **not** the selected row/caption
distribution, loss weights, token cap, or inclusion of rare lengths. Different
caption policies need quality-per-compute comparisons; different plans processing
the same stream can correctly be compared by samples/s.

The agreed design is **CPU boundary optimization → GPU batch-size screening →
optional timing-informed boundary refinement → end-to-end validation**. The first
stage can find an exact optimum for a specified padding-compute model; it does
not establish globally optimal hardware throughput. Stage 4 reuses the procedure
for other model sizes and tunes effective batch through accumulation. This design
remains within the existing profiling allocation, not a new compute budget.

### 8.2 Input distribution and padding-cost model

Compute expected sample-weighted retained-length distributions from row metadata,
mixture weights, and exact caption probabilities across early/middle/late
curriculum. Do not use a flat histogram of stored captions or assume the shortest
and longest caption are deterministic curriculum endpoints. Condition on
resolution/aspect shape and account for caption dropout in execution estimates.
For a single plan spanning a changing curriculum, specify how its phases are
weighted by expected sample exposure and report performance in each phase too.

For one fixed model and image shape, define:

- p(l): probability of sampling retained caption length l, before padding.
- L_cap: configured maximum retained length, taken from the prompt contract.
- h_0 = 0 < h_1 < ... < h_K = L_cap: ordered bucket upper bounds.
- C(l): estimated per-sample compute with text length l for this image shape/model.

For integer lengths, bucket i contains **h_(i-1) + 1 through h_i**. Its lower bound
is implied by the previous upper bound; l_min and l_max are not independent
variables. This matches the sampler's first-fitting-upper-bound contract and
prevents gaps or overlaps. Cover the configured cap even if the empirical tail
has zero observed mass; do not reduce the token cap as an optimization shortcut.

The expected padding-compute waste is:

```text
W(h_1, ..., h_K) = sum_i sum_{l=h_(i-1)+1}^{h_i}
                  p(l) * [C(h_i) - C(l)]
```

The unpadded term sum_l p(l) * C(l) is independent of the boundaries. Therefore
minimizing W is equivalent to minimizing expected padded compute under this model.

A useful initial DiT work proxy is:

```text
C(l) = alpha * (I + l) + gamma * (I + l)^2
```

I is image-token count; nonnegative architecture-dependent coefficients represent
linear projection/MLP work and quadratic attention work. Estimate and record their
relative scale from the actual architecture or a small calibration, not universal
constants. Overall positive scaling does not change the optimal boundaries, but
the linear/quadratic balance can. This is a compute proxy, not an exact latency
model for the frozen encoder, kernels, optimizer, or data path.

This objective already balances distribution and raw lengths: a wide interval
with little probability mass may reasonably share a bucket, while a dense region
or an expensive increase in sequence length can justify narrower intervals.

### 8.3 Exact CPU boundary optimization, not a general closed form

For an arbitrary empirical distribution, do not assume a closed-form expression
for the optimal boundaries. Solve the ordered interval partition exactly for the
chosen additive cost using dynamic programming. The general interval-clustering
method is described by [Nielsen and Nock, Optimal interval clustering](https://arxiv.org/abs/1403.2485);
the padding objective below is our application of that method.

Define the cost of placing lengths a+1 through b in one bucket:

```text
A(a, b) = sum_{l=a+1}^{b} p(l) * [C(b) - C(l)]

F(b) = sum_{l=1}^{b} p(l)
G(b) = sum_{l=1}^{b} p(l) * C(l)

A(a, b) = C(b) * [F(b) - F(a)] - [G(b) - G(a)]
```

Prefix sums F and G make every interval-cost query constant-time. The recurrence
for the minimum cost covering lengths 1 through b with k buckets is:

```text
D(k, b) = min_{a < b} [D(k-1, a) + A(a, b)]
D(0, 0) = 0
D(0, b > 0) = infinity
```

Use only feasible predecessor states, retain argmin pointers, and backtrack from
D(K, L_cap). With M candidate upper bounds, the straightforward complexity is
O(K * M^2), independent of the number of dataset rows after histogram construction.
The bounded retained-token range permits considering every integer boundary on
the CPU; first measure this solver rather than assuming a coarse grid is needed.

If boundary alignment or CPU cost warrants a restricted candidate set, use an
explicit grid, weighted quantiles, and/or measured transition points, always
including the cap. Label the result exact **within that candidate set**, not over
all integer boundaries. Use deterministic tie-breaking in zero-mass regions.

There is a useful approximate analytic interpretation. For smooth p and C and
sufficiently narrow buckets, expanding the padding cost locally gives an interval
cost of approximately p(l) * C'(l) * width^2 / 2. Minimizing total cost for fixed K
then gives:

```text
local bucket width proportional to 1 / sqrt(p(l) * C'(l))
```

Equivalently, place approximate boundaries at equal increments of the cumulative
integral of sqrt(p(l) * C'(l)). This explains the density/compute tradeoff and can
initialize candidates, but is not an exact formula for sparse/discrete histograms,
wide buckets, or zero-density regions. The dynamic program is the reference
solution for the discrete proxy objective.

### 8.4 GPU batch-size screening after boundary selection

With the CPU-selected intervals fixed, screen batch size for each interval and
image shape. Choose the feasible batch size minimizing measured time per sample,
not simply the largest batch that fits:

```text
B_star(a, b) = argmin_{B feasible} t(a, b, B) / B
```

t(a, b, B) is representative micro-batch time for retained lengths in (a, b],
padded to b. Use the actual architecture and frozen text encoder, a coarse batch
search followed by local refinement, and a declared memory safety margin.

- Include text encoding, DiT forward/backward, and input/preparation overhead.
  Account for optimizer/EMA cost at the reference accumulation setting.
- Check memory after optimizer-state initialization; report both allocated and
  reserved memory, OOM outcomes, and variability.
- Measure representative within-bucket lengths and actual dropout behavior.
  Frozen-encoder time can depend on actual lengths and batch maxima rather than
  only the DiT's final padding bound.
- Record compilation/startup cost separately from steady time. Check rare shapes
  explicitly even if they did not occur in the sampled profiling stream.
- Cache results by model, GPU, software, execution settings, and the profiled
  length-distribution assumptions. Reuse compatible measurements when the input
  distribution changes, but do not silently reuse an incompatible timing estimate.

The existing `scripts/bench/transformer_ceiling.py` is a DiT-only starting point,
not a sufficient end-to-end cost model. A small fixed-boundary screen is the
default; an exhaustive GPU sweep over all possible intervals is not required.

### 8.5 Optional timing-informed boundary refinement

The two-stage result is optimal for the initial padding proxy, not necessarily
for throughput. A slightly smaller upper bound might allow a more efficient
batch size or kernel, creating a runtime transition the FLOP proxy cannot see.

After screening, optionally refine nearby boundaries or rerun the CPU optimizer
on a bounded candidate set with measured/estimated timing costs:

```text
A_time(a, b) = [F(b) - F(a)] * min_{B feasible} t(a, b, B) / B
```

Use the same recurrence with A_time in place of A. A globally optimal result for
this additive timing table requires valid costs for every allowed interval;
unmeasured costs must be estimated explicitly, and unsupported intervals cannot
be treated as free. Only profile additional intervals near promising candidates
or observed memory/performance transitions within the budget.

Add optimizer or amortized compilation costs only where the chosen cost model
does not already include them. Label approximation assumptions. Full-run compile
cache behavior, queue tails, and distributed synchronization need not be additive;
they remain end-to-end validation concerns. Do not average bucket throughputs to
estimate global throughput: sum sample-weighted time/sample and invert instead.

### 8.6 Replay and end-to-end validation

Replay actual sampler behavior on the CPU to inspect emitted row/caption
proportions, incomplete queue tails, sample batches, rank imbalance, and rare
shape cadence. Extra buckets can increase finite-run tails and compilation cost.

Benchmark only the best few candidates end to end, initially approximately three,
using identical row/caption draw streams. Cover rare lengths and all relevant
aspect shapes explicitly rather than waiting for a short random run to hit them.
Report predicted versus measured runtime, steady and startup-inclusive samples/s,
padding, compiled shapes, memory margin, and actual samples/update. Validate
across the curriculum, not only its cheapest phase.

One-GPU results shortlist plans; they do not establish multi-GPU efficiency.
Validate the eventual target topology, including synchronization and memory
headroom. Corrected multi-GPU throughput was not established by Stage 3's final
single-GPU closure. Do not mix historical timing denominators to infer scaling.

## 9. Dataset refresh and precompute

After accepted enrichment is frozen, materialize versioned caption data and
rebuild retained-length metadata. Preserve stable image IDs, split assignments,
original captions, generation provenance, and a reproducible merge manifest.
Regenerate the light eval data with explicit caption variants, while retaining
the old regression version.

Prepare refreshed 256p data and 640p precompute as specified in Stage 3.5 of the
roadmap. Estimate the full precompute/storage cost separately; it is not funded
by the 75 experimental GPU-hours. If only captions change and image transforms
are identical, assess whether existing image latents can be reused rather than
re-encoding them unnecessarily. Crop/transform changes require corresponding
latent validation or recomputation.

A potential update to `kaupane/chinese-painting-collection` remains a separate
release action after validation, schema/provenance review, and a publication
decision. This design document is not a publishing instruction.

## 10. Execution order, outputs, and remaining decisions

1. Audit existing caption coverage from metadata; pin candidate model/provider
   settings, reviewer accounting, and a capped pilot manifest.
2. Run the caption/review pilot; publish grounding results, actual costs, and
   conservative/central/optimistic affordable image counts.
3. Revise production quantities and routing within the agreed budgets. Freeze
   acceptance rules and audit rates before bulk generation.
4. Produce and review enrichment; freeze training/eval revisions and refreshed
   length metadata. Budget and prepare required image precompute separately.
5. Inspect policy exposure on the CPU; freeze beta endpoints, reserve, progress
   clock, stationary-control construction, quality thresholds, and run manifests.
6. Profile buckets and establish safe screening plans within the profiling cap.
7. Run the five from-scratch screens sequentially as specified, with telemetry.
8. Confirm the reference/challenger comparison across two fresh paired seeds if
   screening warrants it; validate final bucket plans and 640p behavior.
9. Record the verdict, actual spending, data versions, policy, plans, and limits.
   Do not enter Stage 4 automatically.

Required artifacts include the pilot report and cost forecast, versioned caption
and review manifests, evaluation rubric/suites, policy configs, per-run cost and
exposure logs, reusable profiling cache/planner output, and a final quality/cost
decision report. Proposed script/config names are implementation decisions, not
existing commands promised by this document.

Before GPU experiments, resolve these deliberately open numeric choices:

- Final accepted-image count and feasible per-stratum length counts.
- Middle-tier choice, provider settings, escalation rules, and audit sample sizes.
- Reviewer capacity/cost treatment and evaluation suite sizes/weights.
- Exact selector endpoints, reserve setting, progress clock, and exposure-matching
  method; no retrospective tuning to the observed winner.
- Short-prompt and visual-quality tolerances, useful adherence gain, uncertainty
  analysis, and confirmation decision rule.
- Per-run horizon/accumulation calibration and profiling memory margins.

Stage 3.5 closes with a defensible caption-policy verdict, budget ledger,
versioned data/metadata/evaluation, completed separately budgeted 640p preparation,
and measured bucket plans with their validity scope. An inconclusive quality
result is an honest possible outcome; it does not justify claiming an improvement
or silently spending beyond the agreed caps.
