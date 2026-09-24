# Ascend stability, Stage 1 — 2026-09-24

Completed at 20:39 CST. Current work and the selected configuration live in
[the Ascend pretraining plan](../ascend_pretraining_0924.md).

The user set the sequence: **stability → repository/config/docs cleanup →
infrastructure pass → hero**. This stage establishes a credible stable model
and configuration through short informative experiments. A successful probe
does not trigger a hero launch. Earlier architecture/optimizer suggestions
remain hypotheses until the relevant measurements support adopting them.

## Telemetry contract

`[telemetry].stability_interval` is the only new control; zero disables the
observer. At the first update and every interval thereafter, rank zero observes
a saved panel of up to four examples, each at base times 0.1, 0.5 and 0.9
(with the normal resolution shift). Captions, encoded text, latents and noise
stay fixed. `stability_inputs.pt` is reused on resume and copied between arms;
its SHA-256 prefix is recorded in every observation. This small panel cannot
represent the complete training distribution.

| Measurement | Decision it informs |
|---|---|
| Conditioning RMS; caption variation at fixed time; time variation at fixed caption; negative-tail fraction | Is useful variation being lost, rather than merely a raw activation statistic changing? |
| Actual conditioning displacement per update; shared and centered parts; final-linear `delta_W @ mean(h)` versus bias displacement | Is the previously isolated update mechanism still large? |
| Fixed-panel loss before/after the actual update; post-update loss with old conditioning; loss with twice the actual conditioning displacement | Does this update encounter a sharp response, and is that response attributable to conditioning on the updated trunk? |
| Prediction change divided by conditioning displacement; per-time loss second differences | Is local sensitivity increasing? Small-displacement bf16 rounding can affect these ratios; interpret with absolute loss/output changes. |
| Per-block image-residual RMS and realized attention/MLP contribution ratios; separate modulation shift/scale/gate RMS; branch norm gains | Are gates or later layers reintroducing amplification despite branch normalization? |
| Sampled matrix weight/update RMS, update ratio, radial contribution, step energy and decay contribution | Are norms growing through aligned updates, step energy, or insufficient decay? Samples cover conditioning and first/middle/last blocks, not all Muon parameters. |
| Actual update execution, skip streak, training loss/gradient norm, live and EMA evaluation | Is apparent health genuine learning, rather than skipped updates, slow learning, or an EMA-only picture? |

The conditioning counterfactual holds the **updated trunk fixed**. It replaces
the conditioning vector with its pre-update value, then with old + 2×actual
displacement. It is a local response measurement, not a universal spike
predictor or a deployed optimizer safeguard. The gain is a dimension-dependent
RMS ratio, not a Jacobian bound; a value below one is not a stability criterion.
All probes are detached, eager,
and outside DDP; gradients, optimizer state and training RNG must be unchanged.
Per-layer/tensor detail goes to `stability.jsonl`; a smaller summary goes to
SwanLab. `training_metrics.jsonl` retains every update and evaluation independent
of cloud access. Measured probe time is included in throughput accounting.

## First experiment: two short continuations from the same learned state

Question: after branch normalization, does recalibrating optimizer steps improve
functional update stability, and what residual problem merits the next change?

Both arms start from the completed `bnprobe-branch-a2b` step-2000 checkpoint,
including optimizer, scheduler, EMA, RNG and sampler. Both use 16×910B,
accumulation 2 with the saved half-micro-batch plan, and stop at step 3000.
Reference: Muon 0.02 / auxiliary AdamW 3e-4. Calibrated: 0.003 / 1e-4.
Both keep Muon decay 0.01, branch norm on and conditioning-input norm off.
The timestep factor stays 1 because these are continuations of factor-1
weights; a factor-1000 architecture experiment must start fresh.

The schedule horizon remains 200k with the checkpoint's 200-step warmup. The
observer runs every 100 steps; live/EMA evaluation every 500 on 48 samples per
caption-length band. Spike skipping is explicitly disabled: persistent skips
must not masquerade as stability. The existing coordinated nonfinite guard
still stops before a nonfinite update. Each arm retains its latest two complete
checkpoints and uses a separate output directory. The launcher refuses to
overwrite an existing experiment.

The pair combines an optimizer intervention rather than sweeping each LR
independently. Holding the initial state and diagnostic panel fixed makes its
first update particularly informative. The following 1000 updates test whether
the response and geometry trend improve while useful learning continues.
It does not establish that a lower LR prevents recurrence indefinitely.

Next decisions depend on the observations: persistent conditioning sensitivity
calls for a conditioning/architecture intervention; dominant outward radial
growth gives decay a measured target; healthy response with poor learning
suggests over-conservative optimizer steps. A fresh combined-recipe experiment
is required before declaring Stage 1 complete after architecture changes.

## Completed continuation comparison (19:57 CST)

Job `ascend-stability1-0924` succeeded. Both endpoint checkpoints were validated.
Each arm applied **all 1000 updates**, processed the same **936,418 samples**,
and produced 11 observations on panel **a5eca79b4a12b640**. No loss spike was
observed. Baseline live/EMA evaluations and pre-update panel statistics match
exactly; the first resumed update uses the new configured LRs.

| Measurement | Reference: Muon .02 / Adam 3e-4 | Calibrated: Muon .003 / Adam 1e-4 |
|---|---:|---:|
| Initial EMA / live loss, step 2000 | .85988654 / .87153464 | identical |
| EMA / live loss, step 2500 | .85047320 / .86187347 | .85228233 / .85143508 |
| Final EMA loss, step 3000 | **.84162499** | .84547391 |
| Final live loss, step 3000 | .85816267 | **.84522120** |
| Gradient norm median / p99 / max | .16466 / .37314 / .47012 | .08007 / .20055 / .30319 |
| Largest absolute sampled conditioning loss effect | .00905746 | .00251842 |
| Final conditioning update RMS | .01975247 | .00956100 |
| Maximum gate RMS, start → end | 19.67 → 26.04 | 19.67 → 25.07 |
| Maximum block residual RMS, start → end | 167.53 → 209.74 | 167.53 → 198.94 |
| Sampled attention matrix RMS, start → end | .42733 → .48662 | .42730 → .42065 |
| Sampled FFN matrix RMS, start → end | .48852 → .54994 | .48850 → .47654 |
| Sampled modulation matrix RMS, start → end | .16699 → .17762 | .16698 → .16201 |
| Steady throughput, samples/s | 643.80 | 645.36 |
| Peak allocated memory, GiB | 37.7 | 37.7 |

The common evaluation definition resolves to **147 examples**: 48 each in the
first three caption-length bands, only three in 513–1024 tokens, none above
1024. Absolute loss is not comparable with the older larger evaluation panel.
The mechanism panel has only **four short-caption examples, 88 text tokens**.

### Decisions from the comparison

- **Carry the calibrated rates into the fresh check.** They improve live loss
  and reduce observed update effects while learning continues. They are not
  an across-the-board quality win: EMA loss is 0.00385 worse at the endpoint.
  This continuation does not measure learning speed from random initialization.
- **Keep Muon decay 0.01.** Sampled matrix norms decline under calibrated rates,
  yet gate/residual magnitudes still rise. Weight growth and gate growth are
  therefore distinct, and lowering weight RMS is not itself the objective.
  There is no demonstrated need to increase decay solely to restrain these norms.
- **Do not call the sensitivity problem solved.** Both arms' largest sampled
  conditioning loss effect occurs at their endpoint at base t=0.1. The
  calibrated effect is smaller, but conditioning-prediction gain reaches
  1.866, versus reference 2.302. This is directional, local evidence on actual
  updates, not a universal spike predictor or proof of indefinite stability.
- **Read activation statistics through function and learning.** Conditioning
  RMS ends at 1.0682 / 1.2082 (reference/calibrated), caption variation/RMS at
  .01911 / .01912, time variation/RMS at .12538 / .10204, and negative-tail
  fraction at .68822 / .65668. A smaller negative-tail fraction alone would
  miss the continued gate growth; a larger gate alone would miss improved loss.

The first update also demonstrates the bf16 ratio caveat: calibrated
conditioning displacement/prediction change RMS is .00750/.01138 versus
reference .02069/.02541. Both absolute changes decrease, but their quotient
increases (1.517 versus 1.228). Median reference probe time is **0.345 s**;
this is not an isolated whole-run overhead benchmark.

Source archive SHA-256:
`bf4fa7989d83405faf31a43a9733c59b6944cc8d0b91087a6f824acd1cba308c`
(70-file manifest verified at startup). Run directories:
`ascend-stability-0924-reference` and `ascend-stability-0924-calibrated`.

## Fresh combined-recipe check (completed 20:39 CST)

Question: does the combined recipe learn from random initialization while
maintaining small functional conditioning-update responses at peak LR?
The continuation cannot answer this initialization question.

- Branch norm on, conditioning-input norm off, timestep factor **1000**.
- Muon **0.003**, auxiliary AdamW **1e-4**, Muon decay **0.01**.
- Same recorded input mixture and half-microbatch plan; 16×910B, accumulation
  2, eager execution and foreach updates. Fresh seed-42 weights and optimizer.
- Horizon **600k**, endpoint **2000**, compressed warmup **200** to exercise
  peak LR early. The future hero retains its requested **20k warmup**.
  The horizon also governs caption progression, so this is not an exact A/B
  against the old 200k-horizon probe.
- Same saved diagnostic inputs, cadence 100; live/EMA evaluation every 500 on
  the same 147 examples. Spike skipping disabled, nonfinite guard retained.
  Checkpoint every 500, latest two retained.

**All 2000 updates applied, zero skipped updates.** All 2000 gradient records
were collected in order and all 21 observations have the same panel identifier.
There were 31 ordinary clipping events during the first 200 warmup steps and
**none after warmup**.

| Updates | Median gradient norm | Maximum gradient norm |
|---|---:|---:|
| 1–200 | .66555 | 1.37057 |
| 201–500 | .37754 | .86745 |
| 501–1000 | .24630 | .52087 |
| 1001–1500 | .19618 | .46816 |
| 1501–2000 | .16420 | .40657 |

| Step | EMA evaluation loss | Live evaluation loss |
|---|---:|---:|
| 0 | 1.98811152 | 1.98811152 |
| 500 | .91275436 | .92416462 |
| 1000 | .88123242 | .89200708 |
| 1500 | .86733491 | .87725872 |
| 2000 | **.85689331** | **.86698182** |

The earlier branch-norm, factor-1, higher-LR checkpoint at step 2000 measured
**.85988654 EMA / .87153464 live** on this same evaluation definition. The
combined fresh recipe is slightly better at this training age; this supports
retaining it without a learning-regression concern. It does not isolate which
change caused the improvement or predict the eventual quality ranking.

Across all sampled updates, the largest absolute conditioning loss effect is
**.00300884**, and the largest absolute second finite difference is **.00124812**.
At step 2000 the largest effect is **.00121850**, with conditioning displacement
RMS **.01941119**, induced prediction change RMS **.01132298**, and gain **.58332**.
The last ten sampled gains range .453–.732 without a sustained rise (the full
trajectory, rather than the inequality gain < 1, informs this decision).

Endpoint feature statistics: condition RMS **3.0380**, caption variation/RMS
**.03476**, time variation/RMS **.15812**, hidden negative-tail fraction **.26071**.
Branch-norm affine gain max is **1.01898**, gate RMS max **17.08185**, residual
RMS max **121.11676**. The first observation from the previous branch-norm
checkpoint has condition RMS 1.0407, caption/time ratios .02178/.09792 and
negative-tail fraction .66146. Different conditioning scales must be read
alongside retained variation and the measured output/loss response.

**Growth is not eliminated and is not the acceptance criterion.** Fresh-run
sampled attention/FFN/modulation matrix RMS ends at .07868/.09073/.03508 and
is still increasing, predominantly through outward radial updates. The final
conditioning-weight shared shift is .010038 RMS versus .000008721 for its
bias update—over 1000 times larger—yet the measured loss effect is small.
Thus neither increasing weight norms nor this shift-to-bias ratio independently
identifies a spike. The decision rests on functional update response, complete
gradient history, actual updates and learning.

Job **`ascend-stability-fresh-0924`** succeeded, with a validated step-2000
checkpoint, 21 observations, **1,879,815 samples**, **656.72 steady samples/s**,
and **37.7 GiB** peak allocated memory. The two jobs together occupied about
**28.53 NPU-hours**, including startup and teardown; all allocations are released.
Throughput differences across these runs are not an isolated optimization A/B.

Fresh source SHA-256:
`935c8d6129501c8a09f32e5b45e8da51e5aef69892eeb6cee4645d3c3bb684a4`
(71-file manifest verified). Historical launcher/config definition is in commit
`d8f2840`: `scripts/ascend/stability_probe.sh fresh`, with `fresh.toml`
layered after reference and calibrated configs. Remote immutable source is
`repo-stability-0924-v2`; output is `runs/ascend-stability-0924-fresh` under
the canonical Ascend root in INSPIRE.md. Raw `stability.jsonl`,
`training_metrics.jsonl` and `stability_inputs.pt` remain with each run.

## Stage-1 decision

**Close the short stability stage and proceed to repository/config/docs cleanup.**
Select branch norm, factor 1000, no conditioning-input norm, Muon .003,
auxiliary AdamW 1e-4 and Muon decay .01. The model retains h1152, 16 heads,
one double-stream plus 24 single-stream blocks; branch norms bring the count
to **532,766,716 parameters**. Double-stream modulation is none, single-stream
modulation is layer, matching the actual tested configuration.

No additional isolated normalization/decay sweep or separate long validation is
required by these results. This is a credible short-run recipe, not proof of
indefinite spike prevention. The mechanism panel is small and short-caption-only;
the later infrastructure pass must exercise the real caption/bucket workload,
resume path and resolution transitions. Production still requires the user's
cleanup and infrastructure stages. The hero remains fresh, 600k steps, 20k
warmup, bias-corrected EMA, and the previously requested dataset mixture.

## Checkpoint correctness and verification

The new `[model].timestep_factor` applies only inside the sinusoidal embedder;
flow interpolation and the objective retain their original time values.
Non-default factors are stored as integer checkpoint metadata and model config.
State-dict inference recovers them; mismatched loading fails even with
`strict=False`. Legacy factor-1 embeddings and state-dict keys are unchanged.
Historical monkey-patched factor-1000 checkpoints lack metadata and are not
inferred automatically. No factor-1 continuation was converted in place.

**798 tests passed**, including legacy/scaled embedding behavior, unchanged input
times, checkpoint mismatch rejection, model round trip, monitor non-interference,
conditioning counterfactuals, and the first resumed update's configured LR.
Distributed tests need loopback sockets outside the sandbox; the sandbox-only
TCPStore EPERM was an environment restriction, not a training failure.

Monitoring uses offline SwanLab and durable local JSONL. Short Inspire job-shell
reads recover complete snapshots; long-lived shells sometimes disconnect.
Frequent shell opens hit a gateway websocket limit; access recovered after a
cooldown. Prefer platform logs for frequent checks and sparse full snapshots.
Automatic approval review rejected extracting a credential from a historical
job command; the submitted jobs read/transmit no credential. No new hero has
been launched. **Stage 1 is closed; cleanup is next.**
