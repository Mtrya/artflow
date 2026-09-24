# Ascend stability, Stage 1 — 2026-09-24

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
predictor or a deployed optimizer safeguard. All probes are detached, eager,
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

## Fresh combined-recipe check (running; launched 19:47 CST)

Question: does a fresh model with the combined choices learn while maintaining
small functional conditioning-update responses through early training at peak
LR? The continuation cannot answer this initialization question. Its midpoint
results justified starting this bounded check while its final 500 updates ran.

- Branch norm on, conditioning-input norm off, timestep factor **1000**.
- Muon **0.003**, auxiliary AdamW **1e-4**, Muon decay **0.01**.
- Same recorded input mixture and half-microbatch plan; 16×910B, accumulation
  2, eager execution and foreach updates. Fresh seed-42 weights and optimizer.
- Horizon **600k**, endpoint **2000**, compressed warmup **200** to exercise
  peak LR early. The future hero retains its requested **20k warmup**.
  The horizon also governs caption progression, so this is not an exact A/B
  against the old 200k-horizon probe.
- Same saved diagnostic inputs, cadence 100; live/EMA evaluation every 500 on
  the same 147 examples. No spike skipping; coordinated nonfinite guard retained.
  Checkpoint every 500, latest two retained. No continuation beyond 2000.
- Initial live/EMA loss **1.98811152**. Through step **1200**, all updates applied,
  maximum gradient 1.37057 during early learning (ordinary clipping, no skips);
  latest sampled conditioning loss effect at most .00012517. EMA/live loss:
  step 500 **.91275436/.92416462**; step 1000 **.88123242/.89200708**.
  Too early to judge the final recipe.

Job **`ascend-stability-fresh-0924`**, HIGH priority, on a second 16-card node.
The original job completed and released its allocation. Fresh source SHA-256:
`935c8d6129501c8a09f32e5b45e8da51e5aef69892eeb6cee4645d3c3bb684a4`
(71-file manifest verified). Repeatable definition:
`scripts/ascend/stability_probe.sh fresh`; `fresh.toml` layered after the
reference and calibrated configs in `configs/experiments/stability_0924/`.
Output: `ascend-stability-0924-fresh`.

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
been launched. **Stage 1 remains open pending the fresh experiment.**
