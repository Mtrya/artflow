# ArtFlow redesign plan

ArtFlow is a bilingual text-to-image flow-matching
DiT, with a 532M-parameter hero and an 8-step student as the release target.
**Pretraining and post-training target Ascend on `main`.** The hero is in its
256p stage; post-training preparation proceeds alongside it.

## Document map

- [Hero recipe](hero_recipe.md): architecture, complete training configuration,
  sampling and checkpoint contracts, and resolution transitions.
- [Pretraining infrastructure](infra_pretrain.md): measured execution gains and
  qualification limits.
- [Dataset plan](dataset_plan.md): domains, caption contract and data additions.
- [Post-training preflight](posttrain_preflight.md): judge/scorer evidence,
  throughput accounting, reward-pipeline requirements and qualification.

Platform accounts, images, storage roots, secrets and job submission details
live in the machine-local `INSPIRE.md`. Completed experiment records with
useful additional evidence remain locally in the gitignored `notes/archive/`.
The living notes contain the evidence needed to act on the current design.

## Locked decisions

| ID | Dimension | Choice |
|---|---|---|
| D1 | Data permissions | Research-only sources are permitted, including WikiArt, ArtBench-10 and FFHQ. Preserve per-sample provenance/license metadata and separate restricted mix entries so a clean variant can be assembled. Permission to use a source is distinct from its presence in the current corpus. |
| D2 | Anatomy coverage | Photos and paintings; balance face and full-body coverage, with roughly equal representation as the curation goal. |
| D3 | Corpus | Fixed eligible pools per resolution; source weights are explicit in the complete run config. See [dataset plan](dataset_plan.md). |
| D4 | Model | `artflow-v2`: h1152, 16 heads, 1 double-stream + 24 single-stream blocks, FFN width 3072, 532,496,992 parameters. |
| D5 | Text | Frozen Qwen3-0.6B, encoded online with a true early exit after layer 20. |
| D6 | Resolution | Variable-aspect 256p → 640p → 896p; 75:20:5 of optimizer updates, ending at 450k / 570k / 600k. |
| D7 | Positions | Centered image-grid RoPE; text pinned to a fixed diagonal. Progressive training supplies resolution transfer. |
| D8 | Platform account | Account-selected Inspire project, configured in `INSPIRE.md`. |
| D9 | Training hardware | Ascend for both pretraining and post-training; qualify each workload on its actual allocation. |
| D10 | Captioning | API-based VLM captioning with cached responses and recorded model/prompt provenance. |
| D11 | Modulation | Independent image/text and attention/MLP modulation in the double-stream block; shared attention/MLP modulation within each single-stream block. |
| D12 | Optimizer | Chunked Muon with original scaling, LR 0.02 and decay 0.0015; auxiliary AdamW LR 1e-4, ordinary decay 0.01 and conditioning-matrix decay 0.4. Full settings in [hero recipe](hero_recipe.md). |
| D13 | Monitoring | Fixed bilingual panels, loss probes and final KID; report observed strengths and limitations. Capability forecasts are not launch gates. |
| D14 | Release execution | Reproducible sampling settings; deployment device and performance targets are selected during publication work. |

The model uses rectified flow, logit-normal timestep sampling with resolution
time shift, AdaLN-zero initialization, QK-RMSNorm, gated SiLU FFNs, 2×2 latent
patches, fused pooled-text/time conditioning and branch RMSNorm before gates.
The Qwen-Image VAE supplies 16-channel, factor-8 latents. Caption dropout is 0.1.
Architecture and execution mechanisms live in code; numerical tunables live in
one strict run config covering the entire resolution curriculum.

## Stage status and decision evidence

| Stage | State | Result / next checkpoint |
|---|---|---|
| 0: platform setup | Complete | Repeatable launch/recovery tools and machine-local platform instructions. |
| 1: data curation | Complete; targeted additions continue | Four domains, precomputed latents and online caption encoding. |
| 2: model experiments | Complete | Width/depth, modulation, text exit, positional scheme and optimizer selected. |
| 3: execution efficiency | Complete | Corrected CUDA measurements supplied the initial implementation evidence. |
| 3.5: captions and buckets | Complete | Multi-caption row sampling, length curriculum and variable-aspect planning. |
| 4: Ascend qualification | Complete | 16×910B2C plans for all resolutions, measured execution gains, native transitions, full-state recovery and monitoring. |
| 5: hero pretraining | Active | Finish 256p, activate the finalized recipe at 450k, monitor the actual resolution transfers, then finish 600k. |
| 6: post-training | Preparation | Ascend execution/reward qualification, 640p pilot, then 896p cold start and joint DMD+RL. |
| 7: publication | Pending completed training | Final inference pipeline, `inko` rename, model release and demo. |

The following measurements explain retained choices; their experimental
recipes and hardware define their scope:

- September 5–7 model experiments favored h1152 and single-stream-heavy depth.
  Shared single-stream modulation yielded loss 0.9437 versus 0.9441, KID
  0.0190 versus 0.0195 and 8% lower peak memory in the matched comparison.
- Qwen k20 tied k28 on loss (0.92160 versus 0.92134), improved KID
  (0.00892 versus 0.00922), and won the user's facial-structure review.
- Centered and legacy RoPE tied down to 320p transfer; both failed at
  zero-shot scale factors of at least 1.875. This supports progressive staging.
- The historical RMS-matched Muon experiment reached loss 0.91127 and KID
  0.00699 at 16k, versus AdamW 0.92134 and 0.00922. The current original-scaling
  optimizer is specified and qualified separately in the recipe.
- Corrected Stage-3 single-GPU steady throughput improved 54.99 → 72.98
  samples/s (1.327×); warm-inclusive throughput improved 55.11 → 67.94
  (1.233×). Target-topology performance requires its own measurement.
- September 25 Ascend qualification improved early-256p throughput
  656.29 → 878.91 samples/s (33.92%) with matched experiments and recovery
  checks. See [conditions and limits](infra_pretrain.md).

## Stage 5 — Hero pretraining

Execute [the complete hero recipe](hero_recipe.md) on Ascend. The continuous
600k schedule carries optimizer, EMA, caption curriculum and LR progress across
450k and 570k resolution boundaries. Record actual sample exposure, elapsed
time and NPU-hours throughout training.

All three plans target 16×910B2C, with accumulation **1/3/4**. The 640p/896p
plans use the settled mixtures and measured memory-sized
micro-batches 5–12 / 3–6; native throughput was 173.47 / 83.90 samples/s.
All 280 candidate aspect/length/batch tail cases passed. Measurement scope and
transition/recovery evidence are in [the infrastructure record](infra_pretrain.md).
Apply the complete recipe's future-stage amendments through the explicit
checkpoint migration tool at 450k, preserving the predecessor's original
complete endpoint. Continue normal loss/gradient/functional monitoring when
the actual hero reaches each new resolution.

Review the fixed bilingual panel, live/EMA losses and internal stability
telemetry at the recipe's cadence. Follow the evidence-based review and
recovery rules in [Ascend pretraining](archive/ascend_pretraining.md). A concept
benchmark ran at 200k and 480k and is now retired; run a final capability
benchmark at 600k (form to be decided) for the Stage-6 SFT go/no-go.

Exit: a complete hero checkpoint, verified sampling and resolution transitions,
measured compute/exposure, and a record of observed abilities and limitations.

## Stage 6 — Ascend post-training

The release target is an **8-step student**, trained through optional targeted
SFT, a DMD2 cold start, and joint distribution matching plus RL. The method
follows [DMD2](https://arxiv.org/abs/2405.14867) and
[DMDR](https://arxiv.org/abs/2511.13649). Distribution matching regularizes
reward optimization against the teacher while the reward term supplies the
preference signal:

`L = L_dmd + λ_rl L_rl`.

### Execution and qualification

Use Ascend for the teacher, student, fake-score network and training updates.
The shared VLM judge remains a remote API; place local reward scorers according
to measured support, memory and throughput on the Ascend allocation. CPU
weight-loading evidence is recorded in [preflight](posttrain_preflight.md).

Write the DMD, joint-loss, NFT and reward components per the method notes in
`literature/`, then qualify the real workload: rollout sampling, VAE decode,
fake-score/GAN backward passes, distributed updates, checkpoint/resume, and
reward failure handling. Reuse the qualified
Ascend attention, normalization and launch mechanisms where applicable.

Embed one measurement pass in this bring-up, covering device memory and time
spent in rollout, teacher/fake-score forwards, backward, synchronization,
optimizer and reward waiting. Then run a 640p pilot using the 570k hero
endpoint, followed by the 896p main line using the final 600k checkpoint.
Record end-to-end wall time and NPU-hours for the selected topology before
setting the main-run and ablation budgets. Historical 4090-hour allocations
cannot be converted into an Ascend cap without workload measurements.

### Method and decisions

1. **CFG study, after hero completion:** compare CFG 1 / 1.5 / 2 / 3 using
   conditional/unconditional loss, empty-caption behavior, KID, reward and
   fixed grids. Compare live and stored EMA weights. Use training/evaluation
   sampling helpers while the public generation pipeline awaits Stage 7.
2. **Optional targeted SFT:** the 600k final capability benchmark decides
   the go/no-go and identifies what targeted data could fix. Composition,
   anatomy and texture improvement belong to DMD+RL. Review the
   synthetic teacher and samples before including them.
3. **DMD2 cold start at native 896p:** initialize from the hero; train an online
   fake-score copy on student outputs. Add a classification head on its
   bottleneck, using noise-injected latents, a non-saturating GAN objective and
   real precomputed latents. Start with five fake-score updates per generator
   update. Select the 8-step timestep grid on the final teacher before locking
   the distillation configuration.
4. **Joint DMD+RL:** compare ReFL and DiffusionNFT. Start the λ_rl study near 1
   and adjust using held-out reward and KID; a reward plateau with KID
   regression calls for reducing the RL contribution. Validate the loss-scale
   assumption in the actual implementation. NFT permits asynchronous terminal
   image rollouts/rewards and works with the 8-step student.
5. **Rewards:** combine aesthetic, HPSv2 and VLM rubric signals, with explicit
   criteria for anatomy, prompt adherence and over-smoothed artifacts. Keep
   a separately monitored reward outside the optimized ensemble. Final
   composition and weights are selected in the pilot.
6. **Domain coverage:** mirror the hero mix in rollout prompts. Inspect fixed
   canary grids for Chinese painting, Western painting, people and world
   content; track train/held-out reward divergence and diversity. The user's
   image review decides whether the output looks over-optimized.

Ablation axes are RL algorithm, λ_rl, reward composition/domain weights,
rollout group size G=16/24 and prompts per iteration (initial candidate 48).
This produces 768/1,152 images per iteration; judge throughput can dominate
wall time. Use the corrected accounting in [preflight](posttrain_preflight.md).

Decide the cold-start → joint promotion rule with the 640p pilot. Promotion
must leave room for joint learning before distillation converges. Build a
small 5k-prompt pool for bring-up; finalize the roughly 20k pool with the user
at the 640p checkpoint. Memory measurements set backward-simulation
micro-batches. Main algorithm/λ studies use the completed hero.

Exit: an 8-step student with improved reward and panel review, preserved
KID/diversity, recorded reward/λ configuration, domain comparison grids and
measured Ascend costs.

## Stage 7 — Publication

Rewrite the standalone generation pipeline and demo **after all model training
is complete**, using finalized checkpoint metadata, text conditioning and
sampling behavior.

Rename the repository and model to **`inko`** before publication. Update
imports, configs, comments, documentation, model links and demo metadata;
preserve an explicit mapping for historical artifact identifiers.

Publish the selected 8-step student to Hugging Face with weights, inference
configuration, model card, provenance/license constraints, evaluation and
limitations. Deploy and smoke-test a Hugging Face Space. Write the README with
installation, minimal inference/training examples and a navigable document map;
verify its quickstart in a clean environment.

Publish self-contained explanations of Flow Matching and DMD2/DMDR/NFT,
including equations, training/sampling procedures, implementation mapping and
limitations. Explain methods directly; private run nicknames or unavailable
discussions cannot carry the explanation. Estimate publication and hosting
costs before deployment.

## Current risks

- Persistent internal scale growth: follow the live/EMA, fixed-panel and
  checkpoint evidence in [Ascend pretraining](archive/ascend_pretraining.md).
- Resolution transitions: qualify each plan and preserve complete endpoints.
- Caption/judge provider changes: cache responses and record exact request
  identity; measure limits again on the intended workload.
- Reward hacking and diversity loss: keep held-out signals and domain review
  throughout joint training.
- Source restrictions: preserve provenance and separate restricted entries;
  release permissions depend on the selected data and artifacts.
