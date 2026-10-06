# Pretraining recipe

The complete runnable recipe is
[`configs/pretrain.toml`](../configs/pretrain.toml). It explicitly specifies every
training tunable for all three resolution stages and rejects missing or
unknown fields. Stage selection and checkpoint resume are operations.
Recovery requirements are described below; machine-local run identity lives in `INSPIRE.md`.

## Model and optimization

| Component | Selected setting |
|---|---|
| Architecture | `inko`, h1152, 16 heads, 1 double-stream + 24 single-stream blocks, FFN width 3072 (ratio 8/3), 532,496,992 parameters |
| Text | Frozen Qwen3-0.6B, online true early exit at layer 20, hidden size 1024 |
| Image latents | Qwen-Image VAE, 16 channels, factor 8; 2×2 latent patches |
| Conditioning | Fused pooled text/time, factor-1000 timestep features, branch RMSNorm before gates |
| Positions/modulation | Centered image RoPE and fixed text diagonal; independent double-stream and per-block shared single-stream modulation |
| Muon | Original per-chunk scaling, LR 0.02, weight decay 0.0015, momentum 0.95 |
| Auxiliary AdamW | Peak LR 1e-4, betas 0.9/0.95, epsilon 1e-8, ordinary decay 0.01 |
| Conditioning decay | 0.4 on `txt_pooled_proj.weight`, `c_mlp.0.weight`, `c_mlp.2.weight` |
| Gradient clipping | Global norm 1.0 |
| EMA | Bias-corrected, decay 0.9999, updated every optimizer step |

Muon chunks fused QKV/modulation matrices before orthogonalization. Its update
multiplier is `sqrt(max(1, rows/cols))` for each chunk. Conditioning matrices
belong to AdamW. The selected decay 0.4 was installed through a full-state
migration at 56k. The checkpoint migration record retains the original recipe
and state hashes; the selected decay applies to those conditioning matrices only.

## Schedule and sampling

One continuous 600k-update schedule uses 20k warmup and cosine decay. AdamW
starts at 1e-5, peaks at 1e-4 and reaches 5e-6. Stage endpoints are 450k, 570k
and 600k (256p / 640p / 896p, a 75:20:5 step split).

Rows are sampled from explicit weighted source pools; a caption is selected
within the row. Caption-length beta changes linearly from −1 to +1 over the
run. A 20% short-caption reserve uses a 256-token threshold. Caption dropout
is independently sampled at probability 0.1. Loss weighting uses log2 caption
length relative to 128 tokens. Logit-normal timestep parameters are 0 and 1;
resolution time shift is native. Text tokenization/encoding runs online;
precompute stores latents and cleaned caption text with sampling metadata.

All three bucket plans target 16×910B2C. Each contains 20 caption-length
buckets per aspect ratio, up to 2,048 tokens.

| Stage | Micro-batch per rank | Accumulation | Expected samples/update |
|---|---:|---:|---:|
| 256p | 8–72 | 1 | Depends on caption progress; about 950 in the early qualification |
| 640p | 5–12 | 3 | 528 planned; 532 measured |
| 896p | 3–6 | 4 | 373 planned; 375 measured |

The later-stage measurements below describe the previously qualified mixtures.
Correcting source weights or filtering captions changes the distribution used
by the planner; reassess expected batches and qualify the resulting plans on
the full workload. The 52 GiB planner ceiling includes model, encoder,
optimizer, EMA, gradients and DDP residency. Measurement conditions and limits
are in [the infrastructure record](infra_pretrain.md).

## Resolution-specific source mixture

For each resolution, count eligible training rows in the final dataset artifacts
and assign source mass `rows × multiplier`, then normalize over that stage:

- 1.8: `d1`, `d2-wikiart`, `d2-museum`, `d3-pexels`.
- 1.5: `d4-extra`, `d4-extra2`, `d4-extra3`.
- 1.0: every other source.

The counts must come from that resolution after filtering. Caption variants
are choices within an image row, not extra rows. Split sources such as
`d3-people-a/b` and `d4-relaion-p0..4` each contribute their own eligible rows;
splitting a source must not multiply its aggregate weight.

The previous 640p/896p weights reused 256p counts, overemphasizing sources
whose higher-resolution eligible pools had shrunk. That error does not change
the completed 256p stage. The legacy 640p continuation is stopped; corrected
later-stage weights and their count evidence must be audited before restarting.
The selected restart is the 480k checkpoint. Locating and migrating the real
artifact, handling changed row pools, and launching training are separate
operations from the repository cleanup.

Strict resume compares the full recorded recipe, including future stages.
[`migrate_checkpoint.py`](../scripts/pretrain/migrate_checkpoint.py) permits
explicit future-stage data/bucket/accumulation changes, while preserving model
capacity, optimizer policy and the global schedule. Current-stage entries may
be reweighted but cannot be added, removed or reordered. A resolution
transition constructs a fresh sampler. See [dataset provenance](dataset_plan.md).

## Evaluation and telemetry

- Fixed loss probe every 500 updates, requesting 512 held-out cases per
  caption-length band with evaluation batch size 8. Record actual counts when
  eligibility or buckets reduce the requested set; compare matched panels/counts.
- Image grids every 2,500 updates and at the explicit stage `grid_steps`.
  [`monitor.jsonl`](../configs/prompts/monitor.jsonl) contains
  12 scenes × Chinese/English × short/long captions = 48 images. Each scene
  shares its seed across its four variants. Review anatomy, architecture,
  style and layout as improved / unchanged / regressed / uncertain.
- Grid sampling: stored EMA, Euler 50, CFG 1, resolution time shift;
  solver times/state at least FP32, model forward and VAE decode BF16.
- End-of-global-training KID uses 2,000 generated images.
- Health telemetry every 250 updates, stability probes every 100, caption/source
  logging every 25. Preserve per-update samples, losses and gradient records.

Compare live and stored EMA loss together: smoothing can hide a live-model
regression. Read caption-band sample counts with their losses: underfilled bands are
reported, and an empty band has no loss estimate. Inspect fixed-prompt
generations before attributing an internal-scale trend to quality regression.

At the completed pretraining checkpoint, evaluate live weights first, then compare
stored EMA on loss, grids and KID before selecting the post-training teacher.
Post-hoc EMA is an option only if enough suitable checkpoint history was
retained; the rotating two-checkpoint policy cannot reconstruct arbitrary
historical decay profiles.

## Launch, checkpoints and resolution transitions

Supply the prepared external artifact root through required `--storage-root`.
Dataset, model and output paths are relative to that root and stay untracked.
Prompt suites and bucket plans live in `configs/`; their paths resolve from
the repository, independent of the working directory or TOML location. The
resolved checkpoint recipe records both sets of absolute paths. Platform
preparation and ordinary SwanLab login/API-key setup belong in local `INSPIRE.md`.

```bash
python -m src.pretrain.train --config configs/pretrain.toml --stage 256p --storage-root /external/inko --check_config
python -m scripts.pretrain.launch --config configs/pretrain.toml --stage 256p --storage-root /external/inko --nproc_per_node 16
```

The launcher sets the qualified NPU expandable-segments allocator policy and
one OpenMP thread per worker before initialization. It holds a writer lock,
mirrors attempt output and terminates the worker group on NPU OOM.

Save every 2,000 updates and retain the latest two complete checkpoints plus
all configured stage endpoints. All stages share one run directory. Checkpoints contain the complete run config, model metadata, active
bucket table, model/EMA, optimizers, schedulers, sampler and per-rank RNG state.
Recovery chooses the latest complete same-stage checkpoint; a stage's first
launch uses its predecessor's endpoint. `--resume` selects a complete checkpoint
explicitly. Same-stage recovery also checks bucket-table equality.

Preserve the complete 450k and 570k endpoints before transitioning. Qualify
the new bucket plan, accumulation, memory, monitoring and full-state recovery
while retaining global schedule progress. Any required recipe change uses a
tested migration with provenance. SwanLab records the complete actual recipe,
architecture version, selected stage and active bucket contents.

SwanLab always starts online; configuration/authentication failures stop startup.
Each checkpoint must contain `tracking.json` with `swanlab_project` and
`swanlab_run_id`. Ordinary resume requires the recorded project and uses
`resume="must"`. A missing record, project mismatch or invalid ID is an error;
the trainer never reads identity from the checkpoint's parent directory.

Training-state continuation and experiment identity are separate operations.
`--new-experiment` requires a checkpoint and explicitly starts a fresh experiment
in the configured project while restoring its full training state. The initial
experiment config records the source project, run ID, checkpoint and global step.
Subsequent checkpoints record the new experiment's identity; ordinary recovery
from those checkpoints omits `--new-experiment`. A fresh experiment explicitly
uses `resume="never"`.

The 256p and discarded 640p experiment histories remain in `artflow`. The
corrected continuation uses `inko`. The model class/module is `Inko` /
`src.models.inko`, and its architecture identifier is `inko`; tensor names,
shapes and numerical operators are unchanged by this rename. The migration
reader alone recognizes `artflow-v2` and records the metadata conversion.

Before adopting a new checkout or recipe, migrate a complete checkpoint into
an independent destination. For a checkpoint without its own tracking record,
supply the verified source experiment ID explicitly:

```bash
python -m scripts.pretrain.migrate_checkpoint \
  --source /external/inko/keep/SOURCE/checkpoint_step_STEP \
  --destination /external/inko/keep/pretrain-start/checkpoint_step_STEP \
  --config configs/pretrain.toml --storage-root /external/inko \
  --source-run-id SOURCE_RUN_ID \
  --reason "Import checkpoint metadata and correct future-stage mixture"

python -m scripts.pretrain.launch \
  --config configs/pretrain.toml --stage 640p --storage-root /external/inko \
  --nproc_per_node 16 \
  --resume /external/inko/keep/pretrain-start/checkpoint_step_STEP \
  --new-experiment --verify_resume_state
```

The paths above are examples; the external storage directory need not share
the repository's name. Migrate against the final checkout location: resolved
config-asset paths are part of the checkpoint recipe. Repository renaming
itself does not move datasets, weights, checkpoints or experiment histories.

The tool validates the source inventory, verifies relocated prompt bytes and
completed/current bucket contents, and records both recipes, both model
identities and file hashes. Model, optimizer, scheduler, EMA, sampler and RNG
artifacts remain byte-identical. It publishes the destination only after all
copies and metadata checks pass. The original remains untouched.

This metadata migration does not adapt saved
sampler cycles or queued row IDs to filtered/replaced datasets. At a mid-stage
restart such as 480k, that requires a separate explicit sampler-state decision;
`--new-experiment` changes tracking identity only. A migrated checkpoint
retains its source tracking identity until the explicit new
experiment begins; migration itself creates no SwanLab experiment.

Repository tests verify artifact copying, state restoration contracts and
scheduling mechanics. They do not establish training quality or a successful
NPU restart. Qualify the actual migrated checkpoint with a device restore,
then inspect real generations and telemetry under the corrected mixture.
Historical measurements are in [the infrastructure record](infra_pretrain.md).

## Reading stability telemetry

Read per-update loss and pre-clip gradient distributions together with nearby
live/EMA evaluations, caption lengths, source mix and bucket workload. Cloud
plot downsampling cannot establish a gradient median or rare-event frequency.
Compare fixed-panel responses only at matching panel IDs, sample counts, text
lengths and shifted times. The small fixed panel does not cover the full caption
curriculum.

Nonfinite loss or gradients stop all ranks before an optimizer update. Finite
spikes are not skipped and there is no automatic rollback. Raw-gradient clipping
does not bound Muon's parameter-update norm after momentum orthogonalization.
Internal RMS, gains and directional sensitivities have no universal failure
threshold; inspect individual layers, absolute variation, model response and
real generations before changing an optimizer setting.

If finite loss and gradients deteriorate persistently, preserve the first
traceback, onset logs, panels, complete checkpoints and exact recipe before
retention removes a useful predecessor. Establish a reproducible mechanism,
then qualify any recovery or intervention with a recorded state migration.
Metric definitions live in [stability.py](../src/pretrain/stability.py),
[health.py](../src/pretrain/health.py) and [train.py](../src/pretrain/train.py).
