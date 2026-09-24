# Ascend hero recipe

Current direction: fresh Ascend pretraining on `main`. The complete runnable
recipe is [`configs/hero.toml`](../configs/hero.toml). It contains every tunable
for all three resolution stages. Missing or unknown fields fail validation;
there is no base config, config overlay, or environment/CLI hyperparameter
source. Stage selection and checkpoint resume are operations.

## Selected settings

- Architecture `artflow-v2`: h1152, 16 heads, 1 double-stream + 24 single-stream
  blocks, gated MLP expansion 2.67. Branch RMS normalization, fused pooled-text
  conditioning, centered RoPE, and factor-1000 timestep features are native.
  Double-stream modulation is independent; single-stream modulation is shared
  within each block. Conditioning-input LayerNorm is absent.
- Muon uses original per-chunk scaling at LR 0.02, decay 0.0015, momentum 0.95.
  Auxiliary AdamW uses LR 1e-4, decay 0.01, betas 0.9/0.95. Clip norm is 1.0.
- One continuous 600k-step schedule, with 20k warmup. Stage endpoints are
  450k, 570k, and 600k. All dataset weights, including the higher 256p D4
  share and proportional shard weights, are explicit in the run file.
- Bias-corrected EMA, decay 0.9999. Linear beta caption selection, log2 caption
  loss weighting, and logit-normal timestep sampling are native mechanisms;
  their numerical parameters remain explicit tunables.
- Recovery checkpoints every 2k updates; keep the latest two complete copies
  within each stage. Each checkpoint records its complete run config, model
  architecture/capacity metadata, and the active bucket table.

## Readiness

The short stability stage passed with the preceding RMS-matched Muon recipe.
Original scaling changes the FFN down-projection update coefficient, so the
infrastructure pass must exercise this exact implementation and configuration.
See [the stability evidence](archive/ascend_stability_stage1_0924.md) and
[current plan](ascend_pretraining_0924.md). This is a credible stability
candidate, not a claim of optimality or long-run stability.

The run file names new Ascend bucket artifacts. They are **not installed or
qualified yet**. Accumulations 2/4/5 are explicit candidates for the infra
pass, not measured production choices at all three resolutions. Startup fails
if the selected plan is missing. Qualify the plans and set their final
accumulations before launching the hero.

## Launch and recovery

Prepare the environment, models, and datasets using machine-local platform
instructions. Set `paths.storage_root` in the complete run file to the actual
artifact root. Relative storage roots resolve against the config file's
location. Data/model/output/bucket paths resolve against that root; the prompt
file resolves against the config file. There is no environment interpolation.

```bash
python -m src.pretrain.train --config configs/hero.toml --stage 256p --check_config
python -m scripts.pretrain.launch --config configs/hero.toml --stage 256p --nproc_per_node 16
```

Select `640p` or `896p` after its predecessor finishes. The launcher finds the
latest complete same-stage checkpoint, otherwise the predecessor's endpoint.
It holds a writer lock throughout training, mirrors the current attempt's
output, and terminates the worker group on NPU OOM. Explicit `--resume` selects
a particular complete checkpoint. Resume requires the recorded recipe; the
trainer also verifies that a same-stage bucket table has not changed.

SwanLab receives the entire resolved recipe, architecture version, selected
stage, and active bucket contents. Fixed mechanisms do not produce individual
boolean fields. Older configurations and architecture variants require their
historical Git revision. The [prior CUDA hero record](archive/cuda_hero_recipe_0924.md)
is retained as evidence, not a current launch procedure.
