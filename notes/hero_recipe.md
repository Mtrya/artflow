# Ascend hero recipe

Current direction: fresh Ascend pretraining on `main`. The complete runnable
recipe is [`configs/hero.toml`](../configs/hero.toml). It contains every tunable
for all three resolution stages. Missing or unknown fields fail validation;
there is no base config, config overlay, or environment/CLI hyperparameter
source. Stage selection and checkpoint resume are operations.

## Selected settings

- Architecture `artflow-v2`: h1152, 16 heads, 1 double-stream + 24 single-stream
  blocks, gated MLP expansion 8/3 (width 3072), 532,496,992 parameters.
  Branch RMS normalization, fused pooled-text
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
Original scaling changes the FFN down-projection update coefficient. The
infrastructure pass exercised the selected convention in fresh 600-update
runs and full-state replay, with the production 20k warmup horizon. Those
checks cover early warmup, not peak-LR or long-run stability.
See [the stability evidence](archive/ascend_stability_stage1_0924.md) and
[current plan](ascend_pretraining_0924.md). This is a credible stability
candidate, not a claim of optimality or long-run stability.

The 256p plan is installed and qualified on 16×910B2C, with micro-batches 8–72
and accumulation 1. The real-workload and padded 2048-token tail probes peaked
at 52.61 and 49.10 GiB allocated, respectively. September 25's FFN alignment
comparison measured 843.29 versus 765.34 samples/s with identical sample
identities: 10.19% faster for 0.05% fewer total parameters. See the
[infra evidence](infra_pass.md) for conditions and limits.

Native NPU RMSNorm and SwiGLU are measured execution mechanisms. The latter
adds 4.22% throughput on a matched-node 600-update comparison, with comparable
short-run loss/gradient distributions and unchanged parameter layout. The
measured early-256p training rate is 878.91 samples/s; all-rank peak is
49.68 GiB. These are short-run measurements, not long-run stability guarantees.

Later-stage plans remain absent; accumulations 4/5 are candidates. These plans,
accumulation settings, and transitions must be qualified before those stages
run; they do not gate this pass or the 256p launch. Startup fails if the selected
plan is missing. The September 25 infrastructure pass is closed. The fresh
hero is now launched; see the [maintenance handoff](pretrain_hero_handoff.md). Exact
full-state/RNG restoration passed on all 16 ranks, followed by a matched
3,200-record replay, corrected prompt grids and clean shutdown. The local
suite passes 798 tests (six NPU skips); device-specific checks passed live.

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

The launcher fixes the qualified NPU allocator policy (expandable segments)
and one OpenMP thread per process before workers initialize. These execution
mechanisms are not recipe switches or inherited caller choices.

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

## September 25 launch

Fresh `ascend-hero-256p` runs on 16×910B2C after the pretraining audit fixes in
`8b4257d`. Platform job `ascend-hero-256p-0925-r3` uses the qualified HIGH
allocation and reports directly to [SwanLab](https://swanlab.cn/@mtrya/artflow/runs/i9r8wnah).
Its complete deployed config preserves all numerical settings above; only
storage/output/bucket artifact paths differ. The new output root prevents
resuming an older hero accidentally. Source/config hashes, actual progress,
logs, full-state recovery and later-stage duties are in the
[maintenance handoff](pretrain_hero_handoff.md).
