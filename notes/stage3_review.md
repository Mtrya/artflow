# Stage 3 review follow-up — 2026-09-09

## Sampling balance

Measured analytically from the existing per-caption sidecars using
`python -m scripts.bench.shard_balance --workroot <artflow-root>`.
The calculation uses the Stage-3 mix, exact within-row curriculum probabilities,
and the gate's per-length batch sizes. It estimates steady-state sample rates;
bounded incomplete queue tails and finite-window randomness are excluded.

| Ranks | Curriculum | Largest / smallest expected samples per micro-batch |
| --- | --- | --- |
| 4 | 0.0 | 1.000166823 |
| 4 | 0.5 | 1.000157176 |
| 4 | 1.0 | 1.000147529 |
| 8 | 0.0 | 1.000250852 |
| 8 | 0.5 | 1.000242192 |
| 8 | 1.0 | 1.000235416 |

The theoretical fixed-shard bias exists, but the measured length-driven rate
spread is at most 0.017% on four ranks and 0.025% on eight. No sampler change
is justified for this corpus on that evidence. Independent full-pool row
sampling on each rank is a simple alternative if a future corpus has severe
shard imbalance, with the tradeoff of possible cross-rank duplicate draws.

## Implemented changes

- Throughput timing waits for optimizer and EMA CUDA work to complete and
  includes telemetry/logging overhead. Only evaluation/checkpoint intervals
  are excluded from the final summary.
- Sidecar cache identity uses saved dataset state, shard size/mtime, tokenizer
  identity and local tokenizer file size/mtime. Routine checks never scan
  Arrow contents or model weights. Legacy sidecars rebuild once. This is an
  accidental-staleness check, not protection against stat-preserving edits or
  changes to a remote tokenizer revision under the same model ID.
- Attention tests initialize nonzero output/modulation weights, assert nonzero
  attention gradients, and compare both forward outputs and parameter
  gradients for masked and unmasked inputs. Duplicate test definitions removed.
- Final-layer text early-exit behavior is unchanged at the user's request.

Local validation: `.venv/bin/python -m pytest -q` — 178 passed, 1 skipped,
21 existing warnings; whitespace check and benchmark-script compilation passed.

## Corrected throughput rerun

Prepared by `scripts/bench/gate_ab.py`: sequential baseline/full-stack arms on
the same GPU allocation, 400 optimizer steps each, fixed 512-sample eval-loss
probes every 50 steps, first 50 steps excluded from steady-state throughput.
Both arms use the same current base config and row/caption sampler seed.
The baseline explicitly restores full text forward and disables all fast paths.

Inspire job: `s3-review-ab-1g-0909`, succeeded 2026-09-09 15:11:42 CST,
one RTX 4090 (48 GB), node `qb-prod-4090-gpu170`. The queued four-GPU job
`s3-review-ab-4g-0909` was stopped following the user's decision to review
Stage 3 using the single-GPU result. No corrected multi-GPU scaling claim is made.

| Arm | Actual samples | Training seconds | Samples/s, all steps | Samples/s, steady | Peak allocated GB |
| --- | --- | --- | --- | --- | --- |
| Baseline | 102,208 | 1,854.6 | 55.11 | 54.99 | 25.2 |
| Optimized | 102,208 | 1,504.3 | 67.94 | 72.98 | 17.8 |
| Speedup | | | 1.233× | **1.327×** | |

Both arms average 255.52 actual samples/optimizer step. The **steady-state
1.25× throughput gate passes**. All-step throughput, which includes compile
warm-up, does not pass 1.25×; it is reported to make the cost of short runs
visible. Dataset metadata preparation and model startup are outside the
training-loop timer; evaluation/checkpoint time is also excluded, so neither
throughput figure is total submitted-job wall time.

### Accuracy review

| Optimizer step | Baseline eval/loss | Optimized eval/loss | Relative difference |
| --- | --- | --- | --- |
| 50 | 2.00067 | 2.00067 | 0.000% |
| 100 | 1.98088 | 1.98087 | −0.001% |
| 150 | 1.93909 | 1.93904 | −0.003% |
| 200 | 1.87264 | 1.87222 | −0.022% |
| 250 | 1.79136 | 1.79142 | +0.003% |
| 300 | 1.72450 | 1.72505 | +0.032% |
| 350 | 1.68897 | 1.68925 | +0.017% |
| 400 | 1.68437 | 1.68865 | **+0.254%** |

The largest overall-loss difference is at the last probe. Timestep-stratified
losses at that point range from −0.066% (`t090`) to +0.497% (`t040`). There is
no divergent trajectory in this short run, but the late positive difference
should not be described as bitwise equivalence or as proven harmless in long
training. In conjunction with the meaningful forward/backward tests and earlier
matched A/B trajectories (including cases where the compiled arm was better),
the accuracy evidence is accepted for **Stage-3 infrastructure closure**.
This is a review judgment, not a newly invented statistical tolerance or a
long-run non-inferiority test. Later quality probes still need the fixed eval set.

### Reproduction and artifacts

- Runner: `scripts/bench/gate_ab.py`; base configuration: `configs/base.toml`.
  Command from the uploaded source root:
  `python -m scripts.bench.gate_ab --workroot <artflow-root> --name <fresh-name> --gpus 1 --steps 400`.
- Source base: commit `c496da063b29da61b2548e364cda379d62439542`, with the
  review fixes recorded here. Uploaded archive SHA-256:
  `76f6fa311d66bcde031b330cfa399d91b6bae9b421c3dcd8aa305812cb25921d`.
- Shared-disk artifacts: `<artflow-root>/runs/stage3/s3-review-ab-1g-0909/`
  contains `gate.toml`, `baseline.log`, `fullstack.log`, `result.json`, and
  per-arm checkpoints/runtime configuration. The isolated source is under
  `<artflow-root>/tmp_artifacts/review-0909/`.
- [Structured results](stage3_review_result.json) preserve the summary and all
  printed overall/timestep eval probes in the repository. Loss precision is
  that of the printed logs; ratios use the printed throughput summaries.

The old gate's end-to-end timing and scaling figures are historical, not
revalidated by this run. Stage 3.5 and Stage 4 must measure their new caption
distribution, resolutions, and target GPU topology before committing a hero
sample budget.
