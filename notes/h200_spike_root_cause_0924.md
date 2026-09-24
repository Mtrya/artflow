# H200 spike root-cause investigation — 2026-09-24

The user requested a mechanism, rather than further LR reductions or a
hardware switch. The H200 hero remains stopped; independent Ascend jobs are
not modified. Launch, incident, and trajectory-replay history:
[H200 operational record](h200_smoke_0923.md).

## Same-state numerical and cross-card test (09:39 CST)

An observer captured the first >10 gradient event in a replay from the
protected 12k checkpoint, before optimizer/EMA update: step **12,829**,
loss **0.973399**, gradient norm **41.212498**, 709 samples across four
ranks and three micro-batches per rank. All four complete model states,
parameter hashes, and clipped-gradient hashes are identical. The capture
contains exact DiT inputs, frozen text features, outputs, weights and
gradients; it is **not a resumable checkpoint** (no optimizer/RNG state).

The same weights and all 709 inputs were replayed without optimizer updates:

| Device / execution | Loss | Gradient norm | Cosine to captured gradient |
| --- | ---: | ---: | ---: |
| H200 compiled bf16 | 0.973478 | 41.166180 | 0.999970 |
| H200 eager bf16 | 0.974043 | 41.756997 | 0.999869 |
| H200 math attention bf16 | 0.974089 | 41.646670 | 0.999884 |
| H200 math attention fp32, TF32 off | 0.972703 | 41.709637 | 0.999695 |
| RTX 4090 eager bf16 | 0.975249 | 41.758911 | 0.999827 |
| RTX 4090 math attention fp32, TF32 off | 0.972703 | 41.709521 | 0.999695 |

The spike is a large mathematical gradient at this model state. Immediate
bf16 arithmetic, compilation, distributed reduction, or Ascend-specific
kernels are not required. The fp32 results agree across cards to ~0.0003%
in gradient norm. This does **not** exclude small numerical differences
earlier in training moving the trajectory into this state at different
steps. Serial accumulation and chunking change reduction order; bitwise
identity to the original DDP backward was not expected.

In fp32, `c_mlp.2.weight` norm is **39.715924**, accounting for **90.66%**
of squared total gradient norm. Next are `txt_pooled_proj.weight` (9.600027),
`c_mlp.2.bias` (6.372995), and `c_mlp.0.weight` (5.274025). All four belong
to auxiliary AdamW. This localizes the gradient; it does not establish why
that path became sensitive. The earlier full-fp32 Muon NS trajectory arm
also spiked and produced EMA texture collapse, rejecting NS promotion as a
sufficient fix.

Jobs `h200-spike-replay-0924` and `h200-spike-reference-4090-0924` both
succeeded (09:31:55 / 09:33:50 CST). Shared artifacts under
`$W/runs/h200-hero-256p-investigation-10282`:

- `capture-0924/capture/event_012829`: raw captured tensors and identities.
- `spike-replay-0924` / `spike-replay-4090-0924`: per-mode results including
  all parameter gradient comparisons, micro-batch losses and logs.
- `numerical-replay-summary.json`: combined numerical results.

## Conditioning-path investigation (running)

`h200-condition-path-0924` replays the same captured inputs with live weights
from 10k, 12k, spike 12,829, and original 14k, one H200 per state. All use
eager fp32 math attention, TF32 off, no optimizer updates. It measures:

- Input activation `h`, output `c`, and backward gradient at the final
  shared-conditioning linear layer. For each sample, its weight-gradient
  contribution has norm `||h|| * ||dL/dc||` (with actual global weighting).
- Each block modulation branch's separate contribution to `dL/dc`, checked
  against their sum; modulation component scales.
- LayerNorm / QK RMSNorm input scales and local backward amplification.
  Identity clones isolate each normalization branch so residual gradients
  are not incorrectly counted as its Jacobian amplification.

Comparison outputs: `condition-path-0924/<state>/result.json`. These are
observations, not proposed recipe changes. The next intervention depends
on the measured source of amplification.
