# Timestep-embedding factor — a structural lead upstream of the spike chain

> Archived record. Current pretraining decisions and status are in
> [the Ascend plan](../ascend_pretraining_0924.md); old launch commands and
> configuration switches require the source revision used for that run.

2026-09-24. Companion to [H200 spike root-cause investigation](h200_spike_root_cause_0924.md).
That document causally isolates the triggering **update** (AdamW's update to
`c_mlp.2`, amplified 144× into a shared conditioning bias shift). This note
records what sits **upstream** of it, plus the code facts needed to act on it.

Where the two agree: that document already flagged the untested lead
"the current sinusoidal time embedding receives t in [0,1] without a
frequency multiplier; FLUX uses `time_factor=1000`" and measured the
consequences without a cause (t-branch mean-vector norm 19.78 → 163.40;
all 709×1152 hidden preactivations negative by 10k; `c` centered RMS
0.25 → 0.20 contracted while modulation spectral norm grew 66 → 121).
The measurements below supply the cause.

## 1. The finding

`src/models/dit_blocks.py:130` `TimestepEmbeddings.forward`:

```python
emb = torch.exp(exponent)                                 # freqs 1 … 1/10000
emb = timesteps.unsqueeze(1).float() * emb.unsqueeze(0)   # t multiplied raw
```

Our `t ∈ [0, 1]` is the flow-matching path parameter (`x_t = (1-t)ε + t·x_data`,
`src/flow/paths.py`). Canonical sinusoidal embeddings multiply *integer*
timesteps in [0, 1000]; FLUX, which also uses t ∈ [0, 1], applies an explicit
`time_factor = 1000`. **We do neither**, so the phase argument spans ~1 radian
instead of ~1000.

Measured on a freshly initialised `TimestepEmbeddings` (hidden 1152, t on 21
uniform points over [0,1]) — pure function, no weights involved:

| | per-dim span over t (mean) | dims with span > 0.5 |
|---|---:|---:|
| as-is | 0.0649 | **41 / 1152 (3.6 %)** |
| `× 1000` | 1.4041 | 894 / 1152 (77.6 %) |

**96.4 % of the embedding is constant with respect to t.** The model has 41
usable dimensions to encode "how noisy is this sample". Reproduced
independently by two runs of different code (`scripts/diag/layer_probe.py`
author's check and the note author's).

This is architecture, not training: it holds for a randomly initialised
embedder, before any weight is touched. It also cannot be fixed by more
steps, a different lr, or a different accelerator.

## 2. Why it starts the chain

A conditioning input that barely varies with t forces the downstream
conditioning MLP to manufacture variation. `c_mlp` has finite capacity to do
so, and the recorded evidence shows it failing in a specific, self-reinforcing
way: the hidden units drift into the negative saturated tail of SiLU
(54 % below −5 at 12k, 21 % sitting within 0.1 of SiLU's minimum), which makes
`c` nearly constant across samples (centered RMS 0.20 against mean norm 17).
Once `c` is a near-constant shared offset:

- `grad_W[i,j] ≈ grad_bias[i] · h̄[j]` with `h̄` almost constant across samples,
- Adam's coordinate-wise normalization removes the magnitude of `h̄[j]`, so
  many columns push in the same direction,
- the resulting `ΔW · h̄` is equivalent to a bias shift shared by every block
  (**144×** the explicit bias update),
- and the downstream modulation path is sharply sensitive in that direction
  (curvature **5130×** the 12k state).

One update then moves the conditioning vector by 0.206 RMS while the entire
between-sample spread of conditioning signals is only 0.196 — the step has to
cross the cliff. This is the mechanism the companion note proves.

## 3. Change footprint — small, but not free

- `TimestepEmbeddings` has exactly **one** instantiation (`artflow.py:79`) and
  one call site (`artflow.py:210`). No second implementation exists anywhere
  (`train.py`'s sin/cos is the cosine LR scheduler).
- Inference and eval go through the same class
  (`pipeline/artflow_pipeline.py:216` builds `ArtFlow(**config)`;
  `evaluation/eval_loss.py` calls the same model), so one change propagates
  consistently. No separate training/inference edits.

Constraints:

1. **Every existing checkpoint becomes incompatible.** All hero7 (18 ckpts),
   H200 and Ascend weights had `t_embedder`... — actually `t_embedder` is
   parameter-free; the weights that adapt to this input are `c_mlp` and the
   modulation MLPs. Either way, changing the input distribution invalidates
   the trained conditioning path unless the change is made from a fresh
   trajectory. So the factor must be an **explicit construct-time parameter
   defaulting to 1.0**, with new runs opting in.
2. **The factor must be applied at the embedder entrance only.** `t` also
   drives the interpolation `z_t`, the loss weighting and `shift_timesteps`;
   scaling it anywhere upstream would change the training objective.
3. Registering it as a buffer (so it enters `state_dict`) makes checkpoint and
   code self-consistent and avoids silently loading weights under the wrong
   factor.

## 4. Ruling out the alternatives (code facts)

**Residual scaling: absent.** No `res_scale` / depth-scaling / `1/sqrt(2N)`
anywhere. The residual is bare `x = x + gate·out` (`dit_blocks.py:972`, `:983`).

**qk-norm: present.** `q_norm`/`k_norm` RMSNorm on head dim in every attention
(`dit_blocks.py:465-468`, `:764-765`, `:1009-1010`) — the one SD3-style
stabilizer we do have.

**Sandwich norm: absent.** Only pre-norm; attention and MLP outputs enter the
residual without a post-norm.

**Configuration-side causes are ruled out** (measured by
`layer_probe.py` across hero7 ckpts 2000→18000): trunk activations grow
**exponentially with training step** — every 2000 steps roughly doubles
(332 → 1596 → 4157 → 8372 → 1.77e4 for blocks 12–22), reaching **1433×** by
12k, while `model_output` stays at 5.1–5.4. So the network is internally
reparameterised: output preserved, activations three orders of magnitude
larger. That is the macroscopic face of the same contraction-plus-gain story.
The same growth appears in an independent H200 trajectory replayed on a 4090
(12000→16000, 13.5×), so it is a property of the recipe, not of any
accelerator.

## 5. Code diff: 0918 snapshot vs now — live update math is unchanged

The 4090 hero run used the `repo-0918` snapshot (pre-`src/pretrain` rename).
Comparing it against the current tree, nothing that changes the **live weight
update** differs:

| file | difference | effect |
|---|---|---|
| `models/artflow.py` | adds `view_as_real(rope_freqs)` | mathematically equivalent (torchair has no complex dtype) |
| `models/dit_blocks.py` | NPU `npu_fusion_attention`, real-rope freq layout | mathematically equivalent (execution path) |
| `pretrain/muon.py` | `_NS_DTYPE` switch (default bf16), `fused_adamw` (default off) | default behaviour identical |
| `pretrain/train.py` | EMA decay → `ema_decay_at(..., warmup)`; adds `grad_spike_skip`; `npu_fused_adamw`; resume re-applies config `base_lrs` | EMA does not feed gradients; others default off |
| `configs/hero.toml` | adds `grad_spike_skip = 10.0` | not numerical |

**Consequence:** "the 4090 run was fine" cannot be explained by code. It is
also not a solid premise — that run's checkpoint carried a lagging EMA
(gradient-free, smoothing), so it had no observation point on the live
weights. A recipe-level defect would look exactly like this from the outside.

## 6. Experiments

- **Level 1 (forward only, no training)**: `layer_probe.py --timestep-factor`
  compares, at fixed weights and a fixed batch, the `c_mlp` hidden saturation
  (`frac_negative`, `frac_lt_neg5`) and `c` statistics between factor 1 and
  1000. Caveat: weights trained under one factor are mismatched to the other,
  so this shows what the input change *is*, not what a retrained model
  *becomes*. Use `c_mlp.0` (Linear, unbounded) for the saturation statistic —
  on the SiLU output a fully saturated hidden can report `frac_negative = 0.0`
  because `sigmoid(-100)` underflows to `-0.0`.
- **Level 2 (short training)**: `ascend-tf1000-256p`, 16×910B, patched copy of
  the code (`repo-tf1000`, so the shared `repo-ascend` stays clean), recipe
  identical to hero7 (MUON_LR 0.02, ADAM_LR 3e-4, warmup 20000, accum 1,
  ascend-0922 plan, eager) except `t × 1000`, stopping at step 5000 with
  checkpoints every 1000. Compare against hero7's ckpt-2000/4000 on the same
  probe: t-branch output norm, hidden saturation, and the blocks 12–22 scale
  slope. Level 2 is the only one that can answer whether the spike disappears.

Status: level 2 launched 2026-09-24 12:24 CST and training normally
(loss 2.02 → 1.52 by step 175, grad_norm 0.9 → 0.65, ~870 samples/s), i.e.
the patch is not destructive. Level 1 in progress.

### Level 1 results (measured)

`layer_probe.py --timestep-factor` on hero7 checkpoints, fixed 4-row batch,
t = 0.5, `c_mlp.0` = the first conditioning Linear (unbounded, so a valid
saturation statistic; on the SiLU output a fully saturated hidden can report
`frac_negative = 0.0` because `sigmoid(-100)` underflows to `-0.0`):

| ckpt step (factor 1, training path) | `c_mlp.0` frac_negative | frac_lt_neg5 | `c_mlp.1` absmax | blocks.12 img absmax |
|---|---:|---:|---:|---:|
| 2000 | 0.929 | 0.157 | 4.500 | 244 |
| 4000 | 0.974 | 0.370 | 2.922 | 1968 |
| 12000 | **1.000** | 0.395 | **0.279** | 5.53e5 |

Two things to read off this. First, it independently reproduces the companion
note's "all hidden preactivations negative by 10k" and dates its onset: the
conditioning hidden is already **92.9 % negative at step 2000**, climbing
monotonically to 100 %. Second, `c_mlp.1` (SiLU output) absmax **collapses**
4.50 → 0.279, which is the signature of drifting into SiLU's negative
saturated tail. Trunk scale (right column) grows on the same schedule, so
contraction and expansion advance together rather than one causing the other
after the fact.

Same weights, factor swapped (ckpt-12000):

| factor | `c_mlp.0` neg | frac_lt_neg5 | `c_mlp.0` absmax | `c_mlp.1` absmax | blocks.12 img absmax |
|---|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.395 | 14.250 | 0.279 | 5.53e5 |
| 1000 | **0.896** | **0.132** | 8.562 | **0.988** | 2.654e6 |

Feeding the same weights a t-sensitive embedding **immediately** pulls the
hidden out of deep saturation: `frac_lt_neg5` drops 3× (0.395 → 0.132) and the
SiLU output amplitude recovers 3.5×. That is directional evidence for the
causal step "weak t signal → conditioning hidden saturates".

It does **not** show the recipe improves, and the last column says why: with
weights trained under factor 1, factor 1000 *raises* the trunk scale
(5.53e5 → 2.65e6). Mismatched weights make the conditioning output move
somewhere the rest of the network was never trained for. Only a retrained
model answers the real question — hence level 2.
