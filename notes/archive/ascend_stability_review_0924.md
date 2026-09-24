# Remaining stability work after branch normalization — 2026-09-24

Archived pre-experiment assessment. The [completed Stage-1 evidence](ascend_stability_stage1_0924.md)
records the resulting decision. Apply the
[user's experiment policy](../ascend_pretraining_0924.md): maximize useful decisions
per unit of compute, combine justified changes, and use the hero itself for
longer validation.

## What is established

The accumulation-matched 2k-step [probe](../muon_weight_growth_0924.md) improves
evaluation loss from 0.92154 to 0.91259 and last-block activation RMS from
12001 to 73.58. This supports adopting `branch_norm`. It does not measure
the new model's response to an actual conditioning update or its future spike
rate. These activation probes use four captions at fixed `t=0.5`.

## Remaining risks and the decisions they imply

1. **Effective Muon update size deserves recalibration.** Our implementation
   multiplies the NS result by `0.2*sqrt(max(rows, cols))`. The
   [Keller Jordan reference](https://github.com/KellerJordan/Muon/blob/master/muon.py)
   uses `sqrt(max(1, rows/cols))` with default LR 0.02. For a 1152-square
   matrix, our update at the same LR is 6.79 times larger, holding the NS result
   fixed. LR 0.00295 in our convention matches that reference example.
   [Moonlight §2.2](https://arxiv.org/html/2502.16982v1) introduced our scaling
   to reuse AdamW-like LRs. [CMuon Table 8](https://arxiv.org/html/2608.02502v1)
   uses the same base scaling with LRs 2e-4–3e-4, with an additional chunk
   rescaling absent here. Our code's claim that RMS matching justifies LRs
   0.01–0.05 is unsupported. This is a convention mismatch worth investigating,
   not proof of an optimizer implementation bug: our Stage-2 LR 0.02 also has
   positive 16k-step quality evidence. Proposed first candidate: Muon LR 0.003.

2. **The conditioning-update mechanism is not yet checked after branch norm.**
   The [H200 replay](../h200_spike_root_cause_0924.md) causally isolated the
   shared displacement `delta_W @ mean(h)` from the final conditioning linear
   layer. Branch norm is before the learned gates, so gates and affine norm
   gains remain possible amplification paths. Aggregate modulation RMS rises
   0.735→12.74; this does not identify which component grew or establish harm.
   Measure separate shift/scale/gate outputs, the gated branch contribution,
   and fixed-batch loss/gradient response to actual conditioner updates.
   Proposed auxiliary AdamW LR: 1e-4, supported as a starting prior by
   [Scaling Muon for DiTs, Table 5](https://arxiv.org/html/2608.20818v3).
   A smaller step reduces the demonstrated displacement but is not a structural
   cure if sensitivity continues growing.

3. **Weight growth and SiLU saturation need functional interpretation.**
   Branch norm leaves weight RMS higher (0.556→0.585) and increases the fraction
   of conditioning preactivations below -5 (55.5%→66.6%), while hidden
   cross-caption variation/RMS improves (0.00922→0.01562) and loss improves.
   Neither raw statistic is therefore a sufficient failure criterion.
   `muon_wd` is already 0.01. Weight decay is supported by Moonlight; the DiT
   study above uses 0.01, so the literature does not uniquely prescribe 0.1.
   Inspect update/weight alignment and the decay contribution before increasing
   decay solely to lower weight RMS. Keep 0.01 in the first candidate.

4. **Time-feature bandwidth is a separate concern.** Recommend factor 1000
   for a fresh model, revising the earlier factor-1 recommendation:
   [official FLUX](https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py)
   explicitly uses it. The
   [Diffusers flow scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)
   likewise defaults to a 1000-unit timestep convention. The user's tf1000
   result establishes that it is insufficient to prevent spikes alone.
   It changes temporal frequencies, not embedding norm, and does not solve
   the conditioning-update mechanism. Serialize the factor and share it across
   training/evaluation; do not change it on an existing factor-1 checkpoint.

## Initial check within Stage 1

The user's subsequent clarification sets four stages: stability experiments,
repository/config/docs cleanup, an infrastructure pass, and then the hero.
The replay below is an initial diagnostic, not a direct hero-launch gate.
See the [completed Stage-1 experiment record](ascend_stability_stage1_0924.md).

Reuse a retained branch-norm checkpoint and optimizer state for a bounded
update replay. On a few representative fixed batches/timesteps, compare the
current update with the proposed Muon/AdamW step sizes, including cheap
optimizer-group counterfactuals. Collect the displacement, gate/contribution,
loss/gradient response, and radial weight-update measurements above in the
same pass. Include steps larger than the proposed step to assess local margin.
These are immediate-response measurements, not a forecast of all future steps.

This check informs whether Muon recalibration helps locally, whether the
conditioner still dominates sensitivity, and whether stronger decay has a
measured target. Follow it with short instrumented training experiments that
test the resulting decisions. The proposed fresh-model settings remain
candidates. If conditioner sensitivity remains severe, address its
parameterization before merely extending warmup. Complete the user's cleanup
and infrastructure stages before production.
