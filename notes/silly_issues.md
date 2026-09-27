# Silly issues — training and sampling audit, September 25, 2026

## Current validity — September 27, 2026

Rechecked the working tree based on `df1910b`, then fixed issue 6 in the
follow-up below. **Only issue 4 remains open.** The September 25 pretraining
disposition below records the earlier launch state.

| Issue | Current status | Verification |
|---|---|---|
| 1. Rounded sampling timesteps | Fixed | Euler and Heun construct FP32 model times, including Heun's corrector. |
| 2. BF16 ODE accumulation | Fixed | Both solvers promote low-precision state and velocities before integration; explicit FP64 state is preserved. |
| 3. Microbatch-dependent caption dropout | Fixed | No forced restoration; partition-invariance, singleton, all-dropped and real-tokenizer regressions pass. |
| 4. Standalone generation contract | Still valid | Reproduced missing `txt_pooled` with a tiny real ArtFlow in both CFG and non-CFG calls. A wrapper spy confirms no encoder-layer selection (the hero still selects layer 20). Toy-model probes reproduce endpoint CFG and conditioning-batch mismatches. |
| 5. Final encoder-layer early exit | Fixed | The final exit hooks Qwen's final norm; real small-Qwen training/full-forward equivalence tests pass at first, intermediate and final layers. |
| 6. Decode-helper dtype | Fixed September 27 | The helper now uses the VAE parameter's `.dtype` and moves/casts latents in one `.to(device=device, dtype=dtype)` call. Regression coverage uses a real Conv3d-backed stub decoder with FP32/BF16 inputs and weights. |

The two-step CFG probe gives `3.5*x_initial` instead of `4*x_initial` with
Euler, and `4.28125*x_initial` instead of `6.25*x_initial` with Heun. With
two prompts and two images per prompt, latent/text batch sizes are 4/2
without CFG and 8/4 with CFG. These probes isolate the pipeline logic using
mocked text features and toy velocities; they do not measure image quality.

Audit validation: `OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m pytest tests/test_solvers.py
tests/test_training_caption_dropout.py tests/test_fast_paths.py
tests/test_prompt_grid.py tests/test_kid_eval.py tests/test_time_shift.py
tests/test_timestep_factor.py -q` — **76 passed**, no skips, 36.68 seconds.
Additional CPU reproductions ran inline; no probe scripts were retained.
Training grids and KID still use their own wrappers, pass pooled text and
the configured exit layer, and decode directly with the VAE. The initial audit
did not rerun platform qualification or the complete test suite and changed
only this note.

Issue 6 follow-up: `tests/test_vae_codec.py` reproduces the dtype error in both
conversion directions before the fix (two failures, two matching-dtype passes).
It also checks batched, non-square RGB output and clipping to image range.
After the fix, all four regression cases pass (3.02 seconds).
The full suite with the same offline/thread environment passes outside the
sandbox: **820 passed, six NPU skips**, nine warnings, 44.99 seconds
(`.venv/bin/python -m pytest tests/ -q --tb=short`). The initial sandboxed
full-suite run reported failures and stalled; it was interrupted. Those
failures did not recur outside the sandbox. No platform qualification was run.
The implementation change is confined to the decode helper; the standalone
pipeline, training recipe and platform jobs are unchanged.

## Pretraining disposition — September 25

The launch follow-up fixed the issues that affect pretraining:

- **1–2, sampling precision:** fixed in `72b287c`. Euler/Heun use FP32
  timesteps and at least FP32 integration state; native NPU monitoring grids
  and full-state replay passed during the infra pass.
- **3, caption dropout:** removed forced caption restoration. Dropout is now
  independent per sample, including singleton and entirely dropped batches.
  Regression tests check exposure across micro-batch partitions, probabilities
  zero/one, and real-tokenizer empty prompts through the training encoder.
- **5, final-layer exit:** the final exit now hooks Qwen's final normalization;
  intermediate exits still hook decoder blocks. Real small-Qwen regressions
  compare training with full-forward evaluation at first, intermediate and
  final layers. The selected hero layer 20 of 28 is unchanged.
- **4, standalone generation:** deferred. Pretraining loss, prompt grids and
  KID use separate wrappers and do not call this pipeline.
- **6, unused decode helper:** deferred. The pretraining monitoring wrappers
  decode directly with the VAE; they do not call this helper.

Validation after these fixes: **806 passed, six NPU skips**, nine warnings,
43.65 seconds for the complete local suite. This establishes the regressions
above; it does not claim better generated images or long-run training stability.
The real Ascend launch preflight also passed: an empty prompt retained five
Qwen k20 tokens and the full hero DiT produced finite forward/backward results.
The [maintenance handoff](pretrain_hero_handoff.md) records the running hero.
The original findings below describe the reviewed revision.

Requested follow-up to [the timestep-factor investigation](archive/timestep_factor_0924.md).
Reviewed the working tree based on `4fa399b`, including the pre-existing local
infra edits. This is a review, not an implementation change. No model, recipe,
dataset or platform job was modified. CPU reproductions below use the local
PyTorch/Transformers installations; no image-quality improvement is claimed.

Scope: model/attention/conditioning, flow objective and solvers, pretraining
update/EMA/resume paths, caption sampling, latent preprocessing, evaluation,
standalone generation, and a scan of the post-training primitives. This is
not an exhaustive audit of every data-fetching or platform script.

## 1. Sampling rounds timesteps before factor-1000 embedding

`src/flow/solvers.py:28,39,44` constructs model timesteps with `dtype=x.dtype`.
Both `sample_prompt_images` and KID generation start with BF16 noise. For the
raw EMA model under autocast, predictions and the solver state remain BF16.
Consequently timesteps reach the embedder already rounded; converting them
back to FP32 inside `TimestepEmbeddings` cannot recover the lost precision.
Training samples its logit-normal times in FP32.

Direct call to the real Euler step: requested `t=0.9`, received
`0.8984375` (`torch.bfloat16`). After the fixed factor 1000 this is a
1.5625-radian phase error at the highest frequency.

For the 50-step Euler schedule, comparing the real h1152 embedder on exact
FP32 schedule values and their BF16 round trips gives:

| Square resolution | Maximum phase error, radians | RMS feature error over the schedule | Maximum per-time RMS feature error |
|---|---:|---:|---:|
| 256p | 1.87498 | 0.13527 | 0.28917 |
| 640p | 1.80286 | 0.10006 | 0.27960 |
| 896p | 1.82289 | 0.10475 | 0.28227 |

The embedding itself has RMS `sqrt(1/2)`. This is a concrete training/sampling
conditioning mismatch, present in the current EMA monitoring path, rather
than a proposed architecture change. The fixed-loss probe uses FP32 times and
does not have this particular error. Both Euler and Heun need FP32 model times.

## 2. Sampling also accumulates the ODE state in BF16

`src/flow/solvers.py:30,42,49` performs the update in the incoming tensor
dtype. With BF16 state and BF16 model output, sufficiently small increments
are rounded away on every step.

Reproduction through the real `sample_ode`, 50 Euler steps over `[0,1]`,
`x(0)=1`, constant velocity `0.1`:

| State dtype | Result | Analytical result |
|---|---:|---:|
| BF16 | 1.0 | 1.1 |
| FP32 | 1.0999987 | 1.1 |

This is independent of timestep quantization: the velocity is constant in
both time and state. Keep solver state and update arithmetic in FP32, while
allowing the model forward to use mixed precision. Cast to the VAE dtype at
decode. The magnitude of the effect on trained ArtFlow images is unmeasured.

## 3. Caption dropout depends on local microbatch size

`src/pretrain/train.py:939-940` forcibly restores one caption whenever the
entire local microbatch was dropped. For batch size one, dropout is therefore
impossible. With nominal probability `p` and local batch size `B`, the actual
per-sample dropout probability is `p - p**B / B`.

Executed the actual nested `select_captions` function extracted from the
trainer, with `p=0.1`, seed 42 and 10,000 batches per size:

| B | Observed dropout | Expected from this implementation |
|---|---:|---:|
| 1 | 0 | 0 |
| 2 | 0.0968 | 0.095 |
| 8 | 0.098825 | approximately 0.1 |

Changing memory-driven bucket batch sizes thus changes unconditional
training exposure. In particular, any B=1 bucket has none. The current hero
bucket artifacts are still awaiting qualification; this review does not
claim that its final plans contain B=1. Remove the forced restoration if the
recipe intends ordinary independent caption dropout. The empty prompt is
valid: the cached real Qwen tokenizer retains five chat suffix tokens, so an
all-dropped batch is not an all-masked attention problem.

## 4. Standalone generation has diverged from the training contract

These findings concern `src/pipeline/artflow_pipeline.py`; training grids and
KID use their own sampling wrappers.

- **Missing pooled text, line 376.** `__call__` discards `_encode_text`'s
  pooled output and calls the transformer without `txt_pooled`. A tiny real
  current `ArtFlow`, mocked text features and a two-step pipeline call fail
  immediately with `TypeError: ArtFlow.forward() missing 1 required positional
  argument: 'txt_pooled'`. Preserve and pass pooled conditioning in both CFG
  and non-CFG paths.
- **Wrong text layer, lines 420-422.** The pipeline calls `encode_text` with
  only `pooling=True`; a spy on that real wrapper confirmed the arguments.
  Its default selects final normalized Qwen features, whereas the hero uses
  layer 20 features before final normalization. Fixing the missing argument
  alone would still feed the DiT the wrong feature distribution. Recover the
  training text-encoder settings when loading and apply them at generation.
- **CFG combines final trajectories, lines 390-392.** The solver separately
  evolves conditional and unconditional states, then combines their endpoints.
  CFG requires combining velocities evaluated at the same current state at
  every solver evaluation. A permissive toy transformer isolates this from
  the missing-argument failure: for `v_cond(x)=x`, `v_uncond(x)=0`, guidance 2
  and two Euler steps, the actual pipeline returns `3.5*x_initial`; guided
  Euler returns `4*x_initial`. Combine the branch velocities inside `model_fn`
  and advance a single state, including at Heun predictor/corrector calls.

There is also a static batching mismatch: `num_images_per_prompt` expands
the noise batch at line 359 without repeating text features/masks to match.
Repair that alongside pooled-text batching; the default value one avoids it.

## 5. Early exit is not equivalent at the final encoder layer

`src/utils/encode_text.py:54-67` always captures the decoder block output.
For an intermediate layer this agrees with `hidden_states[k]`. At the final
layer, full-forward `hidden_states[-1]` includes Qwen's final RMSNorm, which
the exception-based early exit bypasses.

Reproduced with an actual, randomly initialized three-layer
`Qwen3ForCausalLM` (hidden width 32), using both real encoding helpers:

- Layer 2: bitwise identical output.
- Layer 3: full-forward RMS 0.99924 versus early-exit RMS 0.025887;
  maximum absolute difference 3.10935.

Those numerical values describe this small random instance, not trained Qwen.
The normalization mismatch is structural. Current k=20 of 28 avoids it, but
the schema permits selecting the final layer, making training and evaluation
silently disagree for that supported setting. Apply the final norm when
stopping at the last layer, or consistently define another explicit contract.

## 6. Decode helper assigns a device to its dtype variable

`src/utils/vae_codec.py:64` says
`dtype = next(model.parameters()).device`. Its subsequent `.to(dtype)` is
therefore a second device move, with no dtype conversion.

Passing FP32 latents to the actual helper with a BF16 Conv3d-backed stub VAE
reaches decode still in FP32 and raises
`RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same`.
Use the parameter's `.dtype`. This helper is not the decoder used by current
prompt grids or the standalone pipeline, so it does not explain their results.

## Checks and limits

- The original factor-1000 fix is present, applies only inside the embedder,
  and rejects incompatible factor state. Training time remains in `[0,1]`.
- The flow velocity target, integration direction, and resolution time-shift
  convention agree by inspection and existing tests.
- Checked the real cached Qwen tokenizer: the prompt prefix is exactly 38
  tokens, matching `DROP_IDX`, and padding is on the right.
- The reviewed gradient accumulation normalizes by global accumulated loss
  weight; EMA parameters remain FP32 in the current trainer. No additional
  factor-1000-like defect in the default training conditioning path was
  established by this review.
- Initial full-suite runs were interrupted after sandbox IPC failures/hangs.
  A direct `socket.socketpair().send(...)` raises `PermissionError(1,
  'Operation not permitted')` there. Rerunning with local IPC available:
  `OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/ -q` — **787 passed,
  4 skipped**, 9 warnings, 43.73 seconds. The initial failures did not recur.

No one-off reproduction scripts were added to the repository. Fixes above
need behavioral regression tests: the existing tests primarily exercise
FP32 solver inputs, intermediate encoder exits and isolated model components.
