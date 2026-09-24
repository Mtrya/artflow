# Stage 4 — Fixed qualitative monitoring panel

Status: 2026-09-14. Implemented monitoring specification for the fixed hero recipe.
There is no numerical capability target, formal confirmation suite, or
capability-based launch gate. The panel detects implementation failures and
regressions; it does not forecast the trained model's eventual quality.

## 1. Frozen panel

[hero_monitor_v1.jsonl](../assets/eval/hero_monitor_v1.jsonl) contains 48 authored
prompts: 12 scenes × Chinese/English × short/long. Each scene has one explicit
seed shared by all four variants and by every checkpoint. Short and long
versions retain the same core scene; the long version adds observable details.
These are authored monitoring prompts, not captions selected from training rows.
They are not an independent statistical benchmark.

| Category | Scenes |
| --- | --- |
| Full-body figures | Old man in a stream; baker in a doorway |
| Hand–object interaction | Two-handed teacup hold; reading an open book |
| Faces | Gongbi woman; impressionistic man |
| Architecture | Waterside pavilion; stone church |
| Layout | Jug/apples/flowers/cloth; foreground boat, middle-ground bridge, distant mountains |
| Style-focused scenes | Ink-wash pines; impressionistic flowers |

The six styles are ink wash, light-color ink, gongbi, representational oil,
watercolor, and impressionistic oil. Aspect ratios are 1:1, 3:4, and 4:3, mapped
to the nearest actual evaluation-data bucket at each resolution.

The 24 short prompts have 29–49 retained tokens. Four scenes (eight bilingual
long prompts) occupy each band: 256–512, 512–1024, and 1024–2048. Their measured
ranges are 302–322, 530–582, and 1066–1223 respectively. Lengths include the
training prompt wrapper and retained-token convention, not a word-count proxy.
They were checked with Qwen3-0.6B tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`; none reaches the 2048-token cap.
The tails exercise the lower portion of the longest band, not its ceiling.

Recheck a local tokenizer without downloading or generating anything:

```bash
python -m scripts.eval.validate_monitor_panel --tokenizer /path/to/Qwen3-0.6B
```

The manifest records full text, scene/category/style, language/variant, aspect,
seed, target length band, and actual retained length. Freeze it with the run's
code revision; do not replace prompts or seeds after inspecting outputs.
The older `prompts_v1.jsonl` remains for reproducing earlier experiments.

## 2. Rendering and cadence

Hero configs select the new panel. SwanLab keeps the `grid/` namespace, with
one scene grid per aspect bucket: Chinese left, English right, short on the top
row and long on the bottom row. Captions identify each cell and its seed.
Grid PNGs and a `samples/panel_step_<step>.json` record preserve full prompts,
resolved seeds/shapes, global step, weights policy, and sampling settings.

Use EMA weights, Euler with 50 steps, bf16 sampling, no CFG (scale 1), and the
training text encoder's selected exit layer and pooling setting. Sampling uses
the shared resolution-aware time-shift path. The same seed fixes starting noise
within a resolution; it does not promise bitwise-identical output across hardware,
batch shapes, or resolutions. The model revision and resolved training config
identify the checkpoint and encoder/VAE artifacts alongside the panel record.

Render every 10k global updates, at stage endpoints, at the incoming checkpoint
before updates at each new resolution, and after its first 2k updates.
`[eval].grid_steps` adds absolute global-step triggers to the regular cadence;
overlaps render once per loop step. An explicitly listed startup/resume step
renders before training, and a crash resume does not reset the +2k trigger.
The hero launcher derives these steps from the final global total T, not a
hard-coded 400k total. Same-step retries can overwrite the same deterministic
panel file. Evaluation preserves the training Torch RNG stream.

Stage stopping and predecessor/T validation are implemented separately from
image scheduling. The launcher sets `stop_at_step` and endpoints are saved before
evaluation. Their distributed smoke test still belongs to the final infrastructure
pass; the next resolution is a separate launch. Include the doubled panel size
and transition checks in the measured all-in budget; training throughput alone
does not account for this overhead.

## 3. Review and response

Review full images and relevant detail crops for missing content, anatomy and
geometry errors, hand contact, faces, style, and basic layout. Compare each scene
with its own earlier outputs, including short-versus-long and Chinese-versus-English
behavior. Use improved / unchanged / regressed / uncertain notes, with concrete
examples, not an aggregate pass rate. Judge structure in a way appropriate to
each painting style; missing or unassessable features are not correct features.
VLM confidence and automatic similarity scores are not ground truth. Escalate
consequential uncertainty or systematic disagreement to the user.

Compare the incoming new-resolution baseline with subsequent checkpoints at
that resolution. A raw loss change across resolutions is not by itself a
regression. Preserve review notes and uncertainty flags with the images.
Follow [hero_recipe.md](archive/cuda_hero_recipe_0924.md)'s operational stop/review rules.

No thousands-image confirmation, minimum pass rate, confidence-bound gate,
judge-calibration quota, or per-capability scaling fit is required. Loss/KID and
these panels are diagnostics. Final reports describe observed strengths and
limitations without claiming a statistical capability target has been reached.
