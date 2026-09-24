# ArtFlow Reboot — Redesign Plan

Personal side project. Model ≤0.7B params. **Compute-frugal by design**:
provisional budget ≈ **2.9–3.7K RTX4090-hours** on Inspire.
Wall-clock time and queueing are the practical constraints. Reference point: the old hero run was ~800 RTX4090-h
(256p only, unoptimized stack, 19.2M samples seen). The former ~60–100M-sample
projection inside ~2K 4090-h was a **preimplementation estimate, not a measured
feasibility result**. Stage 4 must derive the hero recipe from the actual available
budget, corrected Stage-3 throughput, and measurements on the Stage-3.5 data and
bucket plans. The provisional ledger below also needs allowances for Stages 3.5
and 7 before it becomes an approved complete budget.

Organized as a linear pipeline of stages, each with goal / tasks / exit criteria /
compute cap. A follow-up agent should be able to execute stage by stage from this file
plus `notes/dataset_plan.md` (data-source detail).

Current pretraining direction (user, 2026-09-24): **pure Ascend pretraining**.
This pivot happens on **`main`**, with no separate Ascend pretraining branch.
Further pretraining debugging and optimization target the Ascend stack;
the earlier native-CUDA/H200 production direction is superseded. See
[decision and evidence boundaries](ascend_pretraining_0924.md). Use literature
priors and short experiments that change decisions. The current sequence is
**stability experiments and telemetry → repository/config/documentation cleanup
→ infrastructure pass on the selected recipe → hero**. The short stability
stage passed on September 24; repository/config/docs cleanup passed local
checks on September 25. Infrastructure optimization and qualification is next:
target approximately 2× speedup over the whole curriculum, with focused
experiments and an evidence-driven stopping rule, not a fixed time or compute
budget. See the [agreed infra goal and measurement policy](infra_pass.md#goal-and-stopping-rule--agreed-september-25-2026).
The [evidence and selected recipe](archive/ascend_stability_stage1_0924.md) do not
trigger an immediate hero launch. Do not require
an exhaustive candidate A/B matrix or a separate long-validation run.

## Locked decisions

| # | Decision | Choice |
|---|---|---|
| D1 | License | Research-only OK (WikiArt, ArtBench-10, FFHQ unlocked). Per-sample `license` field; NC data in separate mix entries so a clean variant stays one mix-string away |
| D2 | Anatomy data | Photos + paintings both; ~50/50 face vs full-body |
| D3 | Corpus size | Fixed eligible pools per resolution; counts and source mixtures in [hero recipe](archive/cuda_hero_recipe_0924.md) |
| D4 | Hero model | **532,766,716 parameters: h1152, 16 heads, 1 double-stream + 24 single-stream blocks**, branch normalization and timestep factor 1000 selected after the September 24 stability checks; [current decision](ascend_pretraining_0924.md) |
| D5 | Text encoder | Qwen3-0.6B, frozen, online; early-exit layer ablated k ∈ {8,16,28} + follow-up {20,24} → **k=20 (user verdict 2026-09-07**, 2.3a-followup) |
| D6 | Resolution curriculum | **256p → 640p → 896p**, variable aspect at every stage; **75:20:5** of final optimizer steps; **1024p definitively dropped** (final recipe decision, 2026-09-14) |
| D7 | RoPE | **Centered image grid + text pinned to fixed diagonal** (2.1 resolved 2026-09-05: 256p eval/loss+KID tie, 640p transfer tie — both arms collapse identically at 2.5× — 480p/384p/320p ladder tie → final tie-break on Qwen-Image adoption prior). Zero-shot ≥1.875× transfer fails for both variants → progressive staging mandatory |
| D8 | Inspire home | Account-selected project (machine-local configuration) |
| D9 | Compute class | **Ascend for pretraining** (user pivot, 2026-09-24); debugging and optimization focus on the Ascend stack. Current probes use 16×910B. Select the hero recipe from literature, existing evidence and short checks; the hero itself is the longer validation. The earlier 4×H200 and RTX 4090 pretraining paths are historical references. See [Ascend decision](ascend_pretraining_0924.md) |
| D10 | VLM captioning | **Via API** (Qwen-VL-class), not self-hosted — caption cost is money + rate limits, not GPU-hours. No GPU-with-internet workspace needed |
| D11 | Modulation | **Independent double-stream modulation; shared per-layer single-stream modulation** — 2.2a resolved 2026-09-05: layer wins eval/loss@end (0.9437 vs 0.9441) with a persistent t040 advantage (5/5 probes from 3K, -0.0002→-0.0010), KID agrees (0.0190 vs 0.0195), +0.6% faster, -8% peak mem; tie-break prior (PixArt/DiT-Air) points the same way. The selected 1+24 model uses independent image/text and attention/MLP modulation in its double-stream block, and shared attention/MLP modulation in single-stream blocks |
| D12 | Optimizer | **Muon (chunked orthogonalization, original scaling), LR 0.02 / auxiliary AdamW 1e-4**, Muon decay 0.0015 / AdamW decay 0.01, per the user's September 24 convention change. The [short stability checks](archive/ascend_stability_stage1_0924.md) used RMS matching at LR 0.003; those results do not directly validate every aspect of the changed scaling and decay. See [current decision](ascend_pretraining_0924.md). Historical Stage-2 RMS-matched LR 0.02 won at 16k (eval/loss 0.91127 vs AdamW 0.92134, KID 0.00699 vs 0.00922). |
| D13 | Quality monitoring | Small bilingual panels inspect anatomy, architecture, style and layout; no numerical capability target or forecast gate. Report observed strengths and limitations; [Stage-4 plan](archive/stage4_plan.md) |
| D14 | Inference scope | No prescribed inference device, VRAM ceiling, or latency launch gate; use reproducible evaluation settings. Deployment optimization and distillation are optional later work |

## Design dimension ledger (2026-09-04, agreed with user)

Which architecture/training choices are fixed by literature vs decided by stage-2
experiments. Stage-2 arms below implement column C.

### A. Literature-locked (no ablation)

| Dimension | Choice | Anchor |
|---|---|---|
| Objective | rectified flow + logit-normal(0,1) + resolution time shift | SD3 (Esser et al. 2024); FLUX/Qwen-Image follow |
| AdaLN-zero init | on | DiT; universal, already in code |
| QK-RMSNorm | on | SD3/FLUX and everything since; already in code |
| FFN | gated SiLU, ratio 8/3 (width 3072 at h1152; aligned September 25) | measured Ascend gain; see [infra pass](infra_pass.md) |
| Patch size | 2 | DiT-XL/2, SD3, Qwen-Image |
| Pooled text in AdaLN | **fused** (old runs used `pure` — flip default for all stage-2 arms) | SD3 (pooled CLIP), FLUX (pooled T5), Qwen-Image |
| CFG caption dropout | 0.1 | convention |
| VAE | Qwen-Image VAE (16ch f8) | physically locked by stage-1 256p precompute; switching (e.g. DC-AE) = full re-precompute, out of scope |
| Text encoder | Qwen3-0.6B frozen (exit layer ablated in 2.3) | encoder-size gains saturate early (DeepFloyd IF et al.); params go to the DiT |
| Optimizer after original Stage 2 | **Muon (chunked orthogonalization); auxiliary AdamW for parameters routed outside Muon** | Original scaling with Muon LR 0.02 / AdamW 1e-4 is current. Earlier experiments used RMS-matching scaling; see D12. |
| Evaluation inference knobs (solver/steps/CFG/precision/offload) | **recorded and correctness-checked in Stage 4** for reproducible comparisons | No deployment-speed gate; distillation is optional later work |

### B. Considered and excluded

| Dimension | Reason |
|---|---|
| Cross-attention conditioning (PixArt/Hunyuan style) | SD3's own comparison favors joint attention; all post-2024 SOTA is joint; our RoPE/text pipeline assumes joint |
| Full block weight sharing (ALBERT/looped DiT) | niche literature, risky; the param saving is dominated by modulation sharing + depth/width tuning anyway |
| muP / LR-transfer machinery | 0.4–0.7B span too narrow to need it |
| High-compression VAE (DC-AE et al.) | would void the stage-1 precompute |

### C. Experiment axes → stage 2

Fixed protocol for **every** arm: same data mix, same steps, same seed, EMA on,
`fused` conditioning, logit-normal(0,1) + shift=1 at 256p, same LR schedule
(2.5 excepted: per-optimizer LR), eval suite at end. Only compare arms run on the
same platform; cross-platform comparisons are qualitative only.

## Stage 0 — Infra onboarding (≤10 4090-h, mostly CPU)

**Goal**: both compute environments usable end-to-end.

- Inspire: confirm workspace / remote paths / base image with user;
  `inspire init --scope project`; write `INSPIRE.md` (machine-local project configuration).
  Bake deps into a project image (torch 2.9, diffusers, transformers, datasets, accelerate).
  Verify HF access from CPU side (mirror if needed); verify shared-disk r/w from both
  `CPU资源空间` and the GPU workspace.
- **Locate the 4090 groups**: account sees `4090`, `4090-2`, `4090-cuda12.8`,
  `4090-cuda12.8-2`, `4090-cuda13.2-2` — find which workspace hosts them and their quota
  rows via `inspire job quota --workspace <ws>` / `resources availability`; record in
  `INSPIRE.md`.
- local RTX 4060 Ti workstation: SSH smoke — torch sees the 4060 Ti, repo tests pass, a 256p mini-run trains.
  Note VRAM (assume 16GB): ablation arms there must use small micro-batches + grad accum.
- VLM API: pick provider/model (Qwen-VL-Max-class), store key, verify a test call.

**Exit**: trivial GPU jobs succeed on Inspire (nvidia-smi + disk r/w + HF download) and
on local RTX 4060 Ti workstation (`pytest` + 100-step 256p run).

## Stage 1 — Dataset curation (≤30 4090-h + VLM API spend, mostly CPU/network)

**Goal**: all training domains assembled on shared disk as HF datasets with the caption
schema; eval set built.

- 1.1 Source probes: pull ~1K images per source (Met, Smithsonian/Freer, NPM-TW, AIC, NGA,
  WikiArt mirror, ArtBench-10, FFHQ). Verify metadata fields, license flags, download yield,
  dedup rate. Pin the full-body photo source.
- 1.2 Caption probe (API): candidate models on ~200 images; human-rate quality; measure
  **cost and latency per 1K images**, rate limits, concurrency ceiling → pick model and
  caption budget. Caption schema per `dataset_plan.md` (meta short / mid / long / zh),
  aligned with the conditioning prompt in `src/utils/encode_text.py`. Cap ~256 tokens.
- 1.3 Full harvest + phash dedup + curation rules (no-flip for 国画/calligraphy; flip OK
  for photos). Targets: 国画 40–80K; impressionism 30–55K (CC0+NC split); people 60–120K
  (FFHQ + portraits + full-body); world 200K–1M (sized in stage 4).
- 1.4 API captioning at scale: async batch with retries/rate-limit handling; **cache raw
  API responses to disk** (reproducibility + re-run safety); then materialize per-domain HF
  datasets (`save_to_disk`) with fields:
  `image, caption_meta, caption_mid, caption_long, caption_zh, artist, title, date, source,
  license, aesthetic_score?, pwatermark?`
- 1.5 Eval set: fixed prompt suite (style / anatomy: faces-hands-figures / zh / variable
  aspect) + held-out image sets per domain for KID.
- 1.6 Precompute all domains @256p (VAE on GPU — local RTX 4060 Ti workstation is fine; batch encode).

**Text-side tradeoff (2026-09-01, decided: keep as-is)** — whether to pre-encode
prompts (tokenize / Qwen3-0.6B hidden states) during precompute vs. encode online
during training. Decision: **precompute stores cleaned multi-field caption text only;
tokenize + Qwen encode stay online in the training loop.** Rationale:

- Storage: pre-storing Qwen3-0.6B hidden states (hidden 1024, bf16, ~800 tok/sample)
  costs ~1.6MB/sample → ~1.6TB for the 1M-sample 256p bucket (and each higher-res
  bucket doubles that) — infeasible on GPFS. Token ids (~3.2KB/sample, Qwen BPE is
  1:1 on Chinese) are larger than the source text (~1KB) and save nothing: the
  tokenizer itself is cheap (Rust backend ~50K tok/s/process → minutes over the corpus).
  Text-only storage is ~1GB/bucket — the cheapest and already the current design.
- Compute: the real cost is the Qwen forward (token→hidden), not tokenization.
  Throughput ~50–80K tok/s on an A100 at batch 64/seq 800 → ~2–3 GPU-h per epoch per
  million samples (same order as a 256p DiT training epoch). It cannot be precomputed
  (storage-infeasible), and it must run per-batch since hidden states cannot be cached
  in RAM either (~1.6TB/epoch-bucket). Each sample is encoded ~once per epoch (caption
  dropout is masking, not re-encoding; the short→long curriculum varies fields across
  epochs, not within).
- Flexibility: online encoding keeps the caption curriculum, per-field sampling,
  language dropout, and prompt-template changes free — pre-encoding would lock all of
  them in and force a full re-encode on every template change.
- If the 2–3 GPU-h/epoch ever becomes painful: switch to a smaller text encoder
  (0.1–0.2B class) or cap sequence length at 512 — both halve the cost without
  changing the pipeline shape.

**Scope clarification (2026-08-26, user)**: stage 1 covers **all domains including world** —
the world pool is assembled to a comfortable upper bound (~2M captioned records from
`relaion-art-recap-zh` / PD12M are already cheap to keep), and the stage-4 scaling probe
only decides how much of it enters the mix, not whether to collect it. Resolution scope:
stage 1 precomputes **256p only**; Stage 3.5 prepares 640p after caption enrichment.
896p precompute remains in stages 4/5 when the resolution recipe is known.

**Exit**: domain datasets validated; eval suite committed; 256p precomputed sets ready;
`data/` layout documented in `INSPIRE.md`.

## Stage 2 — Ablations @256p, small scale (≤400 4090-h on Inspire + local RTX 4060 Ti workstation hours free)

**Goal**: pick architecture (depth/width, modulation, stream schedule), text-encoder
exit layer, optimizer, and validate the RoPE fix — cheaply, fairly. Fixed protocol per
design-dimension ledger §C. **Only compare arms run on the same platform** (local RTX 4060 Ti workstation ≠
Inspire hardware; cross-platform comparisons are qualitative only). All per-arm results
are consolidated into the decision memo below (raw working record archived 2026-09-07).

- local RTX 4060 Ti workstation (4060 Ti; arms ≤15K steps, batch ≤64 via accum — it runs ~¼ 4090 speed):
  - 2.1 **RoPE fix smoke + A/B** (D7): new centered-grid/fixed-text-diagonal RoPE trains
    stably; then old-vs-new on resolution transfer (train 256p → sample 640p; artifact
    rate). This is the gate for everything below.
  - 2.2a **Modulation sharing**: `none` vs `layer`, h=1024, d=24, all-single-stream,
    10K-step screen on loss-curve separation. Runs **first** — its winner feeds
    2.2b–2.2d and fixes the param matching there. (Split out from the old 2.2a, which
    varied h, d, and mod simultaneously and could not attribute the delta.)
    — **RESOLVED 2026-09-05: mod=layer wins** (D11). Scenario-A shapes in §5's shape
    table are active.
  - 2.3a **Text-encoder early exit, qualitative** (D5): `--text_encoder_exit_layer`
    (`output_hidden_states`, one-line change in `encode_text.py`); k ∈ {8,16,28} short
    runs, eval-loss separation check.
  - 2.4 Mixture sanity: stage-1 mix vs old 80/10/10 mix (quantify the obvious).
- Inspire 4090 (the fair arms that decide the hero config; 20K steps, batch 128):
  - 2.2b **Depth/width iso-param ~500M**: `h=1152,d=20` vs `h=1024,d=30`, both at the
    2.2a-winning modulation.
  - 2.2c **iso-FLOP**: ~670M vs ~400M (`h=1024,d=24`), smaller run proportionally
    longer; step ratio from fvcore-measured FLOPs/step, not param ratio.
  - 2.2d **Stream schedule** (new axis): all-single vs hybrid (~1:2 double:single param
    split, e.g. 6 double + 12 single @ h=1024) vs all-double (12 double), iso-param
    ~500M at the 2.2a-winning mod; 10K-step screen, winner 20K confirm. Prior from
    DiT-Air (arXiv:2503.10618, shared AdaLN + concatenated single-stream processing is
    most param-efficient) and FLUX (hybrid) leans single-heavy/hybrid; drop the
    all-double arm first if budget pinches.
  - 2.3b exit-layer confirmation at the 2.2-winning config (if 2.3a was ambiguous).
  - 2.5 **Muon vs AdamW** (new axis, 2026-09-04): literature now covers DiTs at our
    scale — CMuon (arXiv:2608.02502) shows vanilla Muon hits a late-stage plateau on
    DiTs because fused tensors (6×dim AdaLN output, fused QKV) couple subspaces under
    orthogonalization, and that chunking those matrices before Newton–Schulz fixes it
    (>2× speedup over AdamW, FID 1.18 on ImageNet-256 in 200 epochs at 675M);
    Scaling-Muon-for-DiT (arXiv:2608.20818) shows the quality advantage persists
    1.3–15B. Arm design:
    - Param split: 2D hidden weights → Muon **with chunked orthogonalization for the
      fused qkv/modulation matrices** (chunk to q/k/v and per-modulation-piece);
      embeddings, patch conv, final_layer, norms, biases, t/c-MLPs → AdamW. Weight
      decay on both groups (Moonlight finding).
    - LR probe first: 3 Muon LRs × 3–5K steps (Muon needs its own LR — do not reuse
      AdamW's 3e-4), pick by eval loss; then Muon vs AdamW 20K-step confirm at the
      2.2-winning arch, same steps/data/seed.
    - Watch: attention-logit growth (QK-RMSNorm already mitigates), NS5 step overhead
      (expect <10% at our matrix sizes — record it in the throughput table).
- Record throughput (samples/s) for every arm → stage-3 baseline.

**Exit**: decision memo — arch config (h/d, stream schedule, modulation), exit layer,
optimizer + LR, RoPE scheme — plus throughput table.

### Stage-2 decision memo (2026-09-07, all arms done — working record archived)

**Hero recipe (256p stage-2 winner, ~485M):**

| Dimension | Winner | Evidence |
|---|---|---|
| RoPE | centered-grid (image), text fixed diagonal | 2.1: tie everywhere → Qwen-Image prior (user rule) |
| Modulation | mod=layer (shared per-layer MLP) | 2.2a: eval/loss + t040 5/5 + KID + speed/mem |
| Width×depth | h1152 × d24 | 2.2b: wide > deep at every probe, 0.28% @16K, KID agrees |
| Stream | all-single (no double-stream blocks) | 2.2d: monotone gradient 0.37%/0.65% @8K vs hybrid/all-double |
| Size | ~485M | 2.2c: iso-FLOP ~400M > ~664M → 664M deferred to stage-4 probe |
| Text exit | **k=20** | 2.3a: k28 < k8 < k16 at matched 4K; follow-up (2.3a-followup, 16K): k20 loss 0.92160 vs k28 0.92134 (Δ0.00026, tie) but **KID 0.00892 vs 0.00922 (k20 best of all arms)**; user visual verdict on portrait/face grids (11K + final): k20 wins on facial structure → hero exit = 20 |
| Mix | stage-2 (art-forward) mix | 2.4: tie on eval-loss; kept per intent |
| Optimizer | Muon (chunked NS), LR 0.02 | 2.5: 16K confirm -1.1% eval/loss, -24% KID, +10% time |
| Throughput | ~2.9 s/it AdamW / ~3.2 s/it Muon @ batch 128, 1×4090 (~55 samples/s) | arms' telemetry |

Fixed protocol for the winner: fused conditioning, qkv_bias, gated FFN mlp_ratio
2.67, rectified flow logit-normal(0,1) shift=1 @256p, batch 128, seed 42, EMA
0.999 (16K runs), curriculum 0→1, caption dropout 0.1, warmup 500.
Stage-2 GPU-h total ≈ 2.1(8) + 2.2a(12) + batch2(2×11.5+2×3+2×6) + batch3
(2×13+2×4.8) + 2.5(3×1.5+16) ≈ **130 GPU-h** (of 400 budget; trough/user-policy
actual). Stage 3/4 re-derive: size×steps point (4.1/4.3), Muon LR schedule at
640p+, and the resolution curriculum — hero intent (art-forward) confirmed at
every gate.

**2.3a-followup closing (2026-09-07, exit k ∈ {20,24} at hero arch, 16K AdamW,
resumed from ckpt-6000; user verdict):**

| metric @16K | k20 | k24 | k28 (s2-wide) |
|---|---|---|---|
| eval/loss | 0.92160 | 0.92363 | **0.92134** |
| KID | **0.00892±0.00328** | 0.00917±0.00337 | 0.00922±0.00324 |

k20 vs k28 is a statistical tie on loss (Δ +0.00026) with KID favoring k20
(−0.00030, best of all three arms); k24 trails loss at every per-step bucket
(+0.0012~+0.0039) with KID a wash. **User verdict (2026-09-07): k20 wins** —
manual grid inspection (step 11000 and final) found k20 clearly best on
facial structure in close-up portraits (e.g. elderly-man close-up: k24/k28
draw a single eye + oversized nose while k20 renders both eyes/nose/mouth;
baroque noblewoman k20=k24>k28; 少女侧脸特写 k20≈k28>k24), a quality axis
eval-loss/KID under-weight. Because slicing `hidden_states[k]` after a full
forward is bit-identical to stopping the frozen encoder forward at layer k,
these results validate a **true early exit at k=20** with zero feature change:
skips layers 21–28 → ≈29% of the text-encoder forward compute saved. Stage 3
implements it; hero recipe exit layer = 20.

## Stage 3 — Infra and efficiency on the decided architecture

Stage 3 keeps the Stage-2 hero architecture, dataset mix, caption curriculum,
and chunked Muon plus auxiliary AdamW optimizer. Its current implementation and
throughput gate are defined by [`notes/stage3_gate.md`](archive/stage3_gate.md).

The work is row-preserving: sample a dataset and row first, select one caption
inside that row with the existing curriculum, then use `(resolution,
retained_prompt_length)` queues with per-bucket local batch sizes. Actual global
samples, not nominal batch arithmetic, determine the accumulated gradient mean
and the reported samples/s. The complete prompt/tokenizer cap, drop boundary,
and sidecar contract are shared by offline metadata generation and training.

Static bucket padding/compile, the slice-vs-true-stop text-encoder choice,
data loading, and resolution switching are retained only when matched eval/loss
and actual global samples/s support them. The gate document is the sole current
Stage-3 scope and acceptance reference.

**Outcome (2026-09-09, review rerun):** Stage 3 closes on the single-GPU
evidence, per the user's scope decision. Corrected timing includes completed
optimizer/EMA work and telemetry/logging, excluding evaluation/checkpoint time.
Sequential 256p A/B, 400 optimizer steps and 102,208 actual samples per arm:
baseline **54.99 → 72.98 samples/s steady state (1.327×)**, above the 1.25×
gate. Including compile warm-up: 55.11 → 67.94 samples/s (**1.233×**, below
1.25×; warm-up cost is not hidden). The fixed eval-loss trajectory differs by
at most **0.254%**, at the final probe (1.68437 baseline vs 1.68865 optimized);
this is accepted alongside the correctness tests and earlier A/B evidence,
not a claim of exact numerical equivalence or long-run quality proof.
The queued 4-GPU rerun was stopped at the user's request. Earlier multi-GPU
measurements retain diagnostic value, but their old timing and scaling figures
are **not revalidated** and must not be combined with this corrected denominator.
Kept mechanisms: host-side caption dropout, bincount telemetry, vectorized text
slice, hoisted RoPE/attention tables, true early exit at k=20, per-block
`torch.compile`, batched Muon Newton-Schulz, boundary-only DDP reduction.
Rejected on measurement: cuDNN attention, text-encoder prefetch, CUDA-graph
`reduce-overhead` compile, whole-model `torch.compile`. Details, per-resolution
ceilings, and the Stage-4 sizing inputs are in
[`notes/stage3_gate.md`](archive/stage3_gate.md).

## Stage 3.5 — Long-caption experiments, dataset refresh, and bucket planning

**Goal**: prepare the caption distribution and efficient per-resolution batching
before the Stage-4 scaling probes. The detailed [Stage-3.5 plan](archive/stage3_5_plan.md)
sets a separate 75-hour experiment/profiling cap and approximately $100 API budget;
full-corpus precompute/storage require separate costing. Its
[execution record](archive/stage3_5_pilot.md) records subsequent design amendments and
measured outcomes, which supersede older provisional details below.

- 3.5.1 **Long-caption bias experiment**: compare the current curriculum with
  a controlled preference for longer captions *within an already selected row*.
  Keep row/dataset marginals unchanged. Compare matched actual sample budgets,
  eval loss, prompt adherence, caption-length coverage, throughput, and memory;
  keep a bias only if the quality/cost tradeoff warrants it.
- 3.5.2 **Systematic caption enrichment through ZenMux**: add more
  **256–2048-token** captions to existing rows, not extra copies of those rows.
  Pilot grounding/quality, token-length coverage, API cost, and rate limits;
  cache raw responses and record provider/model version and generation settings.
  Potentially update `kaupane/chinese-painting-collection` after validating the
  enriched records and dataset release metadata.
- 3.5.3 **Regenerate training/evaluation data**: materialize the accepted captions,
  re-precompute the affected datasets, rebuild retained-prompt length metadata,
  and regenerate the light eval dataset. Preserve train/eval row separation and
  pin dataset revisions so all later arms use the same evaluation version.
  Include **640p precompute** and refreshed 256p inputs.
- 3.5.4 **Find the optimal `bucket_plan`**: screen length boundaries and
  per-bucket micro-batch sizes, then finalize on the enriched caption distribution
  at 256p and 640p. Measure end-to-end samples/s, padding/compile overhead, and
  peak memory over the full 2048-token range and all aspect buckets. Record a
  memory-safe, measured winner per resolution, with multi-GPU headroom.

**Exit**: caption-bias verdict, versioned data/metadata and light eval set,
640p precompute, measured bucket plans, and actual preparation costs recorded.
Stage 4 tunes effective batch through `gradient_accumulation_steps`, rather than
reopening per-bucket micro-batch-size tuning. A larger model or new resolution
still requires a memory-safety check and, if necessary, an explicitly recorded
bucket-plan revision.

## Stage 4 — Complete hero recipe and infrastructure validation (≤450 4090-h)

**Goal (user, 2026-09-11)**: deliver the **entire executable hero-run
recipe**, from scratch through **256p → 640p → 896p**. Stage 5
must be able to proceed directly, **without a planned Stage 4.5**. Full design,
literature anchors, evaluation contract, experiment budget, and exit checklist:
[`notes/stage4_plan.md`](archive/stage4_plan.md).

The 450-hour experiment/profiling cap is separate from Stage 3.5's 75 hours and
the hero working range of **1,600–2,200 RTX 4090 hours**. Empirical diffusion
scaling informs the recipe, but neither borrowed coefficients nor low flow loss
establish the required capabilities.

- **Qualitative monitoring:** ordinary full-body figures, everyday
  hand–object interaction, plausible faces, common architecture, six agreed
  Chinese/Western painting styles, and basic layout. Use ordinary short and long
  prompts in Chinese and English without polishing. The recipe's small fixed panel
  supports VLM-first monitoring and escalation of uncertain judgments; there is
  no numerical capability target or formal qualification suite.
- **Infrastructure optimization and evaluation:** measure, optimize and remeasure
  eight-GPU execution, memory, communication and input throughput on the selected
  data/caption policy and bucket plans. Implement and validate bottleneck-driven
  changes; measurement alone is not the deliverable. Record fixed evaluation
  solver/steps/guidance/precision/offload settings; deployment hardware and
  inference latency do not gate the hero launch.
- **Size, exposure, and effective batch:** the 533M model and per-stage mixtures
  are selected in [hero_recipe.md](archive/cuda_hero_recipe_0924.md). Validate its bucket plans and
  accumulation 1/5/7 against mean effective-batch targets 640/512/400 on eight
  ranks. Distinguish actual draws from unique rows and measure realized exposure;
  no additional model-size, corpus-size, or mixture sweep is a launch requirement.
- **Complete resolution/optimization schedule:** test 256p→640p and 640p→896p
  continuation with continuous global LR/caption schedules. The stage split is
  75:20:5; 1024p is definitively dropped. Derive final total steps, sample budgets
  and exact endpoints from optimized infra rates, with the recipe's operational checks.
- **Fixed recipe, observed capability:** model size, GPU budget, optimizer and
  data/training policies are already selected. Remaining work optimizes execution
  efficiency and feasible training exposure. Record the trained model's strengths
  and limitations; do not claim a capability guarantee from the chosen recipe or
  require a small-run capability forecast before launch.
- **Executable handoff:** write `notes/archive/cuda_hero_recipe_0924.md`, runnable configs and
  launch/resume commands, pinned data/metadata, precompute and storage plan,
  evaluation/telemetry, evaluation sampling configuration, and complete cost/uncertainty
  ledger. Required implementation and relevant tests/smokes finish inside Stage 4.

**Exit**: an evidence-backed go verdict and complete budget-feasible recipe;
no material recipe decision or enabling implementation is deferred to Stage 5.
Every measurement is traceable and every extrapolation labeled. Insufficient
correctness or cost-feasibility evidence blocks launch; resolve execution issues
or request an explicit redesign without exceeding the 2,200-hour ceiling. There
is no capability-accuracy threshold to predict or certify before or after training.

## Stage 5 — Execute the hero recipe (1.6–2.2K 4090-h working range)

**Goal**: execute and verify the complete Stage-4 recipe, starting directly after
its go handoff. All resolution allocations, mixtures, optimization, evaluation sampling
settings, and operational checks come from `notes/archive/cuda_hero_recipe_0924.md`, not the earlier
guessed step counts. The selected path is 256p → 640p → 896p, with a 75:20:5
optimizer-step split and no 1024p stage. Eight-GPU throughput must be measured;
the selected topology alone does not establish the wall-clock budget.

- Run the finalized data-preparation, training, checkpoint, and resume workflows.
  Full production high-resolution precompute may execute here only with its
  policy, tested procedure, storage, validation, and cost allocation already settled.
- Follow the selected chunked-Muon/auxiliary-AdamW recipe and exact per-stage
  dataset/quality/caption schedules; keep license-restricted entries identifiable.
- Track actual samples and GPU-hours against the full ledger. Do not reuse the
  old **60–100M samples in 2K 4090-h** projection or extend beyond 2,200 hours
  without a new explicit budget decision.
- Review the frozen 48-image bilingual short/long-prompt panel at the recipe's
  cadence, alongside the fixed loss probe and end-of-global-training KID.
  These are regression diagnostics, not additional per-domain KID, memorization
  or capability-qualification suites. Follow the recipe's stop/review rules;
  extensions, rollback or scientific changes require explicit approval.
- Verify the final hero checkpoint at the recorded evaluation sampling
  settings. Unexpected failures may require redesign, not automatic post-training rescue.

**Exit**: a hero checkpoint from the completed budgeted recipe,
with measured costs, observed abilities and limitations. Preserve verified checkpoints and
follow the recipe's review/stop rules if a resolution transition degrades.

## Stage 6 — Last-mile post-training: optional SFT → DMD2 cold start → joint DMD+RL

**Goal**: turn the hero checkpoint into the final deliverable — an **8-step distilled
model** with preference alignment. Only the 8-step student is released; the hero
(multi-step) model is an intermediate artifact. Method follows the **DMDR** route
([arXiv:2511.13649](https://arxiv.org/abs/2511.13649)): DMD2-style distribution-matching
distillation ([arXiv:2405.14867](https://arxiv.org/abs/2405.14867)) with RL folded in as
a joint objective rather than a separate stage. Budget: ≈300–500 4090-h main line under
PRIORITY 1 (wall-clock, not GPU-hour constrained) + 100 4090-h reserved for ablations.
Context: hero budget is 2,200 nominal, realistically 2,500–3,000 4090-h, so this stage
sits at ~15–25% of pretraining — already an order of magnitude above the 1–5% typical in
the diffusion literature (DiffusionNFT: GenEval 0.24→0.98 in 1K steps; DMDR: 1.5K+1.5K
steps). The risk here is correctness and system structure, not throughput: **no separate
infra stage**; one embedded measurement pass, then a 640p pilot, then the 896p main line.

**Platform decision (2026-09-23): all Stage-6 work runs on NVIDIA cards** (4090 jobs in
可上网GPU资源; judge/HF/swanlab integrations are already proven there). The Ascend
line carries the hero pretraining only — the 910B bring-up cost (shm coin-flip crashes,
torchair GE blockers, tbe shutdown hangs, 30–70 min log lag, 16-card整机 quota) is
tolerable for a fixed, already-validated recipe but unacceptable for research-grade
bring-up where every bug would become "model bug or platform bug". Stage-6 compute is
modest (533M model, 8-step rollouts, 100 4090-h ablation budget), so the free Ascend
capacity buys nothing here.

**Why joint, not sequential** (DMDR §2.3, Fig. 3): RL on an already-distilled model
anchors on a mode-impoverished distribution → reward hacking and collapse; RL before
distillation loses gains to the "distillation gap" and pays multi-step rollout costs.
Joint `L = L_dmd + λ_rl · L_rl` lets DMD regularize RL (reward ascent is projected back
onto the teacher manifold — the primary defense against the "greasy" over-optimized look)
while RL breaks the teacher ceiling. Z-Image-Turbo is the industrial 8-step precedent
(decoupled-DMD + DMDR).

- 6.0 **CFG study (after hero completes, before locking Stage-6 configs)**: old-run
  evidence says CFG=1 was already optimal. Probe the final hero checkpoint: unconditional
  branch health (empty-caption samples, cond/uncond eval-loss gap), CFG ∈ {1, 1.5, 2, 3}
  sweep with KID + reward + eyeball grids, and whether the old observation was an EMA-lag
  artifact. DiffusionNFT is CFG-free by design and interprets CFG as offline reinforcement
  guidance, so "CFG=1 is genuinely best" is a feature, not a bug. **Do not cut the CFG
  code path until this study concludes.**
- 6.1 **Optional targeted SFT**: after the hero checkpoint, probe capability gaps
  (bilingual panel + blind review); if a gap is best fixed by SFT rather than RL
  (e.g. a compositional or texture deficit), generate a targeted synthetic batch with the
  existing caption/generation infra and SFT briefly. Skip if no such gap. Z-Image-Turbo
  synthetic data was high quality; the failed East-Asian synth set was Qwen-Image/Ernie —
  teacher choice matters.
- 6.2 **DMD2 cold start at 896p** (896p is the native resolution; no reliance on
  640p→896p transfer of the student). Fake score = online copy trained on student outputs;
  GAN = classification head on the fake-score backbone bottleneck, operating on
  noise-injected latents (no pixel-space decode in the loop), non-saturating objective,
  real side from our precomputed latents. Two time-scale fake/generator updates starting
  at 5:1. **DynaDG dropped** (teacher = student architecture/init, manifold gap is small);
  revisit only if cold start stalls. If a 1024p rope-scaling stage ever happens, rope-scale
  the teacher first, then distill.
- 6.3 **Joint DMD+RL**: RL term is an ablation axis, **ReFL vs DiffusionNFT** (NFT is
  SFT-form, shares the velocity-MSE units with the DMD loss, and is natively off-policy →
  rollouts/rewards can be fully asynchronous; it needs only terminal images, so 8-step
  student rollouts are valid). λ_rl sweep starts near 1.0 (same-unit losses) and is
  adjusted by rule, not schedule: test-reward plateau + KID regression → lower λ.
- 6.4 **Rewards**: LAION aesthetic + HPSv2 (local, cheap) + VLM-as-judge rubric
  (anatomy, prompt adherence, and **explicit negative criteria for the greasy/over-smoothed
  look** so artifact-y high-reward samples land in NFT's negative branch). Heterogeneous
  rewards dilute single-scorer biases. Probe the judge first: thinking on/off, model tier
  (non-flagship preferred), async throughput; cap total judge spend before the main run.
  A **held-out reward** (never optimized) is monitored to detect hacking of the trained
  rewards.
- 6.5 **Multi-domain guardrails**: rollout prompt pool mirrors the hero data mix (no
  re-sweep; heuristic merge only). Per-domain canary prompt grids (国画 / oil-impressionism /
  people / world) at fixed seeds, eyeballed on a schedule — the user is the final arbiter
  of "greasy". Watch test-vs-train reward divergence and precision/recall drift as early
  hacking/collapse signals.

**Ablation axes** (100 4090-h): RL algorithm (ReFL vs DiffusionNFT; GRPO/DPO
excluded per DMDR's negative evidence), λ_rl, reward
composition and multi-domain weighting, rollout group size G (16 vs 24; NFT
default 24, initial value 16 to save VLM scoring cost), and prompts per
iteration (initial value 48 like NFT → effective RL batch ~768 at G=16;
precedents: NFT 48×24=1152, ReFL 128×1=128). G and prompts/iter are
cost-dominated by the reward scorer (bills, not GPU), so probe both alongside
the VLM judge probe. Everything else starts from literature defaults.

**Fixed before any run**: 8-step timestep grid (sweep Euler grids on the hero checkpoint,
lock before distillation); backward-simulation
micro-batch from a one-off memory measurement (embed in the framework bring-up).
Cold-start → joint promotion rule is **deferred**: decided together with the 640p
checkpoint (2026-09-21), not fixed now — candidates are the PromotionGate reward-reliability
probe vs DMDR's fixed step threshold; the settled constraint is only "not at distillation
convergence" (that would collapse into the rejected sequential pipeline).

**Dependencies — what can start now vs what waits for the hero checkpoint:**

- *Now (no hero checkpoint needed)*: DMDR training loop (fake score, GAN head, backward
  simulation) with unit tests on small random models; async reward pipeline; local reward
  scorer integration; VLM judge probes on existing grids; a small (5k) rollout prompt pool
  from the hero mix for pipeline bring-up only — the full ~20k pool is deferred to the 640p
  checkpoint and will be built jointly with the user (2026-09-21 decision); per-domain
  canary grids (adapt the eval panel); reward budget calculation; KID reference sets;
  backward-simulation memory measurement on the 533M architecture.
- *Methodology now, rerun on the final checkpoint*: 8-step timestep grid sweep (build the
  harness on the current checkpoint, re-run at the end).
- *Waits for the hero checkpoint*: CFG study (6.0), capability probe → SFT decision (6.1),
  λ_rl / RL-algorithm ablations, and all main runs. The 640p pipeline pilot may use the
  640p-stage endpoint checkpoint (~step 360K) when it appears, without waiting for 480K.

**Exit**: an 8-step student that beats the hero teacher on reward and panel review without
KID/diversity regression; documented λ_rl and reward configuration; before/after grids per
domain.

## Stage 7 — Publication readiness

**Goal**: a usable public release whose explanations stand on their own.

- 7.0 **Rename to `inko` (user decision, 2026-09-12)**: rename the repository
  and model from `artflow` to `inko` before publication. Update comments,
  documentation, model cards, examples, and repository/model links consistently;
  reconcile affected package/import names, configuration references, and demo
  metadata, then verify the renamed quickstart and model-loading path. Preserve
  historical artifact identifiers where needed for reproducibility, with an
  explicit old-to-new name mapping. The actual rename is deferred to Stage 7.
- 7.1 Upload the selected model to **Hugging Face**, with model card, weights,
  inference configuration, provenance/license constraints, evaluation results,
  and limitations; deploy and smoke-test a **Hugging Face Space** demo.
- 7.2 Add/rewrite **README.md** as the public entry point: what the model does,
  installation, minimal inference/training examples, model/demo links, data and
  license caveats, evaluation, and a navigable documentation map.
- 7.3 Remove or rewrite internal/temporary terminology and opaque references.
  Core principle: **“不要把解释清楚一件事物的责任转包给外部不可见的指代”**.
  Audit the repository, especially `notes/`; remove obsolete working notes or
  rewrite them into self-contained explanations. An internal stage name, run
  nickname, private dashboard, or unavailable discussion must not substitute
  for explaining a method, decision, configuration, or result. Preserve useful
  evidence in accessible, reproducible form and keep private operational details
  out of the public-facing documentation.
- 7.4 Add substantive explanatory notes on **Flow Matching** and the Stage-6
  **post-training stack (DMD2/DMDR distillation, negative-aware RL)**: motivation,
  equations/notation, training and sampling procedures, implementation mapping,
  assumptions, limitations, and accessible references—not merely experiment logs.

**Exit**: repository and model renamed to `inko`, with comments/docs and affected
references updated; downloadable model and working Space; README quickstart verified from
a clean environment; public terminology/reference audit complete; Flow Matching
and NFT notes readable without access to internal conversations or artifacts.
Publication/demo hosting costs are estimated separately before deployment.

---

## Compute ledger (RTX4090-hours unless noted; caps)

| Stage | Cap | Notes |
|---|---|---|
| 0 Infra | 10 | mostly CPU; smokes on both platforms |
| 1 Data | 30 + API spend | API captioning replaces GPU captioning; cache raw responses |
| 2 Ablations | 400 | Inspire fair arms only; local RTX 4060 Ti workstation takes smoke/qualitative arms (free, ~¼ speed); cap raised 200→400 on 2026-09-04 to fit stream-schedule + Muon axes |
| 3 Efficiency | 80 | buys back far more than it costs — gates stage 5 |
| 3.5 Captions / buckets / 640p | 75 experiments/profiling + separate precompute | approximately $100 API budget; full-corpus precompute/storage costed separately; see detailed plan and execution record |
| 4 Complete recipe / infra | 450 | separate experimental cap; execution optimization, cost estimates and ready-to-launch hero handoff; no capability or serving-performance gate |
| 5 Hero | 1,600–2,200 provisional | target topology and wall time must be justified by Stage 4 |
| 6 Last-mile post-training (SFT/DMD2/joint RL) | 300–500 + 100 ablation | sampling-bound |
| 7 Publication | TBD + hosting spend | model/Space release, documentation and reproducibility checks |
| **Original subtotal** | **≈2.9–3.7K 4090-h** | excludes new Stages 3.5/7 and API/storage/hosting spend; Stage 4 must update the complete ledger |

**Current Stage-4/5 budget boundary (user, 2026-09-11)**: Stage 4 has a separate
450-hour experiment/profiling cap; the hero working range remains 1,600–2,200 hours.
The historical off-peak flexibility below is **not standing authorization** to
exceed these constraints. An execution plan that cannot fit the hero ceiling needs
an explicit recipe/budget decision before launch. Scheduling priority remains
as documented below.

**Historical budget semantics (2026-08-26, extended 2026-09-06)**: the
Inspire budget is about **not crowding out other users**, not an absolute hours
cap — scheduling GPU work into off-peak (late-night) troughs is explicitly
sanctioned, and exceeding the nominal ledger (e.g. 5K h) is acceptable if it
runs in troughs. **2026-09-06 priority policy (user)**: stage 2–4 experiments
(ablation arms, screens, probes, precomputes) run at **medium-high priority**
(`--priority 4`, platform maps to HIGH/NORMAL); **only the stage-5 hero run**
(the long multi-day training) sits at **LOW (priority 1, preemptible)** — idle
cards fill it whenever free and it can be stopped anytime. Preemption/resume
path proven by the 2.1 restarts and the k20/24 16:12 auto-restart.
Conversely, **avoid late-night runs on local RTX 4060 Ti workstation** (shared desktop). Practical
rule: long Inspire jobs (stage 5 hero) run at priority 1; local RTX 4060 Ti workstation arms run
daytime/evening.

## Open risks

- ~~Centered-vs-legacy RoPE~~ Resolved (2.1, 2026-09-05): tie at every probe down to
  320p → centered-grid RoPE adopted on the Qwen-Image prior. Zero-shot resolution
  extrapolation fails from 1.875× up regardless of variant → progressive fine-tuning
  is the only path to 640p/896p.
- ~~Early-exit text features unvalidated~~ Resolved (2.3a-followup, 2026-09-07):
  k20 chosen — metrics tie k28 with best KID, user visual verdict wins on
  portrait facial structure; early exit at layer 20 is feature-identical to the
  validated slice. Stage 3 must implement the true early exit (stop at layer 20).
- The Stage-2 optimizer decision propagates unchanged: Stage 3–5 quality, throughput,
  step/quality knees, and LR schedules use chunked Muon for 2D hidden weights plus
  auxiliary AdamW for the remaining parameters, with Muon LR 0.02. An AdamW-only
  run is historical reference data, not a Stage-3/4 baseline. Chunked
  orthogonalization is part of the optimizer definition, not an optional tweak.
- Cross-platform comparability: local RTX 4060 Ti workstation results inform, never decide — fair arms live on
  Inspire 4090.
- 4090 PCIe-only DDP: earlier experiments identified per-micro-batch
  synchronization as a major cost; boundary-only reduction is retained.
  The old 92% scaling figure predates the corrected timer and was not rerun
  in the final single-GPU review. Re-measure target-topology scaling for Stage 4;
  do not assume eight-GPU efficiency from the old four-GPU result.
- VLM API dependency: rate limits / cost drift / provider model updates → cache raw
  responses; record exact model version in dataset metadata.
- Resolution-transition regressions: validate 256p→640p→896p with the fixed
  monitoring panel and preserve verified checkpoints. 1024p and an upscaler
  fallback are not part of the hero recipe.
- NC-tagged data → no commercial release; `license` column + separate mix entries keep a
  clean variant feasible.
- Reward hacking / diversity collapse in RL post-training → guardrails in 6.5 are load-bearing.
