# ArtFlow Reboot — Redesign Plan

Personal side project. Model ≤0.7B params. **Compute-frugal by design**:
provisional budget ≈ **2.9–3.7K RTX4090-hours** on Inspire (4090 @0.33 pt/h ⇒ ~1.0–1.2K points,
trivial vs the ~589K-pt budget of project 自动化科研 — wall-clock and queueing, not
money, are the constraints). Reference point: the old hero run was ~800 RTX4090-h
(256p only, unoptimized stack, 19.2M samples seen). The former ~60–100M-sample
projection inside ~2K 4090-h was a **preimplementation estimate, not a measured
feasibility result**. Stage 4 must derive the hero recipe from the actual available
budget, corrected Stage-3 throughput, and measurements on the Stage-3.5 data and
bucket plans. The provisional ledger below also needs allowances for Stages 3.5
and 7 before it becomes an approved complete budget.

Organized as a linear pipeline of stages, each with goal / tasks / exit criteria /
compute cap. A follow-up agent should be able to execute stage by stage from this file
plus `notes/dataset_plan.md` (data-source detail).

## Locked decisions

| # | Decision | Choice |
|---|---|---|
| D1 | License | Research-only OK (WikiArt, ArtBench-10, FFHQ unlocked). Per-sample `license` field; NC data in separate mix entries so a clean variant stays one mix-string away |
| D2 | Anatomy data | Photos + paintings both; ~50/50 face vs full-body |
| D3 | Corpus size | Set empirically by stage-4 scaling probe |
| D4 | Params | **485M: h1152×d24** — 2.2b (wide > deep at iso-param, every probe), 2.2c (iso-FLOP: ~400M > ~664M → 664M deferred to stage-4 probe), 2.2d (all-single) — all resolved 2026-09-05/06 |
| D5 | Text encoder | Qwen3-0.6B, frozen, online; early-exit layer ablated k ∈ {8,16,28} + follow-up {20,24} → **k=20 (user verdict 2026-09-07**, 2.3a-followup) |
| D6 | Resolution curriculum | 256p → 640p → 896p → optional 1024p polish; variable aspect at every stage |
| D7 | RoPE | **Centered image grid + text pinned to fixed diagonal** (2.1 resolved 2026-09-05: 256p eval/loss+KID tie, 640p transfer tie — both arms collapse identically at 2.5× — 480p/384p/320p ladder tie → final tie-break on Qwen-Image adoption prior). Zero-shot ≥1.875× transfer fails for both variants → progressive staging mandatory |
| D8 | Inspire home | Project 自动化科研 |
| D9 | Compute class | **RTX 4090 48GB on Inspire** (single 8-GPU node max; no NVLink → DDP over PCIe). Small ablations offloaded to **Andromeda** (SSH-reachable, RTX 4060 Ti ≈ ¼ 4090 throughput) |
| D10 | VLM captioning | **Via API** (Qwen-VL-class), not self-hosted — caption cost is money + rate limits, not GPU-hours. No GPU-with-internet workspace needed |
| D11 | Modulation | **Shared per-layer modulation MLP (`mod=layer`)** — 2.2a resolved 2026-09-05: layer wins eval/loss@end (0.9437 vs 0.9441) with a persistent t040 advantage (5/5 probes from 3K, -0.0002→-0.0010), KID agrees (0.0190 vs 0.0195), +0.6% faster, -8% peak mem; tie-break prior (PixArt/DiT-Air) points the same way. All stage-2+ arms use it |
| D12 | Optimizer | **Muon (chunked orthogonalization), LR 0.02** — 2.5 resolved 2026-09-06: 16K confirm muon 0.91127 vs AdamW 0.92134 eval/loss (-1.1%), KID 0.00699 vs 0.00922, +10% step time (<15% bar); AdamW leads early, muon overtakes by 8K and pulls away (CMuon-style late gain) |

## Design dimension ledger (2026-09-04, agreed with user)

Which architecture/training choices are fixed by literature vs decided by stage-2
experiments. Stage-2 arms below implement column C.

### A. Literature-locked (no ablation)

| Dimension | Choice | Anchor |
|---|---|---|
| Objective | rectified flow + logit-normal(0,1) + resolution time shift | SD3 (Esser et al. 2024); FLUX/Qwen-Image follow |
| AdaLN-zero init | on | DiT; universal, already in code |
| QK-RMSNorm | on | SD3/FLUX and everything since; already in code |
| FFN | gated SiLU, ratio 2.67 (iso-param ≈ standard 4.0) | universal post-2024 |
| Patch size | 2 | DiT-XL/2, SD3, Qwen-Image |
| Pooled text in AdaLN | **fused** (old runs used `pure` — flip default for all stage-2 arms) | SD3 (pooled CLIP), FLUX (pooled T5), Qwen-Image |
| CFG caption dropout | 0.1 | convention |
| VAE | Qwen-Image VAE (16ch f8) | physically locked by stage-1 256p precompute; switching (e.g. DC-AE) = full re-precompute, out of scope |
| Text encoder | Qwen3-0.6B frozen (exit layer ablated in 2.3) | encoder-size gains saturate early (DeepFloyd IF et al.); params go to the DiT |
| Optimizer after Stage 2 | **Muon (chunked orthogonalization), LR 0.02; auxiliary AdamW for parameters routed outside Muon** | Stage-2 D12 winner; fixed for Stage 3–5; the Stage-2 comparison is recorded in §2.5 |
| Inference knobs (solver/steps/CFG/guidance distill) | deferred to stages 5/6 | orthogonal to architecture |

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
  `inspire init --scope project`; write `INSPIRE.md` (project 自动化科研).
  Bake deps into a project image (torch 2.9, diffusers, transformers, datasets, accelerate).
  Verify HF access from CPU side (mirror if needed); verify shared-disk r/w from both
  `CPU资源空间` and the GPU workspace.
- **Locate the 4090 groups**: account sees `4090`, `4090-2`, `4090-cuda12.8`,
  `4090-cuda12.8-2`, `4090-cuda13.2-2` — find which workspace hosts them and their quota
  rows via `inspire job quota --workspace <ws>` / `resources availability`; record in
  `INSPIRE.md`.
- Andromeda: SSH smoke — torch sees the 4060 Ti, repo tests pass, a 256p mini-run trains.
  Note VRAM (assume 16GB): ablation arms there must use small micro-batches + grad accum.
- VLM API: pick provider/model (Qwen-VL-Max-class), store key, verify a test call.

**Exit**: trivial GPU jobs succeed on Inspire (nvidia-smi + disk r/w + HF download) and
on Andromeda (`pytest` + 100-step 256p run).

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
- 1.6 Precompute all domains @256p (VAE on GPU — Andromeda is fine; batch encode).

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

## Stage 2 — Ablations @256p, small scale (≤400 4090-h on Inspire + Andromeda hours free)

**Goal**: pick architecture (depth/width, modulation, stream schedule), text-encoder
exit layer, optimizer, and validate the RoPE fix — cheaply, fairly. Fixed protocol per
design-dimension ledger §C. **Only compare arms run on the same platform** (Andromeda ≠
Inspire hardware; cross-platform comparisons are qualitative only). All per-arm results
are consolidated into the decision memo below (raw working record archived 2026-09-07).

- Andromeda (4060 Ti; arms ≤15K steps, batch ≤64 via accum — it runs ~¼ 4090 speed):
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
throughput gate are defined by [`notes/stage3_gate.md`](stage3_gate.md).

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
[`notes/stage3_gate.md`](stage3_gate.md).

## Stage 3.5 — Long-caption experiments, dataset refresh, and bucket planning

**Goal**: prepare the caption distribution and efficient per-resolution batching
before the Stage-4 scaling probes. GPU/API/storage costs are TBD from small
probes, not implicitly covered by the old compute total.

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

## Stage 4 — Scaling-law probes on the stage-3 infra (≤450 4090-h nominal)

**Goal (user, 2026-09-06)**: with the budget relaxed (priority-1 idle fill),
fit the scaling law empirically and produce the COMPLETE training recipe —
the hero can scale up from the base model. Four axes:

- **Model size**: 等比 scaling of the confirmed architecture (same
  h/d/stream/mod structure, width/depth up) — e.g. 485M → ~0.7B candidates;
- **Effective batch size**, tuned via `gradient_accumulation_steps` with the
  Stage-3.5 bucket plans fixed, and **training steps** (compute axes);
- **Data size** is an INDEPENDENT axis (not coupled to compute).

Scaling one axis alone is suboptimal; Stage 4 determines the ratio across
axes for the actual budget envelope. Small-card probes only.

- 4.0 **Re-estimate the hero recipe from budget and implemented performance**:
  establish remaining GPU-hour, API/storage, and practical wall-clock budgets,
  including queueing/preemption and preparation/evaluation/checkpoint costs.
  Start from corrected Stage-3 single-GPU throughput and memory; refresh
  end-to-end measurements on the Stage-3.5 data, caption bias, and bucket plans
  at each proposed resolution/model size. Measure target-GPU scaling when
  resources permit; otherwise label conservative assumptions explicitly.
  High-resolution DiT-only ceilings and old-timer DDP results are not measured
  end-to-end hero rates. Derive feasible model size, resolution-stage allocation,
  accumulation, actual samples/step, sample budgets, and step counts; for each
  stage, training GPU-hours = actual samples / measured global samples/s ×
  GPU count / 3600, with startup, evaluation, checkpoint, and restart allowances
  added separately. Replace the preimplementation estimates before selecting
  the hero recipe; revisit this estimate as the probes finish.
- 4.1 **Data-scaling probe** @256p: corpus arms 50K / 200K / 500K, fixed steps →
  KID + eval-loss slope → data-bound vs step-bound verdict → corpus size (D3).
- 4.2 **Resolution-transfer probe**: 256p→640p continued vs 640p-from-scratch (short arms)
  → confirm staging saves compute; check 640p→896p continued-training transfer (2.1
  already showed zero-shot 2.5× sampling transfer fails hard for both RoPE variants —
  progressive fine-tuning is the only path).
- 4.3 **Steps/quality curve** at chosen config → place the knee → steps per stage.
- 4.4 **Size/batch/steps grid** (small iso-compute arms around the 485M winner,
  e.g. 485M@32K vs 0.6-0.7B@~iso-compute, accumulation settings targeting
  average effective batches around 128 vs 256) → scaling-law slopes
  → recommended hero size×batch×steps for the stage-5 envelope.
- 4.5 Hero recipe card: corpus, mixture, stage steps, LR schedule, exit layer,
  arch/size, per-resolution bucket plans, `gradient_accumulation_steps`, measured
  actual samples/step and total samples, throughput/memory, GPU-hour and wall-clock
  estimates with uncertainty and overhead allowances — written to `notes/hero_recipe.md`.

**Exit**: budget-feasible recipe card committed; every measured number traceable
to a probe arm, every extrapolation labeled, and the old hero estimates replaced.

## Stage 5 — Hero run (≈1.6–2.2K 4090-h; single 8-GPU 4090 node, ≈8–12 days wall)

**Goal**: the model, using the budget-derived Stage-4 recipe. The following stage
fractions and headline compute/wall-clock ranges remain provisional:
640p bulk (~60–70% of steps) → 896p tail (~20–25%) → optional 1024p
polish with NTK scaling (~10%, only if 896p samples are clean and 1024p is wanted).

- Effective batch/accumulation and LR schedule from Stage 4, retaining chunked
  Muon LR 0.02 plus auxiliary AdamW as the optimizer starting point; logit-normal(0,1) +
  resolution time shift; caption dropout 0.1; caption-length curriculum; EMA 0.9999.
- Do not carry forward the old **60–100M samples in 2K 4090-h** projection as
  established throughput. Stage 4 supplies the actual per-resolution sample/step
  allocation and cost from measured performance. If its probes find the recipe
  step-bound rather than data-bound, consider extending hero hours within the
  agreed scheduling/budget policy.
- Checkpoints + eval suite every interval; watch: KID per domain, anatomy prompt pass
  rate, 1024p-collapse check, memorization probes on small domains.
- Mixture per locked table (dataset_plan.md §mixture; NC entries separate).

**Exit**: hero checkpoint passing the eval suite; if 1024p polish degrades, ship 896p and
note the latent-upscaler fallback.

## Stage 6 — NFT post-training (≈300–500 4090-h)

**Goal**: preference alignment on top of the hero checkpoint — the quality polish that
small base models can't get from data alone. Method: **DiffusionNFT** (Negative-aware
Fine-Tuning, [arXiv:2509.16117](https://arxiv.org/abs/2509.16117)) — online RL on the
*forward* process: per prompt, sample K candidate trajectories from the old policy, score
with a reward ensemble, normalize rewards to advantages, optimize the flow-matching
velocity predictor with a positive/negative contrast. No likelihood estimation, no SDE
reverse process — fits this codebase directly.

- 6.1 Reward ensemble (keep simple, watch hacking): LAION aesthetic scorer (cheap, local)
  + VLM-as-judge rubric for anatomy (hands/faces/body) and prompt adherence — the API VLM
  from stage 1 doubles as judge — + optional style classifier for 国画/impressionism
  authenticity.
- 6.2 Quick arm first: LoRA-NFT vs full-parameter NFT at 640p, short; pick on reward gain
  vs diversity loss. (LoRA is also the 4060 Ti-friendly variant.)
- 6.3 Main NFT run at 640p (sampling-bound: each policy step = K full denoises + VAE
  decode + reward scoring; 896p only if budget allows).
- 6.4 Guardrails: KL budget vs hero reference, diversity metric + canary prompts every
  interval, small LR, early stop on reward plateau or KID regression.

**Exit**: reward gain over hero baseline with no KID/diversity regression; final model +
before/after grids on the eval suite.

## Stage 7 — Publication readiness

**Goal**: a usable public release whose explanations stand on their own.

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
- 7.4 Add substantive explanatory notes on **Flow Matching** and
  **Negative-aware Fine-Tuning (NFT)**: motivation, equations/notation,
  training and sampling procedures, implementation mapping, assumptions,
  limitations, and accessible references—not merely experiment logs.

**Exit**: downloadable model and working Space; README quickstart verified from
a clean environment; public terminology/reference audit complete; Flow Matching
and NFT notes readable without access to internal conversations or artifacts.
Publication/demo hosting costs are estimated separately before deployment.

---

## Compute ledger (RTX4090-hours unless noted; caps)

| Stage | Cap | Notes |
|---|---|---|
| 0 Infra | 10 | mostly CPU; smokes on both platforms |
| 1 Data | 30 + API spend | API captioning replaces GPU captioning; cache raw responses |
| 2 Ablations | 400 | Inspire fair arms only; Andromeda takes smoke/qualitative arms (free, ~¼ speed); cap raised 200→400 on 2026-09-04 to fit stream-schedule + Muon axes |
| 3 Efficiency | 80 | buys back far more than it costs — gates stage 5 |
| 3.5 Captions / buckets / 640p | TBD + API/storage spend | pilot, then cost caption enrichment, refreshed precompute/eval, and bucket search |
| 4 Scaling ladder | 450 nominal | small runs; rebase hero recipe and budget on measured Stage-3/3.5 performance |
| 5 Hero | 1,600–2,200 provisional | target topology and wall time must be justified by Stage 4 |
| 6 NFT | 300–500 | sampling-bound |
| 7 Publication | TBD + hosting spend | model/Space release, documentation and reproducibility checks |
| **Original subtotal** | **≈2.9–3.7K 4090-h** | excludes new Stages 3.5/7 and API/storage/hosting spend; Stage 4 must update the complete ledger |

**Budget semantics (2026-08-26, user clarification; extended 2026-09-06)**: the
Inspire budget is about **not crowding out other users**, not an absolute hours
cap — scheduling GPU work into off-peak (late-night) troughs is explicitly
sanctioned, and exceeding the nominal ledger (e.g. 5K h) is acceptable if it
runs in troughs. **2026-09-06 priority policy (user)**: stage 2–4 experiments
(ablation arms, screens, probes, precomputes) run at **medium-high priority**
(`--priority 4`, platform maps to HIGH/NORMAL); **only the stage-5 hero run**
(the long multi-day training) sits at **LOW (priority 1, preemptible)** — idle
cards fill it whenever free and it can be stopped anytime. Preemption/resume
path proven by the 2.1 restarts and the k20/24 16:12 auto-restart.
Conversely, **avoid late-night runs on Andromeda** (shared desktop). Practical
rule: long Inspire jobs (stage 5 hero) run at priority 1; Andromeda arms run
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
- Cross-platform comparability: Andromeda results inform, never decide — fair arms live on
  Inspire 4090.
- 4090 PCIe-only DDP: earlier experiments identified per-micro-batch
  synchronization as a major cost; boundary-only reduction is retained.
  The old 92% scaling figure predates the corrected timer and was not rerun
  in the final single-GPU review. Re-measure target-topology scaling for Stage 4;
  do not assume eight-GPU efficiency from the old four-GPU result.
- VLM API dependency: rate limits / cost drift / provider model updates → cache raw
  responses; record exact model version in dataset metadata.
- 896p→1024p may degrade → polish stage optional; latent upscaler as documented fallback.
- NC-tagged data → no commercial release; `license` column + separate mix entries keep a
  clean variant feasible.
- Reward hacking / diversity collapse in NFT → guardrails in 6.4 are load-bearing.
