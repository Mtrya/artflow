# ArtFlow Reboot — Redesign Plan

Personal side project. Model ≤0.7B params. **Compute-frugal by design**:
total budget ≈ **2.9–3.7K RTX4090-hours** on Inspire (4090 @0.33 pt/h ⇒ ~1.0–1.2K points,
trivial vs the ~589K-pt budget of project 自动化科研 — wall-clock and queueing, not
money, are the constraints). Reference point: the old hero run was ~800 RTX4090-h
(256p only, unoptimized stack, 19.2M samples seen). The new recipe sees ~60–100M
samples at 640p-equivalent cost inside ~2K 4090-h — feasibility rests on the stage-3
efficiency work, which is why stage 3 gates stage 5.

Organized as a linear pipeline of stages, each with goal / tasks / exit criteria /
compute cap. A follow-up agent should be able to execute stage by stage from this file
plus `notes/dataset_plan.md` (data-source detail).

## Locked decisions

| # | Decision | Choice |
|---|---|---|
| D1 | License | Research-only OK (WikiArt, ArtBench-10, FFHQ unlocked). Per-sample `license` field; NC data in separate mix entries so a clean variant stays one mix-string away |
| D2 | Anatomy data | Photos + paintings both; ~50/50 face vs full-body |
| D3 | Corpus size | Set empirically by stage-4 scaling probe |
| D4 | Params | **485M: h1152×d24** — 2.2b (wide > deep at iso-param, every probe), 2.2c (iso-FLOP: ~400M > ~664M → 664M deferred to stage-4 probe), 2.2d (all-single) — all resolved 2026-09-05/06, records in stage2_ablations.md |
| D5 | Text encoder | Qwen3-0.6B, frozen, online; add early-exit-layer knob, ablate k ∈ {8,16,28} |
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
| Optimizer baseline | AdamW + linear_cosine + EMA 0.9999 | convention; Muon challenger in 2.5 |
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
stage 1 precomputes **256p only**; 640p/896p precompute is deferred to stages 4/5 when
the recipe (and thus the exact bucket sets) is known.

**Exit**: domain datasets validated; eval suite committed; 256p precomputed sets ready;
`data/` layout documented in `INSPIRE.md`.

## Stage 2 — Ablations @256p, small scale (≤400 4090-h on Inspire + Andromeda hours free)

**Goal**: pick architecture (depth/width, modulation, stream schedule), text-encoder
exit layer, optimizer, and validate the RoPE fix — cheaply, fairly. Fixed protocol per
design-dimension ledger §C. **Only compare arms run on the same platform** (Andromeda ≠
Inspire hardware; cross-platform comparisons are qualitative only). Per-arm configs,
telemetry spec, budget roll, and result records live in `notes/stage2_ablations.md`.

- Andromeda (4060 Ti; arms ≤15K steps, batch ≤64 via accum — it runs ~¼ 4090 speed):
  - 2.1 **RoPE fix smoke + A/B** (D7): new centered-grid/fixed-text-diagonal RoPE trains
    stably; then old-vs-new on resolution transfer (train 256p → sample 640p; artifact
    rate). This is the gate for everything below.
  - 2.2a **Modulation sharing**: `none` vs `layer`, h=1024, d=24, all-single-stream,
    10K-step screen on loss-curve separation. Runs **first** — its winner feeds
    2.2b–2.2d and fixes the param matching there. (Split out from the old 2.2a, which
    varied h, d, and mod simultaneously and could not attribute the delta.)
    — **RESOLVED 2026-09-05: mod=layer wins** (D11; records + curves in
    stage2_ablations.md). Scenario-A shapes in §5's shape table are active.
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

### Stage-2 decision memo (2026-09-06, all arms done — records in stage2_ablations.md)

**Hero recipe (256p stage-2 winner, ~485M):**

| Dimension | Winner | Evidence |
|---|---|---|
| RoPE | centered-grid (image), text fixed diagonal | 2.1: tie everywhere → Qwen-Image prior (user rule) |
| Modulation | mod=layer (shared per-layer MLP) | 2.2a: eval/loss + t040 5/5 + KID + speed/mem |
| Width×depth | h1152 × d24 | 2.2b: wide > deep at every probe, 0.28% @16K, KID agrees |
| Stream | all-single (no double-stream blocks) | 2.2d: monotone gradient 0.37%/0.65% @8K vs hybrid/all-double |
| Size | ~485M | 2.2c: iso-FLOP ~400M > ~664M → 664M deferred to stage-4 probe |
| Text exit | k=28 (last hidden state) | 2.3a: k28 < k8 < k16 at matched 4K |
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

## Stage 3 — Efficiency optimization (≤80 4090-h)

**Goal**: maximize samples/s before spending real compute. Gate: loss curves on a fixed
2K-step run match pre-optimization within noise (no numerics change).

4090-specific context: 48GB VRAM is roomy for ≤0.7B + online Qwen3-0.6B; **no NVLink** —
gradients all-reduce over PCIe, so amortize with grad accum (sync once per effective
batch) and DDP bucket overlap; single 8-GPU node, no multi-node.

- torch.compile per resolution bucket (static shapes within a bucket); fix graph breaks.
- SDPA flash/efficient backend selection, bf16 end-to-end, fused AdamW.
- Activation checkpointing **off** (48GB fits 640p/896p easily at 0.7B; it costs ~30%
  compute — re-enable only if the 1024p polish OOMs).
- Grad-accum tuning: fewer, larger micro-batches; measure sync overhead per accum step.
- Dataloader: benchmark latent-read throughput from shared disk; pre-shuffled shards,
  `num_workers`, pin_memory, prefetch. IO must never starve the GPUs.
- Online text encoding: batch/compile the frozen Qwen3; the stage-2 exit layer already
  cuts 40–70% of encoder FLOPs.
- Async checkpoint save; EMA off the critical path.
- VAE precompute throughput (batched GPU encode) for stage-4/5 precomputes.

**Exit**: throughput targets hit — ≥15–20 samples/s/GPU @256p, ≥4–6 @640p (post-optimization
measured values become the stage-5 sizing input); regression check passed.

## Stage 4 — Scaling ladder, small scale only (≤450 4090-h)

**Goal**: derive the hero recipe empirically instead of guessing. All at chosen arch/config.

- 4.1 **Data-scaling probe** @256p: corpus arms 50K / 200K / 500K, fixed steps →
  KID + eval-loss slope → data-bound vs step-bound verdict → corpus size (D3).
- 4.2 **Resolution-transfer probe**: 256p→640p continued vs 640p-from-scratch (short arms)
  → confirm staging saves compute; check 640p→896p continued-training transfer (2.1
  already showed zero-shot 2.5× sampling transfer fails hard for both RoPE variants —
  progressive fine-tuning is the only path).
- 4.3 **Steps/quality curve** at chosen config → place the knee → steps per stage.
- 4.4 Hero recipe card: corpus, mixture, stage steps, LR schedule, exit layer, arch —
  written to `notes/hero_recipe.md`.

**Exit**: recipe card committed; every number in it traceable to a probe arm.

## Stage 5 — Hero run (≈1.6–2.2K 4090-h; single 8-GPU 4090 node, ≈8–12 days wall)

**Goal**: the model. 640p bulk (~60–70% of steps) → 896p tail (~20–25%) → optional 1024p
polish with NTK scaling (~10%, only if 896p samples are clean and 1024p is wanted).

- Effective batch 256; LR 3e-4 linear_cosine (tuned per stage 4); logit-normal(0,1) +
  resolution time shift; caption dropout 0.1; caption-length curriculum; EMA 0.9999.
- At stage-3 throughput, 2K 4090-h ≈ **60–100M samples seen** at 640p-equivalent cost —
  3–5× the old run's 19.2M, at higher resolution, with better data and RoPE. If the
  stage-4 probe says we're step-bound rather than data-bound, extending hero hours is the
  sanctioned lever.
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
reverse process — fits this codebase directly. (If "NFT" meant crypto-minting instead,
flag it — that's a different stage.)

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

---

## Compute ledger (RTX4090-hours unless noted; caps)

| Stage | Cap | Notes |
|---|---|---|
| 0 Infra | 10 | mostly CPU; smokes on both platforms |
| 1 Data | 30 + API spend | API captioning replaces GPU captioning; cache raw responses |
| 2 Ablations | 400 | Inspire fair arms only; Andromeda takes smoke/qualitative arms (free, ~¼ speed); cap raised 200→400 on 2026-09-04 to fit stream-schedule + Muon axes |
| 3 Efficiency | 80 | buys back far more than it costs — gates stage 5 |
| 4 Scaling ladder | 450 | small runs only |
| 5 Hero | 1,600–2,200 | the only big spend; 8×4090, 8–12 days wall |
| 6 NFT | 300–500 | sampling-bound |
| **Total** | **≈2.9–3.7K 4090-h** | ≈1.0–1.2K pts @0.33 pt/h |

**Budget semantics (2026-08-26, user clarification)**: the Inspire budget is about
**not crowding out other users**, not an absolute hours cap — scheduling GPU work into
off-peak (late-night) troughs is explicitly sanctioned, and exceeding the nominal ledger
(e.g. 5K h) is acceptable if it runs in troughs. Conversely, **avoid late-night runs on
Andromeda** (shared desktop). Practical rule: long Inspire jobs (stage 2 fair arms,
stage 5 hero) are submitted to run overnight; Andromeda arms run daytime/evening.

## Open risks

- ~~Centered-vs-legacy RoPE~~ Resolved (2.1, 2026-09-05): tie at every probe down to
  320p → centered-grid RoPE adopted on the Qwen-Image prior. Zero-shot resolution
  extrapolation fails from 1.875× up regardless of variant → progressive fine-tuning
  is the only path to 640p/896p.
- Early-exit text features unvalidated → stage-2.3 decides; fallback k=28.
- Muon outcome propagates: if 2.5 picks Muon, the stage-4 steps/quality knee and the
  stage-5 LR schedule must be measured with Muon — no AdamW carryover. If chunking is
  skipped the arm tests a known-bad configuration (CMuon plateau), so chunked
  orthogonalization is part of the arm definition, not an optional tweak.
- Cross-platform comparability: Andromeda results inform, never decide — fair arms live on
  Inspire 4090.
- 4090 PCIe-only DDP: if stage 3 shows sync-bound training at batch 256, raise accum or
  shrink effective batch — don't buy multi-node.
- VLM API dependency: rate limits / cost drift / provider model updates → cache raw
  responses; record exact model version in dataset metadata.
- 896p→1024p may degrade → polish stage optional; latent upscaler as documented fallback.
- NC-tagged data → no commercial release; `license` column + separate mix entries keep a
  clean variant feasible.
- Reward hacking / diversity collapse in NFT → guardrails in 6.4 are load-bearing.
