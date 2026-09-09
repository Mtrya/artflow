# Stage 3.5 pilot — measurements and decisions

Date: 2026-09-09.  Companion to [stage3_5_plan.md](stage3_5_plan.md).  Everything
below is measured unless marked as an estimate.

## Caption coverage of the current corpus

Read from the twelve `length_metadata.npz` sidecars with the training prompt
contract (Qwen3-0.6B, system prefix dropped, 2048 cap).  Script:
`scripts/caption/audit_coverage.py`.

| Threshold (retained tokens) | Captions | Share of 2,895,079 |
|---|---:|---:|
| >= 128 | ~620,000 | 21% |
| >= 256 | 3,819 | 0.13% |
| >= 512 | 17 | 0.0006% |
| >= 1024 | 4 | 0.0001% |

Weighted by the Stage-3 mixture and the within-row curriculum, the expected
share of caption draws at or above 256 tokens is 0.22%, and at or above 512 it
is zero.  Existing captions sit at 120-250 tokens (Chinese median 181, English
162).  **The 256-2048 range does not exist in the corpus; enrichment creates it
from scratch.**  Row-level ceiling (the most a within-row bias could reach
without new text) is 0.3% at 256 tokens.

## Where captioning can run

Measured 2026-09-09: the ZenMux API is reachable from this workstation only.
Both the Inspur notebook and the local 4060 Ti host time out on
`zenmux.ai` (DNS resolves, TCP connect fails).  Image transfer
Inspur -> workstation over `inspire notebook scp` measured 600 MB in 25.6 s
(24.6 MB/s) on a clean run, but an earlier identical transfer managed only
230 KB/s and stalled, so bulk transfer needs chunked retries.

## Artifacts in the training view

Merged the two Stage-1 artifact passes into one lookup
(`scripts/caption/build_artifact_index.py`): 103,434 labelled rows, **41,292
flagged (39.9%)**.

| Source | Rows | Flagged | Rate |
|---|---:|---:|---:|
| npm_tw c0 | 38,693 | 6,029 | 15.6% |
| npm_tw c1 | 18,627 | 5,850 | 31.4% |
| npm_tw c2 | 7,777 | 6,889 | 88.6% |
| npm_tw c3 | 24,233 | 22,524 | 92.9% |
| museum sets (15 sets) | 16,170 | 1,429 | 8.8% |

Kinds: label 35,432, colour chart 31,801, ruler 28,120, desk 8,432, other 1,759,
glare 355.  The first-round bounding box does not always remove them: on
`npm_tw-19656-K2A003655N000000000PAW` the box `[0, 194, 1000, 1000]` removes
the accession strip at the top and leaves the Kodak grey-scale bar on the left.

Consequences: (a) artifact-flagged rows are excluded from caption enrichment,
which reduces the Chinese-painting dataset from 90,669 to 51,397 usable rows;
(b) a second cropping round is being run over all 40,247 flagged rows that
reach a manifest, asking for the artwork box inside the already-cropped view and
composing the two boxes (`scripts/caption/refine_bbox.py`).

## Model comparison, 40 shared images

Same images, same request (language, format, length band) per model.

| Model | Accepted | In band | Cost/image | Latency | Reasoning tokens |
|---|---:|---:|---:|---:|---:|
| google/gemini-3.5-flash-lite | 67.5% | 77.5% | $0.0019 | 5.7 s | 0 |
| google/gemini-3.1-flash-lite | 75% (n=8) | 88% | $0.0018 | 7.6 s | 0 |
| google/gemini-3.8-flash | 67.5% | 72.5% | $0.0040 | 13.4 s | ~1,150/req |
| z-ai/glm-5.3-flash | 0% | 0% | - | 84 s | 1,293-3,206/req |
| bytedance/doubao-seed-2.0-lite | 0% | 0% | - | 58 s | 1,625/req |
| deepseek/deepseek-v4-flash-vision-exp | 67% (n=6) | 67% | $0.0088 | 113 s | 12,340/req |
| qwen/qwen3.8-flash | 38% (n=8) | 38% | $0.0009 | 27 s | 1,052/req |
| openrouter:nex-agi/nex-n2.5-pro:free | not finished | | $0 | 16 s | 318/req |

The price difference is real and matches the published list exactly (verified
against billed usage): glm is 4x cheaper on input and 10x cheaper on output
than gemini-3.5-flash-lite.  It is nevertheless unusable here: the provider
refuses to disable reasoning ("该模型始终思考，不支持关闭思考"), and with this
prompt it spends the whole output budget thinking.  `effort=low` still used
1,293-1,345 reasoning tokens of a 1,350 budget; `effort=minimal` used 3,206 per
request.  All 48 test requests returned empty content.

Length compliance of gemini-3.5-flash-lite falls off with the band:

| Band | In band | Mean retained | Target |
|---|---:|---:|---:|
| 256-511 | 100% | 387 | 383 |
| 512-1023 | 83% | 593 | 767 |
| 1024-1535 | 20% | 851 | 1,279 |
| 1536-2048 | 0% | 1,354 | 1,792 |

The model under-runs long targets (median error -54 tokens, p10 -414).
gemini-3.8-flash reaches 50% in the 1536-2048 band, and gemini-3.1-flash-lite
produced a 2,048-token caption on its single long-band test.

## Review of generated captions

Contact sheets of four images at 512 px with six deliberately corrupted captions
injected (`scripts/caption/review.py`).  Twenty of the forty outputs were read
against their images.

Grounding was accurate in every case examined, including seal positions,
inscription text ("雨餘春樹緑陰成", "一月七日文壁記", "雲間莫是龍"), tent shapes,
animal markings and tree structure.  Two systematic weaknesses:

- **Inscription-heavy captions.** On calligraphy pages the model spends most of
  its length transcribing text, which is accurate but turns the caption into a
  text dump rather than a description.
- **Controls are hard to catch at contact-sheet resolution.** One injected
  single-fact error in a 1,500-token caption was not detectable at 512 px.  The
  production audit therefore needs full-resolution single-image reads for its
  random sample, not contact sheets.

## Cost arithmetic at the requested scale

250,000 enriched rows, at the observed acceptance rates:

| Routing | Cost |
|---|---:|
| all gemini-3.5-flash-lite (370K requests at $0.0019) | ~$700 |
| all gemini-3.1-flash-lite | ~$600 |
| mixed: flash-lite/lite for short bands, 3.8-flash for long bands | ~$750 |
| all glm-5.3-flash | not achievable |

The plan's ~$100 API ceiling does not cover 250,000 rows with any working
model.  This is the open budget decision.
