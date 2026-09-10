# Long-caption enrichment pilot — measurements and decisions

Date: 2026-09-09.  Design and budget in [stage3_5_plan.md](stage3_5_plan.md);
roadmap in [redesign_plan.md](redesign_plan.md).  Everything below is measured
unless marked as an estimate.

## Caption coverage of the current corpus

Read from the twelve `length_metadata.npz` sidecars with the training prompt
contract (Qwen3-0.6B, system prefix dropped, retained-token cap).  Script:
`scripts/caption/audit_coverage.py`.

| Threshold (retained tokens) | Captions | Share of 2,895,079 |
|---|---:|---:|
| >= 128 | ~620,000 | 21% |
| >= 256 | 3,819 | 0.13% |
| >= 512 | 17 | 0.0006% |
| >= 1024 | 4 | 0.0001% |

Weighted by the training mixture and the within-row curriculum, the expected
share of caption draws at or above 256 tokens is 0.22%, and at or above 512 it
is zero.  Existing captions sit at 120-250 tokens (Chinese median 181, English
162).  **The long-caption range does not exist in the corpus at all; enrichment
creates it from scratch.**  The cap was lowered from 2048 to 1280 retained tokens
after the pilot, so the target bands are now 256-511, 512-1023 and 1024-1279.  Row-level ceiling (the most a within-row bias could reach
without new text) is 0.3% at 256 tokens.

## Where captioning can run

Measured 2026-09-09: the ZenMux API is reachable from this workstation only.
Both the Inspur notebook and the local 4060 Ti host time out on
`zenmux.ai` (DNS resolves, TCP connect fails).  Image transfer
Inspur -> workstation over `inspire notebook scp` measured 600 MB in 25.6 s
(24.6 MB/s) on a clean run, but an earlier identical transfer managed only
230 KB/s and stalled, so bulk transfer needs chunked retries.

## Artifacts in the training view

Merged the two artifact-detection passes into one lookup
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


The model under-runs long targets (median error -54 tokens, p10 -414).
gemini-3.8-flash reached 50% in the 1536-2048 band before the cap was lowered, and
gemini-3.1-flash-lite produced a 2,048-token caption on its single long-band test.

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

The ~$100 API ceiling assumed when this work started does not cover 250,000
rows with any working model.  This is the open budget decision.

## Length cap and target revised (user, 2026-09-09)

Maximum caption length lowered from 2048 to **1280 retained tokens**
(`MAX_SEQUENCE_LENGTH`), which removes the 1536-2048 band.  Bands are now
**256-511 / 512-1023 / 1024-1279**.  Target reduced from 250,000 to **150,000**
enriched rows.  Two acceptance improvements were made at the same time: the
length target moved from the band midpoint to one third of the way into the
band (models overshoot more than they undershoot), and markdown decoration
(code fences, `**`, leading `#`) is normalised away instead of causing a
reject.

## Provider and model routing after the revision

Measured on 40 shared images, same request per model, with the 1280-token cap:

| Model | Cost/request | 256-511 | 512-1023 | 1024-1279 | Overall |
|---|---:|---:|---:|---:|---:|
| google/gemini-2.5-flash-lite ($0.1/$0.4) | $0.00047 | 84% | 83% | 0% | 65% |
| google/gemini-3.1-flash-lite ($0.25/$1.5) | $0.00133 | 72% | 100% | 56% | 72% |

2.5-flash-lite cannot write a 1024-token caption at all, so the proposed
routing is 2.5-flash-lite for the 256-511 and 512-1023 bands and
3.1-flash-lite for 1024-1279.  At a 50/30/20 band split this costs about
**$140** for 150,000 accepted captions; at 50/35/15 about **$120**.

### DeepSeek (official API)

The user asked whether DeepSeek's own API could be used.  Findings:

- `deepseek-v4-pro` is **text only**.  On the official API it returns 200 but
  ignores the image: `prompt_tokens` was 101 for a 768 px image and the answer
  was "I'm unable to describe the image because it was not provided in a
  supported format".  ZenMux rejects it outright with 400 "Model do not
  support image input", and OpenRouter lists it as text-only too.  It does not
  route to a multimodal model.
- `deepseek-v4-flash-vision-exp` **does** accept images: 339 prompt tokens for
  a 768 px image, correct answer to a counting question.
- Published price per 1M tokens (off-peak / peak), from
  <https://api-docs.deepseek.com/quick_start/pricing>: v4-flash and
  v4-flash-vision-exp input cache-miss $0.22/$0.44, output $0.66/$1.32;
  v4-pro $0.66/$1.32 and $1.98/$3.96.  Off-peak is every hour except
  01:00-04:00 and 06:00-10:00 UTC Monday-Friday.
- Thinking mode is on by default and can be switched off with
  `thinking: {"type": "disabled"}` in the OpenAI-format request.  That matters:
  the earlier reasoning models we tried could not be switched off, which is why
  they returned empty answers.

At off-peak prices a 700-token caption costs roughly $0.0006 per request,
comparable to gemini-2.5-flash-lite, so `deepseek-v4-flash-vision-exp` is a
candidate for the long band if it complies with length instructions.

## Transfer constraint

The Inspur-to-workstation link is unstable: 24.6 MB/s on one clean run, 17 KB/s
during this session.  The 40,247 second-round thumbnails were reduced to 640 px
(2.1 GB) on the notebook, and transfer proceeds in 50 MB chunks so a stall
costs one chunk rather than the whole archive.

### DeepSeek vision measured through the pipeline (40 images)

`deepseek:deepseek-v4-flash-vision-exp` with `thinking: {"type": "disabled"}` —
the OpenAI-format switch, confirmed to work:

| Band | Acceptance | Mean retained | Target | Cost/request |
|---|---:|---:|---:|---:|
| 256-511 | 40% | 541 | 341 | $0.00059 |
| 512-1023 | 100% | 774 | 682 | $0.00059 |
| 1024-1279 | 56% | 1,076 | 1,109 | $0.00059 |

40 billed requests cost $0.0238, no reasoning tokens, 6.6 s mean latency.  It
overshoots badly on the shortest band (mean 541 against a 341 target), which is
where its 40% acceptance comes from; on the two longer bands it is competitive
with gemini-3.1-flash-lite at less than half the price.

### Routing settled by measurement (24 images per cell, 2026-09-09 evening)

The switch that turns DeepSeek's thinking off matters more than any other
setting: with no switch the model spends the whole output budget reasoning and
returns a near-empty caption (measured: 2,400 completion tokens, 78 characters
of content, `finish_reason=length`).  Both documented switches work and were
verified: OpenAI-format `thinking: {"type": "disabled"}` and
`reasoning_effort: "none"`.  The production configuration uses
`reasoning_effort: "none"`, and the two are interchangeable in cost and length
(mean retained 872 vs 924 tokens over ten long-band images).

| Band | Model | Setting | Accepted | In band | Mean retained | Cost/request |
|---|---|---|---:|---:|---:|---:|
| 256-511 | google/gemini-2.5-flash-lite | target 0.5 | 100% | 100% | 340 | $0.00035 |
| 512-895 | deepseek-v4.1-flash, reasoning off | target 0.6 | 96% | 88% | 657 | $0.00063 |
| 896-1280 | deepseek-v4.1-flash, reasoning off | target 1.0 | 83% | 25% | 872 | $0.00063 |

ZenMux-only alternative for the middle band: gemini-2.5-flash-lite at target
0.75 accepts 81% and lands 69% in band at a mean of 624 tokens for
$0.00043/request.  For the long band the alternative is gemini-3.1-flash-lite
(56% in band on the pre-1280 measurements, $0.0016/request).

Cost for 150,000 accepted captions at the 55/30/15 split, including the
requests that fail acceptance and have to be repeated:

| Plan | Cost |
|---|---:|
| gemini-2.5-flash-lite short + DeepSeek medium and long | ~$70 |
| gemini-2.5-flash-lite on all three bands, long band with gemini-3.1-flash-lite | ~$117 |

DeepSeek's off-peak window is every hour except 01:00-04:00 and 06:00-10:00 UTC
on weekdays, i.e. Beijing time 09:00-12:00 and 14:00-18:00 are peak.  Night
runs get the half rate used above.  The DeepSeek account balance is 75.74 CNY
(about $10.6), which funds roughly 17,000 requests — enough for the long band
but not for the middle band as well.  ZenMux funds the rest.

`deepseek-v4.1-flash-expires-on-0910` is the current id; on 2026-09-10 it
becomes `deepseek-v4-pro`, which by then routes to the same model.  That is a
one-line change in `configs/caption/routing.json`.

## Directives received after the routing review (user, 2026-09-09)

1. **Bands and shares**: 256-511 / 512-895 / 896-1280 at **55 / 30 / 15**.
2. **Routing**: 256-511 -> gemini-2.5-flash-lite; 512-895 -> gemini-3.1-flash
   (no model with that exact id exists on ZenMux or OpenRouter; the closest is
   gemini-3.1-flash-lite); 896-1280 -> DeepSeek.  Both bands 2 and 3 were
   re-measured against DeepSeek on the final band edges (see the routing table
   above) and DeepSeek wins on both length compliance and cost, so the settled
   routing is gemini-2.5-flash-lite for the short band and DeepSeek for the two
   longer ones.
3. **Markdown is acceptable** in captions.  Do not reject or strip it.
4. **Caption voice**: captions must read like a *user's prompt* to an image
   generator, not like an art catalogue entry.  Structured wording is allowed,
   but analytical headings such as "值得注意的细节" or "媒介、技法与画法" are too
   stiff.  The user's own example of a good caption is flowing prose that walks
   through the picture; the bad example is a six-heading art analysis.
5. **Text in the image goes into the caption itself.**  Any inscription,
   signature or seal text that is legible must be transcribed inside the
   caption, not in a separate field.  The earlier passes stored it in an
   `ocr_text` field; that field now has to be merged into the captions of the
   rows that already have it.

## Row selection: how the 150,000 are drawn

`scripts/caption/select_rows.py` runs against the precompute manifests
(`$W/data/meta/precompute`, one JSONL per training dataset) and emits one
request per selected row.  Rules, in order:

1. **Budget by draw share.**  Each dataset gets a share of the row budget equal
   to its weight in the training mixture, so enrichment lands where the trainer
   actually draws samples.  The mixture used is
   `d1:0.15 d2-wikiart:0.191 d2-museum:0.009 d3-human:0.076 d3-people:0.074
   d4-vintage:0.094 d4-zimage:0.022 d4-megalith:0.003 d4-inat:0.001
   d4-pd12m:0.079 d4-relaion:0.301`.
2. **Specialised top-up.**  A further 30% of the budget goes to Chinese painting
   on top of its draw share.  *This rule was inert until 2026-09-09 evening*:
   the allocation looked the dataset domain up in a table keyed by sub-source
   name (`d1_npm_tw`) while the mixture refers to datasets by manifest name
   (`d1`), so the lookup always missed and the top-up was spread over every
   dataset, leaving the allocation exactly proportional to the mixture.  The
   pilot selection shows it: `d1` received 48 of 320 rows, exactly its 15%
   weight.  The domain is now derived from the rows each manifest contains, and
   `tests/test_caption_selection.py` pins the behaviour.
3. **Stratify within a dataset.**  Rows are bucketed by sub-source, aspect
   shape and existing caption length, then drawn in a fixed pseudo-random order
   per bucket (seed 42), so a partial draw still spans the source.
4. **Length band by available detail.**  55/30/15 over 256-511 / 512-895 /
   896-1280; a detail crop or an image under 512x512 is capped at the middle
   band because it cannot support a 1,000-token description.
5. **Language and format by hash.**  Deterministic and independent of the band:
   Chinese/English 50:50 overall, 80:20 inside the specialised Chinese-art
   portion; prose/structured 50:50.
6. **Availability caps.**  A dataset cannot supply more rows than it has.  The
   shortfall is redistributed over the datasets that still have room, in
   proportion to their weights.

With the top-up fixed and `d1` capped by its usable rows, the 150,000 divide as
roughly 51,400 Chinese painting (its whole usable pool, absorbing the 30%
specialised budget), 34,900 relaion, 22,200 wikiart, 10,900 vintage, 9,200
pd12m, 8,800 human-recaption, 8,600 people, 2,600 z-image, 1,000 museum, and
under 500 across megalith and iNaturalist.  Without the fix `d1` would receive
only 22,500.

Artifact-flagged rows are excluded from the pool unless the second-round crop
recovers them, which is what caps `d1` (see below).

## Second-round crop: model choice

40,247 flagged rows were re-cropped on the already-cropped view (640 px
thumbnails, 2.1 GB, transferred in 41 x 50 MB chunks after the link proved
unstable).  Three models were tried on the same 40 rows:

| Model | Cost/request | Rows still showing artifacts | Artwork cut off |
|---|---:|---:|---|
| google/gemini-2.5-flash-lite | $0.00021 | 13/40 | yes, on 2 of 3 inspected rows |
| google/gemini-3.1-flash-lite | $0.00045 | 35/40 | not inspected |
| google/gemini-3.5-flash-lite | $0.00056 | 22/40 | no |

2.5-flash-lite is the cheapest but crops inside the artwork (on
`npm_tw-15554-K2A000006N000000000PAB` it removed the lower half of a scroll and
on `npm_tw-14718-K2A000011N000000000PAC` most of the painted panel), which is
the one failure this pass must not make.  3.1-flash-lite is too strict and
would discard most recoverable rows.  **3.5-flash-lite is used for the
production pass** at about $22.5 for 40,247 rows, inside the $25 the user
approved for it.  Mean box agreement with 2.5-flash-lite was IoU 0.76 and with
3.1-flash-lite 0.86.

## Scope narrowed to the Chinese-painting set (user, 2026-09-09)

Decision: finish the Chinese-painting component first and publish it for
review before touching any other dataset.

- **Two cropping rounds are enough.** The second-round box becomes *the* box:
  no `bbox_v2` field survives into the metadata or the dataset, there is one
  `bbox` per row.  No third verification pass.
- **No rows are dropped for artifacts.** All rows whose view is `full`,
  `mounted` or `detail` are kept; the second-round crop is applied as the
  training view.  (`rolled` and `junk` views were already removed when the
  metadata was built — 1,781 and 9 raw rows — so nothing further to drop.)
- **A part of the set gets a new long caption**, then the dataset is updated on
  HuggingFace and reviewed by hand.  Only after that does enrichment extend to
  the other datasets, and there it only adds long captions.

Measured on the second-round boxes produced so far: the new box keeps a median
of 90.4% of the first-round crop area (p10 68.1%), never larger than the
first-round box, and 58 of 9,946 rows so far shrink below 40% — those are the
rows whose color chart or label strip took up a large part of the frame.

### What the artifact list means (correction)

The second-round prompt shows the model the *already cropped* image and asks
for the artwork box inside it, plus which non-artwork objects are visible in
that same image.  The `artifacts` field therefore describes the view *before*
the new box is applied, not the result after cropping.  An earlier claim in
this note that it reports the state "after the second crop" was wrong.  Since
the decision is to keep every row anyway, the field is now informational only.

### DeepSeek caption quality (verified by reading images against captions)

Three images were read at 1024 px against their generated captions:

| Image | Checked | Result |
|---|---|---|
| npm_tw-15703-K2A001048N000000000PAG | layout, objects, calligraphy | accurate; the 燈草 instruction text transcribed character-for-character, matching the two visible columns |
| npm_tw-1186-K2A000824N000000000PAI | two travellers, yoke load, pine | accurate, including the round bundle on the front end of the pole and the rectangular basket behind |
| npm_tw-3709-K2A001072N000000000PBE | English long caption | accurate: the seal-script column reads 釀酒清醑聊適餘興毋殫沉酣愆儀兼 plus 令; wall, gate, figure with black dog, two figures at a table, scholar's rock, bamboo all present and correctly placed |

Voice is the requested one (a user describing the picture they want), and the
band-2 sample contained no evaluation phrases.  The one systematic weakness is
length: on the long band the model stops around 870 tokens against a 1,220
target, which still lands inside the acceptance window 83% of the time.

## Model choice for the Chinese-painting set: transcription decided it

On the same image (`npm_tw-3293-K2A000494N000000000PAW`) the two candidate
models were asked for the same caption band.  DeepSeek read the colophon as
雨餘春樹綠陰成 / 最愛西山向晚明 / 應有人家在山足 / 隔岸遥見白烟生 — which is
what the image says.  gemini-2.5-flash-lite produced 而餘春树标传成 /
承爱石山向收明 / 应有人家在山业 / 隔医逸见白姐生, and on
`npm_tw-3709-K2A001072N000000000PBE` it did not transcribe the seal-script
column at all where DeepSeek read 釀酒清醑聊適餘興毋殫沉酣愆儀兼 correctly.

Since a transcription that is wrong is worse than no transcription, the
Chinese-painting run uses DeepSeek on **all three bands**
(`configs/caption/routing_chinese_painting.json`) rather than the cheaper
gemini-2.5-flash-lite on the short band.  Cost for 60,750 rows is about $40 at
the off-peak rate, within the topped-up balance (373.49 CNY).

## Artist and title in the caption (user, 2026-09-09)

The harvested metadata carries the artist and the work's title, and the caption
may now name them where that reads naturally — or leave them out and describe
only the picture.  The prompt states both options and forbids naming anything
the metadata does not give.  The harvested strings are catalogue-shaped (the
artist lists Chinese and romanised names, the title carries a Chinese and an
English version), so `metadata_for_caption()` keeps the part matching the
caption's language and drops the romanisation before it reaches the model.

Verified on three rows: 范寬/雪山蕭寺圖 produced "画面上是范宽的《雪山萧寺图》。",
文徵明 was named inside a sentence about the inscription, and two rows with
metadata left the name out entirely.

## Enrichment runs on the notebook, not the workstation

The Inspur link truncates large transfers: 50 MB chunks arrived cut at 16–28 MB
and needed repeated retries, so moving 11 GB of thumbnails for the
Chinese-painting set would have taken hours with frequent stalls.  The notebook
**can** reach `api.deepseek.com` (verified: HTTP 401 without a key) even though
it cannot reach ZenMux, so with the Chinese-painting routing on DeepSeek the
whole enrichment runs where the images already are and no image is transferred.

Two changes made that possible:

- `scripts/caption/generate.py --no-tokenizer` skips token counting, which needs
  `transformers` (absent on the notebook).  Lengths are then measured locally by
  `scripts/caption/finalize_lengths.py`, which recomputes retained tokens with
  the training tokenizer and applies the acceptance window.  The request itself
  never depends on the tokenizer, so the captions are identical.
- Image reading is batched (`--batch-size`, default 2,000 rows per wave);
  holding every encoded image at once would be ~9 GB for 60,750 rows against
  15 GB of machine memory.

Measured on the notebook: 12.6 requests/s at concurrency 48, ~4 s mean latency,
about $0.00055 per caption by the price list, bands landing at 56/28/16 and
Chinese/English at 51/49 on the first 1,600 rows.

## Publication format: parquet with embedded images (user, 2026-09-09)

The HuggingFace copy must be a parquet dataset with the images inside it, so the
dataset viewer can preview every image; the WebDataset tar shards cannot be
previewed.  `scripts/data/hf_d1_parquet.py` writes one row per image with an
``image`` column of type ``struct<bytes: binary, path: string>`` — the layout the
``datasets`` library uses for its ``Image`` feature — plus the tombstone fields,
both captions, the long caption and the single final ``bbox``.  Shards are
bounded at 1 GB.  Publishing replaces the tar shards as the dataset's main form;
whether to delete the tars from the repository is open.

The preview images under `data/hf_d1/previews/` had a coordinate-frame bug: the
final box (normalised to the original photograph) was applied to a thumbnail
that was already cropped once, which made every crop look as if all four sides
had been cut.  The previews now use the second-round box as the model returned
it, i.e. normalised to the view it was shown.  The training and captioning paths
were never affected: they apply the metadata's ``bbox`` to the original image.

## Length is no longer a reason to drop a caption (user, 2026-09-09)

A caption that came out shorter or longer than its band asked for is kept: the
achieved length is recorded and the training bucket plan can accommodate any
value.  Only defects of the text itself (boilerplate opening, repeated
sentences) disqualify a caption.  Consequence for the bucket plan: it must not
assume captions stop at 1,280 retained tokens, so no truncation there.

Verified in the run's own output: all three bands were produced by
`deepseek-v4.1-flash-expires-on-0910` — 256-511, 512-895 and 896-1280 alike
(`configs/caption/routing_chinese_painting.json`), because that model reads
calligraphy correctly where gemini-2.5-flash-lite garbles it.

## Only repeated sentences disqualify a caption (user, 2026-09-09)

Boilerplate openings are now removed by a rule instead of causing a reject, and
a caption whose opening cannot be removed cleanly is kept anyway — regenerating
a caption to fix its first clause is not worth the API spend.  What is still
dropped is a caption that repeats itself, which is a defect of the text.

`strip_boilerplate()` drops a leading "this image shows" / "这张图片展示了" /
"这是一幅" and the verb that follows it, capitalises an English remainder, and
only acts when what remains is still a substantial part of the caption.
Examples from the test set: "这张图片展示了一位老人坐在松下，手里拿着一卷书。"
→ "一位老人坐在松下，手里拿着一卷书。"; "这幅画描绘了山峦叠嶂的景象…" →
"山峦叠嶂的景象…"; "This image shows a woman in a red robe…" → "A woman in a
red robe…".  "画面是一册打开的册页…" is left alone, because that opening reads
naturally.  The pass runs again over captions already generated, so nothing has
to be re-requested.

## Length policy for the remaining 70% (user, 2026-09-09)

The prompt still asks for a word/character count, but the model's answer is used
whatever length it comes out.  Training truncates at **2,048** retained tokens
(`MAX_SEQUENCE_LENGTH`, raised back from 1,280) and the bucket plan is decided
over that range, so the 845 captions over 1,280 tokens produced for the
Chinese-painting set are not lost.  `retained_length(..., truncate=False)` is
the measurement used to plan buckets; `truncate=True` is the training contract.

## Remaining work in this stage, in order (2026-09-10)

Frozen on 2026-09-11: 59,861 accepted captions for the Chinese-painting set, and
104,769 captions for the 105,001 selected rows of the other datasets
(`data/caption_enrich/production/captions_frozen.jsonl`), against 37,417 Pexels
captions and 2,433 evaluation captions composed separately.  All thirteen
training sources and the evaluation set now have 256p precompute directories
carrying rebuilt `length_metadata.npz` sidecars.

| # | Work | Gate | Owner | State |
|---|---|---|---|---|
| 1 | Chinese-painting republication: upload, viewer check, delete the 53 tar shards | none | agent | done |
| 2 | Hand-review of the published set | — | **user** | done 2026-09-10 |
| 3 | Enrich the other 105,001 rows (~$55–60 at the measured routing) | #2 | agent after #2 | done |
| 4 | Merge accepted captions into the training metadata and rebuild length sidecars | #3 | agent | done, in `precomputed_enriched/` |
| 5 | Selector policy: beta-softmax + short reserve, exposure simulation on the CPU | none | agent | code landed; no simulated exposure recorded |
| 6 | Bucket planner: exact boundary optimisation, then batch-size screening on the GPU | #4 for the real distribution | agent | planner landed; screening not run |
| 7 | Caption-length telemetry and evaluation suites/rubric | none for the code | agent | code and frozen suite landed |
| 8 | Screens A/B/C, then D/E; two-seed confirmation | #6, frozen tolerances | agent, user decides | not started |
| 9 | Precompute for the refreshed dataset (256p main, 640p subset) | #4 | agent | 256p done; 640p not run |

Decisions only the user can make: tolerances for the caption-policy comparison
(short-prompt regression, visual quality, minimum useful adherence gain) and the
evaluation suite sizes; bucket count K and the per-resolution micro-batch
settings; whether the 640p subset is prepared before or after the screens.

### Row budget

The 150,000-row target splits 105,000 broad / 45,000 specialised, and the
specialised half is Chinese painting.  Selecting the broad half while excluding
the Chinese-painting rows that the separate run already covers leaves
105,001 rows rather than 89,250, so the two selections together are 165,751
rows.  Either the count is accepted as is or the non-Chinese-painting selection
is trimmed back to 89,250; the difference is about $15 of API spend.

## Model availability after the DeepSeek id expired (2026-09-10)

The expiring id is gone in practice.  On 2026-08-10 the same eight long-band
requests that it had answered before returned **seven empty captions**
(`finish_reason=length`, 18,418 reasoning tokens): the `reasoning_effort: none`
switch no longer suppresses its thinking.  It cannot be used again.

Provider reachability from the notebook, measured:

| Endpoint | Result |
|---|---|
| `api.deepseek.com` | reachable, 0.14 s |
| `openrouter.ai` | reachable, 0.23 s |
| `zenmux.ai` | **blocked**: DNS resolves to 157.240.6.35, TCP 443 never connects |
| Gemini ids via OpenRouter | reachable but **HTTP 403 "not available in your region"** |

So from the machine that holds the images the only working caption models are
DeepSeek's own.  ZenMux is reachable from the workstation, but the thumbnails
are on the notebook, and moving them is the transfer that the earlier plan
avoided.

### Replacement model, measured on real selected rows

`deepseek-v4-flash-vision-exp` with `reasoning_effort: none` (reasoning tokens
zero on every request), 10 rows per band from the non-Chinese-painting
selection, lengths measured with the training tokenizer:

| Band | n | empty | text accepted | in band | median | min–max | cost/request |
|---|---:|---:|---:|---:|---:|---|---:|
| 256-511 | 10 | 0 | 100% | 70% | 493 | 383–544 | $0.00049 |
| 512-895 | 20 | 0 | 100% | 85% | 636 | 486–829 | $0.00062 |
| 896-1280 | 20 | 0 | 95% | 35% | 853 | 485–3,178 | $0.00087 |

For comparison, the expired id measured 100% / 96% / 83% acceptance and 100% /
88% / 25% in band on the same bands.  The replacement is close except on the
long band, where its median lands just under the floor and one caption ran to
3,178 tokens.

Grounding was checked by reading two of the probe images against their
captions: a Saint Francis receiving the stigmata panel (halo, red seraph,
stigmata rays, red church, gilded arch and frame) and a brass-framed portrait
miniature (ring, tapered stem, rivets around the oval band).  Every claim
matched; no boilerplate openings, no invented provenance.

Cost for the 105,001 selected rows at these rates is about **$61–65**, against a
DeepSeek balance of 264.73 CNY (about $37).

## User decisions, 2026-09-10

1. **Row budget**: 105,001 rows for the non-Chinese-painting batch are accepted,
   so the two batches together are 165,751 rows.  If the balance runs out
   mid-run, stop and report rather than improvise.
2. **Publishing**: only `kaupane/chinese-painting-collection` is published.  The
   other datasets are too heterogeneous to be a coherent public dataset.
3. **Bucket count K**: an input to the planner, not a Stage 3.5 decision; use any
   default for testing (10 is fine).  Micro-batch size is screened later by
   fixing the boundaries and the model, then measuring throughput per candidate
   batch size.
4. **DeepSeek**: the expired id is not to be revived; use `deepseek-v4.1-flash`
   when it launches officially.
