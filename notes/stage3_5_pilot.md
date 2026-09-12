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

## A stalled screening run logs nothing (2026-09-11)

The batch-size pilot screened nine of the ten length buckets at batch 4 and 16,
then stalled on `res1-len8` at batch 16: no output for 57 minutes, a zero-byte
`run.log`, and the training process still alive when the job was stopped.

Two candidate explanations were checked rather than assumed.

*The bucket is not too rare to fill a batch.*  Counting retained lengths in the
sidecars gives 3,834 rows carrying 3,834 captions in that bucket, against a
micro-batch of 16, so the sampler has material to draw from.

*Shared memory is at the platform default.*  A one-GPU probe job reports
`shm 64M 0 64M 0% /dev/shm`, and `/proc/mounts` agrees at `size=65536k`.  That
does not make it the cause: a batch in this pipeline carries latent tensors and
token ids, not the text encoder's hidden states, and is far smaller than 64 MB.
Every later submission passes `--shm-size 16` regardless, because no training
step should depend on a 64 MB mount.

The zero-byte log had a mundane cause and is fixed.  A run's stdout is a pipe,
so Python block-buffers it and nothing reaches the log until the process exits,
which makes a stalled run indistinguishable from one that never started.  The
screen now sets `PYTHONUNBUFFERED=1` for the runs it launches.

## What isolating one bucket costs in sampling (2026-09-11)

A run of the stalling case was reproduced with unbuffered output and a job-shell
look at its GPU: utilization 0% while 69% of the device stayed allocated.  So it
was not training, and not a deadlock either — it was waiting for data.

The screen isolates a bucket by giving every other bucket a batch size no run
can reach, so a run emits only the bucket under test.  Filling one micro-batch
of that bucket then costs as many draws as it takes for enough rows to land in
it, and the rare buckets are rare.  Measured on the CPU against the real
sidecars and the frozen plan — `res1`, 30 steps, accumulation 16, the
13-source mixture:

| bucket bound | micro-batch | draws per micro-batch | draws for one run | sampling time |
|---:|---:|---:|---:|---:|
| 90 | 4 | 125 | 0.06M | seconds |
| 952 | 4 | 7,688 | 3.7M | 1.3 min |
| 621 | 16 | 20,577 | 9.9M | 2.3 min |
| 952 | 16 | 40,998 | 19.7M | 4.9 min |
| 2048 | 16 | 414,114 | 198.8M | 46.3 min |

One draw costs 14–21 microseconds, so the cost follows how rare the bucket is,
not how large the micro-batch is: the 2048-token bucket holds 391 captions
across the whole corpus, and filling a micro-batch of 16 from it takes about
four hundred thousand draws.  `--min-caption-share 0.005` already refuses that
bucket, which is why the pilot never started it.

These are single-process numbers, and a run has eight dataloader workers drawing
in parallel, so the wall-clock share is smaller than the table.  They are still
large enough to explain a run at the 952-token bucket not finishing inside its
hour, and they set the timeout the screen needs: 30 minutes per run, with the
rarest buckets bounded by the share filter rather than by patience.

Should this become the binding constraint, the sampler's draw loop is where to
fix it: a draw walks the corpus and most of the work in an isolated run is rows
the sink buckets immediately discard.

The sweep is therefore split in two rounds, because the cost of isolating a
bucket depends on the bucket and the payoff of a large micro-batch does not:
the long buckets are already flat between micro-batch 4 and 16, while the
common buckets are still 2.3x faster at 16 than at 4.

- **Round one** screens micro-batches 64 and 128, and only buckets holding at
  least 5% of a resolution's draw mass — the common buckets, where a large
  micro-batch is both affordable to sample and likely to pay.
- **Round two** screens micro-batches 16 and 32 over every bucket holding at
  least 0.1%, which stops at the 952-token bucket. The 2048-token bucket holds
  391 captions across the corpus and falls through to the declared fallback of
  16 rather than being screened.

The two rounds are complementary: no (bucket, candidate) pair is screened
twice, and the union covers four candidates for the common buckets and two for
the rest.

Round one has finished for every resolution.  Micro-batch 128 is out of reach:
it ran out of memory in 39 of the 40 combinations, and the one it completed
(resolution 4, smallest bucket) came in at 8.87 ms/sample against 9.05 for 64 —
under 2% for a batch that sits on the device limit.  Micro-batch 64 is the
feasible optimum for the common buckets, and it still beats 16 there (res1
bucket 0: 16 measured 10.84, 64 measured 9.21).

## 640p precompute: buckets and measured shapes (2026-09-11)

The 640p pass runs the same pipeline over the same manifests with the 640p
bucket list, so it needs no new code beyond the resolution table.  A smoke run
over 40 real rows of `d4_vintage` confirms the whole path and reports what each
bucket actually produces, which the 640p length-bucket plan needs as its image
token count:

| resolution id | bucket | latent shape | patch-2 image tokens |
|---:|---|---:|---:|
| 1 | 640x640 | (16, 80, 80) | 1600 |
| 2 | 848x480 | (16, 60, 106) | 1590 |
| 3 | 480x848 | (16, 106, 60) | 1590 |
| 4 | 736x560 | (16, 70, 92) | 1610 |
| 5 | 560x736 | (16, 92, 70) | 1610 |

All 40 rows survived the filters, and the run encoded 2.73 rows per second on
one GPU.  Across the 1.64M manifest rows split over eight workers that is about
20 hours of wall clock, which is why the pass was launched before the batch-size
sweep finished rather than after.

The 640p length-bucket plan waits for that pass: the caption lengths are the
same as at 256p, but the rows it keeps are the high-resolution subset, and only
the finished sidecars say which those are.  The plan is not on the critical path
— the resolution enters hero training long after the first stages do.

## Choosing micro-batches: fill the card, then line the buckets up (2026-09-11)

The sweep finished with 111 successful runs, 31 out-of-memory results, and no
time-outs.

Peak memory turns out to be almost exactly linear in micro-batch size, with a
constant term that is the same for every bucket (9.40-9.65 GB: weights,
optimizer state, compiled kernels) and a slope that follows the sequence
length (0.28 GB per sample at the 26-token bucket, 0.73 at 478).  Because the
time per sample falls with micro-batch size, the best size is the largest one
the device holds, which makes this a memory question rather than a search over
throughput.

Filling the card also lines the buckets up in time, which is not obvious and
was worth checking: `time = batch x per-sample time`, and both the per-sample
time and the memory slope rise with sequence length, so the two partly cancel.
Measured micro-batch times fell from a 3.5x spread (570-1976 ms) to 1.1-1.35x
(904-1235 ms) under that rule.

That rule cannot be applied blindly, though.  The fit puts the small-token
buckets at 100-110, but the sweep only reached 64 there: micro-batch 128 ran
out of memory in 39 of 40 combinations, and the one it completed sat within 2%
of 64.  Two buckets with nearly identical slopes (0.2797 and 0.2835) disagreed
on whether 128 fits, so the limit is set by allocation peaks rather than by
average use.  The plan therefore stays inside the screened sizes.

What the plan does use is the alignment idea, solved over measured candidates
only.  Every rank draws its own micro-batches and the reduction at the step
boundary waits for the slowest one, so a step's wall clock is the slowest
rank's.  Choosing per bucket the largest candidate whose micro-batch time stays
under a target `T`, and sweeping `T`, gives:

| rule | slowest rank over mean | effective ms/sample |
|---|---:|---:|
| each bucket at its fastest (T = none) | +12.65% | 12.522 |
| aligned, T = 800 ms | +6.84% | 12.034 |
| aligned, T = 1300 ms | +8.54% | 12.486 |

**Aligned at 800 ms wins by 4.2%.**  It does not drive the wait to zero: the
small-token buckets cannot be made slower to match the long ones, because 64 is
already their memory ceiling, so some spread survives.  Pushing the target down
to 500 ms is worse (13.0 ms/sample) because past that point the rule buys
alignment by shrinking the step.

The frozen plan is identical across all five resolutions: micro-batch 64 for
the buckets up to 90 tokens, 32 up to 223, and 16 beyond.  This is the plan the
caption-policy comparison runs use.

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

## Coverage of original versus enriched captions is not measured (2026-09-11)

The telemetry section of the plan asks for a breakdown of original versus added
captions.  It is dropped as a metric rather than left unimplemented by accident:
a row's caption list is assembled from whatever rounds produced its dataset, so
the first entry is no more "original" than the last is "enriched".  Several of
these corpora — `human-recaption`, `relaion-art-recap-zh`, `vintage-photography-captions`,
`artbench-captions`, `wikiart-captions` — already carry captions that a model
wrote in an earlier round, which is why re-captioning them was worth doing at
all.  A split along that line would describe the production history of the corpus
rather than a property of the training stream, and the sampler does not act on
it.  Caption length, which the sampler and the bucket plan do act on, is counted
in full.

## Screening arms, revised by exposure measurement (2026-09-11)

The exposure simulation (`scripts/simulate_exposure.py`) was run on the real
corpus with the 13-source mixture, 3.0M nominal draws per arm.  Share of draws
whose retained caption length is at least 256 tokens:

| arm | ramp average | curriculum held at its end value |
|---|---:|---:|
| A legacy | 7.68% | 7.68% |
| B beta -1..+1 | 6.24% | 11.58% |
| C beta -0.5..+1.5 | 8.75% | 12.88% |
| D stationary at B | 6.25% | 11.58% |
| E earlier transition | 8.92% | 11.58% |

Two things follow.  E is dropped: its total ties C (262.4k against 267.5k draws)
*and* its timing nearly matches (40.1% against 41.5% of those draws delivered by
halfway), so it is a second path to roughly the same stream and carries the least
information of the five.  B against D stays, because it is the only ordering
contrast whose totals are matched by construction (187.3k against 187.4k, and
31.7% against 55.0% delivered by halfway).

A gradient-weighting arm replaces E.  Weighting each length band by the value the
log curve takes there (about 1 / 1.5 / 2.45 / 3.0 / 3.55) moves the long bands'
share of the gradient from 6.2% to 11.0% on the ramp, and from 11.6% to 19.7%
when the curriculum is held at its end value.  That is a larger move than the
whole spread across A/B/C/D/E, and it composes with any of them, so it is the
stronger lever of the two.

The screens as designed run the ramp, where long-caption exposure is 6-9%.  The
hero run is expected to hold the curriculum at its end value for its later
resolution stages, where the same policies give 11.6-12.9%.  A screening
difference therefore understates what the hero run would see, and the endpoint
choice matters less there than the ramp shape suggests (+1.3 percentage points
between B and C, against +2.5 on the ramp).

Removing pd12m from the mixture was simulated as a separate scenario and moves
the long share by 0.3 percentage points at most; it is not a decision input.

## Densify pass over the coarse grid, and two launch failures worth recording

The first screen swept a coarse grid (16/32/64, with 128 on the common
buckets).  128 ran out of memory in 39 of 40 combinations, which is exactly
what the fitted memory line predicts - at 0.85 of the 47.37 GB cards the
formula's own ceiling for the small-token buckets is about 108, so 128 was
never a candidate the model believed in.  The grid simply skipped the region
the model points at.  A densify pass now probes that region: 80/96 where 64
passed and 128 failed, 40/48 (or 48/64) where 64 was never tried, 20/24 where
32 failed.  The screen tool grew a --bucket-batch-sizes option for this: each
bucket gets its own candidate list, buckets no entry names are skipped, and
every run records peak reserved memory alongside allocated (the trainer now
logs it), so the pass doubles as calibration data for the reserved-memory
offset the formula needs.

First res5 point, as a sanity check of the linear model: 27 tokens, batch 96,
allocated 36.6 GB against a linear prediction of 36.5, reserved 40.1 - a
~3.5 GB allocator overhead on top of allocated, comfortably inside the card.

The caption-policy arms failed twice at launch before the first training step:

1. The job launcher did not export ARTFLOW_ROOT, the same failure the 640p
   precompute launcher had; the guard at the top of the script caught it.
2. With that fixed, the smoke run died in 13 seconds: the generated run
   config put eval_interval under [eval], but the schema owns it under
   [train] (base.toml has it there, between checkpoint_interval and
   steady_state_skip_steps).  The smoke overlay had the same mistake, and the
   three caption-policy files had never been synced to the cluster repo at
   all.  All four stacks now pass src.train.config.load_config locally before
   anything was resubmitted - that check is free and catches exactly this
   class of error, so it is part of the submission routine from here on.
3. The third failure went one loader further: the trainer tried to load the
   text encoder as the hub id "Qwen/Qwen3-0.6B", because base.toml names hub
   ids for local development and the generated cluster config overrode
   neither [text_encoder].path nor [paths].vae.  Both now point at the flat
   directories under the shared models root, every path the config names was
   verified to exist on the shared disk before resubmission, and the full
   layer stack passes load_config locally.  Lesson folded into the routine:
   a cluster run config must override every hub id in base.toml, and the
   existence check costs one exec call.

## Gradient weighting: continuous curve

The user confirmed the gradient weighting on 2026-09-11 as a continuous
curve, not a table: f(L) = max(1, log2(L/128)) evaluated at the sample's own
retained length (128->1.0, 512->2.0, 1024->3.0, 2048->4.0).  A per-band
table would invent boundaries the training has no use for and would need a
code change every time the length cap moved; the curve extends itself to
whatever length shows up, which is exactly why the 2048-token ceiling needed
no decision and no code change.  Config: caption_loss_weight_curve = "log2",
caption_loss_weight_reference = 128 ("none" is the default and means
unweighted).  The gradient-weighting arm derives from the B/C winner: the
winner's selection policy plus this curve, nothing else changed, so the arm
isolates the weighting's own contribution.  Watch caption/weight_mean (mean
weight over conditioned samples) and caption/grad_weighted_mean_tokens
(sum of w*L over sum of w) during the run and compare against the offline
prediction of the same two numbers, to confirm the realized gradient share
matches the intent and no single dataset comes to dominate through the
weighting.

## Swanlab telemetry overhaul (2026-09-11, revised later the same day)

The logged-metric set had grown to ~350 series - every counter dumped
windowed *and* cumulative, plus three series per (resolution, bucket) pair -
and the user called it unreadable self-indulgence.  The first pass trimmed
the set but kept banded shares with a fifth band bolted on at 1281_2048; the
user then rejected bands outright: the 256/511/895/1280 edges were target
lengths from an old captioning prompt and carry no meaning for training, and
five bands distinguished for no reason is historical baggage.  The redesign
asks what a training-curve review actually looks at and reports only that:
caption length mean and p50/p90/p99 (histogram ceiling at the 2048-token
cap), dropout rate, padding fraction, samples per micro-batch, repeat rate
and the run's unique-row count, policy strength - plus the trainer's own
loss/lr/grad_norm/samples_per_sec/mem_peak and per-dataset mix ratios.  On
weighted runs the banded weight shares are replaced by the two moments above
(weight_mean, grad_weighted_mean_tokens), which answer "where did the
gradient go" without inventing boundaries.  Runs already in flight keep the
old format; the next launches pick this up.

## The unscreened tail bucket is what the smoke run is for

The fourth arms launch died 12 steps into the smoke run with a CUDA OOM in the
DiT forward.  Root cause: the plan's longest bucket (2048 retained tokens)
carried batch 16, a *fallback* value - the coarse screen never measured it
(round b stopped at the ~1000-token bucket, because 2048 holds 391 captions in
the whole corpus and isolating it costs more sampling than the measurement is
worth).  Extrapolating the fitted memory slope gives ~27 GB allocated at batch
8 versus ~44 GB at batch 16, right at the physical edge where allocator noise
decides.  Earlier launches never reached a long-bucket micro-batch, so only
the smoke run's full-mix stream hit it - which is exactly the smoke run's job,
ten minutes instead of fifteen hours.  The interim plan now caps the 2048
bucket at batch 8 in every resolution; the final plan will give this bucket a
computed size from the analytic model plus one sparse validation point, which
is cheaper than screening it through the sampler.

## Full-mix OOM: not fragmentation, genuinely live memory growth

The fifth arms launch OOMed at the same step as the fourth even after the
2048-token bucket was capped at batch 8, so the offender was never that
bucket.  A 60-step diagnostic on the screened plan with
expandable_segments=True then separated the two hypotheses cleanly: the
reserved-but-unallocated pool dropped from 3.2 GB to 0.1 GB - the allocator
change works - yet the run still OOMed, at step 17 instead of 12, with 46.68
GB *allocated*.  The later failure point with tighter packing means the
process's live memory grows step over step until the card is full; it is not
one oversized shape and not fragmentation.  Isolated-bucket screen runs never
see this because they hold one shape for 30 steps; the full-mix stream cycles
through every (resolution, bucket) shape.

Two diagnostics now run with allocation-history capture (a new env-gated
hook, ARTFLOW_MEM_SNAPSHOT, dumps which allocation stacks hold memory at
exit): the screened plan under expandable_segments, and the uniform batch-16
plan that earlier full-mix runs used, under the stock allocator.  If the
uniform plan also fills the card, the growth lives in the current training
code (the caption telemetry and loss-weight path are new since those runs)
and no plan is safe until it is found; if it survives, the constraint is
plan-level and the arms can start on the uniform plan while the screened plan
gets a full-mix validation gate before adoption.

Until this is resolved the arms stay unsubmitted: five launches have now
failed at or before step 17, and iterating hypotheses on 4-card jobs wastes
more GPU time than the diagnosis costs.

## Full-mix OOM, corrected: one forward pass, not a leak

The "live memory grows step over step" reading above did not survive the
allocation snapshots.  Three snapshot dumps (screened plan, uniform batch-16
plan, and a fully eager control - `bisect-eager-0911`, which also OOMed at
step 30 with the identical 46.6 GB allocated) all show the same picture: the
active blocks attributed to the DiT forward (`train.py` model call) number
857 and come in per-layer multiples of 23/24 (23-24 blocks of each size
class, e.g. 235.0 MiB x23, 88.0 MiB x46 = two per layer).  That is exactly
one forward pass's activation graph, held for backward - not an accumulation
across steps.  Eager failing identically to compiled rules out
torch.compile; the accumulator and caption telemetry were re-read and hold
only detached scalars and host floats, so no Python-side reference retention
either.

What the snapshots say the killer micro-batch is: fitting the five
self-consistent block-size classes (qkv output 132 MiB bf16, gated-FFN up
235 MiB, SiLU halves 117.5 MiB, fp32 RMSNorm outputs 88 MiB, bf16 44 MiB)
gives B x S = 20,050-20,112 elements across the board - batch 16 with a
~1250-token padded text at 256p (256 image tokens + a ~1000-token caption
bucket).  One such forward needs ~37.7 GiB of activations; with the ~9.5 GiB
baseline (weights, EMA, optimizer state, text encoder) that is 46-47 GiB,
right at the 47.4 GiB card.  Why "the same shape" screened clean at 29.7 GB
(`res5-len8/B16`, 30 steps) while the full-mix run dies on it is the open
question; the per-micro-batch shape log (new env-gated `[shape]` line ahead
of the forward, ARTFLOW_LOG_SHAPES) plus a full-mix rerun
(`shapeprobe-fullmix-0911`) pins down the exact bucket and the memory level
the forward started from.

Two earlier claims are retracted: that the 2048-token bucket cap at batch 8
was the fix (the OOM shape is the ~1000-token bucket, not 2048), and that
the failure proves step-over-step growth (the step-wise climb in the memory
telemetry is the curriculum/sampler stream reaching heavier buckets, and
each OOM lands on the first micro-batch whose single forward exceeds the
card).

## Root cause closed: dynamo recompile limit, not memory pressure

The shape probe (`shapeprobe-fullmix-0911`, per-micro-batch `[shape]` log)
settled the open question above.  The killer micro-batch was `res=4,
txt_hi=1000, B=16` (S = 252 image tokens + 1000 text), and the log shows
memory flat at 8.57 GB the moment its forward started - nothing accumulates
across steps or shapes.  The same log carries
`torch._dynamo hit config.recompile_limit (64)`: 38 distinct shapes had been
served by step 17, and dynamo spends more than one cache entry per shape
while specializing, so the 64-entry per-code-object cache (shared by all 24
compiled blocks) was already exhausted.  A shape first served after the
limit runs **eagerly**, and an eager forward at this size keeps ~38 GB of
activations (fp32 norm outputs, unfused RoPE, gated-FFN halves) versus ~20
GB for the compiled graph - which is why the isolated screen (one shape,
always compiled) measured the very same micro-batch at 29.7 GB peak while
the mixed stream dies on it at 46.6 GB.  The step at which a run died (12 /
17 / 30) was simply when the first big bucket arrived after cache
exhaustion, and the eager control reproduced the failure because eager is
the fallback state itself.

Fix: the recompile-limit floor in `train.py` goes from 64 to 512, above any
plausible plan's shape count (5 resolutions x 10 length buckets, plus
specialization variants).  Consequence worth remembering: the screened plan's
batch sizes are only valid while every served shape stays compiled; an eager
fallback at a B16 long-caption bucket does not fit the card.  Validation:
`fullmix-fix-0911` reruns the 60-step full-mix stream on the screened plan
with the shape log on.

Validation passed: `fullmix-fix-0911` ran the full 60-step mixed stream on the
screened plan with no OOM - peak 39.1 GB allocated / 43.8 GB reserved on the
47.4 GB card, steady 51.8 samples/s.  The run served and survived both the
former killer shape (res5/1001-token/B16, twice) and the never-screened
2048-token bucket at batch 8 (twice), so every shape in the plan is now
empirically covered.  The caption-policy arms were resubmitted as
`caption-arms-0911f` (smoke gate first, then three one-card arms).

## Analytic memory model vs measurement (2026-09-11 late)

The isolated screen measurements make the memory model essentially exact.
Over 163 successful runs (all five resolutions, all buckets, batch sizes
16-128), a per-bucket linear fit `peak_allocated = const + slope x batch`
gives const = 9.40-9.60 GB everywhere (weights, optimizer state, compiled
kernels) and a worst-case residual of 0.09 GB.  The slope itself is linear in
total sequence length (image tokens + bucket text ceiling): slope =
0.00101 GB per sample per token, with a max fit residual of 0.005.  So for
any shape, allocated peak is

    9.5 GB + batch x seq_len x 0.00101 GB

with sub-0.1 GB error on every one of the 40 screened (resolution, bucket)
pairs.  This is the "the second step does not need a GPU" result: batch sizes
follow from the architecture, and measurements only calibrate the two
constants.

Reserved memory sits a roughly constant 3.0 GB above allocated (60
measurements from the densify pass, which records both; range 1.2-4.4, larger
offsets at smaller allocations).  The one full-mix run on the screened plan
measured 39.1 GB allocated / 43.8 GB reserved, an offset of 4.7 GB.

One honest gap remains.  The model predicts the heaviest shapes of the
screened plan at ~28-30 GB allocated (B16 x 1252 tokens -> 29.7, B8 x 2304
-> 28.1), but the full-mix run peaked at 39.1 GB - about 10 GB above any
single-shape prediction.  The shape log shows only ~8.6 GB live before each
forward, so the surplus is not baseline drift; the mechanism is per-shape
retained allocations that accumulate as the mixed stream compiles and serves
its ~50 shapes, which an isolated screen (one shape) never sees.  The
surcharge grows with the sizes of the shapes in the plan: a later variant
with the short buckets at 96/80 instead of 64 died inside its first step at
44.9 GB live, so it cannot be treated as a fixed constant either.  The
practical rule: analytic sizes from the formula, then a mixed-stream run as
the final check - the plan is only valid once that run survives.

## Time-alignment rule lands in the merge tool; T=1000 confirmed (2026-09-12)

`merge_screen_results.py` now implements the frozen selection rule instead of
per-bucket lowest-ms/sample: per bucket pick the largest measured candidate
whose micro-batch time (batch x ms/sample) is at most T, sweep T, score each
T by simulating the 8-rank step (wall clock = slowest rank; effective
ms/sample = slowest-rank step time over mean samples), keep the T with the
lowest effective time.  Bucket draw mass comes from the sampler-faithful
model (dataset by weight, then row, then caption) via the new
`--bucket-mass-out` on `batch_size_screen.py`; without `--bucket-mass` the
merge refuses to run rather than silently degrading to per-bucket minimum.

On the real caches the optimum is exactly T=1000 ms: effective 12.506
ms/sample versus 13.183 for "each bucket at its fastest" (alignment buys
5.1%), slowest-rank premium +4.50%.  The earlier ad-hoc scan's 11.597
reproduces only under its approximate mass model (weight x captions instead
of weight/rows per caption); the optimum and the chosen sizes are identical
under both, so the conclusion stands and the absolute number was corrected.

Finding worth recording: the screened plan the caption-policy arms run on is
the T=800 choice over the pre-densify rounds plus the manual 8-cap on the
2048-token bucket, i.e. it predates both the densify measurements and the
T=1000 optimum.  That is fine for the arm comparison (all arms share the
plan, and it is validated), but the final 256p plan for the next phase is the
T=1000 output: short buckets go from 64 to 96/80, and the 2048 bucket stays
at 8.  Whether the larger short-bucket sizes survive the full-mix memory
surcharge (~10 GB over the single-shape prediction, see the memory-model
section) is exactly what `fullmix-t1000-0912` tests: the first 60 steps of
the real stream on the T=1000 plan, same protocol as the earlier full-mix
validation.

## 640p plan: preliminary boundaries, and two data findings (2026-09-12)

Boundary solving runs on the notebook over the real 640p sidecars (14
sources; d3-people and d4-zimage still being precomputed, d4-inat excluded -
see below).  Image tokens at 640p: 1600/1590/1590/1610/1610.  K=10 boundaries
land at roughly 24/48/88/120/175/230/430/650/950/2048 per resolution with
padding overhead 1.2-1.8% against the padding-free stream (80-88% less waste
than uniform bounds).  Batch sizes in `640p-k10.prelim.json` are placeholders;
they get filled by the analytic model once the sparse 640p probe
(`probe-640p-0912`, buckets 0/5/7/9 on resolutions 1 and 4, candidates
bracketing the model's prediction) confirms the 256p constants transfer to
~1600 image tokens.

Two data findings from the 640p precompute close-out:

- d4-inat has zero 640p-eligible rows: every row fails the pre-resolution
  filter (res_pre drop), so the source is 256p-only.  It is excluded from the
  640p mix rather than carried as an empty entry.
- The precompute worker's per-source "done" check (dataset_info.json exists)
  is fooled by an all-dropped source: d4-inat@640p holds a dataset_info.json
  with no data shards.  When the remaining-sources job finishes, verify each
  output has data shards, not just the info file.

## T=1000 plan does not fit the full mix; frozen plan is the final 256p plan (2026-09-12)

`fullmix-t1000-0912` ran the 60-step mixed-stream diagnostic on the T=1000
plan (short buckets at 96/80 instead of 64) and died inside the first
optimizer step: a text-encoder attention allocation (~2.5 GB, the B8 x
2048-token bucket's encode) failed with 44.94 GB already live.  The stream
had served several B96/B80 micro-batches before that point, so the mixed-run
memory surcharge is not a fixed constant: it grows with the sizes of the
shapes served (per-compiled-shape retained allocations), and with the larger
short buckets it pushed the run past the card.  The isolated screens stay
exact - every candidate measured there fits on its own - but the full-mix
peak is an empirical property of the whole plan, and the only check that
matters is the mixed-stream run.

Consequences:

- The frozen screened plan (short buckets at 64, 2048-token bucket at 8) is
  already at the ceiling: its full-mix validation measured 39.1 GB
  allocated / 43.8 GB reserved on the 47.4 GB card.  The memory-constrained
  version of the T rule (fixed-surcharge model, 45 GB reserved ceiling) moves
  almost nothing, and where it disagrees with the frozen plan the frozen
  plan's own measurement wins, so **the frozen plan stands as the final 256p
  plan**.  The ~2% the unconstrained T=1000 choice would have bought is not
  available at this memory ceiling.
- Plan batch sizes are only valid after a mixed-stream validation; isolated
  screens bound, they do not certify.  This is now the standing rule, and it
  is what the 640p plan will do: analytic sizes from the memory model
  (calibrated by the sparse probe), then one mixed-stream run to confirm.

## The stage delivers the planning pipeline, not a plan (user, 2026-09-12)

The bucket plan cannot be finalized in this stage on principle: the model
shape and the data mixture are only decided later, and both move the
boundaries and the batch sizes.  What this stage delivers is the pipeline
that produces a plan, as tools plus the order they run in:

1. `scripts/plan_buckets.py` - boundaries from the corpus' retained-length
   distribution (the sampler-faithful draw model: dataset by weight, row,
   caption).
2. `scripts/bench/size_micro_batches.py table` - analytic per-bucket batch
   sizes from the memory model (const/slope/reserved-offset/surcharge are
   parameters; a new model shape or resolution recalibrates them with a
   sparse probe, not a full screen).
3. Mixed-stream validation - a short run over the real mixture with
   ARTFLOW_LOG_SHAPES=1; the plan is only valid once this survives.
4. `scripts/bench/size_micro_batches.py trim` - if validation OOMs, the log
   (live memory + failed allocation + last [shape] line) yields the largest
   batch that fits by proportion through the linear model: one computation,
   never a retreat to the next power of two.  Trim, re-validate, repeat if
   needed.

Where measurements already exist (the 256p screens), the time-alignment rule
in `merge_screen_results.py` still applies between steps 2 and 3 to trade
per-bucket speed for step alignment, subject to the memory ceiling.

Sanity anchor for the analytic table: at the calibrated 256p constants it
reproduces the frozen plan's binding constraints - batch 8 for the
2048-token bucket (the empirically validated cap) and 16 at the ~1250-token
bucket - while showing the headroom the isolated screens saw at short
buckets (73 against the full-mix-validated 64), which is precisely the gap
the mixed-stream validation arbitrates.

## Directives (user, 2026-09-12 early morning)

1. Arm verdict delegated: when the caption-policy arms finish, if one arm is
   a clear winner on eval loss/KID plus sample inspection, pick it without
   waiting and launch the gradient-weighting arm (winner's policy + the log2
   loss-weight curve, nothing else changed) immediately.  Only stop for the
   user when there is no clear winner.
2. The synthetic East-Asian set (d3-synth, 20k rows) is too low-quality to
   keep as-is.  Rescue plan: (a) a VLM filter pass that drops images with
   fundamental flaws - broken anatomy, a person seen from behind wearing
   clothes drawn for the front, wrong subject counts, garbled rendered text,
   and similar hard failures; (b) a fresh caption pass over the survivors,
   where the captioning prompt must NOT include the generation prompt (it
   biases what the model says toward what was requested rather than what is
   there) - image in, caption out, following the same natural-language,
   length-tiered prompt contract as the other recaptioning rounds.  The
   precomputed d3-synth entries and its HF publication need to be redone from
   the rescued manifest afterwards.

## The 256p memory constants transfer to 640p exactly (2026-09-12)

First four points of the sparse 640p probe (`probe-640p-0912`), against the
model fitted at 256p (allocated = 9.5 + batch x seq x 0.00101 GB):

| point | seq tokens | predicted | measured |
|---|---|---:|---:|
| res1 bucket0 B16 | 1624 | 35.7 GB | 35.7 GB |
| res4 bucket0 B16 | 1635 | 35.9 GB | 35.9 GB |
| res1 bucket5 B16 | 1775 | 38.2 GB | 38.1 GB |
| res4 bucket5 B16 | 1789 | 38.4 GB | 39.3 GB |
| res1 bucket0 B24 | 1624 | 48.9 GB | OOM |
| res1 bucket5 B20 | 1775 | 45.4 GB | OOM |

Both fits land on the prediction and both predicted OOMs OOM.  The per-token
activation slope is an architecture property, not a resolution property, so
the model needs no recalibration at ~1600 image tokens; bucket 7/9 points
will confirm the long-caption end.  Throughput at 640p is 67-81 ms/sample
against ~11.6 at 256p - the expected ~6x.  This makes the 640p plan a pure
computation: boundaries from the planner, sizes from the formula, then one
mixed-stream validation run.

Bucket 7/9 points, added after the probe finished (2026-09-12):

| point | seq tokens | predicted | measured |
|---|---|---:|---:|
| res1 bucket7 B12 | 2170 | 35.8 GB | 35.7 GB |
| res1 bucket7 B16 | 2170 | 44.6 GB | 44.5 GB |
| res4 bucket7 B12 | 2253 | 36.8 GB | 36.7 GB |
| res4 bucket7 B16 | 2253 | 45.9 GB alloc -> OOM | OOM |

The long-caption end confirms the model the same way: fits land within
0.1 GB and the predicted OOM OOMs.  Two operational consequences:

- At 640p the long-caption buckets are memory-bound at small batches:
  res1 bucket7 fits at most B16 (44.5 GB allocated, 46.8 GB reserved, a
  1.2% margin on the 47.4 GB card), and per-sample time was still falling
  at that edge - throughput wants a bigger batch, memory forbids it.
- The long tail is too rare at 640p to screen at all: res1/res4 bucket9
  (998-2048] holds ~0.2% of draws (1018 captions), so the sampler needs
  ~5900 draws per emitted sample (~162 ms of pure sampling CPU per sample,
  against ~110 ms of GPU time).  Even bucket7 pays 8-31 ms/sample of
  sampler work.  Boundary planning at 640p has to weigh merging or
  truncating the far tail, not just the GPU cost of the bucket.

## Synthetic set rescue: full run (2026-09-12)

The 60-row pilot's verdict held at scale, with one prompt iteration in
between.  v4 of the filter prompt missed two defect classes that are plainly
visible in the corpus: studio gear and photographers intruding into the scene
(softboxes, light stands, a hand holding a phone reaching in from the edge),
and pictures nested inside a drawn frame (book pages, hanging-scroll mounts,
fan and oval borders, plain decorative borders).  v6 adds a checklist item and
two flaw labels for them (`intrusion`, `frame`).  On a 14-image regression
set - 7 corrupted controls, 5 confirmed pilot misses, 2 clean images - the v6
union (a row passes only when both Gemini 2.5 Flash Lite and Gemini 3.1 Flash
Lite find nothing) scores 14/14.

Full run over all 20,000 rows:

- 2.5 alone keeps 82.3%, 3.1 alone keeps 57.7%, union keeps **10,893
  (54.5%)**.  Rejections: frame 3625, anatomy 3585, garbled text 3094,
  intrusion 1976, cloned repetition 910, fusion 419, quality 243.  The two v6
  classes account for 40% of rejections - they were the difference between
  the pilot prompt and this one.  Spot checks of rejected thumbnails confirmed
  the catches are real (scroll-mounted paintings, oval fan frames, softboxes
  flanking the subject).
- Re-captioning of the survivors: 10,881 accepted of 10,893 (10 repeated
  sentences, 2 empty).  Bands 64-255/256-511 went to 2.5, 512-895/896-1280 to
  3.1 (2.5 measurably cannot write the long bands); split 40/25/25/10,
  language follows the row's original caption (zh 5,286 / en 5,607).
  Hedging phrases remain in 25% of captions - recorded, not rejected.
- Cost: filter $25.0 ($7.0 + $18.0), re-caption $9.3, total **$34.4**.
- Transfer note: the notebook-to-local channel decays to ~10 KB/s on
  long-lived connections; re-chunking the 1.9 GB of thumbnails into 40 x 36 MB
  tars, each fetched on a fresh connection, moved everything in 2 minutes.

Publication: `kaupane/east-asian-dress-synthetic` gains `rescue_ok`,
`rescue_flaws` and `caption_rescued` columns with an updated card; all 20,000
rows stay, the screened subset is a filter away.  The precomputed d3-synth
datasets (256p/640p) are NOT rebuilt yet - data-side rebuilds wait until the
arm experiments finish (user, 2026-09-12).

## 640p d3_people: timeout kill, resubmitted split across both GPUs (2026-09-12)

`precompute-640p-rest-0911` was stopped by its time limit with d3_people at
86% (98,250/114,361).  save_to_disk writes only when a source finishes, so
the interrupted source left nothing - 5.4 GPU-hours lost.  d4_zimage (45,248
rows) had finished cleanly before the stop (38 shards on disk).

Fix: the people manifest was split into alternating halves
(`d3_people_a.jsonl` / `d3_people_b.jsonl`, 57,181/57,180 rows) and
resubmitted as `precompute-640p-people-0912`, one half per GPU
(`precompute_640p_people_split.sh`).  The two output dirs
(`d3-people-a@640p`, `d3-people-b@640p`) enter the 640p mix as two entries,
each with half of the people weight - no merge step, no format risk.  Each
half measures ~11.6 rows/s (vs 3.7 when it shared the node with zimage), so
completion is ~1.3 h instead of a further 8.6 h on one GPU.

## 640p full-mix plan: boundaries + analytic sizes, validation launched (2026-09-12)

`precompute-640p-people-0912` finished both halves (46 shards each, rc=0), so
every 640p source now exists on disk.  The boundary pass was re-run over the
full 640p mix: the 256p domain weights minus d4-inat (no 640p-eligible rows),
d3-people carried as its two halves at 0.0298 each, d4-zimage at 0.0177, 17
mix entries total.  K=10 boundaries land at roughly 25/48/88/120/180/235/
420/640/990/2048 with padding overhead 1.3-1.8% against the padding-free
stream (81-87% less waste than uniform bounds), essentially unchanged from
the 14-source preliminary pass
(`bucket_plans/640p-k10.full.json` + report).

`size_micro_batches.py table` filled the batch sizes analytically from the
calibrated memory model (const 9.5 GB, slope 0.00101 GB/token, 10 GB
surcharge, 47.4 GB card): 12 for the short buckets, 11 through the middle,
9-10 at ~430-650 tokens, 7-8 at ~1000, 5 at the 2048 cap
(`bucket_plans/640p-k10.full.sized.json`).  The long tail stays rare in the
full mix (share above 1024 tokens: 0.04-0.12% of draws per resolution), so
the sampler-cost caveat from the probe stands.

Mixed-stream validation is `diag-fullmix-640p-0912` (1x4090, 60 steps,
ARTFLOW_LOG_SHAPES=1, configs `run-640p.toml` + `diag-fullmix-640p.toml`).
The plan is only valid if this run survives; if it OOMs, `size_micro_batches.py
trim` computes the fitting size from the log.  Note d3-synth@640p is still
the pre-rescue build - its rebuild waits for the arm experiments (see the
directive above), which shifts the synth length distribution slightly; the
boundaries get one more pass when that lands.

## Eval probe OOM: full-probe text encode, fixed by chunked encoding (2026-09-12)

The first 640p mixed-stream validation (`diag-fullmix-640p-0912`) died before
the first training step: `EvalLossProbe.__init__` pre-encodes its whole
caption set in ONE `encode_text` call, and that call runs the text encoder
with `output_hidden_states=True`, keeping every layer's hidden state for the
whole batch live simultaneously.  The octave-banded probe (5 bands x 512
samples) draws ~1364 captions from light-eval; the forward's own retained
hidden states grew to 44.94 GB and the next 2.5 GB attention allocation
failed.  The earlier probes survived only because the previous banding drew
~589 captions - the failure is a probe-size threshold, not a 640p property,
and real 640p training runs would have hit the same wall at startup.

Fix: `src/evaluation/eval_loss.py` encodes the probe captions in chunks of 64
(`_encode_caption_chunks`) and re-pads the chunk outputs to the widest chunk.
Right padding is masked, so per-row embeddings are unchanged - the probe
stays comparable with the already-running arms.  424 tests pass, including
two new ones for the chunking.  Validation resubmitted as
`diag-fullmix-640p-0912b`.

## 640p full-mix validation: PASSED (2026-09-12)

`diag-fullmix-640p-0912b` ran the full 60-step 640p stream on the analytic
plan (17 mix entries, `640p-k10.full.sized.json`) with the chunked eval probe
fix in place: no OOM, peak 30.1 GB allocated / 30.9 GB reserved on the 47.4 GB
card, 10.35 samples/s steady-state.  Shape logging shows the long end was
actually served within the 60 steps - txt_hi 2048, 998, 986, 872 all appear -
so the validation covers the memory-binding buckets, not just the common
shapes.

One calibration finding: the observed mixed-stream surcharge is much smaller
than the model's conservative 10 GB constant (the peak sits ~17 GB under the
card).  The plan is valid as sized; when Stage 4 builds the real plan it can
recalibrate the surcharge against this measurement for denser batches, and
re-validate.  The pipeline order holds: boundaries -> analytic sizes ->
mixed-stream validation, with trim as the fallback.

## Caption-policy verdict: arm B (beta selector + short reserve, linear ramp) (2026-09-12)

Final eval loss at step 5000: A (legacy reference) 0.88691, B 0.88668,
C (shifted ramp) 0.88840.  A and B are a numerical tie on aggregate; C is
behind on aggregate and on every band.  Per-band at 5000 (with the caveat
that the 512_895 band holds only 3 eval samples, so its differences are
noise): band_lt256 A 0.92519 / B 0.92489 / C 0.92612; band_256_511 A 0.63201
/ B 0.63224 / C 0.63721; band_512_895 A 0.64080 / B 0.64276 / C 0.64738.

Tiebreak by sample inspection (user, on the SwanLab generations): B's faces
are clearly better than A and C on two of the three inspected portrait
prompts (baroque noblewoman; 少女侧脸特写), and slightly weaker on the third
(elderly man close-up).  With the numbers tied and the visual read 2:1 for B,
**arm B's policy is the selection**: the beta selector on exact retained
lengths with the 0.20 short reserve under the linear -1 -> +1 ramp.  C's
upward-shifted ramp is rejected - it pays short-band quality for nothing.

The gradient-weighting arm (B's policy + caption_loss_weight_curve="log2",
reference 128, everything else unchanged) launched as `arm-log2w-0912`,
config `configs/caption-policy/w-beta-ramp-log2.toml`, run name arm-log2w.
Watch caption/weight_mean and caption/grad_weighted_mean_tokens against the
offline prediction.  KID-at-end numbers for A/B/C were still computing at
verdict time; they will be appended when the arms job exits.

## Synth v2 precompute launched; cleanup requirements for wrap-up (user, 2026-09-12)

The rescued synthetic manifest `d3_synth_v2.jsonl` holds the 10,893 surviving
rows, each carrying the original user-style grid prompt plus the rescued
caption where one was accepted (10,881 of 10,893; the 12 rejected by the
caption pass keep only the prompt).  `precompute-synth-v2-0912` encodes it at
256p and 640p into `d3-synth-v2@{256p,640p}`; the live d3-synth@ dirs are
untouched so the running arms keep their data.  At wrap-up the 640p boundaries
get one more pass with the v2 source, and the run configs repoint.

Wrap-up cleanup requirements (user, 2026-09-12):

1. Delete engineering/temporary scripts rather than keeping them in git.
2. Sweep new comments and docs for internal jargon and references only
   meaningful with private context (an outside reader must be able to
   understand every comment, name and doc).
3. Sweep for stale code superseded by this stage's changes and remove it.
4. Then rebuild-check d3-synth-v2, commit, push, and look at Stage 4.

KID at step 5000 (appended after the arms job exited): A 0.00761±0.00337,
B 0.00818±0.00352, C 0.00673±0.00314 - all three inside each other's error
bars at this sample size, so KID carries no signal for the verdict either
way.  The verdict stands on the eval-loss tie between A and B plus the 2:1
visual read for B.

## 896p eligibility survey (2026-09-12)

Proxy rule: pixel area >= 896^2 and aspect within the 640p bucket family's
range (480/848 .. 848/480).  Exact counts where the manifest carries
width/height, header-probe estimates (500-image sample) where it does not
(d1, d2_wikiart, d3_people, d4_zimage, d4_inat).  Six eligible images per
source were eyeballed with their captions (samples in
$W/data/quality_896p/, contact sheets kept locally).

| source | rows | 896p-eligible | share | visual/caption read |
|---|---:|---:|---:|---|
| d1 | 91,438 | ~81,400 | ~89% | eligible subset dominated by large book-page scans: 5/6 sampled show album-leaf borders, side calibration strips; captions honestly describe the page framing |
| d2_wikiart | 214,460 | ~74,600 | ~35% | 6/6 genuine artworks |
| d2_museum | 10,074 | 4,063 | 40% | good; one framed miniature |
| d3_human | 120,218 | 98,058 | 82% | mostly good photos; 1/6 was a DVD cover with text |
| d3_people | 114,361 | ~100,600 | ~88% | content drifts past people (neon sign, bare sky, coastline) via the merged pexels sets; captions stay honest |
| d3_pexels | 37,417 | 35,814 | 96% | excellent: 6/6 people in cultural dress, sharp |
| d3_synth_v2 | 10,893 | 6,542 | 60% | post-rescue samples look clean |
| d4_vintage | 194,625 | 16,539 | 8.5% | good vintage B&W |
| d4_zimage | 45,248 | 0 | 0% | generated below 896p - out by resolution |
| d4_megalith | 5,504 | 274 | 5% | tiny and junky (web miscellany) |
| d4_inat | 2,823 | 0 | 0% | out by resolution, as at 640p |
| d4_pd12m | 162,307 | 137,108 | 85% | OK; 2/6 show print page borders/mounts; captions plain English with the known prefix |
| d4_relaion | 622,329 | 148,207 | 24% | good; includes some illustration/anime |

Total eligible pool roughly 700k rows with d1, ~620k without.  Sources out
by resolution: d4_zimage, d4_inat.  d4_megalith contributes 274 rows - not
worth a mix entry.  The open question is d1: its 896p-eligible subset skews
hard toward the uncropped book scans, so either the crop pipeline improves
first or d1 enters 896p at a reduced weight.

Correction to the d1 row of the 896p survey above (2026-09-12): the first d1
sample read the raw scans, but training applies the manifest's normalized
[0,1000] bbox at precompute.  Re-sampled with the crop applied: eligibility
is unchanged at ~89% (~81k rows, 445/500 probe; 42% of rows carry a bbox).
The crops did remove the desk background and calibration strips.  What
remains: for the album-leaf subcollection (e.g. 毛诗品物图考) the crop box
follows the whole page, so the printed decorative border and the vertical
column text outside the illustration stay in frame (2/6 of the eligible
sample), and one sample kept a catalog label strip at the bottom (1/6).
Captions describe the page framing honestly.  Options put to the user:
accept as-is, a small targeted VLM pass over the album-leaf rows, or reduced
d1 weight at 896p.

896p decisions (user, 2026-09-12): d1 enters as-is - the album-leaf pages
with their printed borders are valuable data, not noise.  d4_megalith enters
too (its captions are Claude-written, high quality; 274 eligible rows).
d4_zimage and d4_inat stay out (no 896p-eligible rows).  For the 896p
*training mix* (not precompute): Chinese painting, art and people get a
small upweight, d4_pd12m a small downweight - exact weights are a Stage 4
mix decision.

896p precompute launched as two jobs over 12 workers:
`precompute-896p-a-0912` (8 GPUs: wikiart, pd12m, vintage, human, people,
d1, relaion p0/p1) and `precompute-896p-b-0912` (4 GPUs: relaion p2/p3/p4,
pexels + synth_v2 + museum + megalith).  Buckets are the 640p family scaled
by 1.4: (896,896), (1184,672), (672,1184), (1024,768), (768,1024); batch 25
(the VAE encoder's peak at 896px is ~2x the 640p one, halved batch keeps the
same headroom); caption cap 8192 as at 640p.  ~1.58M manifest rows to scan,
~700k expected to land.  Max-time 12 h per job.

896p job A resubmitted with a 24 h limit (2026-09-12): the first launch's
12 h max-time was under the slowest workers' need (pd12m at 3.7 rows/s ->
~12.4 h, vintage -> ~11.6 h), and save_to_disk writes only at source end, so
the deadline would have discarded both.  `precompute-896p-a-0912` stopped
~30 min in (~3.7 GPU-hours lost) and replaced by `precompute-896p-a2-0912`
(same 8 workers, 24 h).  Job B's workers all finish within ~2 h, unchanged.

## Wrap-up sequencing hazard (2026-09-12)

`[eval] num_samples` and `[eval] compute_metrics` were dead: the trainer
renames them to flat names and nothing reads those, so `compute_metrics = true`
in a run config promised metrics that were never computed.  Removing them
changes which keys a config may carry, so the run configs on the shared disk
have to be edited in the same step - and only once no job is loading those
files, because a run resolves its config at process start, resume included.

For the same reason, a config on the shared disk must not be repointed while a
run that reads it is still alive: `run-256p.toml` is layered under every arm's
config, so a crash-resume after a repoint would silently continue that arm on
a different distribution.

## Length-weighting telemetry verified against the offline prediction (2026-09-12)

The prediction reuses the planner's sampler-faithful draw distribution p(l)
(same code path as the boundary solver: dataset by weight, row uniformly,
caption inside the row by the policy's probabilities, three phases merged with
equal weights), then evaluates the curve from `src/train/caption_loss_weights.py`
over it:

| phase | selected_mean_tokens | weight_mean | grad_weighted_mean_tokens | share of  conditioned mass with w > 1 |
| --- | --- | --- | --- | --- |
| early | 78.3 | 1.018 | 87.1 | 0.024 |
| middle | 98.3 | 1.049 | 120.9 | 0.060 |
| late | 121.3 | 1.086 | 159.2 | 0.102 |
| whole run (equal thirds) | 99.3 | 1.051 | 123.2 | 0.062 |

The running gradient-weighting arm reported 78.3 / 1.019 / 88.2 in its early
phase, i.e. the realized gradient share matches the prediction to within
sampling noise.  So the mechanism does what the curve says, and no dataset is
silently coming to dominate through the weighting.

Worth stating plainly because it bounds what this arm can show: with a
reference of 128 tokens, a caption has to exceed 128 retained tokens before
its weight leaves 1.0, and 94% of the run's drawn mass sits below that.  The
mean weight is 1.05 across the run and 1.09 in the late phase.  This is a
mild nudge toward long captions, not a rebalancing; a large eval-loss
difference should not be expected from it, and a tie is not evidence that
length weighting is useless - only that this curve, at this reference, is
nearly inert on this corpus.

## The eval-loss probe changed between arms, so the last arm needs a re-measure (2026-09-12)

The caption-length bands of the fixed eval-loss probe were redesigned (the
user asked for bands derived from the length scale rather than from output
resolutions), after `arm-a`/`arm-b`/`arm-c` had already trained.  The arms'
probe drew 589 samples over the bands `256_511`, `512_895`, `896_1280`, while
the current probe draws 864 samples over `le128`, `129_256`, `257_512`,
`513_1024`, `1025_2048`.

Eval loss is a mean over the probe's samples, so the two are different
rulers: the new probe adds a short-caption band that the held-out set can
actually fill (275 samples in `129_256`), and short captions are easier.  A
`gradient-weighting` arm measured on the new probe therefore starts at a
different level from `arm-b`'s 0.886682 at step 5000 for a reason that has
nothing to do with the weighting.  Comparing the two curves directly would
have credited the change to the arm.

What was measured, matched: `caption/weight_mean` and
`caption/grad_weighted_mean_tokens` are expectations over drawn captions and
carry no probe, so those are comparable and were checked (previous section).
What was not: `eval/loss`.

Fix, so the arm still has a usable comparison: re-measure `arm-b`'s endpoint
on the current probe.  `reprobe-256p.toml` + `jobs/reprobe_arm.sh` resume the
arm's last checkpoint (step 4000), train the remaining 1000 steps with the
mix and plan pinned in the config file itself, and evaluate the probe once at
step 5000 under the name `arm-b-reprobe`.  The mix and plan are pinned rather
than inherited from the shared run config, so repointing that config later
cannot change what the reproduction trains on.  `arm-log2w` produces its step
5000 point natively, so the two are then on one ruler.

The same trap applies to `kid_eval.py`-based comparisons only if that module
changes; it did not change between the arms and this run, so end-of-run KID
stays comparable, and the fixed-prompt sample grids are unchanged.

## Where a bucket plan stands, and one measurement the pipeline produced (2026-09-12)

A plan is not a deliverable of this stage: the model shape and the data mix
are both still open, either one moves the boundaries and the batch sizes, and
a plan solved now gets re-solved later.  What this stage delivers is the
pipeline that produces one (see "The stage delivers the planning pipeline,
not a plan" above).

Worth recording because I got it wrong once: I re-solved the 256p and 640p
boundaries against the rescued synthetic set so that the plan files would
match the mix the run configs name, and put one of them through a card
validation.  Nothing needed that.  The run configs live on the shared disk
rather than in the repository and a later stage solves the plans anyway.  The
re-solved plans were deleted and the configs put back as the caption-policy
arms used them.  The plans that exist remain: `256p-k10-13src.screened.json`
(the arms' plan) and `640p-k10.full.sized.json`.

The exercise did run step 3 of the pipeline for real, and that produced one
measurement Stage 4's calibration can use.  `diag-fullmix-256p-0912`, 60
steps of the full mix on one card, on a plan whose heaviest shape the
analytic model put at 30.3 GB:

    steps=60  samples=49784  samples_per_step=829.7
    peak_mem_gb=30.2  peak_mem_reserved_gb=31.0  (card 47.37)
    every bucket served, including the 2048-token top.  No OOM.

So on that plan the mixed-stream surplus was ~0 at 256p.  The earlier
full-mix run on a different 256p plan measured 39.1 GB against a 31.9 GB
heaviest-shape prediction, i.e. 7.2 GB of surplus, and that reading is where
the 10 GB constant comes from.  Same card, same code, same resolution,
similar shapes; I could not reconcile the two from the artefacts at hand, so
the constant stays conservative until a calibration run settles it.  The
practical effect is that a plan sized by it leaves roughly a third of a
47.37 GB card unused at 256p.

One number not to read as a throughput result: a 60-step run contains the
first-time compile of ~500 (resolution, bucket) shapes.  The same run reports
45.5 samples/s steady against 74.4 for a 5000-step run, and that gap is
compile, not the plan.

## Length weighting adopted without a comparative verdict; stage closed (user, 2026-09-12)

The user closed the question the gradient-weighting arm was meant to answer:
given that the curve at reference 128 is a mild nudge (mean weight 1.05 over
the run, 94% of drawn mass below the reference, per the prediction table
above), a 5000-step comparison is unlikely to separate it from the unweighted
winner on eval loss, so the comparison is not worth the cards.  Decision:
the weighting ships.  Final design is the one under test -
`caption_loss_weight_curve = "log2"`, `caption_loss_weight_reference = 128`,
f(L) = max(1, log2(L/128)) on the sample's retained length - on top of the
caption-policy winner (arm B: beta selector, 0.20 short-caption reserve,
linear -1 to +1 ramp).  What the arm did establish before being stopped is
the mechanism check: realized `weight_mean` / `grad_weighted_mean_tokens`
match the offline prediction, so the curve does what it says.

`arm-log2w-0912` (at ~1200/5000 steps) and `arm-b-reprobe-0912` were both
stopped on this decision; the re-measure described in the previous section
is no longer needed.  The 896p precompute jobs are unaffected and run on.

This closes the stage.  Deliverables: the two design choices above, the
bucket-planning pipeline (boundaries from the length distribution, batch
sizes from the analytic memory model, a short GPU calibration pass for
time-alignment), and the data work (enriched captions, rescued synthetic
set, screening, sidecars merged into precompute).  Bucket plans themselves
are Stage 4's job, per "The stage delivers the planning pipeline, not a
plan".
