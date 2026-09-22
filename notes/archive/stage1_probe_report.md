# Stage 1.1 — Source Probe Report

> **存档文件(2026-08-25 探针)**:当时的实测与上限估算,部分结论已被后续决策
> 推翻/扩展。**当前组成与数字以 `notes/dataset_plan.md` 为准**,本文仅保留
> 探针细节备查。
>
> Date: 2026-08-25. Probes run from local machine; fetchers in `scripts/data/fetch_*.py`,
> outputs in `data/raw/<source>/` (gitignored). Python env: `.venv-data/`.
> Network notes: HF requires local proxy (`HTTPS_PROXY=http://127.0.0.1:7897`,
> ~0.84 MB/s to HF CDN, flaky under concurrency — retry+resume mandatory).
> AIC IIIF host (www.artic.edu) is Cloudflare-challenged — image downloads need a US
> proxy exit. api.github.com needs proxy; raw.githubusercontent.com and COCO work direct.

## Per-source results (probe scale: ~1K images each)

### Met Museum (`data/raw/met`, scripts/data/fetch_met.py) — CC0
- Discovery: `/search?hasImages=true&departmentId=X&q=*` returns full ID list in one call.
  Asian Art (dept 6): **34,153 IDs** (~32K CC0 est.); European Paintings (dept 11): 2,667.
- Yield: 1000/1000 smalls + 20/20 full-res. phash dup rate 0%.
- Metadata: title 100%, artist 61% (Asian mostly anonymous), date 80%, classification 100%.
- Resolution: `primaryImageSmall` ≈600px (probe-grade only); full `primaryImage`:
  75% ≥1024px both dims (n=20), median ~2057×2980.
- Quirks: intermittent **403 bursts** at ~8 req/s (transient — retry with backoff or
  throttle to 4–5 req/s); 0.4% stale 404s; no isPublicDomain search filter (per-object fetch).
- Full-harvest estimate: ~15 h / ~60 GB at full-res for ~37K objects.

### AIC (`data/raw/aic`, scripts/data/fetch_aic.py) — CC0
- Discovery: ES query on `/api/v1/artworks/search`, paginate with constant `limit=100` +
  explicit sort (API offset bug otherwise). Watch 10k result-window cap → id-range partitions.
- Counts: PD paintings w/ image (all dates) **668**; 1860–1910 PD paintings only **286**
  (small but high quality — Monet/Renoir/Cassatt/Seurat); Asian PD w/ image **10,724**;
  all PD w/ image 59,038.
- Yield: 1000/1000. phash dup 0%. artist 72%, other fields ~100%.
- Resolution: native 95% ≥1024px both dims (median 3000×3000) via IIIF `full/!3000,3000/`.
- Quirks: `www.artic.edu` IIIF behind Cloudflare challenge → downloads via US proxy exit;
  `match_phrase: "Arts of Asia"` (not `match: "Asian"`).

### NGA (`data/raw/nga`, scripts/data/fetch_nga.py) — CC0
- Bulk CSVs (objects.csv 82MB, published_images.csv 89MB, cached in `_csv/`; updated
  daily — pin a snapshot per run). No API key/rate limit on `api.nga.gov/iiif`.
- Open-access: 63,419 unique objects w/ primary image; classification=Painting 2,915;
  painting-like medium **20,510** (1,473 dated 1860–1910; 239 by core impressionists).
- Yield: 999/1000 (1 transient). phash dup 0%. title/artist/classification ~100%.
- Resolution: **99.87%** of painting-like ≥1024px both dims at source (median 3459×4328).
- Quirks: filter `viewtype=="primary"` + `openaccess==1`; use numeric beginyear/endyear
  for date filtering; `maxpixels` column is all-null.

### NPM Taipei (`data/raw/npm_tw`, scripts/data/fetch_npm_tw.py) — CC0 (1MP) / CC-BY-4.0 (6MP)
- Canonical host `digitalarchive.npm.gov.tw`; POST /opendata/Pub/Search with
  `allow_redirects=False`, PageSize=100. No bulk CSV exists.
- 繪畫 total: **16,690** (+法書 5,754, 法帖 7,147). Full harvest = 167 pages ≈ 11 h.
- Yield: 1000/1000. phash dup **2.7%** (multi-version paintings of same motif).
- Metadata (zh, free zh captions): title 100%, artist 83%, dynasty 79%,
  theme tags 92% — 人物-tagged 38%, 山水 63%, 花草/花鳥 41%. Dynasty mix: 清 41%, 宋/明 14.5% ea.
- **Resolution blocker**: scriptable route (`GetImage`) serves only **600px long side**.
  Tiered downloads (1MP/6MP) are captcha-gated; the documented IIIF API returns HTTP 500
  (broken server-side as of probe date — worth re-checking periodically).
  → NPM-TW currently only supports the 256p stage. High-res 国画 must come from
  Met/AIC/NGA Asian collections unless IIIF recovers or captcha-tier access is arranged.
- Quirks: flaky TLS (alternate direct↔proxy on connection errors); no rate limit at ~0.5 req/s.

### HF mirrors (investigation; no fetcher yet)
- **WikiArt → `huggan/wikiart`** (non-gated, non-commercial research): 81,444 rows full-res,
  33.7 GB. Impressionism 13,060; +Post-Imp+Expressionism = **26,246**. Measured 100%
  ≥1024px (n=345). Harvest: `snapshot_download` shards → local parquet filter on style
  (12=Impressionism, 20=Post_Impressionism, 9=Expressionism). ~11 h at current proxy speed.
  Metadata-poor (artist/genre/style labels only). `Artificio/WikiArt_Full` (103K rows @256px,
  title/artist/date/description) usable as metadata cross-ref if needed.
- **ArtBench-10 → `zguo0525/ArtBench`** (non-gated, research-use): impressionism =
  label 7 → exactly 5,000 train + 1,000 test @256px, 1.94 GB total. Official CSV at
  artbench.eecs.berkeley.edu (direct) has artist/is_public_domain flags.
- **FFHQ → probe `merkol/ffhq-256` (7.4 GB, streams fine); harvest `marcosv/ffhq-dataset`
  (1024×1024 originals, 95.8 GB, per-file `hf_hub_download`, resumable, ~33 h at current
  proxy speed)**. License CC BY-NC-SA 4.0 (research OK per D1).

## Full-body photo source — pinned
**Primary: PD12M (`Spawning/PD12M`, non-gated, CDLA-permissive-2.0, Florence-2 captions)
caption-filtered for full-body people** (also doubles as license-clean Domain-4 world data).
**Fallback/supplement: COCO2017 person subset** (reachable direct; bbox/keypoints allow
filtering to full-body-visible instances; captions included; Flickr CC terms, research-OK).
Faces covered by FFHQ; ~50/50 face/body balance enforced at mix-assembly time.

## Not probed / blocked
- **Smithsonian Freer/Sackler**: api.si.edu returns 403 without an api.data.gov key
  (free signup requires email verification). Backup Asian-art source; CC0 domains are
  already covered by Met+AIC+NGA — recommend skip unless a key is provided.
- Kaggle no longer needed (ArtBench-10 via HF).

## Estimated corpus ceilings (post-probe)
| Domain | Sources | Realistic ceiling |
|---|---|---|
| 国画 | NPM-TW 16.7K @600px + Met Asian ~32K + AIC Asian 10.7K + NGA + SAPGAN/CCLAP | 40–60K ✓ (high-res subset ~35K) |
| Impressionism | WikiArt 26.2K (NC) + AIC 286 + NGA 1.5K + Met 2.3K (CC0) | ~30K ✓ |
| People | FFHQ 70K + PD12M/COCO full-body + museum portraits | 60–120K ✓ |
| World | PD12M subset / LAION | sized in stage 4 |
