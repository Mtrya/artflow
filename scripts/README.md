# Project tools

Run Python commands from the repository root as `python -m scripts.<area>.<name>`.
Each command documents its arguments through `--help`. Run corpus processing
where the data is mounted. Platform setup and credentials are in the local
`INSPIRE.md`; training settings come from one complete run TOML.

## Pretraining (`scripts/pretrain/`)

| Command | Purpose |
|---|---|
| `pretrain.launch` | Launch or resume one curriculum stage from a complete run config. |
| `pretrain/ascend.sh` | Set up the Ascend environment and invoke `pretrain.launch`. |
| `pretrain.migrate_stage_recipe` | Apply an explicit, recorded recipe amendment to a checkpoint copy before resume. |
| `pretrain.plan_buckets` | Fit measured calibration data and plan length boundaries and micro-batches against caption draws. |
| `pretrain.resize_bucket_plan` | Draft fixed-boundary batch/accumulation candidates for validation on the full workload. |
| `pretrain.pin_artifacts` | Record and verify input hashes before a training launch. |

Use `python -m src.pretrain.precompute --help` for VAE precomputation. Specify
the manifest, output, resolution and device for each job. Job wrappers must
check every worker's exit status before declaring success.

## Monitoring (`scripts/monitor/`)

| Command | Purpose |
|---|---|
| `monitor.hero_watch` | Print recent SwanLab training and internal-health metrics for `--run USER/PROJECT/RUN`. |
| `monitor.hero_curves` | Render metric histories for `--run USER/PROJECT/RUN --out DIRECTORY`; optionally shade `--highlight-steps START END`. |

## Evaluation (`scripts/eval/`)

| Command | Purpose |
|---|---|
| `eval.blind_panel panel` | Assemble blinded comparisons from directories of `<prompt_id>.png` images. |
| `eval.blind_panel tally` | Count wins and ties from a completed ballot. |

## Caption production (`scripts/caption/`)

Use `caption.select_rows` → `caption.generate` → `caption.freeze_captions` →
`data.apply_caption_enrichment`. Freezing consolidates retries, measures retained
length with the training tokenizer, applies the text-quality checks and emits
one measured record per image. `caption.review` packs image/caption bundles
for human review.

## Data acquisition and transfer (`scripts/data/`)

| Command | Purpose |
|---|---|
| `data.fetch_pexels` | Harvest resumable Pexels query batches with subject and shape filters. |
| `data.fetch_inat` | Harvest iNaturalist photos for count-capped `term \| cap` query files. |
| `data.fetch_museum` | Harvest Met and AIC public-domain artworks for count-capped `term \| cap` query files. |
| `data.fetch_commons` | Harvest Wikimedia Commons JPEG photos for count-capped `term \| cap` query files. |
| `data.fetch_gbif` | Harvest GBIF occurrence photos (taxon-resolved, iNat reexport excluded) for count-capped `term \| cap` query files. |
| `data.fetch_openverse` | Harvest Openverse CC-licensed photos for count-capped `term \| cap` query files. |

| Command | Purpose |
|---|---|
| `data.build_precompute_manifest` | Build D2–D4 training manifests and their held-out evaluation rows. |
| `data.apply_caption_enrichment` | Attach frozen captions and length metadata to an existing precomputed dataset. |
| `data.transfer_precomputed` | Upload a configured stage's datasets or download a repository snapshot through Hugging Face or ModelScope. |
| `data.publish_hf_d1` | Stage D1 metadata and optional image parquet, then optionally publish one release. |
| `data.publish_hf_pexels` | Publish photo references, attribution and captions. |

For D1, `publish_hf_d1 --out-dir <fresh-directory>` writes the metadata table;
`--images` adds bounded image-parquet shards. `--upload --repo-id OWNER/DATASET`
publishes the staged release to an existing repository. An image release
replaces its image-parquet shard set; a metadata-only release updates the table.

```bash
python -m scripts.data.transfer_precomputed upload \
  --provider hf --repo-id OWNER/DATASET --config configs/hero.toml --stage 896p
python -m scripts.data.transfer_precomputed download \
  --provider modelscope --repo-id OWNER/DATASET --data-root /mounted/precomputed_dataset
```

Uploads include every training source and the evaluation set in the selected
stage. `--data-root` relocates those directory names to another mounted copy.
Downloads fetch the selected repository. Credentials come from `HF_TOKEN`, or
`MS_TOKEN` / `MODELSCOPE_TOKEN_PATH` for ModelScope.
