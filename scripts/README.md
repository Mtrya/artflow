# Project tools

Run Python commands from the repository root as `python -m scripts.<area>.<name>`.
Each command documents its arguments through `--help`. Run corpus processing
where the data is mounted. Platform setup and credentials are in the local
`INSPIRE.md`; training settings come from one complete run TOML.

## Pretraining (`scripts/pretrain/`)

| Command | Purpose |
|---|---|
| `pretrain.launch` | Launch or resume one curriculum stage from a complete run config. |
| `pretrain/launch.sh` | Set up the Ascend environment and invoke `pretrain.launch`. |
| `pretrain.migrate_checkpoint` | Migrate model/tracking metadata and declared recipe changes on a verified checkpoint copy. |
| `pretrain.plan_buckets` | Fit measured calibration data and plan length boundaries and micro-batches against caption draws. |
| `pretrain.resize_bucket_plan` | Draft fixed-boundary batch/accumulation candidates for validation on the full workload. |
| `pretrain.pin_artifacts` | Record and verify input hashes before a training launch. |

Use `python -m src.pretrain.precompute --help` for VAE precomputation. Specify
the manifest, output, resolution and device for each job. Job wrappers must
check every worker's exit status before declaring success.

Launchers, planners, recipe migration and configured dataset transfer require
`--storage-root`. Prompts and bucket plans are tracked under `configs/` and
resolve from the repository. The runtime root is not stored in the TOML.

Ordinary resume requires the checkpoint-owned SwanLab project and run ID.
`--new-experiment` explicitly restores training state into a fresh experiment
in the configured project; it never modifies the source experiment. Migration
requires `--source-run-id` when the source lacks a tracking record. An explicit
`--source-assets RECORD.json` supplies hash-verified prompt and bucket bytes
when the source recipe paths are unavailable. The shell launcher uses the
package overlay at `<storage-root>/runtime/python`. See the
[checkpoint migration procedure](../notes/pretrain_recipe.md#launch-checkpoints-and-resolution-transitions).

The planner takes `--config`, `--stage` and `--storage-root`; dataset weights
and caption policy come from that recipe. Calibration is a flat JSON list of
`latent_hw: [H, W]`, `txt_len`, `micro_batch`, `peak_mem_gb` and `ms_per_step`.
Memory is peak allocated GiB (bytes / 2**30); time is milliseconds for the
whole micro-batch DiT forward/backward. An OOM record contains the three shape
fields and `error: "oom"`, without measurements. Calibration inputs must
conform to this schema. Planner estimates require device validation.

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
  --provider hf --repo-id OWNER/DATASET --config configs/pretrain.toml --stage 896p --storage-root /external/inko
python -m scripts.data.transfer_precomputed download \
  --provider modelscope --repo-id OWNER/DATASET --storage-root /external/inko
```

Uploads include every training source and the evaluation set in the selected
stage. `--storage-root` selects the external data/model/output tree.
Downloads fetch the selected repository. Credentials come from `HF_TOKEN`, or
`MS_TOKEN` / `MODELSCOPE_TOKEN_PATH` for ModelScope.
