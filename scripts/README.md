# Project tools

Run Python commands from the repository root as `python -m scripts.<area>.<name>`.
Each command documents its arguments through `--help`. Run corpus processing
where the data is mounted. Platform setup and credentials are in the local
`INSPIRE.md`; training settings come from one complete run TOML.

## Training and planning

| Command | Purpose |
|---|---|
| `pretrain.launch` | Launch or resume one curriculum stage from a complete run config. |
| `ascend/pretrain.sh` | Set up the Ascend environment and invoke `pretrain.launch`. |
| `pretrain.migrate_conditioning_decay` | Convert a checkpoint to the conditioning optimizer groups required by the current recipe. |
| `bench.plan_buckets` | Fit measured calibration data and plan length boundaries and micro-batches against caption draws. |
| `bench.resize_bucket_plan` | Draft fixed-boundary batch/accumulation candidates for validation on the full workload. |
| `bench.pin_artifacts` | Record and verify input hashes. |

Use `python -m src.pretrain.precompute --help` for VAE precomputation. Specify
the manifest, output, resolution and device for each job. Job wrappers must
check every worker's exit status before declaring success.

## Data preparation

| Command | Purpose |
|---|---|
| `data.fetch_pexels` | Harvest resumable Pexels query batches with subject and shape filters. |
| `data.scan_resolution` | Measure image dimensions from file headers and summarize resolution coverage. |
| `data.build_d1_metadata` | Assemble Chinese-painting metadata from clean records and VLM labels. |
| `data.apply_final_bbox` | Apply final crop boxes, filter views and merge OCR into D1 captions. |
| `data.publish_hf_d1` | Stage D1 metadata and optional image parquet, then optionally publish one release. |
| `data.build_d1_manifest` | Build D1 training rows from published metadata and accepted long captions. |
| `data.build_precompute_manifest` | Build D2–D4 training manifests and their held-out evaluation rows. |
| `data.apply_caption_enrichment` | Attach frozen captions and length metadata to an existing precomputed dataset. |
| `data.publish_hf_pexels` | Publish photo references, attribution and captions. |
| `data.publish_hf_synth` | Stage and publish the synthetic traditional-dress dataset. |
| `data.transfer_precomputed` | Upload a configured stage's datasets or download a repository snapshot through Hugging Face or ModelScope. |

Museum harvesting lives in `src.dataset.fetchers`; use its source-specific
module commands for AIC, NGA, NPM Taiwan, FSG, Princeton and Rijksmuseum.

The synthetic-data sequence is `data.synth_prompt_grid` →
`data.synth_assign_models` → `data.synth_generate` → `data.synth_manifest`.
It builds a balanced traditional-dress grid, assigns teacher models, generates
images and assembles their training rows. Generation hardware follows the
teacher model's requirements.

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

## Caption production

Use `caption.select_rows` → `caption.generate` → `caption.freeze_captions` →
`data.apply_caption_enrichment`. Freezing consolidates retries, measures retained
length with the training tokenizer, applies the text-quality checks and emits
one measured record per image.

| Command | Purpose |
|---|---|
| `caption.audit_coverage` | Summarize caption coverage and lengths by source. |
| `caption.review` | Build image/caption bundles for human review. |
| `caption.build_artifact_index` | Collect artifact labels for crop refinement. |
| `caption.refine_bbox` | Request refined boxes for images with surrounding artifacts. |
| `caption.synth_filter` | Check synthetic traditional-dress captions and image quality. |
| `caption.synth_rescue_selection` | Select synthetic rows for targeted recaptioning. |

## Monitoring, evaluation and post-training

| Command | Purpose |
|---|---|
| `ascend.hero_watch` | Print recent SwanLab training and internal-health metrics for `--run USER/PROJECT/RUN`. |
| `ascend.hero_curves` | Render metric histories for `--run USER/PROJECT/RUN --out DIRECTORY`; optionally shade `--highlight-steps START END`. |
| `eval.validate_monitor_panel` | Validate the frozen bilingual prompt panel. |
| `eval.blind_panel panel` | Assemble blinded comparisons from directories of `<prompt_id>.png` images. |
| `eval.blind_panel tally` | Count wins and ties from a completed ballot. |
| `posttrain.build_rollout_prompts` | Sample tagged prompts from the precomputed datasets in `--config` and `--stage`. |

Rollout counts follow the config's source probabilities. Rows are sampled
without replacement; if a source is exhausted, its shortfall is redistributed
by the remaining source weights. Each sampled row contributes one uniformly
chosen usable caption. The command prints final source counts for review.
