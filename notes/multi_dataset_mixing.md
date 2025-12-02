# Multi-Dataset Mixing Plan

## Objectives
- Combine multiple precomputed HF datasets in Stage 1 training without duplicating storage.
- Preserve resolution-bucket batching, distributed sharding, and per-sample metadata.
- Enforce user-defined sampling ratios (e.g., 90% Dataset A @80k rows, 10% Dataset B @1M rows).

## CLI and Config Surface
1. Replace `--precomputed_dataset_path` with `--dataset_mix "pathA:0.9 pathB:0.1"`.
2. Parse into structured entries: `[{"path": Path, "weight": float, "alias": str}]`. Provide defaults if only one path is supplied.
3. Persist parsed spec in `args` so checkpoints/logging know the intended distribution.

## Dataset Assembly Pipeline
1. Load every dataset separately via `load_from_disk(entry.path)`.
2. Add a string/int `dataset_id` column before concatenation so downstream samplers and logs can distinguish examples.
3. Concatenate the datasets (`concatenate_datasets`) to obtain a single `Dataset` object compatible with Accelerate + PyTorch DataLoader.
4. Optionally pre-shuffle each dataset with its own seed to guarantee local randomness before samplers receive indices.

## Weighted Bucket Sampler Design
1. Extend `ResolutionBucketSampler` signature to accept `dataset_weights: Dict[str, float]`.
2. While building buckets, group indices by `(dataset_id, resolution_bucket_id)` to maintain both constraints simultaneously.
3. Keep per-rank sharding (`indices[self.rank :: self.num_replicas]`) but apply it inside each dataset group so every process sees the right proportion locally.
4. During iteration:
   - Maintain a map `{dataset_id: deque_of_bucket_batches}`. Each deque stores ready-made batches for that dataset.
   - Draw the next dataset via `random.choices(dataset_ids, weights=...)` and pop one batch from its deque.
   - When a dataset runs out of full batches, drop it from the chooser and renormalize weights.
5. Continue until no dataset has complete batches left. This guarantees the long dataset does not dominate once the small dataset is exhausted.

## Training-Loop Hooks
- Each batch already carries `dataset_id`; log per-source loss, stage, and effective counts via `accelerator.log` to validate the empirical ratio.
- For EMA/eval checkpoints, store the current dataset histogram so downstream analysis can reconcile metrics with sampling skew.

## Alternative: `datasets.interleave_datasets`
- Hugging Face provides `interleave_datasets([ds_a, ds_b], probabilities=[0.9, 0.1], seed=..., stopping_strategy="first_exhausted")` which returns a single map-style dataset that already respects the ratio.
- Advantages: zero custom sampler work, works with existing DataLoader immediately.
- Limitations:
  1. Default `stopping_strategy="first_exhausted"` stops once the smallest dataset runs out. With 80k (A) and 1M (B) at 90/10, the mixed dataset yields ≈88.9k samples total (since each A sample is used exactly once). Training keeps the ratio but terminates early relative to B’s capacity.
  2. Memory/time overhead: `interleave_datasets` materializes index mappings eagerly; repeated shuffles require re-running the interleave op.
  3. Harder to expose per-batch dataset IDs unless you manually add a column before interleaving.
- Use this path only if you accept the shortened epoch length or plan to periodically re-run interleave after reloading A.

## Recommended Next Steps
1. Implement parser + dataset loader changes behind a feature flag.
2. Extend `ResolutionBucketSampler` with weighted mixing and unit tests covering deterministic ratios and distributed shards.
3. Add telemetry in `train_stage1.py` to monitor per-dataset sample counts per N steps.
4. Once stable, consider optional fallback to `interleave_datasets` for experiments without distributed sampling.
