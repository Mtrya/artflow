"""Predict rank-local samples/microbatch from Stage-3 metadata, without latents.

This is the long-run expectation, ignoring bounded incomplete bucket tails.
It uses the exact within-row curriculum probabilities and the gate bucket table.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.bench.gate_ab import BUCKETS, MIX
from src.dataset.length_metadata import RowLengthMetadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workroot", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    stages = [0.0, 0.5, 1.0]
    costs = {world: np.zeros((len(stages), world)) for world in args.ranks}
    for name, weight in MIX.items():
        path = Path(args.workroot) / "precomputed_dataset" / f"{name}@256p"
        meta = RowLengthMetadata.load(path / "length_metadata.npz")
        counts = np.diff(meta.caption_offsets)
        rows = np.repeat(np.arange(meta.num_rows), counts)
        tokens = meta.curriculum_lengths.astype(float)
        totals = np.bincount(rows, weights=tokens, minlength=meta.num_rows)
        means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        deviations = np.divide(tokens - means[rows], means[rows],
                               out=np.zeros_like(tokens), where=means[rows] > 0)
        bucket_indices = np.searchsorted(np.array(BUCKETS)[:, 0], meta.prompt_lengths)
        inverse_batch = 1.0 / np.array(BUCKETS)[bucket_indices, 1]
        max_probs = np.maximum(.15, np.minimum(.80, 1.0 - .15 * (counts[rows] - 1)))
        for index, stage in enumerate(stages):
            scores = np.maximum(1 + 2 * (stage - .5) * deviations, 1e-6)
            sums = np.bincount(rows, weights=scores, minlength=meta.num_rows)
            probabilities = np.clip(scores / sums[rows], .15, max_probs)
            sums = np.bincount(rows, weights=probabilities, minlength=meta.num_rows)
            probabilities /= sums[rows]
            row_costs = np.bincount(rows, weights=probabilities * inverse_batch,
                                    minlength=meta.num_rows)
            for world in args.ranks:
                for rank in range(world):
                    valid = counts[rank::world] > 0
                    costs[world][index, rank] += weight * row_costs[rank::world][valid].mean()
    for world in args.ranks:
        for index, stage in enumerate(stages):
            rates = 1 / costs[world][index]
            print(json.dumps({"ranks": world, "stage": stage,
                              "expected_samples_per_micro": rates.tolist(),
                              "max_min_rank_rate": float(rates.max() / rates.min())}), flush=True)


if __name__ == "__main__":
    main()
