"""Caption-length coverage of the training mix, read from sidecar metadata.

The trainer draws a dataset, then a row, then one caption inside that row.  How
much long text the model can actually be conditioned on is therefore not the
length histogram of all stored captions: it is a function of the dataset
mixture, the per-row caption lengths, and the within-row selection policy.

This tool reports three different quantities, because they answer different
questions:

``caption histogram``
    Every stored caption, unweighted.  Describes the corpus, not the training
    signal.  Reported for completeness only.

``draw mass``
    Expected share of caption draws at or above a length threshold, under the
    current within-row curriculum and the dataset mixture.  This is the actual
    exposure of the current recipe.

``row coverage``
    Share of row draws whose row contains at least one caption at or above the
    threshold.  This is the ceiling that any within-row bias could reach
    without new text, and it is the quantity the enrichment plan reports as
    ``C(>=L)``.

Reads only ``length_metadata.npz`` sidecars, so it is CPU-only and never
touches latents or images.  Lengths are the retained prompt lengths defined by
``src/utils/prompt_contract.py``: prompt template applied, system prefix
dropped, capped at 2048 tokens, floored at 1.

CLI:
    python -m scripts.caption.audit_coverage \
        --dataset-root /path/to/precomputed_dataset \
        --mix "d1@256p:0.15 d4-relaion@256p:0.301" \
        --out /tmp/coverage.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

THRESHOLDS = (128, 256, 512, 1024, 1536, 2048)
BANDS = ((1, 255), (256, 511), (512, 1023), (1024, 1535), (1536, 2048))


@dataclass
class DatasetCoverage:
    """Coverage of one dataset, measured over its rows and captions."""

    name: str
    weight: float
    num_rows: int
    num_captions: int
    caption_lengths: np.ndarray
    row_max_length: np.ndarray
    row_lengths: List[np.ndarray] = field(default_factory=list)

    @property
    def captions_per_row(self) -> float:
        return self.num_captions / self.num_rows if self.num_rows else 0.0

    def row_coverage(self, threshold: int) -> float:
        """Share of rows holding at least one caption of >= threshold tokens."""
        if not self.num_rows:
            return 0.0
        return float(np.mean(self.row_max_length >= threshold))

    def draw_mass(self, threshold: int, stage: float) -> float:
        """Expected share of caption draws at >= threshold under the curriculum.

        The within-row distribution is the one the trainer uses
        (``captions.caption_probabilities_from_token_counts``), evaluated at a
        fixed curriculum position and averaged over rows with equal row weight.
        """
        if not self.row_lengths:
            return 0.0
        from src.dataset.captions import caption_probabilities_from_token_counts

        total = 0.0
        for lengths in self.row_lengths:
            counts = lengths.tolist()
            if len(counts) == 1:
                total += 1.0 if counts[0] >= threshold else 0.0
                continue
            probs = caption_probabilities_from_token_counts(counts, stage=stage)
            total += float(sum(p for p, n in zip(probs, counts) if n >= threshold))
        return total / self.num_rows

    def band_shares(self, stage: float) -> Dict[str, float]:
        """Expected draw share per length band under the curriculum."""
        from src.dataset.captions import caption_probabilities_from_token_counts

        mass = {f"{lo}-{hi}": 0.0 for lo, hi in BANDS}
        if not self.row_lengths:
            return mass
        for lengths in self.row_lengths:
            counts = lengths.tolist()
            probs = (
                [1.0]
                if len(counts) == 1
                else caption_probabilities_from_token_counts(counts, stage=stage)
            )
            for p, n in zip(probs, counts):
                for lo, hi in BANDS:
                    if lo <= n <= hi:
                        mass[f"{lo}-{hi}"] += p
                        break
        return {key: value / self.num_rows for key, value in mass.items()}

    def summary(self, stage: float) -> Dict:
        hist, edges = np.histogram(self.caption_lengths, bins=[0, *[hi for _, hi in BANDS]])
        return {
            "name": self.name,
            "weight": self.weight,
            "num_rows": self.num_rows,
            "num_captions": self.num_captions,
            "captions_per_row": self.captions_per_row,
            "captions_per_row_hist": _captions_per_row_hist(self),
            "caption_histogram": {
                f"{lo}-{hi}": int(hist[i]) for i, (lo, hi) in enumerate(BANDS)
            },
            "row_max_length_percentiles": {
                q: float(np.percentile(self.row_max_length, q))
                for q in (50, 75, 90, 95, 99)
            },
            "row_coverage": {str(t): self.row_coverage(t) for t in THRESHOLDS},
            "draw_mass": {str(t): self.draw_mass(t, stage) for t in THRESHOLDS},
            "band_draw_share": self.band_shares(stage),
        }


def _captions_per_row_hist(coverage: DatasetCoverage) -> Dict[str, int]:
    counts = np.array([len(lengths) for lengths in coverage.row_lengths])
    values, frequencies = np.unique(counts, return_counts=True)
    return {str(int(v)): int(f) for v, f in zip(values, frequencies)}


def resolve_dataset_dir(root: Path, name: str) -> Path:
    """Find a dataset directory, tolerating an appended resolution suffix.

    The mix is written in terms of dataset names; on disk the precomputed
    datasets are stored per resolution (``d1@256p``).  Both forms are accepted
    so the same mix string can be used here and in a training config.
    """
    direct = root / name
    if (direct / "length_metadata.npz").is_file():
        return direct
    matches = sorted(p for p in root.glob(f"{name}@*") if (p / "length_metadata.npz").is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no dataset with sidecar for {name!r} under {root}")
    raise FileNotFoundError(f"ambiguous resolution suffix for {name!r}: {matches}")


def load_dataset_coverage(path: Path, name: str, weight: float) -> DatasetCoverage:
    sidecar = path / "length_metadata.npz"
    with np.load(sidecar, allow_pickle=False) as data:
        offsets = data["caption_offsets"]
        lengths = data["prompt_lengths"]
    row_lengths = [lengths[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    row_max = np.array([int(lengths[offsets[i]:offsets[i + 1]].max())
                        if offsets[i + 1] > offsets[i] else 0
                        for i in range(len(offsets) - 1)])
    return DatasetCoverage(
        name=name,
        weight=weight,
        num_rows=len(offsets) - 1,
        num_captions=int(offsets[-1]),
        caption_lengths=lengths,
        row_max_length=row_max,
        row_lengths=row_lengths,
    )


def parse_mix(spec: str) -> List[tuple]:
    entries = []
    for part in spec.split():
        if ":" in part:
            name, weight = part.rsplit(":", 1)
            entries.append((name, float(weight)))
        else:
            entries.append((part, 1.0))
    total = sum(w for _, w in entries)
    return [(name, weight / total) for name, weight in entries]


def aggregate(coverages: Sequence[DatasetCoverage], stage: float) -> Dict:
    total_weight = sum(c.weight for c in coverages)
    row_coverage = {
        str(t): sum(c.weight * c.row_coverage(t) for c in coverages) / total_weight
        for t in THRESHOLDS
    }
    draw_mass = {
        str(t): sum(c.weight * c.draw_mass(t, stage) for c in coverages) / total_weight
        for t in THRESHOLDS
    }
    bands: Dict[str, float] = {}
    for c in coverages:
        for key, value in c.band_shares(stage).items():
            bands[key] = bands.get(key, 0.0) + c.weight * value / total_weight
    return {"row_coverage": row_coverage, "draw_mass": draw_mass, "band_draw_share": bands}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                        help="directory containing <name>/length_metadata.npz")
    parser.add_argument("--mix", required=True,
                        help='space separated "name:weight" entries')
    parser.add_argument("--stage", type=float, default=1.0,
                        help="within-row curriculum position (0 shortest, 1 longest)")
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    coverages = [load_dataset_coverage(resolve_dataset_dir(root, name), name, weight)
                 for name, weight in parse_mix(args.mix)]
    report = {
        "stage": args.stage,
        "thresholds": list(THRESHOLDS),
        "datasets": [c.summary(args.stage) for c in coverages],
        "aggregate": aggregate(coverages, args.stage),
    }

    print(f"within-row curriculum stage = {args.stage}")
    header = f"{'dataset':>16s} {'w':>6s} {'rows':>9s} {'caps/row':>9s}"
    for t in THRESHOLDS:
        header += f" {'C>=' + str(t):>9s}"
    print(header)
    for c in coverages:
        line = f"{c.name:>16s} {c.weight:6.3f} {c.num_rows:9d} {c.captions_per_row:9.2f}"
        for t in THRESHOLDS:
            line += f" {100 * c.row_coverage(t):8.1f}%"
        print(line)
    agg = report["aggregate"]
    line = f"{'AGGREGATE':>16s} {1.0:6.3f} {'':>9s} {'':>9s}"
    for t in THRESHOLDS:
        line += f" {100 * agg['row_coverage'][str(t)]:8.1f}%"
    print(line + "   <- row coverage (ceiling for within-row bias)")
    line = f"{'AGGREGATE':>16s} {1.0:6.3f} {'':>9s} {'':>9s}"
    for t in THRESHOLDS:
        line += f" {100 * agg['draw_mass'][str(t)]:8.1f}%"
    print(line + "   <- expected draw mass (current recipe)")
    print("\ndraw mass by band:")
    for key, value in agg["band_draw_share"].items():
        print(f"  {key:>10s}  {100 * value:6.2f}%")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
