"""Decide which generator renders each prompt in the synthetic grid.

Two generators are used because a single one imprints its own look on the whole
set, and the set exists precisely to broaden what the model has seen.  The
choice is made per prompt, not per generator run, so the cost of either choice
is even.

The assignment is balanced inside every cell of the grid: for each combination
of language, family, subject group and frame shape, the prompts are split
evenly between the generators.  Balancing on the marginals alone would not do -
a plain alternation down the file would hand one generator most of the square
frames, because the frame shape cycles with the row number - and a generator
that is systematically tied to a subject or a frame shape would put that
association into the training data.

CLI:
    python -m scripts.data.synth_assign_models --grid grid.jsonl \\
        --models ernie-image-turbo,qwen-image --out grid_assigned.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

STRATA = ("language", "family", "subject_group", "aspect")


def assign(rows: List[Dict], models: List[str]) -> List[Dict]:
    """Set ``model`` on every row, split evenly inside each grid cell.

    Cells take turns starting with a different generator: a cell with an odd
    number of rows hands one generator an extra row, and if every cell started
    with the same generator those extra rows would add up across the whole grid.
    """
    buckets: Dict[tuple, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[tuple(row[key] for key in STRATA)].append(index)
    for cell, indexes in enumerate(buckets.values()):
        for position, index in enumerate(indexes):
            rows[index]["model"] = models[(position + cell) % len(models)]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", required=True,
                        help="comma-separated recipe names, in the order they take turns")
    args = parser.parse_args()

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if len(models) < 2:
        raise SystemExit("at least two generators are needed to split the grid")

    rows = [json.loads(line) for line in Path(args.grid).open(encoding="utf-8")
            if line.strip()]
    assign(rows, models)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(rows)} prompts -> {out}")
    counts = Counter(row["model"] for row in rows)
    print(f"  model    {dict(counts)}")
    cells: Dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        cells[tuple(row[key] for key in STRATA)][row["model"]] += 1
    worst_cell = 0
    for counter in cells.values():
        per_model = [counter[model] for model in models]
        worst_cell = max(worst_cell, max(per_model) - min(per_model))
    print(f"  {len(cells)} grid cells; largest imbalance inside a cell: {worst_cell}")
    for key in STRATA:
        table = Counter((row[key], row["model"]) for row in rows)
        worst = 0
        for value in {row[key] for row in rows}:
            per_model = [table[(value, model)] for model in models]
            worst = max(worst, max(per_model) - min(per_model))
        print(f"  {key:<14} largest imbalance across a value: {worst}")


if __name__ == "__main__":
    main()
