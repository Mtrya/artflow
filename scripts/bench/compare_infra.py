"""Compare matched update records; never extrapolates an eight-GPU hero rate."""

import argparse
import json
from pathlib import Path


def read_records(path):
    rows = {}
    seen = set()
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        shapes = {tuple(item["shape"]) for item in row["shapes"]}
        row["repeat_shape"] = shapes <= seen
        seen.update(shapes)
        if row["step"] in rows:
            raise ValueError("duplicate update record")
        rows[row["step"]] = row
    return rows


def compare(baseline, candidate):
    pairs = []
    for step in sorted(baseline.keys() & candidate.keys()):
        a, b = baseline[step], candidate[step]
        for key in ("rank", "global_samples", "progress", "shapes"):
            if a[key] != b[key]:
                raise ValueError(f"unmatched {key} at step {step}")
        if not a["profiled"] and not b["profiled"]:
            pairs.append((a, b))

    def stats(rows):
        if not rows:
            return None
        a = sum(x["seconds"] for x, _ in rows)
        b = sum(y["seconds"] for _, y in rows)
        return dict(updates=len(rows), baseline_seconds=a, candidate_seconds=b,
                    speedup=a / b, baseline_mean_seconds=a / len(rows),
                    candidate_mean_seconds=b / len(rows),
                    max_absolute_loss_difference=max(abs(x["loss"] - y["loss"]) for x, y in rows))

    return dict(matched_nonprofiled=stats(pairs),
                matched_repeat_shapes=stats([(a, b) for a, b in pairs
                                            if a["repeat_shape"] and b["repeat_shape"]]),
                caveat="Shape-repeat subsets change the sampled distribution. Matching records "
                       "does not prove identical hardware/clocks/cache state or full numerical equivalence. "
                       "No all-in hero gate is established.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(read_records(args.baseline), read_records(args.candidate)), indent=2))


if __name__ == "__main__":
    main()
