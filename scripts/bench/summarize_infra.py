"""Summarize per-rank infra records without extrapolating lower-rank results.

Training intervals exclude evaluation/checkpoint work. This report is diagnostic
evidence, not an all-in hero budget or a saturation/readiness verdict. A first
observed shape is a compile-risk marker, not proof that compilation occurred.
"""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(directory, *, skip=50, window=25):
    if skip < 0 or window < 1:
        raise ValueError("skip must be nonnegative and window positive")
    ranks = {}
    for path in sorted(Path(directory).glob("rank-*.jsonl")):
        records = {}
        rank = int(path.stem.split("-")[-1])
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["rank"] != rank or row["step"] in records:
                raise ValueError(f"rank mismatch or duplicate step in {path}")
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError(f"invalid timing in {path}")
            records[row["step"]] = row
        ranks[rank] = records
    if not ranks:
        raise ValueError("no per-rank records found")
    if set(ranks) != set(range(len(ranks))):
        raise ValueError("rank IDs must start at zero and be contiguous")
    common = sorted(set.intersection(*(set(rows) for rows in ranks.values())))
    seen = {rank: set() for rank in ranks}
    updates = []
    for step in common:
        rows = [records[step] for records in ranks.values()]
        if len({row["global_samples"] for row in rows}) != 1:
            raise ValueError(f"inconsistent global sample count at step {step}")
        new_shapes = 0
        for row in rows:
            shapes = {tuple(shape["shape"]) for shape in row["shapes"]}
            new_shapes += len(shapes - seen[row["rank"]])
            seen[row["rank"]].update(shapes)
        updates.append(dict(step=step, seconds=max(row["seconds"] for row in rows),
                            samples=rows[0]["global_samples"], new_shapes=new_shapes,
                            profiled=any(row["profiled"] for row in rows)))

    def stats(rows):
        if not rows:
            return None
        times = sorted(row["seconds"] for row in rows)
        total = sum(times)
        return dict(steps=len(rows), first_step=rows[0]["step"], last_step=rows[-1]["step"],
                    seconds=total, mean_seconds=statistics.mean(times),
                    median_seconds=statistics.median(times),
                    p95_seconds=times[math.ceil(.95 * len(times)) - 1],
                    samples_per_second=sum(row["samples"] for row in rows) / total,
                    first_seen_shape_updates=sum(row["new_shapes"] > 0 for row in rows),
                    profiled_updates=sum(row["profiled"] for row in rows))

    all_rows = [row for records in ranks.values() for row in records.values()]
    return dict(
        observed_ranks=len(ranks), all_in_hero_gate_established=False,
        caveat="Rank files alone do not establish expected world size or job completion. "
               "Times use the maximum recorded rank interval per aligned update, exclude "
               "eval/checkpoint overhead, and can include compilation/profiling. "
               "Repeated-shape subsets change the sampled distribution and are diagnostic only.",
        common_steps=len(common), unaligned_records=len(all_rows) - len(common) * len(ranks),
        all_updates=stats(updates), after_skip=stats(updates[skip:]),
        repeated_shape_updates=stats([row for row in updates if not row["new_shapes"]]),
        windows=[stats(updates[start:start + window]) for start in range(0, len(updates), window)],
        unique_shapes_per_rank={rank: len(shapes) for rank, shapes in seen.items()},
        peak_allocated_bytes=max((r["peak_allocated_bytes"] for r in all_rows), default=0),
        peak_reserved_bytes=max((r["peak_reserved_bytes"] for r in all_rows), default=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Stage run's infra/ directory")
    parser.add_argument("--skip", type=int, default=50)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, skip=args.skip, window=args.window), indent=2))


if __name__ == "__main__":
    main()
