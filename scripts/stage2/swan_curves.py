#!/usr/bin/env python
"""
Print eval/loss curve tables for one or more swanlab runs at probe steps.

Must run where the swanlab netrc login lives (Andromeda). Run ids are resolved
by run NAME under the project, so pass the same names the arm scripts used.

Usage (Andromeda):
  python swan_curves.py --project artflow-stage2 --runs s2-wide s2-deep
Adds --ids to pin run ids instead of names. Metrics default to eval/loss and
the per-t buckets; --metrics overrides.
"""

import argparse

from swanlab import Api


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--runs", nargs="+", default=[])
    p.add_argument("--ids", nargs="+", default=[])
    p.add_argument("--steps", type=int, nargs="+",
                   default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    p.add_argument("--metrics", nargs="+",
                   default=["eval/loss", "eval/loss_t015", "eval/loss_t040",
                            "eval/loss_t065", "eval/loss_t090"])
    args = p.parse_args()

    api = Api()
    names = {r.name: r.run_id for r in api.runs(f"mtrya/{args.project}")}
    lookup = {}
    for name in args.runs:
        if name not in names:
            raise SystemExit(f"run name not found: {name} (have {sorted(names)})")
        lookup[name] = names[name]
    for rid in args.ids:
        lookup.setdefault(rid, rid)

    data = {}
    for name, rid in lookup.items():
        run = api.run(f"mtrya/{args.project}/{rid}")
        data[name] = {}
        for mk in args.metrics:
            keys = [k for k in run.series(metric_type="SCALAR", search=mk)
                    if k.key == mk]
            if not keys:
                data[name][mk] = {}
                continue
            data[name][mk] = {m["index"]: m["data"]
                              for m in keys[0].metric()["metrics"]}

    names = list(lookup)
    print("metric" + " " * 10 + "step" + "".join(f"{n:>14}" for n in names))
    for mk in args.metrics:
        for st in args.steps:
            row = [data[n][mk].get(st) for n in names]
            if all(v is None for v in row):
                continue
            cells = "".join(f"{v:>14.5f}" if v is not None else f"{'-':>14}"
                            for v in row)
            print(f"{mk:16s} {st:5d}{cells}")
        print()


if __name__ == "__main__":
    main()
