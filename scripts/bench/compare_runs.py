"""Compare SwanLab runs at matched steps.

Reads the scalar series of several runs in one project and prints, for each
requested metric key, a table of per-run values aligned by step, so two runs
(e.g. a baseline and an ablated variant) can be compared at the same
training step. Runs not found in the project are reported and skipped.

Usage:
    python scripts/bench/compare_runs.py --project <entity>/<project> \
        --runs baseline-run variant-run --keys eval/loss
"""

import argparse
import netrc
import os


def _api_key() -> str:
    path = os.environ.get("SWANLAB_NETRC")
    if path and os.path.exists(path):
        _, (_, _, password) = list(netrc.netrc(path).hosts.items())[0]
        return password
    key = os.environ.get("SWANLAB_API_KEY")
    if not key:
        raise SystemExit("set SWANLAB_NETRC or SWANLAB_API_KEY")
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="<entity>/<swanlab project>")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--keys", nargs="+", default=["eval/loss"])
    parser.add_argument("--last", type=int, default=0, help="keep only the last N points")
    args = parser.parse_args()

    import swanlab

    api = swanlab.Api(api_key=_api_key())
    available = list(api.runs(args.project))
    series = {}
    for name in args.runs:
        matches = [run for run in available if run.name == name]
        if not matches:
            print(f"!! {name}: not found in {args.project}")
            continue
        run = matches[-1]
        payload = run.metrics(list(args.keys))
        entry = {}
        for block in payload.get("list", []):
            points = [(p["index"], p["data"]) for p in block.get("metrics", [])]
            if args.last:
                points = points[-args.last :]
            entry[block["key"]] = points
        series[name] = entry
        print(f"# {name}: run_id={run.run_id} state={run.state}")

    for key in args.keys:
        print(f"\n### {key}")
        steps = sorted({step for entry in series.values() for step, _ in entry.get(key, [])})
        header = "step\t" + "\t".join(series.keys())
        print(header)
        for step in steps:
            row = [str(step)]
            for name in series:
                value = dict(series[name].get(key, [])).get(step)
                row.append(f"{value:.6f}" if isinstance(value, float) else "-")
            print("\t".join(row))


if __name__ == "__main__":
    main()
