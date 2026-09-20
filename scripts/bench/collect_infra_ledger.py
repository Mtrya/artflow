"""Read-only Inspire job accounting, retaining evidence and conservative bounds.

CLI status timestamps are epoch milliseconds; event timestamps use the platform
timezone (explicit below). Missing allocation evidence is not zero-cost evidence.
Output contains platform metadata: keep it local, not in a public commit.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def observed_runtime_hours(events, *, gpus_per_instance, timezone):
    """Sum observed container intervals, not delayed controller cleanup.

    Killing is a stop request, so this deliberately excludes termination grace.
    Only pair events belonging to the same instance; never multiply the first
    node's interval by every node in a distributed allocation.
    """
    def timestamp(event):
        return datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=ZoneInfo(timezone)).timestamp()

    starts = {}
    for event in events:
        instance = event.get("instance")
        if (instance and event.get("reason") == "Started"
                and event.get("message") == "Started container pytorch"):
            starts.setdefault(instance, []).append(timestamp(event))
    total = 0.0
    for instance, times in starts.items():
        start = min(times)
        ends = [timestamp(event) for event in events
                if event.get("instance") == instance
                and ((event.get("reason") == "Killing"
                      and event.get("message") == "Stopping container pytorch")
                     or event.get("reason") == "PodReservingStart")]
        eligible = [end for end in ends if end >= start]
        if eligible:
            total += (min(eligible) - start) * gpus_per_instance / 3600
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--event-timezone", default="Asia/Shanghai")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retain closed successful rows; refresh open or failed rows from --out")
    args = parser.parse_args()

    def query(*parts):
        result = subprocess.run(
            ["inspire", "--json", "job", *parts, "--workspace", args.workspace],
            capture_output=True, text=True, timeout=120,
        )
        if not result.stdout.strip():
            raise RuntimeError(f"CLI exit {result.returncode}: {result.stderr[-1000:]}")
        data = json.loads(result.stdout)
        if result.returncode or not data.get("success"):
            raise RuntimeError(data.get("error", result.stderr))
        return data["data"]

    previous = None
    if args.retry_errors:
        previous = json.loads(args.out.read_text())
    all_items = [] if previous else query("list", "--all")["items"]
    counts = Counter(item["name"] for item in all_items)
    occurrence = Counter()
    targets = []
    for item in all_items:
        name = item["name"]
        occurrence[name] += 1
        if item["project"] == args.project and name.startswith(tuple(args.prefix)):
            targets.append((name, occurrence[name] if counts[name] > 1 else None))
    if previous:
        targets = [(row["name"], row["pick"]) for row in previous["jobs"]
                   if "error" in row or row.get("allocation_open")]

    def collect(target):
        name, pick = target
        selection = [name] + (["--pick", str(pick)] if pick else [])
        row = {"name": name, "pick": pick}
        try:
            status = query("status", *selection)
            events = query("events", *selection, "--tail", "10000")
            row.update(status=status, events=events)
            gpus = status["resource"]["gpu"] * status["resource"]["nodes"]
            created = int(status["created_at"]) / 1000
            finish = status.get("finished_at")
            row["allocation_open"] = not bool(finish)
            # An open allocation is not a query failure. Bound it through
            # this observation, retaining the explicit unfinished status.
            finished = (int(finish) / 1000 if finish else
                        datetime.now(ZoneInfo("UTC")).timestamp())
            row["accounted_through_unix"] = finished
            scheduled = [
                datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=ZoneInfo(args.event_timezone)).timestamp()
                for event in events["items"] if event.get("reason") == "Scheduled"
            ]
            if scheduled:
                start = min(scheduled)
                if not created - 1 <= start <= finished:
                    raise ValueError("Scheduling timestamp outside job lifetime")
                row["accounting"] = "scheduled_to_observation" if not finish else "scheduled_to_finished"
            else:
                # Conservatively includes all queue time. Never infer that
                # absent/expired scheduling events establish zero allocation.
                start = created
                row["accounting"] = ("creation_to_observation_upper_bound" if not finish else
                                     "creation_to_finished_upper_bound")
            row["gpu_hours_bound"] = (finished - start) * gpus / 3600
            # A deliberately smaller observed interval: omit provisioning,
            # retention and all later restarts. This can establish overspend
            # even when the conservative allocation bound is too generous.
            row["observed_runtime_gpu_hours"] = observed_runtime_hours(
                events["items"], gpus_per_instance=status["resource"]["gpu"],
                timezone=args.event_timezone,
            )
        except Exception as exc:
            row["error"] = str(exc)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(collect, targets))
    if previous:
        replacements = {(row["name"], row["pick"]): row for row in rows}
        rows = [replacements.get((row["name"], row["pick"]), row) for row in previous["jobs"]]
    report = {
        "queried_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "prefixes": args.prefix, "event_timezone": args.event_timezone,
        "jobs": rows,
        "gpu_hours_bound": sum(row.get("gpu_hours_bound", 0) for row in rows),
        "observed_runtime_gpu_hours": sum(row.get("observed_runtime_gpu_hours", 0)
                                          for row in rows),
        "errors": sum("error" in row for row in rows),
        "open_allocations": sum(row.get("allocation_open", False) for row in rows),
        "scope_note": "Validate selected prefixes for completeness; errors invalidate total. "
                      "Open bounds stop at observation time; future allocation is not included. "
                      "Creation-based bounds include queue time and are not an invoice.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "jobs"}, indent=2))
    for row in rows:
        print(row["name"], row.get("pick"), row.get("gpu_hours_bound"),
              row.get("accounting", row.get("error")))


if __name__ == "__main__":
    main()
