"""Wait for a whole GPU node without reserving or preempting resources.

Reports availability only; the execution agent must validate the final job
snapshot and budget before submitting. Transient query failures do not imply
that resources are available. Checks default to five minutes apart.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def available_groups(payload, groups):
    if not payload.get("success"):
        raise ValueError("resource query did not succeed")
    return [row["compute_group"] for row in payload["data"]["items"]
            if row["compute_group"] in groups
            and row.get("free_nodes", 0) >= 1
            and row.get("available_gpus", 0) >= 8
            and row.get("gpus_per_node", 0) >= 8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--timeout-hours", type=float, default=12)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if args.interval < 60 or args.timeout_hours <= 0:
        parser.error("interval must be >=60 seconds and timeout positive")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.timeout_hours * 3600
        while time.monotonic() < deadline:
            state = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "workspace": args.workspace, "groups": args.group}
            try:
                result = subprocess.run(
                    ["inspire", "--json", "resources", "availability", "--workspace",
                     args.workspace, "--all"], capture_output=True, text=True, timeout=90)
                if result.returncode:
                    raise RuntimeError(f"resource query exit {result.returncode}")
                payload = json.loads(result.stdout)
                ready = available_groups(payload, args.group)
                state.update(status="ready" if ready else "waiting", ready_groups=ready,
                             resources=[row for row in payload["data"]["items"]
                                        if row["compute_group"] in args.group])
            except Exception as exc:
                state.update(status="query_error", error=str(exc))
            tmp = args.state.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            tmp.replace(args.state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if state["status"] == "ready":
                return 0
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        print("Resource watchdog timed out; no GPU job was submitted.", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
