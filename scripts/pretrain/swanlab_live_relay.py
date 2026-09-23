"""Relay shared offline SwanLab 0.9.7 logs from an internet-enabled CPU host.

The training process remains offline. This reader retries partial tail records,
preserves metric steps, and uploads images through the public SDK. It never
marks an active source finished merely because it reached the current EOF.
The binary reader's seek fields are version-specific; install swanlab==0.9.7.
"""

import argparse
import fcntl
import json
from pathlib import Path
import time


def read_available(path, position, reader_factory, record_factory):
    """Return complete records and a cursor before any incomplete tail record."""
    reader = reader_factory()
    try:
        reader.open(path)
    except AssertionError:
        # A newly created log may be visible before its header is flushed.
        reader.close()
        return [], position
    if position is not None:
        reader._fp.seek(position)
        reader._index = position
    records = []
    try:
        while True:
            position = reader._fp.tell()
            try:
                blob = reader.scan()
            except (AssertionError, ValueError):
                break
            except Exception as exc:
                # The pinned reader uses this for a partly written payload.
                if type(exc).__name__ != "DataStoreError":
                    raise
                break
            if blob is None:
                break
            record = record_factory()
            record.ParseFromString(blob)
            records.append(record)
    finally:
        reader.close()
    return records, position


def source_config(run_dir):
    import yaml

    path = run_dir / "files/config.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    return {key: value.get("value") if isinstance(value, dict) else value
            for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--end-step", type=int, required=True)
    args = parser.parse_args()

    import swanlab
    from swanlab.proto.swanlab.record.v1.record_pb2 import Record
    from swanlab.sdk.internal.core_python.store import DataStoreReader

    if swanlab.__version__ != "0.9.7":
        raise RuntimeError("This binary-log relay requires swanlab==0.9.7")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity_path = args.state_dir / "identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
    positions, sources, last_steps = {}, {}, {}
    run = None
    count = 0
    latest_step = -1
    while True:
        source_finished = False
        for path in sorted(args.log_dir.glob("run-*/*.swanlab")):
            if sources.get(path) is False:
                continue
            records, position = read_available(path, positions.get(path), DataStoreReader, Record)
            if path not in sources:
                start = next((r.start for r in records if r.HasField("start")), None)
                if start is None:
                    continue
                sources[path] = start if start.name == args.source_name else False
                if sources[path] is False:
                    continue
            if run is None:
                config = source_config(path.parent)
                if config is None:
                    continue
                start = sources[path]
                if identity is None:
                    identity = {"id": start.id, "project": start.project,
                                "name": args.source_name}
                    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
                run = swanlab.init(
                    id=identity["id"], project=identity["project"],
                    name=identity["name"], resume="allow", mode="online",
                    workspace=start.workspace or None,
                    description="H200 offline training; metrics and images relayed from shared storage by a CPU notebook. Platform job status is authoritative during preemption or logging outages.",
                    config=config, log_dir=str(args.state_dir / "swanlog"),
                    settings=swanlab.Settings(interactive=False, probe=swanlab.Settings.Probe(
                        hardware=False, monitor=False, runtime=False,
                        requirements=False, conda=False, git=False)),
                )
                print("RELAY_STARTED", json.dumps(identity), flush=True)
            for record in records:
                if record.HasField("scalar"):
                    metric = record.scalar
                    if metric.key.startswith("__swanlab__."):
                        continue
                    if metric.step <= last_steps.get(metric.key, -1):
                        continue
                    swanlab.log({metric.key: metric.value.number}, step=metric.step)
                    last_steps[metric.key] = metric.step
                    latest_step = max(latest_step, metric.step)
                    count += 1
                elif record.HasField("media"):
                    metric = record.media
                    if metric.step <= last_steps.get(metric.key, -1):
                        continue
                    images = [swanlab.Image(str(path.parent / "media/image" / item.filename),
                                            caption=item.caption)
                              for item in metric.value.items]
                    swanlab.log({metric.key: images}, step=metric.step)
                    last_steps[metric.key] = metric.step
                    count += 1
                elif record.HasField("finish"):
                    source_finished = record.finish.state == 1
            positions[path] = position
        status = {"updated_unix": time.time(), "latest_source_step": latest_step,
                  "records_forwarded": count, "cloud_identity": identity,
                  "source_files": [str(p) for p, s in sources.items() if s is not False]}
        temporary = args.state_dir / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.state_dir / "status.json")
        if source_finished and latest_step >= args.end_step:
            swanlab.finish()
            print("RELAY_FINISHED", json.dumps(status), flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
