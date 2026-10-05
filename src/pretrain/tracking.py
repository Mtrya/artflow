"""Persist the experiment identity across checkpoint recovery and stage changes."""

import json
from pathlib import Path

TRACKING_RECORD = "tracking.json"


def validate_run_id(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or any(c.isspace() or ord(c) < 32 or c in "/\\#?%:" for c in value)
    ):
        raise ValueError(
            "missing or invalid SwanLab run ID; refusing a new experiment on resume"
        )
    return value


def resume_run_id(checkpoint):
    root = Path(checkpoint)
    record = root / TRACKING_RECORD
    if not record.exists():
        # Completed v1 checkpoints from the active pretrain kept identity in
        # the containing run directory. Keep this specific recovery contract.
        record = root.parent / "runtime.json"
    try:
        payload = json.loads(record.read_text())
        return validate_run_id(payload["swanlab_run_id"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot recover SwanLab identity from {record}: {exc}"
        ) from exc


def write_tracking_record(checkpoint, run_id):
    Path(checkpoint, TRACKING_RECORD).write_text(
        json.dumps({"swanlab_run_id": validate_run_id(run_id)}) + "\n"
    )
