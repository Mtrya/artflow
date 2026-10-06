"""Checkpoint-owned experiment identity and explicit experiment forks."""

import json
from pathlib import Path

TRACKING_RECORD = "tracking.json"


def validate_run_id(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or any(c.isspace() or ord(c) < 32 or c in "/\\#?%:" for c in value)
    ):
        raise ValueError("missing or invalid SwanLab run ID")
    return value


def read_tracking_record(checkpoint):
    record = Path(checkpoint) / TRACKING_RECORD
    try:
        payload = json.loads(record.read_text())
        if set(payload) != {"swanlab_project", "swanlab_run_id"}:
            raise ValueError("tracking record requires exactly project and run ID")
        project = payload["swanlab_project"]
        if not isinstance(project, str) or not project.strip():
            raise ValueError("missing SwanLab project")
        validate_run_id(payload["swanlab_run_id"])
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot recover SwanLab identity from {record}: {exc}"
        ) from exc


def tracker_init_kwargs(checkpoint, project, *, new_experiment=False):
    """A training-state restore need not continue the source experiment."""
    if new_experiment and not checkpoint:
        raise ValueError("--new-experiment requires --resume")
    kwargs = {"mode": "online", "resume": "never"}
    if checkpoint:
        identity = read_tracking_record(checkpoint)
        if not new_experiment:
            if identity["swanlab_project"] != project:
                raise ValueError(
                    "checkpoint SwanLab project differs; use --new-experiment "
                    "to fork training state explicitly"
                )
            kwargs.update(id=identity["swanlab_run_id"], resume="must")
    return kwargs


def write_tracking_record(checkpoint, project, run_id):
    if not isinstance(project, str) or not project.strip():
        raise ValueError("missing SwanLab project")
    Path(checkpoint, TRACKING_RECORD).write_text(
        json.dumps(
            {"swanlab_project": project, "swanlab_run_id": validate_run_id(run_id)}
        )
        + "\n"
    )
