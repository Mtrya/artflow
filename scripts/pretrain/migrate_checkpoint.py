"""Migrate checkpoint metadata and recipe on an independent, verified copy.

Model, optimizer, scheduler, EMA and training RNG artifacts remain byte-identical.
Sampler state is preserved unless --reset-sampler explicitly rebuilds it for
filtered row pools, discarding queued/prefetched draws at the saved global step.
Interior stage endpoints preserve sampler files: the next stage starts fresh.
Future stages may change data/batching; the current stage may only reweight
the same entries. Model capacity, completed stages and the global schedule
cannot change. Old artifact conventions are read here, never in the trainer.
"""

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.models.inko import Inko
from src.pretrain.config import load_config, stage_caption_policy
from src.pretrain.stage_control import (
    CHECKPOINT_RECORD,
    validate_checkpoint,
    validate_checkpoint_inventory,
    write_checkpoint_record,
)
from src.pretrain.tracking import (
    TRACKING_RECORD,
    read_tracking_record,
    validate_run_id,
    write_tracking_record,
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_source_assets(path):
    """Read an explicitly supplied archive of source config bytes."""
    assets = json.loads(Path(path).read_text())
    if not isinstance(assets, dict):
        raise TypeError("source assets must map original paths to text and sha256")
    result = {}
    for name, entry in assets.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"text", "sha256"}
            or not isinstance(entry["text"], str)
        ):
            raise ValueError(f"invalid source asset: {name}")
        content = entry["text"].encode()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"source asset hash mismatch: {name}")
        result[name] = content
    return result


def read_source_asset(path, archived_assets):
    if archived_assets is None:
        return Path(path).read_bytes()
    if path not in archived_assets:
        raise ValueError(f"source assets record is missing: {path}")
    return archived_assets[path]


def check_recipe_change(old, new, *, step, active_bucket, archived_assets=None):
    """Allow future data/batching changes and verified artifact relocation only."""
    expected = copy.deepcopy(old)
    if [s["name"] for s in old["stages"]] != [s["name"] for s in new["stages"]]:
        raise ValueError("stage names and order must remain unchanged")
    if [s["end_step"] for s in old["stages"]] != [s["end_step"] for s in new["stages"]]:
        raise ValueError("stage endpoints and global schedule must remain unchanged")
    active = next(
        (i for i, s in enumerate(old["stages"]) if step <= s["end_step"]), None
    )
    if active is None:
        raise ValueError("checkpoint is outside the source curriculum")

    # Run output locations are operational; optimizer/model/data behavior is
    # checked below against the entire resolved recipe.
    expected["paths"]["output_dir"] = new["paths"]["output_dir"]
    expected["train"]["run_name"] = new["train"]["run_name"]
    expected["telemetry"]["swanlab_project"] = new["telemetry"]["swanlab_project"]
    if old["eval"]["prompts_file"] != new["eval"]["prompts_file"]:
        if (
            read_source_asset(old["eval"]["prompts_file"], archived_assets)
            != Path(new["eval"]["prompts_file"]).read_bytes()
        ):
            raise ValueError("relocated evaluation prompts must be byte-identical")
        expected["eval"]["prompts_file"] = new["eval"]["prompts_file"]

    amendments = []
    for i, (before, after) in enumerate(zip(old["stages"], new["stages"])):
        if i > active:
            for key in (
                "datasets",
                "bucket_plan",
                "gradient_accumulation_steps",
                "eval_dataset_path",
            ):
                expected["stages"][i][key] = after[key]
                if before[key] != after[key]:
                    amendments.append(
                        {
                            "stage": before["name"],
                            "field": key,
                            "old": before[key],
                            "new": after[key],
                        }
                    )
            if not Path(after["bucket_plan"]).is_file():
                raise ValueError(
                    f"future stage plan is missing: {after['bucket_plan']}"
                )
        elif i == active or before["bucket_plan"] != after["bucket_plan"]:
            reference = (
                active_bucket
                if i == active
                else json.loads(
                    read_source_asset(before["bucket_plan"], archived_assets)
                )
            )
            if reference != json.loads(Path(after["bucket_plan"]).read_text()):
                raise ValueError(
                    "completed/current stage bucket contents cannot change"
                )
            expected["stages"][i]["bucket_plan"] = after["bucket_plan"]
        if i == active and before["datasets"] != after["datasets"]:
            # Saved queues and cycles refer to the exact dataset entry order.
            before_paths = [d["path"] for d in before["datasets"]]
            after_paths = [d["path"] for d in after["datasets"]]
            if after_paths != before_paths:
                raise ValueError(
                    "current-stage data amendments may only reweight the same datasets"
                )
            expected["stages"][i]["datasets"] = after["datasets"]
            amendments.append(
                {
                    "stage": before["name"],
                    "field": "datasets",
                    "old": before["datasets"],
                    "new": after["datasets"],
                }
            )
    if expected != new:
        raise ValueError(
            "only future-stage data/batching and verified operational locations may change"
        )
    return amendments


def source_tracking(source, old_recipe, source_run_id):
    """Import identity explicitly; never infer it from a mutable parent run."""
    project = old_recipe["telemetry"]["swanlab_project"]
    path = source / TRACKING_RECORD
    if path.is_file():
        saved = json.loads(path.read_text())
        if set(saved) not in (
            {"swanlab_run_id"},
            {"swanlab_project", "swanlab_run_id"},
        ):
            raise ValueError("invalid source tracking record")
        if "swanlab_project" in saved:
            project = read_tracking_record(source)["swanlab_project"]
        run_id = validate_run_id(saved["swanlab_run_id"])
        if source_run_id is not None and source_run_id != run_id:
            raise ValueError("--source-run-id disagrees with the checkpoint")
    else:
        if source_run_id is None:
            raise ValueError(
                "checkpoint has no tracking record; --source-run-id is required"
            )
        run_id = validate_run_id(source_run_id)
    return {"swanlab_project": project, "swanlab_run_id": run_id}


def reset_sampler_files(staging, config, *, step, world_size):
    """Write fresh rank-local cycles for the target pools, outside the trainer."""
    import torch
    from datasets import load_from_disk

    from src.dataset.length_metadata import RowLengthMetadata
    from src.dataset.sampler import RowLengthQueueBatchSampler, load_bucket_plan

    stage = next(s for s in config.stages if step <= s.end_step)
    metadata, datasets = [], []
    for entry in stage.datasets:
        path = Path(entry.path)
        sidecar = path / "length_metadata.npz"
        lengths = RowLengthMetadata.load(sidecar)
        lengths.validate_against_dataset(load_from_disk(str(path)))
        metadata.append(lengths)
        datasets.append({
            "path": str(path), "rows": lengths.num_rows,
            "state_sha256": sha256(path / "state.json"),
            "sidecar_sha256": sha256(sidecar),
        })
    plan = load_bucket_plan(
        stage.bucket_plan,
        {int(r) for m in metadata for r in m.resolution_ids},
    )
    policy, _ = stage_caption_policy(config, stage.name)
    progress = config.data.curriculum_start + (
        config.data.curriculum_end - config.data.curriculum_start
    ) * step / config.max_steps
    ranks = []
    for rank in range(world_size):
        path = staging / f"sampler_state_rank_{rank:05d}.pt"
        old = torch.load(path, map_location="cpu", weights_only=False)
        if old.get("version") != RowLengthQueueBatchSampler.STATE_VERSION:
            raise ValueError("unsupported source sampler version for reset")
        next_id = old["next_batch_id"]
        if type(next_id) is not int or next_id < 0:
            raise ValueError("invalid source sampler batch counter")
        seed = config.train.seed + rank
        sampler = RowLengthQueueBatchSampler(
            metadata, plan, [d.weight for d in stage.datasets],
            num_replicas=world_size, rank=rank, seed=seed,
            initial_stage=progress, caption_policy=policy,
        )
        state = sampler.state_dict()
        state["next_batch_id"] = next_id
        torch.save(state, path)
        ranks.append({
            "rank": rank, "seed": seed, "next_batch_id": next_id,
            "source_cycle_rows": [len(c) for c in old["cycles"]],
            "destination_cycle_rows": [len(c) for c in state["cycles"]],
            "discarded_pending_rows": sum(map(len, old["queues"].values()))
            + sum(len(b) for k in ("inflight", "replay", "ready_batches")
                  for b in old[k]),
        })
    return {
        "policy": "Fresh shuffled cycles; empty queues and prefetch; seed = recipe seed + rank; batch counters preserved.",
        "stage": stage.name, "curriculum_position": progress,
        "datasets": datasets, "ranks": ranks,
    }


def migrate(
    source,
    destination,
    config_path,
    *,
    storage_root,
    reason,
    source_run_id=None,
    source_assets=None,
    reset_sampler=False,
):
    if not reason.strip():
        raise ValueError("a migration reason is required")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    config = load_config(config_path, storage_root=storage_root)
    recipe = asdict(config)
    record = json.loads((source / CHECKPOINT_RECORD).read_text())
    step = record["global_step"]
    if (
        record.get("version") != 1
        or record.get("complete") is not True
        or type(step) is not int
        or not 0 <= step <= config.max_steps
        or source.name != f"checkpoint_step_{step:06d}"
        or record.get("max_steps") != config.max_steps
        or record.get("scheduler_count") != 2
        or record.get("use_ema") is not True
        or record.get("device_type") != "npu"
    ):
        raise ValueError(
            "source must be a complete checkpoint of the same training schedule"
        )
    validate_checkpoint_inventory(source, record)
    if destination.name != source.name or destination.exists():
        raise ValueError(
            "destination must be a new directory with the same checkpoint_step_* name"
        )
    if source in destination.parents:
        raise ValueError("destination must be outside the source checkpoint")
    old = json.loads((source / "run_config.json").read_text())
    active_bucket = json.loads((source / "bucket_plan.json").read_text())
    archived_assets = (
        load_source_assets(source_assets) if source_assets is not None else None
    )
    amendments = check_recipe_change(
        old,
        recipe,
        step=step,
        active_bucket=active_bucket,
        archived_assets=archived_assets,
    )
    metadata = json.loads((source / "transformer_config.json").read_text())
    # This is an explicit metadata-only import; tensor keys and geometry agree.
    source_architecture = metadata.get("architecture")
    if source_architecture not in {"artflow-v2", Inko.ARCHITECTURE} or metadata != dict(
        recipe["model"], architecture=source_architecture
    ):
        raise ValueError("model metadata differs from the target recipe")
    target_metadata = dict(recipe["model"], architecture=Inko.ARCHITECTURE)
    recipe_digest = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()
    ).hexdigest()
    provenance_name = f"checkpoint_migration_{recipe_digest[:12]}.json"
    if provenance_name in record["files"]:
        raise ValueError("this target recipe already has migration provenance")
    identity = source_tracking(source, old, source_run_id)
    source_hashes = {name: sha256(source / name) for name in record["files"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-migration-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        for name in record["files"]:
            shutil.copy2(source / name, staging / name)
            if sha256(staging / name) != source_hashes[name]:
                raise ValueError(f"checkpoint artifact changed while copying: {name}")
        (staging / "run_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
        (staging / "transformer_config.json").write_text(
            json.dumps(target_metadata, indent=2) + "\n"
        )
        write_tracking_record(
            staging, identity["swanlab_project"], identity["swanlab_run_id"]
        )
        # The next stage constructs its own sampler at a resolution transition.
        at_interior_endpoint = any(
            step == stage.end_step for stage in config.stages[:-1]
        )
        sampler_reset = (
            reset_sampler_files(
                staging, config, step=step, world_size=record["world_size"]
            )
            if reset_sampler and not at_interior_endpoint else None
        )
        destination_hashes = {p.name: sha256(p) for p in staging.iterdir()}
        provenance = {
            "migration": "checkpoint-metadata",
            "reason": reason,
            "source": str(source),
            "global_step": step,
            "source_completion_sha256": sha256(source / CHECKPOINT_RECORD),
            "source_assets": {
                "path": str(Path(source_assets).resolve()),
                "sha256": sha256(source_assets),
            }
            if source_assets is not None
            else None,
            "source_sha256": source_hashes,
            "destination_sha256": destination_hashes,
            "source_recipe": old,
            "destination_recipe": recipe,
            "amendments": amendments,
            "source_model": metadata,
            "destination_model": target_metadata,
            "source_tracking": identity,
            "new_experiment_required": identity["swanlab_project"]
            != config.telemetry.swanlab_project,
            "sampler_reset": sampler_reset,
            "preserved": (
                "Model, optimizer, scheduler, EMA and training RNG artifacts are byte-identical; sampler files are explicitly rebuilt."
                if sampler_reset is not None else
                "Model, optimizer, scheduler, EMA, sampler and RNG artifacts are byte-identical."
            ),
        }
        (staging / provenance_name).write_text(json.dumps(provenance, indent=2) + "\n")
        write_checkpoint_record(
            staging,
            step=step,
            max_steps=config.max_steps,
            scheduler_count=2,
            use_ema=True,
            world_size=record["world_size"],
            device_type="npu",
        )
        validate_checkpoint(staging, max_steps=config.max_steps, device_type="npu")
        if destination.exists():
            raise ValueError("destination appeared during migration")
        staging.rename(destination)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument(
        "--reset-sampler", action="store_true",
        help="Rebuild shuffled cycles from target datasets and discard queued draws; preserve global training state",
    )
    parser.add_argument(
        "--source-assets",
        help="Explicit JSON record of original config paths, text and SHA-256 hashes",
    )
    parser.add_argument(
        "--source-run-id",
        help="Required for a source checkpoint without its own tracking record",
    )
    args = parser.parse_args()
    report = migrate(
        args.source,
        args.destination,
        args.config,
        storage_root=args.storage_root,
        reason=args.reason,
        source_run_id=args.source_run_id,
        source_assets=args.source_assets,
        reset_sampler=args.reset_sampler,
    )
    print(
        json.dumps(
            {
                "destination": args.destination,
                "step": report["global_step"],
                "preserved": report["preserved"],
            }
        )
    )
