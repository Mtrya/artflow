"""Apply explicit recipe amendments to a complete checkpoint copy.

Model, optimizer, scheduler, EMA, sampler and RNG artifacts remain byte-identical.
Future stages may change data/batching freely; the current stage may only
reweight existing datasets and append new ones (appended rows sit above every
saved dataset id, so the saved sampler state restores cleanly); completed
stages and the global schedule cannot change. This operation records the old
and new recipes; ordinary resume remains strict.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from src.pretrain.config import load_config
from src.pretrain.stage_control import (
    CHECKPOINT_RECORD,
    validate_checkpoint,
    write_checkpoint_record,
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_recipe_change(old, new, *, step, active_bucket):
    """Allow future data/batching changes and verified artifact relocation only."""
    expected = copy.deepcopy(old)
    if [s["name"] for s in old["stages"]] != [s["name"] for s in new["stages"]]:
        raise ValueError("stage names and order must remain unchanged")
    if [s["end_step"] for s in old["stages"]] != [s["end_step"] for s in new["stages"]]:
        raise ValueError("stage endpoints and global schedule must remain unchanged")
    active = next((i for i, s in enumerate(old["stages"]) if step <= s["end_step"]), None)
    if active is None:
        raise ValueError("checkpoint is outside the source curriculum")

    # Run output locations are operational; optimizer/model/data behavior is
    # checked below against the entire resolved recipe.
    expected["paths"]["output_dir"] = new["paths"]["output_dir"]
    expected["train"]["run_name"] = new["train"]["run_name"]
    if old["eval"]["prompts_file"] != new["eval"]["prompts_file"]:
        if sha256(old["eval"]["prompts_file"]) != sha256(new["eval"]["prompts_file"]):
            raise ValueError("relocated evaluation prompts must be byte-identical")
        expected["eval"]["prompts_file"] = new["eval"]["prompts_file"]

    amendments = []
    for i, (before, after) in enumerate(zip(old["stages"], new["stages"])):
        if i > active:
            for key in ("datasets", "bucket_plan", "gradient_accumulation_steps", "eval_dataset_path"):
                expected["stages"][i][key] = after[key]
                if before[key] != after[key]:
                    amendments.append(dict(stage=before["name"], field=key, old=before[key], new=after[key]))
            if not Path(after["bucket_plan"]).is_file():
                raise ValueError(f"future stage plan is missing: {after['bucket_plan']}")
        elif i == active or before["bucket_plan"] != after["bucket_plan"]:
            reference = active_bucket if i == active else json.loads(Path(before["bucket_plan"]).read_text())
            if reference != json.loads(Path(after["bucket_plan"]).read_text()):
                raise ValueError("completed/current stage bucket contents cannot change")
            expected["stages"][i]["bucket_plan"] = after["bucket_plan"]
        if i == active and before["datasets"] != after["datasets"]:
            # Mid-stage data amendments may reweight existing entries and append
            # new ones; removal or reordering breaks the saved sampler's
            # dataset-id prefix (see sampler load_state_dict).
            before_paths = [d["path"] for d in before["datasets"]]
            after_paths = [d["path"] for d in after["datasets"]]
            if len(after_paths) < len(before_paths) or after_paths[:len(before_paths)] != before_paths:
                raise ValueError(
                    "current-stage data amendments may only reweight existing datasets and append new ones")
            expected["stages"][i]["datasets"] = after["datasets"]
            amendments.append(dict(stage=before["name"], field="datasets",
                                   old=before["datasets"], new=after["datasets"]))
    if expected != new:
        raise ValueError("only future-stage data/batching and verified operational locations may change")
    return amendments


def migrate(source, destination, config_path, *, reason):
    if not reason.strip():
        raise ValueError("a migration reason is required")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    config = load_config(config_path)
    recipe = asdict(config)
    record = json.loads((source / CHECKPOINT_RECORD).read_text())
    step = validate_checkpoint(
        source, max_steps=config.max_steps, require_record=True,
        scheduler_count=2, use_ema=True, world_size=record["world_size"], device_type="npu",
    )
    if destination.name != source.name or destination.exists():
        raise ValueError("destination must be a new directory with the same checkpoint_step_* name")
    if source in destination.parents:
        raise ValueError("destination must be outside the source checkpoint")
    old = json.loads((source / "run_config.json").read_text())
    active_bucket = json.loads((source / "bucket_plan.json").read_text())
    amendments = check_recipe_change(old, recipe, step=step, active_bucket=active_bucket)
    metadata = json.loads((source / "transformer_config.json").read_text())
    if metadata != dict(recipe["model"], architecture="artflow-v2"):
        raise ValueError("model metadata differs from the target recipe")
    recipe_digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    provenance_name = f"stage_recipe_migration_{recipe_digest[:12]}.json"
    if provenance_name in record["files"]:
        raise ValueError("this target recipe already has migration provenance")
    source_hashes = {name: sha256(source / name) for name in record["files"]}
    destination.mkdir(parents=True)
    for name in record["files"]:
        shutil.copy2(source / name, destination / name)
    (destination / "run_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
    destination_hashes = {name: sha256(destination / name) for name in record["files"]}
    if any(source_hashes[n] != destination_hashes[n] for n in record["files"] if n != "run_config.json"):
        raise ValueError("an artifact outside the declared recipe amendment changed")
    provenance = dict(
        migration="stage-recipe-v2", reason=reason, source=str(source), global_step=step,
        source_completion_sha256=sha256(source / CHECKPOINT_RECORD),
        source_sha256=source_hashes, destination_sha256=destination_hashes,
        source_recipe=old, destination_recipe=recipe, amendments=amendments,
        preserved="All training-state artifacts byte-identical; only run_config.json amended.",
    )
    # Distinct from earlier conditioning-decay provenance, which stays intact.
    (destination / provenance_name).write_text(json.dumps(provenance, indent=2) + "\n")
    write_checkpoint_record(
        destination, step=step, max_steps=config.max_steps, scheduler_count=2,
        use_ema=True, world_size=record["world_size"], device_type="npu",
    )
    validate_checkpoint(destination, max_steps=config.max_steps, require_record=True, device_type="npu")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    report = migrate(args.source, args.destination, args.config, reason=args.reason)
    print(json.dumps({"destination": args.destination, "step": report["global_step"], "preserved": report["preserved"]}))
