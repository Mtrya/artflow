"""Split a historical single AdamW group without resetting training state.

This is an explicit offline migration, never an automatic resume fallback.
The destination must be new. Only conditioning-matrix decay and the run's
output/name may change; the original checkpoint remains untouched. Run with
``python -m scripts.pretrain.migrate_conditioning_decay --help``.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import torch

from src.models.artflow import ArtFlow
from src.pretrain.config import load_config
from src.pretrain.muon import CONDITIONING_WEIGHTS, build_param_groups
from src.pretrain.stage_control import (
    CHECKPOINT_RECORD, validate_checkpoint, write_checkpoint_record,
)
from src.pretrain.state_verification import require_exact_state


def split_adam_state(model, optimizer, saved, scheduler):
    """Map historical parameter IDs by name, preserving every moment tensor."""
    if len(saved["param_groups"]) != 1 or len(optimizer.param_groups) != 2:
        raise ValueError("migration requires one historical and two destination AdamW groups")
    old_group = saved["param_groups"][0]
    names = {id(p): name for name, p in model.named_parameters()}
    current_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    # The historical single group followed model.named_parameters() order.
    old_params = [(name, p) for name, p in model.named_parameters() if id(p) in current_ids]
    old_ids = old_group["params"]
    if old_ids != list(range(len(old_params))) or not saved["state"].keys() <= set(old_ids):
        raise ValueError("historical AdamW parameter inventory differs")
    old_by_name = dict(zip((name for name, _ in old_params), old_ids))
    selected = {names[id(p)] for p in optimizer.param_groups[1]["params"]}
    if selected != CONDITIONING_WEIGHTS:
        raise ValueError("conditioning group must contain exactly the three matrices")

    migrated = {"state": {}, "param_groups": []}
    mapping = {}
    for live_group, serialized in zip(optimizer.param_groups, optimizer.state_dict()["param_groups"]):
        for key in ("betas", "eps", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
            if old_group[key] != serialized[key]:
                raise ValueError(f"AdamW setting changed: {key}")
        group = copy.deepcopy(old_group)
        group.update(params=serialized["params"], weight_decay=serialized["weight_decay"])
        migrated["param_groups"].append(group)
        for p, new_id in zip(live_group["params"], serialized["params"]):
            name = names[id(p)]
            old_id = old_by_name[name]
            mapping[name] = dict(old_id=old_id, new_id=new_id)
            if old_id in saved["state"]:
                state = saved["state"][old_id]
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].shape != p.shape:
                        raise ValueError(f"moment shape mismatch: {name}.{key}")
                migrated["state"][new_id] = state

    migrated_scheduler = copy.deepcopy(scheduler)
    for key in ("base_lrs", "_last_lr", "lr_lambdas"):
        values = scheduler[key]
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"expected a one-group LambdaLR state: {key}")
        migrated_scheduler[key] = values * 2
    if scheduler["lr_lambdas"] != [None]:
        raise ValueError("expected the native schedule's stateless lambda")
    if old_group["lr"] != scheduler["_last_lr"][0]:
        raise ValueError("AdamW and scheduler learning rates disagree")
    return migrated, migrated_scheduler, mapping


def check_recipe_change(old, new):
    expected = copy.deepcopy(old)
    if "adam_conditioning_wd" in expected["optim"]:
        raise ValueError("source already uses separate conditioning decay")
    expected["optim"]["adam_conditioning_wd"] = new["optim"]["adam_conditioning_wd"]
    expected["paths"]["output_dir"] = new["paths"]["output_dir"]
    expected["train"]["run_name"] = new["train"]["run_name"]
    if expected != new:
        raise ValueError("only conditioning decay, output_dir and run_name may change")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def migrate(source, destination, config_path, *, reason):
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
    old_recipe = json.loads((source / "run_config.json").read_text())
    check_recipe_change(old_recipe, recipe)
    metadata = json.loads((source / "transformer_config.json").read_text())
    if metadata != dict(recipe["model"], architecture=ArtFlow.ARCHITECTURE):
        raise ValueError("model metadata differs from the target recipe")
    with torch.device("meta"):
        model = ArtFlow(**metadata)
    o = config.optim
    _, adam = build_param_groups(
        model, muon_lr=o.muon_lr, muon_wd=o.muon_wd, adam_lr=o.learning_rate,
        adam_wd=o.adam_wd, adam_conditioning_wd=o.adam_conditioning_wd,
        adam_eps=o.adam_eps, adam_betas=tuple(o.adam_betas), muon_momentum=o.muon_momentum,
    )
    saved = torch.load(source / "optimizer_1.bin", map_location="cpu", weights_only=False)
    scheduler = torch.load(source / "scheduler_1.bin", map_location="cpu", weights_only=False)
    if saved["param_groups"][0]["weight_decay"] != o.adam_wd:
        raise ValueError("historical AdamW decay differs from its recipe")
    if scheduler["last_epoch"] != step or scheduler["base_lrs"] != [o.learning_rate]:
        raise ValueError("historical scheduler step or base LR differs")
    migrated, migrated_scheduler, mapping = split_adam_state(model, adam, saved, scheduler)
    source_hashes = {name: sha256(source / name) for name in record["files"]}
    # Copy independently: editing a hardlinked config/state would corrupt the
    # protected input. Publish the new completion record only after verification.
    destination.mkdir(parents=True)
    for name in record["files"]:
        shutil.copy2(source / name, destination / name)
    torch.save(migrated, destination / "optimizer_1.bin")
    torch.save(migrated_scheduler, destination / "scheduler_1.bin")
    (destination / "run_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
    actual = torch.load(destination / "optimizer_1.bin", map_location="cpu", weights_only=False)
    for name, ids in mapping.items():
        if ids["old_id"] in saved["state"]:
            require_exact_state(saved["state"][ids["old_id"]], actual["state"][ids["new_id"]], label=name)
    require_exact_state(migrated_scheduler, torch.load(destination / "scheduler_1.bin", weights_only=False), label="scheduler")
    destination_hashes = {name: sha256(destination / name) for name in record["files"]}
    changed = {"optimizer_1.bin", "scheduler_1.bin", "run_config.json"}
    if any(source_hashes[n] != destination_hashes[n] for n in record["files"] if n not in changed):
        raise ValueError("an artifact outside the declared migration changed")
    provenance = dict(
        migration="split-conditioning-adamw-v1", reason=reason, source=str(source),
        global_step=step, source_completion_sha256=sha256(source / CHECKPOINT_RECORD),
        source_sha256=source_hashes, destination_sha256=destination_hashes,
        source_recipe=old_recipe, destination_recipe=recipe, parameter_mapping=mapping,
        preserved="Model, EMA, Muon, sampler and RNG files byte-identical; AdamW moments exact by parameter name; schedule position unchanged.",
    )
    (destination / "migration.json").write_text(json.dumps(provenance, indent=2) + "\n")
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
