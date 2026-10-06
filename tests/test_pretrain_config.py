"""Path ownership, recipe-derived policy, and deliberate recovery migration."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.pretrain import migrate_checkpoint as migration
from scripts.pretrain.plan_buckets import calibration_points, load_sidecar_lengths
from src.dataset.captions import CaptionPolicy
from src.dataset.length_metadata import RowLengthMetadata, prompt_metadata_contract
from src.dataset.mix import DatasetEntry
from src.pretrain.config import (
    REPOSITORY_ROOT,
    DatasetConfig,
    load_config,
    stage_caption_policy,
)
from src.pretrain.stage_control import (
    validate_checkpoint,
    validate_checkpoint_recipe,
    write_checkpoint_record,
)
from src.pretrain.tracking import (
    read_tracking_record,
    tracker_init_kwargs,
    write_tracking_record,
)


@pytest.fixture
def recipe(tmp_path):
    # The shipped file supplies a complete input, not any expected test value.
    config = load_config(
        REPOSITORY_ROOT / "configs/pretrain.toml", storage_root=tmp_path / "heavy"
    )
    return replace(
        config,
        stages=[
            replace(stage, name=f"s{i}", end_step=step, grid_steps=[])
            for i, (stage, step) in enumerate(zip(config.stages, (10, 20, 40)))
        ],
        data=replace(
            config.data,
            curriculum_start=0.2,
            curriculum_end=0.8,
            caption_beta_start=-2,
            caption_beta_end=2,
            caption_short_reserve=0.1,
            caption_short_threshold=4,
        ),
    )


def test_config_assets_resolve_from_repository_after_chdir(tmp_path, monkeypatch):
    path = tmp_path / "recipe.toml"
    text = (REPOSITORY_ROOT / "configs/pretrain.toml").read_text()
    text = text.replace(
        "configs/prompts/monitor.jsonl", "configs/prompts/fixture.jsonl"
    )
    path.write_text(text)
    monkeypatch.chdir(tmp_path)
    config = load_config(path, storage_root=tmp_path / "heavy")
    assert (
        Path(config.eval.prompts_file)
        == REPOSITORY_ROOT / "configs/prompts/fixture.jsonl"
    )
    assert all(
        Path(s.bucket_plan).parent == REPOSITORY_ROOT / "configs/bucket_plans"
        for s in config.stages
    )
    assert Path(config.paths.vae).is_relative_to(tmp_path / "heavy")
    assert all(
        Path(d.path).is_relative_to(tmp_path / "heavy")
        for s in config.stages
        for d in s.datasets
    )


def test_recipe_controls_caption_policy_and_global_stage_interval(recipe):
    policy, interval = stage_caption_policy(recipe, "s1")
    assert interval == pytest.approx((0.35, 0.5))  # .2 + .6 * (10/40, 20/40)
    assert policy.beta(0.35) == pytest.approx(-0.6)
    assert policy.short_reserve == 0.1
    assert policy.short_threshold == 4


def test_planner_weights_images_then_captions_with_explicit_policy(tmp_path):
    RowLengthMetadata(np.array([1, 1]), np.array([0, 2, 3]), np.array([2, 6, 6])).save(
        tmp_path / "length_metadata.npz"
    )
    records = load_sidecar_lengths(
        [DatasetEntry(tmp_path, 1.0, "fixture")],
        policy=CaptionPolicy(
            beta_start=1, beta_end=1, short_reserve=0.2, short_threshold=4
        ),
        progress_start=0.2,
        progress_end=0.7,
        progress_grid=2,
    )
    # Each image has half the mass. First row's captions split it 2/5 : 3/5.
    assert records[0].weights[1].tolist() == pytest.approx([0.2, 0.3, 0.5])


@pytest.mark.parametrize("bad", [None, "", "../run", "bad run", 123])
def test_resume_rejects_missing_or_invalid_identity(tmp_path, bad):
    checkpoint = tmp_path / "checkpoint_step_000015"
    checkpoint.mkdir()
    (checkpoint / "tracking.json").write_text(
        json.dumps({"swanlab_project": "experiment-project", "swanlab_run_id": bad})
    )
    with pytest.raises(ValueError, match="SwanLab"):
        read_tracking_record(checkpoint)


def test_resume_never_uses_parent_identity_or_hides_corruption(tmp_path):
    checkpoint = tmp_path / "checkpoint_step_000015"
    checkpoint.mkdir()
    (tmp_path / "runtime.json").write_text('{"swanlab_run_id": "old-run"}')
    with pytest.raises(ValueError):
        read_tracking_record(checkpoint)
    write_tracking_record(checkpoint, "original-project", "saved-run")
    assert tracker_init_kwargs(checkpoint, "original-project") == {
        "mode": "online",
        "id": "saved-run",
        "resume": "must",
    }
    with pytest.raises(ValueError, match="project differs"):
        tracker_init_kwargs(checkpoint, "another-project")
    assert tracker_init_kwargs(checkpoint, "another-project", new_experiment=True) == {
        "mode": "online",
        "resume": "never",
    }
    with pytest.raises(ValueError, match="requires --resume"):
        tracker_init_kwargs(None, "another-project", new_experiment=True)
    (checkpoint / "tracking.json").write_text('{"swanlab_run_id": null}')
    with pytest.raises(ValueError):
        tracker_init_kwargs(checkpoint, "another-project", new_experiment=True)


@pytest.mark.parametrize("archive_source_assets,reset_sampler", [(False, False), (True, False), (True, True)])
def test_migration_rebrands_metadata_preserves_training_state_and_records_origin(
    tmp_path, recipe, monkeypatch, capsys, archive_source_assets, reset_sampler
):
    if reset_sampler:
        from datasets import Dataset

        pool = tmp_path / "filtered-pool"
        Dataset.from_dict({"captions": [["a"], ["b"], ["c"]]}).save_to_disk(str(pool))
        RowLengthMetadata(
            np.array([1, 1, 1]), np.array([0, 1, 2, 3]), np.array([2, 3, 4]),
            metadata_info=prompt_metadata_contract(3),
        ).save(pool / "length_metadata.npz")
        stages = list(recipe.stages)
        stages[1] = replace(stages[1], datasets=[DatasetConfig(str(pool), 1.0)])
        recipe = replace(recipe, stages=stages)
    old = asdict(recipe)
    old["telemetry"]["swanlab_project"] = "original-project"
    old_prompt = tmp_path / "old-prompts.jsonl"
    old_prompt.write_bytes(Path(recipe.eval.prompts_file).read_bytes())
    old["eval"]["prompts_file"] = str(old_prompt)
    for i, stage in enumerate(old["stages"]):
        old_plan = tmp_path / f"old-plan-{i}.json"
        old_plan.write_bytes(Path(stage["bucket_plan"]).read_bytes())
        stage["bucket_plan"] = str(old_plan)
    source = tmp_path / "original" / "checkpoint_step_000015"
    source.mkdir(parents=True)
    (source / "run_config.json").write_text(json.dumps(old))
    (source / "transformer_config.json").write_text(
        json.dumps(dict(asdict(recipe.model), architecture="artflow-v2"))
    )
    (source / "bucket_plan.json").write_bytes(
        Path(recipe.stages[1].bucket_plan).read_bytes()
    )
    artifacts = [
        "model.safetensors",
        "ema_weights.pt",
        "optimizer.bin",
        "optimizer_1.bin",
        "scheduler.bin",
        "scheduler_1.bin",
        "sampler_state_rank_00000.pt",
        "random_states_0.pkl",
        "npu_rng_state_rank_00000.pt",
    ]
    for name in artifacts:
        (source / name).write_bytes(b"opaque saved training state")
    if reset_sampler:
        torch.save({
            "version": 1, "cycles": [[100, 101]], "next_batch_id": 37,
            "queues": {(1, 0): [(0, 100, 0, 1, 2, 0, 16, -1)]},
            "inflight": [[(0, 101, 0, 1, 2, 0, 16, 36)]],
            "replay": [], "ready_batches": [],
        }, source / "sampler_state_rank_00000.pt")
    write_tracking_record(source, "original-project", "continuing-run")
    write_checkpoint_record(
        source,
        step=15,
        max_steps=40,
        scheduler_count=2,
        use_ema=True,
        world_size=1,
        device_type="npu",
    )
    # Source fixture predates checkpoint-owned tracking. Parent identity is
    # deliberately misleading: the import must require an explicit run ID.
    (source / "tracking.json").unlink()
    record_path = source / "training_state.json"
    record = json.loads(record_path.read_text())
    del record["files"]["tracking.json"]
    record_path.write_text(json.dumps(record))
    (source.parent / "runtime.json").write_text('{"swanlab_run_id": "wrong-run"}')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    monkeypatch.setattr(migration, "load_config", lambda *a, **kw: recipe)
    destination = tmp_path / "migrated" / source.name
    with pytest.raises(ValueError, match="--source-run-id is required"):
        migration.migrate(
            source,
            destination,
            "input.toml",
            storage_root=tmp_path,
            reason="Import preserved checkpoint",
        )
    assert not destination.exists()
    report = migration.migrate(
        source,
        destination,
        "input.toml",
        storage_root=tmp_path,
        reason="Relocate tracked configuration",
        source_run_id="continuing-run",
        reset_sampler=reset_sampler,
    )
    assert (
        validate_checkpoint(destination, max_steps=40, world_size=1, device_type="npu")
        == 15
    )
    assert read_tracking_record(destination) == {
        "swanlab_project": "original-project",
        "swanlab_run_id": "continuing-run",
    }
    assert (
        json.loads((destination / "transformer_config.json").read_text())[
            "architecture"
        ]
        == "inko"
    )
    with pytest.raises(ValueError, match="project differs"):
        tracker_init_kwargs(destination, recipe.telemetry.swanlab_project)
    assert tracker_init_kwargs(
        destination, recipe.telemetry.swanlab_project, new_experiment=True
    ) == {"mode": "online", "resume": "never"}
    assert json.loads((destination / "run_config.json").read_text()) == asdict(recipe)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    preserved = [n for n in artifacts if not (reset_sampler and n.startswith("sampler_state_"))]
    assert all((destination / name).read_bytes() == before[name] for name in preserved)
    if reset_sampler:
        state = torch.load(destination / "sampler_state_rank_00000.pt", weights_only=False)
        assert sorted(state["cycles"][0]) == [0, 1, 2]
        assert state["cursors"] == [0]
        assert state["stage"] == pytest.approx(0.425)  # .2 + .6 * 15/40
        assert state["next_batch_id"] == 37
        assert state["queues"] == {}
        assert state["inflight"] == state["replay"] == state["ready_batches"] == []
        assert report["sampler_reset"]["ranks"][0]["discarded_pending_rows"] == 2
    validate_checkpoint_recipe(destination, asdict(recipe), architecture="inko")
    # The launcher must carry the explicit fork into the trainer command.
    from scripts.pretrain import launch

    monkeypatch.setattr(launch, "load_config", lambda *a, **kw: recipe)
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch",
            "--config",
            "input.toml",
            "--storage-root",
            str(tmp_path),
            "--stage",
            "s1",
            "--nproc_per_node",
            "1",
            "--resume",
            str(destination),
            "--new-experiment",
            "--dry_run",
        ],
    )
    launch.main()
    command = json.loads(capsys.readouterr().out)["command"]
    assert "--new-experiment" in command
    assert command[command.index("--resume") + 1] == str(destination)

    # A failed copy never publishes a partial destination or alters the source.
    failed = tmp_path / "failed" / source.name

    def broken_copy(*args):
        raise OSError("simulated copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(migration.shutil, "copy2", broken_copy)
        with pytest.raises(OSError, match="copy failure"):
            migration.migrate(
                source,
                failed,
                "input.toml",
                storage_root=tmp_path,
                reason="Import",
                source_run_id="continuing-run",
            )
    assert not failed.exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    # Relocation is not permission to change policy or the active plan.
    changed = copy.deepcopy(asdict(recipe))
    changed["data"]["caption_beta_end"] = 3
    with pytest.raises(ValueError):
        migration.check_recipe_change(
            old,
            changed,
            step=15,
            active_bucket=json.loads((source / "bucket_plan.json").read_text()),
        )
    old_prompt.write_text("different prompt suite")
    with pytest.raises(ValueError, match="byte-identical"):
        migration.check_recipe_change(
            old,
            asdict(recipe),
            step=15,
            active_bucket=json.loads((source / "bucket_plan.json").read_text()),
        )

    if archive_source_assets:
        import hashlib

        old_prompt.write_bytes(Path(recipe.eval.prompts_file).read_bytes())
        paths = [old_prompt] + [Path(s["bucket_plan"]) for s in old["stages"]]
        saved = {
            str(p): {
                "text": p.read_text(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        }
        archive = tmp_path / "source-assets.json"
        archive.write_text(json.dumps(saved))
        for p in paths:
            p.unlink()
        restored = tmp_path / "from-archive" / source.name
        with pytest.raises(FileNotFoundError):
            migration.migrate(
                source,
                restored,
                "input.toml",
                storage_root=tmp_path,
                reason="Import",
                source_run_id="continuing-run",
            )
        report = migration.migrate(
            source,
            restored,
            "input.toml",
            storage_root=tmp_path,
            reason="Import with archived config bytes",
            source_run_id="continuing-run",
            source_assets=archive,
            reset_sampler=reset_sampler,
        )
        assert report["source_assets"]["path"] == str(archive)
        assert all((restored / name).read_bytes() == before[name] for name in preserved)
        assert {p.name: p.read_bytes() for p in source.iterdir()} == before
        missing = copy.deepcopy(saved)
        del missing[str(old_prompt)]
        archive.write_text(json.dumps(missing))
        with pytest.raises(ValueError, match="record is missing"):
            migration.check_recipe_change(
                old,
                asdict(recipe),
                step=15,
                active_bucket=json.loads((source / "bucket_plan.json").read_text()),
                archived_assets=migration.load_source_assets(archive),
            )
        saved[str(old_prompt)]["text"] = "corrupted archive"
        archive.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match="hash mismatch"):
            migration.load_source_assets(archive)


def test_recipe_migration_locks_current_entries_but_allows_next_stage_at_boundary(
    recipe,
):
    old = asdict(recipe)
    new = copy.deepcopy(old)
    new["stages"][1]["datasets"] = [{"path": "/new/source", "weight": 3.0}]
    bucket = json.loads(Path(recipe.stages[0].bucket_plan).read_text())
    changes = migration.check_recipe_change(old, new, step=10, active_bucket=bucket)
    assert [(c["stage"], c["field"]) for c in changes] == [("s1", "datasets")]
    bucket = json.loads(Path(recipe.stages[1].bucket_plan).read_text())
    with pytest.raises(ValueError, match="same datasets"):
        migration.check_recipe_change(old, new, step=15, active_bucket=bucket)


def test_sampler_migration_partitions_filtered_rows_without_advancing_training_rng(tmp_path, recipe):
    import random

    from datasets import Dataset

    from src.dataset.sampler import BucketPlan, RowLengthQueueBatchSampler

    pool = tmp_path / "filtered"
    Dataset.from_dict({"captions": [["a"], ["b"], [], ["d"], ["e"], ["f"]]}).save_to_disk(str(pool))
    metadata = RowLengthMetadata(
        np.ones(6), np.array([0, 1, 2, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6]),
        metadata_info=prompt_metadata_contract(6),
    )
    metadata.save(pool / "length_metadata.npz")
    plan_path = tmp_path / "buckets.json"
    plan_path.write_text('{"1": [{"max_length": 8, "batch_size": 1}]}')
    stages = list(recipe.stages)
    stages[1] = replace(stages[1], datasets=[DatasetConfig(str(pool), 1.0)], bucket_plan=str(plan_path))
    recipe = replace(recipe, stages=stages)
    staging = tmp_path / "staging"
    staging.mkdir()
    for rank in range(2):
        torch.save({
            "version": 1, "cycles": [[1000]], "next_batch_id": 20 + rank,
            "queues": {}, "inflight": [], "replay": [], "ready_batches": [],
        }, staging / f"sampler_state_rank_{rank:05d}.pt")
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    migration.reset_sampler_files(staging, recipe, step=15, world_size=2)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for rank, expected_rows in enumerate(([0, 4], [1, 3, 5])):
        state = torch.load(staging / f"sampler_state_rank_{rank:05d}.pt", weights_only=False)
        assert sorted(state["cycles"][0]) == expected_rows
        assert state["stage"] == pytest.approx(0.425)
        sampler = RowLengthQueueBatchSampler(
            [metadata], BucketPlan({1: [(8, 1)]}), num_replicas=2, rank=rank,
            caption_policy=CaptionPolicy(-2, 2, 0.1, 4),
        )
        sampler.load_state_dict(state)
        stream = iter(sampler)
        batches = [next(stream) for _ in expected_rows]
        assert sorted(b[0].row_idx for b in batches) == expected_rows
        assert [b[0].batch_id for b in batches] == list(range(20 + rank, 20 + rank + len(expected_rows)))


def test_resume_rejects_old_model_identity_even_with_identical_capacity(tmp_path):
    recipe = {"model": {"width": 32}}
    (tmp_path / "run_config.json").write_text(json.dumps(recipe))
    (tmp_path / "transformer_config.json").write_text(
        json.dumps({"width": 32, "architecture": "old-identity"})
    )
    with pytest.raises(ValueError, match="model metadata differs"):
        validate_checkpoint_recipe(tmp_path, recipe, architecture="current-identity")


@pytest.mark.parametrize(
    "fault", [None, "nested", "per_sample", "missing", "nan", "fractional"]
)
def test_calibration_schema_and_whole_batch_units(fault, capsys):
    row = {
        "latent_hw": [8, 12],
        "txt_len": 5,
        "micro_batch": 8,
        "peak_mem_gb": 2.0,
        "ms_per_step": 16.0,
    }
    payload = [
        row,
        {"latent_hw": [8, 12], "txt_len": 5, "micro_batch": 16, "error": "oom"},
    ]
    if fault == "nested":
        payload = {"results": payload}
    elif fault == "per_sample":
        row["ms_per_sample"] = row.pop("ms_per_step")
    elif fault == "missing":
        del row["txt_len"]
    elif fault == "nan":
        row["ms_per_step"] = float("nan")
    elif fault == "fractional":
        row["micro_batch"] = 1.5
    if fault:
        with pytest.raises(ValueError):
            calibration_points(payload)
    else:
        points = calibration_points(payload)
        assert len(points) == 1
        assert points[0].img_tokens == 24
        assert points[0].ms_per_step == 16
        assert "excluded 1" in capsys.readouterr().out
