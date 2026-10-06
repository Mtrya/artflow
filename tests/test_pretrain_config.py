"""Path ownership, recipe-derived policy, and deliberate recovery migration."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from scripts.pretrain import migrate_checkpoint as migration
from scripts.pretrain.plan_buckets import calibration_points, load_sidecar_lengths
from src.dataset.captions import CaptionPolicy
from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.mix import DatasetEntry
from src.pretrain.config import REPOSITORY_ROOT, load_config, stage_caption_policy
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


def test_migration_rebrands_metadata_preserves_training_state_and_records_origin(
    tmp_path, recipe, monkeypatch, capsys
):
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
    migration.migrate(
        source,
        destination,
        "input.toml",
        storage_root=tmp_path,
        reason="Relocate tracked configuration",
        source_run_id="continuing-run",
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
    assert all((destination / name).read_bytes() == before[name] for name in artifacts)
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
