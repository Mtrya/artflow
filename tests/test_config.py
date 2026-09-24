"""A full curriculum is explicit, strict, and independent of cwd/environment."""

import copy
from dataclasses import asdict, fields
from pathlib import Path
import tomllib

import pytest

from src.pretrain.config import TrainConfig, _decode, _validate, flatten, load_config

RECIPE = Path(__file__).resolve().parents[1] / "configs/hero.toml"


def test_full_curriculum_has_shared_schedule_and_stage_inputs():
    config = load_config(RECIPE)
    assert config.max_steps == 600000
    for name, start, end in [
        ("256p", 0, 450000),
        ("640p", 450000, 570000),
        ("896p", 570000, 600000),
    ]:
        flat = flatten(config, name)
        assert (flat["stage_start"], flat["stop_at_step"], flat["max_steps"]) == (
            start,
            end,
            600000,
        )
        assert flat["muon_lr"] == 0.02 and flat["muon_wd"] == 0.0015
        assert flat["learning_rate"] == 0.0001 and flat["lr_warmup_steps"] == 20000
        assert name in flat["eval_dataset_path"] and name in flat["bucket_plan"]
    assert all(
        type(v) is not bool
        for section in asdict(config).values()
        if isinstance(section, dict)
        for v in section.values()
    )


def test_no_layers_or_partial_files(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_config([RECIPE, RECIPE])
    path = tmp_path / "partial.toml"
    path.write_text("[optim]\nmuon_lr = 0.02\n")
    with pytest.raises(ValueError, match="missing required"):
        load_config(path)


def _field_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield prefix + (key,)
            yield from _field_paths(item, prefix + (key,))
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        yield from _field_paths(value[0], prefix + (0,))


RAW = tomllib.loads(RECIPE.read_text())


@pytest.mark.parametrize("path", list(_field_paths(RAW)))
def test_every_recipe_field_is_required(path):
    payload = copy.deepcopy(RAW)
    parent = payload
    for part in path[:-1]:
        parent = parent[part]
    del parent[path[-1]]
    with pytest.raises(ValueError, match="missing required"):
        _decode(TrainConfig, payload, "recipe")


@pytest.mark.parametrize(
    "table,key,value",
    [
        ("model", "branch_norm", True),
        ("optim", "muon_lr", True),
        ("optim", "muon_lr", float("inf")),
        ("model", "hidden_size", 1153),
        ("train", "checkpoint_keep_last", -1),
        ("train", "ema_decay", 1.0),
        ("data", "caption_dropout_prob", 1.1),
        ("telemetry", "stability_interval", -1),
    ],
)
def test_unknown_types_and_invalid_values_rejected(table, key, value):
    payload = copy.deepcopy(RAW)
    payload[table][key] = value
    with pytest.raises(ValueError):
        _validate(_decode(TrainConfig, payload, "recipe"))


def test_duplicate_or_reversed_stages_rejected():
    for mutate in (
        lambda p: p["stages"].reverse(),
        lambda p: p["stages"][1].update(name="256p"),
        lambda p: p["stages"][1].update(grid_steps=[1]),
        lambda p: p["stages"][0].update(gradient_accumulation_steps=0),
    ):
        payload = copy.deepcopy(RAW)
        mutate(payload)
        with pytest.raises(ValueError):
            _validate(_decode(TrainConfig, payload, "recipe"))


def test_paths_ignore_cwd_and_hyperparameter_environment(tmp_path, monkeypatch):
    expected = load_config(RECIPE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUON_LR", "123")
    monkeypatch.setenv("ARTFLOW_ROOT", "/wrong")
    assert load_config(RECIPE) == expected
    assert Path(expected.paths.storage_root).is_absolute()


def test_cli_rejects_architecture_and_optimizer_overrides():
    from src.pretrain.train import parse_args

    for argument in ("--branch_norm", "--muon_lr", "--compile", "--run_name"):
        with pytest.raises(SystemExit):
            parse_args().parse_args(
                ["--config", str(RECIPE), "--stage", "256p", argument, "1"]
            )


def test_cli_rejects_repeated_config_files():
    from src.pretrain.train import parse_args

    with pytest.raises(SystemExit):
        parse_args().parse_args(
            ["--config", str(RECIPE), "--config", str(RECIPE), "--stage", "256p"]
        )


def test_frozen_dataset_and_shard_weights_are_preserved():
    config = load_config(RECIPE)
    assert [len(stage.datasets) for stage in config.stages] == [13, 17, 14]
    weights = {
        Path(item.path).name: item.weight
        for stage in config.stages
        for item in stage.datasets
    }
    assert weights["d4-relaion@256p"] == 33.3715
    assert weights["d3-people-a@640p"] == 5.876955
    assert weights["d3-people-b@640p"] == 5.883045
    assert weights["d4-relaion-p4@896p"] == 3.229245


def test_dataset_paths_with_spaces_are_literal(tmp_path):
    import json
    from src.dataset.mix import parse_dataset_mix

    path = tmp_path / "run.toml"
    storage = tmp_path / "space and $literal"
    path.write_text(
        RECIPE.read_text().replace(
            'storage_root = ".."', "storage_root = " + json.dumps(str(storage))
        )
    )
    config = load_config(path)
    entries = parse_dataset_mix(flatten(config, "256p")["dataset_mix"])
    assert (
        len(entries) == 13 and entries[0].path.parent == storage / "precomputed_dataset"
    )
