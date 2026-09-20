import tomllib

import pytest

from scripts.bench.render_hero_stage import render


@pytest.mark.parametrize("stage,count", [("256p", 13), ("640p", 17), ("896p", 14)])
def test_frozen_input_templates_are_root_portable_without_selecting_T(stage, count):
    config = tomllib.loads(render(stage, "/artifact-root"))
    assert len(config["data"]["mix"].split()) == count
    assert config["eval"]["dataset_path"] == f"/artifact-root/precomputed_dataset/light-eval@{stage}"
    assert config["data"]["bucket_plan"].endswith(f"hero-{stage}-k20.json")
    assert "train" not in config


def test_path_substitution_is_literal_and_toml_safe():
    root = '/artifacts/"quotes"/\\/$UNEXPANDED'
    config = tomllib.loads(render("256p", root))
    assert config["paths"]["vae"] == root + "/models/e2e-qwenimage-vae"
    with pytest.raises(ValueError, match="absolute"):
        render("256p", "relative")
    with pytest.raises(ValueError, match="whitespace"):
        render("256p", "/has space")


def test_preserves_measured_shard_weights():
    mid = tomllib.loads(render("640p", "/artifacts"))["data"]["mix"]
    late = tomllib.loads(render("896p", "/artifacts"))["data"]["mix"]
    assert "d3-people-a@640p:5.876955" in mid
    assert "d3-people-b@640p:5.883045" in mid
    assert "d4-relaion-p4@896p:3.229245" in late
