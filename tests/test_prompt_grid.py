"""Prompt grids must include every rank and retain suite order."""

from contextlib import nullcontext
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import diffusers
import pytest
import torch

from src.evaluation import prompt_grid


@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1), (8, 0), (8, 7)])
def test_grid_collects_all_prompts_in_suite_order(monkeypatch, tmp_path, world_size, rank):
    prompts = [
        {"id": str(i), "text": f"prompt {i}", "_bucket": i % 2}
        for i in range(5)
    ]
    shapes = {0: (2, 2), 1: (2, 2)}

    def images_for(records):
        return [(p, torch.full((3, 2, 2), int(p["id"]) / 5)) for p in records]

    local = prompts[rank::world_size]
    sampler = Mock(return_value=images_for(local))

    def gather(results):
        assert [p["id"] for p, _ in results] == [p["id"] for p in local]
        return [
            pair
            for other_rank in range(world_size)
            for pair in images_for(prompts[other_rank::world_size])
        ]

    collector = Mock(side_effect=gather)
    monkeypatch.setattr(prompt_grid, "gather_object", collector)
    monkeypatch.setattr(prompt_grid, "load_prompt_plan", lambda *args: (prompts, shapes))
    monkeypatch.setattr(prompt_grid, "sample_prompt_images", sampler)
    monkeypatch.setattr(
        diffusers.AutoencoderKLQwenImage, "from_pretrained", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        prompt_grid, "get_vae_stats", lambda *args, **kwargs: (torch.tensor(0.), torch.tensor(1.))
    )
    grid_writer = Mock()
    monkeypatch.setattr(prompt_grid, "make_image_grid", grid_writer)
    monkeypatch.setattr(prompt_grid.swanlab, "Image", Mock())
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), num_processes=world_size, process_index=rank,
        is_main_process=rank == 0, autocast=nullcontext, log=Mock(),
    )
    model = torch.nn.Linear(1, 1)
    prompt_grid.run_prompt_grid_eval(
        accelerator, model, "unused", str(tmp_path), 42, None, None, False
    )

    assert sampler.call_args.args[6] == local
    collector.assert_called_once()
    assert model.training
    if rank != 0:
        grid_writer.assert_not_called()
        accelerator.log.assert_not_called()
        return

    assert grid_writer.call_count == 2
    for call, bucket in zip(grid_writer.call_args_list, (0, 1)):
        expected = torch.stack([image for _, image in images_for(prompts[bucket::2])])
        torch.testing.assert_close(call.args[0], expected)
    assert accelerator.log.call_count == 2
    assert all(call.kwargs["step"] == 42 for call in accelerator.log.call_args_list)


def test_seed_override_and_legacy_fallback():
    assert prompt_grid.resolved_prompt_seed({"id": "old"}) == prompt_grid.prompt_seed("old")
    assert prompt_grid.resolved_prompt_seed({"id": "zh", "seed": 17}) == 17
    assert prompt_grid.resolved_prompt_seed({"id": "en", "seed": 17}) == 17


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "17"])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        prompt_grid.resolved_prompt_seed({"id": "x", "seed": seed})


def test_explicit_seed_controls_sample_noise(monkeypatch):
    prompts = [{"id": name, "seed": seed, "text": name, "_bucket": 0}
               for name, seed in [("zh", 17), ("en", 17), ("other", 18)]]
    monkeypatch.setattr(prompt_grid, "encode_text", lambda *a, **k: (None, None, None))
    noises = []

    def sample(model_fn, noise, **kwargs):
        noises.append(noise.clone())
        return noise

    monkeypatch.setattr(prompt_grid, "sample_ode", sample)
    vae = SimpleNamespace(decode=lambda x: SimpleNamespace(sample=x[:, :3]))
    prompt_grid.sample_prompt_images(
        None, vae, torch.tensor(0.), torch.tensor(1.), None, None,
        prompts, {0: (2, 2)}, batch_size=2, ode_steps=50, pooling=False,
        device=torch.device("cpu"),
    )
    noise = torch.cat(noises)
    torch.testing.assert_close(noise[0], noise[1], rtol=0, atol=0)
    assert not torch.equal(noise[0], noise[2])


def test_frozen_monitor_panel():
    root = Path(__file__).resolve().parents[1]
    prompts = prompt_grid.load_prompt_suite(str(root / "assets/eval/hero_monitor_v1.jsonl"))
    assert len(prompts) == 48
    assert len({p["scene_id"] for p in prompts}) == 12
    assert len({p["seed"] for p in prompts}) == 12
    assert Counter(p["category"] for p in prompts) == dict.fromkeys(
        ["figures", "hand_object", "faces", "architecture", "layout", "style"], 8
    )
    assert len({p["style"] for p in prompts}) == 6
    assert Counter(p["length_band"] for p in prompts) == {
        "short": 24, "256-512": 8, "512-1024": 8, "1024-2048": 8,
    }
    for p in prompts:
        lo, hi = (1, 255) if p["variant"] == "short" else map(int, p["length_band"].split("-"))
        assert lo <= p["retained_tokens"] <= hi


@pytest.mark.parametrize("fault", ["duplicate", "incomplete", "seed", "aspect", "scene_id"])
def test_malformed_scene_rejected(tmp_path, fault):
    prompts = [{"id": f"{lang}_{variant}", "scene_id": "scene", "seed": 17,
                "lang": lang, "variant": variant, "text": "x"}
               for variant in ("short", "long") for lang in ("zh", "en")]
    if fault == "duplicate":
        prompts[1]["id"] = prompts[0]["id"]
    elif fault == "incomplete":
        prompts.pop()
    elif fault == "scene_id":
        prompts[0]["scene_id"] = "../escape"
    else:
        prompts[0][fault] = 18 if fault == "seed" else "3:4"
    path = tmp_path / "suite.jsonl"
    path.write_text("\n".join(map(json.dumps, prompts)))
    with pytest.raises(ValueError):
        prompt_grid.load_prompt_suite(str(path))


def test_scene_grid_layout_and_manifest(monkeypatch, tmp_path):
    # Deliberately shuffled input and gathered output must still render zh/en
    # columns with short/long rows.
    prompts = [{"id": str(i), "scene_id": "tea", "seed": 17, "_bucket": 0,
                "text": "full prompt " * 50, "lang": lang, "variant": variant}
               for i, (lang, variant) in enumerate(
                   [("en", "long"), ("zh", "short"), ("zh", "long"), ("en", "short")])]
    monkeypatch.setattr(prompt_grid, "load_prompt_plan", lambda *a: (prompts, {0: (2, 2)}))
    pairs = [(p, torch.full((3, 2, 2), int(p["id"]) / 4)) for p in prompts]
    monkeypatch.setattr(prompt_grid, "sample_prompt_images", Mock(return_value=pairs))
    monkeypatch.setattr(prompt_grid, "gather_object", lambda x: x[::-1])
    monkeypatch.setattr(diffusers.AutoencoderKLQwenImage, "from_pretrained", Mock(return_value=Mock()))
    monkeypatch.setattr(prompt_grid, "get_vae_stats", lambda *a, **k: (torch.tensor(0.), torch.tensor(1.)))
    writer, media = Mock(), Mock()
    monkeypatch.setattr(prompt_grid, "make_image_grid", writer)
    monkeypatch.setattr(prompt_grid.swanlab, "Image", media)
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True,
                                  autocast=nullcontext, log=Mock())
    prompt_grid.run_prompt_grid_eval(
        accelerator, torch.nn.Linear(1, 1), "unused", str(tmp_path), 300000,
        None, None, False, weights="ema",
    )
    assert writer.call_args.kwargs["cols"] == 2
    torch.testing.assert_close(writer.call_args.args[0], torch.stack([pairs[i][1] for i in [1, 3, 2, 0]]))
    assert list(accelerator.log.call_args.args[0]) == ["grid/tea_bucket0_16x16"]
    assert "zh / short / seed=17" in media.call_args.kwargs["caption"]
    record = json.loads((tmp_path / "samples/panel_step_300000.json").read_text())
    assert (record["step"], record["ode_steps"], record["cfg_scale"], record["weights"]) == (300000, 50, 1.0, "ema")
    assert record["precision"] == "bf16"
    assert record["solver_precision"] == record["timestep_precision"] == "fp32"
    assert record["prompts"][0]["text"] == prompts[0]["text"]


@pytest.mark.parametrize("total", [400000, 420000])
def test_absolute_transition_triggers(total):
    for start, end in [(total * 75 // 100, total * 95 // 100), (total * 95 // 100, total)]:
        extra = [start, start + 2000, end]
        for step in extra:
            assert prompt_grid.grid_due(step, 10000, extra)
        assert not prompt_grid.grid_due(start + 2001, 10000, extra)
        assert prompt_grid.grid_due(10000, 10000, extra)
        # Resuming 1000 updates after the boundary doesn't move the +2k check.
        assert start + 2000 in [s for s in range(start + 1001, start + 2001)
                                if prompt_grid.grid_due(s, 10000, extra)]
    assert not prompt_grid.grid_due(0, 10000, [])
    assert prompt_grid.grid_due(0, 0, [0])
    assert not prompt_grid.grid_due(10000, 0, [])




def test_training_grid_wiring_preserves_rng_and_has_preloop_baseline():
    import ast
    import inspect
    from src.pretrain import train

    tree = ast.parse(inspect.getsource(train))
    helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "evaluate_grid")
    assert any(isinstance(n, ast.With) and "torch.random.fork_rng" in ast.unparse(n.items[0].context_expr)
               for n in ast.walk(helper))
    startup = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "global_step in args.grid_steps")
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                and ast.unparse(n.test) == "global_step < end_step")
    assert startup.lineno < loop.lineno
    assert "evaluate_grid()" in ast.unparse(startup)
    assert "grid_due(global_step, args.eval_interval, args.grid_steps)" in ast.unparse(loop)
