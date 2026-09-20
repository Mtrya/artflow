from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import datasets
import diffusers
import pytest
import torch

from src.evaluation import kid_eval


def test_shape_groups_accept_default_huggingface_list_format():
    dataset = datasets.Dataset.from_dict({"latents": [
        torch.zeros(16, h, w).tolist() for h, w in [(2, 3), (3, 2), (2, 3)]
    ]})
    assert isinstance(dataset[0]["latents"], list)
    assert kid_eval._group_by_shape(dataset, [0, 1, 2]) == [[0, 2], [1]]


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1), (8, 0), (8, 7)])
@pytest.mark.parametrize("missing_rank", [False, True])
@pytest.mark.parametrize("num_fake", [3, 11])
def test_kid_collects_full_sets_with_empty_shards(monkeypatch, tmp_path, world, rank, missing_rank, num_fake):
    dataset = datasets.Dataset.from_dict({
        "latents": [torch.zeros(16, 2 + i % 2, 3 - i % 2).tolist() for i in range(5)],
        "captions": [["short", f"caption {i}"] for i in range(5)],
    })
    monkeypatch.setattr(datasets, "load_from_disk", lambda _: dataset)
    vae = SimpleNamespace(decode=lambda value: SimpleNamespace(sample=value[:, :3]))
    vae.to = lambda _: vae
    monkeypatch.setattr(diffusers.AutoencoderKLQwenImage, "from_pretrained", lambda *a, **k: vae)
    monkeypatch.setattr(kid_eval, "get_vae_stats", lambda *a, **k: (torch.tensor(0.), torch.tensor(1.)))
    monkeypatch.setattr(kid_eval, "encode_text", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(kid_eval, "aspect_resize_crop",
                        lambda images, _: torch.nn.functional.adaptive_avg_pool2d(images, (2, 2)))
    sample = Mock(side_effect=lambda fn, noise, **kwargs: noise)
    monkeypatch.setattr(kid_eval, "sample_ode", sample)
    metric = Mock(return_value=(0.25, 0.01))
    monkeypatch.setattr(kid_eval, "calculate_kid", metric)
    calls = []

    def gather(parts):
        count = 5 if not calls else num_fake
        assert isinstance(parts, list) and len(parts) == 1
        assert len(parts[0]) == len(range(count)[rank::world])
        calls.append(count)
        result = [torch.zeros(len(range(count)[r::world]), 3, 2, 2, dtype=torch.uint8)
                  for r in range(world)]
        if missing_rank:
            result[0] = result[0][:-1]
        return result

    monkeypatch.setattr(kid_eval, "gather_object", gather)
    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=world,
                                  process_index=rank, is_main_process=rank == 0,
                                  autocast=nullcontext, log=Mock())
    # Deliberately no gather_object method: the real Accelerator has none.
    model = torch.nn.Linear(1, 1)
    if missing_rank and rank == 0:
        with pytest.raises(ValueError, match="complete real/fake"):
            kid_eval.run_kid_eval(accelerator, model, "unused", str(tmp_path), 66,
                                 None, None, False, num_fake=num_fake, batch_size=2, ode_steps=17)
        metric.assert_not_called()
        return
    metrics = kid_eval.run_kid_eval(accelerator, model, "unused", str(tmp_path), 66,
                                   None, None, False, num_fake=num_fake, batch_size=2, ode_steps=17)
    assert calls == [5, num_fake]
    assert model.training
    assert all(call.kwargs["steps"] == 17 for call in sample.call_args_list)
    # Shape regrouping and rank sharding must preserve unique global fake IDs:
    # repeated conditioning rows receive distinct noise, never duplicated fakes.
    observed = [tuple(noise.flatten().tolist())
                for call in sample.call_args_list for noise in call.args[1]]
    expected = []
    for fake_id in range(num_fake)[rank::world]:
        shape = torch.as_tensor(dataset[fake_id % len(dataset)]["latents"]).shape
        noise = torch.randn(tuple(shape), generator=torch.Generator().manual_seed(123 + fake_id))
        expected.append(tuple(noise.to(torch.bfloat16).flatten().tolist()))
    assert sorted(observed) == sorted(expected)
    if rank == 0:
        assert metrics["kid/num_real"] == 5 and metrics["kid/num_fake"] == num_fake
        assert len(metric.call_args.args[0]) == 5 and len(metric.call_args.args[1]) == num_fake
    else:
        assert metrics == {}
        metric.assert_not_called()
