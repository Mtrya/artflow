import contextlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import evaluation  # noqa: E402


class DummyAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0
        self.use_distributed = False
        self.logged = []

    def log(self, payload, step):
        self.logged.append((payload, step))

    @contextlib.contextmanager
    def autocast(self):
        yield

    def gather_object(self, obj):
        return [obj]

    def broadcast_object_list(self, obj_list):
        return obj_list


class DummyModel(nn.Module):
    def forward(self, latents, timesteps, txt, txt_pooled, txt_mask):
        return torch.zeros_like(latents)


class DummyDataset:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def select(self, indices):
        selected = [self._items[i] for i in indices]
        return DummyDataset(selected)

    def __iter__(self):
        return iter(self._items)


def _install_encode_text_stub(monkeypatch):
    module = types.ModuleType("encode_text")

    def _encode(prompts, text_encoder, processor, pooling):
        batch = len(prompts)
        seq = 2
        hidden = 4
        txt = torch.zeros(batch, seq, hidden)
        txt_mask = torch.ones(batch, seq, dtype=torch.long)
        pooled = torch.zeros(batch, hidden) if pooling else None
        return txt, txt_mask, pooled

    module.encode_text = _encode
    monkeypatch.setitem(sys.modules, "encode_text", module)


def _install_diffusers_stub(monkeypatch):
    module = types.ModuleType("diffusers")

    class _Autoencoder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            self.device = device
            return self

        def decode(self, latents):
            tensor = latents.to(torch.float32).squeeze(2)
            upsampled = torch.nn.functional.interpolate(
                tensor, scale_factor=8, mode="nearest"
            )
            base = upsampled.mean(dim=1, keepdim=True)
            sample = base.unsqueeze(2).repeat(1, 3, 1, 1, 1)
            return types.SimpleNamespace(sample=sample)

    module.AutoencoderKLQwenImage = _Autoencoder
    monkeypatch.setitem(sys.modules, "diffusers", module)


def _install_swanlab_stub(monkeypatch):
    class _Image:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(evaluation.swanlab, "Image", _Image)


def _install_flow_stub(monkeypatch, sample_history):
    flow_module = types.ModuleType("flow")
    solvers_module = types.ModuleType("flow.solvers")

    def _sample_ode(model_fn, sample_z0, steps, t_start, t_end):
        sample_history.append(
            (
                sample_z0.shape[0],
                sample_z0.shape[-2] * 8,
                sample_z0.shape[-1] * 8,
            )
        )
        return torch.zeros_like(sample_z0)

    solvers_module.sample_ode = _sample_ode
    flow_module.solvers = solvers_module
    monkeypatch.setitem(sys.modules, "flow", flow_module)
    monkeypatch.setitem(sys.modules, "flow.solvers", solvers_module)
    return solvers_module


def _build_dataset_items():
    bucket_resolutions = {
        1: (256, 256),
        2: (336, 192),
        3: (192, 336),
        4: (288, 224),
        5: (224, 288),
    }
    items = []
    for bucket_id, (h, w) in bucket_resolutions.items():
        repeats = 3 if bucket_id == 1 else 1
        for idx in range(repeats):
            latents = torch.full((16, h // 8, w // 8), float(bucket_id + idx))
            items.append(
                {
                    "latents": latents,
                    "captions": [f"prompt-{bucket_id}-{idx}"],
                    "resolution_bucket_id": bucket_id,
                }
            )
    return items, bucket_resolutions


def test_run_evaluation_light_handles_resolution_buckets(tmp_path, monkeypatch):
    _install_encode_text_stub(monkeypatch)
    _install_diffusers_stub(monkeypatch)
    _install_swanlab_stub(monkeypatch)

    items, bucket_resolutions = _build_dataset_items()
    dataset = DummyDataset(items)

    import datasets as datasets_module

    monkeypatch.setattr(datasets_module, "load_from_disk", lambda path: dataset)

    sample_history = []
    _install_flow_stub(monkeypatch, sample_history)

    fid_calls = {}
    kid_calls = {}
    clip_calls = {}

    def _fid(real, fake, device=None):
        fid_calls["shapes"] = (tuple(real.shape), tuple(fake.shape))
        assert real.shape[-2:] == (256, 256)
        assert fake.shape[-2:] == (256, 256)
        return 0.1

    def _kid(real, fake, device=None):
        kid_calls["shapes"] = (tuple(real.shape), tuple(fake.shape))
        return 0.2

    def _clip(images, prompts, device=None):
        clip_calls["count"] = (images.shape[0], len(prompts))
        assert images.shape[0] == len(prompts)
        return 0.3

    monkeypatch.setattr(evaluation, "calculate_fid", _fid)
    monkeypatch.setattr(evaluation, "calculate_kid", _kid)
    monkeypatch.setattr(evaluation, "calculate_clip_score", _clip)

    accelerator = DummyAccelerator()
    model = DummyModel()
    save_dir = tmp_path / "grids"

    metrics = evaluation.run_evaluation_light(
        accelerator=accelerator,
        model=model,
        vae_path="dummy",
        save_path=str(save_dir),
        current_step=1200,
        text_encoder=object(),
        processor=object(),
        pooling=True,
        dataset_path="ignored",
        num_samples=16,
        batch_size=2,
    )

    assert metrics == {"fid": 0.1, "kid": 0.2, "clip_score": 0.3}
    assert fid_calls["shapes"][0][-2:] == (256, 256)
    assert kid_calls["shapes"][1][-2:] == (256, 256)
    assert clip_calls["count"][0] == clip_calls["count"][1]

    expected_calls = [
        (2, 256, 256),
        (1, 256, 256),
        (1, 336, 192),
        (1, 192, 336),
        (1, 288, 224),
        (1, 224, 288),
    ]
    assert sample_history == expected_calls

    step_tag = round(1200 / 1000)
    for bucket_id, (h, w) in bucket_resolutions.items():
        grid_path = Path(save_dir) / f"samples_step_{step_tag}k_bucket{bucket_id}_{h}x{w}.png"
        assert grid_path.exists()

    assert accelerator.logged, "Expected accelerator.log to be called"
