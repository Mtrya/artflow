"""Tests for the fixed eval-loss probe (caption bands, weighting, determinism)."""

import re

import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk

from src.dataset.length_metadata import RowLengthMetadata, sidecar_path
from src.evaluation import eval_loss as eval_loss_module
from src.evaluation.eval_loss import LENGTH_BINS, EvalLossProbe
from src.flow.paths import shift_timesteps
from src.models.artflow import ArtFlow
from src.utils.prompt_contract import DROP_IDX

CHANNELS = 16
LEGACY_KEYS = {
    "eval/loss",
    "eval/loss_t015",
    "eval/loss_t040",
    "eval/loss_t065",
    "eval/loss_t090",
}


def _save_dataset(tmp_path, rows, name="tiny_eval"):
    """Save rows of ``(caption texts, latent HxW)``; latents identify their row."""
    data = {"latents": [], "captions": [], "resolution_bucket_id": []}
    for index, (captions, shape) in enumerate(rows):
        data["latents"].append(
            torch.full((CHANNELS, shape[0], shape[1]), float(index)).numpy()
        )
        data["captions"].append(list(captions))
        data["resolution_bucket_id"].append(1)
    path = str(tmp_path / name)
    Dataset.from_dict(data).save_to_disk(path)
    return path


def _save_sidecar(dataset_path, lengths_per_row):
    """Write the retained-length sidecar for rows whose captions are known."""
    offsets = [0]
    for lengths in lengths_per_row:
        offsets.append(offsets[-1] + len(lengths))
    RowLengthMetadata(
        resolution_ids=np.zeros(len(lengths_per_row), dtype=np.int64),
        caption_offsets=np.array(offsets, dtype=np.int64),
        prompt_lengths=np.array(
            [length for row in lengths_per_row for length in row], dtype=np.int64
        ),
    ).save(sidecar_path(dataset_path))


class _LengthTokenizer:
    """Tokenizer stand-in whose retained length is the number in the caption."""

    def __call__(self, prompts, truncation=True, max_length=None, padding=False):
        input_ids = []
        for prompt in prompts:
            retained = int(re.search(r"retained_(\d+)", prompt).group(1))
            input_ids.append(list(range(retained + DROP_IDX)))
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }


class _FakeEncoder:
    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])


def _probe_kwargs(path, **overrides):
    kwargs = dict(
        dataset_path=path,
        text_encoder=_FakeEncoder(),
        tokenizer=None,
        pooling=False,
        exit_layer=None,
        vae_mean=torch.zeros(CHANNELS, 1, 1),
        vae_std=torch.ones(CHANNELS, 1, 1),
        num_samples=8,
        batch_size=4,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return kwargs


class _ZeroModel(torch.nn.Module):
    """Zero velocity, so a sample's loss is the energy of its own target."""

    def forward(self, z_t, t, txt=None, txt_pooled=None, txt_mask=None):
        return torch.zeros_like(z_t)


class _RecordingModel(_ZeroModel):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, z_t, t, txt=None, txt_pooled=None, txt_mask=None):
        self.calls.append((t.detach().clone(), z_t.detach().clone().float()))
        return super().forward(
            z_t, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask
        )


def _fake_encode_text(texts, model, tokenizer, pooling, exit_layer=None):
    b = len(texts)
    emb = torch.zeros(b, 5, 1024)
    mask = torch.ones(b, 5, dtype=torch.long)
    pooled = torch.zeros(b, 1024) if pooling else None
    return emb, mask, pooled


def _model_inputs(probe, index, t_val):
    """The (z_t, t) the model is called with for one sample at one timestep."""
    z1 = probe.latents[index].unsqueeze(0).to(torch.bfloat16)
    z0 = probe.noise[index].unsqueeze(0).to(torch.bfloat16)
    t = shift_timesteps(torch.full((1,), t_val), z1)
    return (1.0 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z1, t


def _calls_matching(calls, expected_z_t, expected_t):
    matched = 0
    for t_values, z_t in calls:
        for row_index in range(z_t.shape[0]):
            if torch.equal(t_values[row_index], expected_t[0]) and torch.allclose(
                z_t[row_index], expected_z_t[0], rtol=0.0, atol=1e-6
            ):
                matched += 1
    return matched


@pytest.fixture(autouse=True)
def _fake_text_encoder(monkeypatch):
    monkeypatch.setattr(eval_loss_module, "encode_text", _fake_encode_text)


def _set_sample_losses(probe, scales):
    """Force per-sample losses to ``scale ** 2`` (exact in bf16 and float32)."""
    for index, scale in enumerate(scales):
        shape = probe.latents[index].shape
        probe.latents[index] = torch.zeros(shape)
        probe.noise[index] = torch.full(shape, float(scale))


# ---------------------------------------------------------------------------
# Caption bands
# ---------------------------------------------------------------------------

BAND_ROWS = [
    (["row0_short", "row0_long", "row0_over"], (32, 32)),
    (["row1_mid"], (32, 32)),
    (["row2_short", "row2_mid"], (32, 16)),
    (["row3_long"], (32, 16)),
    (["row4_short", "row4_also_short"], (32, 32)),
    (["row5_band512"], (32, 32)),
]
BAND_LENGTHS = {
    "row0_short": 100,
    "row0_long": 900,
    "row0_over": 1300,
    "row1_mid": 300,
    "row2_short": 200,
    "row2_mid": 400,
    "row3_long": 1000,
    "row4_short": 150,
    "row4_also_short": 150,
    "row5_band512": 600,
}
BAND_LENGTHS_PER_ROW = [[100, 900, 1300], [300], [200, 400], [1000], [150, 150], [600]]


def _band_dataset(tmp_path, name="bands"):
    path = _save_dataset(tmp_path, BAND_ROWS, name=name)
    _save_sidecar(path, BAND_LENGTHS_PER_ROW)
    return path


def test_bands_pick_captions_of_their_own_length(tmp_path):
    probe = EvalLossProbe(**_probe_kwargs(_band_dataset(tmp_path), num_samples=4))

    assert probe.banded
    assert probe.band_sample_counts == {
        "le128": 1,
        "129_256": 2,
        "257_512": 2,
        "513_1024": 3,
        "1025_2048": 1,
    }
    for name, low, high in LENGTH_BINS:
        chosen = [c for c, band in zip(probe.captions, probe.bands) if band == name]
        assert len(chosen) == probe.band_sample_counts[name]
        assert chosen
        assert all(low <= BAND_LENGTHS[caption] <= high for caption in chosen)

    # A caption the shorter bands did not name lands in the extended last one.
    assert "row0_over" in probe.captions
    # Two captions of one row inside one band: the earliest one is used.
    assert "row4_short" in probe.captions
    assert "row4_also_short" not in probe.captions


def test_sidecar_written_by_the_builder_is_read(tmp_path):
    """The probe reads the CSR layout the offline builder actually writes."""
    rows = [
        (["retained_100", "retained_900"], (32, 32)),
        (["retained_300"], (32, 32)),
        (["retained_600"], (32, 32)),
    ]
    path = _save_dataset(tmp_path, rows)
    metadata = RowLengthMetadata.from_hf_dataset(
        load_from_disk(path), _LengthTokenizer()
    )
    metadata.save(sidecar_path(path))

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=2))

    assert probe.banded
    assert probe.band_sample_counts == {
        "le128": 1,
        "129_256": 0,
        "257_512": 1,
        "513_1024": 2,
        "1025_2048": 0,
    }
    assert sorted(probe.captions) == [
        "retained_100",
        "retained_300",
        "retained_600",
        "retained_900",
    ]
    assert probe.paired_images == 1


def test_each_band_is_filled_to_the_requested_size(tmp_path):
    path = _band_dataset(tmp_path)
    full = EvalLossProbe(**_probe_kwargs(path, num_samples=4))
    assert full.band_sample_counts == {
        "le128": 1,
        "129_256": 2,
        "257_512": 2,
        "513_1024": 3,
        "1025_2048": 1,
    }
    # Shrinking the budget cuts every band to it where the data allows.
    smaller = EvalLossProbe(**_probe_kwargs(path, num_samples=2))
    assert smaller.band_sample_counts == {
        "le128": 1,
        "129_256": 2,
        "257_512": 2,
        "513_1024": 2,
        "1025_2048": 1,
    }
    assert len(smaller.latents) == 8


def test_the_length_cap_is_inside_the_last_band(tmp_path):
    # The last band runs to the 2048-token cap, which is also the largest
    # retained length the sidecar accepts, so no caption a banded probe holds
    # can fall outside every band.
    rows = [([f"row{i}"], (32, 32)) for i in range(2)]
    path = _save_dataset(tmp_path, rows)
    _save_sidecar(path, [[2048], [100]])

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=2))

    assert probe.banded
    assert set(probe.bands) == {"1025_2048", "le128"}
    assert probe.band_sample_counts["1025_2048"] == 1
    assert probe.band_sample_counts["le128"] == 1


def test_band_shortfall_is_reported(tmp_path):
    probe = EvalLossProbe(**_probe_kwargs(_band_dataset(tmp_path), num_samples=4))
    metrics = probe.evaluate(_ZeroModel())

    assert metrics["eval/samples/band_le128"] == 1.0
    assert metrics["eval/shortfall/band_le128"] == 3.0
    assert metrics["eval/samples/band_513_1024"] == 3.0
    assert metrics["eval/shortfall/band_513_1024"] == 1.0
    assert metrics["eval/samples/band_1025_2048"] == 1.0
    assert metrics["eval/shortfall/band_1025_2048"] == 3.0
    assert "eval/loss/band_le128" in metrics


def test_band_without_candidates_reports_the_gap_not_a_number(tmp_path):
    path = _save_dataset(
        tmp_path, [([f"row{i}"], (32, 32)) for i in range(3)], name="short_only"
    )
    _save_sidecar(path, [[100], [100], [100]])
    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=5))
    metrics = probe.evaluate(_ZeroModel())

    assert metrics["eval/samples/band_513_1024"] == 0.0
    assert metrics["eval/shortfall/band_513_1024"] == 5.0
    assert "eval/loss/band_513_1024" not in metrics
    # The band that does have samples carries the whole aggregate.
    assert metrics["eval/loss"] == pytest.approx(metrics["eval/loss/band_le128"])


# ---------------------------------------------------------------------------
# Sample-count weighting
# ---------------------------------------------------------------------------


def test_loss_is_weighted_by_sample_count(tmp_path):
    # Four rows of one latent shape and three of another: the shape groups
    # become batches of 4 and 3 samples, and each batch mixes caption bands.
    lengths = [[100], [100], [300], [300], [100], [300], [600]]
    rows = [
        ([f"row{i}"], (32, 32) if i < 4 else (32, 16)) for i in range(len(lengths))
    ]
    path = _save_dataset(tmp_path, rows)
    _save_sidecar(path, lengths)

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=8))
    scales = list(range(1, len(probe.latents) + 1))
    _set_sample_losses(probe, scales)
    losses = [float(scale) ** 2 for scale in scales]

    metrics = probe.evaluate(_ZeroModel())
    weighted = sum(losses) / len(losses)
    assert metrics["eval/loss"] == pytest.approx(weighted)

    # Averaging the batch means instead would give a different number, which is
    # what the probe used to report.
    batch_means = []
    for group in probe.groups:
        for start in range(0, len(group), probe.batch_size):
            batch = group[start : start + probe.batch_size]
            batch_means.append(sum(losses[i] for i in batch) / len(batch))
    mean_of_batch_means = sum(batch_means) / len(batch_means)
    assert mean_of_batch_means != pytest.approx(weighted)
    assert metrics["eval/loss"] != pytest.approx(mean_of_batch_means)

    for ti, t_val in enumerate(probe.t_grid):
        tag = f"{t_val:.2f}".replace(".", "").ljust(3, "0")
        assert metrics[f"eval/loss_t{tag}"] == pytest.approx(weighted)

    for name in probe.band_names:
        members = [i for i, band in enumerate(probe.bands) if band == name]
        if not members:
            assert f"eval/loss/band_{name}" not in metrics
            continue
        assert metrics[f"eval/loss/band_{name}"] == pytest.approx(
            sum(losses[i] for i in members) / len(members)
        )
        assert metrics[f"eval/samples/band_{name}"] == float(len(members))


# ---------------------------------------------------------------------------
# Comparability: one image, several bands, one noise and one timestep
# ---------------------------------------------------------------------------


def test_paired_samples_share_noise_and_timestep(tmp_path):
    rows = [([f"row{i}_short", f"row{i}_long"], (32, 32)) for i in range(3)]
    path = _save_dataset(tmp_path, rows)
    _save_sidecar(path, [[100, 900]] * 3)

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=3))
    indices_per_row = {}
    for index, row in enumerate(probe.sample_rows):
        indices_per_row.setdefault(row, []).append(index)
    assert sorted(len(entries) for entries in indices_per_row.values()) == [2, 2, 2]

    model = _RecordingModel()
    probe.evaluate(model)

    for entries in indices_per_row.values():
        first, second = entries
        assert probe.bands[first] != probe.bands[second]
        assert probe.captions[first] != probe.captions[second]
        assert torch.equal(probe.noise[first], probe.noise[second])
        assert torch.equal(probe.latents[first], probe.latents[second])
        for t_val in probe.t_grid:
            z_t, t = _model_inputs(probe, first, t_val)
            other_z_t, other_t = _model_inputs(probe, second, t_val)
            assert torch.equal(z_t, other_z_t)
            assert torch.equal(t, other_t)
            # Both samples really reached the model with that input.
            assert _calls_matching(model.calls, z_t, t) == 2


def test_probe_is_deterministic_with_bands(tmp_path):
    kwargs = _probe_kwargs(_band_dataset(tmp_path), num_samples=4)
    first = EvalLossProbe(**kwargs)
    second = EvalLossProbe(**kwargs)

    assert first.captions == second.captions
    assert first.bands == second.bands
    assert first.sample_rows == second.sample_rows
    assert first.paired_images == second.paired_images
    assert len(first.noise) == len(second.noise)
    for one, other in zip(first.noise, second.noise):
        assert torch.equal(one, other)

    metrics = first.evaluate(_ZeroModel())
    other_metrics = second.evaluate(_ZeroModel())
    assert metrics.keys() == other_metrics.keys()
    for key, value in metrics.items():
        assert value == pytest.approx(other_metrics[key], rel=0, abs=1e-7)


def test_paired_images_counts_images_in_several_bands(tmp_path):
    paired = _save_dataset(
        tmp_path, [([f"row{i}_short", f"row{i}_long"], (32, 32)) for i in range(3)],
        name="paired",
    )
    _save_sidecar(paired, [[100, 900]] * 3)
    probe = EvalLossProbe(**_probe_kwargs(paired, num_samples=3))
    metrics = probe.evaluate(_ZeroModel())
    assert probe.paired_images == 3
    assert metrics["eval/paired_images"] == 3.0

    single = _save_dataset(
        tmp_path,
        [(["row0"], (32, 32)), (["row1"], (32, 32)), (["row2"], (32, 32))],
        name="single_band",
    )
    _save_sidecar(single, [[100], [300], [600]])
    unpaired = EvalLossProbe(**_probe_kwargs(single, num_samples=3))
    assert unpaired.paired_images == 0
    assert unpaired.evaluate(_ZeroModel())["eval/paired_images"] == 0.0


# ---------------------------------------------------------------------------
# No usable sidecar
# ---------------------------------------------------------------------------


def test_falls_back_to_the_positional_caption_without_a_sidecar(tmp_path, capsys):
    rows = [
        (["row0_only"], (32, 32)),
        (["row1_first", "row1_second"], (32, 32)),
        (["row2_only"], (32, 32)),
    ]
    path = _save_dataset(tmp_path, rows)
    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=3))

    assert not probe.banded
    assert probe.band_sample_counts == {}
    assert sorted(probe.captions) == ["row0_only", "row1_second", "row2_only"]
    assert len(probe.latents) == 3
    assert "caption bands are off" in capsys.readouterr().out

    metrics = probe.evaluate(_ZeroModel())
    assert set(metrics) == LEGACY_KEYS


def test_sidecar_for_another_dataset_falls_back(tmp_path, capsys):
    rows = [([f"row{i}_first", f"row{i}_second"], (32, 32)) for i in range(3)]
    path = _save_dataset(tmp_path, rows)
    _save_sidecar(path, [[100, 900], [100, 900]])  # two rows, dataset has three

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=3))

    assert not probe.banded
    assert set(probe.evaluate(_ZeroModel())) == LEGACY_KEYS
    assert "caption bands are off" in capsys.readouterr().out


def test_unreadable_sidecar_falls_back(tmp_path, capsys):
    rows = [([f"row{i}"], (32, 32)) for i in range(2)]
    path = _save_dataset(tmp_path, rows)
    sidecar_path(path).write_bytes(b"not an archive")

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=2))

    assert not probe.banded
    assert set(probe.evaluate(_ZeroModel())) == LEGACY_KEYS
    assert "caption bands are off" in capsys.readouterr().out


def test_stale_sidecar_falls_back(tmp_path, capsys):
    # Same row count, but the caption lists the sidecar describes have moved:
    # keeping it would read every row's band off another row's lengths.
    rows = [([f"row{i}_short", f"row{i}_long"], (32, 32)) for i in range(2)]
    path = _save_dataset(tmp_path, rows)
    _save_sidecar(path, [[100], [900]])

    probe = EvalLossProbe(**_probe_kwargs(path, num_samples=2))

    assert not probe.banded
    assert set(probe.evaluate(_ZeroModel())) == LEGACY_KEYS
    assert "caption bands are off" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Existing guarantees, unchanged by banding
# ---------------------------------------------------------------------------


def _write_tiny_dataset(tmp_path):
    gen = torch.Generator().manual_seed(0)
    rows = {
        "latents": [],
        "captions": [],
        "resolution_bucket_id": [],
    }
    # Two shape groups: 32x32 and 32x16 latents (16ch)
    for i in range(8):
        h, w = (32, 32) if i % 2 == 0 else (32, 16)
        rows["latents"].append(torch.randn((16, h, w), generator=gen).numpy())
        rows["captions"].append([f"short caption {i}", f"a longer caption {i}"])
        rows["resolution_bucket_id"].append(1 if i % 2 == 0 else 2)
    ds = Dataset.from_dict(rows)
    path = tmp_path / "tiny_eval"
    ds.save_to_disk(str(path))
    return str(path)


@pytest.fixture()
def probe_env(tmp_path):
    path = _write_tiny_dataset(tmp_path)

    model = ArtFlow(
        hidden_size=64,
        num_heads=4,
        double_stream_depth=0,
        single_stream_depth=2,
        conditioning_scheme="pure",
        qkv_bias=False,
        ffn_type="gated",
    )
    return model, _probe_kwargs(path, num_samples=8, batch_size=4)


def test_probe_is_deterministic(probe_env):
    model, kwargs = probe_env
    probe_a = EvalLossProbe(**kwargs)
    probe_b = EvalLossProbe(**kwargs)
    ma = probe_a.evaluate(model)
    mb = probe_b.evaluate(model)
    assert ma.keys() == mb.keys()
    for k in ma:
        assert ma[k] == pytest.approx(mb[k], rel=0, abs=1e-7)
    # repeated evaluation on the same probe is also stable
    ma2 = probe_a.evaluate(model)
    assert ma["eval/loss"] == pytest.approx(ma2["eval/loss"], rel=0, abs=1e-7)


def test_probe_grouping_and_metrics_keys(probe_env):
    model, kwargs = probe_env
    probe = EvalLossProbe(**kwargs)
    assert len(probe.groups) == 2  # two latent shapes were grouped separately
    metrics = probe.evaluate(model)
    assert "eval/loss" in metrics
    assert set(k for k in metrics if k.startswith("eval/loss_t")) == {
        f"eval/loss_t{tag}" for tag in ["015", "040", "065", "090"]
    }
    assert all(v == v and v >= 0 for v in metrics.values())  # no NaN


def test_probe_restores_training_mode(probe_env):
    model, kwargs = probe_env
    model.train()
    probe = EvalLossProbe(**kwargs)
    probe.evaluate(model)
    assert model.training


def test_encode_caption_chunks(monkeypatch):
    """Chunked pre-encoding keeps row order and re-pads to the widest chunk."""
    calls = []

    def fake_encode(texts, model, tokenizer, pooling, exit_layer=None):
        calls.append(list(texts))
        b = len(texts)
        width = 3 + b  # the trailing short chunk comes back narrower
        emb = torch.stack([torch.full((width, 4), float(t)) for t in texts])
        mask = torch.ones(b, width, dtype=torch.long)
        pooled = torch.stack([torch.full((4,), float(t)) for t in texts]) if pooling else None
        return emb, mask, pooled

    monkeypatch.setattr(eval_loss_module, "encode_text", fake_encode)
    monkeypatch.setattr(eval_loss_module, "_ENCODE_CHUNK", 2)

    txt, mask, pooled = eval_loss_module._encode_caption_chunks(
        ["0", "1", "2", "3", "4"], None, None, True, None
    )

    assert calls == [["0", "1"], ["2", "3"], ["4"]]
    assert txt.shape == (5, 5, 4)
    assert mask.shape == (5, 5)
    assert pooled.shape == (5, 4)
    assert [row[0, 0].item() for row in txt] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert mask[4].tolist() == [1, 1, 1, 1, 0]  # narrow chunk is zero-padded
    assert [row[0].item() for row in pooled] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_encode_caption_chunks_rejects_empty():
    with pytest.raises(ValueError):
        eval_loss_module._encode_caption_chunks([], None, None, False, None)
