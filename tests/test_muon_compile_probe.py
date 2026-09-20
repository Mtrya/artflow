import torch

from scripts.bench.muon_compile_probe import batch_shapes
from src.models.artflow import ArtFlow


def test_probe_counts_all_muon_chunks_without_model_allocation():
    from src.train.muon import Muon, build_param_groups

    with torch.device("meta"):
        model = ArtFlow(hidden_size=64, num_heads=4, double_stream_depth=1,
                        single_stream_depth=2)
    shapes = batch_shapes(model)
    assert shapes and all(n > 0 and r > 0 and c > 0 for n, r, c in shapes)
    optimizers = build_param_groups(model, muon_lr=.02)
    expected = sum(p.numel() for opt in optimizers if isinstance(opt, Muon)
                   for group in opt.param_groups for p in group["params"]
                   if sum(group["chunks"] for _ in group["params"]) > 1)
    assert sum(n * r * c for n, r, c in shapes) == expected
