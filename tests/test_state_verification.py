import copy

import pytest
import torch
from safetensors.torch import save_file

from src.train.state_verification import require_exact_state, verify_restored_training_state


def test_exact_state_rejects_values_types_metadata_and_nonfinite():
    reference = {"state": [torch.tensor([1., 2.])], "lr": .01}
    assert require_exact_state(reference, copy.deepcopy(reference), label="test")["exact"]
    for actual in ({"state": [torch.tensor([1., 3.])], "lr": .01},
                   {"state": [torch.tensor([1., 2.], dtype=torch.float64)], "lr": .01},
                   {"state": [torch.tensor([1., 2.])], "lr": .02},
                   {"state": [torch.tensor([1., float("nan")])], "lr": .01}):
        with pytest.raises(ValueError):
            require_exact_state(reference, actual, label="test")


def test_verifies_live_loaded_state_and_detects_post_load_mutation(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    ema = copy.deepcopy(model)
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.bin")
    torch.save(scheduler.state_dict(), tmp_path / "scheduler.bin")
    torch.save(ema.state_dict(), tmp_path / "ema_weights.pt")
    report = verify_restored_training_state(tmp_path, model, [optimizer], [scheduler], ema)
    assert report["exact"] and len(report["files"]) == 4
    with torch.no_grad():
        model.weight[0, 0].add_(.1)
    with pytest.raises(ValueError, match="model.safetensors"):
        verify_restored_training_state(tmp_path, model, [optimizer], [scheduler], ema)
