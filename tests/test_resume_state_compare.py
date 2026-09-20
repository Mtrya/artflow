import numpy as np
import pytest
import torch

from scripts.bench.compare_resume_state import compare_values


def test_nested_optimizer_and_rng_state_comparison():
    state = {"momentum": torch.ones(3), "rng": (np.arange(4, dtype=np.uint32), 42)}
    other = {"momentum": torch.ones(3), "rng": (np.arange(4, dtype=np.uint32), 42)}
    assert compare_values(state, other) == dict(
        exact=True, tensors=2, max_absolute_difference=0.0)
    other["momentum"][1] += 0.125
    result = compare_values(state, other)
    assert not result["exact"]
    assert result["max_absolute_difference"] == 0.125
    assert not compare_values({"step": 7}, {"step": 8})["exact"]


@pytest.mark.parametrize("other", [torch.ones(4), torch.ones(3, dtype=torch.float64),
                                  torch.tensor([1., float("nan"), 1.])])
def test_invalid_comparison_is_not_exact_evidence(other):
    with pytest.raises(ValueError):
        compare_values(torch.ones(3), other)
