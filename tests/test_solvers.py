"""Sampling precision must not inherit the model's autocast dtype."""

import pytest
import torch

from src.flow.solvers import Euler, Heun, sample_ode


@pytest.mark.parametrize("solver", [Euler, Heun])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_solver_step_preserves_time_and_small_increments(solver, dtype):
    seen = []
    velocity = torch.tensor(0.1, dtype=dtype)

    def model(x, t):
        seen.append(t.clone())
        assert x.dtype == torch.float32
        return torch.full_like(x, velocity.item(), dtype=dtype)

    result = solver().step(torch.ones(2, 1, 2, 2, dtype=dtype), 0.9, 0.002, model)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full_like(result, 1 + velocity.item() * 0.002),
                               rtol=0, atol=1e-7)
    for index, t in enumerate(seen):
        assert t.dtype == torch.float32
        torch.testing.assert_close(t, torch.full_like(t, 0.9 + index * 0.002),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_low_precision_velocity_accumulates_across_sampling_steps(solver):
    z0 = torch.ones(1, 1, 2, 2, dtype=torch.bfloat16)
    velocity = torch.tensor(0.1, dtype=torch.bfloat16).item()
    result, states = sample_ode(
        lambda x, t: torch.full_like(x, velocity, dtype=torch.bfloat16),
        z0, steps=50, solver=solver, time_shift=1, return_intermediates=True,
    )
    assert len(states) == 51
    assert all(state.dtype == torch.float32 for state in states)
    torch.testing.assert_close(states[0], z0.float(), rtol=0, atol=0)
    torch.testing.assert_close(result, torch.full_like(result, 1 + velocity),
                               rtol=0, atol=4e-6)
    assert (result > 1).all()  # BF16 state previously lost every increment.


def test_heun_adds_velocities_after_promoting_precision():
    velocities = iter([1.0, 1.0078125])
    result = Heun().step(
        torch.zeros(1, dtype=torch.bfloat16), 0, 1,
        lambda x, t: torch.full_like(x, next(velocities), dtype=torch.bfloat16),
    )
    torch.testing.assert_close(result, torch.tensor([1.00390625]), rtol=0, atol=0)


def test_explicit_double_precision_state_is_preserved():
    result = sample_ode(lambda x, t: x * 0, torch.ones(1, 1, 2, 2, dtype=torch.float64),
                        steps=1, time_shift=1)
    assert result.dtype == torch.float64
