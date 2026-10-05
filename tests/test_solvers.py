"""Flow integration checked against analytic ODE solutions and training time shifts."""

import math

import pytest
import torch

from src.flow.paths import resolution_time_shift, shift_timesteps
from src.flow.solvers import sample_ode


@pytest.mark.parametrize('solver', ['euler', 'heun'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float64])
def test_constant_velocity_solution_preserves_small_increments(solver, dtype):
    initial = torch.ones(1, 1, 2, 2, dtype=dtype)
    velocity = torch.tensor(0.1, dtype=dtype).item()
    result = sample_ode(lambda x, t: torch.full_like(x, velocity, dtype=dtype),
                        initial, steps=50, solver=solver, time_shift=1)
    expected = torch.full_like(result, 1 + velocity)
    torch.testing.assert_close(result, expected, rtol=0, atol=4e-6)


@pytest.mark.parametrize('solver,tolerance', [('euler', .015), ('heun', .00005)])
def test_linear_ode_matches_exponential_solution(solver, tolerance):
    initial = torch.tensor([[[[1., -2.]]]], dtype=torch.float64)
    result = sample_ode(lambda x, t: x, initial, steps=100, solver=solver, time_shift=1)
    torch.testing.assert_close(result, initial * math.e, rtol=tolerance, atol=0)


def test_shifted_solver_integrates_over_training_time_interval():
    initial = torch.zeros(1, 16, 64, 64)
    endpoints = shift_timesteps(torch.tensor([.2, .7]), initial)
    # For v(t)=t, the exact integral is half the change in squared time.
    expected = (endpoints[1].square() - endpoints[0].square()) / 2
    result = sample_ode(lambda x, t: t[:, None, None, None].expand_as(x), initial,
                        steps=5, solver='heun', t_start=.2, t_end=.7,
                        time_shift=resolution_time_shift(initial))
    torch.testing.assert_close(result, torch.full_like(result, expected))


@pytest.mark.parametrize('width,shift', [(32, 1.), (128, 4.)])
def test_time_shift_matches_sd3_token_count_anchors(width, shift):
    # SD3 Eq. 23 gives shifts 1 and 4 for 256 and 4096 image tokens.
    assert resolution_time_shift(torch.empty(1, 16, width, width)) == shift
