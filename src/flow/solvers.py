"""
ODE and SDE solvers for sampling from flow matching and diffusion models.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union, List
import torch


class Solver(ABC):
    """Abstract base class for solvers."""

    @abstractmethod
    def step(
        self, x: torch.Tensor, t: float, dt: float, model_fn: Callable
    ) -> torch.Tensor:
        pass


class Euler(Solver):
    """Euler method for ODEs."""

    def step(
        self, x: torch.Tensor, t: float, dt: float, model_fn: Callable
    ) -> torch.Tensor:
        # x is current state at time t
        # model_fn(x, t) returns velocity v(x, t)
        t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0])
        v = model_fn(x, t_tensor)
        return x + v * dt


class Heun(Solver):
    """Heun's method (Improved Euler) for ODEs."""

    def step(
        self, x: torch.Tensor, t: float, dt: float, model_fn: Callable
    ) -> torch.Tensor:
        t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0])
        v1 = model_fn(x, t_tensor)

        x_guess = x + v1 * dt
        t_next = t + dt
        t_next_tensor = torch.tensor(t_next, device=x.device, dtype=x.dtype).expand(
            x.shape[0]
        )
        v2 = model_fn(x_guess, t_next_tensor)

        return x + 0.5 * (v1 + v2) * dt


class EulerMaruyama(Solver):
    """Euler-Maruyama method for SDEs."""

    def step(
        self,
        x: torch.Tensor,
        t: float,
        dt: float,
        model_fn: Callable,
        drift_fn: Callable,
        diffusion_fn: Callable,
    ) -> torch.Tensor:
        """
        Args:
            x: Current state
            t: Current time
            dt: Time step
            model_fn: Predicts score or relevant term
            drift_fn: Returns f(x, t)
            diffusion_fn: Returns g(t)
        """
        # Note: This signature is slightly different because SDEs need drift/diffusion terms
        t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0])

        score = model_fn(x, t_tensor)

        f = drift_fn(x, t_tensor)
        g = diffusion_fn(t_tensor)

        reverse_drift = f - (g**2) * score

        # Noise
        z = torch.randn_like(x)

        x_next = x + reverse_drift * dt + g * torch.abs(torch.tensor(dt)).sqrt() * z
        return x_next


class ScoreMatchingODE(Solver):
    """
    Probability Flow ODE solver for Score Matching (VP-SDE).
    dx = -0.5 * beta(t) * (x + score) * dt
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def step(
        self, x: torch.Tensor, t: float, dt: float, model_fn: Callable
    ) -> torch.Tensor:
        # t is current time
        # model_fn(x, t) returns score s(x, t)

        t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0])
        score = model_fn(x, t_tensor)

        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)

        # Probability Flow ODE: dx = -0.5 * beta(t) * (x + score) dt
        # Note: This is the vector field v(x, t)
        velocity = -0.5 * beta_t * (x + score)

        return x + velocity * dt


def sample_ode(
    model_fn: Callable,
    z0: torch.Tensor,
    steps: int = 100,
    solver: str = "euler",
    solver_instance: Optional[Solver] = None,
    t_start: float = 0.0,
    t_end: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    return_intermediates: bool = False,
    time_shift: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Sample using ODE solver.

    Args:
        time_shift: If provided, the uniform timestep schedule is shifted via
            t' = (s * t) / (1 + (s - 1) * t). When None, the shift is
            automatically computed from z0's spatial shape using the same
            resolution-dependent formula used in training.
    """
    from .paths import resolution_time_shift, shift_timesteps

    if solver_instance is not None:
        s = solver_instance
    elif solver == "euler":
        s = Euler()
    elif solver == "heun":
        s = Heun()
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if time_shift is None:
        time_shift = resolution_time_shift(z0)

    target_device = torch.device(device) if device is not None else z0.device
    x = z0.to(target_device)

    # Build shifted timestep schedule
    uniform_ts = torch.linspace(t_start, t_end, steps + 1)
    shifted_ts = [shift_timesteps(u, z0, time_shift=time_shift).item() for u in uniform_ts]

    intermediates = []
    if return_intermediates:
        intermediates.append(x.cpu())

    for i in range(steps):
        t = shifted_ts[i]
        dt = shifted_ts[i + 1] - shifted_ts[i]
        x = s.step(x, t, dt, model_fn)
        if return_intermediates:
            intermediates.append(x.cpu())

    if return_intermediates:
        return x, intermediates
    return x
