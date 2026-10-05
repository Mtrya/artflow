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
        # Accumulate outside the model's mixed precision: small increments
        # disappear in BF16, and rounded times corrupt high-frequency features.
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        t_tensor = torch.tensor(t, device=x.device, dtype=torch.float32).expand(x.shape[0])
        v = model_fn(x, t_tensor).to(x.dtype)
        return x + v * dt


class Heun(Solver):
    """Heun's method (Improved Euler) for ODEs."""

    def step(
        self, x: torch.Tensor, t: float, dt: float, model_fn: Callable
    ) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        t_tensor = torch.tensor(t, device=x.device, dtype=torch.float32).expand(x.shape[0])
        v1 = model_fn(x, t_tensor).to(x.dtype)

        x_guess = x + v1 * dt
        t_next = t + dt
        t_next_tensor = torch.tensor(t_next, device=x.device, dtype=torch.float32).expand(
            x.shape[0]
        )
        v2 = model_fn(x_guess, t_next_tensor).to(x.dtype)

        return x + 0.5 * (v1 + v2) * dt


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
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Sample with FP32 times and at least FP32 integration state.

    The model may use autocast and return lower-precision velocities. Cast the
    final samples to the decoder's dtype at the decode boundary.

    Args:
        time_shift: If provided, the uniform timestep schedule is shifted via
            t' = t / (s - (s - 1) * t) (SD3 Eq. 23 adapted to this repo's
            t=0-noise convention, see flow.paths.apply_time_shift). When None,
            the shift is automatically computed from z0's spatial shape using
            the same resolution-dependent formula used in training.
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
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    # Build shifted timestep schedule
    uniform_ts = torch.linspace(t_start, t_end, steps + 1, dtype=torch.float32)
    shifted_ts = [shift_timesteps(u, z0, time_shift=time_shift).item() for u in uniform_ts]

    intermediates = []
    if return_intermediates:
        intermediates.append(x.cpu())

    for i in range(steps):
        if progress_callback is not None:
            progress_callback(i + 1, steps)
        t = shifted_ts[i]
        dt = shifted_ts[i + 1] - shifted_ts[i]
        x = s.step(x, t, dt, model_fn)
        if return_intermediates:
            intermediates.append(x.cpu())

    if return_intermediates:
        return x, intermediates
    return x
