import torch


def test_apply_time_shift_monotonic_and_bounded():
    from src.flow.paths import apply_time_shift

    t = torch.linspace(0.0, 1.0, 100)
    shifted = apply_time_shift(t, 2.5)

    assert torch.all(shifted >= 0)
    assert torch.all(shifted <= 1)
    assert torch.all(shifted[1:] >= shifted[:-1])


def test_apply_time_shift_moves_toward_noise():
    """In this repo t=0 is noise and t=1 is data; higher resolutions need
    noisier timesteps (SD3 Eq. 23), so shift > 1 must decrease t."""
    from src.flow.paths import apply_time_shift

    t = torch.linspace(0.0, 1.0, 100)
    shifted = apply_time_shift(t, 3.0)
    assert torch.all(shifted[1:-1] < t[1:-1])

    # endpoints are fixed points
    assert apply_time_shift(torch.tensor(0.0), 3.0).item() == 0.0
    assert apply_time_shift(torch.tensor(1.0), 3.0).item() == 1.0

    # shift=1 is the identity
    assert torch.allclose(apply_time_shift(t, 1.0), t)


def test_resolution_time_shift_anchors():
    from src.flow.paths import resolution_time_shift

    # 256 tokens anchor: H=W=32 latent with patch_size=2 => 16x16 patches => 256
    z_256 = torch.zeros(1, 16, 32, 32)
    assert resolution_time_shift(z_256) == 1.0

    # SD3 Eq. 23: shift = sqrt(m/n) in token counts; 4096 tokens => sqrt(16) = 4
    z_4096 = torch.zeros(1, 16, 128, 128)
    assert resolution_time_shift(z_4096) == 4.0


def test_training_and_inference_shift_convention_matches():
    from src.flow.paths import shift_timesteps, resolution_time_shift
    from src.flow.solvers import sample_ode

    z0 = torch.randn(2, 16, 64, 64)
    s = resolution_time_shift(z0)

    # Inference convention: shift the solver schedule. For one Euler step, the first time equals shift(t_start).
    seen_ts = []

    def model_fn(x, t):
        seen_ts.append(t.detach().cpu())
        return torch.zeros_like(x)

    _ = sample_ode(model_fn, z0=z0, steps=1, t_start=0.2, t_end=0.7, time_shift=s)

    # model_fn is called once; it should see the shifted t_start
    assert len(seen_ts) == 1

    t_start = torch.tensor(0.2)
    t_start_used = shift_timesteps(t_start, z0, time_shift=s).expand(z0.shape[0])
    assert torch.allclose(seen_ts[0], t_start_used, atol=1e-6)
