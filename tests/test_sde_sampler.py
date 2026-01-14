import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.flow.sde_sampler import sde_step_with_logprob

def test_sde_step_shapes():
    """Test that output shapes are correct."""
    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    model_output = torch.randn(B, C, H, W)
    t = 0.5
    dt = -0.01
    
    prev_sample, log_prob, mean, std = sde_step_with_logprob(x, model_output, t, dt)
    
    print(f"✓ Shapes: prev_sample={prev_sample.shape}, log_prob={log_prob.shape}")
    
    assert prev_sample.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {prev_sample.shape}"
    assert mean.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {mean.shape}"
    assert log_prob.shape == (B,), f"Expected {(B,)}, got {log_prob.shape}"
    assert isinstance(std, float), f"Expected float, got {type(std)}"

def test_sde_step_deterministic():
    """Test that noise_level=0 yields ODE step results."""
    B, C, H, W = 1, 4, 8, 8
    x = torch.randn(B, C, H, W)
    model_output = torch.randn(B, C, H, W)
    t = 0.5
    dt = -0.1   
    
    # ODE: x_next = x + v * dt (when noise_level=0)
    expected = x + model_output * dt
    
    prev_sample, _, _, _ = sde_step_with_logprob(x, model_output, t, dt, noise_level=0.0)
    
    print(f"✓ ODE mode: max diff = {(prev_sample - expected).abs().max().item():.6f}")
    
    # Allow small float error
    assert torch.allclose(prev_sample, expected, atol=1e-5), "ODE mode should match expected"

def test_sde_gradients():
    """Ensure gradients flow from log_prob to model_output."""
    B, C, H, W = 1, 4, 8, 8
    x = torch.randn(B, C, H, W, requires_grad=False)
    model_output = torch.randn(B, C, H, W, requires_grad=True)
    t = 0.5
    dt = -0.01
    
    prev_sample, log_prob, _, _ = sde_step_with_logprob(x, model_output, t, dt)

    loss = log_prob.sum()
    loss.backward()
    
    print(f"✓ Gradients: grad_norm = {model_output.grad.norm().item():.6f}")
    
    assert model_output.grad is not None, "Gradients should exist"
    assert not torch.all(model_output.grad == 0), "Gradients should be non-zero"

if __name__ == "__main__":
    try:
        print("Running SDE Sampler Tests...")
        print("=" * 50)
        test_sde_step_shapes()
        test_sde_step_deterministic()
        test_sde_gradients()
        print("=" * 50)
        print("✅ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
