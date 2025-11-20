"""
Comprehensive tests for ArtFlow models
"""
import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.artflow_uncond import ArtFlowUncond
from models.artflow_pure import ArtFlowPure
from models.artflow_fused import ArtFlowFused
from models.artflow_hybrid import ArtFlowHybrid


def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")


def print_test_result(passed, message=""):
    """Print test result with formatting"""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: {message}")


def test_artflow_uncond():
    """Test ArtFlowUncond model"""
    print_test_header("ArtFlowUncond")
    
    # Setup parameters
    batch_size = 2
    in_channels = 4
    input_size = 32
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    
    # Create inputs
    x = torch.randn(batch_size, in_channels, input_size, input_size)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Initialize model
    model = ArtFlowUncond(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads
    )
    
    # Forward pass
    out = model(x, t)
    
    # Check shape
    shape_correct = out.shape == x.shape
    print_test_result(shape_correct, f"Output shape: {out.shape} (expected: {x.shape})")
    
    # Check for non-zero output
    non_zero = not torch.allclose(out, torch.zeros_like(out))
    print_test_result(non_zero, "Output contains non-zero values")
    
    return shape_correct and non_zero


def test_artflow_pure():
    """Test ArtFlowPure model"""
    print_test_header("ArtFlowPure")
    
    # Setup parameters
    batch_size = 2
    in_channels = 4
    input_size = 32
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    txt_in_features = 64
    txt_seq_len = 10
    
    # Create inputs
    x = torch.randn(batch_size, in_channels, input_size, input_size)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)
    
    # Initialize model
    model = ArtFlowPure(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        txt_in_features=txt_in_features
    )
    
    # Forward pass
    out = model(x, t, txt)
    
    # Check shape
    shape_correct = out.shape == x.shape
    print_test_result(shape_correct, f"Output shape: {out.shape} (expected: {x.shape})")
    
    # Check for non-zero output
    non_zero = not torch.allclose(out, torch.zeros_like(out))
    print_test_result(non_zero, "Output contains non-zero values")
    
    return shape_correct and non_zero


def test_artflow_fused():
    """Test ArtFlowFused model"""
    print_test_header("ArtFlowFused")
    
    # Setup parameters
    batch_size = 2
    in_channels = 4
    input_size = 32
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    txt_in_features = 64
    txt_seq_len = 10
    
    # Create inputs
    x = torch.randn(batch_size, in_channels, input_size, input_size)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)
    txt_pooled = torch.randn(batch_size, txt_in_features)
    
    # Initialize model
    model = ArtFlowFused(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        txt_in_features=txt_in_features
    )
    
    # Forward pass
    out = model(x, t, txt, txt_pooled)
    
    # Check shape
    shape_correct = out.shape == x.shape
    print_test_result(shape_correct, f"Output shape: {out.shape} (expected: {x.shape})")
    
    # Check for non-zero output
    non_zero = not torch.allclose(out, torch.zeros_like(out))
    print_test_result(non_zero, "Output contains non-zero values")
    
    return shape_correct and non_zero


def test_artflow_hybrid():
    """Test ArtFlowHybrid model"""
    print_test_header("ArtFlowHybrid")
    
    # Setup parameters
    batch_size = 2
    in_channels = 4
    input_size = 32
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    # Hybrid specific depths
    double_depth = 1
    single_depth = 1
    txt_in_features = 64
    txt_seq_len = 10
    
    # Create inputs
    x = torch.randn(batch_size, in_channels, input_size, input_size)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)
    txt_pooled = torch.randn(batch_size, txt_in_features)
    
    # Initialize model
    model = ArtFlowHybrid(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        double_depth=double_depth,
        single_depth=single_depth,
        num_heads=num_heads,
        txt_in_features=txt_in_features
    )
    
    # Forward pass
    out = model(x, t, txt, txt_pooled)
    
    # Check shape
    shape_correct = out.shape == x.shape
    print_test_result(shape_correct, f"Output shape: {out.shape} (expected: {x.shape})")
    
    # Check for non-zero output
    non_zero = not torch.allclose(out, torch.zeros_like(out))
    print_test_result(non_zero, "Output contains non-zero values")
    
    return shape_correct and non_zero


def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS FOR models")
    print("="*60)
    
    results = {}
    
    # Run all tests
    results['ArtFlowUncond'] = test_artflow_uncond()
    results['ArtFlowPure'] = test_artflow_pure()
    results['ArtFlowFused'] = test_artflow_fused()
    results['ArtFlowHybrid'] = test_artflow_hybrid()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
