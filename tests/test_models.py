"""
Comprehensive tests for ArtFlow models
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.artflow_uncond import ArtFlowUncond
from src.models.artflow import ArtFlow


def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{'=' * 60}")
    print(f"Testing: {test_name}")
    print(f"{'=' * 60}")


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
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2

    # Create inputs
    H, W = 32, 32
    x = torch.randn(batch_size, in_channels, H, W)
    t = torch.randint(0, 1000, (batch_size,))

    # Initialize model
    model = ArtFlowUncond(
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
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
    """Test ArtFlow with Pure conditioning (timestep only)"""
    print_test_header("ArtFlow Pure")

    # Setup parameters
    batch_size = 2
    in_channels = 4
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    txt_in_features = 64
    txt_seq_len = 10

    # Create inputs
    H, W = 32, 32
    x = torch.randn(batch_size, in_channels, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)

    # Initialize model
    model = ArtFlow(
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_heads=num_heads,
        txt_in_features=txt_in_features,
        conditioning_scheme="pure",
        double_stream_depth=depth,
        single_stream_depth=0,
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
    """Test ArtFlow with Fused conditioning (timestep + pooled text)"""
    print_test_header("ArtFlow Fused")

    # Setup parameters
    batch_size = 2
    in_channels = 4
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    txt_in_features = 64
    txt_seq_len = 10

    # Create inputs
    H, W = 32, 32
    x = torch.randn(batch_size, in_channels, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)
    txt_pooled = torch.randn(batch_size, txt_in_features)

    # Initialize model
    model = ArtFlow(
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_heads=num_heads,
        txt_in_features=txt_in_features,
        conditioning_scheme="fused",
        double_stream_depth=depth,
        single_stream_depth=0,
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
    """Test ArtFlow with Hybrid architecture (double + single stream)"""
    print_test_header("ArtFlow Hybrid")

    # Setup parameters
    batch_size = 2
    in_channels = 4
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    double_depth = 1
    single_depth = 1
    txt_in_features = 64
    txt_seq_len = 10

    # Create inputs
    H, W = 32, 32
    x = torch.randn(batch_size, in_channels, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)

    # Initialize model
    model = ArtFlow(
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_heads=num_heads,
        txt_in_features=txt_in_features,
        conditioning_scheme="pure",
        double_stream_depth=double_depth,
        single_stream_depth=single_depth,
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


def test_modulation_strategies():
    """Test different modulation sharing strategies"""
    print_test_header("Modulation Strategies")

    # Setup parameters
    batch_size = 2
    in_channels = 4
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    txt_in_features = 64
    txt_seq_len = 10

    # Create inputs
    H, W = 32, 32
    x = torch.randn(batch_size, in_channels, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)

    strategies = ["none", "stream", "layer", "all"]
    all_passed = True

    for strategy in strategies:
        model = ArtFlow(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            txt_in_features=txt_in_features,
            conditioning_scheme="pure",
            double_stream_depth=depth,
            single_stream_depth=0,
            modulation_share=strategy,
        )

        out = model(x, t, txt)
        shape_correct = out.shape == x.shape
        print_test_result(
            shape_correct, f"Modulation '{strategy}': Output shape {out.shape}"
        )
        all_passed = all_passed and shape_correct

    return all_passed


def test_ffn_types():
    """Test different FFN types"""
    print_test_header("FFN Types")

    # Setup parameters
    batch_size = 2
    in_channels = 4
    patch_size = 2
    hidden_size = 32
    num_heads = 4
    depth = 2
    txt_in_features = 64
    txt_seq_len = 10

    # Create inputs
    H, W = 32, 32
    x = torch.randn(batch_size, in_channels, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    txt = torch.randn(batch_size, txt_seq_len, txt_in_features)

    ffn_types = ["gated", "standard"]
    all_passed = True

    for ffn_type in ffn_types:
        model = ArtFlow(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            txt_in_features=txt_in_features,
            conditioning_scheme="pure",
            double_stream_depth=depth,
            single_stream_depth=0,
            ffn_type=ffn_type,
        )

        out = model(x, t, txt)
        shape_correct = out.shape == x.shape
        print_test_result(shape_correct, f"FFN '{ffn_type}': Output shape {out.shape}")
        all_passed = all_passed and shape_correct

    return all_passed


def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS FOR models")
    print("=" * 60)

    results = {}

    # Run all tests
    results["ArtFlowUncond"] = test_artflow_uncond()
    results["ArtFlow Pure"] = test_artflow_pure()
    results["ArtFlow Fused"] = test_artflow_fused()
    results["ArtFlow Hybrid"] = test_artflow_hybrid()
    results["Modulation Strategies"] = test_modulation_strategies()
    results["FFN Types"] = test_ffn_types()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

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


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
