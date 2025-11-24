"""
Comprehensive tests for dit_blocks.py modules
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dit_blocks import (
    TimestepEmbeddings,
    apply_rotary_emb,
    MSRoPE,
    SingleStreamAttention,
    SingleStreamDiTBlock,
    DoubleStreamAttention,
    DoubleStreamDiTBlock,
    UnconditionalAttention,
    UnconditionalDiTBlock,
    FeedForward,
    modulate,
)


def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{'=' * 60}")
    print(f"Testing: {test_name}")
    print(f"{'=' * 60}")


def print_test_result(passed, message=""):
    """Print test result with formatting"""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: {message}")


def test_timestep_embeddings():
    """Test TimestepEmbeddings module"""
    print_test_header("TimestepEmbeddings")

    hidden_size = 256
    batch_size = 4

    model = TimestepEmbeddings(hidden_size=hidden_size)
    timesteps = torch.randn(batch_size)

    output = model(timesteps)

    # Check shape
    expected_shape = (batch_size, hidden_size)
    shape_correct = output.shape == expected_shape
    print_test_result(
        shape_correct, f"Output shape: {output.shape} (expected: {expected_shape})"
    )

    # Check that output is not all zeros
    non_zero = not torch.allclose(output, torch.zeros_like(output))
    print_test_result(non_zero, "Output contains non-zero values")

    # Check determinism
    output2 = model(timesteps)
    deterministic = torch.allclose(output, output2)
    print_test_result(deterministic, "Output is deterministic")

    return shape_correct and non_zero and deterministic


def test_apply_rotary_emb():
    """Test apply_rotary_emb function"""
    print_test_header("apply_rotary_emb")

    B, S, H, D = 2, 10, 4, 128
    x = torch.randn(B, S, H, D)
    freqs = torch.randn(S, D // 2, dtype=torch.complex64)

    output = apply_rotary_emb(x, freqs)

    # Check shape preservation
    shape_correct = output.shape == x.shape
    print_test_result(
        shape_correct, f"Output shape: {output.shape} (expected: {x.shape})"
    )

    # Check dtype matches input
    dtype_correct = output.dtype == x.dtype
    print_test_result(
        dtype_correct, f"Output dtype: {output.dtype} (expected: {x.dtype})"
    )

    # Check output is different from input (rotation applied)
    modified = not torch.allclose(output, x)
    print_test_result(modified, "Rotation was applied (output differs from input)")

    return shape_correct and dtype_correct and modified


def test_msrope():
    """Test MSRoPE module"""
    print_test_header("MSRoPE")

    rope = MSRoPE(theta=10000, axes_dim=[64, 64])
    img_hw = (32, 32)
    txt_seq_len = 77
    device = torch.device("cpu")

    img_freqs, txt_freqs = rope(img_hw, txt_seq_len, device)

    # Check image frequencies shape
    expected_img_shape = (32 * 32, 64)
    img_shape_correct = img_freqs.shape == expected_img_shape
    print_test_result(
        img_shape_correct,
        f"Image freqs shape: {img_freqs.shape} (expected: {expected_img_shape})",
    )

    # Check text frequencies shape
    expected_txt_shape = (77, 64)
    txt_shape_correct = txt_freqs.shape == expected_txt_shape
    print_test_result(
        txt_shape_correct,
        f"Text freqs shape: {txt_freqs.shape} (expected: {expected_txt_shape})",
    )

    # Check dtype is complex
    img_complex = (
        img_freqs.dtype == torch.complex64 or img_freqs.dtype == torch.complex128
    )
    txt_complex = (
        txt_freqs.dtype == torch.complex64 or txt_freqs.dtype == torch.complex128
    )
    print_test_result(
        img_complex and txt_complex,
        f"Frequencies are complex (img: {img_freqs.dtype}, txt: {txt_freqs.dtype})",
    )

    # Test caching
    img_freqs2, txt_freqs2 = rope(img_hw, txt_seq_len, device)
    cache_works = torch.allclose(img_freqs, img_freqs2) and torch.allclose(
        txt_freqs, txt_freqs2
    )
    print_test_result(cache_works, "Caching works correctly")

    return (
        img_shape_correct
        and txt_shape_correct
        and img_complex
        and txt_complex
        and cache_works
    )


def test_feedforward():
    """Test FeedForward module"""
    print_test_header("FeedForward")

    dim = 512
    hidden_dim = 2048
    batch_size = 2
    seq_len = 16

    model = FeedForward(dim=dim, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, dim)

    output = model(x)

    # Check shape preservation
    shape_correct = output.shape == x.shape
    print_test_result(
        shape_correct, f"Output shape: {output.shape} (expected: {x.shape})"
    )

    # Check output is different from input
    modified = not torch.allclose(output, x)
    print_test_result(modified, "MLP transformation applied")

    return shape_correct and modified


def test_modulate():
    """Test modulate function"""
    print_test_header("modulate")

    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    shift = torch.randn(B, D)
    scale = torch.randn(B, D)

    output = modulate(x, shift, scale)

    # Check shape preservation
    shape_correct = output.shape == x.shape
    print_test_result(
        shape_correct, f"Output shape: {output.shape} (expected: {x.shape})"
    )

    # Check modulation was applied
    modified = not torch.allclose(output, x)
    print_test_result(modified, "Modulation was applied")

    # Test with zero scale and shift (should be close to x)
    zero_shift = torch.zeros(B, D)
    zero_scale = torch.zeros(B, D)
    output_zero = modulate(x, zero_shift, zero_scale)
    identity = torch.allclose(output_zero, x)
    print_test_result(identity, "Identity test (zero shift/scale)")

    return shape_correct and modified and identity


def test_single_stream_attention():
    """Test SingleStreamAttention module"""
    print_test_header("SingleStreamAttention")

    dim = 512
    num_heads = 4  # head_dim = 128
    batch_size = 2
    seq_len = 100

    model = SingleStreamAttention(dim=dim, num_heads=num_heads, rope_axes_dim=[64, 64])
    x = torch.randn(batch_size, seq_len, dim)

    # Generate dummy frequencies
    # head_dim = 128, so we need 128/2 = 64 complex frequencies
    freqs = torch.randn(seq_len, 64, dtype=torch.complex64)

    output = model(x, freqs)

    # Check shape preservation
    shape_correct = output.shape == x.shape
    print_test_result(
        shape_correct, f"Output shape: {output.shape} (expected: {x.shape})"
    )

    # Check attention was applied
    modified = not torch.allclose(output, x)
    print_test_result(modified, "Attention transformation applied")

    # Test with attention mask
    mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[:, :, :, 50:] = False  # Mask second half
    output_masked = model(x, freqs, attention_mask=mask)
    mask_applied = not torch.allclose(output, output_masked)
    print_test_result(mask_applied, "Attention mask affects output")

    return shape_correct and modified and mask_applied


def test_single_stream_dit_block():
    """Test SingleStreamDiTBlock module"""
    print_test_header("SingleStreamDiTBlock")

    dim = 512
    num_heads = 4  # head_dim = 128
    c_dim = 256
    batch_size = 2
    img_hw = (16, 16)
    txt_seq_len = 77

    model = SingleStreamDiTBlock(
        dim=dim, num_heads=num_heads, c_dim=c_dim, rope_axes_dim=[64, 64]
    )

    img_tokens = torch.randn(batch_size, img_hw[0] * img_hw[1], dim)
    txt_tokens = torch.randn(batch_size, txt_seq_len, dim)
    c = torch.randn(batch_size, c_dim)

    # Test without mask
    out_img, out_txt = model(img_tokens, txt_tokens, c, img_hw, txt_seq_len)

    # Check shapes
    img_shape_correct = out_img.shape == img_tokens.shape
    txt_shape_correct = out_txt.shape == txt_tokens.shape
    print_test_result(
        img_shape_correct,
        f"Image output shape: {out_img.shape} (expected: {img_tokens.shape})",
    )
    print_test_result(
        txt_shape_correct,
        f"Text output shape: {out_txt.shape} (expected: {txt_tokens.shape})",
    )

    # Check transformation was applied
    img_modified = not torch.allclose(out_img, img_tokens)
    txt_modified = not torch.allclose(out_txt, txt_tokens)
    print_test_result(img_modified, "Image tokens were transformed")
    print_test_result(txt_modified, "Text tokens were transformed")

    # Test with attention mask
    txt_mask = torch.ones(batch_size, txt_seq_len)
    txt_mask[:, 50:] = 0  # Mask last tokens
    out_img_masked, out_txt_masked = model(
        img_tokens, txt_tokens, c, img_hw, txt_seq_len, txt_attention_mask=txt_mask
    )

    mask_affects = not torch.allclose(out_txt, out_txt_masked)
    print_test_result(mask_affects, "Attention mask affects text output")

    return (
        img_shape_correct
        and txt_shape_correct
        and img_modified
        and txt_modified
        and mask_affects
    )


def test_double_stream_attention():
    """Test DoubleStreamAttention module"""
    print_test_header("DoubleStreamAttention")

    dim = 512
    num_heads = 4  # head_dim = 128
    batch_size = 2
    img_hw = (16, 16)
    txt_seq_len = 77

    model = DoubleStreamAttention(dim=dim, num_heads=num_heads, rope_axes_dim=[64, 64])

    img_tokens = torch.randn(batch_size, img_hw[0] * img_hw[1], dim)
    txt_tokens = torch.randn(batch_size, txt_seq_len, dim)

    # Test without mask
    out_img, out_txt = model(img_tokens, txt_tokens, img_hw, txt_seq_len)

    # Check shapes
    img_shape_correct = out_img.shape == img_tokens.shape
    txt_shape_correct = out_txt.shape == txt_tokens.shape
    print_test_result(
        img_shape_correct,
        f"Image output shape: {out_img.shape} (expected: {img_tokens.shape})",
    )
    print_test_result(
        txt_shape_correct,
        f"Text output shape: {out_txt.shape} (expected: {txt_tokens.shape})",
    )

    # Check transformation was applied
    img_modified = not torch.allclose(out_img, img_tokens)
    txt_modified = not torch.allclose(out_txt, txt_tokens)
    print_test_result(img_modified, "Image tokens were transformed")
    print_test_result(txt_modified, "Text tokens were transformed")

    # Test with attention mask
    txt_mask = torch.ones(batch_size, txt_seq_len)
    txt_mask[:, 50:] = 0
    out_img_masked, out_txt_masked = model(
        img_tokens, txt_tokens, img_hw, txt_seq_len, txt_attention_mask=txt_mask
    )

    mask_affects = not torch.allclose(out_txt, out_txt_masked)
    print_test_result(mask_affects, "Attention mask affects text output")

    return (
        img_shape_correct
        and txt_shape_correct
        and img_modified
        and txt_modified
        and mask_affects
    )


def test_double_stream_dit_block():
    """Test DoubleStreamDiTBlock module"""
    print_test_header("DoubleStreamDiTBlock")

    dim = 512
    num_heads = 4  # head_dim = 128
    c_dim = 256
    batch_size = 2
    img_hw = (16, 16)
    txt_seq_len = 77

    model = DoubleStreamDiTBlock(
        dim=dim, num_heads=num_heads, c_dim=c_dim, rope_axes_dim=[64, 64]
    )

    img_tokens = torch.randn(batch_size, img_hw[0] * img_hw[1], dim)
    txt_tokens = torch.randn(batch_size, txt_seq_len, dim)
    c = torch.randn(batch_size, c_dim)

    # Test without mask
    out_img, out_txt = model(img_tokens, txt_tokens, c, img_hw, txt_seq_len)

    # Check shapes
    img_shape_correct = out_img.shape == img_tokens.shape
    txt_shape_correct = out_txt.shape == txt_tokens.shape
    print_test_result(
        img_shape_correct,
        f"Image output shape: {out_img.shape} (expected: {img_tokens.shape})",
    )
    print_test_result(
        txt_shape_correct,
        f"Text output shape: {out_txt.shape} (expected: {txt_tokens.shape})",
    )

    # Check transformation was applied
    img_modified = not torch.allclose(out_img, img_tokens)
    txt_modified = not torch.allclose(out_txt, txt_tokens)
    print_test_result(img_modified, "Image tokens were transformed")
    print_test_result(txt_modified, "Text tokens were transformed")

    # Test with attention mask
    txt_mask = torch.ones(batch_size, txt_seq_len)
    txt_mask[:, 50:] = 0
    out_img_masked, out_txt_masked = model(
        img_tokens, txt_tokens, c, img_hw, txt_seq_len, txt_attention_mask=txt_mask
    )

    mask_affects = not torch.allclose(out_txt, out_txt_masked)
    print_test_result(mask_affects, "Attention mask affects text output")

    return (
        img_shape_correct
        and txt_shape_correct
        and img_modified
        and txt_modified
        and mask_affects
    )


def test_unconditional_attention():
    """Test UnconditionalAttention module"""
    print_test_header("UnconditionalAttention")

    dim = 512
    num_heads = 4  # head_dim = 128
    batch_size = 2
    img_hw = (16, 16)
    seq_len = img_hw[0] * img_hw[1]

    model = UnconditionalAttention(dim=dim, num_heads=num_heads, rope_axes_dim=[64, 64])
    x = torch.randn(batch_size, seq_len, dim)

    output = model(x, img_hw)

    # Check shape preservation
    shape_correct = output.shape == x.shape
    print_test_result(
        shape_correct, f"Output shape: {output.shape} (expected: {x.shape})"
    )

    # Check attention was applied
    modified = not torch.allclose(output, x)
    print_test_result(modified, "Attention transformation applied")

    return shape_correct and modified


def test_unconditional_dit_block():
    """Test UnconditionalDiTBlock module"""
    print_test_header("UnconditionalDiTBlock")

    dim = 512
    num_heads = 4  # head_dim = 128
    c_dim = 256
    batch_size = 2
    img_hw = (16, 16)

    model = UnconditionalDiTBlock(
        dim=dim, num_heads=num_heads, c_dim=c_dim, rope_axes_dim=[64, 64]
    )

    img_tokens = torch.randn(batch_size, img_hw[0] * img_hw[1], dim)
    c = torch.randn(batch_size, c_dim)

    out = model(img_tokens, c, img_hw)

    # Check shape
    shape_correct = out.shape == img_tokens.shape
    print_test_result(
        shape_correct, f"Output shape: {out.shape} (expected: {img_tokens.shape})"
    )

    # Check transformation was applied
    modified = not torch.allclose(out, img_tokens)
    print_test_result(modified, "Tokens were transformed")

    return shape_correct and modified


def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS FOR dit_blocks.py")
    print("=" * 60)

    results = {}

    # Run all tests
    results["TimestepEmbeddings"] = test_timestep_embeddings()
    results["apply_rotary_emb"] = test_apply_rotary_emb()
    results["MSRoPE"] = test_msrope()
    results["FeedForward"] = test_feedforward()
    results["modulate"] = test_modulate()
    results["SingleStreamAttention"] = test_single_stream_attention()
    results["SingleStreamDiTBlock"] = test_single_stream_dit_block()
    results["DoubleStreamAttention"] = test_double_stream_attention()
    results["DoubleStreamDiTBlock"] = test_double_stream_dit_block()
    results["UnconditionalAttention"] = test_unconditional_attention()
    results["UnconditionalDiTBlock"] = test_unconditional_dit_block()

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
