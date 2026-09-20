import pytest

from scripts.bench.attention_block_probe import comparison_names, compile_mode


def test_autotune_comparison_holds_attention_and_rope_fixed():
    names = comparison_names(True, True, True)
    assert names == ["NATIVE_FLASH_VARLEN_REAL_ROPE", "NATIVE_FLASH_VARLEN_REAL_ROPE_AUTOTUNE"]
    assert [compile_mode(name) for name in names] == ["default", "max-autotune-no-cudagraphs"]
    assert all(name.startswith("NATIVE_FLASH_VARLEN") and "_REAL_ROPE" in name for name in names)


@pytest.mark.parametrize("native,real", [(False, False), (True, False), (False, True)])
def test_autotune_rejects_changed_backend_policy(native, real):
    with pytest.raises(ValueError):
        comparison_names(native, real, True)


def test_original_backend_comparison_is_unchanged():
    assert comparison_names(False, False) == ["EFFICIENT_ATTENTION", "CUDNN_ATTENTION"]
    assert len(comparison_names(True, True)) == 4
