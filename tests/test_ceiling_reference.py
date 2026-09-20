from scripts.bench.transformer_ceiling import PEAK_BF16_TFLOPS


def test_4090_bf16_reference_is_dense_with_fp32_accumulation():
    assert PEAK_BF16_TFLOPS == 165.2
