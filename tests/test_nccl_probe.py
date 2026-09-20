import pytest

from scripts.bench.nccl_allreduce_bench import parse_args


def test_default_matches_actual_hero_fp32_payload():
    args = parse_args([])
    assert args.elements * 4 == 2130827248
    assert args.dtype == "fp32" and args.bucket_mib == 25
    assert args.trace_dir is None


def test_trace_is_optional_and_separate():
    args = parse_args(["--trace-dir", "traces/protocol"])
    assert str(args.trace_dir) == "traces/protocol"


@pytest.mark.parametrize("flag", ["--elements", "--iterations", "--bucket-mib"])
def test_probe_rejects_invalid_sizes(flag):
    with pytest.raises(SystemExit):
        parse_args([flag, "0"])
