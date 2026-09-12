import json
import math

import pytest

from scripts.bench.size_micro_batches import (
    _image_tokens, largest_batch, parse_oom_log, read_plan, main)


PLAN = {"1": [{"max_length": 26, "batch_size": 16},
              {"max_length": 2048, "batch_size": 8}],
        "2": [{"max_length": 90, "batch_size": 16}]}


def write_plan_file(tmp_path, plan=PLAN):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return str(path)


def test_largest_batch_follows_the_linear_model():
    # available = 45 - 9.5 - 10 - 4.7 = 20.8; seq 282 -> floor(20.8/0.2848) = 73
    assert largest_batch(282, const_gb=9.5, slope=0.00101,
                         reserved_offset_gb=4.7, surcharge_gb=10.0,
                         ceiling_gb=45.0) == 73


def test_largest_batch_never_returns_zero():
    assert largest_batch(100000, const_gb=9.5, slope=0.00101,
                         reserved_offset_gb=4.7, surcharge_gb=10.0,
                         ceiling_gb=45.0) == 1


def test_largest_batch_rejects_a_ceiling_below_the_fixed_terms():
    with pytest.raises(ValueError, match="leaves nothing"):
        largest_batch(282, const_gb=9.5, slope=0.00101,
                      reserved_offset_gb=4.7, surcharge_gb=40.0,
                      ceiling_gb=45.0)


def test_image_tokens_scalar_and_mapping():
    assert _image_tokens("256") == {"*": 256}
    assert _image_tokens('{"1": 256, "2": 252}') == {"1": 256, "2": 252}
    with pytest.raises(ValueError):
        _image_tokens("0")
    with pytest.raises(ValueError):
        _image_tokens("[1, 2]")


def test_table_fills_every_bucket(tmp_path, capsys):
    plan_path = write_plan_file(tmp_path)
    out = tmp_path / "filled.json"
    assert main(["table", "--plan", plan_path, "--image-tokens", "256",
                 "--out", str(out)]) == 0
    filled = read_plan(str(out))
    # seq = 256 + 26 = 282 -> 73; seq = 256 + 2048 = 2304 -> floor(20.8/2.327) = 8
    assert [b["batch_size"] for b in filled["1"]] == [73, 8]
    assert [b["max_length"] for b in filled["1"]] == [26, 2048]
    assert "validate" in capsys.readouterr().out


OOM_LOG = """\
[shape] step=0 micro=3 res=1 txt_hi=26 txt_len=26 B=96 latent=32x32 mem_gb=8.57
[shape] step=0 micro=4 res=1 txt_hi=2048 txt_len=2048 B=8 latent=32x32 mem_gb=8.58
Traceback (most recent call last):
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 47.37 GiB of which 1.90 GiB is free. Of the allocated memory 44.94 GiB is allocated by PyTorch, and 61.05 MiB is reserved by PyTorch but unallocated.
"""


def test_parse_oom_log_reads_live_failed_and_last_shape(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(OOM_LOG)
    live, tried, shape = parse_oom_log(str(log))
    assert live == pytest.approx(44.94)
    assert tried == pytest.approx(2.50)
    assert shape == (1, 2048, 8)


def test_parse_oom_log_accepts_mib_allocation(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(OOM_LOG.replace("2.50 GiB", "90.00 MiB", 1))
    _, tried, _ = parse_oom_log(str(log))
    assert tried == pytest.approx(90.0 / 1024.0)


def test_parse_oom_log_requires_shape_attribution(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(OOM_LOG.split("Traceback")[1])
    with pytest.raises(ValueError, match="ARTFLOW_LOG_SHAPES"):
        parse_oom_log(str(log))


def test_parse_oom_log_requires_an_oom(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("all good\n")
    with pytest.raises(ValueError, match="no CUDA OOM"):
        parse_oom_log(str(log))


def test_trim_shrinks_the_named_bucket_by_the_measured_ratio(tmp_path, capsys):
    plan_path = write_plan_file(tmp_path)
    log = tmp_path / "run.log"
    log.write_text(OOM_LOG)
    out = tmp_path / "trimmed.json"
    assert main(["trim", "--plan", plan_path, "--image-tokens", "256",
                 "--log", str(log), "--out", str(out)]) == 0
    trimmed = read_plan(str(out))
    # peak estimate 44.94 + 2.50 = 47.44, ceiling 44.99, margin 2.0:
    # floor(8 * 42.99 / 47.44) = 7
    assert trimmed["1"][1]["batch_size"] == 7
    assert trimmed["1"][0]["batch_size"] == 16  # untouched
    assert "re-run the mixed-stream validation" in capsys.readouterr().out


def test_trim_rejects_a_shape_that_is_not_in_the_plan(tmp_path):
    plan_path = write_plan_file(tmp_path)
    log = tmp_path / "run.log"
    log.write_text(OOM_LOG.replace("txt_hi=2048 txt_len=2048 B=8",
                                   "txt_hi=2048 txt_len=2048 B=16"))
    with pytest.raises(ValueError, match="ran at batch"):
        main(["trim", "--plan", plan_path, "--image-tokens", "256",
              "--log", str(log), "--out", str(tmp_path / "o.json")])


def test_trim_rejects_a_bound_the_plan_does_not_have(tmp_path):
    plan_path = write_plan_file(tmp_path)
    log = tmp_path / "run.log"
    log.write_text(OOM_LOG.replace("txt_hi=2048", "txt_hi=1024"))
    with pytest.raises(ValueError, match="does not belong"):
        main(["trim", "--plan", plan_path, "--image-tokens", "256",
              "--log", str(log), "--out", str(tmp_path / "o.json")])
