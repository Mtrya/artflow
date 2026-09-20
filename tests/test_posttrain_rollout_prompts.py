import json

import pytest

from scripts.posttrain.build_rollout_prompts import allocate, build_pool


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


def test_allocate_proportional_to_weight_times_size():
    counts = allocate({"a": 100, "b": 100}, {"a": 2.0, "b": 1.0}, 90)
    assert counts == {"a": 60, "b": 30}
    # Bigger pool with equal weight gets more samples.
    counts = allocate({"a": 300, "b": 100}, {}, 100)
    assert counts == {"a": 75, "b": 25}
    # Largest remainder fills rounding slack to exactly `total`.
    counts = allocate({"a": 10, "b": 10, "c": 10}, {}, 100)
    assert sum(counts.values()) == 100
    with pytest.raises(ValueError):
        allocate({"a": 10}, {"a": 0.0}, 5)
    with pytest.raises(ValueError):
        allocate({"a": 10}, {}, 0)


def test_build_pool_deterministic_and_tagged(tmp_path):
    rows_a = [{"caption_zh": f"甲{i}", "caption_en": f"a{i}"} for i in range(20)]
    rows_b = [{"caption_long": f"long b{i}"} for i in range(20)]
    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    _write_jsonl(pa, rows_a)
    _write_jsonl(pb, rows_b)
    kw = dict(total=16, seed=7)
    pool1 = build_pool({"a": pa, "b": pb}, {"a": 1.0, "b": 3.0}, **kw)
    pool2 = build_pool({"a": pa, "b": pb}, {"a": 1.0, "b": 3.0}, **kw)
    assert pool1 == pool2
    assert len(pool1) == 16
    assert sum(1 for r in pool1 if r["source"] == "b") == 12
    assert all(r["prompt"] and r["field"] for r in pool1)
    # Source b only has caption_long; every b row must use it.
    assert all(r["field"] == "caption_long" for r in pool1 if r["source"] == "b")


def test_build_pool_skips_rows_without_captions(tmp_path):
    rows = [{"caption_zh": "有"}, {"note": "no captions"}]
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, rows)
    pool = build_pool({"x": p}, {}, 1, seed=0)
    assert all(r["prompt"] == "有" for r in pool)


def test_build_pool_reads_captions_list_format(tmp_path):
    # Precomputed training manifests carry {"captions": [str, ...]}.
    rows = [{"captions": [f"c{i}", f"alt{i}"]} for i in range(10)]
    p = tmp_path / "m.jsonl"
    _write_jsonl(p, rows)
    pool = build_pool({"m": p}, {}, 5, seed=0)
    assert len(pool) == 5
    assert all(r["field"] == "captions" for r in pool)


def test_build_pool_rejects_source_without_any_captions(tmp_path):
    p = tmp_path / "empty.jsonl"
    _write_jsonl(p, [{"note": "no captions"}, {"other": 1}])
    with pytest.raises(ValueError, match="no usable captions"):
        build_pool({"x": p}, {}, 4)


def test_build_pool_redistributes_exhausted_allocation(tmp_path):
    # Heavy weight on a tiny source: allocation asks for more than it holds;
    # the shortfall must move to other sources, not silently vanish.
    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    _write_jsonl(pa, [{"caption_zh": f"甲{i}"} for i in range(10)])
    _write_jsonl(pb, [{"caption_en": f"b{i}"} for i in range(1000)])
    pool = build_pool({"a": pa, "b": pb}, {"a": 100.0, "b": 1.0}, 100, seed=0)
    assert len(pool) == 100
    assert sum(1 for r in pool if r["source"] == "a") == 10
    assert sum(1 for r in pool if r["source"] == "b") == 90


def test_build_pool_raises_when_total_exceeds_capacity(tmp_path):
    p = tmp_path / "tiny.jsonl"
    _write_jsonl(p, [{"caption_zh": "只有一条"}])
    with pytest.raises(ValueError, match="usable rows"):
        build_pool({"x": p}, {}, 5)
