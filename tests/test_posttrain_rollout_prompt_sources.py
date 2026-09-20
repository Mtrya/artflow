import json
import random

from scripts.posttrain import build_rollout_prompt_sources as bprs


def _manifest(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


def test_read_rows_keeps_only_captioned_rows(tmp_path):
    p = tmp_path / "d4_inat.jsonl"
    _manifest(p, [
        {"image_id": "a", "captions": ["有"]},
        {"image_id": "b", "captions": []},
        {"image_id": "c", "captions": ["", "x"]},
        {"image_id": "d"},
    ])
    assert list(bprs.read_rows(p)) == [
        {"image_id": "a", "captions": ["有"]},
        {"image_id": "c", "captions": ["x"]},
    ]


def test_subsample_caps_deterministically_and_reports_source_size():
    rows = [{"image_id": str(i)} for i in range(100)]
    kept, seen = bprs.subsample(iter(rows), 10, random.Random(0))
    assert seen == 100
    assert len(kept) == 10
    assert kept == bprs.subsample(iter(rows), 10, random.Random(0))[0]
    # A source smaller than the pool is kept whole.
    short, seen = bprs.subsample(iter(rows[:5]), 10, random.Random(0))
    assert (len(short), seen) == (5, 5)


def test_main_stages_one_pool_sized_file_per_source(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _manifest(manifests / "d4_inat.jsonl",
              [{"image_id": f"i{i}", "captions": [f"c{i}", f"alt{i}"]}
               for i in range(10)])
    monkeypatch.setitem(bprs.POOL_ROWS, "d4-inat", 4)
    monkeypatch.setattr("sys.argv", [
        "build_rollout_prompt_sources",
        "--manifest-dir", str(manifests),
        "--out-dir", str(tmp_path / "out"),
        "--only", "d4-inat",
    ])
    bprs.main()
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "d4-inat.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["captions"] for r in rows)
