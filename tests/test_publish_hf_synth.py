import json
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq

from scripts.data.publish_hf_synth import build_rows, write_shards


def record(tmp_path, prompt_id, width=64, height=64, generator="ernie-image-turbo",
           seconds=2.5):
    path = tmp_path / "images" / f"{prompt_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), "white").save(path)
    return {
        "prompt_id": prompt_id, "image_id": prompt_id, "text": f"prompt {prompt_id}",
        "language": "zh", "family": "photograph", "subject": "hanfu-crossed",
        "subject_group": "hanfu", "aspect": "1x1", "width": width, "height": height,
        "recipe": generator, "generator": generator, "steps": 8, "guidance": 1.0,
        "seconds": seconds, "path": str(path),
    }


def test_rows_carry_the_prompt_as_the_caption(tmp_path):
    verdicts = {"syn-000000": {"ok": True, "flaws": []}}
    rows, stats = build_rows([record(tmp_path, "syn-000000")], verdicts, {})
    assert len(rows) == 1
    assert rows[0]["text"] == "prompt syn-000000"
    assert rows[0]["image"]["path"] == "syn-000000.jpg"
    assert rows[0]["image"]["bytes"].startswith(b"\xff\xd8\xff")
    assert not stats


def test_a_missing_picture_is_reported_not_written(tmp_path):
    gone = record(tmp_path, "syn-000001")
    Path(gone["path"]).unlink()
    rows, stats = build_rows([record(tmp_path, "syn-000000"), gone], {}, {})
    assert [row["image_id"] for row in rows] == ["syn-000000"]
    assert stats["image missing"] == 1


def test_one_row_per_picture(tmp_path):
    rows, stats = build_rows([record(tmp_path, "syn-000000"), record(tmp_path, "syn-000000")], {}, {})
    assert len(rows) == 1
    assert stats["duplicate record"] == 1


def test_rows_name_the_model_behind_the_recipe(tmp_path):
    # Generation records carry the recipe that ran, not the model's name.
    entry = record(tmp_path, "syn-000000", generator="qwen-image-lightning")
    entry.pop("generator")
    rows, _stats = build_rows([entry], {}, {})
    assert rows[0]["generator"] == "Qwen-Image with the 8-step distilled sampler"


def test_an_unknown_recipe_is_passed_through(tmp_path):
    entry = record(tmp_path, "syn-000000", generator="something-else")
    entry.pop("generator")
    rows, _stats = build_rows([entry], {}, {})
    assert rows[0]["generator"] == "something-else"


def test_shards_are_named_for_the_reader_and_hold_the_bytes(tmp_path):
    records = [record(tmp_path, f"syn-{index:06d}") for index in range(5)]
    rows, _stats = build_rows(records, {}, {})
    # Let each shard take two pictures, so the split is decided by the test and
    # not by how well a solid-colour JPEG happens to compress.  A shard closes
    # once it reaches the limit, so the limit has to land inside the second row.
    limit_bytes = 1.5 * len(rows[0]["image"]["bytes"])
    shards = write_shards(rows, tmp_path / "data", shard_gb=limit_bytes / 1024 ** 3,
                          row_group_rows=2)
    assert [path.name for path in shards] == [
        "train-00000-of-00003.parquet", "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
    ]
    table = pq.read_table(shards[0])
    assert table.num_rows == 2
    assert table.column("text").to_pylist() == ["prompt syn-000000", "prompt syn-000001"]
    payload = table.column("image").to_pylist()[0]
    assert payload["bytes"].startswith(b"\xff\xd8\xff")
    assert b"huggingface" in (table.schema.metadata or {})


def test_rows_carry_the_rescue_outcome(tmp_path):
    records = [record(tmp_path, "syn-000000"), record(tmp_path, "syn-000001")]
    verdicts = {"syn-000000": {"ok": True, "flaws": []},
                "syn-000001": {"ok": False, "flaws": ["frame", "anatomy"]}}
    rescued = {"syn-000000": "一位穿交领襦裙的女子站在庭院里。"}
    rows, _stats = build_rows(records, verdicts, rescued)
    assert rows[0]["rescue_ok"] is True
    assert rows[0]["rescue_flaws"] == []
    assert rows[0]["caption_rescued"] == "一位穿交领襦裙的女子站在庭院里。"
    assert rows[1]["rescue_ok"] is False
    assert rows[1]["rescue_flaws"] == ["frame", "anatomy"]
    assert rows[1]["caption_rescued"] == ""
