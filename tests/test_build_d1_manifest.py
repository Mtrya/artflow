"""The Chinese-painting manifest must carry the published crop and captions.

The published table is the version of record, and the earlier manifest predates
it: every crop box in that one differs from the published box.  These tests pin
the two things the rebuild decides - which captions a row ends up with, and that
a row whose photograph is not on the shared disk is reported rather than written
with a path that points nowhere.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data.build_d1_manifest import crop_box, read_long_captions, write_manifest


def write_metadata(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def metadata_row(image_id, **overrides):
    # bbox is a JSON string in the published table, not a list.
    row = {"image_id": image_id, "shard": "npm_tw_c0", "source": "npm_tw",
           "object_no": "故畫000001", "title": "title", "artist": "artist",
           "category": "繪畫", "culture": "chinese", "view_type": "full",
           "caption_zh": "中文描述", "caption_en": "an English description",
           "ocr_text": "", "artifacts": "[]", "bbox": None}
    row.update(overrides)
    return row


def place_image(root, image_id, shard="npm_tw_c0"):
    path = root / shard / "images" / f"{image_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")
    return path


def test_long_caption_is_added_after_the_short_ones(tmp_path):
    place_image(tmp_path / "clean", "img-1")
    write_metadata(tmp_path / "meta.parquet",
                   [metadata_row("img-1", bbox="[10, 20, 30, 40]")])
    write_long = tmp_path / "long.jsonl"
    write_long.write_text(
        json.dumps({"image_id": "img-1", "text": "a long caption", "accepted": True}) + "\n"
        + json.dumps({"image_id": "img-2", "text": "rejected", "accepted": False}) + "\n",
        encoding="utf-8")

    stats = write_manifest(str(tmp_path / "meta.parquet"), str(tmp_path / "clean"),
                           read_long_captions(str(write_long)), str(tmp_path / "out.jsonl"))

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert stats["rows"] == 1
    assert stats["1 captions"] == 0
    assert stats["3 captions"] == 1
    assert stats["with crop box"] == 1
    assert rows[0]["captions"] == ["中文描述", "an English description", "a long caption"]
    assert rows[0]["source"] == "d1_npm_tw"
    assert rows[0]["bbox"] == [10, 20, 30, 40]


def test_a_row_without_its_photograph_is_reported_not_written(tmp_path):
    place_image(tmp_path / "clean", "img-1")
    write_metadata(tmp_path / "meta.parquet",
                   [metadata_row("img-1"), metadata_row("img-missing")])
    write_long = tmp_path / "long.jsonl"
    write_long.write_text("", encoding="utf-8")

    stats = write_manifest(str(tmp_path / "meta.parquet"), str(tmp_path / "clean"),
                           read_long_captions(str(write_long)), str(tmp_path / "out.jsonl"))

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [row["image_id"] for row in rows] == ["img-1"]
    assert stats["image missing"] == 1


def test_a_row_without_any_caption_is_reported_not_written(tmp_path):
    place_image(tmp_path / "clean", "img-1")
    write_metadata(tmp_path / "meta.parquet",
                   [metadata_row("img-1", caption_zh=None, caption_en=None)])
    write_long = tmp_path / "long.jsonl"
    write_long.write_text("", encoding="utf-8")

    stats = write_manifest(str(tmp_path / "meta.parquet"), str(tmp_path / "clean"),
                           read_long_captions(str(write_long)), str(tmp_path / "out.jsonl"))

    assert stats["rows"] == 0
    assert stats["no captions"] == 1


def test_the_published_crop_box_is_a_string_and_must_be_parsed():
    assert crop_box("[0, 252, 1000, 824]") == [0.0, 252.0, 1000.0, 824.0]
    assert crop_box("[]") is None
    assert crop_box(None) is None
    assert crop_box([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]


def test_a_parsed_crop_box_reaches_the_manifest(tmp_path):
    place_image(tmp_path / "clean", "img-1")
    write_metadata(tmp_path / "meta.parquet",
                   [metadata_row("img-1", bbox="[0, 252, 1000, 824]")])
    write_long = tmp_path / "long.jsonl"
    write_long.write_text("", encoding="utf-8")

    write_manifest(str(tmp_path / "meta.parquet"), str(tmp_path / "clean"),
                   read_long_captions(str(write_long)), str(tmp_path / "out.jsonl"))

    row = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
    assert row["bbox"] == [0.0, 252.0, 1000.0, 824.0]
