import json
from pathlib import Path

from PIL import Image

from scripts.data.synth_manifest import build, read_records


def record(tmp_path, prompt_id, width=64, height=64, **overrides):
    path = tmp_path / "images" / f"{prompt_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), "white").save(path)
    row = {"prompt_id": prompt_id, "image_id": prompt_id, "text": f"prompt {prompt_id}",
           "language": "zh", "family": "photograph", "subject": "hanfu",
           "subject_group": "hanfu", "aspect": "1x1", "width": width, "height": height,
           "recipe": "z-image", "steps": 9, "guidance": 0.0, "path": str(path)}
    row.update(overrides)
    return row


def write_records(tmp_path, rows, shard=0):
    target = tmp_path / "records" / f"generated_shard{shard}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_manifest_rows_carry_the_prompt_as_the_caption(tmp_path):
    write_records(tmp_path, [record(tmp_path, "syn-000000"), record(tmp_path, "syn-000001")])
    write_records(tmp_path, [record(tmp_path, "syn-000002")], shard=1)

    rows = read_records(str(tmp_path / "records"))
    manifest, stats = build(rows)

    assert len(rows) == 3
    assert [entry["image_id"] for entry in manifest] == ["syn-000000", "syn-000001", "syn-000002"]
    assert manifest[0]["captions"] == ["prompt syn-000000"]
    assert manifest[0]["source"] == "d3_synth"
    assert manifest[0]["width"] == 64 and manifest[0]["height"] == 64
    assert stats["aspect 1x1"] == 3


def test_a_row_whose_image_is_missing_or_the_wrong_size_is_reported(tmp_path):
    present = record(tmp_path, "syn-000000")
    absent = record(tmp_path, "syn-000001")
    Path(absent["path"]).unlink()
    wrong = record(tmp_path, "syn-000002", width=32, height=32)
    wrong["width"], wrong["height"] = 64, 64          # grid asked for more than was produced

    manifest, stats = build([present, absent, wrong])

    assert [entry["image_id"] for entry in manifest] == ["syn-000000"]
    assert stats["image missing"] == 1
    assert stats["size mismatch"] == 1


def test_a_repeated_prompt_id_is_counted_once(tmp_path):
    row = record(tmp_path, "syn-000000")

    manifest, stats = build([row, row])

    assert len(manifest) == 1
    assert stats["duplicate record"] == 1
