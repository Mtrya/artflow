import json

import pyarrow.parquet as pq
import pytest

from scripts.data.publish_hf_d1 import upload_release, write_release


def fixture_rows(tmp_path):
    image = tmp_path / "painting.jpg"
    image.write_bytes(b"original image bytes")
    metadata = tmp_path / "metadata.jsonl"
    rows = [{"image_id": str(i), "local_path": str(image), "shard": "museum",
             "source": "museum", "caption_zh": "画作\n题字", "caption_en": "painting",
             "bbox": "[1,2,999,998]", "artifacts": ["frame"]} for i in range(3)]
    metadata.write_text("\n".join(json.dumps(row) for row in rows))
    captions = tmp_path / "captions.jsonl"
    captions.write_text("\n".join(json.dumps(row) for row in [
        {"image_id": "0", "text": "old", "accepted": True},
        {"image_id": "0", "text": "latest", "accepted": True},
        {"image_id": "0", "text": "bad retry", "accepted": False,
         "reject_reasons": ["repeated sentences"]},
        {"image_id": "1", "text": "length-only", "accepted": False,
         "reject_reasons": ["length 16 outside requested band"]},
    ]))
    return metadata, captions, image


def test_release_metadata_and_image_rows_agree(tmp_path):
    metadata, captions, image = fixture_rows(tmp_path)
    out = tmp_path / "release"
    stats = write_release(metadata, captions, out, images=True, shard_gb=1e-9,
                          row_group_rows=1)
    assert stats["rows"] == 3 and stats["image_shards"] == 3
    table = pq.read_table(out / "metadata/metadata.parquet").to_pylist()
    assert "local_path" not in table[0]
    assert table[0]["caption_long"] == "latest"
    assert table[1]["caption_long"] == "length-only"
    assert table[0]["bbox"] == [1, 2, 999, 998]
    for row, path in zip(table, sorted((out / "default/train").glob("*.parquet"))):
        shard = pq.read_table(path)
        record = shard.to_pylist()[0]
        assert record.pop("image") == {"bytes": image.read_bytes(), "path": image.name}
        assert record == row
        features = json.loads(shard.schema.metadata[b"huggingface"])["info"]["features"]
        assert features["image"]["_type"] == "Image"
    with pytest.raises(ValueError, match="empty"):
        write_release(metadata, captions, out)


def test_metadata_only_needs_no_local_images(tmp_path):
    metadata, captions, image = fixture_rows(tmp_path)
    image.unlink()
    out = tmp_path / "metadata_only"
    assert write_release(metadata, captions, out)["image_shards"] == 0
    assert not (out / "default").exists()
    with pytest.raises(FileNotFoundError, match="image 0"):
        write_release(metadata, captions, tmp_path / "full", images=True)


def test_explicit_missing_captions_and_empty_metadata_fail(tmp_path):
    metadata, _, _ = fixture_rows(tmp_path)
    with pytest.raises(FileNotFoundError):
        write_release(metadata, tmp_path / "missing.jsonl", tmp_path / "release")
    metadata.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        write_release(metadata, None, tmp_path / "release")


@pytest.mark.parametrize("images", [False, True])
def test_upload_replaces_only_the_requested_artifacts(tmp_path, monkeypatch, images):
    import huggingface_hub

    calls = []
    class Api:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    upload_release(tmp_path, "owner/d1", images=images)
    assert calls[0]["repo_id"] == "owner/d1"
    assert calls[0]["delete_patterns"] == (["default/train/*.parquet"] if images else None)
