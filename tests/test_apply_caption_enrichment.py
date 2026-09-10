"""The enrichment merge must caption the row it thinks it is captioning.

A saved dataset stores no image id, only a row position, so the merge recovers
the image id by matching a row's captions back to the manifest that produced
the dataset.  Precompute drops rows that fail its filters, so the stored rows
are a subsequence of the manifest rather than the same list, and a merge that
compared positions one for one would caption the wrong pictures from the first
dropped row onward.  These tests pin the walk, the outcome for the three kinds
of row (caption available, caption rejected, no caption), and the refusal to
run when the manifest does not describe the data at all.
"""

import json

from datasets import Dataset

from scripts.data.apply_caption_enrichment import apply
from src.dataset.captions import clean_caption


def save_dataset(path, captions):
    """A saved dataset shaped like the precomputed ones.

    Captions are stored as precompute leaves them: every string has been through
    ``clean_caption``.  Building the fixture any other way would test a
    comparison the real data never sees.
    """
    dataset = Dataset.from_dict({
        "captions": [[clean_caption(caption) for caption in row] for row in captions],
        "latents": [[[[float(i), 0.5]]] for i in range(len(captions))],
        "resolution_bucket_id": [0] * len(captions),
    })
    dataset.save_to_disk(str(path))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest_rows(captions, ids=None):
    ids = ids or [f"img-{index}" for index in range(len(captions))]
    return [
        {"image_id": image_id, "source": "src", "captions": row,
         "local_path": f"/images/{image_id}.jpg", "width": 1024, "height": 1024,
         "bbox": None}
        for image_id, row in zip(ids, captions)
    ]


def caption_table(path, entries):
    write_jsonl(path, [
        {"image_id": image_id, "text": text, "accepted": accepted}
        for image_id, text, accepted in entries
    ])


def test_appends_enriched_caption_and_keeps_latents(tmp_path):
    captions = [["a short caption"], ["another short one"], ["third"]]
    save_dataset(tmp_path / "ds", captions)
    write_jsonl(tmp_path / "manifest.jsonl", manifest_rows(captions))
    caption_table(tmp_path / "captions.jsonl",
                  [("img-0", "a much longer caption", True),
                   ("img-1", "rejected caption", False)])

    summary = apply(str(tmp_path / "ds"), str(tmp_path / "manifest.jsonl"),
                    str(tmp_path / "captions.jsonl"), str(tmp_path / "out"))

    merged = Dataset.load_from_disk(str(tmp_path / "out"))
    assert summary["enriched"] == 1
    assert merged["captions"] == [["A short caption", "a much longer caption"],
                                  ["Another short one"],
                                  ["Third"]]
    # Latents are copied, not recomputed.
    original = Dataset.load_from_disk(str(tmp_path / "ds"))
    assert merged["latents"] == original["latents"]
    assert merged["resolution_bucket_id"] == original["resolution_bucket_id"]


def test_alignment_survives_rows_the_precompute_dropped(tmp_path):
    manifest = manifest_rows([["first picture"], ["a dropped row"], ["last picture"]],
                             ids=["img-0", "img-dropped", "img-2"])
    save_dataset(tmp_path / "ds", [["first picture"], ["last picture"]])
    write_jsonl(tmp_path / "manifest.jsonl", manifest)
    caption_table(tmp_path / "captions.jsonl",
                  [("img-0", "longer for the first", True),
                   ("img-dropped", "longer for the dropped", True),
                   ("img-2", "longer for the last", True)])

    apply(str(tmp_path / "ds"), str(tmp_path / "manifest.jsonl"),
          str(tmp_path / "captions.jsonl"), str(tmp_path / "out"))

    merged = Dataset.load_from_disk(str(tmp_path / "out"))
    assert merged["captions"] == [["First picture", "longer for the first"],
                                  ["Last picture", "longer for the last"]]


def test_repeated_captions_take_manifest_positions_in_order(tmp_path):
    # Two pictures share a caption.  The walk must use them in the order they
    # appear, so each picture keeps the enrichment meant for it.
    taken = ["picture of a cat", "picture of a cat"]
    save_dataset(tmp_path / "ds", [[caption] for caption in taken])
    write_jsonl(tmp_path / "manifest.jsonl",
                manifest_rows([[caption] for caption in taken], ids=["img-a", "img-b"]))
    caption_table(tmp_path / "captions.jsonl",
                  [("img-a", "the tabby cat", True), ("img-b", "the black cat", True)])

    apply(str(tmp_path / "ds"), str(tmp_path / "manifest.jsonl"),
          str(tmp_path / "captions.jsonl"), str(tmp_path / "out"))

    merged = Dataset.load_from_disk(str(tmp_path / "out"))
    assert merged["captions"] == [["Picture of a cat", "the tabby cat"],
                                  ["Picture of a cat", "the black cat"]]


def test_stops_when_the_manifest_does_not_describe_the_rows(tmp_path):
    save_dataset(tmp_path / "ds", [["one"], ["two"]])
    write_jsonl(tmp_path / "manifest.jsonl",
                manifest_rows([["completely different"], ["also different"]]))
    caption_table(tmp_path / "captions.jsonl", [("img-0", "longer", True)])

    try:
        apply(str(tmp_path / "ds"), str(tmp_path / "manifest.jsonl"),
              str(tmp_path / "captions.jsonl"), str(tmp_path / "out"))
    except ValueError as error:
        assert "could be matched" in str(error)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected the merge to refuse an unrelated manifest")


def test_manifest_mode_appends_without_needing_a_dataset(tmp_path):
    from scripts.data.apply_caption_enrichment import add_captions_to_manifest

    write_jsonl(tmp_path / "manifest.jsonl",
                manifest_rows([["an alt text"], ["another alt text"]],
                              ids=["pexels-1", "pexels-2"]))
    caption_table(tmp_path / "captions.jsonl",
                  [("pexels-1", "a much longer caption", True),
                   ("pexels-2", "rejected", False)])

    stats = add_captions_to_manifest(str(tmp_path / "manifest.jsonl"),
                                     str(tmp_path / "captions.jsonl"),
                                     str(tmp_path / "out.jsonl"))

    rows = [json.loads(line) for line in
            (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats == {"rows": 2, "enriched": 1}
    assert rows[0]["captions"] == ["an alt text", "a much longer caption"]
    assert rows[1]["captions"] == ["another alt text"]


def test_merging_twice_does_not_duplicate_the_caption(tmp_path):
    captions = [["a short caption"], ["another short one"]]
    save_dataset(tmp_path / "ds", captions)
    write_jsonl(tmp_path / "manifest.jsonl", manifest_rows(captions))
    caption_table(tmp_path / "captions.jsonl", [("img-0", "the long caption", True)])

    apply(str(tmp_path / "ds"), str(tmp_path / "manifest.jsonl"),
          str(tmp_path / "captions.jsonl"), str(tmp_path / "first"))
    apply(str(tmp_path / "first"), str(tmp_path / "manifest.jsonl"),
          str(tmp_path / "captions.jsonl"), str(tmp_path / "second"))

    merged = Dataset.load_from_disk(str(tmp_path / "second"))
    assert merged["captions"][0] == ["A short caption", "the long caption"]
