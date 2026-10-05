"""Success budgets refill within a page and preserve its cursor across runs."""

import importlib
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.data.fetch_reinforce_common import download_image


@pytest.fixture(params=["commons", "inat", "gbif", "openverse", "museum"])
def fetcher(request, tmp_path, monkeypatch):
    name = request.param
    module = importlib.import_module(f"scripts.data.fetch_{name}")
    queries = tmp_path / "queries.txt"
    queries.write_text("subject | 10\n")
    flags = [
        "--out",
        str(tmp_path / "out"),
        "--queries",
        str(queries),
        "--target",
        "1",
        "--workers",
        "3",
    ]
    if name == "museum":
        flags += ["--apis", "aic"]
    args = module.build_parser().parse_args(flags)
    monkeypatch.setattr(module.Pacer, "wait", lambda self: None)
    monkeypatch.setattr(module.Pacer, "note", lambda *args: None)
    monkeypatch.setattr(module, "thread_session", lambda: None)
    requests = []
    ids = range(1, 5)

    def record(i):
        prefix = {"commons": "com-", "inat": "inat-", "museum": "aic-"}.get(name, "")
        return dict(
            source_id=prefix + str(i),
            query="subject",
            source_width=1000,
            source_height=1000,
            image_url=str(i),
            original_url=str(i),
            license="cc0",
        )

    def page(*args, **kwargs):
        requests.append((args, kwargs))
        if name == "commons":
            return {"query": {"pages": {str(i): {"pageid": i} for i in ids}}}
        if name == "inat":
            return {
                "results": [{"taxon": {"id": 1}, "photos": [{"id": i} for i in ids]}],
                "total_results": 1,
            }
        if name == "gbif":
            return {
                "results": [{"media": [{"identifier": str(i)}]} for i in ids],
                "endOfRecords": True,
            }
        if name == "openverse":
            return {"results": [{"id": i} for i in ids], "page_count": 1}
        return {
            "data": [
                {"id": i, "image_id": str(i), "is_public_domain": True} for i in ids
            ],
            "config": {"iiif_url": "https://example.invalid"},
            "pagination": {"total_pages": 1},
        }

    if name == "commons":
        monkeypatch.setattr(module, "get_json", page)
        monkeypatch.setattr(
            module, "candidate_from_page", lambda term, item, *a: record(item["pageid"])
        )
    elif name == "inat":
        monkeypatch.setattr(module, "observations", page)
        monkeypatch.setattr(
            module, "record_for", lambda term, obs, photo, *a: record(photo["id"])
        )
    elif name == "gbif":
        monkeypatch.setattr(module, "match_taxon", lambda *a: {"usageKey": 1})
        monkeypatch.setattr(module, "get_json", page)
        monkeypatch.setattr(
            module,
            "candidate_from_occurrence",
            lambda term, item, idx, media, *a: record(media["identifier"]),
        )
    elif name == "openverse":
        monkeypatch.setattr(module, "get_json", page)
        monkeypatch.setattr(
            module, "candidate_from_result", lambda term, item, *a: record(item["id"])
        )
    else:
        monkeypatch.setattr(module, "aic_search", page)
        monkeypatch.setattr(
            module, "aic_record", lambda term, item, *a: record(item["id"])
        )
        monkeypatch.setattr(module, "aic_image_url", lambda prefix, image_id: image_id)
    return module, args, queries, requests


@pytest.mark.parametrize("limit", ["target", "term"])
def test_last_page_refills_failures_and_resumes_unattempted_candidates(
    fetcher, monkeypatch, limit
):
    module, args, queries, requests = fetcher
    if limit == "term":
        args.target = 10
        queries.write_text("subject | 1\n")
    attempts = []

    def download(session, url, destination, **kwargs):
        attempts.append(url)
        return {"download_ok": url != "1", "width": 1000, "height": 1000}

    monkeypatch.setattr(module, "download_image", download)
    module.harvest(args, None)
    assert attempts == ["1", "2"]
    assert len(requests) == 1
    # Stopping at the budget on the final page must not mark the page exhausted.
    args.target = 10
    queries.write_text("subject | 10\n")
    module.harvest(args, None)
    assert sorted(attempts) == ["1", "2", "3", "4"]
    assert len(requests) == 2
    from pathlib import Path

    records = [
        json.loads(line)
        for line in (Path(args.out) / "metadata.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    assert sum(row["download_ok"] for row in records) == 3
    module.harvest(args, None)
    assert len(requests) == 2  # Now genuinely exhausted.


def test_interruption_resumes_page_after_recorded_success(fetcher, monkeypatch):
    module, args, _, _ = fetcher
    args.target, args.workers = 10, 1
    attempts = []

    def interrupted(session, url, destination, **kwargs):
        attempts.append(url)
        if url == "2":
            raise RuntimeError("interrupted")
        return {"download_ok": True}

    monkeypatch.setattr(module, "download_image", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.harvest(args, None)
    assert attempts == ["1", "2"]
    monkeypatch.setattr(
        module,
        "download_image",
        lambda session, url, destination, **kwargs: attempts.append(url)
        or {"download_ok": True},
    )
    module.harvest(args, None)
    assert attempts == ["1", "2", "2", "3", "4"]


@pytest.mark.parametrize(
    "shape,expected",
    [((800, 1600), (500, 1000)), ((1600, 800), (1000, 500)), ((300, 600), (300, 600))],
)
@pytest.mark.parametrize("existing", [False, True])
def test_download_limits_longest_side_without_upscaling(
    tmp_path, shape, expected, existing
):
    pixels = np.random.default_rng(4).integers(
        0, 256, (shape[1], shape[0], 3), dtype=np.uint8
    )
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
    dest = tmp_path / "image.jpg"
    if existing:
        dest.write_bytes(buffer.getvalue())

    def get(*args, **kwargs):
        assert not existing
        return SimpleNamespace(content=buffer.getvalue(), raise_for_status=lambda: None)

    result = download_image(
        SimpleNamespace(get=get), "https://example.invalid", dest, box=1000
    )
    assert result["download_ok"]
    assert (result["width"], result["height"]) == expected
    with Image.open(dest) as image:
        assert image.size == expected
