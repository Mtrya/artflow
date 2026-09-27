"""Resumption state and the queries file of the Pexels fetcher.

The fetch is long-running and resumable, so two properties matter.  The ids
already recorded are skipped, and ``--target`` counts the pictures on disk -
counting every row instead would make the requested "N more downloads" stop
immediately on a resumed run.  The queries file additionally carries an
optional per-term page budget, because the families of terms need different
amounts of the search API.
"""

from scripts.data.fetch_pexels import load_state, make_keep_matcher, parse_queries


def test_load_state_counts_only_downloaded_rows(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        '{"photo_id": 11, "download_ok": true}\n'
        '{"photo_id": 12, "download_ok": false, "skip_reason": "no person in alt text"}\n'
        '{"photo_id": 13, "download_ok": true}\n'
        '{"photo_id": 14, "download_ok": true\n',  # torn last line
        encoding="utf-8",
    )
    done, kept = load_state(path)
    assert done == {11, 12, 13}
    assert kept == 2


def test_load_state_of_missing_file(tmp_path):
    done, kept = load_state(tmp_path / "absent.jsonl")
    assert done == set()
    assert kept == 0


def test_parse_queries_optional_page_budget():
    text = "# body framing\n\nsexy woman | 3\nhanfu\n  \n# comment\n"
    assert parse_queries(text) == [("sexy woman", 3), ("hanfu", None)]


def test_keep_matcher_default_keeps_person_photographs():
    matcher = make_keep_matcher(None)
    assert matcher.search("A woman walking through a market")
    assert not matcher.search("A mountain lake at sunset")


def test_keep_matcher_custom_pattern_keeps_limbs():
    matcher = make_keep_matcher(r"\b(hand|hands|finger|fingers|leg|legs|foot|feet)\b")
    assert matcher.search("Close-up of hands holding a cup")
    assert matcher.search("Crossed legs on a wooden bench")
    assert matcher.search("Hands folded in prayer")  # case-insensitive
    assert not matcher.search("A mountain lake at sunset")
