import json
from pathlib import Path

CANARY = Path(__file__).resolve().parents[1] / "assets/eval/canary_v1.jsonl"

DOMAINS = {"d1_chinese_painting", "d2_western_art", "d3_people", "d4_world"}


def _rows():
    return [json.loads(line) for line in CANARY.read_text().splitlines() if line.strip()]


def test_canary_schema_and_domain_balance():
    rows = _rows()
    assert rows, "canary file is empty"
    for row in rows:
        assert {"id", "domain", "aspect", "seed", "lang", "text"} <= row.keys()
        assert row["domain"] in DOMAINS
        assert row["lang"] in {"zh", "en"}
        assert row["text"].strip()
    per_domain = {d: sum(1 for r in rows if r["domain"] == d) for d in DOMAINS}
    assert len(set(per_domain.values())) == 1, per_domain


def test_canary_seeds_pair_zh_en():
    rows = _rows()
    by_seed = {}
    for row in rows:
        by_seed.setdefault((row["domain"], row["seed"]), set()).add(row["lang"])
    # Every (domain, seed) scene exists in both languages, so a domain
    # regression cannot hide behind a seed/language change.
    assert all(langs == {"zh", "en"} for langs in by_seed.values())
