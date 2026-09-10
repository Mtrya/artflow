"""Row-budget allocation for caption enrichment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.caption.select_rows import Row, allocate, dataset_domain  # noqa: E402


def _row(manifest, source, image_id):
    return Row(image_id=image_id, source=source, manifest=manifest,
               local_path=f"/x/{image_id}.jpg", captions=["a caption"],
               width=1000, height=800, bbox=None)


def test_dataset_domain_reads_the_rows_not_the_name():
    rows = [
        _row("d1", "d1_npm_tw", "a"),
        _row("d1", "d1_met_china", "b"),
        _row("d4-relaion", "d4_relaion", "c"),
    ]
    domains = dataset_domain(rows)
    assert domains["d1"] == "chinese_painting"
    assert domains["d4-relaion"] == "generic"


def test_specialised_share_lands_on_chinese_painting():
    mix = {"d1": 0.25, "d4-relaion": 0.75}
    available = {"d1": 100_000, "d4-relaion": 100_000}
    domains = {"d1": "chinese_painting", "d4-relaion": "generic"}

    budgets = allocate(1000, mix, 0.30, available, domains)

    # Broad part by draw share, all of the specialised part on d1.
    assert budgets["d1"] == round(700 * 0.25) + 300
    assert budgets["d4-relaion"] == round(700 * 0.75)
    assert sum(budgets.values()) == 1000


def test_specialised_share_is_spread_when_nothing_is_chinese():
    mix = {"d2-wikiart": 0.5, "d4-relaion": 0.5}
    available = {name: 100_000 for name in mix}
    domains = {"d2-wikiart": "western_art", "d4-relaion": "generic"}

    budgets = allocate(1000, mix, 0.30, available, domains)

    assert budgets["d2-wikiart"] == 500
    assert budgets["d4-relaion"] == 500
    assert sum(budgets.values()) == 1000


def test_a_capped_dataset_returns_its_shortfall_to_the_rest():
    mix = {"d1": 0.25, "d4-relaion": 0.75}
    available = {"d1": 100, "d4-relaion": 100_000}
    domains = {"d1": "chinese_painting", "d4-relaion": "generic"}

    budgets = allocate(1000, mix, 0.30, available, domains)

    assert budgets["d1"] == 100
    assert budgets["d4-relaion"] == 900
