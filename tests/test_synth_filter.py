"""The synthetic-image filter's verdict parsing, and the rescue selection rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.caption.synth_filter import FLAW_LABELS, parse_verdict  # noqa: E402
from scripts.caption.synth_rescue_selection import (  # noqa: E402
    BAND_SHARES, apportion, band_cycle, caption_language, sample_rows,
)


def test_verdict_parses_with_check_lines_around_it():
    text = ("1. anatomy：无\n2. text：右上角有乱码\n\n"
            '{"ok": false, "flaws": ["text"], "reason": "右上角是乱码伪文字。"}')
    ok, flaws, reason, error = parse_verdict(text)
    assert ok is False
    assert flaws == ["text"]
    assert reason == "右上角是乱码伪文字。"
    assert error is None


def test_verdict_parses_inside_a_code_fence():
    text = '```json\n{"ok": true, "flaws": [], "reason": "无缺陷。"}\n```'
    ok, flaws, reason, error = parse_verdict(text)
    assert (ok, flaws, reason, error) == (True, [], "无缺陷。", None)


def test_the_last_json_object_wins():
    text = ('{"ok": true, "flaws": [], "reason": "中间的对象"}\n'
            '结论：\n{"ok": false, "flaws": ["anatomy", "repetition"], "reason": "手指粘连，人物重复。"}')
    ok, flaws, _, _ = parse_verdict(text)
    assert ok is False
    assert flaws == ["anatomy", "repetition"]


def test_a_string_verdict_and_a_single_flaw_string_are_coerced():
    ok, flaws, _, error = parse_verdict('{"ok": "false", "flaws": "quality", "reason": "拼接块。"}')
    assert ok is False
    assert flaws == ["quality"]
    assert error is None


def test_a_missing_json_is_a_parse_error_not_a_verdict():
    ok, flaws, reason, error = parse_verdict("1. anatomy：无\n\n结论：一切正常。")
    assert ok is None
    assert flaws == []
    assert error is not None


def test_every_label_the_prompt_offers_is_counted():
    assert "repetition" in FLAW_LABELS
    assert set(FLAW_LABELS) == {"anatomy", "orientation", "fusion", "text",
                                "garment", "quality", "repetition",
                                "intrusion", "frame"}


def test_language_follows_the_existing_caption():
    assert caption_language(["一位汉族女性穿交领襦裙"]) == "zh"
    assert caption_language(["A film-grain portrait of a man"]) == "en"
    assert caption_language(["", "English then Chinese 汉服"]) == "zh"
    assert caption_language([]) == "en"


def test_band_apportionment_meets_the_share_exactly():
    assert apportion(60, BAND_SHARES) == {"64-255": 24, "256-511": 15,
                                          "512-895": 15, "896-1280": 6}
    counts = apportion(42, BAND_SHARES)
    assert sum(counts.values()) == 42
    assert counts["64-255"] == 17  # .8 remainder rounds it up, not down


def test_band_cycle_uses_every_quota_and_interleaves():
    counts = apportion(42, BAND_SHARES)
    cycle = band_cycle(counts, [name for name, _ in BAND_SHARES])
    assert len(cycle) == 42
    assert cycle.count("64-255") == counts["64-255"]
    assert cycle.count("896-1280") == counts["896-1280"]
    # The long band is 10% of the rows; they do not arrive as one block.
    long_positions = [i for i, band in enumerate(cycle) if band == "896-1280"]
    assert max(long_positions) - min(long_positions) > 4


def test_sampling_takes_every_333rd_row_of_each_generator():
    rows = [{"image_id": f"syn-{i:06d}",
             "local_path": ("/w/gen-ernie-image-turbo/" if i % 2 else "/w/gen-qwen-image-lightning/")
                           + f"syn-{i:06d}.jpg"} for i in range(1000)]
    sample = sample_rows(rows, stride=333, per_model=3)
    picked = sorted(row["image_id"] for row in sample)
    assert len(picked) == 4  # 3 per generator, but only 2 rows per stride step exist
    assert picked[0] == "syn-000000"
