"""Folding a transcription of in-image text into a caption."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.ocr_merge import append_ocr_block, clean_ocr_lines  # noqa: E402


def test_catalogue_lines_are_dropped():
    text = "故畫 003655N000000008\n毛詩品物圖考 (三) 册 8\n流離之子"
    assert clean_ocr_lines(text) == ["流離之子"]


def test_bare_codes_and_numeric_lines_are_dropped():
    assert clean_ocr_lines("12345\nK2A003655N000000000PAW\n1234567") == []


def test_seal_prefix_is_stripped_and_seal_text_kept():
    assert clean_ocr_lines("【印】乾隆御覽之寶\n【印】") == ["乾隆御覽之寶"]


def test_mostly_unreadable_lines_are_dropped():
    assert clean_ocr_lines("□□□□□□□\n□□印") == []


def test_duplicate_lines_are_kept_once():
    assert clean_ocr_lines("石渠寶笈\n石渠寶笈") == ["石渠寶笈"]


def test_append_block_uses_the_caption_language():
    zh = append_ocr_block("山水立軸。", ["石渠寶笈"], "zh")
    en = append_ocr_block("A hanging scroll.", ["石渠寶笈"], "en")
    assert zh.endswith("画面上的文字（按原行款录出）：\n石渠寶笈")
    assert en.endswith("Text visible in the image, transcribed as it appears:\n石渠寶笈")


def test_append_block_without_lines_is_a_no_op():
    assert append_ocr_block("caption", [], "zh") == "caption"
    assert append_ocr_block("caption", ["", "  "], "zh") == "caption"
