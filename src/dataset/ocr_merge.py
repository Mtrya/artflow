"""Fold a transcription of in-image text into a caption.

Calligraphy, inscriptions, signatures and seals carry information a caption
cannot paraphrase: the characters themselves.  An earlier labelling pass stored
them in a separate ``ocr_text`` field, which the training manifest ignored, so
that information never reached the model.  This module moves it into the caption
instead of dropping it.

The raw field mixes two things: text that is part of the artwork, and text that
belongs to the museum's own label strip (accession numbers, album titles) which
the crop removes from the training view.  Lines that look like catalogue data
are therefore dropped, as are lines that are mostly unreadable placeholders.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Lines the museum printed on its label strip, or bare catalogue codes.  The
# training crop excludes that strip, so transcribing it would describe text the
# model never sees.
_CATALOGUE = re.compile(
    r"(故畫|故画|故宮|故宫)\s*[A-Z0-9]|"
    r"^[A-Z]{0,3}\d{5,}[A-Z0-9]*$|"
    r"^\d+$|"
    r"^(册|冊|卷|開|开|頁|页|圖|图)\s*\d+$|"
    r"(册|冊|卷)\s*[0-9一二三四五六七八九十]+\s*$|"
    r"[（(]\s*[0-9一二三四五六七八九十]+\s*[)）]"
)
# A seal or a column of calligraphy is mostly Chinese characters.  Latin letters
# and digits in bulk mean the line is a label or a mis-read code.
_NON_CJK = re.compile(r"[A-Za-z0-9]")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

_SEAL_PREFIX = re.compile(r"^[【\[]?\s*(印|印文|印章|seal)\s*[】\]]?\s*[:：]?\s*", re.IGNORECASE)

_ZH_LEAD = "画面上的文字（按原行款录出）："
_EN_LEAD = "Text visible in the image, transcribed as it appears:"


def _is_cjk(ch: str) -> bool:
    return bool(_CJK.match(ch))


def clean_ocr_lines(text: str, min_cjk: int = 2) -> List[str]:
    """Keep the lines that plausibly transcribe the artwork itself."""
    kept: List[str] = []
    seen = set()
    for raw in (text or "").splitlines():
        line = _SEAL_PREFIX.sub("", raw.strip()).strip(" 　")
        if not line:
            continue
        if _CATALOGUE.search(line):
            continue
        cjk = sum(1 for ch in line if _is_cjk(ch))
        if cjk < min_cjk:
            continue
        if cjk / max(len(line), 1) < 0.5:
            continue
        if line.count("□") > cjk:
            continue
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return kept


def append_ocr_block(caption: str, lines: Iterable[str], language: str) -> str:
    """Append a transcription block to ``caption``, or return it unchanged."""
    lines = [line for line in lines if line.strip()]
    if not lines:
        return caption
    lead = _ZH_LEAD if language == "zh" else _EN_LEAD
    block = lead + "\n" + "\n".join(lines)
    body = (caption or "").rstrip()
    return f"{body}\n\n{block}" if body else block
