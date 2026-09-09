"""Prompts for long-caption enrichment, and the checks applied to their output.

The corpus being enriched already contains one short caption per row.  This
module builds the requests that add a second, much longer caption, and defines
what "acceptable" means for the result.

Length is specified in the units a language model obeys most reliably (words
for English, characters for Chinese) and then measured in the units training
actually consumes: retained tokens under the frozen Qwen3-0.6B tokenizer and
the prompt contract in ``src/utils/prompt_contract.py``.  The conversion
factors below were measured on 200 existing captions of this corpus:

    English    0.757 words  per retained token
    Chinese    1.21  characters per retained token

They are a starting point for the request, never a substitute for measuring the
answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROMPT_VERSION = "cap-v1"

# Retained-token bands from the plan.  A request targets the middle of its band
# and is accepted anywhere inside it.
LENGTH_BANDS = {
    "256-511": (256, 511),
    "512-1023": (512, 1023),
    "1024-1535": (1024, 1535),
    "1536-2048": (1536, 2048),
}
BAND_NAMES = tuple(LENGTH_BANDS)

WORDS_PER_TOKEN = {"en": 0.757, "zh": 1.21}

LANGUAGES = ("en", "zh")
FORMATS = ("prose", "structured")

# Openers the existing corpus already forbids; a caption that starts with one
# is rejected rather than trimmed, because the rest of the sentence is usually
# built around it.
BOILERPLATE_OPENERS = (
    "this image", "the image", "this picture", "the picture", "this painting shows",
    "the painting shows", "in this image", "the artwork depicts", "this artwork",
    "这张图片", "这幅图片", "这幅画", "该作品", "这是一幅", "图中", "画面中",
)


@dataclass
class CaptionRequest:
    """One caption to generate for one image."""

    image_id: str
    language: str
    format: str
    band: str
    domain: str
    metadata: Dict[str, str] = field(default_factory=dict)
    extra_notes: str = ""

    def __post_init__(self) -> None:
        if self.language not in LANGUAGES:
            raise ValueError(f"unsupported language {self.language!r}")
        if self.format not in FORMATS:
            raise ValueError(f"unsupported format {self.format!r}")
        if self.band not in LENGTH_BANDS:
            raise ValueError(f"unsupported length band {self.band!r}")

    @property
    def low_tokens(self) -> int:
        return LENGTH_BANDS[self.band][0]

    @property
    def high_tokens(self) -> int:
        return LENGTH_BANDS[self.band][1]

    @property
    def target_tokens(self) -> int:
        return (self.low_tokens + self.high_tokens) // 2

    def target_units(self) -> int:
        return int(round(self.target_tokens * WORDS_PER_TOKEN[self.language]))

    def unit_range(self) -> tuple:
        return (
            int(self.low_tokens * WORDS_PER_TOKEN[self.language] * 0.95),
            int(self.high_tokens * WORDS_PER_TOKEN[self.language] * 1.05),
        )

    @property
    def prompt_version(self) -> str:
        return f"{PROMPT_VERSION}-{self.language}-{self.format}"

    def prompt(self) -> str:
        return _ZH_TEMPLATE.format(**self._fields()) if self.language == "zh" \
            else _EN_TEMPLATE.format(**self._fields())

    def _fields(self) -> Dict[str, str]:
        low, high = self.unit_range()
        return {
            "unit": "words" if self.language == "en" else "字",
            "target": str(self.target_units()),
            "low": str(low),
            "high": str(high),
            "format_rules": _STRUCTURED_RULES[self.language] if self.format == "structured"
            else _PROSE_RULES[self.language],
            "domain_rules": _DOMAIN_RULES[self.language].get(
                self.domain, _DOMAIN_RULES[self.language]["generic"]),
            "metadata_block": self._metadata_block(),
            "extra_notes": self.extra_notes.strip(),
        }

    def _metadata_block(self) -> str:
        if not self.metadata:
            return _METADATA_ABSENT[self.language]
        lines = [f"- {key}: {value}" for key, value in self.metadata.items() if value]
        if not lines:
            return _METADATA_ABSENT[self.language]
        return _METADATA_PRESENT[self.language].format(lines="\n".join(lines))


_EN_TEMPLATE = """You are writing training captions for a text-to-image model. \
Write one caption for the single image attached to this message.

Length: about {target} {unit}. Acceptable range {low}-{high} {unit}. This is a \
hard requirement: a caption outside the range is unusable. Reach the length by \
describing more of what is actually there, never by repeating yourself, listing \
synonyms, or padding with generic praise.

{format_rules}

Grounding rules:
- Every statement must be visible in this image or come from the metadata below.
- Do not invent an artist, title, date, place, collection history, symbolism, \
or the maker's intention.
- Do not guess at what is outside the frame, and do not describe a different \
image from the one attached.
- If something is unclear or ambiguous, say so briefly instead of choosing a \
specific answer.
- Do not open with "This image", "The image", "This painting", or any similar \
phrase. Start with the content itself.

{domain_rules}
{metadata_block}{extra_notes}
Output only the caption text. No headings, no preamble, no closing remarks."""


_ZH_TEMPLATE = """你为文生图模型撰写训练用的图像描述。请为随本条消息附上的这张图写一条描述。

长度：约 {target} {unit}。可接受范围 {low}–{high} {unit}。这是硬性要求，超出范围的描述无法使用。\
请通过描述画面中真实存在的内容来达到长度，不要靠重复、堆砌近义词或空泛的赞美凑字数。

{format_rules}

依据要求：
- 每一句话都必须来自这张图上可见的内容，或来自下方给出的元数据。
- 不要编造作者、标题、年代、地点、收藏史、象征含义或创作意图。
- 不要猜测画面之外的内容，也不要描述与附图无关的另一张图。
- 看不清或不确定的地方，简短说明不确定，不要随便给一个具体答案。
- 不要用"这张图片""这幅画""该作品""这是一幅"之类的开头，直接从内容写起。

{domain_rules}
{metadata_block}{extra_notes}
只输出描述正文，不要标题、前言或结尾语。"""


_PROSE_RULES = {
    "en": "Format: continuous descriptive prose. One or more paragraphs, no "
          "labels, no bullet points, no lists.",
    "zh": "格式：连贯的叙述性文字。可以分段，但不要小标题、不要项目符号、不要罗列。",
}

_STRUCTURED_RULES = {
    "en": "Format: grouped description. Write short labelled groups, each a "
          "sentence or two of continuous prose, covering in order: main "
          "subjects and their attributes; placement and spatial relationships; "
          "setting and background; medium, technique and rendering; light and "
          "colour; notable details. Label each group on its own line. Do not "
          "write it as instructions for editing an image.",
    "zh": "格式：分组描述。每组用一行小标题开头，后面接一到两句连贯文字，依次覆盖：主体与其特征；\
位置与空间关系；环境与背景；媒介、技法与画法；光线与色彩；值得注意的细节。不要写成修图或改图的指令。",
}

_DOMAIN_RULES = {
    "en": {
        "generic": "Use precise, concrete nouns for objects, materials and "
                   "actions. Name what a viewer can point at.\n",
        "chinese_painting": "This is a Chinese ink or colour painting. Use the "
            "accurate vocabulary of the medium: brushwork (outline, texture "
            "strokes, wet or dry ink), ink tonality, washes, silk or paper "
            "ground, mounting, seals and inscriptions. If you can read an "
            "inscription or a seal, quote only the characters you can actually "
            "read and say where it sits; if you cannot read it, describe its "
            "position and shape instead. Never turn unreadable text into "
            "invented characters.\n",
        "western_art": "This is a Western painting or drawing. Use accurate "
            "vocabulary for medium and technique (support, ground, impasto, "
            "glazing, brush or pencil handling), and for pictorial "
            "construction (perspective, modelling, chiaroscuro, palette).\n",
        "photograph": "This is a photograph. Describe the subject, the light, "
            "the framing, the depth of field, and any visible lens or exposure "
            "characteristics. Do not call it a painting.\n",
    },
    "zh": {
        "generic": "用准确、具体的名词描述物体、材质和动作，写清观者能够指认的东西。\n",
        "chinese_painting": "这是一幅中国画。请使用准确的媒介词汇：笔法（勾勒、皴法、干笔湿笔）、\
墨色浓淡、设色与渲染、绢本或纸本、装裱形制、印章与题跋。如果题字或印文能够辨认，只引用你确实认得的字，\
并说明它所在的位置；如果认不出来，就描述它的位置和形态，不要把认不出的字编造出来。\n",
        "western_art": "这是一幅西洋绘画或素描。请使用准确的媒介与技法词汇（基底、底子、厚涂、罩染、\
笔触或铅笔线条），以及构图词汇（透视、明暗塑造、明暗对照、色调）。\n",
        "photograph": "这是一张照片。请描述主体、光线、取景、景深，以及可见的镜头或曝光特征，\
不要把它说成绘画。\n",
    },
}

_METADATA_ABSENT = {
    "en": "\nNo metadata is available for this image. Rely only on what you see.\n",
    "zh": "\n这张图没有提供元数据，只依据画面本身来写。\n",
}

_METADATA_PRESENT = {
    "en": "\nVerified metadata (use only the parts that agree with the image; "
          "ignore anything that does not):\n{lines}\n",
    "zh": "\n已核对的元数据（只用与画面相符的部分，与画面不符的忽略）：\n{lines}\n",
}


# ---------------------------------------------------------------------------
# Output checks.  These are the structural gates every generated caption must
# pass before it is eligible for review; they never judge whether the words
# match the picture, which requires looking at the image.
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    retained_tokens: int = 0
    in_band: bool = False

    @property
    def usable(self) -> bool:
        return self.ok


def check_caption(text: str, request: CaptionRequest, tokenizer) -> CheckResult:
    """Apply the non-visual acceptance rules to one generated caption."""
    reasons: List[str] = []
    cleaned = (text or "").strip()
    if not cleaned:
        return CheckResult(ok=False, reasons=["empty"])

    retained = retained_length(cleaned, tokenizer)
    low, high = request.low_tokens, request.high_tokens
    in_band = low <= retained <= high
    if not in_band:
        reasons.append(f"length {retained} outside {low}-{high}")

    head = cleaned.lower()[:60]
    for opener in BOILERPLATE_OPENERS:
        if head.startswith(opener):
            reasons.append(f"boilerplate opener: {opener}")
            break

    if _has_duplicate_sentences(cleaned):
        reasons.append("repeated sentences")

    if _has_markdown_noise(cleaned):
        reasons.append("markdown or fence noise")

    return CheckResult(ok=not reasons, reasons=reasons, retained_tokens=retained,
                       in_band=in_band)


def retained_length(text: str, tokenizer) -> int:
    """Retained prompt tokens for a caption, using the training contract."""
    from ..utils.prompt_contract import (
        DROP_IDX, MAX_SEQUENCE_LENGTH, PROMPT_TEMPLATE, RETAINED_MIN_LENGTH, SYSTEM_PROMPT,
    )

    prompt = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=text)
    encoded = tokenizer(prompt, truncation=True,
                        max_length=MAX_SEQUENCE_LENGTH + DROP_IDX, padding=False)
    return max(len(encoded["input_ids"]) - DROP_IDX, RETAINED_MIN_LENGTH)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s*")


def _has_duplicate_sentences(text: str) -> bool:
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 25]
    return len(sentences) != len(set(sentences))


def _has_markdown_noise(text: str) -> bool:
    return "```" in text or text.lstrip().startswith("#") or "**" in text
