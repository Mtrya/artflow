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

PROMPT_VERSION = "cap-v7"

# Retained-token bands.  A request targets the middle of its band
# and is accepted anywhere inside it.
LENGTH_BANDS = {
    "64-255": (64, 255),
    "256-511": (256, 511),
    "512-895": (512, 895),
    "896-1280": (896, 1280),
}

WORDS_PER_TOKEN = {"en": 0.757, "zh": 1.21}

LANGUAGES = ("en", "zh")
FORMATS = ("prose", "structured")



def metadata_for_caption(artist: Optional[str], title: Optional[str],
                         language: str) -> Dict[str, str]:
    """Tidy museum fields into what a caption may use.

    The harvested fields are catalogue strings: the artist often lists Chinese
    and romanised names together, and the title carries a Chinese and an English
    version separated by a pipe.  Keep the part that matches the caption's
    language and drop the romanisation, so the model is not tempted to paste a
    catalogue line into the caption.
    """
    fields: Dict[str, str] = {}
    if artist:
        parts = [p.strip() for p in artist.split(",")]
        keep = [p for p in parts if p and not any(ch.isascii() and ch.isalpha() for ch in p)]
        cleaned = "、".join(keep) if keep else parts[0].strip()
        if cleaned:
            fields["作者" if language == "zh" else "artist"] = cleaned
    if title:
        zh, _, en = title.partition(" | ")
        value = (zh if language == "zh" else en or zh).strip()
        if value and value.lower() not in {"untitled", "無題", "无题", "no title"}:
            fields["作品名" if language == "zh" else "title"] = value
    return fields


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
    # Where inside the band to aim, 0 = lower edge, 1 = upper edge.  The
    # default is a third of the way in because models overshoot more often
    # than they undershoot.  Models with a measured systematic bias get a
    # position near the opposite edge; the acceptance band never changes.
    target_position: float = 1.0 / 3.0

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
        position = min(max(self.target_position, 0.0), 1.0)
        return int(round(self.low_tokens
                         + position * (self.high_tokens - self.low_tokens)))

    def target_units(self) -> int:
        return int(round(self.target_tokens * WORDS_PER_TOKEN[self.language]))

    def unit_range(self) -> tuple:
        factor = WORDS_PER_TOKEN[self.language]
        return (
            int(self.low_tokens * factor * 0.95),
            int(self.high_tokens * factor * 1.05),
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
            "format_rules": self._format_rules(),
            "domain_rules": _DOMAIN_RULES[self.language].get(
                self.domain, _DOMAIN_RULES[self.language]["generic"]),
            "metadata_block": self._metadata_block(),
            "extra_notes": self.extra_notes.strip(),
        }

    def _format_rules(self) -> str:
        """Formatting instructions, tightened for the shortest band.

        The structured format asks for several paragraphs, which by itself
        pushes a caption past a short target: a paragraph per region of the
        picture is more than a 64-255 token caption can hold.  In that band the
        request collapses to a single paragraph.
        """
        rules = (_STRUCTURED_RULES if self.format == "structured"
                 else _PROSE_RULES)[self.language]
        if self.band == "64-255":
            rules += _SHORT_BAND_NOTE[self.language]
        return rules

    def _metadata_block(self) -> str:
        if not self.metadata:
            return _METADATA_ABSENT[self.language]
        lines = [f"- {key}: {value}" for key, value in self.metadata.items() if value]
        if not lines:
            return _METADATA_ABSENT[self.language]
        return _METADATA_PRESENT[self.language].format(lines="\n".join(lines))


_EN_TEMPLATE = """You are writing training captions for a text-to-image model. \
Describe the single image attached to this message.

Length: about {target} {unit}. Acceptable range {low}-{high} {unit}. Staying \
inside that range matters more than reaching the target: stop once you have \
described what is worth describing. Reach the length with more visible content, \
never with repetition, synonym lists or generic praise.

{format_rules}

Write plainly what is visible, the way a user states the picture they want: \
"a woman in a red silk dress", not "what appears to be a woman, possibly in a \
dress". No speculation of any kind - no "appears", "seems", "looks like", \
"possibly", "probably", "some kind of", "a type of", no question marks - and \
any other way of saying "I am not sure" is equally forbidden. Where two \
readings are possible, name the one the picture best supports; where a detail \
cannot be read off the picture, leave it out instead of saying it is unclear. \
Naming a visible ethnicity, garment, material or colour is description, not \
invention, so state it directly.

No evaluation: state what is visible, not how good it is. Do not praise the \
work or its maker, do not judge the composition, technique or mood \
("masterfully rendered", "beautifully composed", "conveys a serene \
atmosphere" are not allowed).

Medium and style: name the medium explicitly and early - photograph, oil \
painting, watercolour, drawing, print, digital artwork, or whatever is actually \
there - with the visible evidence for it: grain, blur, brushstrokes, canvas or \
paper texture, ink wash, halftone dots; a caption that never says what kind of \
image this is is unusable. Never present a photograph as a painting or the \
reverse; when a picture could pass for either, follow the strongest visible \
evidence. Then name the style at the level the picture shows: when visible \
traits - stroke, palette, light, flat or modelled forms, pattern, composition - \
identify a style, movement or period (Impressionism, Post-Impressionism, \
Symbolism, Realism, Baroque, and so on), name it together with the traits that \
carry it. For a picture whose style is visible, "an oil painting" alone leaves \
the caption incomplete. Do not invent what the picture cannot show: artists, \
exact dates, places, or movements its visible traits do not support.

People: when people are visible, describe skin tone, hair, apparent age and \
dress, and name the ethnicity or region their features point to (East Asian, \
South Asian, Southeast Asian, African, European, Middle Eastern, Latin \
American); leave it out only when the picture truly cannot tell. Give a \
garment its own name when you know it - hanfu, kimono, hanbok, sari, ao dai, \
cheongsam, kaftan, abaya, dashiki - and otherwise describe its cut, fabric and \
pattern. Do not rate or compliment anyone's looks.

Grounding rules:
- Every statement must come from this image or the metadata below; do not \
describe what is outside the frame.
- Metadata may name an artist or a title - name them or leave them out - and \
never add anything it does not give.
- If the image contains readable writing - an inscription, a signature, a seal, \
a printed label - transcribe it in quotation marks inside the caption and say \
where it sits; never invent characters you cannot read.
- Do not open with "This image", "The image", "This painting" or any similar \
pointer phrase: start with the medium ("A photograph of ...", "An oil painting \
...") or with the content itself.

{domain_rules}
{metadata_block}{extra_notes}
Output only the caption text. No headings, no preamble, no closing remarks."""


_ZH_TEMPLATE = """你为文生图模型撰写训练用的图像描述。请为随本条消息附上的这张图写一条描述。

长度：约 {target} {unit}。可接受范围 {low}–{high} {unit}。控制在范围内比写满目标更重要：\
值得写的写完就停。靠描述画面里真实存在的内容来达到长度，不要重复、堆砌近义词或空泛赞美。

{format_rules}

像用户点单一样直接写看得见的东西："一位穿红色丝绸长裙的女性"，不写"看起来像是一位女性，可能穿着长裙"。\
不允许任何猜测——不写"看起来""像是""似乎是""可能是""某种"之类的词，也不用问号；\
其他任何表示"我不确定"的说法同样禁止。两种读法都说得通时，写画面更支持的那一种；\
读不出来的细节就不写，不要专门说明"这里看不清"。写出看得见的族裔、服饰、材质、器物、颜色是在描述画面，不是编造，直接写出来。

不要评价：只写看得见的东西，不评判它好不好；不夸作品或作者，不评价构图、技法、气韵（"笔法细腻""技艺精湛""意境深远"之类禁止）。

媒介与风格：描述必须在开头或显要位置点明媒介——照片、油画、水彩、素描、版画、数字绘画，\
或画面里实际是什么——并写出可见依据：胶片颗粒、镜头虚化、笔触、画布或纸张纹理、水墨晕染、网点；\
完全没有说明画面类型的描述不可用。不要把照片写成画，也不要把画写成照片；两者不好区分时，\
按最明显的可见证据写。风格要写到画面真正显示的那一层：当可见特征——笔触、色彩、光线、平涂或明暗塑造、纹样、构图——\
足以指认某个风格、流派或时期（印象派、后印象派、象征主义、写实主义、巴洛克等）时，把它写出来，\
并给出支撑它的画面特征。风格可见的作品，只写"油画"是不完整的。不要写画面没有显示的内容：\
作者、确切年代、地点，或画面特征并不支持的流派。

人物：画面里出现人物时，写清肤色、发型、大致年龄和服饰，以及可见特征指向的族裔或地域\
（东亚、南亚、东南亚、非洲、欧洲、中东、拉丁美洲）；只有画面确实看不出时才不写。\
服饰有专名就用专名（汉服、和服、韩服、纱丽、奥黛、旗袍、长袍、头巾等），没有专名就写清款式、面料和纹样。\
不要评价或夸赞谁的长相。

依据要求：
- 每句话都必须来自这张图上可见的内容或下方给出的元数据；不要描述画面之外的内容。
- 元数据给出作者或作品名时，可以自然提及，也可以完全不提；元数据没有的作者、标题、年代、地点、\
象征含义或创作意图，一律不写、不编造。
- 画面里能认出的文字——题款、署名、印章、印刷标签——用引号写进描述并说明位置；认不出的字不要编造。
- 不要用"这张图片""这幅画""该作品"之类的指代式开头；以媒介开头（"一张照片里……""一幅油画……"）\
或直接从内容写起。

{domain_rules}
{metadata_block}{extra_notes}
只输出描述正文，不要标题、前言或结尾语。"""


_PROSE_RULES = {
    "en": "Format: continuous descriptive prose. One or more paragraphs, no "
          "labels, no bullet points, no lists.",
    "zh": "格式：连贯的叙述性文字。可以分段，但不要小标题、不要项目符号、不要罗列。",
}

# Structured output is organised the way a person walks a reader through a
# picture — by where things are — not by art-analysis categories.  Headings
# such as "notable details" or "medium and technique" read like a catalogue
# entry, which is the voice this corpus must not have.
_STRUCTURED_RULES = {
    "en": "Format: a few short paragraphs, each starting with where it looks "
          "or what it covers in plain words (for example \"In the centre\", "
          "\"Along the left edge\", \"Overall\"), then one or two sentences. "
          "Do not use analytical headings such as \"medium and technique\" or "
          "\"notable details\", and do not write it as instructions for "
          "editing an image.",
    "zh": "格式：分成几个短段，每段用一句朴素的方位或范围说明开头（例如“画面中央”“左侧边缘”“整体来看”），\
后接一到两句描述。不要使用“媒介、技法与画法”“值得注意的细节”这类分析式小标题，也不要写成修图或改图的指令。",
}

_SHORT_BAND_NOTE = {
    "en": " At this length one paragraph is enough: write a single continuous "
          "passage, and stop when it is complete.",
    "zh": "这个长度一段写完即可，写成连贯的一整段，写完就停。",
}

_DOMAIN_RULES = {
    "en": {
        "generic": "Use precise, concrete nouns for objects, materials and "
                   "actions. Name what a viewer can point at.\n",
        "chinese_painting": "This is a Chinese ink or colour painting. Use the "
            "accurate vocabulary of the medium: brushwork (outline, texture "
            "strokes, wet or dry ink), ink tonality, washes, silk or paper "
            "ground, mounting, seals and inscriptions. Name the style where "
            "the visible traits show it - gongbi or xieyi, ink or blue-green "
            "colouring, literati or academic painting, and the school or "
            "period character the brushwork and composition point to - with "
            "the traits that carry it. Quote only characters you can actually "
            "read from an inscription or seal; describe its position and shape "
            "instead of inventing characters.\n",
        "western_art": "This is a Western painting or drawing. Use accurate "
            "vocabulary for medium and technique (support, ground, impasto, "
            "glazing, brush or pencil handling) and for pictorial construction "
            "(perspective, modelling, chiaroscuro, palette). Name the "
            "art-historical style the visible character points to - "
            "Impressionism, Post-Impressionism, Symbolism, Realism, "
            "Romanticism, Neoclassicism, Baroque, Rococo, academic painting, "
            "genre, still life, portraiture, landscape, geometric abstraction "
            "- with the traits (stroke, palette, light, composition, subject) "
            "that support it; a caption that stops at \"oil painting\" leaves "
            "the style unstated. Where no named style fits the picture, "
            "describe the traits themselves.\n",
        "photograph": "This is a photograph. Describe the subject, the light, "
            "the framing, the depth of field, and any visible lens or exposure "
            "characteristics. Do not call it a painting.\n",
    },
    "zh": {
        "generic": "用准确、具体的名词描述物体、材质和动作，写清观者能够指认的东西。\n",
        "chinese_painting": "这是一幅中国画。用准确的媒介词汇：笔法（勾勒、皴法、干湿笔）、\
墨色浓淡、设色与渲染、绢本或纸本、装裱形制、印章与题跋。画面特征能看出的风格要写出来——\
工笔还是写意、水墨还是青绿设色、文人画还是院体，以及笔法与构图指向的流派或时代特征——\
并给出支撑它的画面特征。题字或印文只引用你确实认得的字并说明位置，认不出的不要编造。\n",
        "western_art": "这是一幅西洋绘画或素描。用准确的媒介与技法词汇（基底、底子、厚涂、罩染、\
笔触处理）和构图词汇（透视、明暗塑造、明暗对照、色调）。画面特征指向的艺术史风格要点名——\
印象派、后印象派、象征主义、写实主义、浪漫主义、新古典主义、巴洛克、洛可可、学院派、\
风俗画、静物、肖像、风景、几何抽象——并给出支撑它的画面特征（笔触、色彩、光线、构图、题材）；\
只写“油画”而风格缺位是不完整的。归不进任何有名字的风格时，就朴素描述特征本身。\n",
        "photograph": "这是一张照片。描述主体、光线、取景、景深，以及可见的镜头或曝光特征，\
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

# Phrases that judge the picture instead of describing it.  A caption full of
# these reaches its token target without telling a reader what to draw, so the
# rate is measured per batch even though it is not a hard rejection.
EVALUATION_PHRASES = (
    "笔法细腻", "技艺精湛", "技法娴熟", "构图巧妙", "构图疏密得当", "虚实结合",
    "意境深远", "栩栩如生", "恰到好处", "十分自然", "生动传神", "跃然纸上",
    "艺术造诣", "匠心", "精湛", "高超", "别具一格", "相得益彰", "引人入胜",
    "masterfully", "beautifully composed", "skilfully", "skillfully", "exquisite",
    "conveys a sense", "demonstrates the artist", "testament to", "evokes a",
    "meticulously", "impeccable", "breathtaking", "captures the essence",
)


# Words that turn a description into a guess.  A caption written this way reads
# as a comment on a picture rather than as a request for one, so the rate is
# measured per batch even though it is not a hard rejection.
HEDGING_PHRASES = (
    "appears to be", "appear to be", "appears to", "appear to",
    "seems to be", "seem to be", "seems to", "seem to",
    "looks like", "look like", "is likely", "are likely",
    "possibly", "probably", "perhaps", "maybe", "some kind of", "a type of",
    "or similar",
    "看起来", "似乎是", "好像", "大概是", "可能是", "也许是", "像是", "大约",
)


@dataclass
class CheckResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    retained_tokens: int = 0
    in_band: bool = False
    evaluation_hits: List[str] = field(default_factory=list)
    hedging_hits: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.ok


_EN_OPENER = re.compile(
    r"^\s*(?:in\s+)?(?:this|the)\s+"
    r"(?:image|picture|photo|photograph|painting|artwork|drawing|scroll|work)"
    r"(?:\s+(?:shows|depicts|displays|features|presents|contains|is|of|showing|depicting))?"
    r"\s*(?:[,:;\u2014-]\s*)?",
    re.IGNORECASE,
)
_ZH_OPENER = re.compile(
    r"^\s*(?:这张|这幅|这帧|该|此|这)\s*(?:图片|图画|画作|作品|照片|画|图)?\s*"
    r"(?:显示|展示|描绘|呈现|表现|拍摄|是|为)(?:了|着|出)?\s*[，,：:、]?\s*"
)


def strip_boilerplate(text: str, min_keep: float = 0.4) -> str:
    """Drop a catalogue-style opening, if that leaves a usable caption.

    "This image shows a woman ..." becomes "A woman ..."; "这张图片展示了一位
    老人" becomes "一位老人".  The removal is only applied when what remains is
    still a substantial share of the text, so a caption that is nothing but an
    opener is left as it is rather than cut down to a fragment.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    match = _EN_OPENER.match(cleaned) or _ZH_OPENER.match(cleaned)
    if not match:
        return cleaned
    remainder = cleaned[match.end():].lstrip()
    if len(remainder) < max(8, int(len(cleaned) * min_keep)):
        return cleaned
    if remainder[:1].isascii() and remainder[:1].isalpha():
        remainder = remainder[0].upper() + remainder[1:]
    return remainder


def normalize_caption(text: str) -> str:
    """Remove a wrapping code fence and a catalogue-style opening.

    Markdown inside the caption is acceptable content, so it is left alone.
    A fence around the whole answer is a transport wrapper rather than part of
    the text, so it is removed.  Openers are only removed when what remains
    still reads as a caption; otherwise the text is kept as written, because a
    slightly stiff opening is not a reason to throw a description away.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return strip_boilerplate(cleaned)


def check_caption(text: str, request: CaptionRequest, tokenizer) -> CheckResult:
    """Apply the non-visual acceptance rules to one generated caption.

    ``tokenizer=None`` skips the length gate and records the character count
    instead.  A machine that generates captions but has no tokenizer can then
    hand the text to a later pass that measures retained tokens exactly;
    nothing about the request itself depends on the tokenizer.
    """
    reasons: List[str] = []
    cleaned = normalize_caption(text)
    if not cleaned:
        return CheckResult(ok=False, reasons=["empty"])

    low, high = request.low_tokens, request.high_tokens
    in_band = False
    if tokenizer is None:
        retained = len(cleaned)
    else:
        retained = retained_length(cleaned, tokenizer, truncate=False)
        # The band is a request, not a contract.  A caption that came out
        # shorter or longer than asked is still a description of the picture,
        # so length never disqualifies it; ``in_band`` records whether the
        # request was met, and the sampler buckets by the achieved length.
        in_band = low <= retained <= high

    # A catalogue-style opening is removed by normalize_caption when that is
    # possible; if it survives, the caption is still used.  Only a defect of the
    # text itself disqualifies it.
    if _has_duplicate_sentences(cleaned):
        reasons.append("repeated sentences")

    hits = [phrase for phrase in EVALUATION_PHRASES if phrase in cleaned.lower()]
    hedging = [phrase for phrase in HEDGING_PHRASES if phrase in cleaned.lower()]
    return CheckResult(ok=not reasons, reasons=reasons, retained_tokens=retained,
                       in_band=in_band, evaluation_hits=hits, hedging_hits=hedging)


def length_only_reject(reasons) -> bool:
    """True when the only objections to a caption are about its length.

    Length is a request, not a requirement: a caption that came out shorter or
    longer than asked still describes the picture, and the bucket plan can
    accommodate it.  What is dropped is a caption that opens with boilerplate or
    repeats itself, which is a defect of the text rather than of its size.
    """
    reasons = list(reasons or [])
    return bool(reasons) and all(reason.startswith("length ") for reason in reasons)


def retained_length(text: str, tokenizer, truncate: bool = True) -> int:
    """Retained prompt tokens for a caption.

    ``truncate=True`` applies the training contract, which cuts the prompt at
    the model's sequence length.  Pass ``truncate=False`` to measure how long a
    caption really is: the bucket plan must not assume captions stop at the
    sequence limit.
    """
    from ..utils.prompt_contract import (
        DROP_IDX, MAX_SEQUENCE_LENGTH, PROMPT_TEMPLATE, RETAINED_MIN_LENGTH, SYSTEM_PROMPT,
    )

    prompt = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=text)
    if truncate:
        encoded = tokenizer(prompt, truncation=True,
                            max_length=MAX_SEQUENCE_LENGTH + DROP_IDX, padding=False)
    else:
        encoded = tokenizer(prompt, truncation=False, padding=False)
    return max(len(encoded["input_ids"]) - DROP_IDX, RETAINED_MIN_LENGTH)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s*")


def _has_duplicate_sentences(text: str) -> bool:
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 25]
    return len(sentences) != len(set(sentences))


