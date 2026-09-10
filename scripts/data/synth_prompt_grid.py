"""Build the prompt grid for the synthetic East Asian / traditional-dress set.

The corpus is thin on people from East Asia and on garments that have their own
name (hanfu, Tibetan chuba, Miao silver festival dress), and stock photography
does not cover them.  This grid fills that gap with generated pictures.

Each grid row is used twice: it is the prompt the image generator is given, and
it is the caption the produced picture is trained on.  That is why the wording
is written the way a person describes a picture they want - one plain sentence,
no hedging, no tag list - and why every row is a combination of slots rather
than free text: a caption and the picture it labels cannot drift apart if the
picture is made from the caption.

Slots are only combined where the combination makes sense: indoor light never
appears on a grasslands picture, and the subject noun phrase carries its own
article so the English reads correctly whatever the subject starts with.

Output is JSONL, one row per picture to generate:

    prompt_id      stable id, also the image id stem
    language       "zh" or "en"
    text           the prompt, and the caption
    aspect         one of the five aspect buckets, with the size to generate
    family         "photograph" or "painting"
    subject        slot keys, so a batch can be checked for coverage

Usage:
    python -m scripts.data.synth_prompt_grid --count 10000 --out grid.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# The five aspect buckets, as (width, height).  These are exactly the frames
# the chosen generator was trained on, so every picture comes out at a shape it
# renders well and needs no reshaping; the names are the ratio each one is
# closest to.
ASPECTS: Tuple[Tuple[str, int, int], ...] = (
    ("1x1", 1024, 1024),
    ("3x2", 1264, 848),
    ("2x3", 848, 1264),
    ("16x9", 1376, 768),
    ("9x16", 768, 1376),
)


@dataclass(frozen=True)
class Subject:
    """One person-and-garment combination.

    ``en`` carries its own article: "a Han Chinese woman ..." and "an East Asian
    woman ..." cannot be produced by one rule, and an article mistake would be
    trained on as text.
    """

    key: str
    group: str
    zh: str
    en: str


SUBJECTS: Tuple[Subject, ...] = (
    # Han Chinese dress, by period.
    Subject("hanfu-ming", "hanfu", "汉族女性穿明制立领长袄配马面裙",
            "a Han Chinese woman in a Ming-style standing-collar jacket and a mamian skirt"),
    Subject("hanfu-tang", "hanfu", "汉族女性穿唐制齐胸襦裙",
            "a Han Chinese woman in a Tang-style chest-high ruqun"),
    Subject("hanfu-song", "hanfu", "汉族女性穿宋制褙子与宋裤",
            "a Han Chinese woman in a Song-style beizi jacket and trousers"),
    Subject("hanfu-crossed", "hanfu", "汉族女性穿交领襦裙",
            "a Han Chinese woman in a crossed-collar ruqun"),
    Subject("hanfu-roundcollar", "hanfu", "汉族男性穿圆领袍戴幞头",
            "a Han Chinese man in a round-collar robe and a futou cap"),
    Subject("hanfu-daopao", "hanfu", "汉族男性穿明代道袍系丝绦",
            "a Han Chinese man in a Ming-style daopao robe tied with a silk sash"),
    Subject("hanfu-feiyu", "hanfu", "汉族男性穿明制飞鱼服",
            "a Han Chinese man in a Ming-style feiyu robe"),
    Subject("hanfu-modern", "hanfu", "汉族女性穿汉元素日常上衣配长裙",
            "a Han Chinese woman in everyday hanfu-inspired clothing and a long skirt"),
    Subject("qipao", "hanfu", "汉族女性穿素色丝绸旗袍",
            "a Han Chinese woman in a plain silk qipao"),
    Subject("zhongshan", "hanfu", "汉族男性穿藏青色中山装",
            "a Han Chinese man in a navy Zhongshan suit"),

    # Ethnic-minority dress of China.
    Subject("manchu", "minority", "满族女性穿旗装配大拉翅头饰",
            "a Manchu woman in qizhuang with a tall dalachii headdress"),
    Subject("mongol", "minority", "蒙古族男性穿蒙古袍系腰带",
            "a Mongolian man in a deel robe with a sash"),
    Subject("mongol-woman", "minority", "蒙古族女性戴头巾穿镶边蒙古袍",
            "a Mongolian woman in a trimmed deel and a headscarf"),
    Subject("tibetan", "minority", "藏族女性穿藏袍系邦典围裙",
            "a Tibetan woman in a chuba with a striped pangden apron"),
    Subject("tibetan-kham", "minority", "藏族康巴男性穿镶豹皮边藏袍",
            "a Kham Tibetan man in a chuba trimmed with leopard fur"),
    Subject("uyghur", "minority", "维吾尔族女性穿艾德莱斯绸连衣裙戴花帽",
            "a Uyghur woman in an atlas-silk dress and a doppa cap"),
    Subject("kazakh", "minority", "哈萨克族女性戴羽毛帽穿刺绣长袍",
            "a Kazakh woman in an embroidered robe and a feathered hat"),
    Subject("miao-silver", "minority", "苗族女性戴银冠银项圈穿盛装",
            "a Miao woman in festival dress with a silver crown and neck rings"),
    Subject("miao-bird", "minority", "苗族女性穿百鸟衣",
            "a Miao woman in a hundred-bird festival garment"),
    Subject("yi", "minority", "彝族男性披察尔瓦戴鸡冠帽",
            "a Yi man in a chawa cloak and a rooster-shaped hat"),
    Subject("zhuang", "minority", "壮族女性穿壮锦上衣",
            "a Zhuang woman in a brocade jacket"),
    Subject("bai", "minority", "白族女性穿扎染服饰",
            "a Bai woman in tie-dyed dress"),
    Subject("tujia", "minority", "土家族女性穿西兰卡普织锦上衣",
            "a Tujia woman in a xilankapu brocade jacket"),
    Subject("chaoxian", "minority", "朝鲜族女性穿短衣长裙",
            "an ethnic Korean woman from China in a short jacket and a long skirt"),
    Subject("dong", "minority", "侗族女性穿亮布对襟上衣",
            "a Dong woman in a glossy-cloth jacket"),
    Subject("yao", "minority", "瑶族女性穿盘王印纹样服饰",
            "a Yao woman in dress patterned with panwang seals"),
    Subject("qiang", "minority", "羌族女性穿绣花围腰配云云鞋",
            "a Qiang woman in an embroidered apron and cloud-pattern shoes"),
    Subject("dai", "minority", "傣族女性穿筒裙配傣锦披肩",
            "a Dai woman in a tube skirt and a brocade shawl"),
    Subject("li", "minority", "黎族女性穿黎锦上衣与筒裙",
            "a Li woman in a brocade jacket and a tube skirt"),
    Subject("she", "minority", "畲族女性穿凤凰装",
            "a She woman in phoenix costume"),
    Subject("naxi", "minority", "纳西族女性披七星羊皮披肩",
            "a Naxi woman in a seven-star sheepskin cape"),

    # Neighbouring East and Southeast Asian dress.
    Subject("furisode", "neighbour", "日本女性穿振袖和服",
            "a Japanese woman in a furisode kimono"),
    Subject("yukata", "neighbour", "日本女性穿浴衣系腰带",
            "a Japanese woman in a yukata with an obi sash"),
    Subject("hanbok", "neighbour", "韩国女性穿韩服",
            "a Korean woman in a hanbok"),
    Subject("aodai", "neighbour", "越南女性穿奥黛戴斗笠",
            "a Vietnamese woman in an ao dai and a conical hat"),
    Subject("bingata", "neighbour", "琉球女性穿红型染和服",
            "a Ryukyuan woman in a bingata-dyed robe"),

    # Everyday East Asian people: the corpus needs ordinary modern dress too,
    # not only festival costume.
    Subject("modern-woman", "modern", "东亚女性穿毛衣配牛仔裤",
            "an East Asian woman in a knit sweater and jeans"),
    Subject("modern-man", "modern", "东亚男性穿衬衫配西裤",
            "an East Asian man in a shirt and tailored trousers"),
    Subject("street-youth", "modern", "中国城市青年穿连帽卫衣",
            "a young Chinese person in a hooded sweatshirt"),
    Subject("rural-elder", "modern", "中国乡村老年女性穿棉布对襟褂",
            "an elderly Chinese woman in a cotton button-front jacket"),
    Subject("child-festival", "modern", "中国儿童穿节日新衣",
            "a Chinese child in new clothes for a festival"),
)

# (key, zh, en, indoor).  Light is drawn only from what the setting admits.
SCENES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("garden", "站在江南园林的回廊里", "standing in a covered walkway of a Jiangnan garden", False),
    ("palace", "站在宫殿前的石阶上", "standing on the stone steps in front of a palace", False),
    ("village-bridge", "站在水乡村口的石桥上", "standing on the stone bridge at a waterside village entrance", False),
    ("market", "站在山顶集市的人群中", "standing among the crowd at a hilltop market", False),
    ("grassland", "站在内蒙古草原上", "standing on the Inner Mongolian grassland", False),
    ("pasture", "站在雪山下的牧场上", "standing on a pasture below snow mountains", False),
    ("temple", "站在寺庙山门前", "standing in front of a temple gate", False),
    ("courtyard", "站在徽派民居的天井里", "standing in the open courtyard of a Huizhou house", False),
    ("city-street", "站在现代城市的街道上", "standing on a modern city street", False),
    ("rapeseed", "站在油菜花田边", "standing at the edge of a rapeseed field", False),
    ("raft", "站在漓江的竹筏上", "standing on a bamboo raft on the Li river", False),
    ("saltlake", "站在盐湖的水边", "standing at the edge of a salt lake", False),
    ("study", "坐在书房的书桌前", "sitting at a desk in a study", True),
    ("hall", "站在徽派民居的堂屋里", "standing in the main hall of a Huizhou house", True),
    ("studio", "站在灰色影棚背景前", "standing in front of a plain grey studio backdrop", True),
)

# (key, zh, en, where): "indoor", "outdoor" or "any".
LIGHTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("dawn", "清晨的侧光", "side light at dawn", "outdoor"),
    ("noon", "正午的硬光", "hard light at noon", "outdoor"),
    ("backlit", "黄昏的逆光", "backlight at dusk", "outdoor"),
    ("overcast", "阴天的柔光", "soft overcast light", "any"),
    ("window", "室内的窗光", "window light", "indoor"),
    ("candle", "烛光", "candlelight", "indoor"),
    ("softbox", "影棚的柔光箱", "a studio softbox", "indoor"),
)

FRAMINGS: Tuple[Tuple[str, str, str], ...] = (
    ("full", "全身", "full-length"),
    ("three-quarter", "七分身", "three-quarter-length"),
    ("half", "半身", "waist-up"),
    ("closeup", "面部特写", "close-up"),
    ("profile", "侧脸", "profile"),
    ("back", "背影", "back-view"),
    ("group", "群像", "group"),
)

PHOTO_STYLES: Tuple[Tuple[str, str], ...] = (
    ("film35", "135 胶片颗粒", "film-grain"),
    ("medium", "中画幅数码", "medium-format"),
    ("phone", "手机抓拍", "phone-camera"),
    ("studio", "影棚布光", "studio-lit"),
)

# Adjectives.  The English carries its article because "a fine-line" and "an
# ink-wash" cannot share one rule, and the template prefixes it to "figure
# painting".
PAINTING_STYLES: Tuple[Tuple[str, str], ...] = (
    ("gongbi", "工笔重彩", "a fine-line heavy-colour"),
    ("xieyi", "水墨写意", "an ink-wash"),
    ("dancai", "淡彩", "a light-colour"),
    ("mural", "壁画风格", "a mural-style"),
    ("juanshe", "绢本设色", "a silk-ground"),
)

MOUNTS: Tuple[Tuple[str, str], ...] = (
    ("scroll", "立轴", "a hanging scroll"),
    ("album", "册页", "an album leaf"),
    ("fan", "扇面", "a fan painting"),
    ("none", "", ""),
)

# Sentence patterns, so the grid does not read as one template filled in ten
# thousand times.  Subjects carry their own article in English.
PHOTO_TEMPLATES = {
    "zh": (
        "{photo_style}人像照片：{framing}，一位{subject}，{scene}，{light}。",
        "拍摄一位{subject}的{photo_style}{framing}人像照片，{scene}，{light}。",
        "{scene}的{photo_style}照片，画面里是{framing}：一位{subject}，{light}。",
        "以一位{subject}为主体的{framing}{photo_style}人像照片，{scene}，{light}。",
    ),
    "en": (
        "A {photo_style} {framing} portrait of {subject}, {scene}, {light}.",
        "A {photo_style} photograph of {subject}, {framing} composition, {scene}, {light}.",
        "A {photo_style} {framing} photograph of {subject}, {scene}, {light}.",
        "A {framing} portrait of {subject}, {scene}, {light}, {photo_style}.",
    ),
}

PAINTING_TEMPLATES = {
    "zh": (
        "{paint_style}人物画：一位{subject}，{scene}。{mount}",
        "一幅{paint_style}人物画，画的是一位{subject}，{scene}。{mount}",
        "一位{subject}，{scene}，{paint_style}人物画。{mount}",
    ),
    "en": (
        "{paint_style} figure painting of {subject}, {scene}.{mount}",
        "{paint_style} figure painting showing {subject}, {scene}.{mount}",
        "{paint_style} figure painting set {scene}, with {subject} as the main figure.{mount}",
    ),
}


def _mount_fields(mount: Tuple[str, str], language: str) -> Dict[str, str]:
    """Mount wording, already substituted: a suffix cannot nest its own format."""
    if mount[0] == "none":
        return {"mount": ""}
    if language == "zh":
        return {"mount": f"装裱为{mount[1]}。"}
    return {"mount": f" Mounted as {mount[2]}."}


def build_rows(count: int, seed: int, zh_share: float, photo_share: float) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    seen = set()
    attempts = 0
    while len(rows) < count and attempts < count * 200:
        attempts += 1
        language = "zh" if rng.random() < zh_share else "en"
        family = "photograph" if rng.random() < photo_share else "painting"
        subject = rng.choice(SUBJECTS)
        # A studio backdrop belongs to the studio scene only: it makes no sense
        # as the setting of a painting, or as the light on a grasslands picture.
        scene = rng.choice([s for s in SCENES
                            if s[0] != "studio" or family == "photograph"])
        framing = rng.choice(FRAMINGS)

        if family == "photograph":
            allowed = [light for light in LIGHTS
                       if light[3] == "any"
                       or light[3] == ("indoor" if scene[3] else "outdoor")]
            if scene[3] and scene[0] == "studio":
                allowed = [light for light in LIGHTS if light[3] != "outdoor"]
            light = rng.choice(allowed)
            style = rng.choice([s for s in PHOTO_STYLES
                                if (s[0] == "studio") == (scene[0] == "studio")])
            templates = PHOTO_TEMPLATES[language]
            mount = ("none", "", "")
        else:
            light = None
            style = rng.choice(PAINTING_STYLES)
            templates = PAINTING_TEMPLATES[language]
            mount = rng.choice(MOUNTS)

        key = (language, family, subject.key, scene[0], framing[0],
               style[0], light[0] if light else "-", mount[0])
        if key in seen:
            continue
        seen.add(key)

        fields = {
            "subject": subject.zh if language == "zh" else subject.en,
            "scene": scene[1] if language == "zh" else scene[2],
            "light": (light[1] if language == "zh" else light[2]) if light else "",
            "framing": framing[1] if language == "zh" else framing[2],
            "photo_style": style[1] if language == "zh" else style[2],
            "paint_style": style[1] if language == "zh" else style[2],
        }
        fields.update(_mount_fields(mount, language))
        aspect = ASPECTS[len(rows) % len(ASPECTS)]     # even share per shape
        text = rng.choice(templates).format(**fields).strip()
        if language == "en":
            text = text[0].upper() + text[1:]
        rows.append({
            "prompt_id": f"syn-{len(rows):06d}",
            "language": language,
            "text": text,
            "aspect": aspect[0],
            "width": aspect[1],
            "height": aspect[2],
            "family": family,
            "subject": subject.key,
            "subject_group": subject.group,
            "style": style[0],
            "scene": scene[0],
            "framing": framing[0],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--zh-share", type=float, default=0.5)
    parser.add_argument("--photo-share", type=float, default=0.8)
    args = parser.parse_args()

    rows = build_rows(args.count, args.seed, args.zh_share, args.photo_share)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def by(key: str) -> Dict[str, int]:
        return dict(Counter(row[key] for row in rows))

    print(f"{len(rows)} prompts -> {out}")
    print(f"  language {by('language')}")
    print(f"  family   {by('family')}")
    print(f"  aspect   {by('aspect')}")
    print(f"  group    {by('subject_group')}")
    print(f"  distinct texts {len({row['text'] for row in rows})}")


if __name__ == "__main__":
    main()
