"""Concept vocabulary assembly for the concept benchmark.

Sources: WordNet noun synsets, ImageNet-1k labels, Getty AAT preferred labels
(pre-extracted to JSONL) and curated Chinese lists. Each entry is cleaned,
tagged with one of the three benchmark axes (entity / technique / scene),
annotated with wordfreq Zipf frequency, and deduplicated on the normalized
English label. Curated entries are exempt from the frequency floor.

Axis convention (see notes/concept_benchmark.md):
- entity: nouns, including material, garment, pose/activity and pattern terms
- technique: art styles, techniques and periods; color lives here
- scene: places and environments
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

AXES = ("entity", "technique", "scene")

# WordNet lexical categories mapped onto benchmark axes.
WORDNET_MAP = {
    "noun.animal": ("entity", "animal"),
    "noun.artifact": ("entity", "artifact"),
    "noun.body": ("entity", "body"),
    "noun.food": ("entity", "food"),
    "noun.object": ("entity", "object"),
    "noun.plant": ("entity", "plant"),
    "noun.substance": ("entity", "material"),
    "noun.person": ("entity", "person_role"),
    "noun.location": ("scene", "location"),
}

# Minimal profanity blocklist: the person-role slice of WordNet contains
# vulgar synonyms that must never reach a prompt.
BLOCKLIST = {
    "cunt", "twat", "whore", "slut", "prick", "fucker", "motherfucker",
    "cocksucker", "wanker", "nigger", "faggot", "dyke", "kike", "spic",
    "chink", "gook", "wop", "wetback", "retard",
}

_FORM_OK = re.compile(r"^[a-z][a-z0-9 \-]{1,28}[a-z0-9]$")
_QUALIFIER = re.compile(r"\s*\(([^()]*)\)\s*$")
_AAT_STYLE_QUAL = re.compile(r"(style|movement|period|art)\b", re.I)
_AAT_STYLE_QUAL_BAD = re.compile(r"^\(?culture or style\)?$", re.I)
_AAT_STYLE_SFX = re.compile(
    r"(ism|deco|nouveau|gothic|baroque|rococo|renaissance|classical|revival"
    r"|figure|byzantine|romanesque|mannerist|ukiyo-e)\b", re.I)


@dataclass
class Concept:
    en: str                          # normalized English label, qualifier-free
    axis: str
    sub: str
    sources: list[str] = field(default_factory=list)
    zh: str | None = None
    qualifier: str | None = None
    freq_en: float | None = None
    freq_zh: float | None = None
    curated: bool = False
    aliases: list[str] = field(default_factory=list)  # other surface forms
    family: str | None = None        # cluster key, assigned by family clustering


def normalize_en(label: str) -> str:
    """Lowercase, strip accents, collapse whitespace; keep the qualifier out."""
    label = unicodedata.normalize("NFKD", label)
    label = "".join(c for c in label if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", label).strip().lower()


def split_qualifier(label: str) -> tuple[str, str | None]:
    """'ibriks (coffee makers)' -> ('ibriks', 'coffee makers')."""
    m = _QUALIFIER.search(label)
    if m:
        return label[: m.start()].strip(), m.group(1).strip()
    return label.strip(), None


def wordnet_form_ok(lemma: str) -> bool:
    """Common-noun filter: lowercase-only drops proper nouns and abbreviations."""
    return bool(_FORM_OK.match(lemma)) and not lemma.startswith(("a ", "an ", "the "))


def zipf(term: str, lang: str = "en") -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(term, lang)


def unescape_ntriples(text: str) -> str:
    """NTriples escapes non-ASCII as literal \\uXXXX sequences; undo that."""
    return re.sub(r"\\u([0-9a-fA-F]{4})",
                  lambda m: chr(int(m.group(1), 16)), text)


# WordNet noun.location mixes real places with abstract locative senses
# (time zone, tendency, destination). Keep only synsets passing through a
# natural-feature or settlement ancestor; AAT covers the built side.
_SCENE_ANCESTORS = {
    "geological_formation.n.01", "body_of_water.n.01",
    "natural_depression.n.01", "natural_elevation.n.01",
    "tract.n.01", "jungle.n.01", "settlement.n.06", "shore.n.01",
}


def _scene_synset_ok(synset) -> bool:
    for path in synset.hypernym_paths():
        if any(h.name() in _SCENE_ANCESTORS for h in path):
            return True
    return False


def singularize_last(en: str) -> str:
    """Crude head-noun singularization, used only as a merge key."""
    words = en.split()
    w = words[-1]
    if w.endswith("ies") and len(w) > 3:
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "ses", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return " ".join(words[:-1] + [w])


def load_wordnet(min_zipf: float = 2.5) -> list[Concept]:
    """Noun synsets in the mapped lexical categories, frequency-floored.

    One Concept per synset: the most frequent lemma is canonical, the rest
    are aliases (synonyms must not become separate sampling units).
    """
    from nltk.corpus import wordnet as wn

    out = []
    for synset in wn.all_synsets("n"):
        mapped = WORDNET_MAP.get(synset.lexname())
        if mapped is None:
            continue
        axis, sub = mapped
        if axis == "scene" and not _scene_synset_ok(synset):
            continue
        lemmas = []
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if wordnet_form_ok(name) and name not in BLOCKLIST:
                lemmas.append(name)
        if not lemmas:
            continue
        scored = sorted(((zipf(lm), lm) for lm in lemmas), reverse=True)
        freq, canonical = scored[0]
        if freq < min_zipf:
            continue
        out.append(Concept(en=normalize_en(canonical), axis=axis, sub=sub,
                           sources=["wordnet"], freq_en=freq,
                           aliases=[normalize_en(lm) for _, lm in scored[1:]]))
    return out


def load_imagenet(path: str) -> list[Concept]:
    labels = json.load(open(path))
    out = []
    for label in labels:
        key = normalize_en(label)
        out.append(Concept(en=key, axis="entity", sub="imagenet",
                           sources=["imagenet"], freq_en=zipf(key)))
    return out


def aat_style_ok(base: str, qualifier: str | None) -> bool:
    """Styles and Periods mixes true styles with peoples and demonyms.

    Keep qualifier-annotated styles/periods/movements (but not the bare
    'culture or style' qualifier, which is almost always an ethnic group) and
    unqualified names carrying a well-known movement suffix.
    """
    if qualifier is not None:
        if _AAT_STYLE_QUAL_BAD.match(qualifier):
            return False
        return bool(_AAT_STYLE_QUAL.search(qualifier))
    return bool(_AAT_STYLE_SFX.search(base))


def load_aat(path: str, min_zipf: float = 2.5) -> list[Concept]:
    """Pre-extracted AAT English preferred labels, routed by hierarchy path.

    The hierarchy path (parentString) decides the axis: Styles and Periods and
    Processes and Techniques and Color are technique; Settlements and
    Landscapes and Built Complexes and Districts are scene; the remaining
    Objects/Materials/Agents/Physical Attributes mass is entity. Other
    Activities hierarchies (events, functions, disciplines) are abstract and
    dropped.
    """
    out = []
    for line in open(path):
        row = json.loads(line)
        label, facet = row["label"], row["facet"]
        if label.startswith("<") or facet is None:
            continue
        path = set(row.get("path") or [])
        base, qualifier = split_qualifier(label)
        key = normalize_en(base)
        if not key:
            continue
        freq = zipf(key)

        if facet == "Styles and Periods Facet":
            if not aat_style_ok(base, qualifier):
                continue
            out.append(Concept(en=key, axis="technique", sub="style",
                               sources=["aat"], qualifier=qualifier,
                               freq_en=freq))
        elif "Processes and Techniques (hierarchy name)" in path:
            out.append(Concept(en=key, axis="technique", sub="process",
                               sources=["aat"], qualifier=qualifier,
                               freq_en=freq))
        elif "Color (hierarchy name)" in path:
            out.append(Concept(en=key, axis="technique", sub="color",
                               sources=["aat"], qualifier=qualifier,
                               freq_en=freq))
        elif "Settlements and Landscapes (hierarchy name)" in path:
            out.append(Concept(en=key, axis="scene", sub="settlement_landscape",
                               sources=["aat"], qualifier=qualifier,
                               freq_en=freq))
        elif "Built Complexes and Districts (hierarchy name)" in path:
            out.append(Concept(en=key, axis="scene", sub="built_complex",
                               sources=["aat"], qualifier=qualifier,
                               freq_en=freq))
        elif facet in ("Objects Facet", "Materials Facet", "Agents Facet",
                       "Physical Attributes Facet"):
            if freq < min_zipf:
                continue
            sub = {"Objects Facet": "object", "Materials Facet": "material",
                   "Agents Facet": "agent",
                   "Physical Attributes Facet": "attribute"}[facet]
            out.append(Concept(en=key, axis="entity", sub=sub,
                               sources=["aat"], qualifier=qualifier,
                               freq_en=freq))
        # Associated Concepts, Brand Names and the abstract Activities
        # hierarchies carry no reliably visual concepts; skip them.
    return out


def load_curated(paths: list[str]) -> list[Concept]:
    """Curated zh/en lists; exempt from the frequency floor by construction."""
    out = []
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            en = normalize_en(row["en"])
            concept = Concept(en=en, axis=row["axis"], sub=row.get("sub", "curated"),
                              sources=["curated"], zh=row["zh"], curated=True)
            concept.freq_en = zipf(en)
            concept.freq_zh = zipf(row["zh"], "zh")
            out.append(concept)
    return out


def merge(concepts: list[Concept]) -> list[Concept]:
    """Deduplicate on (axis, singularized label), merging provenance+aliases."""
    by_key: dict[tuple[str, str], Concept] = {}
    for c in concepts:
        if c.en in BLOCKLIST:
            continue
        key = (c.axis, singularize_last(c.en))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = c
            continue
        existing.sources = sorted(set(existing.sources) | set(c.sources))
        existing.curated = existing.curated or c.curated
        if c.en != existing.en:
            existing.aliases = sorted(set(existing.aliases) | {c.en})
        existing.aliases = sorted(set(existing.aliases) | set(c.aliases))
        if existing.zh is None:
            existing.zh = c.zh
        if existing.qualifier is None:
            existing.qualifier = c.qualifier
        freqs = [f for f in (existing.freq_en, c.freq_en) if f is not None]
        existing.freq_en = max(freqs) if freqs else None
    return list(by_key.values())


def report(concepts: list[Concept]) -> str:
    from collections import Counter
    lines = [f"concepts: {len(concepts)}"]
    for axis in AXES:
        sub = [c for c in concepts if c.axis == axis]
        by_source = Counter(s for c in sub for s in c.sources)
        lines.append(f"  {axis}: {len(sub)}  sources={dict(by_source)}")
        subs = Counter(c.sub for c in sub)
        lines.append(f"    subs={dict(subs.most_common(12))}")
    return "\n".join(lines)
