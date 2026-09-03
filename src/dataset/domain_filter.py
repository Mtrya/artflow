"""Rule-based domain assignment for harvested museum records.

Assigns each metadata record a coarse domain tag used to route samples into the
per-domain pools (see notes/dataset_plan.md Work breakdown item 3). The rules only
use metadata fields; ambiguous leftovers are tagged "other" and can be reclassified
by the VLM at caption time.

Tags: guo_hua (Domain 1 Chinese painting/calligraphy), western (Domain 2/3 Western
art), japanese_print, object (non-painting artifacts), other.

CLI:
    python -m src.dataset.domain_filter --in data/raw/met/metadata.jsonl
"""

import argparse
import json
from collections import Counter
from typing import Dict

_GUOHUA_CLASSES = ("paintings", "painting", "calligraphy", "繪畫", "法書")

# AIC's `classification` is medium-like and lowercase; painting-like values below.
_AIC_PAINTING_CLASSES = (
    "painting", "hanging scroll", "handscroll", "album leaf", "fan", "screen",
    "miniature painting", "ink or chalk wash",
)
_AIC_PAINTING_PREFIXES = ("oil on", "watercolor", "tempera", "acrylic", "gouache")


def _has(text: str, *keywords: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def assign_domain(source: str, rec: Dict) -> str:
    culture = rec.get("culture", "") or ""
    classification = rec.get("classification", "") or ""
    department = rec.get("department", "") or ""

    if source == "npm_tw":
        return "guo_hua"  # 書畫部绘画/法书,全部由 fetcher 按 RegisterType 限定

    if source == "met":
        if _has(culture, "china") and _has(classification.lower(), "painting", "calligraphy"):
            return "guo_hua"
        if department == "European Paintings":
            return "western"
        if _has(culture, "japan") and _has(classification.lower(), "print"):
            return "japanese_print"
        return "object" if not _has(classification.lower(), "painting") else "other"

    if source == "aic":
        # No place_of_origin in harvested records; culture comes from
        # style_title/period ("Chinese (culture or style)", "Japanese ...").
        style = f"{rec.get('style_title', '')} {rec.get('period', '')}"
        cls = classification.lower()
        is_painting = cls in _AIC_PAINTING_CLASSES or cls.startswith(_AIC_PAINTING_PREFIXES)
        if _has(style, "chinese", "yüan", "yuan"):
            return "guo_hua" if is_painting or cls == "" else "object"
        if _has(style, "japanese"):
            return "japanese_print" if is_painting else "object"
        if is_painting:
            return "other"  # culture unknown — VLM reroutes at caption time
        return "object"

    if source == "nga":
        return "western" if classification == "Painting" else "other"

    if source == "fsg":
        if _has(culture, "chin") and _has(classification.lower(), "painting", "calligraphy"):
            return "guo_hua"
        if _has(culture, "japan") and _has(classification.lower(), "painting", "calligraphy", "print"):
            return "japanese_print"
        return "other" if _has(classification.lower(), "painting", "calligraphy") else "object"

    if source == "princeton":
        # Harvest is already restricted to Asian Art {Paintings, Calligraphy} and
        # records carry no culture field: provisional guo_hua, VLM reroutes.
        if _has(classification.lower(), "painting", "calligraphy"):
            return "guo_hua"
        return "other"

    return "other"


def run(meta_path: str) -> None:
    with open(meta_path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    counts: Counter = Counter()
    for r in recs:
        source = r.get("source") or r["image_id"].split("-")[0]
        counts[assign_domain(source, r)] += 1
    total = sum(counts.values())
    print(f"{meta_path}: {total} records")
    for tag, n in counts.most_common():
        print(f"  {tag:15s} {n:6d}  ({n / max(1, total) * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Domain assignment stats for a metadata.jsonl")
    ap.add_argument("--in", dest="meta", required=True)
    args = ap.parse_args()
    run(args.meta)


if __name__ == "__main__":
    main()
