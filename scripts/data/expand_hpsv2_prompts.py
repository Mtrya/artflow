"""Expand each HPS v2 seed prompt into six drawable training prompts.

The seeds are short aesthetic prompts.  Six variants are drawn from each, one
per *axis*, so a batch of 3,200 seeds becomes ~19,200 prompts that stay near
their seed while covering different subjects, media, light and framings - and,
on the adjacent axis, the neighbouring concept of one element, which is how
rare style x subject combinations enter the set.

The variants are written to the caption contract of this corpus
(``src/dataset/caption_prompts.py``): the same length bands, the same
prose/structured formats, the same bans on hedging, evaluation and invented
attribution.  A variant that fails the contract checks is re-asked, and only
the failed axes are re-asked, so one bad variant does not cost six.

Language, format and length band come from the same hash assignment
``scripts/caption/select_rows.py`` uses for harvested rows, with independent
draws, so no language or length can become confounded with what a seed happened
to be about.

CLI:
    python -m scripts.data.expand_hpsv2_prompts \\
        --seeds data/hpsv2/seeds.jsonl --out data/hpsv2/variants.jsonl --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

from src.dataset.caption_client import CaptionClient, Response, summarise
from src.dataset.caption_prompts import (
    EVALUATION_PHRASES, HEDGING_PHRASES, CaptionRequest, check_caption,
    normalize_caption, _PROSE_RULES, _STRUCTURED_RULES,
)

PROMPT_VERSION = "expand-hpsv2-v1"
MODEL = "zenmux:deepseek-flash"

AXES = ("setting", "subject", "medium", "light", "vantage", "adjacent")

AXIS_RULES = {
    "setting": "keep the seed's subject and medium, move the scene somewhere "
               "else: another place, another kind of room or landscape, another "
               "season or era that suits it",
    "subject": "keep the seed's setting, medium and mood, and put a different "
               "but equally concrete subject in it (another person, animal, "
               "object, vehicle or building)",
    "medium": "keep the subject and the scene, change how the picture is made - "
              "photograph, oil painting, ink painting, watercolour, pencil "
              "drawing, print, digital painting, 3D render - and give the new "
              "medium its visible traits",
    "light": "keep subject, scene and medium, and change the light: another "
             "time of day, another weather, indoor instead of outdoor light, "
             "warm instead of cold",
    "vantage": "keep subject, scene and medium, and change the framing: wide "
               "view, close-up, elevated view, low angle, or a detail of one "
               "part of the scene",
    "adjacent": "keep the seed's core and swap exactly one element for its "
                "nearest neighbour in the same family - the neighbouring "
                "subject, garment, building type, tool, plant or animal - never "
                "a distant one",
}

# The length mix of the Pexels caption batch (notes/dataset_plan.md, Caption
# 增补契约), so synthetic rows carry the same distribution as harvested ones.
BAND_SHARES = (("64-255", 0.4488), ("256-511", 0.2753),
               ("512-895", 0.1658), ("896-1280", 0.1101))
CONTRACT_SEED = 42

# Attribution the caption contract forbids: a caption states what is visible,
# not who supposedly made it.  Seeds name artists, studios and platforms; the
# variants describe the visible traits those names stand for instead.
ATTRIBUTION_NAME = re.compile(r"\b[Bb]y\s+[A-Z][a-zà-ÿ]{2,}")
ATTRIBUTION_WORD = re.compile(
    r"(?i)(\bstyle of\b|inspired by|\bartstation\b|\bdeviantart\b|"
    r"\binstagram\b|\breddit\b|\br/\w+|\bpinterest\b|\btumblr\b|\bmidjourney\b|"
    r"\bunreal engine\b|\bblizzard\b|\bpixar\b|\bdisney\b)")
OPENER = re.compile(
    r"(?i)^\s*(this|the)\s+(image|picture|photo|photograph|painting|artwork|"
    r"drawing|illustration)\b|^\s*(这张|这幅|该)(图片|图画|画作|作品|照片|插画)")
CJK = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")


def stable_hash(value: str, seed: int) -> float:
    """Deterministic uniform draw in [0, 1) for a string under a seed."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_contract(variant_key: str) -> CaptionRequest:
    """Language, format and length band for one variant, drawn by hash."""
    language = "zh" if stable_hash(variant_key, CONTRACT_SEED + 2) < 0.5 else "en"
    fmt = "structured" if stable_hash(variant_key, CONTRACT_SEED + 3) < 0.5 else "prose"
    draw = stable_hash(variant_key, CONTRACT_SEED + 4)
    band = BAND_SHARES[-1][0]
    cumulative = 0.0
    for name, share in BAND_SHARES:
        cumulative += share
        if draw < cumulative:
            band = name
            break
    return CaptionRequest(image_id=variant_key, language=language, format=fmt,
                          band=band, domain="generic")


def plan_for(seed_id: str) -> List[Dict]:
    """The six requests one seed expands into, in axis order."""
    plan = []
    for index, axis in enumerate(AXES):
        plan.append({"axis": axis, "index": index,
                     "request": assign_contract(f"{seed_id}-{index}")})
    return plan


def variant_prompt(seed: str, jobs: List[Dict]) -> str:
    """The request for the variants of one seed that are still missing."""
    lines = []
    for job in jobs:
        request: CaptionRequest = job["request"]
        unit = "words" if request.language == "en" else "字"
        low, high = request.unit_range()
        fmt = (_STRUCTURED_RULES if request.format == "structured"
               else _PROSE_RULES)[request.language]
        lines.append(
            f"{job['index'] + 1}. axis={job['axis']} - {AXIS_RULES[job['axis']]}\n"
            f"   language: {request.language} (write only in this language)\n"
            f"   length: about {request.target_units()} {unit}, acceptable "
            f"{low}-{high} {unit}\n"
            f"   format: {fmt}")
    block = "\n".join(lines)

    hedging = ", ".join(f'"{p}"' for p in HEDGING_PHRASES[:14])
    evaluation = ", ".join(f'"{p}"' for p in EVALUATION_PHRASES[:12])
    return f"""You expand one seed prompt into training prompts for a text-to-image model.

Seed prompt (the reference for relation and difference; never copy its wording):
{seed}

Write {len(jobs)} new prompt(s), one per variant below. Keep each a step away from the seed along its axis, and make the prompts differ from each other by more than a word:

{block}

Every prompt you write:
- describes one specific picture that a camera or a painter could produce, and nothing else;
- keeps the seed's aesthetic core: the mood, the palette family and roughly the level of detail the seed implies;
- names the medium (photograph, oil painting, watercolour, pencil drawing, print, digital painting, ink painting, 3D render) early, in the words of that medium, and gives the visible traits that make the medium readable;
- uses concrete nouns and plain words for materials, colours, light, texture, and for where things sit;
- describes people by appearance, dress and the region their features point to, naming a garment by its own name when it has one (hanfu, kimono, sari, cheongsam, kaftan, abaya, hijab);
- stays drawable: no legible text unless the seed is about signage, no diagrams, no dense ornament.

Never write:
- doubt or guessing: {hedging};
- evaluation or art criticism: {evaluation};
- names of artists, studios, brands, websites or platforms - where the seed names one, describe the visible traits it stands for instead;
- an opening that points at the picture ("This image shows", "The picture depicts", "这张图片展示");
- instructions about adding text, watermarks, borders or frames;
- any mention of the seed, of this task, or of these rules.

Return JSON only, no code fence:
{{"variants": [{{"axis": "<axis>", "text": "<the prompt>"}}, ...]}}
with exactly {len(jobs)} item(s), in the order given above, each `axis` copied from that item."""


def validate(text: str, request: CaptionRequest, tokenizer) -> List[str]:
    """Reasons this text cannot be used; empty when it is acceptable."""
    cleaned = normalize_caption(text)
    if not cleaned or len(cleaned) < 20:
        return ["too short"]
    problems = list(check_caption(cleaned, request, tokenizer).reasons)
    if request.language == "zh":
        if len(CJK.findall(cleaned)) / max(len(cleaned), 1) < 0.3:
            problems.append("not written in Chinese")
    elif CJK.search(cleaned):
        problems.append("not written in English")
    if OPENER.search(cleaned):
        problems.append("opens by pointing at the picture")
    if ATTRIBUTION_NAME.search(cleaned) or ATTRIBUTION_WORD.search(cleaned):
        problems.append("names an artist, studio or platform")
    lowered = cleaned.lower()
    for phrase in HEDGING_PHRASES:
        if phrase in lowered:
            problems.append(f"hedging: {phrase}")
            break
    for phrase in EVALUATION_PHRASES:
        if phrase in lowered:
            problems.append(f"evaluation: {phrase}")
            break
    return problems


def content_words(text: str) -> set:
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", text.lower())
    return {token for token in value.split() if len(token) > 2}


def near_duplicate(text: str, others: List[str]) -> bool:
    """True when ``text`` describes the same picture as one already accepted."""
    mine = content_words(text)
    if len(mine) <= 12:
        return False
    for other in others:
        theirs = content_words(other)
        if len(theirs) <= 12:
            continue
        if len(mine & theirs) / min(len(mine), len(theirs)) > 0.92:
            return True
    return False


def parse_variants(text: str) -> List[Dict[str, str]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:160]!r}")
    chunk = text[start:end + 1]
    try:
        payload = json.loads(chunk)
    except ValueError:
        # A newline inside a string is the one defect the generator produces
        # often enough to be worth accepting; anything else still raises.
        payload = json.loads(chunk, strict=False)
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"no variants list: {str(payload)[:160]!r}")
    return [{"axis": str(item.get("axis", "")).strip(), "text": str(item["text"]).strip()}
            for item in variants
            if isinstance(item, dict) and item.get("text")]


async def expand_seed(client: CaptionClient, http: httpx.AsyncClient, seed: Dict,
                      plan: List[Dict], tokenizer, extra: Dict,
                      attempts: int = 3) -> Tuple[Dict, List[Response]]:
    """All six variants of one seed, re-asking for the axes that failed."""
    accepted: Dict[int, Dict] = {}
    failures: Dict[int, str] = {}
    responses: List[Response] = []
    for attempt in range(attempts):
        jobs = [job for job in plan if job["index"] not in accepted]
        if not jobs:
            break
        prompt = variant_prompt(seed["prompt"], jobs)
        response = await client.generate_text(
            http, prompt=prompt, prompt_version=PROMPT_VERSION, max_tokens=8192,
            temperature=0.8, extra=extra)
        responses.append(response)
        if response.error:
            failures = {job["index"]: f"api: {response.error}" for job in jobs}
            continue
        try:
            variants = parse_variants(response.text)
        except ValueError as exc:
            failures = {job["index"]: f"parse: {exc}" for job in jobs}
            continue
        by_index = {job["index"]: job for job in jobs}
        seen = [item["text"] for item in accepted.values()]
        for item in variants:
            index = next((i for i, job in by_index.items()
                          if job["axis"] == item["axis"]), None)
            if index is None or index in accepted:
                continue
            request: CaptionRequest = plan[index]["request"]
            problems = validate(item["text"], request, tokenizer)
            cleaned = normalize_caption(item["text"])
            if not problems and near_duplicate(cleaned, seen):
                problems = ["too close to another variant"]
            if problems:
                failures[index] = "; ".join(problems)
                continue
            accepted[index] = {"axis": plan[index]["axis"], "text": cleaned,
                               "language": request.language, "format": request.format,
                               "band": request.band,
                               "prompt_version": request.prompt_version}
            seen.append(cleaned)
            failures.pop(index, None)

    record = {
        "seed_id": seed["seed_id"],
        "seed_prompt": seed["prompt"],
        "variants": [accepted[i] for i in sorted(accepted)],
        "missing": [{"axis": plan[i]["axis"], "reason": failures.get(i, "not returned")}
                    for i in range(len(plan)) if i not in accepted],
    }
    return record, responses


async def run(args) -> None:
    from transformers import AutoTokenizer

    seeds = [json.loads(line) for line in Path(args.seeds).open(encoding="utf-8")]
    if args.limit:
        seeds = seeds[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    client = CaptionClient(cache_dir=args.cache, model=args.model,
                           concurrency=args.concurrency)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    responses: List[Response] = []
    done = 0
    with out.open("w", encoding="utf-8") as sink:
        async with httpx.AsyncClient() as http:
            extra = json.loads(args.extra) if args.extra else None

            async def one(seed: Dict):
                return await expand_seed(client, http, seed, plan_for(seed["seed_id"]),
                                         tokenizer, extra)

            tasks = [asyncio.create_task(one(seed)) for seed in seeds]
            for task in asyncio.as_completed(tasks):
                record, calls = await task
                responses.extend(calls)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(seeds)} seeds", flush=True)

    variants = [json.loads(line) for line in out.open(encoding="utf-8")]
    flat = [v for record in variants for v in record["variants"]]
    missing = sum(len(record["missing"]) for record in variants)
    bands = Counter(v["band"] for v in flat)
    languages = Counter(v["language"] for v in flat)
    formats = Counter(v["format"] for v in flat)
    print(f"{len(flat)} variants from {len(variants)} seeds; {missing} axes failed")
    print("  bands: " + "  ".join(f"{k} {v}" for k, v in bands.most_common()))
    print("  languages: " + "  ".join(f"{k} {v}" for k, v in languages.most_common()))
    print("  formats: " + "  ".join(f"{k} {v}" for k, v in formats.most_common()))
    print("  cost: $" + f"{summarise(responses)['cost_usd']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="data/hpsv2/seeds.jsonl")
    parser.add_argument("--out", default="data/hpsv2/variants.jsonl")
    parser.add_argument("--cache", default="data/hpsv2/expand_cache")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--extra", default=None,
                        help="JSON merged into every request, e.g. "
                             "'{\"reasoning_effort\": \"none\"}'")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
