"""Choose which synthetic rows to rescue, and specify each re-caption request.

Runs against the synthetic precompute manifest (``image_id``, ``local_path``,
``captions``, ``bbox``).  Two jobs, one output schema:

* **Pilot sampling.**  ``--stride``/``--per-model`` draw an even sample inside
  each generation model's rows (every Nth row, so the sample spans the run
  instead of clustering at the start).
* **Re-caption plan.**  With ``--verdicts`` the rows that failed the filter are
  dropped and the survivors get the request they will be captioned with.

The differences from the harvested corpora are deliberate.  The generation
prompt that produced a row lives in ``captions[0]`` and must never reach the
caption prompt - the point of the rescue is a caption written from the pixels,
not a restatement of the request - so it is used only to decide the language.
Language follows that first caption (a Chinese prompt row is captioned in
Chinese), format is drawn by hash, and the length bands are apportioned over
the actual survivor list, largest-remainder, so the requested split is met
exactly rather than in expectation.  The domain is ``generic``: a synthetic
portrait may read as a photograph or as a painting, and the generic vocabulary
rules do not assert a medium the caption pass has to guess at.

CLI:
    # the 60-row pilot sample: 30 rows per generation model, every 333rd row
    python -m scripts.caption.synth_rescue_selection \
        --manifest d3_synth.jsonl --out pilot_sample.jsonl \
        --stride 333 --per-model 30

    # the re-caption plan for the rows that passed the filter
    python -m scripts.caption.synth_rescue_selection \
        --manifest d3_synth.jsonl --verdicts verdicts.jsonl --only-ok \
        --out recaption_selection.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.caption.select_rows import parse_band_shares, stable_hash

# The rescue split: most rows short, a long tail worth a near-token-cap
# description.
BAND_SHARES = (("64-255", 0.40), ("256-511", 0.25), ("512-895", 0.25), ("896-1280", 0.10))

# A synthetic d3-synth row is a generated image of people and clothing, not a
# photographed artwork: the generic vocabulary rules fit it, the per-medium
# ones would assert something the image may not show.
DOMAIN = "generic"

_CJK = re.compile(r"[\u4e00-\u9fff]")


def caption_language(captions: Sequence[str]) -> str:
    """Chinese when the row's existing caption uses Chinese, English otherwise."""
    for caption in captions:
        if caption and _CJK.search(caption):
            return "zh"
    return "en"


def load_manifest(path: str, verdicts: Optional[str] = None,
                  only_ok: bool = False) -> List[Dict]:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    if not verdicts:
        return rows
    available: Dict[str, Optional[bool]] = {}
    for line in Path(verdicts).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        available[record["image_id"]] = record.get("ok")
    if only_ok:
        return [row for row in rows if available.get(row["image_id"]) is True]
    return [row for row in rows if row["image_id"] in available]


def generation_model(local_path: str) -> str:
    """The generator that produced the row, from its directory."""
    return local_path.rsplit("/", 1)[0].rsplit("/", 1)[-1]


def sample_rows(rows: List[Dict], stride: int, per_model: int) -> List[Dict]:
    """Every ``stride``-th row of each generation model, capped per model."""
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_model[generation_model(row["local_path"])].append(row)
    sample: List[Dict] = []
    for model in sorted(by_model):
        group = by_model[model]
        taken = group[::stride][:per_model]
        print(f"{model}: {len(group)} rows, {len(group[::stride])} on the stride, "
              f"{len(taken)} taken")
        sample.extend(taken)
    return sample


def apportion(count: int, shares: Sequence[Tuple[str, float]]) -> Dict[str, int]:
    """Integer counts meeting the shares exactly, largest remainder first."""
    exact = {name: count * weight for name, weight in shares}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = count - sum(counts.values())
    ranking = sorted(shares, key=lambda item: exact[item[0]] - counts[item[0]], reverse=True)
    for name, _ in ranking[:remainder]:
        counts[name] += 1
    return counts


def band_cycle(counts: Dict[str, int], order: Sequence[str]) -> List[str]:
    """Interleave the bands so the quotas are met without long runs of one band.

    Each step hands the row to whichever band is furthest below its own quota,
    measured as a fraction of it; fractions compare exactly, so ties have no
    dependence on floating point.
    """
    taken = {name: 0 for name in order}
    cycle: List[str] = []
    for _ in range(sum(counts.values())):
        best = None
        best_deficit = None
        for name in order:
            quota = counts.get(name, 0)
            if not quota or taken[name] >= quota:
                continue
            deficit = Fraction(quota - taken[name], quota)
            if best_deficit is None or deficit > best_deficit:
                best, best_deficit = name, deficit
        taken[best] += 1
        cycle.append(best)
    return cycle


def build_selection(rows: List[Dict], shares, seed: int) -> List[Dict]:
    """One cap-v6 request row per input row, deterministic under ``seed``."""
    counts = apportion(len(rows), shares)
    order = [name for name, _ in shares]
    cycle = band_cycle(counts, order)
    # The cycle is handed out in a pseudo-random row order, so band is not
    # correlated with anything the manifest's own order carries.
    ordered = sorted(rows, key=lambda row: stable_hash(row["image_id"], seed + 1))
    selection = []
    for row, band in zip(ordered, cycle):
        captions = [c for c in (row.get("captions") or []) if isinstance(c, str)]
        selection.append({
            "image_id": row["image_id"],
            "source": row.get("source") or "d3_synth",
            "domain": DOMAIN,
            "local_path": row["local_path"],
            "bbox": row.get("bbox"),
            "width": row.get("width"),
            "height": row.get("height"),
            "artist": row.get("artist"),
            "title": row.get("title"),
            "language": caption_language(captions),
            "format": "structured" if stable_hash(row["image_id"], seed + 3) < 0.5 else "prose",
            "band": band,
        })
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="precompute manifest JSONL")
    parser.add_argument("--out", required=True, help="selection manifest JSONL")
    parser.add_argument("--verdicts", default=None,
                        help="filter verdict JSONL; only its rows are considered")
    parser.add_argument("--only-ok", action="store_true",
                        help="with --verdicts, keep only rows whose verdict is ok")
    parser.add_argument("--stride", type=int, default=0,
                        help="pilot sampling: take every Nth row of each generation model")
    parser.add_argument("--per-model", type=int, default=0,
                        help="pilot sampling: cap the rows taken per generation model")
    parser.add_argument("--band-shares", default=None,
                        help='length bands as "band:weight ..." (default: the rescue split)')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.verdicts, args.only_ok)
    print(f"{len(rows)} rows"
          + (f" after the filter" if args.verdicts else ""))
    if args.stride and args.per_model:
        rows = sample_rows(rows, args.stride, args.per_model)
    shares = parse_band_shares(args.band_shares) if args.band_shares else BAND_SHARES
    selection = build_selection(rows, shares, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in selection:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    models = Counter(generation_model(record["local_path"]) for record in selection)
    languages = Counter(record["language"] for record in selection)
    bands = Counter(record["band"] for record in selection)
    formats = Counter(record["format"] for record in selection)
    total = max(len(selection), 1)
    print(f"generation models: {dict(models)}")
    print(f"languages: {dict(languages)}")
    print(f"formats: {dict(formats)}")
    print("bands: " + "  ".join(
        f"{name} {bands.get(name, 0)} ({bands.get(name, 0) / total:.0%})"
        for name, _ in shares))
    print(f"wrote {len(selection)} rows to {out_path}")


if __name__ == "__main__":
    main()
