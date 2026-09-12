"""Choose which rows receive a new long caption, and specify each request.

Runs against the precompute manifests on the shared disk (one JSONL per source,
with ``image_id``, ``local_path``, ``captions``, ``bbox``).  Output is a
selection manifest: one line per row to enrich, carrying the row identity and
the exact request to send (language, format, length band, domain).

Selection rules, in the order they are applied:

1. **Budget by draw share.**  Each dataset receives a share of the row budget
   equal to its sampling weight in the training mixture, so enrichment lands
   where the trainer actually draws samples.
2. **Specialised top-up.**  An extra share of the budget is spent on Chinese
   painting on top of its draw share, because that is the domain where the
   corpus can carry the most domain-specific vocabulary.
3. **Within a dataset, stratify.**  Rows are bucketed by sub-source, aspect
   shape, and existing caption length, then drawn in a fixed pseudo-random
   order per bucket, so a partial draw still spans the source.
4. **Length band by available detail.**  Larger images and non-detail views get
   the longer bands; small or cropped views are not asked for 2000 tokens of
   description they cannot support.
5. **Language and format by hash.**  Deterministic, and independent of the
   band, so language/format are not confounded with length or source.

Everything is derived from a seed, so the same command reproduces the same
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.dataset.caption_prompts import LENGTH_BANDS

# Length bands and their intended share of accepted additions
# (notes/stage3_5_plan.md section 3.3).
# A dataset whose captions serve a different purpose can pass its own split on
# the command line; the shares are normalised, so they need not sum to one.
BAND_SHARES = (("256-511", 0.55), ("512-895", 0.30), ("896-1280", 0.15))


def parse_band_shares(spec: str) -> Tuple[Tuple[str, float], ...]:
    """Read ``"64-255:0.4 256-511:0.25 ..."`` into normalised band shares."""
    shares = []
    for part in spec.split():
        name, _, weight = part.rpartition(":")
        if name not in LENGTH_BANDS:
            raise SystemExit(f"unknown length band {name!r}")
        shares.append((name, float(weight)))
    total = sum(weight for _, weight in shares)
    if total <= 0:
        raise SystemExit("band shares must sum to something positive")
    return tuple((name, weight / total) for name, weight in shares)

# Domains that drive the specialised top-up and the vocabulary rules.
DOMAIN_BY_SOURCE = {
    "d1_npm_tw": "chinese_painting",
    "d1_met_china": "chinese_painting",
    "d1_princeton": "chinese_painting",
    "d1_fsg": "chinese_painting",
    "d1_aic_china": "chinese_painting",
    "d2_wikiart": "western_art",
    "d2_met_impression": "western_art",
    "d2_met_portrait": "western_art",
    "d2_nga_impression": "western_art",
    "d2_nga_paintings_all": "western_art",
    "d2_rijksmuseum": "western_art",
    "d4_vintage": "photograph",
    "d4_pd12m": "photograph",
    "d4_relaion": "generic",
    "d4_zimage": "generic",
    "d4_megalith": "photograph",
    "d4_inat": "photograph",
    "d3_human_recaption": "photograph",
    "d3_people_supp": "photograph",
    "d3_pexels": "photograph",
}


def stable_hash(value: str, seed: int) -> float:
    """Deterministic uniform draw in [0, 1) for a string under a seed."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def aspect_bucket(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height or width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    if ratio < 0.8:
        return "portrait"
    if ratio > 1.25:
        return "landscape"
    return "square"


def length_bucket(captions: List[str]) -> str:
    """Coarse proxy for how much text the row already carries."""
    if not captions:
        return "none"
    longest = max(len(c) for c in captions)
    if longest < 120:
        return "short"
    if longest < 260:
        return "medium"
    return "long"


@dataclass
class Row:
    image_id: str
    source: str
    manifest: str
    local_path: str
    captions: List[str]
    width: Optional[int]
    height: Optional[int]
    bbox: Optional[List[float]]
    artist: Optional[str] = None
    title: Optional[str] = None

    @property
    def domain(self) -> str:
        return DOMAIN_BY_SOURCE.get(self.source, "generic")

    @property
    def pixels(self) -> int:
        if self.width and self.height:
            return int(self.width) * int(self.height)
        return 0

    @property
    def is_detail(self) -> bool:
        return "detail" in self.image_id.lower()


def load_artifact_flags(path: Optional[str]) -> Dict[str, List[str]]:
    """Rows whose photograph contains something that is not the artwork.

    Enriching such a row would pay for a caption that either describes the
    colour chart or silently omits a large part of the training view, so they
    are dropped from the enrichment set by default.
    """
    if not path:
        return {}
    flags: Dict[str, List[str]] = {}
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("artifacts"):
            flags[record["image_id"]] = record["artifacts"]
    return flags


def load_rows(manifest_dir: Path, datasets: Optional[Iterable[str]],
              include_eval: bool = False) -> List[Row]:
    """Rows of the training manifests, one per image.

    The held-out manifest is skipped by default: its rows are not part of the
    training mixture, so spending the enrichment budget on them would take
    captions away from rows the trainer actually draws.  ``include_eval`` is
    for the separate job of giving the held-out rows a second caption at a
    different length, which is what makes an evaluation able to report whether
    the model follows long text at all.
    """
    rows: List[Row] = []
    for path in sorted(manifest_dir.glob("*.jsonl")):
        name = path.stem
        if name == "light_eval" and not include_eval:
            continue
        if datasets and name not in datasets:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(Row(
                    image_id=raw["image_id"],
                    source=raw.get("source") or name,
                    manifest=name,
                    local_path=raw["local_path"],
                    captions=[c for c in (raw.get("captions") or []) if isinstance(c, str)],
                    width=raw.get("width"),
                    height=raw.get("height"),
                    bbox=raw.get("bbox"),
                    artist=raw.get("artist"),
                    title=raw.get("title"),
                ))
    return rows


def dataset_domain(rows: Iterable[Row]) -> Dict[str, str]:
    """Domain of each dataset, decided by the rows it actually contains.

    The mixture refers to datasets by manifest name (``d1``), while the domain
    rules are written per sub-source (``d1_npm_tw``).  Deriving the domain from
    the rows keeps the two consistent when a dataset is renamed or a sub-source
    is added, instead of relying on the names lining up.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row.manifest][row.domain] += 1
    return {name: counter.most_common(1)[0][0] for name, counter in counts.items()}


def allocate(total: int, mix: Dict[str, float], specialized_share: float,
             dataset_rows: Dict[str, int], dataset_domains: Dict[str, str]) -> Dict[str, int]:
    """Row budget per dataset: draw share plus a Chinese-painting top-up."""
    broad = total * (1.0 - specialized_share)
    budgets = {name: broad * weight for name, weight in mix.items()}

    chinese = [name for name in mix if dataset_domains.get(name) == "chinese_painting"]
    chinese_weight = sum(mix[name] for name in chinese)
    top_up = total * specialized_share
    if chinese_weight <= 0:
        # No Chinese painting in the mix: spread the top-up by draw share so
        # the budget is still spent.
        for name in budgets:
            budgets[name] += top_up * mix[name]
    else:
        for name in chinese:
            budgets[name] += top_up * (mix[name] / chinese_weight)

    # A dataset cannot supply more rows than it has.
    capped = {name: min(int(round(value)), dataset_rows.get(name, 0))
              for name, value in budgets.items()}
    shortfall = int(round(total)) - sum(capped.values())
    if shortfall > 0:
        # Redistribute whatever the caps left over, proportionally, repeatedly.
        for _ in range(4):
            spare = {name: dataset_rows.get(name, 0) - capped[name]
                     for name in capped if dataset_rows.get(name, 0) > capped[name]}
            if not spare or shortfall <= 0:
                break
            spare_total = sum(spare.values())
            for name, room in spare.items():
                add = min(room, int(round(shortfall * room / spare_total)))
                capped[name] += add
            shortfall = int(round(total)) - sum(capped.values())
    return capped


def pick_rows(rows: List[Row], count: int, seed: int) -> List[Row]:
    """Draw ``count`` rows, spreading the draw over source/shape/length strata."""
    if count <= 0:
        return []
    strata: Dict[Tuple[str, str, str], List[Row]] = defaultdict(list)
    for row in rows:
        key = (row.source, aspect_bucket(row.width, row.height), length_bucket(row.captions))
        strata[key].append(row)

    for key, group in strata.items():
        group.sort(key=lambda r: stable_hash(r.image_id, seed))

    # Allocate the draw across strata proportionally, then round-robin so a
    # small stratum still contributes before a large one is exhausted.
    total_rows = len(rows)
    quota = {key: count * len(group) / total_rows for key, group in strata.items()}
    chosen: List[Row] = []
    taken: Counter = Counter()
    for key in sorted(strata, key=lambda k: stable_hash(str(k), seed)):
        group = strata[key]
        want = min(len(group), int(quota[key]) + 1)
        chosen.extend(group[:want])
        taken[key] = want
    # Top up to the exact count from the remaining rows, largest strata first.
    if len(chosen) < count:
        remainder = []
        for key, group in strata.items():
            remainder.extend(group[taken[key]:])
        remainder.sort(key=lambda r: stable_hash(r.image_id, seed + 1))
        chosen.extend(remainder[:count - len(chosen)])
    return chosen[:count]


def assign_request(row: Row, index: int, seed: int, specialized: bool,
                   band_shares: Tuple[Tuple[str, float], ...] = BAND_SHARES) -> Dict[str, object]:
    """Language, format and length band for one selected row."""
    draw = stable_hash(row.image_id, seed + 2)
    # Language: 50:50 for the broad portion, 80:20 Chinese-first for the
    # specialised portion.
    zh_share = 0.80 if specialized and row.domain == "chinese_painting" else 0.50
    language = "zh" if draw < zh_share else "en"

    fmt = "structured" if stable_hash(row.image_id, seed + 3) < 0.5 else "prose"

    # Length band, biased by how much detail the row can support.  A detail
    # crop or a small image is capped at the second-longest band in use, since
    # it cannot fill a long description with content that is actually there.
    band_draw = stable_hash(row.image_id, seed + 4)
    cumulative = 0.0
    band = band_shares[-1][0]
    for name, share in band_shares:
        cumulative += share
        if band_draw < cumulative:
            band = name
            break
    fallback = band_shares[-2][0] if len(band_shares) > 1 else band
    if band == band_shares[-1][0] and (row.is_detail or 0 < row.pixels < 512 * 512):
        band = fallback

    return {
        "image_id": row.image_id,
        "source": row.source,
        "domain": row.domain,
        "local_path": row.local_path,
        "bbox": row.bbox,
        "width": row.width,
        "height": row.height,
        # Carried so the caption can name the artist or the work when it reads
        # naturally; the caption prompt may leave them out.
        "artist": row.artist,
        "title": row.title,
        "existing_captions": len(row.captions),
        "language": language,
        "format": fmt,
        "band": band,
        "specialized": specialized,
        "index": index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True,
                        help="directory of precompute manifest JSONL files")
    parser.add_argument("--out", required=True, help="selection manifest JSONL")
    parser.add_argument("--target-total", type=int, required=True)
    parser.add_argument("--mix", required=True,
                        help='space separated "dataset:weight" (names match manifest stems)')
    parser.add_argument("--specialized-share", type=float, default=0.30)
    parser.add_argument("--band-shares", default=None,
                        help='length bands as "band:weight ..." (default: the '
                             "enrichment split); normalised, so it need not sum to one")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", default=None,
                        help="comma separated manifest stems to restrict to (default: all)")
    parser.add_argument("--artifact-index", default=None,
                        help="image_id/artifacts JSONL; flagged rows are excluded")
    parser.add_argument("--keep-flagged", action="store_true",
                        help="enrich artifact-flagged rows anyway")
    parser.add_argument("--include-eval", action="store_true",
                        help="also read light_eval.jsonl; used to give the held-out "
                             "rows a caption variant, not to enrich the training mixture")
    parser.add_argument("--flagged-only", action="store_true",
                        help="emit exactly the artifact-flagged rows, no budget or "
                             "stratification; used to build the re-crop pass")
    args = parser.parse_args()

    mix = {}
    for part in args.mix.split():
        name, weight = part.rsplit(":", 1)
        mix[name.replace("_", "-")] = float(weight)
    total_weight = sum(mix.values())
    mix = {name: weight / total_weight for name, weight in mix.items()}

    datasets = set(args.datasets.split(",")) if args.datasets else None
    rows = load_rows(Path(args.manifest_dir), datasets, include_eval=args.include_eval)
    flagged = load_artifact_flags(args.artifact_index)
    if args.flagged_only:
        selected = [row for row in rows if row.image_id in flagged]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps({
                    "image_id": row.image_id, "source": row.source,
                    "local_path": row.local_path, "bbox": row.bbox,
                    "width": row.width, "height": row.height,
                    "artifacts": flagged[row.image_id],
                }, ensure_ascii=False) + "\n")
        print(f"{len(selected)} artifact-flagged rows written to {out_path}")
        return
    if flagged and not args.keep_flagged:
        before = len(rows)
        rows = [row for row in rows if row.image_id not in flagged]
        print(f"excluded {before - len(rows)} artifact-flagged rows "
              f"({(before - len(rows)) / max(before, 1):.1%})")

    per_dataset: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        per_dataset[_dataset_for(row)].append(row)

    available = {name: len(group) for name, group in per_dataset.items()}
    budgets = allocate(args.target_total, mix, args.specialized_share, available,
                       dataset_domain(rows))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    band_shares = parse_band_shares(args.band_shares) if args.band_shares else BAND_SHARES
    written = Counter()
    specialized_counts = Counter()
    band_counts = Counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for name in sorted(mix):
            group = per_dataset.get(name, [])
            budget = budgets.get(name, 0)
            chosen = pick_rows(group, budget, args.seed)
            # How much of this dataset's budget is the specialised top-up.
            broad_budget = args.target_total * (1 - args.specialized_share) * mix[name]
            for position, row in enumerate(chosen):
                specialized = position >= int(round(broad_budget))
                record = assign_request(row, written[name], args.seed, specialized,
                                        band_shares)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written[name] += 1
                band_counts[record["band"]] += 1
                if specialized:
                    specialized_counts[name] += 1

    print(f"{'dataset':>22s} {'available':>10s} {'budget':>9s} {'selected':>9s} {'specialised':>12s}")
    for name in sorted(mix):
        print(f"{name:>22s} {available.get(name, 0):10d} {budgets.get(name, 0):9d} "
              f"{written[name]:9d} {specialized_counts[name]:12d}")
    print(f"{'TOTAL':>22s} {len(rows):10d} {sum(budgets.values()):9d} {sum(written.values()):9d} "
          f"{sum(specialized_counts.values()):12d}")
    total_rows = sum(band_counts.values())
    print("length bands: " + "  ".join(
        f"{band} {band_counts[band]} ({band_counts[band] / max(total_rows, 1):.1%})"
        for band, _ in band_shares))
    print(f"wrote {out_path}")


def _dataset_for(row: Row) -> str:
    """Budget key for a row: the manifest it came from, hyphenated.

    A manifest file is one training dataset (``d2_museum.jsonl`` holds five
    museum sub-sources, ``d3_people.jsonl`` holds five people sub-sources), so
    the file is what the mixture weights refer to.  ``row.source`` is the finer
    sub-source used for domain rules and stratification.
    """
    return row.manifest.replace("_", "-")


if __name__ == "__main__":
    main()
