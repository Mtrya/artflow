"""Append enriched captions to a precomputed dataset without re-encoding latents.

The training set stores one row per image with the captions known when the
latents were computed.  Enrichment adds a second, longer caption for the rows
that got one, and the within-row selector needs it to sit in the same list as
the caption it competes with.

Re-running precompute would redo the VAE pass on identical images, so this
instead rewrites the caption column of the existing Arrow shards.  Latents and
resolution bucket ids are copied byte for byte and shard names are unchanged.

A precomputed dataset records no image id, only a position, so the caption a
row gets is decided by matching the row's captions back to the manifest it was
precomputed from.  The two are walked together rather than compared position by
position, because precompute drops rows that fail its filters: the stored rows
are a subsequence of the manifest, and by the time it reaches the tenth dropped
row every stored row sits a little earlier than its manifest position.  A
walk that matches caption text in order recovers the mapping for every row it
can find, and the merge refuses to run when too few rows can be found at all.

CLI:
    python -m scripts.data.apply_caption_enrichment \\
        --dataset-dir precomputed_dataset/d3-human@256p \\
        --manifest data/meta/precompute/d3_human.jsonl \\
        --captions data/caption_enrich/production/captions_frozen.jsonl \\
        --out-dir precomputed_enriched/d3-human@256p
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa

from src.dataset.captions import clean_caption

RowKey = Tuple[str, ...]


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: str) -> Tuple[List[RowKey], List[str], List[str]]:
    """Caption keys, image ids and sources, one entry per manifest row.

    Only what the merge needs is kept: a manifest holds the full record of
    every image, and a million of those does not fit in memory twice over.
    """
    keys: List[RowKey] = []
    image_ids: List[str] = []
    sources: List[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            keys.append(tuple(clean_caption(caption) for caption in record["captions"]))
            image_ids.append(record["image_id"])
            sources.append(record["source"])
    return keys, image_ids, sources


def shard_names(dataset_dir: Path) -> List[str]:
    """Shard filenames in the order the loader will read them."""
    state_path = dataset_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    names = [item["filename"] for item in state.get("_data_files", [])]
    if not names:
        names = sorted(path.name for path in dataset_dir.glob("data-*.arrow"))
    if not names:
        raise ValueError(f"no Arrow shards under {dataset_dir}")
    return names


def load_caption_table(path: str) -> Dict[str, Dict]:
    table = {}
    for record in read_jsonl(path):
        table[record["image_id"]] = record
    return table


def row_index(manifest_keys: Sequence[RowKey]) -> Dict[RowKey, List[int]]:
    """Manifest positions per caption set, in ascending position order."""
    index: Dict[RowKey, List[int]] = defaultdict(list)
    for position, key in enumerate(manifest_keys):
        index[key].append(position)
    return index


def align(keys: Sequence[RowKey], index: Dict[RowKey, List[int]],
          window: int) -> Tuple[List[Optional[int]], Counter]:
    """Map stored rows to manifest positions by walking both in order.

    A stored row takes the first manifest position at or after the cursor whose
    captions match, searching no further than ``window`` rows ahead.  The
    window bounds both the cost and the damage a repeated caption can do: the
    precompute filters drop single rows, so every stored row is near its
    manifest position, and a match found a thousand rows away would be a
    different picture that happens to share the text.

    A row that already carries an enriched caption no longer equals its
    manifest row, so a key that does not match is retried without its trailing
    caption.  That keeps a second run over an already-merged dataset aligned
    instead of reporting every enriched row as unmatched.
    """
    stats = Counter()
    positions: List[Optional[int]] = []
    cursor = 0
    for key in keys:
        candidates = index.get(key)
        if candidates is None and len(key) > 1:
            candidates = index.get(key[:-1])
            if candidates is not None:
                stats["matched after dropping a trailing caption"] += 1
        if not candidates:
            positions.append(None)
            stats["unmatched"] += 1
            continue
        at = bisect.bisect_left(candidates, cursor)
        if at >= len(candidates):
            positions.append(None)
            stats["unmatched"] += 1
            continue
        found = candidates[at]
        if found - cursor > window:
            positions.append(None)
            stats["beyond window"] += 1
            continue
        if len(candidates) > 1:
            stats["repeated caption"] += 1
        positions.append(found)
        stats["skipped manifest rows"] += found - cursor
        cursor = found + 1
    return positions, stats


def rewrite_shard(source: Path, destination: Path, positions: Sequence[Optional[int]],
                  enrichment: Sequence[str], cursor: int) -> int:
    """Copy one shard, replacing the caption column.  Returns the new cursor."""
    with source.open("rb") as handle:
        reader = pa.ipc.open_stream(handle)
        schema = reader.schema
        column = schema.get_field_index("captions")
        if column < 0:
            raise ValueError(f"{source.name} has no captions column")
        with destination.open("wb") as sink:
            writer = pa.ipc.new_stream(sink, schema)
            for batch in reader:
                count = batch.num_rows
                stored = batch.column(column).to_pylist()
                merged = []
                for offset, captions in enumerate(stored):
                    position = positions[cursor + offset]
                    extra = enrichment[position] if position is not None else ""
                    # Re-running a merge that already happened must not append
                    # the same caption twice.
                    merged.append(captions + [extra] if extra and extra not in captions
                                  else captions)
                writer.write_batch(
                    batch.set_column(column, schema.field(column),
                                     pa.array(merged, type=schema.field(column).type))
                )
                cursor += count
            writer.close()
    return cursor


def apply(dataset_dir: str, manifest_path: str, captions_path: str, out_dir: str,
          window: int = 500, min_matched: float = 0.98) -> Dict:
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    names = shard_names(dataset_dir)

    manifest_keys, image_ids, sources = load_manifest(manifest_path)
    table = load_caption_table(captions_path)
    if not table:
        raise ValueError(f"{captions_path} holds no captions")

    # Captions are read first: the alignment needs every stored row before any
    # of them is written.
    keys: List[RowKey] = []
    for name in names:
        with (dataset_dir / name).open("rb") as handle:
            for batch in pa.ipc.open_stream(handle):
                column = batch.schema.get_field_index("captions")
                if column < 0:
                    raise ValueError(f"{name} has no captions column")
                keys.extend(tuple(row) for row in batch.column(column).to_pylist())

    positions, stats = align(keys, row_index(manifest_keys), window)
    matched = sum(1 for position in positions if position is not None)
    print(f"{len(keys)} stored rows, {len(manifest_keys)} manifest rows, "
          f"{matched} matched ({matched / len(keys):.2%})")
    print("alignment:", dict(stats.most_common()))
    if matched < min_matched * len(keys):
        raise ValueError(
            f"only {matched / len(keys):.2%} of stored rows could be matched to the "
            "manifest; refusing to caption rows this tool cannot identify"
        )

    # A rejected caption is not a usable caption; the row simply keeps the
    # captions it was precomputed with.
    enrichment = [table[image_id]["text"] if table.get(image_id, {}).get("accepted") else ""
                  for image_id in image_ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for name in names:
        cursor = rewrite_shard(dataset_dir / name, out_dir / name, positions, enrichment, cursor)
        print(f"  {name}: {cursor} rows", flush=True)

    for extra in ("dataset_info.json", "state.json"):
        source = dataset_dir / extra
        if source.is_file():
            shutil.copy2(source, out_dir / extra)

    by_source = Counter()
    for position in positions:
        if position is not None and enrichment[position]:
            by_source[sources[position]] += 1
    summary = {"rows": len(keys), "matched": matched, "enriched": sum(by_source.values()),
               "by_source": dict(by_source.most_common())}
    print(f"{summary['enriched']} rows got an enriched caption "
          f"({summary['enriched'] / len(keys):.1%} of stored rows)")
    print("by source:", summary["by_source"])
    print(f"wrote {out_dir}")
    return summary


def add_captions_to_manifest(manifest_path: str, captions_path: str, out_path: str) -> Dict:
    """Append the enriched caption to each manifest row that has one.

    A dataset that has not been precomputed yet still lives as a manifest, so
    its enrichment joins the row before the latents exist and there is nothing
    to rewrite.  No row is dropped: the caption is added to the captions the row
    already carries.
    """
    table = load_caption_table(captions_path)
    if not table:
        raise ValueError(f"{captions_path} holds no captions")
    stats = Counter()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(manifest_path).open(encoding="utf-8") as source, out.open("w", encoding="utf-8") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record = table.get(row["image_id"])
            captions = list(row.get("captions") or [])
            if record and record.get("accepted") and record["text"] not in captions:
                row = dict(row)
                row["captions"] = captions + [record["text"]]
                stats["enriched"] += 1
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
    print(f"{stats['rows']} manifest rows, {stats['enriched']} got an enriched caption "
          f"({stats['enriched'] / max(stats['rows'], 1):.1%})")
    print(f"wrote {out}")
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="the manifest the dataset was precomputed from, or the "
                             "manifest to be written when --out-manifest is used")
    parser.add_argument("--captions", required=True,
                        help="frozen caption table keyed by image id")
    parser.add_argument("--dataset-dir",
                        help="a precomputed dataset whose caption column is rewritten")
    parser.add_argument("--out-dir", help="where the rewritten dataset goes")
    parser.add_argument("--out-manifest",
                        help="write a manifest instead, with the captions appended")
    parser.add_argument("--window", type=int, default=500,
                        help="how far ahead of the cursor a matching manifest row may sit")
    parser.add_argument("--min-matched", type=float, default=0.98,
                        help="abort below this share of stored rows matched to the manifest")
    args = parser.parse_args()

    if args.out_manifest:
        if args.dataset_dir or args.out_dir:
            parser.error("--out-manifest writes a manifest; drop --dataset-dir/--out-dir")
        add_captions_to_manifest(args.manifest, args.captions, args.out_manifest)
        return
    if not (args.dataset_dir and args.out_dir):
        parser.error("give --dataset-dir with --out-dir, or --out-manifest")
    apply(args.dataset_dir, args.manifest, args.captions, args.out_dir,
          window=args.window, min_matched=args.min_matched)


if __name__ == "__main__":
    main()
