"""Freeze the enriched captions into one table keyed by image id.

Generation runs append one record per request, retry rows that failed, and run
on machines that may not have a tokenizer.  The pass that folds these captions
into the training data needs exactly one measured caption per image, so this
step collapses the raw outputs, measures retained length with the training
tokenizer, drops records whose text is defective, and writes the table the
merge reads.  It is the point where caption production stops changing.

CLI:
    python -m scripts.caption.freeze_captions --out frozen.jsonl \\
        --tokenizer models/Qwen3-0.6B captions_short_final.jsonl captions_ds.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.dataset.caption_prompts import CaptionRequest, check_caption

from .consolidate import load


def freeze(records, tokenizer) -> tuple:
    """Measure and gate one record per image, keeping the order of first sight.

    Every row of the table carries the same fields, so a row that never got
    text says so with ``accepted=False`` and an empty caption rather than by
    lacking the fields the merge reads.
    """
    frozen = []
    order = []
    seen = set()
    for key, record in records.items():
        image_id = record["image_id"]
        if image_id in seen:
            raise ValueError(f"{image_id} was captioned by more than one model")
        seen.add(image_id)
        measured = dict(record)
        measured.pop("usage", None)
        measured.pop("request_key", None)
        measured.pop("image_fingerprint", None)
        if record.get("error") or not record.get("text"):
            measured["text"] = record.get("text") or ""
            measured["retained_tokens"] = 0
            measured["in_band"] = False
            measured["accepted"] = False
            measured["reject_reasons"] = ["no text"]
            measured["hedging_hits"] = []
            measured["evaluation_hits"] = []
            measured["frozen_reason"] = "no text"
            frozen.append(measured)
            order.append(image_id)
            continue
        # Only the band matters here: it sets the window the length gate
        # compares against.  The prompt is not rebuilt, so the target position
        # inside the band is irrelevant.
        request = CaptionRequest(
            image_id=image_id,
            language=record["language"],
            format=record["format"],
            band=record["band"],
            domain=record.get("domain") or "generic",
        )
        check = check_caption(record["text"], request, tokenizer)
        measured["retained_tokens"] = check.retained_tokens
        measured["in_band"] = check.in_band
        measured["accepted"] = check.ok
        measured["reject_reasons"] = check.reasons
        measured["hedging_hits"] = check.hedging_hits
        measured["evaluation_hits"] = check.evaluation_hits
        measured["frozen_reason"] = "" if check.ok else ",".join(check.reasons)
        frozen.append(measured)
        order.append(image_id)
    return frozen, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="the frozen caption table JSONL")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--report", default=None, help="optional per-source markdown table")
    parser.add_argument("inputs", nargs="+", help="raw generation outputs (appended files)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    records = load(args.inputs)
    frozen, order = freeze(records, tokenizer)

    kept = [record for record in frozen if record.get("accepted")]
    dropped = [record for record in frozen if not record.get("accepted")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as sink:
        for record in frozen:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(records)} requests -> {len(frozen)} images, {len(kept)} kept, "
          f"{len(dropped)} dropped")
    reasons = Counter(reason.split(":")[0] for record in dropped
                      for reason in (record.get("reject_reasons") or ["no text"]))
    print("drop reasons:", dict(reasons.most_common()))
    lengths = [record["retained_tokens"] for record in kept]
    if lengths:
        print(f"retained tokens: min={min(lengths)} median={statistics.median(lengths):.0f} "
              f"mean={statistics.fmean(lengths):.0f} max={max(lengths)}")
    print("bands:", dict(Counter(record["band"] for record in kept).most_common()))
    print("languages:", dict(Counter(record["language"] for record in kept).most_common()))
    print("models:", dict(Counter(record["model"] for record in kept).most_common()))
    print("prompt versions:", dict(Counter(record["prompt_version"] for record in kept).most_common()))
    print(f"wrote {out_path}")

    if args.report:
        by_source = {}
        for record in kept:
            stats = by_source.setdefault(record["source"], {"n": 0, "tokens": [], "bands": Counter()})
            stats["n"] += 1
            stats["tokens"].append(record["retained_tokens"])
            stats["bands"][record["band"]] += 1
        lines = ["| source | captions | median tokens | 64-255 | 256-511 | 512-895 | 896-1280 |",
                 "|---|---|---|---|---|---|---|"]
        for source, stats in sorted(by_source.items()):
            bands = stats["bands"]
            lines.append(
                f"| {source} | {stats['n']} | {statistics.median(stats['tokens']):.0f} | "
                f"{bands['64-255']} | {bands['256-511']} | {bands['512-895']} | {bands['896-1280']} |"
            )
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
