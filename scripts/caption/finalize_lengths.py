"""Measure generated captions with the training tokenizer and apply the gate.

Generation can run on a machine that has no tokenizer: it records the text and
a character count.  This pass, run where the tokenizer lives, computes retained
tokens under the training prompt contract, applies the length window and the
other acceptance rules, and rewrites the file in place.

CLI:
    python -m scripts.caption.finalize_lengths --captions captions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.dataset.caption_prompts import CaptionRequest, check_caption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", default=None, help="default: rewrite in place")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source = Path(args.captions)
    out_path = Path(args.out) if args.out else source.with_suffix(source.suffix + ".tmp")
    stats = Counter()
    lengths = []
    with source.open(encoding="utf-8") as handle, out_path.open("w", encoding="utf-8") as sink:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["error"] += 1
                continue
            request = CaptionRequest(image_id=record["image_id"],
                                     language=record["language"],
                                     format=record["format"],
                                     band=record["band"],
                                     domain=record.get("domain") or "generic")
            check = check_caption(record["text"], request, tokenizer)
            record["retained_tokens"] = check.retained_tokens
            record["in_band"] = check.in_band
            record["accepted"] = check.ok
            record["reject_reasons"] = check.reasons
            record["evaluation_hits"] = check.evaluation_hits
            lengths.append(check.retained_tokens)
            stats["accepted" if check.ok else "rejected"] += 1
            stats[f"band:{record['band']}"] += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not args.out:
        out_path.replace(source)
    lengths.sort()
    if lengths:
        print(f"retained tokens: min {lengths[0]} median {lengths[len(lengths) // 2]} "
              f"max {lengths[-1]}")
    print(dict(stats))


if __name__ == "__main__":
    main()
