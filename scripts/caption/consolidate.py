"""Collapse an append-only caption output into one usable record per request.

``generate.py`` appends a record per request and never rewrites, so a row that
was retried appears more than once, and a row whose first attempt failed before
a later one succeeded keeps the error text next to a good caption.  Both would
misreport coverage and could carry a stale error into the merge.

This pass keeps, for each (image, model), the last record that carries text; if
none of a row's records carry text, the last one is kept as it is, so a
provider-side refusal stays visible instead of silently disappearing.

CLI:
    python -m scripts.caption.consolidate --out consolidated.jsonl \\
        captions.jsonl captions_retry.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(paths) -> dict:
    records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "image_id" not in record:
                    continue
                key = (record["image_id"], record.get("model"))
                previous = records.get(key)
                # Keep the last record that has text; fall back to the last one.
                if previous is None or record.get("text") or not previous.get("text"):
                    records[key] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("captions", nargs="+")
    args = parser.parse_args()

    records = load(args.captions)
    kept = Counter()
    with Path(args.out).open("w", encoding="utf-8") as sink:
        for (image_id, _model), record in sorted(records.items()):
            if record.get("text"):
                record.pop("error", None)
                record.pop("transient", None)
                kept["with_text"] += 1
            else:
                kept["without_text"] += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(records)} records -> {args.out}")
    print(f"  with text    {kept['with_text']}")
    print(f"  without text {kept['without_text']}")


if __name__ == "__main__":
    main()
