"""Caption token-length distribution of a dataset mix.

Loads each precomputed dataset in the mix, samples up to --max-rows rows per
entry, tokenizes every caption with the real Qwen3 tokenizer, and writes a
JSON file with:
  - a per-caption token-count histogram (all captions)
  - per-sample shortest- and longest-caption histograms: the two endpoints of
    the caption-selection curriculum (position 0.0 prefers the shortest
    caption in a row, 1.0 the longest), which bound what the sampler can pick
  - p50/p90/p99/max token counts per entry and overall

The raw counts are the right key for the length-bucket plan: in training the
chat-template prefix is dropped (encode_text trims the first DROP_IDX=38
tokens) before padding, so the in-train sequence length is roughly the raw
caption token count plus a small constant, capped at MAX_SEQUENCE_LENGTH.

Usage:
    python scripts/bench/caption_lengths.py \
        --mix "p1:w1 p2:w2 ..." --tokenizer $W/models/Qwen3-0.6B \
        --out caption_len.json
"""

import argparse
import json
from collections import Counter

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

BUCKETS = [0, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
           3072, 4096, 8192, 16384, 32768, 10**9]


def bucket_index(n: int) -> int:
    for i, b in enumerate(BUCKETS):
        if n < b:
            return i - 1
    return len(BUCKETS) - 2


def hist_to_dict(counter: Counter) -> dict:
    out = {}
    for i in range(len(BUCKETS) - 1):
        lo = BUCKETS[i]
        hi = "inf" if BUCKETS[i + 1] == 10**9 else BUCKETS[i + 1] - 1
        out[f"{lo}-{hi}"] = counter[i]
    return out


def percentile(sorted_lens, p):
    if not sorted_lens:
        return 0
    k = max(0, min(len(sorted_lens) - 1, int(round(p / 100 * (len(sorted_lens) - 1)))))
    return sorted_lens[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help='space-separated "path:weight" pairs')
    ap.add_argument("--tokenizer", required=True, help="Qwen3 tokenizer dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def n_tokens(texts):
        """Return token-count list for texts, batched."""
        out = []
        for i in range(0, len(texts), args.batch):
            chunk = texts[i : i + args.batch]
            enc = tok(chunk, add_special_tokens=False)
            out.extend(len(ids) for ids in enc["input_ids"])
        return out

    all_lens = []                # every caption, as trained (chat template adds ~40)
    per_sample_shortest = []     # shortest caption per sample (curriculum position 0)
    per_sample_longest = []      # longest caption per sample (curriculum position 1)
    per_entry = {}
    template_overhead = None

    for part in args.mix.split():
        path, weight = part.rsplit(":", 1)
        ds = load_from_disk(path)
        col = next((c for c in ds.column_names if "caption" in c.lower()), None)
        if col is None:
            print(f"SKIP {path}: no caption column in {ds.column_names}", flush=True)
            continue
        n = len(ds)
        take = min(n, args.max_rows)
        sample = ds.select(range(take))[col]

        def norm_cell(cell):
            if isinstance(cell, str):
                return [cell]
            if cell is None:
                return []
            flat = []
            for c in cell:
                if isinstance(c, str):
                    flat.append(c)
                else:  # nested list fallback
                    flat.extend(x for x in c if isinstance(x, str))
            return flat

        per_row = [norm_cell(s) for s in sample]

        row_lens = []
        n_caps_per_row = []
        for row in per_row:
            if not row:
                row_lens.append([0])
                n_caps_per_row.append(0)
                continue
            row_lens.append(n_tokens(row))
            n_caps_per_row.append(len(row))
        all_lens.extend(l for row in row_lens for l in row)
        per_sample_shortest.extend(min(row) for row in row_lens)
        per_sample_longest.extend(max(row) for row in row_lens)

        flat = [l for row in row_lens for l in row]
        hist = Counter(bucket_index(l) for l in flat)
        ent = {
            "rows_sampled": len(row_lens),
            "total_rows": n,
            "weight": float(weight),
            "hist_all_captions": hist_to_dict(hist),
            "n_captions_per_row_p50": percentile(sorted(n_caps_per_row), 50),
            "n_captions_per_row_p99": percentile(sorted(n_caps_per_row), 99),
            "rows_with_single_caption_frac": round(
                sum(1 for c in n_caps_per_row if c == 1) / max(len(n_caps_per_row), 1), 4
            ),
            "p50": percentile(sorted(flat), 50),
            "p90": percentile(sorted(flat), 90),
            "p99": percentile(sorted(flat), 99),
            "p999": percentile(sorted(flat), 99.9),
            "max": max(flat) if flat else 0,
        }
        per_entry[path.split("/")[-1]] = ent
        print(f"{path}: {ent}", flush=True)
        del ds, sample

    # Note: chat-template overhead is dropped in training — encode_text trims
    # the first DROP_IDX=38 tokens (template prefix) before padding, so in-train
    # effective seq ≈ raw caption tokens + small constant (assistant prefix),
    # capped at MAX_SEQUENCE_LENGTH. Raw token counts below are the right
    # bucket key.

    def summarize(lens):
        sl = sorted(lens)
        return {
            "n": len(sl),
            "p50": percentile(sl, 50),
            "p90": percentile(sl, 90),
            "p99": percentile(sl, 99),
            "p999": percentile(sl, 99.9),
            "max": sl[-1] if sl else 0,
            "hist": hist_to_dict(Counter(bucket_index(l) for l in sl)),
        }

    out = {
        "per_caption_raw": summarize(all_lens),
        "per_sample_shortest": summarize(per_sample_shortest),
        "per_sample_longest": summarize(per_sample_longest),
        "note": "raw = caption as stored, pre-template. In-train effective seq "
        "= raw + small constant after encode_text drops the first 38 template "
        "tokens, capped at MAX_SEQUENCE_LENGTH. shortest/longest = the min/max "
        "caption length of each sample, i.e. the endpoints of the "
        "caption-selection curriculum (position 0.0 prefers the shortest "
        "caption, 1.0 the longest). Buckets are token counts.",
        "per_entry": per_entry,
    }
    print("GLOBAL per_caption_raw:", json.dumps(out["per_caption_raw"]))
    print("GLOBAL shortest:", json.dumps(out["per_sample_shortest"]))
    print("GLOBAL longest:", json.dumps(out["per_sample_longest"]))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
