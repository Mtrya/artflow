"""Embed concept labels for family clustering (concept benchmark).

Reads a concepts JSONL (one row per concept, fields en/qualifier/axis/...),
embeds each label with Qwen3-Embedding using last-token pooling and L2
normalization, and writes a float16 .npy matrix aligned with input row order.

Runs on CPU; ~40k short labels with the 0.6B model takes tens of minutes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def concept_text(row: dict) -> str:
    text = row["en"]
    if row.get("qualifier"):
        text = f"{text} ({row['qualifier']})"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", required=True, help="concepts JSONL")
    ap.add_argument("--model", required=True, help="local model dir")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    rows = [json.loads(line) for line in open(args.concepts)]
    texts = [concept_text(r) for r in rows]
    print(f"concepts: {len(texts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        idx = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), idx]

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i : i + args.batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            v = F.normalize(last_token_pool(hidden, enc["attention_mask"]), dim=-1)
            vecs.append(v.to(torch.float16).numpy())
            if (i // args.batch_size) % 20 == 0:
                print(f"{i}/{len(texts)}", flush=True)

    arr = np.concatenate(vecs)
    np.save(args.out, arr)
    print(f"saved {arr.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
