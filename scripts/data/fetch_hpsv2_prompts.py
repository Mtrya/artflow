"""Pull the HPS v2 prompt set that seeds the synthetic validation batch.

The seeds are the 3,200 test prompts of ``Lakonik/t2i-prompts-hpsv2`` (the HPS
v2 benchmark prompt list).  They are kept verbatim - the expansion step is where
each one becomes six drawable prompts, and keeping the seed text untouched means
a later audit can always compare a variant against the prompt it came from.

Row shape: ``{"seed_id": "hps-0000", "prompt": "..."}``.

CLI:
    python -m scripts.data.fetch_hpsv2_prompts --out data/hpsv2/seeds.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "Lakonik/t2i-prompts-hpsv2"
REPO_FILE = "data/test-00000-of-00001.parquet"


def fetch_parquet(repo: str, filename: str) -> Path:
    """Local path of the dataset parquet, downloading it on first use."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))


def read_prompts(parquet: Path) -> list:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["prompt"])
    return [row["prompt"] for row in table.to_pylist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--file", default=REPO_FILE)
    parser.add_argument("--out", default="data/hpsv2/seeds.jsonl")
    args = parser.parse_args()

    prompts = read_prompts(fetch_parquet(args.repo, args.file))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for index, prompt in enumerate(prompts):
            sink.write(json.dumps({"seed_id": f"hps-{index:04d}", "prompt": prompt},
                                  ensure_ascii=False) + "\n")

    unique = len({p.strip() for p in prompts})
    print(f"{len(prompts)} seeds -> {out} ({unique} unique prompts)")
    print(f"mean chars {sum(len(p) for p in prompts) / max(len(prompts), 1):.0f}")


if __name__ == "__main__":
    main()
