"""Publish the Pexels people set as a URL-and-caption dataset.

The photographs are not redistributed.  Pexels' terms allow the images to be
used and modified but not to be served again from an image library, so the
dataset ships the identifier and the links the images can be fetched from,
under the licence their own site states, and the captions written for this
corpus.  The alt text the API returned is used while training but is not
republished: it is Pexels' own copy, and the same terms discourage passing
their content on in bulk.

Run on the machine that holds the fetch metadata and the caption output:

    HF_TOKEN=... python -m scripts.data.publish_hf_pexels \\
        --metadata $W/data/raw/pexels_people/metadata.jsonl \\
        --captions $W/data/caption_enrich/production/pexels_frozen.jsonl \\
        --out $W/hf_pexels_staging --upload
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "kaupane/pexels-people-captions"

SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("photo_id", pa.int64()),
    ("page_url", pa.string()),
    ("image_url", pa.string()),
    ("caption", pa.string()),
    ("caption_language", pa.string()),
    ("length_band", pa.string()),
    ("caption_model", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("photographer", pa.string()),
    ("photographer_url", pa.string()),
    ("search_query", pa.string()),
])

CARD = """---
license: other
license_name: pexels-license
task_categories:
  - text-to-image
  - image-to-text
language:
  - en
  - zh
tags:
  - photograph
  - people
  - caption
  - portrait
size_categories:
  - 10K<n<100K
---

# Pexels people captions

{captions} photographs of people, each with a description written for it in
Chinese or English.  The photographs themselves are not included: every row
carries a link to the original on Pexels instead.

## What a row holds

| column | meaning |
|---|---|
| `image_id` | stable identifier used across the corpus, `pexels-<photo id>` |
| `photo_id` | the Pexels photo id |
| `page_url` | the photo's page on pexels.com |
| `image_url` | the original image file as the API reports it |
| `caption` | the description written for this dataset, 33-3150 tokens long |
| `caption_language` | `zh` or `en` |
| `length_band` | the length the caption was written to: `64-255`, `256-511`, `512-895` or `896-1280`; a caption that came out outside its band is still shipped |
| `caption_model` | the model that wrote the caption |
| `width`, `height` | dimensions of the original image |
| `photographer` | photographer credited by Pexels |
| `photographer_url` | the photographer's page on pexels.com |
| `search_query` | the query the photo was drawn with |

## How it was built

1. **Search** - {queries} queries describing people (portraits, occupations,
   clothing, region) were run against the Pexels search API and their results
   collected.  Duplicates were removed by photo id, and photographs smaller
   than 896 px on the short side were discarded.
2. **Selection** - a stratified draw chose which photographs to describe, spread
   across queries and image shapes.
3. **Captioning** - each selected photograph was described once by
   `{model}` with reasoning disabled, at one of four length bands, in Chinese or
   English, as flowing prose or as grouped facts.  The model was shown the
   photograph at a bounded edge and asked for what is in it, with a ban on
   hedging language and on catalogue-style openings.
4. **Acceptance** - {dropped} captions that repeated a sentence were discarded.
   A caption that came out shorter or longer than its band was kept, since it
   still describes the photograph.

## Using it

```python
import pandas as pd, requests
row = pd.read_parquet("https://huggingface.co/datasets/{repo}/resolve/main/data/train-00000-of-00001.parquet").iloc[0]
image = requests.get(row["image_url"]).content
```

## Licence and attribution

The photographs remain under the [Pexels licence](https://www.pexels.com/license/):
free to use and modify, no attribution required by the licence but appreciated,
and not to be redistributed from another image or wallpaper service - which is
why this dataset links to them rather than shipping them.  Follow the
`page_url` and `photographer_url` columns to credit the photographer.

The captions were written for this dataset and are released under CC-BY-4.0.

## Caveats

- Captions are machine-written.  They are grounded in the photograph, but a
  caption is not a verified description: expect occasional counting or colour
  errors.
- Photographs may be removed from Pexels, in which case a link stops resolving.
- The set is a sample of what the queries returned, not a balanced survey of
  the world's people; the queries deliberately over-sample some regions and
  clothing.
"""


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rows(metadata_path: str, captions_path: str) -> List[Dict]:
    photos = {record["source_id"]: record for record in read_jsonl(metadata_path)}
    captions = {record["image_id"]: record for record in read_jsonl(captions_path)}
    rows = []
    for image_id, caption in captions.items():
        if not caption.get("accepted"):
            continue
        photo = photos.get(image_id)
        if photo is None:
            continue
        rows.append({
            "image_id": image_id,
            "photo_id": int(photo["photo_id"]),
            "page_url": photo["page_url"],
            "image_url": photo["original_url"],
            "caption": caption["text"],
            "caption_language": caption["language"],
            "length_band": caption["band"],
            "caption_model": caption["model"],
            "width": int(photo["source_width"]),
            "height": int(photo["source_height"]),
            "photographer": photo["photographer"],
            "photographer_url": photo["photographer_url"],
            "search_query": photo.get("query"),
        })
    rows.sort(key=lambda row: row["image_id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", required=True, help="staging directory")
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args.metadata, args.captions)
    photos = read_jsonl(args.metadata)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    parquet_path = out / "data" / "train-00000-of-00001.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    card = CARD.format(
        captions=f"{len(rows):,}",
        queries=len({row["search_query"] for row in rows if row["search_query"]}),
        model=next((row["caption_model"] for row in rows), "an external model"),
        dropped=sum(1 for record in read_jsonl(args.captions) if not record.get("accepted")),
        repo=args.repo,
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    print(f"{len(rows)} rows written to {parquet_path}")
    print(f"  photos fetched: {len(photos)}")
    print(f"  bands: {dict(Counter(row['length_band'] for row in rows).most_common())}")
    print(f"  languages: {dict(Counter(row['caption_language'] for row in rows).most_common())}")
    print(f"  photographers: {len({row['photographer'] for row in rows})}")
    print(f"  queries: {len({row['search_query'] for row in rows})}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="dataset")
        print(f"uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
