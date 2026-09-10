#!/usr/bin/env python3
"""Spot-check uploaded D1 shards: decode a few images from a small shard
(full download) and stream the first entries of a large shard."""
import json
import os
import tarfile

import requests
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "kaupane/chinese-painting-collection"
tok = os.environ["HF_TOKEN"]

p = hf_hub_download(REPO, "data/aic_china/shard-00000.tar", repo_type="dataset")
tf = tarfile.open(p)
names = tf.getnames()
jpgs = [n for n in names if n.endswith(".jpg")]
print("aic shard entries:", len(names), "jpgs:", len(jpgs), flush=True)
for n in jpgs[:3]:
    im = Image.open(tf.extractfile(n))
    im.load()
    side = json.loads(tf.extractfile(n.replace(".jpg", ".json")).read())
    print(" ", n, im.size, "|", (side.get("caption_en") or "")[:70], flush=True)

url = ("https://huggingface.co/datasets/" + REPO +
       "/resolve/main/data/npm_tw_c3/shard-00003.tar")
r = requests.get(url, headers={"Authorization": "Bearer " + tok},
                 stream=True, timeout=60)
r.raise_for_status()
tf = tarfile.open(fileobj=r.raw, mode="r|")
count = 0
for member in tf:
    if member.name.endswith(".jpg"):
        im = Image.open(tf.extractfile(member))
        im.load()
        print("c3 stream:", member.name, im.size, flush=True)
        count += 1
        if count >= 3:
            break
print("SPOTCHECK_OK", flush=True)
