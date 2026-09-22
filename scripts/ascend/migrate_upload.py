#!/usr/bin/env python
"""Upload precomputed dataset dirs (qb GPFS) to a private HF repo.

Runs on inko-patrol (CPU notebook with qb-ilm access + HF token).
Usage: python3 migrate_upload.py <resolution>  e.g. 256p
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi

ROOT = "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow/precomputed_dataset"

HERO_DIRS = {
    "256p": ["d1@256p", "d2-museum@256p", "d2-wikiart@256p", "d3-human@256p",
             "d3-people@256p", "d3-pexels@256p", "d3-synth-v2@256p", "d4-inat@256p",
             "d4-megalith@256p", "d4-pd12m@256p", "d4-relaion@256p",
             "d4-vintage@256p", "d4-zimage@256p", "light-eval@256p"],
    "640p": ["d1@640p", "d2-museum@640p", "d2-wikiart@640p", "d3-human@640p",
             "d3-people-a@640p", "d3-people-b@640p", "d3-pexels@640p",
             "d3-synth-v2@640p", "d4-megalith@640p", "d4-pd12m@640p",
             "d4-relaion-p0@640p", "d4-relaion-p1@640p", "d4-relaion-p2@640p",
             "d4-relaion-p3@640p", "d4-relaion-p4@640p", "d4-vintage@640p",
             "d4-zimage@640p", "light-eval@640p"],
    "896p": ["d1@896p", "d2-museum@896p", "d2-wikiart@896p", "d3-human@896p",
             "d3-people@896p", "d3-pexels@896p", "d4-megalith@896p",
             "d4-pd12m@896p", "d4-relaion-p0@896p", "d4-relaion-p1@896p",
             "d4-relaion-p2@896p", "d4-relaion-p3@896p", "d4-relaion-p4@896p",
             "d4-vintage@896p", "light-eval@896p"],
}

WORKERS = 6
ATTEMPTS = 3


def main() -> None:
    res = sys.argv[1]
    repo = f"kaupane/artflow-precomputed-{res}"
    dirs = HERO_DIRS[res]
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"repo={repo} dirs={len(dirs)} workers={WORKERS}", flush=True)

    def up(d: str):
        last = None
        for attempt in range(1, ATTEMPTS + 1):
            t0 = time.time()
            try:
                api.upload_folder(repo_id=repo, repo_type="dataset",
                                  folder_path=os.path.join(ROOT, d),
                                  path_in_repo=d)
                return d, time.time() - t0
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"RETRY {d} attempt={attempt} err={e}", flush=True)
                time.sleep(20 * attempt)
        raise last

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(up, d): d for d in dirs}
        for f in as_completed(futs):
            d = futs[f]
            try:
                name, dt = f.result()
                print(f"DONE {name} {dt:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(d)
                print(f"FAIL {d}: {e}", flush=True)
    print(f"ALL_DONE failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
