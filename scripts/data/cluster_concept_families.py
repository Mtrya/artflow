"""Cluster concept embeddings into sampling families (concept benchmark).

Per axis, scans MiniBatchKMeans over a ladder of k values, picks the elbow of
the inertia curve (max distance to the chord on normalized axes), refits at
that k, and assigns each concept a family key. Family display names are the
3-5 concepts nearest the centroid; the key is the single nearest concept plus
the cluster index for uniqueness.

Outputs a JSONL copy of the concepts file with `family` / `family_label`
filled in, plus a Markdown report with the inertia curves and cluster census.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

# Candidate k ladders per axis (entity is an order of magnitude larger).
K_LADDERS = {
    "entity": [100, 200, 400, 700, 1000, 1400, 2000, 2800],
    "technique": [30, 60, 100, 160, 240, 360, 520],
    "scene": [20, 40, 70, 110, 160, 230, 320],
}


def elbow(curve: list[tuple[int, float]]) -> int:
    """Max-distance-to-chord on the normalized (k, inertia) curve."""
    ks = np.array([k for k, _ in curve], dtype=float)
    ys = np.array([v for _, v in curve], dtype=float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    chord = np.stack([x[[0, -1]], y[[0, -1]]])
    seg = chord[:, 1] - chord[:, 0]
    dists = np.abs(seg[0] * (chord[1, 0] - y) - (chord[0, 0] - x) * seg[1])
    dists /= np.linalg.norm(seg)
    return int(ks[int(np.argmax(dists))])


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def cluster_axis(rows: list[dict], emb: np.ndarray, k_ladder: list[int],
                 seed: int) -> tuple[list[str], dict]:
    from sklearn.cluster import KMeans, MiniBatchKMeans

    n = len(rows)
    ladder = [k for k in k_ladder if k < n]
    curve = []
    for k in ladder:
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=seed,
                             n_init=3, max_iter=200)
        labels = km.fit_predict(emb)
        curve.append((k, float(km.inertia_)))
        print(f"  k={k} inertia={km.inertia_:.1f}", flush=True)
    k_best = elbow(curve)
    print(f"  elbow -> k={k_best}", flush=True)

    km = KMeans(n_clusters=k_best, random_state=seed, n_init=10)
    labels = km.fit_predict(emb)
    centers = km.cluster_centers_
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    families = [""] * n
    census = []
    for cid in range(k_best):
        idx = np.where(labels == cid)[0]
        sims = emb[idx] @ centers[cid]
        nearest = idx[np.argsort(-sims)[:5]]
        names = [rows[i]["en"] for i in nearest]
        key = f"{slug(names[0])}#{cid:04d}"
        for i in idx:
            families[i] = key
        census.append({"key": key, "size": len(idx), "names": names})
    return families, {"curve": curve, "k": k_best, "census": census}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.concepts)]
    emb = np.load(args.embeddings).astype(np.float32)
    assert len(rows) == len(emb), "embeddings misaligned with concepts"
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

    by_axis: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_axis[r["axis"]].append(i)

    report = {}
    for axis, idx in sorted(by_axis.items()):
        print(f"axis={axis} n={len(idx)}", flush=True)
        sub_rows = [rows[i] for i in idx]
        sub_emb = emb[idx]
        families, info = cluster_axis(sub_rows, sub_emb,
                                      K_LADDERS[axis], args.seed)
        label_of = {c["key"]: ", ".join(c["names"]) for c in info["census"]}
        for pos, i in enumerate(idx):
            rows[i]["family"] = f"{axis}:{families[pos]}"
            rows[i]["family_label"] = label_of[families[pos]]
        report[axis] = info

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = ["# Concept families", ""]
    for axis, info in sorted(report.items()):
        n = len(by_axis[axis])
        sizes = sorted((c["size"] for c in info["census"]), reverse=True)
        lines.append(f"## {axis}: k={info['k']}, n={n}")
        lines.append(f"curve: {info['curve']}")
        lines.append(f"cluster sizes: max={sizes[0]} p50={sizes[len(sizes)//2]} "
                     f"min={sizes[-1]} singletons={sum(1 for s in sizes if s == 1)}")
        lines.append("")
        for c in info["census"][:10]:
            lines.append(f"- `{c['key']}` ({c['size']}): {', '.join(c['names'])}")
        lines.append("")
    with open(args.report, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out} and {args.report}", flush=True)


if __name__ == "__main__":
    main()
