#!/usr/bin/env python3
"""Bucket-level training plan: per-resolution availability with capping.

Reads res_stats.jsonl (+ optional parquet stats), groups datasets into
domains, and prints per-bucket (256/640/896/1024) totals and domain mix
under several cap scenarios:

  none        - all min>=R images used
  ds25        - each dataset capped at 25% of uncapped bucket total
  dom30       - each domain capped at 30% of uncapped bucket total
  dom25       - each domain capped at 25% of uncapped bucket total
  ds15dom30   - dataset cap 15%, domain cap 30%

Capping is applied per bucket independently (progressive resolution stages
sample each bucket separately). Also reports extreme aspect ratios.

Usage:  .venv/bin/python scripts/data/res_plan.py --stats res_stats.jsonl
"""
import argparse
import json
import os

DOMAINS = {
    "d1_npm_c0": "D1", "d1_npm_c1": "D1", "d1_npm_c2": "D1",
    "d1_npm_c3": "D1", "d1_aic": "D1", "d1_met": "D1", "d1_fsg": "D1",
    "d1_princeton": "D1",
    "d2_wikiart": "D2", "d2_nga": "D2", "d2_rijks": "D2",
    "d2_nga_imp": "D2", "d2_met_imp": "D2", "d2_met_portrait": "D2",
    "d2_imp_caillebotte": "D2", "d2_imp_cassatt": "D2",
    "d2_imp_cezanne": "D2", "d2_imp_degas": "D2", "d2_imp_gauguin": "D2",
    "d2_imp_manet": "D2", "d2_imp_monet": "D2", "d2_imp_morisot": "D2",
    "d2_imp_pissarro": "D2", "d2_imp_renoir": "D2", "d2_imp_seurat": "D2",
    "d2_imp_sisley": "D2", "d2_imp_toulouse": "D2", "d2_imp_vangogh": "D2",
    "d3_human_recaption": "D3", "d3_people_supp": "D3",
    "d4_pd12m": "D4", "d4_relaion": "D4", "d4_vintage": "D4",
    "z_image_turbo_gen": "D4", "megalith_opus": "D4", "inat_opus": "D4",
}
BUCKETS = ["g256", "g640", "g896", "g1024"]


def load(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def bucket_plan(rows, cap_ds=None, cap_dom=None):
    """Return {bucket: {total, by_domain, by_dataset}} after caps."""
    out = {}
    for b in BUCKETS:
        per_ds = {r["dataset"]: r[b] for r in rows if r.get(b, 0) > 0}
        raw_total = sum(per_ds.values())
        # dataset-level cap
        if cap_ds:
            ds_cap = int(raw_total * cap_ds)
            per_ds = {k: min(v, ds_cap) for k, v in per_ds.items()}
        # domain-level cap
        if cap_dom:
            dom_cap = int(raw_total * cap_dom)
            by_dom = {}
            for ds, n in per_ds.items():
                by_dom[DOMAINS.get(ds, ds)] = by_dom.get(DOMAINS.get(ds, ds), 0) + n
            for d in list(by_dom):
                by_dom[d] = min(by_dom[d], dom_cap)
            # scale datasets down proportionally within capped domains
            for ds in list(per_ds):
                d = DOMAINS.get(ds, ds)
                orig_dom = sum(v for k, v in per_ds.items() if DOMAINS.get(k, k) == d)
                if orig_dom > dom_cap and orig_dom > 0:
                    per_ds[ds] = int(per_ds[ds] * dom_cap / orig_dom)
        by_dom = {}
        for ds, n in per_ds.items():
            d = DOMAINS.get(ds, ds)
            by_dom[d] = by_dom.get(d, 0) + n
        out[b] = {"raw_total": raw_total, "total": sum(per_ds.values()),
                  "by_domain": by_dom, "by_dataset": per_ds}
    return out


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--parquet-stats", default=None)
    args = ap.parse_args()
    rows = load(args.stats) + load(args.parquet_stats)

    ar_tot = sum(r["total"] for r in rows)
    ar_lt05 = sum(r.get("r_lt_0_25", 0) + r.get("r_0_25_0_5", 0) for r in rows)
    ar_gt2 = sum(r.get("r_2_4", 0) + r.get("r_gt_4", 0) for r in rows)
    print(f"aspect ratio: ar<0.5 {ar_lt05} ({pct(ar_lt05, ar_tot)}), "
          f"ar>2 {ar_gt2} ({pct(ar_gt2, ar_tot)})\n")

    scenarios = [("none", None, None), ("ds25", 0.25, None),
                 ("dom30", None, 0.30), ("dom25", None, 0.25),
                 ("ds15dom30", 0.15, 0.30)]
    header = f"{'scenario':>10} | " + " | ".join(
        f"{b}: total (D1/D2/D3/D4)" for b in BUCKETS)
    print(header)
    for name, cds, cdom in scenarios:
        plan = bucket_plan(rows, cds, cdom)
        cells = []
        for b in BUCKETS:
            p = plan[b]
            mix = "/".join(str(p["by_domain"].get(d, 0))
                           for d in ("D1", "D2", "D3", "D4"))
            cells.append(f"{p['total']:,} ({mix})")
        print(f"{name:>10} | " + " | ".join(cells))

    print("\nper-dataset share of the capped (ds25) 256p bucket:")
    plan = bucket_plan(rows, 0.25, None)
    p = plan["g256"]
    for ds in sorted(p["by_dataset"], key=lambda k: -p["by_dataset"][k]):
        print(f"  {ds:24s} {p['by_dataset'][ds]:>8,}  "
              f"{pct(p['by_dataset'][ds], p['total'])}")


if __name__ == "__main__":
    main()
