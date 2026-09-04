"""Reconcile 256p precompute outputs against manifests.

Counts rows in each manifest jsonl and the corresponding @256p dataset dir,
prints a per-source table plus totals. Source dir name == manifest name with
'_' -> '-' plus '@256p' suffix.
"""
import glob
import os
import sys

from datasets import load_from_disk

m_dir = sys.argv[1]
d_dir = sys.argv[2]

fmt = "{:<22}{:>12}{:>12}  {}"
print(fmt.format("manifest", "m_rows", "out_rows", "delta"))
tot_m = tot_o = 0
for mf in sorted(glob.glob(os.path.join(m_dir, "*.jsonl"))):
    name = os.path.basename(mf)[: -len(".jsonl")]
    dname = name.replace("_", "-") + "@256p"
    m_rows = sum(1 for _ in open(mf))
    dpath = os.path.join(d_dir, dname)
    if os.path.isdir(dpath):
        ds = load_from_disk(dpath)
        o_rows = ds.num_rows
    else:
        o_rows = -1
    print(fmt.format(name, m_rows, o_rows, o_rows - m_rows if o_rows >= 0 else "MISSING"))
    tot_m += m_rows
    if o_rows >= 0:
        tot_o += o_rows
print(fmt.format("TOTAL", tot_m, tot_o, tot_o - tot_m))
