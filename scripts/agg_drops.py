"""Aggregate per-source drop categories from the precompute runner log.

Attribution: batch "dropped" lines belong to the source whose "== <name>: rc=0"
marker appears AFTER them (markers lag the batches they summarize). Batches
before the first marker belong to the first source.
"""
import collections
import re
import sys

log = sys.argv[1]
order = []
agg = collections.defaultdict(collections.Counter)
buf = collections.Counter()  # batches not yet flushed to a finished source
for line in open(log, errors="ignore"):
    m = re.search(r"== ([\w]+): rc=0", line)
    if m:
        src = m.group(1)
        order.append(src)
        agg[src] += buf
        buf = collections.Counter()
        continue
    b = re.search(r"dropped \((.*)\)$", line)
    if not b:
        continue
    for f in b.group(1).split(","):
        k, v = f.strip().split("=")
        if v != "0":
            buf[k] += int(v)
for s in order:
    print(f"{s:<14} drop_categories={dict(agg[s])}")
