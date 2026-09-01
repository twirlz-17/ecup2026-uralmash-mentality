"""Is a THIRD cross-encoder affordable? Price its value as a function of coverage.

WHY THIS EXISTS. The largest effect measured today is a third ensemble member:
+0.01777 held-out E_real for adding t152euroA-ep1 on top of CE-1. I called it
unshippable from `q15timing`'s arithmetic -- "mmBERT cannot be ADDED to the
champion at any useful coverage (40% -> 16.14 of 13.0, 30% -> 14.93, 20% ->
13.68, 15% -> 13.05; only 10% fits)" -- without checking how much of the VALUE
survives at the coverage that does fit. That is the wrong order: `cascadebuys
thebudget` says PR-AUC lives at the HEAD of the ranking, and `coversweep`
measured the shipped 0.30 capturing 89% of what FULL coverage buys for CE-2.

So the real question is not "can we afford it at full coverage" (no) but "what
fraction of it survives at 10-15%". Zero GPU, dumps already on disk.

The third member enters exactly as CE-2 does -- band-ranked over the top-k of
the CURRENT blended score, which is how run.py would have to cascade it.

Coverage chosen on SELECT, quoted on REPORT, as everywhere else today.
"""
import sys
import numpy as np
sys.path.insert(0, "tools")
from cascade_read import (macro, cascade, raw, band_rank, SEL, REP,
                          W_CE, CE_COVER)

a, b = raw("t167llm2x-ep1"), raw("ce2-shipped")


def three(ce1, ce2, ce3, idx, cov3, w3=0.20):
    base = cascade(ce1, ce2, idx, W_CE, CE_COVER)
    n = len(idx)
    k = max(1, int(round(cov3 * n)))
    sel = np.argsort(-base, kind="stable")[:k]
    return (1.0 - w3) * base + w3 * band_rank(ce3[idx], sel, n)


base_rep, _ = macro(REP, cascade(a, b, REP))
print("v39 cascade REPORT = %.5f\n" % base_rep)

CANDS = ["alexbge", "t152euroA-ep1", "t176full-ep1", "t136sa-ep1", "t177vol-ep1"]
COVS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
print("%-18s %s" % ("third member (REPORT delta vs v39)",
                    "".join("%9s" % ("cov%.2f" % c) for c in COVS)))
for t in CANDS:
    try:
        p = raw(t)
    except Exception:
        print("%-36s SKIP" % t); continue
    cells = []
    for c in COVS:
        v, _ = macro(REP, three(a, b, p, REP, c))
        cells.append(v - base_rep)
    star = " <-- " if max(cells) > 0.005 else ""
    print("%-36s%s%s" % (t, "".join("%+9.5f" % x for x in cells), star))

print("\n=== fraction of FULL-coverage value retained ===")
for t in CANDS:
    try:
        p = raw(t)
    except Exception:
        continue
    full, _ = macro(REP, three(a, b, p, REP, 1.00))
    fd = full - base_rep
    if fd <= 1e-5:
        continue
    out = []
    for c in COVS[:-1]:
        v, _ = macro(REP, three(a, b, p, REP, c))
        out.append("cov%.2f %3.0f%%" % (c, 100 * (v - base_rep) / fd))
    print("  %-18s full %+.5f | %s" % (t, fd, "  ".join(out)))
