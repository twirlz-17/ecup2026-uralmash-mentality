"""Price v39's ACTUAL cascade, and the marginal value of a partner inside it.

WHY THIS COULD NOT BE RUN UNTIL TODAY. v39 ships CE-1 (mmBERT@1024) + CE-2
(bge-reranker-v2-m3 @256, re-scoring only the top CE_COVER of CE-1) + a GBDT.
Every ensemble number this project owns -- `runens` +0.00615, `runens2` +0.02821
-- was measured against CE-1 SOLO, because a CE-2 score dump existed on NO
machine. `partnerholdout` died on that gap. queue100 produced the dump, so the
question "what does the partner actually buy, and would a different partner buy
more" is answerable for the first time.

THE CASCADE IS REPLICATED FROM THE SHIPPED ARCHIVE, not from memory:
    _rank01(x)          = rankdata(x) / n
    _band_rank(raw,sel) = lo + (1-lo) * rankdata(raw[sel])/k   on the band,
                          lo = 1 - k/n                          off the band
    ce = W_CE*_rank01(ce1) + (1-W_CE)*_band_rank(ce2, top-k)
with W_CE=0.7 and CE_COVER=0.30, read out of submission_v39.zip's run.py.

HONEST-SPLIT DISCIPLINE, same as `blend_search`: E_real is split ITEM-disjoint
by a pid1 hash; anything that looks like a search (the w/cover sweep, partner
ranking) is done on SELECT, and every headline number is quoted on REPORT, which
the search never saw. `v41lb` cost a slot because I trusted a self-flattering
number; this is the guard against repeating it.

KNOWN APPROXIMATION, STATED UP FRONT: the container computes its band over the
FULL test pair set, this reads it over E_real's 36.5k rows, so the top-30% band
is not the same set of pairs. Ratios between arms are the signal; the absolute
level is not the board. `transfergap` forbids projecting either onto the board.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PREV = 0.045
W_CE, CE_COVER = 0.7, 0.30
D = "outputs/dumps/ce_scores_%s.npz"

uv = pd.read_parquet("outputs/universe_view.parquet",
                     columns=["y", "slice", "category", "pid1", "pid2"])
va = ~np.load("outputs/out-t102/ce_split_mask.npz")["in_tr"]
sl, y_all, cat_all = uv["slice"].values, uv.y.values.astype(int), uv.category.values
pk = np.flatnonzero((sl == "pos") & va)
ek = np.flatnonzero((sl == "easy_neg") & va)
E_REAL = np.sort(np.concatenate([pk, ek]))
h = pd.util.hash_pandas_object(uv["pid1"].astype(str), index=False).values
half = (h % 2).astype(bool)[E_REAL]
SEL, REP = E_REAL[half], E_REAL[~half]
print("E_real %d -> SELECT %d / REPORT %d (item-disjoint by pid1 hash)\n"
      % (len(E_REAL), len(SEL), len(REP)))


def raw(tag):
    z = np.load(D % tag, allow_pickle=True)
    k = "ce" if "ce" in z else list(z.keys())[0]
    return z[k].astype(np.float64)


def macro(idx, s):
    y, c = y_all[idx], cat_all[idx]
    out = []
    for cc in np.unique(c):
        m = c == cc
        yy, ss = y[m], s[m]
        p, n = int((yy == 1).sum()), int((yy == 0).sum())
        if p < 10 or n < 10:
            continue
        w = PREV * n / ((1 - PREV) * p)
        out.append(average_precision_score(yy, ss,
                                           sample_weight=np.where(yy == 1, w, 1.0)))
    return float(np.mean(out)), len(out)


def rank01(x):
    return rankdata(x) / len(x)


def band_rank(rawv, sel, n):
    k = len(sel)
    lo = 1.0 - k / float(n)
    out = np.full(n, lo, dtype=np.float64)
    if k > 1:
        out[sel] = lo + (1.0 - lo) * (rankdata(rawv[sel]) / k)
    return out


def cascade(ce1, ce2, idx, w=W_CE, cov=CE_COVER):
    """v39's cascade, restricted to idx (ranks recomputed inside idx)."""
    a, b = ce1[idx], ce2[idx]
    n = len(idx)
    k = max(1, int(round(cov * n)))
    sel = np.argsort(-a, kind="stable")[:k]
    return w * rank01(a) + (1.0 - w) * band_rank(b, sel, n)


CE1, CE2 = "t167llm2x-ep1", "ce2-shipped"
a, b = raw(CE1), raw(CE2)

print("=== 1. WHAT DOES THE SHIPPED PARTNER ACTUALLY BUY? (REPORT half) ===")
s1, nc = macro(REP, a[REP])
s2, _ = macro(REP, b[REP])
sc, _ = macro(REP, cascade(a, b, REP))
print("  CE-1 solo (t167llm2x-ep1)      %.5f   (%d categories)" % (s1, nc))
print("  CE-2 solo (bge@256, shipped)   %.5f" % s2)
print("  v39 cascade w=%.2f cover=%.2f   %.5f   -> partner buys %+.5f"
      % (W_CE, CE_COVER, sc, sc - s1))

print("\n=== 2. IS w=0.7 / cover=0.30 THE RIGHT OPERATING POINT? (SELECT) ===")
base_sel, _ = macro(SEL, a[SEL])
print("  CE-1 solo on SELECT = %.5f" % base_sel)
best = None
for cov in (0.15, 0.30, 0.50, 1.00):
    row = []
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        v, _ = macro(SEL, cascade(a, b, SEL, w, cov))
        row.append("%.5f" % v)
        if best is None or v > best[0]:
            best = (v, w, cov)
    print("   cover=%.2f | " % cov + "  ".join(
        "w=%.1f %s" % (w, s) for w, s in zip((0.5, 0.6, 0.7, 0.8, 0.9, 1.0), row)))
print("  best on SELECT: %.5f at w=%.1f cover=%.2f" % best)
v_ship, _ = macro(REP, cascade(a, b, REP))
v_best, _ = macro(REP, cascade(a, b, REP, best[1], best[2]))
print("  ON REPORT: shipped %.5f  vs  SELECT-best %.5f  -> %+.5f"
      % (v_ship, v_best, v_best - v_ship))

print("\n=== 3. WOULD A DIFFERENT PARTNER BEAT bge IN THE CE-2 SLOT? (REPORT) ===")
print("  baseline = v39 cascade %.5f\n" % sc)
cands = [t for t in sys.argv[1:]] or [
    "t152euroA-ep1", "t159euro610b-ep1", "t136sa-ep1", "t177vol-ep1",
    "t158mlmA2b-ep1", "t148mlmA-ep1", "t172sb4-ep0"]
rows = []
for t in cands:
    try:
        p = raw(t)
    except Exception as exc:
        print("  %-18s SKIP (%s)" % (t, type(exc).__name__)); continue
    solo, _ = macro(REP, p[REP])
    swap, _ = macro(REP, cascade(a, p, REP))
    rows.append((swap - sc, t, solo, swap))
for d, t, solo, swap in sorted(rows, reverse=True):
    print("  %-18s solo %.5f   as CE-2 %.5f   %+.5f %s"
          % (t, solo, swap, d, "<-- beats bge" if d > 0 else ""))
