"""Local twin of box/arm_read.py: read two dumps on E_real and E_mix, paired.

    python tools/arm_read_local.py <arm.npz> <baseline.npz> [arm_label] [base_label]

WHY IT EXISTS. box/arm_read.py runs on a molab box against /marimo paths. Both
GPUs are busy for hours, and the highest-value read available tonight uses dumps
that are already sitting on this laptop. Same pool construction, same PREV, same
paired bootstrap -- so a number from here is comparable to a number from there.

THE READ THIS WAS BUILT FOR. dose2x concluded "Stage A is limited by data
DIVERSITY, not compute" from a contrast between two arms measured on DIFFERENT
instruments: mlmdose's +0.0000 was human-val, dose2x's +0.0147 was E_real. On the
common instrument dose2x reads -0.0003 -- indistinguishable from mlmdose. The
contrast was never made on one ruler, so the conclusion was never supported. This
makes it on one ruler.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PREV = 0.045
W = 0.784
NBOOT = 400

arm_p, base_p = sys.argv[1], sys.argv[2]
arm_l = sys.argv[3] if len(sys.argv) > 3 else arm_p
base_l = sys.argv[4] if len(sys.argv) > 4 else base_p

uv = pd.read_parquet("outputs/universe_view.parquet",
                     columns=["y", "slice", "category"])
va = ~np.load("outputs/out-t102/ce_split_mask.npz")["in_tr"]
sl, y_all, cat_all = uv["slice"].values, uv.y.values.astype(int), uv.category.values

pk = np.flatnonzero((sl == "pos") & va)
ek = np.flatnonzero((sl == "easy_neg") & va)
mk = np.flatnonzero((sl == "mined_neg") & va)
rng = np.random.default_rng(784)
es = rng.choice(ek, min(int(round(len(mk) * (1 - W) / W)), len(ek)), replace=False)
pe = float((y_all[va] == 1).mean())
nn = len(es) + len(mk)
npo = int(round(pe / (1 - pe) * nn))
ps = rng.choice(pk, npo, replace=False) if npo < len(pk) else pk
POOLS = {"E_real": np.sort(np.concatenate([pk, ek])),
         "E_mix": np.sort(np.concatenate([ps, es, mk]))}


def raw(p):
    z = np.load(p, allow_pickle=True)
    k = "ce" if "ce" in z else list(z.keys())[0]
    return z[k].astype(np.float64)


a_all, b_all = raw(arm_p), raw(base_p)
assert len(a_all) == len(b_all) == len(uv), (
    "dump lengths %d/%d vs universe %d -- not the same rows"
    % (len(a_all), len(b_all), len(uv)))


def per_cat(y, s, cat):
    out = {}
    for c in np.unique(cat):
        m = cat == c
        yy, ss = y[m], s[m]
        p, n = int((yy == 1).sum()), int((yy == 0).sum())
        if p < 10 or n < 10:
            continue
        w = PREV * n / ((1 - PREV) * p)
        out[c] = average_precision_score(
            yy, ss, sample_weight=np.where(yy == 1, w, 1.0))
    return out


print("arm      %s" % arm_l)
print("baseline %s" % base_l)
print("PREV=%.3f  E_mix w*=%.3f  paired bootstrap n=%d" % (PREV, W, NBOOT))

for pname, ix in POOLS.items():
    y, cat = y_all[ix], cat_all[ix]
    ap_a, ap_b = per_cat(y, a_all[ix], cat), per_cat(y, b_all[ix], cat)
    cats = sorted(set(ap_a) & set(ap_b))
    va_, vb_ = np.array([ap_a[c] for c in cats]), np.array([ap_b[c] for c in cats])
    d = va_ - vb_
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NBOOT)])
    print("\n=== %s | %d rows, %d categories ===" % (pname, len(ix), len(cats)))
    print("  baseline macro %.5f" % vb_.mean())
    print("  arm      macro %.5f" % va_.mean())
    print("  DELTA          %+.5f   sd %.5f   [%.5f, %.5f] 90%%"
          % (d.mean(), bs.std(), np.percentile(bs, 5), np.percentile(bs, 95)))
    print("  categories improved: %d/%d" % (int((d > 0).sum()), len(d)))
