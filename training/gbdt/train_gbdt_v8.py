"""
E_CUP 2026 -- ROUND 27: TRAIN ON matches_llm, JUDGE ON RULER E.

Two round-20/25 findings meet here.

  1. The test set's negatives are NEAR-DUPLICATES; matches.parquet's are easy.
     That is the whole +0.12, and it is why ruler A reads the leaderboard with
     an error of 0.343 while ruler E reads it to 0.0087.
  2. Manufacturing near-duplicate negatives by mining does NOT work (round 25,
     confirmed on the uncontaminated ruler: dG = -0.00045). Round 3's diagnosis
     was right -- the mined pool contains genuine duplicates, so labelling them
     0 teaches the opposite of truth.

But `matches_llm` is exactly the thing mining was trying to fake, and it comes
with real labels. Round 20's adversarial classifier separated human pairs from
LLM pairs at AUC 1.0000 with `mh_bands_agree` carrying 66% of the gain --
matches_llm IS an LSH-blocked near-duplicate pool. 11.19M pairs of it, labelled
by nine LLM votes rather than by the assumption that a near-duplicate is a
non-match.

So the training distribution we want already exists. This kernel measures
whether using it helps.

RULER CHOICE, which is the part that decides whether the answer means anything.
Round 25's rule: a ruler is valid only for models whose training set does not
share its generator.

    ruler G is LLM pairs      -> CONTAMINATED for an LLM-trained model. Unusable.
    ruler E is mined negatives -> a different generator from matches_llm, but the
                                  same CHARACTER (both are LSH near-duplicates).
                                  E's negatives are assumed 0 and round 3 says
                                  some are really duplicates. An LLM-trained
                                  model that correctly calls those duplicates is
                                  PENALISED by E.

E is therefore biased AGAINST the model under test. That is the right direction
to be wrong in: if the LLM-trained model wins on E anyway, the finding is real
and understated. If it loses, E's bias is a live alternative explanation and the
result is not conclusive -- which the output says explicitly rather than
pretending otherwise.

    h0  human train half only                        (control, what we ship)
    l0  matches_llm only, hard labels at >0.5
    l1  matches_llm only, SOFT labels -- round 23 found 10 distinct values (k/9,
        nine votes) and ~25% of rows in the graded middle, all of which we
        currently flatten to 0/1. The disagreement band is the hard region.
    m0  both, pooled, one NB table fitted on the pooled set

Each model gets NB tables fitted on ITS OWN training data, because that is what
would ship with it. Mixing tables across training sets is the confound that made
round 8b's L3 unreadable.
"""
import base64
import gc
import json
import glob
import os
import subprocess
import sys
import time

try:
    import rapidfuzz  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"],
                   check=False)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from collections import Counter

SEED = 42
N_LLM = 600_000
SHARD = 100_000
MIN_COUNT = 5
TEST_PREV = 0.11130
t0 = time.time()
OUT = "/kaggle/working"


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


os.makedirs(f"{OUT}/src", exist_ok=True)
for fn, blob in (("features.py", "<<<FEATURES_B64>>>"),
                 ("features2.py", "<<<FEATURES2_B64>>>")):
    with open(f"{OUT}/src/{fn}", "w", encoding="utf-8") as f:
        f.write(base64.b64decode(blob).decode("utf-8"))
open(f"{OUT}/src/__init__.py", "w").close()
sys.path.insert(0, OUT)

import lightgbm as lgb  # noqa: E402

from src import features2 as F2  # noqa: E402
from src.features import (  # noqa: E402
    HAVE_RAPIDFUZZ, build_item_features, build_pair_features, build_tfidf,
)

assert HAVE_RAPIDFUZZ
NS = len(F2.NB_STATS)
NB_COLS = F2.NB_NAME_COLS + F2.NB_ATTR_COLS + F2.NB_KEY_COLS


def find_file(name):
    hits = [h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True)
            if "ecup26-matching" in h]
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def macro_ap(y, s, cat):
    per = []
    for c in np.unique(cat):
        k = cat == c
        if k.sum() < 2 or y[k].min() == y[k].max():
            continue
        per.append(average_precision_score(y[k], s[k]))
    return float(np.mean(per)) if per else float("nan")


def featurise(items_df, pairs_df, top_keys):
    feats = build_item_features(items_df)
    tfidf = build_tfidf(feats)
    idx = pd.Index(feats["id"])
    a = idx.get_indexer(pairs_df["id1"].to_numpy())
    b = idx.get_indexer(pairs_df["id2"].to_numpy())
    ok = (a >= 0) & (b >= 0)
    p1, p2 = F2.canonical_order(feats, np.maximum(a, 0), np.maximum(b, 0))
    X = pd.concat([
        build_pair_features(feats, tfidf, p1, p2).reset_index(drop=True),
        F2.build_crowd_features(feats, F2.build_corpus_state(feats), p1, p2)
        .reset_index(drop=True),
        F2.build_ncd_units_lsa(feats, tfidf, p1, p2).reset_index(drop=True),
        F2.build_specialist_features(feats, p1, p2).reset_index(drop=True),
        F2.build_key_conflict_features(feats, p1, p2, top_keys)
        .reset_index(drop=True)], axis=1)
    X = F2.rank_normalise(X)
    nd, ns_, ad = F2.pair_bags(feats, p1, p2)
    return (X, feats["category"][p1], ok,
            (nd, ns_, ad, F2.key_bags(feats, p1, p2)), feats)


BL_IDX = ((0, 1), (2, None), (3, None))


def nb_apply(bags, rows, W3, n):
    out = np.zeros((n, 3 * NS), dtype=np.float32)
    for bi, (bd, bs) in enumerate(BL_IDX):
        d_ = [bags[bd][i] for i in rows]
        s_ = [bags[bs][i] for i in rows] if bs is not None else None
        F2.apply_nb(np.arange(n), W3[bi], d_, s_, out[:, bi * NS:(bi + 1) * NS])
    return out


def nb_fit(bags, rows, y):
    return [F2.fit_nb(rows, y, bags[bd], bags[bs] if bs is not None else None,
                      MIN_COUNT) for bd, bs in BL_IDX]


# ==================================================================== human real
mh = pd.read_parquet(find_file("matches.parquet"))
ih = pd.read_parquet(find_file("items_human.parquet"))
ycol = "target" if "target" in mh.columns else mh.columns[-1]
TOP_KEYS = F2.compute_top_keys(build_item_features(ih))
Xr, cr, okr, bagsr, feats_h = featurise(ih, mh, TOP_KEYS)
sel = np.flatnonzero(okr)
yr = mh[ycol].to_numpy().astype(np.int8)[sel]
Xr, cr = Xr.iloc[sel].reset_index(drop=True), np.asarray(cr)[sel]
BAG_R = tuple([z[i] for i in sel] for z in bagsr)
log(f"human real {Xr.shape}, prevalence {yr.mean():.5f}")

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(Xr))
tr, va = np.sort(perm[:len(perm) // 2]), np.sort(perm[len(perm) // 2:])
CATS = sorted(set(cr.tolist()))
C2I = {c: i for i, c in enumerate(CATS)}
COLS = list(Xr.columns) + NB_COLS

# ============================================ ruler E: mined near-duplicate negs
state = F2.build_corpus_state(feats_h)
band = state["band_of"][0]
ids_h = pd.Index(feats_h["id"])
icat = np.array([C2I[c] for c in feats_h["category"]], dtype=np.int64)
known = set(map(tuple, np.sort(np.column_stack(
    [ids_h.get_indexer(mh["id1"].to_numpy()[sel]),
     ids_h.get_indexer(mh["id2"].to_numpy()[sel])]), axis=1).tolist()))
key = band.astype(np.int64) * 1000 + icat
order = np.argsort(key, kind="stable")
mined, r = [], np.random.default_rng(SEED)
s = 0
while s < len(order) and len(mined) < 240_000:
    e = s
    while e < len(order) and key[order[e]] == key[order[s]]:
        e += 1
    mem = order[s:e]
    if 2 <= len(mem) <= 200:
        for _ in range(min(len(mem), 8)):
            a, b = r.choice(mem, 2, replace=False)
            t = (min(a, b), max(a, b))
            if t not in known:
                mined.append(t)
    s = e
log(f"mined {len(mined)} near-duplicate non-matches for ruler E")
idv = np.asarray(feats_h["id"])
mdf = pd.DataFrame({"id1": idv[[p[0] for p in mined]],
                    "id2": idv[[p[1] for p in mined]]})
Xm, cm, okm, bagsm, _ = featurise(ih, mdf, TOP_KEYS)
jm = np.flatnonzero(okm)
Xm, cm = Xm.iloc[jm].reset_index(drop=True), np.asarray(cm)[jm]
BAG_M = tuple([z[i] for i in jm] for z in bagsm)
del state, band, order, key, mined, feats_h, bagsm, bagsr, mdf
gc.collect()
log(f"mined matrix {Xm.shape}")


def thin_pos(rows, y, target, rr):
    pos, neg = rows[y[rows] == 1], rows[y[rows] == 0]
    want = int(round(target * len(neg) / (1.0 - target)))
    if want >= len(pos):
        return rows
    return np.sort(np.concatenate([rr.choice(pos, want, replace=False), neg]))


# Ruler E exactly as round 19 CALIBRATED it: every real val row, plus only as
# many mined negatives as it takes to reach the test prevalence. Rounds 25-28
# thinned the positives AND added the whole pool, landing at 0.043 -- same rows,
# different ruler, and none of round 19's +-0.0087 applies to that one.
_npos = int(yr[va].sum())
_want = int(round(_npos * (1 - TEST_PREV) / TEST_PREV))
_add = max(0, min(len(Xm), _want - int((yr[va] == 0).sum())))
EK = np.sort(np.random.default_rng(SEED).choice(len(Xm), _add, replace=False))
y_E = np.concatenate([yr[va], np.zeros(len(EK), dtype=np.int8)])
c_E = np.concatenate([cr[va], cm[EK]])
log(f"ruler E: {len(va)} real + {len(EK)} mined = {len(y_E)}, "
    f"prevalence {y_E.mean():.5f}  (round 19 had 0.11226)")

# ====================================================================== llm side
llm = pq.read_table(find_file("matches_llm.parquet")).to_pandas()
tcol = "target" if "target" in llm.columns else llm.columns[-1]
kk = np.sort(np.random.default_rng(SEED).choice(len(llm), N_LLM, False))
llm = llm.iloc[kk].reset_index(drop=True)
need = pd.Index(pd.unique(np.concatenate([llm["id1"].to_numpy(),
                                          llm["id2"].to_numpy()])))
kb = []
for batch in pq.ParquetFile(find_file("items.parquet")).iter_batches(
        batch_size=400_000, columns=["id", "name", "attributes", "category"]):
    d = batch.to_pandas()
    d = d[d["id"].isin(need)]
    if len(d):
        kb.append(d)
items_llm = pd.concat(kb, ignore_index=True).drop_duplicates("id")
del kb
gc.collect()
log(f"llm items {len(items_llm)}")

pos_of = pd.Index(items_llm["id"].to_numpy())
XL, SL, CL, BAG_L = [], [], [], [[], [], [], []]
for s in range(0, len(llm), SHARD):
    sub = llm.iloc[s:s + SHARD].reset_index(drop=True)
    ids = pd.unique(np.concatenate([sub["id1"].to_numpy(), sub["id2"].to_numpy()]))
    rows = pos_of.get_indexer(ids)
    X, c, ok, bags, _ = featurise(
        items_llm.iloc[rows[rows >= 0]].reset_index(drop=True), sub, TOP_KEYS)
    js = np.flatnonzero(ok)
    XL.append(X.iloc[js].reset_index(drop=True))
    SL.append(sub[tcol].to_numpy()[ok])
    CL.append(np.asarray(c)[ok])
    for bi in range(4):
        BAG_L[bi].extend(bags[bi][i] for i in js)
    log(f"  llm shard {s // SHARD + 1}/{-(-len(llm) // SHARD)}: {len(js)}")

XL = pd.concat(XL, ignore_index=True)
SL, CL = np.concatenate(SL), np.concatenate(CL)
YL = (SL > 0.5).astype(np.int8)
BAG_L = tuple(BAG_L)
del items_llm, llm
gc.collect()
log(f"llm matrix {XL.shape}, hard prevalence {YL.mean():.5f}, "
    f"soft mean {SL.mean():.5f}")

PAR = dict(objective="binary", learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, num_threads=4, verbose=-1, seed=SEED)


def code(cat):
    return np.array([C2I.get(c, -1) for c in cat], dtype=np.float32)


def assemble(X, nb, cat):
    return np.column_stack([X.to_numpy(dtype=np.float32), nb, code(cat)])


def eval_E(model, W3):
    """Ruler E, built with the model's OWN nb tables -- what would ship."""
    nb_r = nb_apply(BAG_R, va, W3, len(va))
    nb_m = nb_apply(BAG_M, EK, W3, len(EK))
    M = np.vstack([assemble(Xr.iloc[va], nb_r, cr[va]),
                   assemble(Xm.iloc[EK], nb_m, cm[EK])])
    return macro_ap(y_E, model.predict(M, num_threads=4), c_E)



RES = {}
ORDER = []


def run(name, X, y_hard, label_y, bags, rows, cat, weight=None):
    """Fit this variant's own NB tables on its own training rows, build its own
    columns, train, score on the CORRECTLY built ruler E."""
    W3 = nb_fit(bags, rows, y_hard)
    nb = nb_apply(bags, rows, W3, len(rows))
    M = assemble(X, nb, cat)
    m = lgb.train(PAR, lgb.Dataset(M, label=label_y, weight=weight,
                                   categorical_feature=[M.shape[1] - 1]),
                  num_boost_round=600)
    RES[name] = eval_E(m, W3)
    ORDER.append(name)
    log(f"{name:24s} E = {RES[name]:.5f}   ({len(rows)} rows)")
    del M, nb
    gc.collect()
    return m


nL = len(XL)
rngS = np.random.default_rng(SEED)


def pool(n_llm, soft, weight_llm=1.0, subset=None):
    """human train half + a slice of matches_llm."""
    idx = np.arange(nL) if subset is None else subset
    if n_llm < len(idx):
        idx = np.sort(rngS.choice(idx, n_llm, replace=False))
    X = pd.concat([Xr.iloc[tr].reset_index(drop=True), XL.iloc[idx]],
                  ignore_index=True)
    yh_ = np.concatenate([yr[tr], YL[idx]])
    lab = np.concatenate([yr[tr].astype(np.float64),
                          (SL[idx] if soft else YL[idx]).astype(np.float64)])
    cat = np.concatenate([cr[tr], CL[idx]])
    bags = tuple(list(BAG_R[bi][i] for i in tr) + list(BAG_L[bi][i] for i in idx)
                 for bi in range(4))
    w = None
    if weight_llm != 1.0:
        w = np.concatenate([np.ones(len(tr)),
                            np.full(len(idx), weight_llm)])
    return X, yh_, lab, bags, np.arange(len(X)), cat, w


# =============================================================================
# ROUND 43 -- THE CORRECTED EXPORT, AND THE LAST ONE
#
# Round 42 had two defects and its artefact is discarded:
#
#  1. LABEL LEAKAGE. It fitted the NB tables on the WHOLE pooled set -- which
#     includes the human VALIDATION half -- and then scored ruler E with them.
#     Same construction as round 41 read 0.37206 there and 0.44792 in round 42;
#     that +0.076 is the leak. Round 34 got this right and round 42 regressed it.
#  2. A LUCKY BASELINE. Its "1 member" comparison used member 0 specifically,
#     which scored 0.43687 against a member mean of 0.41994 (+0.8 sd), so the
#     ensemble was measured against an optimistic single. Corrected, the
#     ensemble wins by ~+0.013, matching round 36's clean +0.01355 -- but the
#     kernel had already decided to export one model.
#
# So phase 1 here fits its tables on TRAINING ROWS ONLY and compares the
# ensemble against the MEAN member, not against a chosen one. Phase 2 fits the
# shipped tables on everything, which is correct at inference and leaks nothing.
#
# What ships combines the two effects measured on clean instruments:
#   round 36  rank-average over llm draws        +0.01355
#   round 41  gate llm rows out of the B models  +0.00514  (b3 control passed)
# =============================================================================
from scipy.stats import rankdata

K = 5
N_A, N_B = 2000, 2000
SUB = 300_000
LOSERS = ["Мебель", "Обувь", "Одежда", "Ювелирные изделия",
          "Галантерея и аксессуары"]
MODELS = "/kaggle/working/models"
os.makedirs(MODELS, exist_ok=True)

ALLH = np.arange(len(Xr))
XF = pd.concat([Xr, XL], ignore_index=True)
YF = np.concatenate([yr, YL])
LF = np.concatenate([yr.astype(np.float64), SL.astype(np.float64)])
CF = np.concatenate([cr, CL])
IS_LLM = np.concatenate([np.zeros(len(ALLH), bool), np.ones(len(XL), bool)])
BAGF = tuple(list(BAG_R[bi]) + list(BAG_L[bi]) for bi in range(4))

from src import features as F1  # noqa: E402  -- _tokens for the extra pass
# ===================================================== round 70: the big table
# Round 66, paired over 4 draws: big table +0.00796 (4/4), min_count 25 alone
# +0.00441 (3/4), and TOGETHER +0.01934 (4/4) -- super-additive by +0.0070,
# which is the stale-threshold hypothesis confirmed. min_count 5 was calibrated
# when this table was fitted on 782,827 pairs; at 6.78M it admits ~270k keys
# seen five times in seven million, whose log-odds pin near a constant extreme.
MC_NAME = 25
N_EXTRA = 6_000_000
BLOCK = 1_000_000


def _fin2(p1, n1, p2, n2, npos, nneg, mc):
    """Finalise from TWO counter pairs without merging them -- merging would
    copy ~1.9M keys and this kernel is already holding the pooled matrix."""
    lp, ln = np.log(npos + 2.0), np.log(nneg + 2.0)
    out = {}
    for k in set(p1) | set(n1) | set(p2) | set(n2):
        a = p1.get(k, 0) + p2.get(k, 0)
        b = n1.get(k, 0) + n2.get(k, 0)
        if a + b >= mc:
            out[k] = float((np.log(a + 1.0) - lp) - (np.log(b + 1.0) - ln))
    return out


# The extra pool is matches_llm rows NOT drawn into N_LLM, so it is disjoint
# from training AND from ruler E (whose human half and mined negatives are all
# items_human, and llm items are disjoint from human items).
_lall = pq.read_table(find_file("matches_llm.parquet")).to_pandas()
_tc = "target" if "target" in _lall.columns else _lall.columns[-1]
_used = np.zeros(len(_lall), bool)
_used[np.random.default_rng(SEED).choice(len(_lall), N_LLM, False)] = True
_avail = np.flatnonzero(~_used)
_take = np.sort(np.random.default_rng(SEED + 5).choice(
    _avail, min(N_EXTRA, len(_avail)), replace=False))
_E1 = _lall["id1"].to_numpy()[_take]
_E2 = _lall["id2"].to_numpy()[_take]
_EY = (_lall[_tc].to_numpy()[_take] > 0.5).astype(np.int8)
del _lall, _used, _avail, _take
gc.collect()

_XP, _XN = Counter(), Counter()
_XPOS = _XNEG = 0
_ITEMS = find_file("items.parquet")
_done = 0
for _b0 in range(0, len(_E1), BLOCK):
    _b1 = min(_b0 + BLOCK, len(_E1))
    _a, _b, _yb = _E1[_b0:_b1], _E2[_b0:_b1], _EY[_b0:_b1]
    _need = pd.Index(pd.unique(np.concatenate([_a, _b])))
    _toks, _ids = [], []
    for _batch in pq.ParquetFile(_ITEMS).iter_batches(
            batch_size=400_000, columns=["id", "name"]):
        _bid = _batch.column("id").to_numpy()
        _m = _need.get_indexer(_bid) >= 0
        if _m.any():
            _nm = _batch.column("name").to_pylist()
            _ids.append(_bid[_m])
            _toks.extend(set(F1._tokens(str(_nm[i])))
                         for i in np.flatnonzero(_m))
    _ids = np.concatenate(_ids) if _ids else np.array([], dtype=np.int64)
    _po = pd.Index(_ids)
    _i1, _i2 = _po.get_indexer(_a), _po.get_indexer(_b)
    _ok = (_i1 >= 0) & (_i2 >= 0)
    for _r in np.flatnonzero(_ok):
        _ti, _tj = _toks[_i1[_r]], _toks[_i2[_r]]
        _c = _XP if _yb[_r] else _XN
        _c.update(_ti ^ _tj)
        _c.update("~" + t for t in (_ti & _tj))
    _XPOS += int(_yb[_ok].sum())
    _XNEG += int(_ok.sum()) - int(_yb[_ok].sum())
    _done += int(_ok.sum())
    del _toks, _ids, _po, _i1, _i2, _ok
    gc.collect()
    log(f"  extra block {_b0 // BLOCK + 1}/{-(-len(_E1) // BLOCK)}: "
        f"{_done} pairs, {len(_XP) + len(_XN)} keys")
del _E1, _E2, _EY
gc.collect()


def big_name(rows):
    """Name table over `rows` PLUS the streamed extra pool, pruned at MC_NAME.
    `rows` must be the same rows the caller passes to nb_fit, or phase 1 stops
    being leak-free."""
    p, n = Counter(), Counter()
    for i in rows:
        c = p if YF[i] else n
        c.update(BAGF[0][i])
        c.update("~" + t for t in BAGF[1][i])
    npos = int(YF[rows].sum())
    return _fin2(p, n, _XP, _XN, npos + _XPOS,
                 (len(rows) - npos) + _XNEG, MC_NAME)

LLM_OFF = len(ALLH)
nllm = len(XF) - LLM_OFF
DRAWS = [np.sort(np.random.default_rng(SEED + 17 * k).choice(nllm, SUB, False))
         + LLM_OFF for k in range(K)]
log(f"pooled {len(XF)} rows = {len(ALLH)} human + {nllm} llm, K={K}")

# ===================================================== phase 1: CLEAN measurement
# Tables fitted on the phase-1 TRAINING rows only. Ruler E's human half comes
# from `va`, which is excluded here, so nothing leaks.
TRAIN1 = np.concatenate([tr, np.arange(LLM_OFF, len(XF))])
# fit_nb indexes y AND the bags by GLOBAL row position, so y must be the
# full-length array even when `rows` is a subset. Slicing it here is what
# killed v1 with IndexError at index 782827.
W3m = list(nb_fit(BAGF, TRAIN1, YF))
_shipped_name = W3m[0]
W3m[0] = big_name(TRAIN1)
log(f'phase-1 name table: shipped {len(_shipped_name)} entries -> '
    f'big/mc{MC_NAME} {len(W3m[0])} entries')
NBm = nb_apply(BAGF, np.arange(len(XF)), W3m, len(XF))
Mm = assemble(XF, NBm, CF)
del NBm
gc.collect()
log(f"phase-1 matrix {Mm.shape} (tables fitted on {len(TRAIN1)} training rows)")

nb_r = nb_apply(BAG_R, va, W3m, len(va))
nb_m = nb_apply(BAG_M, EK, W3m, len(EK))
ME = np.vstack([assemble(Xr.iloc[va], nb_r, cr[va]),
                assemble(Xm.iloc[EK], nb_m, cm[EK])])
del nb_r, nb_m
gc.collect()
nE = len(ME)

P1 = []
for k, d in enumerate(DRAWS):
    rows = np.concatenate([tr, d])
    m = lgb.train(PAR, lgb.Dataset(Mm[rows], label=LF[rows],
                                   categorical_feature=[Mm.shape[1] - 1]),
                  num_boost_round=N_A)
    P1.append(m.predict(ME, num_threads=4))
    log(f"  member {k}: E = {macro_ap(y_E, P1[-1], c_E):.5f}")
    del m
    gc.collect()

# gated B models on member 0's training set
base_rows = np.concatenate([tr, DRAWS[0]])
CB = {}
for ci, c in enumerate(CATS):
    sel_ = base_rows[CF[base_rows] == c]
    if c in LOSERS:
        sel_ = sel_[~IS_LLM[sel_]]
    if len(sel_) < 2000 or len(np.unique(YF[sel_])) < 2:
        continue
    CB[ci] = lgb.train(dict(PAR, num_leaves=63),
                       lgb.Dataset(Mm[sel_][:, :-1], label=LF[sel_]),
                       num_boost_round=N_B)
log(f"phase-1 gated B models: {len(CB)} of {len(CATS)}")


def with_b(pa):
    pb = pa.copy()
    for ci, c in enumerate(CATS):
        if ci not in CB:
            continue
        kk = c_E == c
        if kk.any():
            pb[kk] = CB[ci].predict(ME[kk][:, :-1], num_threads=4)
    return 0.5 * rankdata(pa) / nE + 0.5 * rankdata(pb) / nE


singles_b = [macro_ap(y_E, with_b(p), c_E) for p in P1]
ens = np.mean([rankdata(p) / nE for p in P1], axis=0)
ens_b = macro_ap(y_E, with_b(ens), c_E)
singles_a = [macro_ap(y_E, p, c_E) for p in P1]

print("\n=== PHASE 1 (clean tables, ensemble vs the MEAN member) ===")
print(f"  members, A alone     {' '.join(f'{s:.5f}' for s in singles_a)}")
print(f"  mean single, A alone {np.mean(singles_a):.5f}  sd {np.std(singles_a, ddof=1):.5f}")
print(f"  ensemble,    A alone {macro_ap(y_E, ens, c_E):.5f}")
print(f"\n  members + gated B    {' '.join(f'{s:.5f}' for s in singles_b)}")
print(f"  MEAN single + gated B {np.mean(singles_b):.5f}   <- the honest baseline")
print(f"  ensemble  + gated B   {ens_b:.5f}   ({ens_b - np.mean(singles_b):+.5f})")
print(f"\n  for comparison, on clean instruments:")
print(f"    round 34 single + ungated B  0.36570")
print(f"    round 41 single + gated B    0.37720")
SHIP_ENS = ens_b > np.mean(singles_b) + 0.0042
print(f"  -> exporting {K if SHIP_ENS else 1} member(s)")
del Mm, ME, P1, CB, ens
gc.collect()

# ============================================ phase 2: the shipped artefacts
# Tables fitted on EVERYTHING, which is what a model scoring the test set wants
# and leaks nothing, because the test set is not in this data at all.
W3s = list(nb_fit(BAGF, np.arange(len(XF)), YF))
W3s[0] = big_name(np.arange(len(XF)))
log(f'shipped name table: {len(W3s[0])} entries at mc{MC_NAME} '
    f'over {len(XF) + _done} pairs')
for nm, W in zip(("nb_name", "nb_attr", "nb_key"), W3s):
    F2.save_nb_table(f"{MODELS}/{nm}.npz", W)
NBs = nb_apply(BAGF, np.arange(len(XF)), W3s, len(XF))
del BAGF
gc.collect()
Ms = assemble(XF, NBs, CF)
del XF, NBs
gc.collect()
log(f"phase-2 matrix {Ms.shape}, shipped tables saved")

NMEM = K if SHIP_ENS else 1
for k in range(NMEM):
    rows = np.concatenate([ALLH, DRAWS[k]])
    m = lgb.train(PAR, lgb.Dataset(Ms[rows], label=LF[rows],
                                   categorical_feature=[Ms.shape[1] - 1]),
                  num_boost_round=N_A)
    m.save_model(f"{MODELS}/model_a_{k}.txt" if NMEM > 1
                 else f"{MODELS}/model_a.txt")
    log(f"  member {k} saved ({len(rows)} rows)")
    del m
    gc.collect()

kept_b, gated = {}, []
for ci, c in enumerate(CATS):
    rws = np.flatnonzero(CF == c)
    if c in LOSERS:
        rws = rws[~IS_LLM[rws]]
        gated.append(c)
    if len(rws) < 2000 or len(np.unique(YF[rws])) < 2:
        continue
    fb = lgb.train(dict(PAR, num_leaves=63),
                   lgb.Dataset(Ms[rws][:, :-1], label=LF[rws]),
                   num_boost_round=N_B)
    fb.save_model(f"{MODELS}/model_b_{ci}.txt")
    kept_b[c] = int(len(rws))
log(f"{len(kept_b)} B models saved, {len(gated)} of them gated (human rows only)")

meta = {"features": COLS, "categories": CATS, "top_keys": TOP_KEYS,
        "corpus_cols": F2.CORPUS_COLS, "min_count": MIN_COUNT,
        "name_min_count": MC_NAME,
        "name_table_extra_pairs": int(_done),
        "name_table_entries": int(len(W3s[0])),
        "best_iter_a": N_A, "best_iter_b": {c: N_B for c in kept_b},
        "b_train_rows": kept_b, "gated_categories": gated,
        "ensemble_members": NMEM, "llm_rows_per_member": SUB,
        "trained_on": f"matches.parquet (all) + matches_llm soft; {NMEM} member"
                      f"(s) over different llm draws; B models gated in "
                      f"{len(gated)} grid-SKU categories",
        "n_human": int(len(ALLH)), "n_llm_pool": int(nllm),
        "ruler_E_mean_single_plus_gatedB": round(float(np.mean(singles_b)), 5),
        "ruler_E_ensemble_plus_gatedB": round(float(ens_b), 5),
        "ruler_E_member_sd": round(float(np.std(singles_a, ddof=1)), 5),
        "change_vs_v7": ("per-category B models go from 31 leaves/800 rounds to 63/2000 -- they had been at round 34's settings for ~50 rounds with nobody asking why. Round 86 (A trained once per draw and SHARED, so the delta isolates B): +0.01088 on ruler E 3/3 and +0.00381 on ruler G 3/3. Round 88 confirmed it survives SHIPPED DEPTH at +0.00950 3/3 and is additive with the big table (+0.01861 together, vs +0.01940 if perfectly additive). Round 91 priced it on ruler E's HUMAN half at +0.00249 3/3. Weight 0.5 was checked at the same time and is right: 0.25 and 0.75 both lose. Inference cost is more trees in a pass that already runs, and round 50 measured trees as close to free here -- featurisation dominates the container. The cost is ARCHIVE SIZE: B models are 5.1x more leaves."),
        "excluded_min_data_in_leaf_2000": ("+0.00647 on human-labelled pairs standing alone (round 89, 4/4, sd 0.00092) but only +0.00078 at shipped depth (round 91, 3/3) -- the ensemble and B models already capture it. Under the 0.004 bar, so left out."),
        "ruler_E_delta_inflation": ("ROUND 89: ruler E is 182,827 human validation rows PLUS 232,350 LSH-mined negatives, and its DELTAS run 3.7x to 37x hot depending on the change -- its LEVEL is fine, which is why round 19's calibration did not catch this. Round 85 read min_data_in_leaf 20000 at +0.067 on the full ruler; on the human half it LOSES 0.0126. Price every future change on the human half. The v4->v6 leaderboard step (+0.00358) against ruler E's +0.03 to +0.05 for the same change is independent support."),
        "expected_LB": ("v6 scored 0.35502. Round 91 puts v8 at +0.00556 over v6 on human-labelled pairs at shipped depth (3/3, sd 0.00072), so expect ~0.3606. The old +0.019 figure was ruler E's mined half talking."),
        "change_vs_v6": ("name log-odds table fitted on 6.78M pairs instead of "
                        "782,827 and pruned at 25 instead of 5. Round 66 paired over "
                        "4 draws: big alone +0.00796 (4/4), mc25 alone +0.00441 "
                        "(3/4), together +0.01934 (4/4) -- super-additive by "
                        "+0.0070. 127 leaves was REJECTED (3/4, and it lowers "
                        "the delta in combination). Zero inference cost: the "
                        "table is fitted here and shipped as a file."),
        "note": ("phase-1 tables fitted on TRAINING ROWS ONLY -- round 42 "
                 "leaked the validation half into them and read +0.076 high. "
                 "Shipped tables are fitted on everything, which is correct at "
                 "inference. Ruler E is biased against this model's kind of "
                 "competence. The old 'E->LB slope 0.54 to 1.22' claim is "
                 "WITHDRAWN: see ruler_E_delta_inflation.")}
with open(f"{MODELS}/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

tot = sum(os.path.getsize(os.path.join(MODELS, f)) for f in os.listdir(MODELS))
for f_ in sorted(os.listdir(MODELS)):
    log(f"  {f_}  {os.path.getsize(os.path.join(MODELS, f_)) / 1e6:.2f} MB")
print(f"\n=== ROUND 94: THE v8 EXPORT ===")
print(f"members {NMEM}, gated categories {len(gated)}, total {tot / 1e6:.1f} MB")
print(f"phase-1 ensemble + gated B on ruler E: {ens_b:.5f}")
print(f"inference cost: {NMEM} x {N_A} model-A trees plus one B pass.")
log("done")
