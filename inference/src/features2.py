"""
Round-5 feature additions, shared VERBATIM by the training kernel and the
container. Train/inference parity is not negotiable here: every one of these
features is either corpus-fitted or table-driven, so a difference of a single
constant between the two sides silently destroys the model.

Three families, chosen by measured gain per second of feature build:

  NB token log-odds   +0.0405 for  49s   which words, when they appear on ONE
                                         side only, historically meant "not a
                                         duplicate". IDF only knows rarity.
  crowdedness         +0.0081 for  29s   is this match special, or does this
                                         item look like a thousand others?
  NCD/units/align/LSA +0.0056 for  61s   similarities the fuzzy+tf-idf block
                                         structurally cannot express.

Cut after measurement: char-4gram MinHash (-0.00001 for 55s) and neighbour
rank/margin (+0.0015 for 105s). Both won in isolation and neither survived
contact with the rest of the stack.

SCALE-FREE BY CONSTRUCTION. Crowd counts are divided by corpus size and the
corpus-fitted columns are rank-normalised within the run before the trees see
them. This is the fix for the one asymmetry that explains our validation-to-
leaderboard gap: of the three models we have leaderboard scores for, the two
that are scale-free (the cross-encoder, which has no corpus features, and the
rank-averaged lexical probe) calibrate to within 0.03, while the GBDT — the
only one that learns split thresholds on absolute corpus-dependent values — is
off by +0.10. Trees are invariant to monotone transforms of a single feature,
so rank-normalising costs nothing within a run and removes the drift across
runs.
"""
import re
import zlib
from collections import Counter  # noqa: F401  (used by compute_top_keys)

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD

try:
    from rapidfuzz import fuzz as _rf_fuzz
    HAVE_RF = True
except ImportError:  # pragma: no cover
    HAVE_RF = False

SEED = 42
N_PERM, BANDS, ROWS = 128, 32, 4

# corpus-fitted -> rank-normalised within each run. NB columns are NOT here:
# their weights come from a shipped table, so they are already on a fixed scale.
CORPUS_COLS = [
    "tfidf_name_word", "tfidf_name_char", "tfidf_attr_word",
    "idf_miss_a", "idf_miss_b", "idf_miss_max", "idf_miss_min",
    "idf_miss_ratio", "idf_shared", "idf_shared_ratio",
    "mh_jac_word", "mh_bands_agree", "crowd_a", "crowd_b", "crowd_min",
    "crowd_max", "crowd_cat_max", "shared_rarest", "shared_mean_idf",
    "crowd_evidence", "rare_over_crowd", "svd_cos", "svd_l2",
]

_UNITS = ["мг", "кг", "гр", "г", "мл", "л", "мм", "см", "м", "шт",
          "гб", "тб", "мб", "gb", "tb", "mb", "вт", "w", "мач", "mah"]
_SCALE = dict(zip(_UNITS, [1e-3, 1e3, 1.0, 1.0, 1.0, 1e3, 0.1, 1.0, 100.0, 1.0,
                           1e3, 1e6, 1.0, 1e3, 1e6, 1.0, 1.0, 1.0, 1.0, 1.0]))
_FAMILY = dict(zip(_UNITS, ["m", "m", "m", "m", "v", "v", "l", "l", "l", "c",
                            "mem", "mem", "mem", "mem", "mem", "mem", "p", "p",
                            "cap", "cap"]))
_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(" + "|".join(_UNITS) + r")\b")

NB_STATS = ["sum", "mean", "max", "min", "n", "s_sum", "s_mean", "s_max",
            "s_min", "gap"]
NB_NAME_COLS = [f"nbname_{s}" for s in NB_STATS]
NB_ATTR_COLS = [f"nbattr_{s}" for s in NB_STATS]

CROWD_COLS = ["mh_jac_word", "mh_bands_agree", "crowd_a", "crowd_b",
              "crowd_min", "crowd_max", "crowd_cat_max", "shared_rarest",
              "shared_mean_idf", "crowd_evidence", "rare_over_crowd"]
NCD_COLS = ["ncd", "unit_jac", "unit_conflict", "unit_rel_diff", "align_mean",
            "align_min", "align_unmatched", "svd_cos", "svd_l2"]
SPECIALIST_COLS = ["sz_num_eq", "sz_num_conflict", "sz_let_eq",
                   "sz_let_conflict", "sz_present", "sz_gap",
                   "code_shared_max_len", "code_best_ratio", "code_asym",
                   "code_n_diff", "col_conflict", "col_shared", "gen_conflict",
                   "mat_jac", "axis_any"]
EXTRA_COLS = (CROWD_COLS + NCD_COLS + SPECIALIST_COLS
              + NB_NAME_COLS + NB_ATTR_COLS)
# KEYCONF_COLS / NB_KEY_COLS are appended by the round-15 block below; the
# authoritative column order for a build always comes from meta["features"].


# ---------------------------------------------------------------- ordering
def canonical_order(item_feats, p1, p2):
    """The a/b assignment in a pair is arbitrary, so every asymmetric feature
    carries avoidable noise. Order on name length -- deterministic, and free:
    it is a permutation of the inputs, not extra work. Worth +0.0045."""
    nn = item_feats["name_norm"]
    la = np.fromiter((len(nn[i]) for i in p1), dtype=np.int32, count=len(p1))
    lb = np.fromiter((len(nn[i]) for i in p2), dtype=np.int32, count=len(p2))
    sw = la > lb
    return np.where(sw, p2, p1), np.where(sw, p1, p2)


# ---------------------------------------------------------------- minhash
def _signatures(bags, n_items):
    """128-permutation MinHash with no per-item Python loop.

    Every token is hashed once; the permuted values are computed for the UNIQUE
    hashes only, then np.minimum.reduceat collapses each item's slice of the
    flat token array. Turns O(items x perms x tokens) into a few vector passes.
    """
    flat = []
    offs = np.zeros(n_items + 1, dtype=np.int64)
    for i, s in enumerate(bags):
        flat.extend(s)
        offs[i + 1] = len(flat)
    if not flat:
        return np.zeros((n_items, N_PERM), dtype=np.int32)
    inv, uniq = pd.factorize(pd.Series(flat, dtype="string"), sort=False)
    h = (pd.util.hash_pandas_object(pd.Series(uniq), index=False)
         .to_numpy().astype(np.int64) & 0x7FFFFFFF)
    rng = np.random.default_rng(SEED)
    prime = np.int64(2147483647)
    a = rng.integers(1, prime, N_PERM, dtype=np.int64)
    b = rng.integers(0, prime, N_PERM, dtype=np.int64)
    sig = np.full((n_items, N_PERM), np.iinfo(np.int32).max, dtype=np.int32)
    ne = np.nonzero(np.diff(offs) > 0)[0]
    starts = offs[ne]
    for p0 in range(0, N_PERM, 16):
        p1 = min(p0 + 16, N_PERM)
        perm = ((a[p0:p1, None] * h[None, :] + b[p0:p1, None]) % prime).astype(np.int32)
        sig[ne, p0:p1] = np.minimum.reduceat(perm[:, inv], starts, axis=1).T
        del perm
    return sig


def _band(sig, n_items, icode=None):
    band_of = []
    crowd = np.zeros(n_items, dtype=np.float32)
    for bnd in range(BANDS):
        cols = sig[:, bnd * ROWS:(bnd + 1) * ROWS]
        key = np.zeros(n_items, dtype=np.uint64)
        for r in range(ROWS):
            key = key * np.uint64(1000003) ^ cols[:, r].astype(np.uint64)
        if icode is not None:
            key = key * np.uint64(1000003) ^ icode.astype(np.uint64)
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        band_of.append(inv)
        crowd += cnt[inv]
    return band_of, crowd / BANDS


def build_corpus_state(item_feats):
    """Everything derived from the ITEMS file alone. Computed once per run."""
    tok = item_feats["name_tok"]
    n_items = len(tok)
    cats = item_feats["category"]
    uniq_c = {c: i for i, c in enumerate(sorted(set(cats.tolist())))}
    icode = np.array([uniq_c[c] for c in cats], dtype=np.int64)

    sig = _signatures(tok, n_items)
    band_of, crowd = _band(sig, n_items)
    _, crowd_cat = _band(sig, n_items, icode=icode)

    df = Counter()
    for s in tok:
        df.update(s)
    inv_df = {k: float(np.log(n_items / (1.0 + v))) for k, v in df.items()}
    return {"sig": sig, "band_of": band_of, "crowd": crowd,
            "crowd_cat": crowd_cat, "inv_df": inv_df, "n_items": n_items}


def build_crowd_features(item_feats, state, pos1, pos2):
    tok = item_feats["name_tok"]
    n = len(pos1)
    sig, n_items = state["sig"], state["n_items"]
    ja = (sig[pos1] == sig[pos2]).mean(axis=1).astype(np.float32)
    agree = np.zeros(n, dtype=np.float32)
    for inv in state["band_of"]:
        agree += (inv[pos1] == inv[pos2])
    inv_df = state["inv_df"]
    sh_rare = np.empty(n, dtype=np.float32)
    sh_mean = np.empty(n, dtype=np.float32)
    for r in range(n):
        sh = tok[pos1[r]] & tok[pos2[r]]
        if sh:
            v = [inv_df.get(x, 0.0) for x in sh]
            sh_rare[r] = max(v)
            sh_mean[r] = float(np.mean(v))
        else:
            sh_rare[r] = sh_mean[r] = -1.0
    crowd = state["crowd"]
    # divided by corpus size: raw bucket occupancy scales with n, and the
    # container's items file is a different size from the training one
    ca = (crowd[pos1] / n_items).astype(np.float32)
    cb = (crowd[pos2] / n_items).astype(np.float32)
    cc = (np.maximum(state["crowd_cat"][pos1],
                     state["crowd_cat"][pos2]) / n_items).astype(np.float32)
    hi = np.maximum(np.maximum(crowd[pos1], crowd[pos2]), 1.0)
    out = np.column_stack([
        ja, agree, ca, cb, np.minimum(ca, cb), np.maximum(ca, cb), cc,
        sh_rare, sh_mean,
        (ja * np.log1p(1.0 / hi)).astype(np.float32),
        (sh_rare / np.log1p(hi)).astype(np.float32)]).astype(np.float32)
    return pd.DataFrame(out, columns=CROWD_COLS)


# ------------------------------------------------------- ncd / units / align
def _qty(s):
    o = {}
    for v, u in _UNIT_RE.findall(s):
        o.setdefault(_FAMILY[u], set()).add(
            round(float(v.replace(",", ".")) * _SCALE[u], 4))
    return o


def build_ncd_units_lsa(item_feats, tfidf, pos1, pos2):
    names = item_feats["name_norm"]
    raw = item_feats["name"]
    tok = item_feats["name_tok"]
    n = len(pos1)

    clen = np.array([len(zlib.compress(s.encode("utf-8"), 6)) for s in names],
                    dtype=np.float32)
    ncd = np.empty(n, dtype=np.float32)
    for r in range(n):
        i, j = pos1[r], pos2[r]
        cab = len(zlib.compress((names[i] + " " + names[j]).encode("utf-8"), 6))
        ca, cb = clen[i], clen[j]
        hi = max(ca, cb)
        ncd[r] = (cab - min(ca, cb)) / hi if hi else -1.0

    # RAW names: normalisation strips the decimal comma, so "0,5кг" would
    # become "0 5кг" and parse as 5 kg.
    Q = [_qty(s.lower()) for s in raw]
    uj = np.empty(n, dtype=np.float32)
    uc = np.empty(n, dtype=np.float32)
    ur = np.empty(n, dtype=np.float32)
    am = np.empty(n, dtype=np.float32)
    an = np.empty(n, dtype=np.float32)
    au = np.empty(n, dtype=np.float32)
    for r in range(n):
        qa, qb = Q[pos1[r]], Q[pos2[r]]
        fams = set(qa) & set(qb)
        if not fams:
            uj[r] = uc[r] = ur[r] = -1.0
        else:
            hit = conf = 0
            worst = 0.0
            for f in fams:
                if qa[f] & qb[f]:
                    hit += 1
                else:
                    conf += 1
                    va, vb = max(qa[f]), max(qb[f])
                    if max(va, vb) > 0:
                        worst = max(worst, abs(va - vb) / max(va, vb))
            uj[r], uc[r], ur[r] = hit / len(fams), conf, worst
        ti, tj = tok[pos1[r]], tok[pos2[r]]
        oa, ob = list(ti - tj), list(tj - ti)
        if not oa or not ob or not HAVE_RF:
            am[r] = an[r] = -1.0
            au[r] = len(oa) + len(ob)
        else:
            best = [max(_rf_fuzz.ratio(x, z) for z in ob) / 100.0 for x in oa]
            am[r], an[r] = float(np.mean(best)), float(np.min(best))
            au[r] = float(sum(1 for v in best if v < 0.6))

    # n_components must be < vocabulary size. Real corpora are far larger than
    # 64, but a degenerate items file would otherwise raise here and — before
    # the per-block guards in gbdt_v2 — take the NB features down with it.
    W = tfidf["name_word"]
    k = int(min(64, max(1, min(W.shape) - 1)))
    sv = TruncatedSVD(n_components=k, random_state=SEED, n_iter=4)
    Z = sv.fit_transform(W).astype(np.float32)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    scos = np.einsum("ij,ij->i", Z[pos1], Z[pos2]).astype(np.float32)
    sl2 = np.linalg.norm(Z[pos1] - Z[pos2], axis=1).astype(np.float32)

    out = np.column_stack([ncd, uj, uc, ur, am, an, au, scos, sl2]).astype(np.float32)
    return pd.DataFrame(out, columns=NCD_COLS)


# ------------------------------------------------------- variant conflicts
# Round 12's largest gain (+0.0199), and the round-10 autopsy explains why it
# was missing. Our three worst categories by skill score were Обувь (0.361),
# Одежда (0.379) and Ювелирные изделия (0.386) -- and in the first two the
# model-free lexical ranking scored lift BELOW 1.0, i.e. worse than random.
# Two different sneakers are both "Кроссовки мужские Nike Air Max": near
# identical names, different products. There, similarity is evidence in the
# wrong direction, and the entire rest of the stack is built on similarity.
#
# These features encode CONFLICT instead. A colour that disagrees is positive
# evidence of "different product", which no cosine can express -- and round 9
# showed why it has to be hand-encoded: word vectors place красный and синий
# very close together, because they occur in identical contexts, so a learned
# semantic residual runs backwards.
_LETTER_SIZE = {"xxs": 0, "xs": 1, "s": 2, "m": 3, "l": 4, "xl": 5, "xxl": 6,
                "xxxl": 7, "2xl": 6, "3xl": 7, "4xl": 8, "5xl": 9}
_SIZE_KEYS = ("размер", "длина стельки")
_RANGE_RE = re.compile(r"\b(\d{2})\s*[-/]\s*(\d{2})\b")
_NUMSZ_RE = re.compile(r"\b(\d{2}(?:[.,]5)?)\b")
_LET_RE = re.compile(r"\b(xxxl|xxl|xxs|[2-5]xl|xl|xs|s|m|l)\b", re.I)
_CODE_RE = re.compile(r"\b(?=[a-z0-9-]{4,})(?=[a-z-]*\d)[a-z0-9-]+\b", re.I)
_WORD_RE = re.compile(r"[а-яёa-z]+")
_MAT_KEYS = ("материал", "состав", "металл", "вставка")

_COLOR = {}
for _canon, _alts in {
    "black": ["черный", "чёрный", "black"], "white": ["белый", "white"],
    "red": ["красный", "red"], "blue": ["синий", "blue", "голубой"],
    "green": ["зеленый", "зелёный", "green"], "grey": ["серый", "grey", "gray"],
    "beige": ["бежевый", "beige"], "pink": ["розовый", "pink"],
    "brown": ["коричневый", "brown"], "yellow": ["желтый", "жёлтый", "yellow"],
    "silver": ["серебристый", "серебряный", "silver"],
    "gold": ["золотой", "золотистый", "gold"],
    "purple": ["фиолетовый", "сиреневый", "purple"],
    "orange": ["оранжевый", "orange"],
}.items():
    for _a in _alts:
        _COLOR[_a] = _canon

_GENDER = {"мужской": "m", "мужская": "m", "мужские": "m", "муж": "m",
           "женский": "f", "женская": "f", "женские": "f", "жен": "f",
           "детский": "k", "детская": "k", "детские": "k", "унисекс": "u"}


def build_specialist_features(item_feats, pos1, pos2):
    names, attrs, tok = (item_feats["name"], item_feats["attrs"],
                         item_feats["name_tok"])
    n_items = len(names)

    sizes, axes, codes = [], [], []
    for i in range(n_items):
        a = attrs[i]
        txt = names[i].lower()
        for k, v in a.items():
            if any(s in k for s in _SIZE_KEYS):
                txt += " " + str(v).lower()
        nums = set()
        for lo, hi in _RANGE_RE.findall(txt):
            nums.update({float(lo), float(hi)})
        for v in _NUMSZ_RE.findall(txt):
            f = float(v.replace(",", "."))
            if 13 <= f <= 70:            # plausible clothing / shoe range
                nums.add(f)
        lets = {_LETTER_SIZE[x.lower()] for x in _LET_RE.findall(txt)
                if x.lower() in _LETTER_SIZE}
        sizes.append((nums, lets))

        blob = " ".join(str(v).lower() for k, v in a.items()
                        if "цвет" in k or k.startswith("пол"))
        words = set(tok[i]) | set(_WORD_RE.findall(blob))
        mat = set()
        for k, v in a.items():
            if any(s in k for s in _MAT_KEYS):
                mat.update(w for w in _WORD_RE.findall(str(v).lower())
                           if len(w) >= 4)
        axes.append(({_COLOR[w] for w in words if w in _COLOR},
                     {_GENDER[w] for w in words if w in _GENDER}, mat))
        codes.append(set(_CODE_RE.findall(names[i].lower())))

    n = len(pos1)
    out = {k: np.full(n, -1.0, dtype=np.float32) for k in SPECIALIST_COLS}
    for r in range(n):
        i, j = pos1[r], pos2[r]
        na, la = sizes[i]
        nb, lb = sizes[j]
        out["sz_present"][r] = float(bool(na or la) + bool(nb or lb))
        if na and nb:
            out["sz_num_eq"][r] = 1.0 if na & nb else 0.0
            out["sz_num_conflict"][r] = 0.0 if na & nb else 1.0
            out["sz_gap"][r] = float(min(abs(x - z) for x in na for z in nb))
        if la and lb:
            out["sz_let_eq"][r] = 1.0 if la & lb else 0.0
            out["sz_let_conflict"][r] = 0.0 if la & lb else 1.0
            if out["sz_gap"][r] < 0:
                out["sz_gap"][r] = float(min(abs(x - z) for x in la for z in lb))

        ca, cb = codes[i], codes[j]
        sh = ca & cb
        out["code_shared_max_len"][r] = max((len(x) for x in sh), default=0)
        out["code_asym"][r] = float(bool(ca) != bool(cb))
        out["code_n_diff"][r] = len(ca ^ cb)
        if ca and cb and HAVE_RF:
            out["code_best_ratio"][r] = max(
                _rf_fuzz.ratio(x, z) for x in ca for z in cb) / 100.0

        (cola, gena, mata), (colb, genb, matb) = axes[i], axes[j]
        if cola and colb:
            out["col_shared"][r] = float(bool(cola & colb))
            out["col_conflict"][r] = 0.0 if cola & colb else 1.0
        if gena and genb:
            out["gen_conflict"][r] = 0.0 if gena & genb else 1.0
        if mata and matb:
            u = len(mata | matb)
            out["mat_jac"][r] = len(mata & matb) / u if u else -1.0
        out["axis_any"][r] = float(max(out["col_conflict"][r], 0.0)
                                   + max(out["gen_conflict"][r], 0.0)
                                   + max(out["sz_num_conflict"][r], 0.0)
                                   + max(out["sz_let_conflict"][r], 0.0))
    return pd.DataFrame(out, columns=SPECIALIST_COLS)


# -------------------------------------------------- learned per-category keys
# Round 15, +0.0034 on top of the hand-coded conflicts, and it fixed the one
# category hand-coding could not: Ювелирные изделия gained only +0.005 from
# colour/gender/size because its discriminating attributes are проба and
# вставка -- gold fineness and gemstone. This finds them without knowing what
# they are. Обувь +0.013, Ювелирные +0.013.
#
# TOP_KEYS IS FIXED AT TRAINING TIME AND SHIPPED IN meta.json. Recomputing it
# from the container's items file would silently change what column keyconf_7
# means between training and inference.
N_TOP_KEYS = 12
KEYCONF_COLS = [f"keyconf_{i}" for i in range(N_TOP_KEYS)]
NB_KEY_COLS = [f"nbkey_{s}" for s in NB_STATS]
_KEYNUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def compute_top_keys(item_feats, n_top=N_TOP_KEYS):
    """Most common attribute keys per category. Training-time only."""
    per = {}
    for a, c in zip(item_feats["attrs"], item_feats["category"]):
        d = per.setdefault(c, Counter())
        d.update(a.keys())
    return {c: [k for k, _ in d.most_common(n_top)] for c, d in per.items()}


def _keynum(v):
    mt = _KEYNUM_RE.search(v)
    return float(mt.group(0).replace(",", ".")) if mt else None


def build_key_conflict_features(item_feats, pos1, pos2, top_keys):
    """1.0 values agree · 0.5 numeric within 2% · 0.0 conflict · -1 no compare."""
    attrs, cats = item_feats["attrs"], item_feats["category"]
    n = len(pos1)
    Z = np.full((n, N_TOP_KEYS), -1.0, dtype=np.float32)
    for r in range(n):
        i, j = pos1[r], pos2[r]
        ai, aj = attrs[i], attrs[j]
        for s, k in enumerate(top_keys.get(cats[i], ())):
            va, vb = ai.get(k), aj.get(k)
            if va is None or vb is None:
                continue
            if va == vb:
                Z[r, s] = 1.0
                continue
            na, nb = _keynum(va), _keynum(vb)
            if na is not None and nb is not None:
                hi = max(abs(na), abs(nb), 1e-6)
                Z[r, s] = 0.5 if abs(na - nb) / hi < 0.02 else 0.0
            else:
                Z[r, s] = 0.0
    return pd.DataFrame(Z, columns=KEYCONF_COLS)


def key_bags(item_feats, pos1, pos2):
    """Key-level agreement bags, CATEGORY-PREFIXED.

    The existing attribute table keys on key=value strings, so it only scores
    value pairs it has literally seen. This keys on the KEY alone -- D: differs,
    S: same, M: present on one side only -- which generalises to unseen values,
    and most values are unseen. The category prefix gives per-category
    conditioning through a single shipped table instead of twenty.
    """
    attrs, cats = item_feats["attrs"], item_feats["category"]
    out = []
    for r in range(len(pos1)):
        i, j = pos1[r], pos2[r]
        ai, aj = attrs[i], attrs[j]
        c = cats[i]
        b = {f"{c}\x01" + ("S:" if ai[k] == aj[k] else "D:") + k
             for k in set(ai) & set(aj)}
        b |= {f"{c}\x01M:{k}" for k in set(ai) ^ set(aj)}
        out.append(b)
    return out


# ---------------------------------------------------------------- NB tables
def pair_bags(item_feats, pos1, pos2):
    """Symmetric-difference and intersection bags the log-odds tables key on."""
    tok, attrs = item_feats["name_tok"], item_feats["attrs"]
    n = len(pos1)
    name_d, name_s, attr_d = [], [], []
    for r in range(n):
        ti, tj = tok[pos1[r]], tok[pos2[r]]
        name_d.append(ti ^ tj)
        name_s.append(ti & tj)
        ai, aj = attrs[pos1[r]], attrs[pos2[r]]
        attr_d.append({f"{k}={ai[k]}" for k in ai if aj.get(k) != ai[k]} |
                      {f"{k}={aj[k]}" for k in aj if ai.get(k) != aj[k]})
    return name_d, name_s, attr_d


def fit_nb(rows, y, bag_d, bag_s, min_count=5):
    """Log-odds that a token appearing in the symmetric difference (or the
    intersection, prefixed '~') belongs to a duplicate pair."""
    pos, neg = Counter(), Counter()
    for i in rows:
        c = pos if y[i] else neg
        c.update(bag_d[i])
        if bag_s is not None:
            c.update("~" + t for t in bag_s[i])
    npos = int(np.sum(y[rows]))
    nneg = len(rows) - npos
    lp, ln = np.log(npos + 2.0), np.log(nneg + 2.0)
    return {k: float((np.log(pos.get(k, 0) + 1.0) - lp)
                     - (np.log(neg.get(k, 0) + 1.0) - ln))
            for k in set(pos) | set(neg)
            if pos.get(k, 0) + neg.get(k, 0) >= min_count}


def apply_nb(rows, W, bag_d, bag_s, out):
    for i in rows:
        d = [W[x] for x in bag_d[i] if x in W]
        s = [W["~" + x] for x in bag_s[i] if "~" + x in W] if bag_s is not None else []
        out[i, 0] = sum(d) if d else 0.0
        out[i, 1] = float(np.mean(d)) if d else 0.0
        out[i, 2] = max(d) if d else 0.0
        out[i, 3] = min(d) if d else 0.0
        out[i, 4] = len(d)
        out[i, 5] = sum(s) if s else 0.0
        out[i, 6] = float(np.mean(s)) if s else 0.0
        out[i, 7] = max(s) if s else 0.0
        out[i, 8] = min(s) if s else 0.0
        out[i, 9] = out[i, 1] - out[i, 6]


def build_nb_features(bags, w_name, w_attr):
    """Inference path: pure table lookup, no labels involved."""
    name_d, name_s, attr_d = bags
    n = len(name_d)
    a = np.zeros((n, len(NB_STATS)), dtype=np.float32)
    b = np.zeros((n, len(NB_STATS)), dtype=np.float32)
    rows = np.arange(n)
    apply_nb(rows, w_name, name_d, name_s, a)
    apply_nb(rows, w_attr, attr_d, None, b)
    return pd.DataFrame(np.column_stack([a, b]),
                        columns=NB_NAME_COLS + NB_ATTR_COLS)


# ---------------------------------------------------------------- rank norm
def rank_normalise(df, cols=None):
    """Map corpus-fitted columns to their within-run rank in [0, 1].

    Trees are invariant to monotone transforms of an individual feature, so on
    a single run this is a no-op for accuracy. Across runs it is the whole
    point: a threshold like `idf_miss_a > 7.3` means something different when
    IDF is refit on a differently sized corpus, and a rank does not.
    """
    cols = CORPUS_COLS if cols is None else cols
    n = len(df)
    if n == 0:
        return df
    for c in cols:
        if c in df.columns:
            v = np.nan_to_num(df[c].to_numpy(dtype=np.float64), nan=-1.0,
                              posinf=0.0, neginf=-1.0)
            df[c] = (rankdata(v) / n).astype(np.float32)
    return df


def save_nb_table(path, W):
    """Keys as one UTF-8 blob plus a length array.

    NOT a numpy '<U' array: that is fixed-width UTF-32, so 122K attribute keys
    whose longest member is a few hundred characters inflate to ~240MB and take
    28 seconds to load. Measured in the container smoke test. An explicit
    offset table is exact for any key content, including newlines and nulls.
    """
    ks = list(W.keys())
    parts = [k.encode("utf-8") for k in ks]
    lens = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
    blob = np.frombuffer(b"".join(parts), dtype=np.uint8)
    vals = np.array([W[k] for k in ks], dtype=np.float32)
    np.savez_compressed(path, blob=blob, lens=lens, vals=vals)


def load_nb_table(path):
    z = np.load(path, allow_pickle=False)
    blob = z["blob"].tobytes()
    offs = np.concatenate([[0], np.cumsum(z["lens"])]).astype(np.int64)
    vals = z["vals"].tolist()
    return {blob[offs[i]:offs[i + 1]].decode("utf-8"): vals[i]
            for i in range(len(vals))}
