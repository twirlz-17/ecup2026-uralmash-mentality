"""
E_CUP 2026 Matching — lexical / structural pair features for a GBDT.

Rationale: matches_llm and items_human cover disjoint product populations, so
the 365K human pairs are the only in-distribution training signal. Features
computed directly on those pairs sidestep the cross-population transfer problem
the cross-encoder is fighting, and cost seconds instead of minutes.

Per-item work is done ONCE per item (not once per pair) and reused; per-pair
work is either vectorised (rapidfuzz cpdist / sparse TF-IDF row dots) or a tight
Python loop over precomputed sets.

Feature families:
  name_*   fuzzy + token + char-ngram similarity on the product name
  tfidf_*  IDF-weighted cosine (word and char), corpus = the items file itself
  num_*    numeric tokens (sizes, volumes, wattage) — highly discriminative
  code_*   SKU / model / article tokens (letters+digits)
  brand_*  brand agreement from the parsed attributes
  attr_*   structured per-key comparison of the attributes JSON
  meta_*   lengths, token counts
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rapidfuzz import fuzz, process as rf_process
    from rapidfuzz.distance import JaroWinkler

    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - prototype fallback
    HAVE_RAPIDFUZZ = False

_WORD = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁ]+")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_PUNCT = re.compile(r"[^0-9a-zA-Zа-яёА-ЯЁ]+")
_PUNCT_KEEP_DOT = re.compile(r"[^0-9a-zA-Zа-яёА-ЯЁ.]+")
_WS = re.compile(r"\s+")

BRAND_KEYS = ("бренд", "производитель", "торговая марка", "brand", "manufacturer")


# ---------------------------------------------------------------- per item
def _norm(s: str) -> str:
    return _PUNCT.sub(" ", str(s).lower()).strip()


def _tokens(s: str):
    return _WORD.findall(s.lower())


def _char_ngrams(s: str, n: int = 3):
    s = " " + _PUNCT.sub(" ", s.lower()).strip() + " "
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _numbers(s: str):
    out = set()
    for tok in _NUM.findall(s):
        tok = tok.replace(",", ".")
        try:
            v = float(tok)
        except ValueError:
            continue
        out.add(round(v, 4))
    return out


def _codes(tokens):
    """Alphanumeric model/SKU-ish tokens: contain a digit AND a letter, len>=4."""
    out = set()
    for t in tokens:
        if len(t) >= 4 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            out.add(t)
    return out


def _parse_attrs(s):
    try:
        d = json.loads(s)
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        out[str(k).strip().lower()] = str(v).strip().lower()
    return out


def _norm_val(v: str) -> str:
    """Unit/separator-insensitive form: '15,6 ГБ' and '15.6 гб' collapse together."""
    v = v.replace(",", ".").replace(" ", " ")
    v = _PUNCT_KEEP_DOT.sub(" ", v)
    return _WS.sub(" ", v).strip()


_NORM_CACHE: dict = {}


def _norm_pair(v: str):
    """(normalised string, frozenset of numbers), memoised.

    Deliberately called from the PAIR loop for shared keys only, not per item:
    normalising every value on every item costs ~3x more, and attribute values
    repeat heavily across products so the cache hit rate is high.
    """
    hit = _NORM_CACHE.get(v)
    if hit is None:
        if len(_NORM_CACHE) > 2_000_000:
            _NORM_CACHE.clear()
        hit = _NORM_CACHE[v] = (_norm_val(v), frozenset(_numbers(v)))
    return hit


def _brand(attrs: dict):
    for k in BRAND_KEYS:
        if k in attrs and attrs[k]:
            return attrs[k]
    for k, v in attrs.items():
        if any(b in k for b in BRAND_KEYS) and v:
            return v
    return ""


def build_item_features(df: pd.DataFrame):
    """df: id, name, attributes, category -> dict of per-item arrays/lists."""
    names = df["name"].astype(str).tolist()
    attrs_raw = df["attributes"].astype(str).tolist()

    name_norm, name_tok, name_c3, nums, codes = [], [], [], [], []
    for nm in names:
        t = _tokens(nm)
        name_norm.append(_norm(nm))
        name_tok.append(set(t))
        name_c3.append(_char_ngrams(nm))
        nums.append(_numbers(nm))
        codes.append(_codes(t))

    attrs = [_parse_attrs(s) for s in attrs_raw]
    brands = [_brand(a) for a in attrs]
    attr_text = [" ".join(f"{k} {v}" for k, v in a.items()) for a in attrs]
    attr_nums = [_numbers(t) for t in attr_text]

    return {
        "id": df["id"].to_numpy(),
        "category": df["category"].astype(str).to_numpy(),
        "name": names,
        "name_norm": name_norm,
        "name_tok": name_tok,
        "name_c3": name_c3,
        "num": nums,
        "code": codes,
        "attrs": attrs,
        "brand": brands,
        "attr_text": attr_text,
        "attr_num": attr_nums,
    }


# ---------------------------------------------------------------- tfidf
def build_tfidf(item_feats):
    """IDF is fitted on the items file being scored — no external corpus needed."""
    out = {}
    vw = TfidfVectorizer(analyzer="word", token_pattern=r"[0-9a-zA-Zа-яёА-ЯЁ]+",
                         min_df=1, sublinear_tf=True)
    out["name_word"] = vw.fit_transform(item_feats["name_norm"])
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                         sublinear_tf=True, max_features=300_000)
    out["name_char"] = vc.fit_transform(item_feats["name_norm"])
    va = TfidfVectorizer(analyzer="word", token_pattern=r"[0-9a-zA-Zа-яёА-ЯЁ]+",
                         min_df=2, sublinear_tf=True, max_features=300_000)
    out["attr_word"] = va.fit_transform(item_feats["attr_text"])
    # keys prefixed with "_" are side-channel data, not pair matrices
    out["_idf"] = dict(zip(vw.get_feature_names_out(), vw.idf_.astype(np.float32)))
    return out


def _row_cosine(X, i1, i2, chunk=200_000):
    """Elementwise cosine between paired rows of an L2-normalised CSR matrix."""
    out = np.empty(len(i1), dtype=np.float32)
    for s in range(0, len(i1), chunk):
        e = min(s + chunk, len(i1))
        a, b = X[i1[s:e]], X[i2[s:e]]
        out[s:e] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


# ---------------------------------------------------------------- per pair
def _jac(a: set, b: set):
    if not a and not b:
        return -1.0
    u = len(a | b)
    return len(a & b) / u if u else -1.0


def _cont(a: set, b: set):
    m = min(len(a), len(b))
    return len(a & b) / m if m else -1.0


def _chrf(a: set, b: set, beta: float = 2.0):
    """chrF-style F-beta over char n-gram sets (recall weighted, as in chrF++)."""
    if not a or not b:
        return -1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _fuzzy_block(s1, s2):
    """Vectorised rapidfuzz scorers; falls back to a Python loop if unavailable.

    workers is PINNED, never -1. `-1` means "one worker per core rapidfuzz can
    see", and inside a container that is the HOST's core count — which can be
    dozens — while the cgroup quota is 20. The resulting pool thrashes against
    the quota and the run never finishes; the beefier the node, the worse it
    gets. Every submission carrying rapidfuzz timed out, and every one without
    it completed, including a probe running this exact feature code minus the
    five fuzzy columns.
    """
    workers = int(os.environ.get("RF_WORKERS", "4"))
    scorers = [("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio),
               ("tok_sort", fuzz.token_sort_ratio), ("tok_set", fuzz.token_set_ratio),
               ("jw", JaroWinkler.normalized_similarity)]
    out = {}
    for tag, sc in scorers:
        try:
            v = rf_process.cpdist(s1, s2, scorer=sc, workers=workers)
            out[f"name_{tag}"] = np.asarray(v, dtype=np.float32)
        except (AttributeError, TypeError):
            out[f"name_{tag}"] = np.fromiter(
                (sc(a, b) for a, b in zip(s1, s2)), dtype=np.float32, count=len(s1))
    return out


def build_pair_features(item_feats, tfidf, pos1: np.ndarray, pos2: np.ndarray):
    """pos1/pos2: row positions into item_feats arrays. Returns a DataFrame."""
    n = len(pos1)
    f = {}

    nn = item_feats["name_norm"]
    s1 = [nn[i] for i in pos1]
    s2 = [nn[i] for i in pos2]

    if HAVE_RAPIDFUZZ:
        f.update(_fuzzy_block(s1, s2))

    for key, mat in tfidf.items():
        if key.startswith("_"):
            continue
        f[f"tfidf_{key}"] = _row_cosine(mat, pos1, pos2)

    tok, c3, num, code = (item_feats["name_tok"], item_feats["name_c3"],
                          item_feats["num"], item_feats["code"])
    attrs, brand, anum = item_feats["attrs"], item_feats["brand"], item_feats["attr_num"]

    cols = {k: np.empty(n, dtype=np.float32) for k in (
        "name_tok_jac", "name_tok_cont", "name_c3_jac", "name_chrf",
        "num_jac", "num_cont", "num_all_a_in_b", "num_eq", "num_n1", "num_n2",
        "anum_jac", "anum_cont",
        "code_jac", "code_any", "code_n1", "code_n2",
        "brand_eq", "brand_missing",
        "attr_key_jac", "attr_kv_exact", "attr_kv_ratio", "attr_val_sim",
        "attr_kv_norm_ratio", "attr_kv_num_ratio", "attr_kv_conflict",
        "attr_n1", "attr_n2", "attr_shared",
        "meta_len_ratio", "meta_len_diff", "meta_tok1", "meta_tok2", "meta_first_tok",
        # asymmetric IDF mismatch: duplicates differ in NOISE words, non-duplicates
        # differ in INFORMATIVE ones. Cosine similarity averages that distinction
        # away — these keep the two directions and the rarity separate.
        "idf_miss_a", "idf_miss_b", "idf_miss_max", "idf_miss_min",
        "idf_miss_ratio", "idf_shared", "idf_shared_ratio",
        # difference-typing: WHAT kind of token differs, not how many
        "diff_num", "diff_code", "diff_word", "diff_num_ratio",
    )}

    idf = tfidf.get("_idf", {})
    idf_default = float(np.median(list(idf.values()))) if idf else 1.0

    def _idf_of(tok):
        return idf.get(tok, idf_default)

    for r in range(n):
        i, j = pos1[r], pos2[r]
        ti, tj = tok[i], tok[j]
        cols["name_tok_jac"][r] = _jac(ti, tj)
        cols["name_tok_cont"][r] = _cont(ti, tj)
        cols["name_c3_jac"][r] = _jac(c3[i], c3[j])
        cols["name_chrf"][r] = _chrf(c3[i], c3[j])

        only_a, only_b, both_ = ti - tj, tj - ti, ti & tj
        ia = sum(_idf_of(t) for t in only_a)
        ib = sum(_idf_of(t) for t in only_b)
        ish = sum(_idf_of(t) for t in both_)
        cols["idf_miss_a"][r] = ia
        cols["idf_miss_b"][r] = ib
        cols["idf_miss_max"][r] = max(ia, ib)
        cols["idf_miss_min"][r] = min(ia, ib)
        cols["idf_shared"][r] = ish
        tot = ia + ib + ish
        cols["idf_miss_ratio"][r] = (ia + ib) / tot if tot else -1.0
        cols["idf_shared_ratio"][r] = ish / tot if tot else -1.0

        d_num = d_code = d_word = 0
        for t in only_a | only_b:
            if t.isdigit():
                d_num += 1
            elif any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
                d_code += 1
            else:
                d_word += 1
        nd = d_num + d_code + d_word
        cols["diff_num"][r] = d_num
        cols["diff_code"][r] = d_code
        cols["diff_word"][r] = d_word
        cols["diff_num_ratio"][r] = (d_num + d_code) / nd if nd else -1.0

        ni, nj = num[i], num[j]
        cols["num_jac"][r] = _jac(ni, nj)
        cols["num_cont"][r] = _cont(ni, nj)
        cols["num_all_a_in_b"][r] = 1.0 if (ni and ni <= nj) else 0.0
        cols["num_eq"][r] = 1.0 if (ni and ni == nj) else 0.0
        cols["num_n1"][r], cols["num_n2"][r] = len(ni), len(nj)

        cols["anum_jac"][r] = _jac(anum[i], anum[j])
        cols["anum_cont"][r] = _cont(anum[i], anum[j])

        ci, cj = code[i], code[j]
        cols["code_jac"][r] = _jac(ci, cj)
        cols["code_any"][r] = 1.0 if (ci & cj) else 0.0
        cols["code_n1"][r], cols["code_n2"][r] = len(ci), len(cj)

        bi, bj = brand[i], brand[j]
        cols["brand_missing"][r] = 1.0 if (not bi or not bj) else 0.0
        cols["brand_eq"][r] = 1.0 if (bi and bj and bi == bj) else 0.0

        ai, aj = attrs[i], attrs[j]
        ka, kb = set(ai), set(aj)
        shared = ka & kb
        cols["attr_key_jac"][r] = _jac(ka, kb)
        cols["attr_n1"][r], cols["attr_n2"][r] = len(ka), len(kb)
        cols["attr_shared"][r] = len(shared)
        if shared:
            exact = sum(1 for k in shared if ai[k] == aj[k])
            cols["attr_kv_exact"][r] = exact
            cols["attr_kv_ratio"][r] = exact / len(shared)
            norm_hit = num_hit = num_cmp = conflict = 0
            for k in shared:
                va_, na_ = _norm_pair(ai[k])
                vb_, nb_ = _norm_pair(aj[k])
                if va_ == vb_:
                    norm_hit += 1
                if na_ or nb_:
                    num_cmp += 1
                    if na_ == nb_:
                        num_hit += 1
                    elif na_ and nb_:
                        conflict += 1  # both sides state a number and they disagree
            cols["attr_kv_norm_ratio"][r] = norm_hit / len(shared)
            cols["attr_kv_num_ratio"][r] = num_hit / num_cmp if num_cmp else -1.0
            cols["attr_kv_conflict"][r] = conflict
            if HAVE_RAPIDFUZZ:
                cols["attr_val_sim"][r] = float(np.mean(
                    [fuzz.ratio(ai[k], aj[k]) for k in shared]))
            else:
                cols["attr_val_sim"][r] = -1.0
        else:
            cols["attr_kv_exact"][r] = 0.0
            cols["attr_kv_ratio"][r] = -1.0
            cols["attr_val_sim"][r] = -1.0
            cols["attr_kv_norm_ratio"][r] = -1.0
            cols["attr_kv_num_ratio"][r] = -1.0
            cols["attr_kv_conflict"][r] = -1.0

        la, lb = len(s1[r]), len(s2[r])
        cols["meta_len_ratio"][r] = min(la, lb) / max(la, lb) if max(la, lb) else -1.0
        cols["meta_len_diff"][r] = abs(la - lb)
        cols["meta_tok1"][r], cols["meta_tok2"][r] = len(ti), len(tj)
        cols["meta_first_tok"][r] = 1.0 if (
            s1[r][:12] and s1[r].split(" ")[0] == s2[r].split(" ")[0]) else 0.0

    f.update(cols)
    return pd.DataFrame(f)
