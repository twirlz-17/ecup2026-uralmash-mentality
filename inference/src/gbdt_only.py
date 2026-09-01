"""
GBDT scoring with no torch dependency.

Same feature pipeline and models as the blend's GBDT half, lifted out of
utils.py so the module graph never touches torch/transformers. Keeping it
separate rather than adding a flag to utils.py means an accidental import
cannot drag CUDA initialisation back in.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _lexical_score(X):
    """Model-free ranking from the strongest standalone features.

    Rank-averaged so the different scales combine sanely. Measured standalone
    macro PR-AUC on the SEED=42 holdout: tfidf_name_word 0.374,
    name_tok_cont 0.366, tfidf_name_char 0.355, name_chrf 0.348, num_jac 0.345.
    Far from the 0.63 model, but ~3x a constant prediction.
    """
    from scipy.stats import rankdata

    picks = ["tfidf_name_word", "name_tok_cont", "tfidf_name_char",
             "name_chrf", "num_jac"]
    cols = [c for c in picks if c in X.columns]
    if not cols:
        return np.full(len(X), 0.5)
    n = len(X)
    return np.mean([rankdata(np.nan_to_num(X[c].to_numpy(), nan=-1.0)) / n
                    for c in cols], axis=0)


def score_pairs(items_path: str, matches_path: str, gbdt_dir: str, log=print,
                lgb=None):
    if lgb is None:
        import lightgbm as lgb  # noqa: F811

    from .features import (HAVE_RAPIDFUZZ, build_item_features,
                           build_pair_features, build_tfidf)

    if not HAVE_RAPIDFUZZ:
        raise RuntimeError("rapidfuzz unavailable — 6 features would be missing")

    with open(os.path.join(gbdt_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    df = pq.read_table(items_path,
                       columns=["id", "name", "attributes", "category"]).to_pandas()
    log(f"items: {len(df)}")

    feats = build_item_features(df)
    log("item features built")
    tfidf = build_tfidf(feats)
    log("tfidf built")

    m = pd.read_parquet(matches_path, columns=["id1", "id2"])
    index = pd.Index(feats["id"])
    if not index.is_unique:
        keep = ~index.duplicated()
        for k in ("id", "category", "name", "name_norm", "name_tok", "name_c3",
                  "num", "code", "attrs", "brand", "attr_text", "attr_num"):
            v = feats[k]
            feats[k] = v[keep] if isinstance(v, np.ndarray) else \
                [x for x, s in zip(v, keep) if s]
        index = pd.Index(feats["id"])
        log(f"dropped {(~keep).sum()} duplicate item ids")

    pos1 = index.get_indexer(m["id1"].to_numpy())
    pos2 = index.get_indexer(m["id2"].to_numpy())
    miss = (pos1 < 0) | (pos2 < 0)
    if miss.any():
        log(f"WARNING: {int(miss.sum())} pairs have an id missing from items")
    pos1 = np.where(pos1 < 0, 0, pos1)
    pos2 = np.where(pos2 < 0, 0, pos2)

    X = build_pair_features(feats, tfidf, pos1, pos2)[meta["features"]]
    Xv = X.to_numpy(dtype=np.float32)
    log(f"pair features {Xv.shape}")

    # Category encoding must be EXACT-MATCH, never searchsorted.
    # searchsorted maps an unseen or slightly-different string to whatever
    # neighbour it sorts next to, silently feeding the model a wrong value for
    # what is its second-highest-gain feature. Unknown -> -1, which LightGBM
    # routes down the categorical default branch instead of a wrong one.
    cats = list(meta["categories"])
    cat_to_code = {c: i for i, c in enumerate(cats)}
    cat = feats["category"][pos1]
    code = np.array([cat_to_code.get(c, -1) for c in cat], dtype=np.float32)
    unknown = int((code < 0).sum())
    if unknown:
        log(f"WARNING: {unknown}/{len(code)} pairs have a category absent from "
            f"training ({sorted(set(cat[code < 0]))[:5]})")
    Xa = np.column_stack([Xv, code])

    # num_threads=0 means "one thread per visible core". Under a cgroup CPU
    # quota the visible count is the HOST's, so that oversubscribes badly and
    # hangs. Pin it instead — 275K rows through 1715 trees took 39s on a 4-core
    # box, so a small pool is ample.
    nt = int(os.environ.get("N_THREADS", "4"))
    try:
        pred_a = lgb.Booster(model_file=os.path.join(gbdt_dir, "model_a.txt")).predict(
            Xa, num_threads=nt)
        log(f"model A scored (num_threads={nt})")
        # Start from A so any row the per-category models do not cover keeps a
        # real prediction. Previously uncovered rows stayed 0.0, which silently
        # halved the blend into 0.5*pred_a for them.
        pred_b = pred_a.copy()
        covered = np.zeros(len(Xv), dtype=bool)
        for ci, c in enumerate(cats):
            k = cat == c
            if not k.any():
                continue
            path = os.path.join(gbdt_dir, f"model_b_{ci}.txt")
            if os.path.exists(path):
                pred_b[k] = lgb.Booster(model_file=path).predict(Xv[k], num_threads=nt)
                covered |= k
        frac = float(covered.mean())
        log(f"model B covered {frac:.3f} of rows")
        if frac < 0.5:
            # category strings did not line up; the B half is not trustworthy
            log("B coverage too low — using A alone")
            out = pred_a
        else:
            out = 0.5 * pred_a + 0.5 * pred_b
    except Exception as exc:
        # A previous submission scored 0.1113 — exactly the positive rate —
        # because scoring failed and a CONSTANT fallback was written. Constant
        # predictions are worth literally nothing. The features are already
        # computed at this point, so degrade to the strongest single ones
        # instead: tfidf_name_word alone is 0.374 standalone on the holdout,
        # name_tok_cont 0.366, num_jac 0.345.
        log(f"MODEL LOAD/PREDICT FAILED ({exc!r}); falling back to lexical score")
        out = _lexical_score(X)
    # pairs with an unmappable id got row-0 features; a neutral score is more
    # honest than a confident one derived from the wrong item
    out[miss] = float(np.median(out))
    log("scored")
    return out
