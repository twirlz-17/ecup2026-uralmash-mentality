"""
Round-5 scorer. Torch-free, and every feature comes from the same two modules
the training kernel imported, so train/inference parity is structural rather
than something to remember.

Pipeline, with measured cost on 711K items / 366K pairs at 4 cores:
    item features + tf-idf        ~180s
    canonical pair ordering        free (a permutation of the inputs)
    base pair features             ~50s
    crowdedness (word MinHash)     ~29s
    NCD / units / alignment / LSA  ~61s
    NB table lookup                ~25s
    rank-normalise corpus columns  ~2s
Roughly 350s at 4 cores; the grader gives 20, and the parallel stages (the
rapidfuzz block, tf-idf, LightGBM) scale with them.

Failure posture, learned the hard way: a previous submission scored 0.1113 —
exactly the test prevalence — because scoring raised and a CONSTANT fallback
was written. Constants are worth nothing. Every stage here degrades to the best
thing still computable instead: no NB tables -> run without them is impossible
(the models expect the columns), so the columns go to zero, which is the value
a token absent from the table already produces. Total model failure -> the
lexical rank-average, which scored 0.2586 on the leaderboard against 0.1113 for
a constant.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _lexical_score(X):
    """Model-free ranking from the strongest standalone features.

    Rank-averaged so the different scales combine sanely. Standalone macro
    PR-AUC on the holdout: tfidf_name_word 0.374, name_tok_cont 0.366,
    tfidf_name_char 0.355, name_chrf 0.348, num_jac 0.345. Measured 0.2586 on
    the leaderboard — far below the model, ~2.3x a constant.
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
    import time

    from scipy.stats import rankdata

    if lgb is None:
        import lightgbm as lgb  # noqa: F811

    from . import features2 as F2
    from .features import (HAVE_RAPIDFUZZ, build_item_features,
                           build_pair_features, build_tfidf)

    if not HAVE_RAPIDFUZZ:
        raise RuntimeError("rapidfuzz unavailable - 6 features would be missing")

    t0 = time.time()

    def stage(name):
        log(f"  [{time.time() - t0:6.1f}s] {name}")

    with open(os.path.join(gbdt_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    df = pq.read_table(items_path,
                       columns=["id", "name", "attributes", "category"]).to_pandas()
    log(f"items: {len(df)}")
    feats = build_item_features(df)
    stage("item features")
    tfidf = build_tfidf(feats)
    stage("tfidf")

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
        log(f"dropped {int((~keep).sum())} duplicate item ids")

    p1 = index.get_indexer(m["id1"].to_numpy())
    p2 = index.get_indexer(m["id2"].to_numpy())
    miss = (p1 < 0) | (p2 < 0)
    if miss.any():
        log(f"WARNING: {int(miss.sum())} pairs have an id missing from items")
    p1 = np.where(p1 < 0, 0, p1)
    p2 = np.where(p2 < 0, 0, p2)
    pos1, pos2 = F2.canonical_order(feats, p1, p2)

    Xb = build_pair_features(feats, tfidf, pos1, pos2)
    stage(f"base pair features {Xb.shape}")

    # Each block is guarded SEPARATELY. One shared try/except meant a hiccup in
    # the 0.0056-worth LSA block also took out the 0.0405-worth NB block and
    # dropped the whole run to the lexical fallback — a smoke test on a
    # degenerate items file caught exactly that. A block that fails leaves its
    # columns absent; they are then filled with 0.0 below, which costs that
    # block's contribution and nothing else.
    parts = [Xb.reset_index(drop=True)]

    def block(name, fn):
        try:
            parts.append(fn().reset_index(drop=True))
            stage(name)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            log(f"BLOCK '{name}' FAILED ({exc!r}); its columns will be zero-filled")

    block("crowdedness",
          lambda: F2.build_crowd_features(
              feats, F2.build_corpus_state(feats), pos1, pos2))
    block("ncd / units / alignment / lsa",
          lambda: F2.build_ncd_units_lsa(feats, tfidf, pos1, pos2))
    block("variant conflicts",
          lambda: F2.build_specialist_features(feats, pos1, pos2))

    def _nb():
        w_name = F2.load_nb_table(os.path.join(gbdt_dir, "nb_name.npz"))
        w_attr = F2.load_nb_table(os.path.join(gbdt_dir, "nb_attr.npz"))
        log(f"nb tables: {len(w_name)} name entries, {len(w_attr)} attr entries")
        return F2.build_nb_features(F2.pair_bags(feats, pos1, pos2),
                                    w_name, w_attr)

    block("nb log-odds", _nb)

    # round 15: top_keys comes from meta, never recomputed here -- deriving it
    # from the container's items file would change what keyconf_7 means
    top_keys = meta.get("top_keys")
    if top_keys:
        block("per-category key conflicts",
              lambda: F2.build_key_conflict_features(feats, pos1, pos2, top_keys))

        def _keylo():
            w = F2.load_nb_table(os.path.join(gbdt_dir, "nb_key.npz"))
            log(f"key log-odds table: {len(w)} entries")
            bags = F2.key_bags(feats, pos1, pos2)
            out = np.zeros((len(pos1), len(F2.NB_STATS)), dtype=np.float32)
            F2.apply_nb(np.arange(len(pos1)), w, bags, None, out)
            return pd.DataFrame(out, columns=F2.NB_KEY_COLS)

        if os.path.exists(os.path.join(gbdt_dir, "nb_key.npz")):
            block("key log-odds", _keylo)

    X = pd.concat(parts, axis=1)
    for c in meta["features"]:
        if c not in X.columns:
            log(f"WARNING: feature {c} missing, filling 0")
            X[c] = np.float32(0.0)
    X = X[meta["features"]]
    F2.rank_normalise(X, meta.get("corpus_cols"))
    stage(f"assembled {X.shape}")
    Xv = X.to_numpy(dtype=np.float32)

    # EXACT-MATCH category encoding, never searchsorted: searchsorted maps an
    # unseen string to whatever it sorts next to and silently feeds the model a
    # wrong value for a high-gain feature. Unknown -> -1 takes LightGBM's
    # categorical default branch instead.
    cats = list(meta["categories"])
    c2i = {c: i for i, c in enumerate(cats)}
    cat = feats["category"][pos1]
    code = np.array([c2i.get(c, -1) for c in cat], dtype=np.float32)
    unknown = int((code < 0).sum())
    if unknown:
        log(f"WARNING: {unknown}/{len(code)} pairs have an unseen category")
    Xa = np.column_stack([Xv, code])

    # num_threads=0 means one thread per VISIBLE core, which inside a container
    # is the HOST's count, not the cgroup quota. That oversubscription is what
    # hung seven earlier submissions. Pin it.
    nt = int(os.environ.get("N_THREADS", "8"))
    try:
        # A is an ENSEMBLE when several model_a*.txt are present. Round 36
        # measured five pooled models trained on five different matches_llm
        # draws, rank-averaged, at +0.01355 over the mean single model, with
        # near-zero variance -- draw-to-draw sd alone is ~0.015, so a
        # single-model export bets the submission on one draw.
        #
        # Averaged by RANK, not by probability: the members are separately
        # fitted boosters whose probability scales need not agree, and the
        # blend below is a rank blend anyway. With exactly one file this is
        # the identity path and behaves as it always did.
        a_files = sorted(
            os.path.join(gbdt_dir, f) for f in os.listdir(gbdt_dir)
            if f.startswith("model_a") and f.endswith(".txt"))
        if not a_files:
            raise FileNotFoundError("no model_a*.txt in the model directory")
        if len(a_files) == 1:
            pred_a = lgb.Booster(model_file=a_files[0]).predict(
                Xa, num_threads=nt)
        else:
            na = len(Xa)
            acc = np.zeros(na, dtype=np.float64)
            for f in a_files:
                p = lgb.Booster(model_file=f).predict(Xa, num_threads=nt)
                acc += rankdata(p) / na
                stage(f"  ensemble member {os.path.basename(f)}")
            pred_a = acc / len(a_files)
        stage(f"model A scored: {len(a_files)} booster(s), "
              f"num_threads={nt}")
        # start from A so rows no per-category model covers keep a real
        # prediction rather than silently collapsing to 0.5 * pred_a
        pred_b = pred_a.copy()
        covered = np.zeros(len(Xv), dtype=bool)
        for ci, c in enumerate(cats):
            k = cat == c
            if not k.any():
                continue
            path = os.path.join(gbdt_dir, f"model_b_{ci}.txt")
            if os.path.exists(path):
                pred_b[k] = lgb.Booster(model_file=path).predict(
                    Xv[k], num_threads=nt)
                covered |= k
        frac = float(covered.mean())
        stage(f"model B covered {frac:.3f} of rows")
        if frac < 0.5:
            log("B coverage too low - using A alone")
            out = pred_a
        else:
            n = len(pred_a)
            out = 0.5 * rankdata(pred_a) / n + 0.5 * rankdata(pred_b) / n
    except Exception as exc:
        log(f"MODEL LOAD/PREDICT FAILED ({exc!r}); lexical fallback")
        out = _lexical_score(X)

    out = np.asarray(out, dtype=np.float64)
    out[miss] = float(np.median(out))
    stage("scored")
    return out
