"""Cross-encoder half of the v9 blend.

Lifted from src/utils.py (the archive that scored LB 0.4150) with two changes:
  * no module-level thread-env mutation — run.py owns the thread caps;
  * the deadline is an ABSOLUTE epoch timestamp passed in by run.py, so this
    module needs no clock of its own.

The text template MUST stay byte-identical to kaggle-train/train_ce.py.
"""
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


# ---------------------------------------------------------------- item texts
_PAIR_TEXT_CACHE = {}


def _build_texts(items_path, log):
    """Return (id array, text array) with the exact training text template."""
    tbl = pq.read_table(items_path, columns=["id", "name", "attributes", "category"])
    df = tbl.to_pandas()
    del tbl
    log(f"ce: items loaded: {len(df)}")

    name = df["name"].astype(str)
    category = df["category"].astype(str)
    # training: re.sub(r'[{}\[\]"]', ' ', attrs) -> re.sub(r'\s+', ' ', ...) -> [:1500]
    attrs = (
        df["attributes"]
        .astype(str)
        .str.replace(r'[{}\[\]"]', " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 1500)
    )
    text = name + " [SEP] " + category + " [SEP] " + attrs
    # max_len=256 for the PAIR means no side survives past ~625 chars of
    # Russian; truncating here removes ~40% of the tokenizer's work for free.
    text = text.str.slice(0, int(os.environ.get("TEXT_CHAR_CAP", "2000")))
    ids = df["id"].to_numpy()
    del df, name, category, attrs
    log("ce: item texts built")
    return ids, text.to_numpy()


# ---------------------------------------------------------------- pair build
def _build_pairs(matches_path, ids, texts, log):
    m = pd.read_parquet(matches_path, columns=["id1", "id2"])
    index = pd.Index(ids)
    if not index.is_unique:
        keep = ~index.duplicated()
        log(f"ce: WARNING {int((~keep).sum())} duplicate item ids dropped")
        ids, texts = ids[keep], texts[keep]
        index = pd.Index(ids)
    pos1 = index.get_indexer(m["id1"].to_numpy())
    pos2 = index.get_indexer(m["id2"].to_numpy())
    missing = int((pos1 < 0).sum() + (pos2 < 0).sum())
    if missing:
        log(f"ce: WARNING {missing} pair sides had no item row; scored with empty text")

    lut = np.append(texts, "")  # index -1 lands on the empty string
    t1 = lut[pos1]
    t2 = lut[pos2]
    log("ce: pair texts built")
    return t1, t2


# ---------------------------------------------------------------- inference
def _load_model(model_path, log):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        major = torch.cuda.get_device_properties(0).major
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {"attn_implementation": "sdpa"}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=dtype, **kwargs
        )
    except TypeError:  # older transformers
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=dtype, **kwargs
        )
    model.to(device).eval()
    model.config.use_cache = False
    if device.type == "cuda":
        log(f"ce: model loaded on {torch.cuda.get_device_name(0)} dtype={dtype}")
    else:
        log("ce: model loaded on CPU (forced) — slow, deadline guard governs")
    return model, tokenizer, device


def _score(model, tokenizer, t1, t2, device, batch_size, max_len,
                  deadline_ts, log):
    """Same contract as src/ce.py `_score`: returns (scores, complete)."""
    import torch

    n = len(t1)
    scores = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return scores, True

    # ---- single tokenization pass, no padding yet -------------------------
    # Chunked so peak memory stays flat; fast tokenizers release the GIL, but
    # this is one thread by design -- the 'Already borrowed' race needs two.
    ids = [None] * n
    CH = 4096
    t_tok = time.time()
    for s in range(0, n, CH):
        enc = tokenizer(list(t1[s:s + CH]), list(t2[s:s + CH]),
                        padding=False, truncation=True, max_length=max_len)
        ids[s:s + CH] = enc["input_ids"]
    lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=n)
    order = np.argsort(lengths, kind="stable")
    log("ce: pairs sorted by TOKEN length (%.1fs, mean %.0f tok, max %d)"
        % (time.time() - t_tok, lengths.mean(), lengths.max()))

    pad_id = tokenizer.pad_token_id or 0
    right = getattr(tokenizer, "padding_side", "right") == "right"

    def make_batch(i):
        sl = order[i:i + batch_size]
        m = int(lengths[sl].max())
        inp = np.full((len(sl), m), pad_id, dtype=np.int64)
        att = np.zeros((len(sl), m), dtype=np.int64)
        for r, j in enumerate(sl):
            row = ids[j]
            k = len(row)
            if right:
                inp[r, :k] = row
                att[r, :k] = 1
            else:
                inp[r, m - k:] = row
                att[r, m - k:] = 1
        return {"input_ids": torch.from_numpy(inp),
                "attention_mask": torch.from_numpy(att)}

    # ---- producers now only pad; no tokenizer touched after this point ----
    n_prod = max(1, int(os.environ.get("TOKENIZER_THREADS", "4")))
    q = queue.Queue(maxsize=n_prod * 3)
    work = queue.Queue()
    for s in range(0, n, batch_size):
        work.put(s)

    def producer():
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                break
            q.put((i, make_batch(i)))
        q.put(None)

    threads = [threading.Thread(target=producer, daemon=True)
               for _ in range(n_prod)]
    for th in threads:
        th.start()

    done, complete, finished = 0, True, 0
    t_start = time.time()
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                finished += 1
                if finished == n_prod:
                    break
                continue
            i, enc = item
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            out = torch.sigmoid(
                model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            scores[order[i:i + len(out)]] = out
            done += len(out)

            if deadline_ts is not None and done >= batch_size * 4:
                rate = done / max(time.time() - t_start, 1e-6)
                projected = time.time() + (n - done) / rate
                if projected > deadline_ts:
                    log("ce: ABORT at %d/%d - projected finish %.0fs past "
                        "deadline (%.0f pairs/s)"
                        % (done, n, projected - deadline_ts, rate))
                    complete = False
                    break
            if done % (batch_size * 50) < batch_size:
                log("ce: scored %d/%d (%.0f/s)"
                    % (done, n, done / max(time.time() - t_start, 1e-6)))

    if not complete:
        for qq in (work, q):
            try:
                while True:
                    qq.get_nowait()
            except queue.Empty:
                pass
    for th in threads:
        th.join(timeout=5)
    log("ce: scored %d/%d complete=%s" % (done, n, complete))
    return scores, complete


def ce_scores(items_path, matches_path, model_path, batch_size, max_len,
              deadline_ts, log, subset=None):
    """Score every pair with the cross-encoder. Returns (scores, complete)."""
    import torch

    _key = (items_path, matches_path, os.environ.get("TEXT_CHAR_CAP", "2000"))
    _hit = _PAIR_TEXT_CACHE.get(_key)
    if _hit is None:
        ids, texts = _build_texts(items_path, log)
        t1, t2 = _build_pairs(matches_path, ids, texts, log)
        del ids, texts
        _PAIR_TEXT_CACHE[_key] = (t1, t2)
    else:
        t1, t2 = _hit
        log("ce: reusing cached pair texts (second cross-encoder)")
    if subset is not None:
        # CASCADE. Score only the head of the CE-1 ranking; everything else keeps
        # NaN and the caller substitutes CE-1's own rank there. The pair texts
        # are already built and cached, so selection costs one fancy-index.
        import numpy as _np
        sel = _np.asarray(subset)
        full1, full2 = t1, t2
        t1, t2 = full1[sel], full2[sel]
        log(f"ce: cascade subset {len(sel)}/{len(full1)} "
            f"({len(sel) / max(len(full1), 1):.1%}) of pairs")
    model, tokenizer, device = _load_model(model_path, log)
    try:
        if subset is not None:
            import numpy as _np
            part, ok = _score(model, tokenizer, t1, t2, device,
                              batch_size, max_len, deadline_ts, log)
            out = _np.full(len(full1), _np.nan, dtype=_np.float32)
            out[sel] = part
            return out, ok
        return _score(model, tokenizer, t1, t2, device,
                      batch_size, max_len, deadline_ts, log)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
