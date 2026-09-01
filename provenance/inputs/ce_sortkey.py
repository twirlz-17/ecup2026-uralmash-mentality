"""Drop-in replacement for the batching half of src/ce.py's `_score`.

WHY (LEDGER `sortkey`): the shipped path sorts pairs by CHARACTER length and
then pads each batch to its longest member in TOKENS. Characters are a noisy
proxy for tokens in a mixed-script catalogue -- pearson r 0.9603, chars-per-token
p5 2.56 / p50 3.08 / p95 3.50 -- and the residual is padding the GPU processes
for nothing. Measured over 60,000 real pairs at the shipped batch 256, padded
tokens relative to char-sorted @256:

    window   char-sort   token-sort   waste
    256        1.00x       0.92x       9.1%
    512        1.70x       1.27x      34.7%
    1024       2.33x       1.36x      71.4%

Sorting by true token length cuts padded tokens at 1024 by **41.6%**. That model
also explains the whole measured cost curve (predicts MAX_LEN^0.61 against a
measured 0.59), which is what closed `costcurve`.

WHAT THIS DOES AND DOES NOT CHANGE. It changes BATCH COMPOSITION only: which
pairs travel together and therefore how much padding each batch carries. It
never changes the token order inside a pair, and never changes the input any
pair is scored with -- so it is unrelated to the `reordertrunc` / `windowsat`
order-specialisation kill, and it needs no retraining. Every pair still gets
exactly the tokens the shipped tokenizer call would have given it.

HOW, without paying for tokenization twice. Knowing token lengths before
batching means tokenizing up front. A naive length pre-pass would tokenize
everything and then tokenize it again inside the producer threads -- CPU we may
not have inside a 780s private budget. So this does a SINGLE pass that keeps the
encodings and pads batches from them, which removes the producer-thread
tokenization entirely.

RISK, stated plainly: this rewrites a component that has already produced two
subtle bugs (the Rust tokenizer's 'Already borrowed' race and its warmup fix).
The single pass sidesteps that race by construction -- there is exactly one
tokenizer call, from one thread -- but it must still go through the container
verifier and a real timed run before it ships. `verify_sortkey()` below checks
the parts that can be checked without a GPU.
"""
import os
import queue
import threading
import time

import numpy as np


def _score_sorted(model, tokenizer, t1, t2, device, batch_size, max_len,
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
    return scores, complete


def verify_sortkey(tokenizer, t1, t2, max_len=1024, batch_size=256):
    """GPU-free checks of everything that can be checked without a model.

    1. every pair's token ids are EXACTLY what the shipped tokenizer call gives
    2. padding/attention masks match the tokenizer's own padded output
    3. the order -> scores round-trip lands each score on its original index
    4. reports the padded-token saving on this input
    """
    import torch

    n = len(t1)
    ids = []
    for s in range(0, n, 4096):
        ids.extend(tokenizer(list(t1[s:s + 4096]), list(t2[s:s + 4096]),
                             padding=False, truncation=True,
                             max_length=max_len)["input_ids"])
    lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=n)
    tok_order = np.argsort(lengths, kind="stable")
    chars = np.fromiter((len(a) + len(b) for a, b in zip(t1, t2)),
                        dtype=np.int64, count=n)
    char_order = np.argsort(chars, kind="stable")

    pad_id = tokenizer.pad_token_id or 0
    bad = 0
    for i in range(0, n, batch_size):
        sl = tok_order[i:i + batch_size]
        ref = tokenizer([t1[j] for j in sl], [t2[j] for j in sl],
                        padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt")
        m = int(lengths[sl].max())
        inp = np.full((len(sl), m), pad_id, dtype=np.int64)
        att = np.zeros((len(sl), m), dtype=np.int64)
        for r, j in enumerate(sl):
            k = len(ids[j])
            inp[r, :k] = ids[j]
            att[r, :k] = 1
        if not (torch.equal(torch.from_numpy(inp), ref["input_ids"].long())
                and torch.equal(torch.from_numpy(att),
                                ref["attention_mask"].long())):
            bad += 1
    assert bad == 0, "%d batches differ from the tokenizer's own output" % bad

    # round-trip: fake scores = original index, must come back in place
    scores = np.full(n, np.nan, dtype=np.float64)
    for i in range(0, n, batch_size):
        sl = tok_order[i:i + batch_size]
        scores[tok_order[i:i + batch_size]] = sl.astype(np.float64)
    assert np.array_equal(scores, np.arange(n)), "order round-trip is wrong"

    def padded(o):
        t = lengths[o]
        return sum(len(t[s:s + batch_size]) * int(t[s:s + batch_size].max())
                   for s in range(0, len(t), batch_size))

    pc, pt = padded(char_order), padded(tok_order)
    return {"batches_checked": (n + batch_size - 1) // batch_size,
            "padded_char_sort": pc, "padded_token_sort": pt,
            "saving_pct": 100.0 * (1 - pt / pc)}
