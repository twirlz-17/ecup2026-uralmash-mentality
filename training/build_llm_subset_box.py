"""Rebuild items_llm_subset.parquet for a SCALED LLM draw. Runs on the box.

WHY THIS EXISTS. `llmscale2` -- the only experiment that has ever asked whether
Stage A saturates at 2.2M LLM pairs -- was killed on 2026-08-28 after 50 minutes
because `--llm-scale 2.0` drew 4,400,000 pairs but only 2,396,739 of them had
item texts. That was read at the time as a data limit. It is not:

    matches_llm.parquet      11,187,780 pairs
    items.parquet            13,397,761 items, all with name/attributes/category

The texts exist for essentially every pair. The cap is `tools/build_llm_subset.py`,
which hardcodes LLM_SAMPLE = {pos 1.2M, mid 500k, zero 500k} at SEED 42 and writes
texts for ONLY the items behind that 2.2M draw. train_ce.py then scales the PAIR
draw but still looks the texts up in that file, so everything outside the original
draw is silently dropped. The +8.9% we measured is the item overlap between the
two draws, not a signal about the data.

What we actually use today, by class:
    pos  (>=0.5)   1,200,000 of 2,619,567   45.8%
    mid  (0<t<0.5)   500,000 of 1,383,550   36.1%
    zero (==0)       500,000 of 7,184,663    7.0%
    total          2,200,000 of 11,187,780  19.7%

So four fifths of the labelled pool has never reached a model, and the experiment
that would have priced it never ran.

SCALE CEILING. Holding the class ratio fixed (which is what makes --llm-scale a
one-variable DOSE change rather than a MIX change), positives run out first:
2,619,567 / 1,200,000 = 2.18. So 2.0 is the largest round scale that keeps the
mix identical, and anything above 2.18 is a different experiment.

Usage: python build_llm_subset_box.py --llm-scale 2.0
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED = 42
LLM_SAMPLE = {"pos": 1_200_000, "mid": 500_000, "zero": 500_000}
STOR = "/marimo/storage"

ap = argparse.ArgumentParser()
ap.add_argument("--llm-scale", type=float, default=1.0)
ap.add_argument("--llm-zero-scale", type=float, default=1.0,
                help="multiply ONLY the zero class. MUST match the value passed "
                     "to train_ce.py, or the trainer draws zeros this file has "
                     "no texts for and the arm silently shrinks -- which is "
                     "exactly how llmscale2 died.")
ap.add_argument("--llm-pos-scale", type=float, default=1.0,
                help="multiply ONLY the pos class. MUST match train_ce.py.")
ap.add_argument("--llm-mid-scale", type=float, default=1.0,
                help="multiply ONLY the mid class. MUST match train_ce.py. "
                     "0 deletes the soft-label middle (confident-only Stage A).")
ap.add_argument("--clean", action="store_true",
                help="MUST match train_ce.py. The trainer samples from the CLEANED "
                     "frame, so if this file samples from the raw one the two draws "
                     "are different rows and the arm silently shrinks -- the same "
                     "failure this whole script exists to prevent.")
ap.add_argument("--clean-mode", default="contra", choices=["contra", "margin", "both"])
ap.add_argument("--out", default="/marimo/items_llm_subset.parquet")
a = ap.parse_args()

mllm = pd.read_parquet("%s/matches_llm.parquet" % STOR)
if a.clean:
    fl = np.load("/marimo/llm_clean_flags.npz")
    n = int(fl["n"])
    assert n == len(mllm), "flag file has %d rows, matches_llm has %d" % (n, len(mllm))
    ypk = np.unpackbits(fl["ypack"])[:n].astype(bool)
    assert (ypk == (mllm.target.to_numpy() > 0.5)).all(), "flag file is not row-aligned"
    drop = np.zeros(n, dtype=bool)
    if a.clean_mode in ("contra", "both"):
        drop |= np.unpackbits(fl["contra"])[:n].astype(bool)
    if a.clean_mode in ("margin", "both"):
        drop |= np.unpackbits(fl["margin"])[:n].astype(bool)
    print("clean[%s]: dropping %d rows (%.2f%%)"
          % (a.clean_mode, int(drop.sum()), 100.0 * drop.mean()), flush=True)
    mllm = mllm[~drop]
avail = {"pos": int((mllm.target >= 0.5).sum()),
         "mid": int(((mllm.target > 0) & (mllm.target < 0.5)).sum()),
         "zero": int((mllm.target == 0).sum())}
want = {k: int(round(v * a.llm_scale
                     * (a.llm_zero_scale if k == 'zero' else
                        a.llm_mid_scale if k == 'mid' else a.llm_pos_scale)))
        for k, v in LLM_SAMPLE.items()}
short = {k: (avail[k], v) for k, v in want.items() if avail[k] < v}
# Fail loudly. A silently shrunk draw is exactly what made llmscale2 unreadable,
# and it looked like a result rather than a bug for a whole arm.
assert not short, ("scale %.2f/zero %.2f exceeds the pool: %s (have, need)"
                   % (a.llm_scale, a.llm_zero_scale, short))
print("scale %.2f zero-scale %.2f -> %s (avail %s)"
      % (a.llm_scale, a.llm_zero_scale, want, avail), flush=True)

# SEED 42 and the same three .sample() calls in the same order as
# tools/build_llm_subset.py and train_ce.py, so at scale 1.0 this reproduces the
# existing file and at scale 2.0 the original draw is a SUBSET of the new one --
# the dose arm therefore adds rows rather than replacing them.
pos = mllm[mllm.target >= 0.5].sample(n=want["pos"], random_state=SEED)
mid = mllm[(mllm.target > 0) & (mllm.target < 0.5)].sample(n=want["mid"], random_state=SEED)
zero = mllm[mllm.target == 0].sample(n=want["zero"], random_state=SEED)
sub = pd.concat([pos, mid, zero])
need = pd.unique(np.concatenate([sub.id1.to_numpy(), sub.id2.to_numpy()]))
print("subsample %d pairs, %d unique items" % (len(sub), len(need)), flush=True)
del pos, mid, zero, sub, mllm

vs = pa.array(need)
pf = pq.ParquetFile("%s/items.parquet" % STOR)
parts, tot = [], 0
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["id", "name", "attributes", "category"])
    t = t.filter(pc.is_in(t["id"], value_set=vs))
    if t.num_rows:
        parts.append(t)
        tot += t.num_rows
    if rg % 10 == 0:
        print("rg %d/%d: %d" % (rg, pf.metadata.num_row_groups, tot), flush=True)
out = pa.concat_tables(parts)
pq.write_table(out, a.out, compression="zstd")
print("wrote %d rows (%.1f%% of the %d items needed), %.0f MB"
      % (out.num_rows, 100.0 * out.num_rows / len(need), len(need),
         os.path.getsize(a.out) / 1e6))
# The coverage number is the thing to read: llmscale2 died because coverage was
# ~54% and nobody printed it.
assert out.num_rows >= 0.98 * len(need), (
    "only %d of %d needed items have texts -- do NOT train on this, the arm "
    "would silently shrink again" % (out.num_rows, len(need)))
print("COVERAGE OK")
