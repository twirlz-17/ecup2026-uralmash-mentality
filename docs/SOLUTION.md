# The solution

E-CUP 2026 product matching, team *uralmash mentality*. Final public score
**0.5414129346430998**, 6th place. Metric: macro PR-AUC over 20 categories.

---

## 1. The container, in execution order

The grader runs `python -u run.py` in a fixed image on an H100 80GB. Budgets are
**per stage**, and a submission runs check → public → private sequentially:

| stage | pairs | budget |
|---|---|---|
| check | ≤10,000 | 60 s |
| public | ~115,000 | 360 s |
| private | ~275,000 | 780 s |

`run.py` writes its answer progressively — a constant 0.5 first, then the GBDT,
then each better blend — so a stage that dies late still has the best answer it
reached on disk. Everything below is `inference/run.py`.

The archive has **47 entries and carries three cross-encoders**: `models/gbdt/`,
`models/ce-e5-base/`, `models/ce-2/` and `models/ce-3/`. That is worth stating
explicitly because it is easy to misremember — v46 is the *three*-cross-encoder
build, and the fourth stage that v47 added never executed.

The other selected submission, **v45**, is the same archive minus the third
stage: 43 entries, of which 42 are byte-identical to v46's, with the whole
difference in `run.py` amounting to a single inserted hunk of 63 lines. Both
rebuild from this repository and verify CRC-identical against the graded files
(`build_submission.py --variant v45|v46 --verify`).

### 1.1 GBDT — every pair

`src/gbdt_v2.py` + `models/gbdt/` (25 files, 286 MB). Two LightGBM boosters over
**~129 engineered features**:

- string similarity on names (rapidfuzz ratio/partial/token-sort/token-set/
  Jaro-Winkler, chrF, char-3-gram Jaccard);
- TF-IDF cosine over name words, name char-ngrams and attribute words;
- numeric-token and article/code-token overlap (Jaccard, containment,
  asymmetry, longest shared code);
- brand equality and brand-missing flags;
- attribute key/value agreement: key Jaccard, exact k=v matches, normalised and
  numeric value ratios, and a per-category **key-conflict** block over the 12
  most informative keys of each category;
- IDF-miss statistics (how much of each side is vocabulary the other has never
  seen), crowding features (how many catalogue items share the rare tokens);
- naive-Bayes log-odds tables over **131,496 name entries** and 243k attribute
  entries, shipped as `nb_name.npz` / `nb_attr.npz` / `nb_key.npz`;
- unit/size/colour/material conflict detectors and an SVD cosine.

Model **A** is global; model **B** is per-category, and is *gated* — it is used
only in the 5 grid-SKU categories where it helps (`Галантерея и аксессуары`,
`Мебель`, `Обувь`, `Одежда`, `Ювелирные изделия`). Both refit on 100% of the
365,654 human pairs at iteration counts found by early stopping on an inner
split. `models/gbdt/meta.json` carries the full feature list, per-category top
keys, row counts and the change log for each GBDT generation.

Trained by `training/gbdt/train_gbdt_v8.py`. Train/inference parity is
structural: the kernel imports the same `features.py` / `features2.py` modules
the container ships.

### 1.2 CE-1 — every pair

`models/ce-e5-base/` — **`t176full-ep1`**, mmBERT-base (ModernBERT
architecture): 22 layers, hidden 768, vocab 256k, 615 MB.
`MAX_LEN=1024`, `BATCH_SIZE=256`, bf16, SDPA attention.

> The directory name is a fossil. That slot held an e5-base in v12; renaming it
> later would have been a second variable in a build where exactly one thing was
> allowed to change per submission. `run.py:179` reads `models/ce-e5-base`.

### 1.3 CE-2 — top 30% of CE-1's ranking

`models/ce-2/` — **`t120loss-pw0.134-ep1`**, bge-reranker-v2-m3 (XLM-R-large):
24 layers, hidden 1024, vocab 250,002, 1136 MB.
`MAX_LEN_2=256`, `BATCH_SIZE_2=512`, `CE_COVER=0.30`.

```
ce = 0.70 * rank01(CE-1) + 0.30 * band_rank(CE-2, top 30%)
```

### 1.4 CE-3 — top 30% of the CE-1+CE-2 blend

`models/ce-3/` — a teammate's Stage-B bge, same architecture, 1136 MB.
`MAX_LEN_3=256`, `BATCH_SIZE_3=512`, `CE3_COVER=0.30`.

```
ce = 0.75 * ce + 0.25 * band_rank(CE-3, top 30% of the blend)
```

### 1.5 Final blend

```
out = 0.10 * rank(GBDT) + 0.90 * rank(ce)          # W_GBDT = 0.1
```

### 1.6 `band_rank` — how a partial stage is merged

A cascaded stage scores only a band, so it has no global rank; computing one
would need exactly the scores that were deliberately skipped. `band_rank` ranks
the band internally and maps it onto the slice of global rank space the band
already occupies:

```python
k, lo = len(sel), 1.0 - len(sel) / float(n)
out = np.full(n, lo)
out[sel] = lo + (1.0 - lo) * (rankdata(raw[sel]) / k)
```

Off-band positions get exactly `lo`; the tail ordering is carried entirely by
the `w * rank01(CE-1)` term, so there is no seam at the band boundary.

### 1.7 Text format

Identical in training and inference, on both sides of a pair:

```
name [SEP] category [SEP] attributes
```

where attributes have `[{}\[\]"]` and runs of whitespace collapsed to single
spaces, and the whole side is truncated at **2000 characters**
(`TEXT_CHAR_CAP`). 2000 is "off" rather than "looser": no item in the catalogue
exceeds it, while 11.40% exceed the previous 900. Worth +0.00327.

Batches are formed by sorting pairs by **true token length**, not character
length. Characters are a noisy proxy (r = 0.9603, 2.56–3.50 chars per token
across the p5–p95 range) and the residual is padding the GPU processes for
nothing: at the 1024 window, token-sorting cuts padded tokens by 41.6%. The
implementation tokenises once and pads batches from the kept encodings, which
also removes a Rust-tokenizer `Already borrowed` race that had cost a whole
cross-encoder pass.

---

## 2. Why the cascade is the solution and not a speed trick

The obvious reading of a cascade is "we cannot afford to run three big models on
275,000 pairs, so we run the later ones on a shortlist". That is true, but it is
not why the score is what it is.

**Measured contributions, all on held-out data or on the board:**

| change | value |
|---|---|
| CE-2 added to CE-1 alone (inside the cascade) | **+0.01434** |
| CE-3 added on top (board) | **+0.00822** |
| CE-1 checkpoint swap to `t176full-ep1` (board) | **+0.01014** |

CE-3 is **weaker than CE-2 standing alone** — 0.50727 against 0.51373 — and it
still added the second-largest gain of the final day. The same is true across
the whole ensemble study: members that are weaker solo but *differently trained*
add most. What the extra stages buy is decorrelation, not strength.

Two consequences we verified rather than assumed:

**Adding beats substituting, by 3.5×.** The same weights were worth +0.00565
added as a third stage and +0.00161 substituted into the second slot.
Substitution throws away the decorrelation that addition preserves.

**The partner slot is closed.** Eleven checkpoints were swapped into the CE-2
slot; nine lose (−0.003 … −0.012) and the best alternative gains +0.00161,
inside noise on a 0.0015-sd instrument. Notably the ranking is *not* by solo
strength: the best solo of the eleven placed third.

---

## 3. Weight-space merging is dead, and we know why

Model soups, SLERP and TIES-Merging over the same-ancestor mmBERT family went
**0 for 8**, monotone in how much merging was done:

| arm | delta | categories won |
|---|---|---|
| SLERP (2 models, t = 0.5) | −0.00224 | 9/20 |
| TIES 3-member, λ = 1.0 | −0.01021 | 5/20 |
| TIES 4-member, λ = 1.0 | −0.01312 | 3/20 |
| TIES 3-member, λ = 1.3 | −0.01498 | 7/20 |

This is directionally wrong, not mis-tuned — and the mechanism is real, which is
what closes the axis properly. Against the `jhu-clsp/mmBERT-base` ancestor, the
task vectors τ = W − W_base genuinely conflict: cos(τ, τ) is +0.62 … +0.78 in
`layers.21.mlp.Wo`, with 21–28% pairwise sign conflict and **14.0% of parameters
split 2–2**, where a plain average lands on ~zero and deletes a feature both
models learned. TIES is the published fix for exactly that; it was implemented
correctly and still lost. So sign interference was never the binding constraint.

**Diversity lives in the outputs.** Score-space combination is 4 for 4;
weight-space is 0 for 8.

A trap worth repeating: the raw weights are a red herring. cos(W, W) is
+0.998 … +1.000 and ‖τ‖ is 1–4% of ‖W_base‖, so these checkpoints look identical
while their learned shells disagree. **Compare task vectors, never weights.**

---

## 4. Which instrument to trust

This is the most transferable thing in the write-up.

### 4.1 The validation set

**E_real**: 36,542 rows (positives plus easy negatives) drawn from a leak-free
validation half, with macro PR-AUC re-weighted to a prevalence of 0.045 to match
the board. It is split **item-disjoint** into SELECT and REPORT halves by a hash
of `pid1`: candidates are chosen on SELECT and quoted on REPORT, so the number
reported is not the number optimised.

### 4.2 Never price a container change solo

A solo comparison answers "how good is this checkpoint alone". The container
asks "how good is it as the ranking half of `w·rank01(CE1) + (1−w)·band_rank(CE2)`,
then rank-blended with a GBDT". On the final day those differed by **34×**:

| candidate | solo read | cascade read | board |
|---|---|---|---|
| `t177vol-ep1` | +0.00024 | −0.00154 | −0.00111 |
| `t176full-ep1` | **+0.00030** | +0.00272 | **+0.01014** |
| CE-3 add | — | +0.00520 | +0.00822 |

The solo read called `t176full-ep1` flat, and an acceptance bar built on it
would have blocked the single largest gain of the competition. The cascade read
is 3 for 3 on sign, ratios 0.72× / 3.73× / 1.58×.

`evaluation/cascade_read.py` ports the blend arithmetic verbatim out of
`run.py`. Solo reads are for training attribution only.

### 4.3 The determinism floor

v46 and v47 had identical effective compute (v47's extra stage declined for lack
of budget) and scored **9.06e-7** apart. That is the first measurement of the
grader's run-to-run noise floor in this project, and it means board differences
above ~1e-5 are real signal — the +0.0101 and +0.0082 gains are ~10,000× it.

### 4.4 Local level does not predict board level

Local deltas and board deltas agreed on *sign* far more often than on
*magnitude*: two changes with the same local delta moved the board 17× apart.
Every number in this repo that is called "local" is a direction, not a forecast.

---

## 5. Traps that cost real money

These are recorded because each one has a receipt attached, and because most of
them are not specific to this competition.

1. **`FORCE_CE=1` makes in-container budget checks inert.** It sets
   `deadline_ts = None`, so every `left = deadline_ts - time.time()` becomes 1e9
   and no stage can ever decline itself. It is on by default for scored stages
   because the guards were firing falsely and silently dropping cross-encoders.
   Any *new* guard must read `T0 + total_budget * f - time.time()` directly.

2. **A timeout anywhere in the CE section discards every cross-encoder score**
   and ships the GBDT alone (~0.36). The `SIGALRM` watchdog raises inside
   whichever forward pass is running and propagates to the outer handler; a
   per-stage `try/except` protects only its own stage. Our last two submissions
   died exactly here: both exceeded the private budget, returned
   `Error: Container did not finish in time`, and were unselectable — though the
   public stage had already scored 0.54214.

3. **A container shipped without being run scored 0.3611536.** Run every archive
   in the grader image before submitting: `evaluation/grader_run.sh`.

4. **Swapping a checkpoint's own `config.json` silently reverts environment
   fixes.** Cost 0.00087. `evaluation/build_swap.py` compares config and
   tokenizer byte-wise against the base archive and aborts on a mismatch.

5. **The shipped `run.py` is CRLF.** LF patch anchors match nothing, and the
   patch silently no-ops rather than failing.

6. **`zipfile.writestr(zipinfo, data)` mutates the ZipInfo you pass it.** If it
   came from the source archive's `infolist()`, a later CRC comparison sees no
   difference for an entry that really changed. Snapshot CRCs first.

7. **The shipped GBDT cannot be priced locally** — it was fit on 88% of every
   pool we own, so any stacker trained on its output reads "use the GBDT alone".

8. **The cost model was benched on the wrong GPU.** `bench_infer.py` priced the
   champion at 13.44 minutes against a 13.0-minute budget — i.e. it said the
   configuration should not fit, while the board says it did. Do not accept
   "X does not fit" without a measurement on grader-class hardware.

---

## 6. What the container never got to run

`t167llm2x-ep1` as a **fourth** cross-encoder measured +0.00452 on SELECT and
+0.00312 on REPORT — the best of seven candidates and, at the observed transfer
range, worth roughly +0.002 … +0.012 on the board. It shipped twice and never
executed: once its guard declined for lack of budget (v47, identical to v46 to
within the noise floor), and twice the guard was removed to force it, which
overran the 780-second private budget.

The honest summary is that the pipeline was **timing-bound, not idea-bound**, at
the end. The measured next step was a real one; buying the ~5 minutes it needed
was the unsolved problem. The two obvious purchases, neither validated in the
grader image before the deadline:

- **Run the GBDT concurrently with CE-1.** The GBDT is pure CPU (pandas/numpy
  featurisation plus two LightGBM boosters, ~4.6 minutes at private size) and
  CE-1 is pure GPU; the shipped container runs them sequentially, so one of the
  two resources is idle throughout. Overlapping them costs no accuracy. The
  caveat that makes this non-trivial: Python delivers signals only to the main
  thread, so moving the GBDT to a worker while the `_alarm` watchdog is armed
  routes a GBDT overrun into aborting the cross-encoder instead.
- **Raise `BATCH_SIZE` from 256 to 512.** With SDPA/flash attention, attention
  memory is linear rather than quadratic in sequence length, so this is
  semantically free at the 1024 window.
