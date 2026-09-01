# What was tried

`docs/LEDGER.csv` is the full record: 411 experiments, each with a scope, a
**pre-registered acceptance rule written before the number existed**, the
instrument it was read on, the result, and a verdict. This file is the summary
an expert would want first.

Every number below is a local held-out read unless it says "board".

---

## 1. The final day's ladder — the only board receipts that matter

All one-variable steps, each built by a script that refuses to change more than
one entry of the archive and proves it entry-by-entry by CRC:

| sub | change | board | delta |
|---|---|---|---|
| v39 | (starting champion) | 0.5230454881 | — |
| v41 | CE-1 → `t177vol-ep1` | 0.5219365999 | −0.00111 |
| v45 | CE-1 → `t176full-ep1` | 0.5331887135 | **+0.01014** |
| v46 | add CE-3 | **0.5414129346** | **+0.00822** |
| v47 | add CE-4 | 0.5414120288 | −0.0000009 (the stage declined) |
| v52, v53 | CE-4 forced past its guard | — | **Error: container did not finish in time** |

**+0.0184 in one day from two changes.** The v47 → v46 difference of 9.06e-7 is
the determinism floor of the grader, measured for the first time — the two runs
had identical effective compute.

---

## 2. What worked

### 2.1 More Stage-A data at a *low* positive mix (+0.01014 board)

`t176full` draws the entire LLM pool — 11,182,000 of 11,187,780 pairs — at its
natural label mix of **23.4% positive**, against the previous recipe's 54.5%.
That is 5.08× the volume *and* a mix change.

The attribution is clean because the control ran first. `t177vol` took volume
alone to its ceiling (2.18×, where positives run out) with the mix held, and
read +0.01314 against +0.01469 for the earlier 2.0× step — i.e. **the pure
volume axis is flat beyond 2.0×**. So the whole of `t176full`'s gain is
attributable to the mix.

The mechanism is that the two axes are *coupled*: the sampler holds a fixed
positive:mid:zero ratio and positives are the scarce class, so volume at fixed
mix is capped at 2.18×. Dropping the positive share toward the pool's own makes
the entire 11.19M reachable. Zeros were the under-used class all along — we were
taking 45.8% of available positives and 7.0% of available zeros, while the board
runs at ~4.5% prevalence.

### 2.2 A third cross-encoder (+0.00822 board)

Covered in `SOLUTION.md` §2. The short version: the member that added most was
the one that was *weaker* alone.

### 2.3 Character cap 900 → 2000 (+0.00327)

Not a "looser" cap — an *off* cap. No item in the catalogue exceeds 2000
characters, while 11.40% exceed 900. One variable, one submission.

### 2.4 Token-length batch sorting (~3× on padding, +0.0000 on score)

Sorting batches by true token length instead of character length cuts padded
tokens by 41.6% at the 1024 window. Score-neutral by construction — it changes
only which pairs travel together — and it bought the wall-clock that the third
cross-encoder later spent.

### 2.5 A stale pruning threshold in the GBDT (+0.0176)

The naive-Bayes name table's minimum-count threshold had been calibrated at
783k pairs and never rescaled when the table was refit on 6.78M. Fitting on the
larger pool and pruning at 25 instead of 5 was **super-additive**: +0.00796 for
the bigger fit alone, +0.00441 for the threshold alone, +0.01934 together.

---

## 3. What did not work, with the mechanism

These are closed, not merely unfunded. Each one has a reason attached that
would survive being tried again.

### 3.1 Weight-space merging — 0 for 8

Soups, SLERP, TIES-Merging. Monotone in dose: more members and more scaling
meant more damage. The task vectors genuinely conflict (14.0% of parameters
split 2–2 across members), TIES is the published fix for exactly that, it was
implemented correctly, and it still lost. **Diversity lives in the outputs.**
Full detail in `SOLUTION.md` §3.

### 3.2 Substituting the CE-2 partner — 9 of 11 candidates lose

Range −0.003 … −0.012; the best alternative was +0.00161, inside noise. The same
weights were worth 3.5× more *added* as a new stage than *substituted* into an
existing one.

### 3.3 Blend-weight tuning — exhausted

Swept on the final stack, chosen on SELECT and quoted on the held-out REPORT
half:

| knob | SELECT | REPORT | verdict |
|---|---|---|---|
| `W_CE` 0.70 → 0.55 | +0.00151 | **−0.00063** | signs disagree — noise |
| `W_CE3` 0.20 … 0.35 | +0.0003 … +0.0016 | all within ±0.0007 | noise |
| `CE3_COVER` 0.30 → 0.70 | +0.00155 | +0.00063 | sub-noise, and costs time |
| `W_GBDT` 0.1 | +0.00492 (sd 0.00774) | — | sd > mean |

`W_CE` 0.55 *was* worth +0.00247 on the older stack with a clean interval.
Adding CE-3 absorbed it. This is a general pattern worth stating: a knob that
was live on a two-member blend can be dead on a three-member one, and the
earlier receipt is not evidence about the later stack.

### 3.4 Mined negatives

22% of LSH-mined "negative" pairs have byte-identical names. Training on them
showed a fake +0.25 locally while the board lost. The mined pool contains
genuine duplicates, so labelling them 0 teaches the opposite of truth. The
competition's own `matches_llm` is the same near-duplicate pool *with real
labels*, which is why Stage-A distillation works where mining does not.

### 3.5 Stacking / CatBoost on model outputs

Blocked by contamination: the shipped GBDT was fit on 88% of every pool we own,
so a stacker trained on its output reads "use the GBDT alone". Excluding it
leaves 4 highly-correlated rank features, and macro PR-AUC is not what a
stacker's objective optimises anyway.

### 3.6 Structural non-starters

- **Transitivity / graph propagation.** The labelled slice is a matching:
  98.7% of items appear in exactly one pair.
- **Images.** There are none in the data.
- **Per-category blend weights**, in any form.
- **Inference-time reordering of the pair.** Loss is uniform across pairs that
  truncate and pairs that fit, so there is nothing for reordering to recover.
- **A wider window.** At 1024 only 0.66% of pairs truncate at all (against 61.9%
  at 256). The axis that was once worth +0.0197 is exhausted.
- **EuroBERT**, which was the second-best fourth-member candidate and a
  genuinely different backbone: its remote code indexes
  `ROPE_INIT_FUNCTIONS['default']`, which does not exist in the grader's
  transformers. Shipping a vendored older tree to fix it cost 0.00087.

---

## 4. Method notes

These are the working rules that produced the ledger, and they are the part most
likely to transfer.

**Pre-register the acceptance rule before looking at the number.** Every row
does this. It repeatedly stopped motivated reasoning, and the cost of not doing
it is visible in the rows where an early conclusion had to be withdrawn.

**Ship one variable.** One submission moved four things and lost; the next moved
one and won +0.0055. Every build script in `evaluation/` refuses a
multi-variable change by default and proves single-variable-ness from the
artifact — entry by entry, by CRC — never from intent.

**A bar fitted to two points can veto a winner.** A rule requiring ≥ +0.010 on
the local instrument was derived from two board receipts and would have blocked
`t176full-ep1`, the +0.0101 that produced the champion. It is marked refuted in
the ledger. Bars are useful; bars stated on the wrong instrument are dangerous.

**Search a ledger by mechanism, not by flag name.** Two rows closed the `W_CE`
axis while describing the shipped value as 0.5; the container actually ran 0.7,
so the axis had been closed on a false premise about our own container.

**Validate the safety net with a negative test.** Guards that were never made to
fire have twice turned out to be inert in production — most expensively
`FORCE_CE=1`, which quietly disabled every budget check in the container.
