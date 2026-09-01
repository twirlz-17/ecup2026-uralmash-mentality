# Reproducing the submission

Two independent things can be reproduced, and they cost very different amounts:

| | needs | time |
|---|---|---|
| **A. Rebuild the graded archive** from the shipped weights | the 2.9 GB weight bundle | minutes |
| **B. Retrain the components** from the competition data | 1 GPU (≥48 GB), ~14 GPU-hours | days incl. data prep |

Start with A — it is what actually produced 0.5414129346430998.

---

## A. Rebuild and run the graded archive

### A.1 Get the weights

`docs/WEIGHTS.md` lists every file with its size and sha256. Lay them out as:

```
weights/
  ce-1/    t176full-ep1          (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)
  ce-2/    t120loss-pw0.134-ep1  (+ sentencepiece.bpe.model, special_tokens_map.json)
  ce-3/    alexbge-fp16          (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)
```

The GBDT is already in this repo at `inference/models/gbdt/`.

### A.2 Build the archive

```bash
python inference/build_submission.py --weights weights/ --out submission_v46.zip
```

Writes ~2.9 GB, `ZIP_STORED` (safetensors do not compress, and the grader
unpacks under a time budget). The script prints the entry list and a sha256.

### A.3 Run it in the grader image

```bash
bash evaluation/grader_run.sh submission_v46.zip /tmp/work
```

**Do not skip this.** An archive that was shipped without being run once scored
0.3611536 against an expected ~0.52 — the single most expensive mistake of the
competition.

### A.4 Environment knobs

Everything is an environment variable with a shipped default; the defaults *are*
the submission, and are listed here so a reviewer can see the whole
configuration in one place.

| variable | default | meaning |
|---|---|---|
| `FORCE_CE` | `1` | run the cross-encoders on scored stages regardless of the budget estimate (see SOLUTION.md §5.1) |
| `MAX_LEN` / `BATCH_SIZE` | `1024` / `256` | CE-1 |
| `CE_COVER` / `MAX_LEN_2` / `BATCH_SIZE_2` / `W_CE` | `0.30` / `256` / `512` / `0.7` | CE-2 |
| `CE3_COVER` / `MAX_LEN_3` / `BATCH_SIZE_3` / `W_CE3` | `0.30` / `256` / `512` / `0.25` | CE-3 |
| `W_GBDT` | `0.1` | weight of the GBDT rank in the final blend |
| `TEXT_CHAR_CAP` | `2000` | per-side character cap before tokenising |
| `SKIP_CE`, `SKIP_CE2`, `SKIP_CE3` | unset | disable a stage, for ablation |
| `N_THREADS` | `8` | CPU thread cap |

Setting `SKIP_CE2=1 SKIP_CE3=1` reproduces the single-cross-encoder pipeline;
`SKIP_CE=1` reproduces the GBDT-only pipeline (~0.36). Those two commands are
the cheapest way to see where the score comes from.

---

## B. Retrain the components

### B.1 Data

The trainer reads from `$ECUP_STOR` (default `$ECUP_ROOT/storage`,
`ECUP_ROOT=/marimo`):

```
storage/matches.parquet         365,654 human-labelled pairs
storage/items_human.parquet
storage/matches_llm.parquet     11,187,780 LLM-labelled pairs
storage/items.parquet           13,397,761 items
$ECUP_ROOT/items_llm_subset.parquet   item texts for the Stage-A draw
```

`items_llm_subset.parquet` must be **rebuilt for the draw you intend to train
on** — this is not optional and it silently corrupts a run if skipped:

```bash
python training/build_llm_subset_box.py     # writes items_llm_subset.parquet
```

The original `build_llm_subset` hardcoded a 2.2M-pair draw and wrote texts only
for the items behind it. `train_ce.py` scales the *pair* draw but looks texts up
in that file, so every pair outside the original draw is silently dropped. A
5.08× draw with the old subset file trains on 2.4M pairs and reports 11.2M.

### B.2 CE-1 — `t176full-ep1` (~7 GPU-hours)

The whole LLM pool at its natural label mix: 11,182,000 of the 11,187,780
available pairs, **23.4% positive** (against the previous recipe's 54.5%), which
is 5.08× the previous Stage-A volume.

```bash
python training/train_ce.py --tag t176full \
    --model jhu-clsp/mmBERT-base \
    --stage-a 1 --stage-b 2 \
    --max-len 1024 --train-seed 0 --group-by-length 50 \
    --llm-pos-scale 2.180 --llm-mid-scale 2.765 --llm-zero-scale 14.367 \
    --num-workers 2 --ckpt-every 1000
```

Everything not named takes the trainer's default: `--batch-llm 128`,
`--batch-human 96`, `--lr-llm 2e-5`, `--lr-human 1e-5`, AdamW, bf16. Stage A is
one epoch of LLM distillation; Stage B is two epochs of human fine-tuning.
**Ship `ep1`** — the second Stage-B epoch.

`--num-workers 2` is load-bearing at this draw and is the reason the recipe
looks different from the earlier ones. DataLoader workers are forks, and
CPython's refcounter writes to every object header it touches, so copy-on-write
degrades toward a real copy of the ~32 GB tokenised set. At 8 workers the box
went to 899 MiB available with no swap; at 2 workers, 141 GiB. Same pairs, same
batch size, same window. The GPU was already at 97% utilisation, so the loader
was never the bottleneck.

The scale factors sit 0.002 under the exact ratios on purpose: the sampler
asserts on an over-draw rather than silently shrinking, and positives bind at
2,619,567 available against 1,200,000 × 2.183 = 2,619,600.

Reference launches, with the operational notes attached:
`training/recipes/queue94.sh` (original), `queue96.sh` (the worker-count
restart), `queue98.sh` (resume on a fresh box).

### B.3 CE-2 — `t120loss-pw0.134-ep1` (~3 GPU-hours)

Two stages on a different backbone. Stage A first:

```bash
python training/train_ce.py --tag t119vol2x \
    --model BAAI/bge-reranker-v2-m3 --max-len 256 \
    --llm-scale 2.0 --stage-a 1 --stage-b 2 \
    --batch-llm 48 --batch-human 32 --ckpt-every 4000
```

then Stage B with a positive weight of 0.134, resumed off that Stage A:

```bash
python training/train_ce.py --tag t120loss-pw0.134 \
    --model  <ckpt-t119vol2x Stage-A dir> \
    --max-len 256 --stage-a 0 --stage-b 2 --resume-epoch 0 \
    --pos-weight 0.134 --batch-human 32 --ckpt-every 4000
```

**Ship `ep1`**; `ep2` was measured and rejected (+0.00139, under the bar).

Two honest caveats about this checkpoint, both of which a reviewer would
otherwise have to discover from the logs:

- The Stage A it was actually built on is a **mid-epoch partial** — step 76,000
  of 91,666 (83%), recovered after a sandbox died. The learning rate there is
  ~17% of peak and still falling, so the Stage A never annealed. A full Stage A
  should be at least as good; it is not the same weights.
- `--pos-weight` is LightGBM-style `pos_weight` on the BCE loss, so 0.134
  *down*-weights positives. 89.3% of positives are affected. The value was the
  winner of a three-arm sweep (1.0 / 0.134 / 3.0) run against a shared control.

`training/recipes/queue10.sh` and `queue13.sh`.

### B.4 CE-3

A teammate's Stage-B bge-reranker-v2-m3, trained on their own fork of the
recipe. The weights are in the bundle (`ce-3/`, sha256 in `docs/WEIGHTS.md`);
the training code for it is not ours to publish here. Architecturally it is the
same XLM-R-large reranker as CE-2 at the same 256 window — what makes it worth
a stage is that it was trained differently, not that it is better (it is
weaker solo: 0.50727 against CE-2's 0.51373).

**The pipeline degrades cleanly without it.** `SKIP_CE3=1` reproduces v45
(0.5331887134560345) exactly.

### B.5 GBDT v8 (~2 hours, CPU + RAM)

```bash
python training/gbdt/train_gbdt_v8.py
```

Written to run as a Kaggle kernel (`training/gbdt/mk94_generator.py` is the
generator that produced it). It fits the phase-1 tables, then model A on all
365,654 human pairs plus a 300k LLM draw, then the 20 per-category B models at
63 leaves / 2000 rounds, gates B to the 5 grid-SKU categories, and exports
`model_a.txt`, `model_b_<i>.txt`, `nb_{name,attr,key}.npz` and `meta.json` —
i.e. exactly `inference/models/gbdt/`.

Two notes carried in `meta.json` and worth reading before judging the numbers in
it: the phase-1 tables must be fitted on **training rows only** during
validation (an earlier round leaked the validation half into them and read
+0.076 high), and the internal ruler's *deltas* run 3.7×–37× hot even though its
*level* is fine.

---

## C. What cannot be reproduced from this repo

Stated plainly rather than left for a reviewer to find:

- **The exact `.zip` bytes.** The graded archives were built by a chain of
  one-variable patches over previous archives, and the local copies were deleted
  after the deadline. `inference/build_submission.py` rebuilds the same content
  from source; the entry *order* and timestamps differ, so the file hash will
  not match.
- **`inference/src/ce.py` was reconstructed**, not copied, from the three build
  scripts that produced it (`provenance/rebuild_ce.py` shows the chain step by
  step and asserts at every anchor). It compiles and its structure matches the
  build scripts' own post-conditions.
- **The declared base image.** `inference/metadata.json` in this repo carries
  `odsai/ecup26-matching-baseline:1.0`; the harness we ran locally used
  `twirlz/ecup26-matching:1.0`, and `inference/Dockerfile` builds a derived
  image adding `libgomp1`, `lightgbm==4.6.0` and `rapidfuzz==3.14.4`. Which of
  the three the final archive declared should be read off the submitted zip
  rather than trusted from here.
- **Wall-clock timings on grader hardware.** Every local timing was measured on
  an RTX PRO 6000 / RTX 6000-class box, and our own cost model was wrong by
  enough to say the champion configuration should not fit when it did.
