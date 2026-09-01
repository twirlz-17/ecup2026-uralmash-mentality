# Weight manifest

Every hash below was read out of **`submission_v46.zip` itself** — the archive
that scored 0.5414129346430998 — not out of a training directory that happened
to be lying around. `inference/build_submission.py --verify <that zip>` rebuilds
the archive from this repository plus these files and confirms all 47 entries
CRC-identical.

## What is already in this repository

- `inference/models/gbdt/` — the trained GBDT, all 25 files, byte-identical to the archive (286 MB).
- `inference/models/ce-{1,2,3}/` — each checkpoint's `config.json`, `tokenizer_config.json`
  and (for CE-2) `special_tokens_map.json`. These encode *decisions*, not weights, so they
  belong in git. One is load-bearing: see the note at the bottom.
- `inference/run.py`, `inference/src/*.py`, `inference/metadata.json` — byte-identical to the archive,
  CRLF and all.

## What is not (seven files, 2.96 GB)

Lay them out as `weights/ce-1/`, `weights/ce-2/`, `weights/ce-3/` and point
`inference/build_submission.py --weights` at the parent.

### `ce-1/` — t176full-ep1

mmBERT-base (ModernBERT): 22 layers, hidden 768, vocab 256k. **CE-1**, `MAX_LEN=1024`, `BATCH_SIZE=256`, scores every pair. Archive slot `models/ce-e5-base/` (fossil name).

| file | bytes | sha256 |
|---|---:|---|
| `model.safetensors` | 615076194 | `eb51ac28237686daef674fe3c7336e762aea54265571281ebd27cfc79e7bf9dc` |
| `tokenizer.json` | 34363287 | `aebee76d0312011b0d73b08a255d4d11d25c83e9639324870c7b916ea102e13e` |

### `ce-2/` — t120loss-pw0.134-ep1

bge-reranker-v2-m3 (XLM-R-large): 24 layers, hidden 1024, vocab 250,002. **CE-2**, `MAX_LEN_2=256`, `BATCH_SIZE_2=512`, top 30% of CE-1's ranking.

| file | bytes | sha256 |
|---|---:|---|
| `model.safetensors` | 1135559698 | `4695498b8f98d04310a46ea5ec08f51f9e2ecace6c5c174cec6dc734f80b13e3` |
| `tokenizer.json` | 17098085 | `5df1f55d60c9705a501ab9a75550728625740741fe4be308dac4806c16b7d51d` |
| `sentencepiece.bpe.model` | 5069051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |

### `ce-3/` — alexbge (teammate's Stage-B bge)

bge-reranker-v2-m3 (XLM-R-large), same shape as CE-2. **CE-3**, `MAX_LEN_3=256`, `BATCH_SIZE_3=512`, top 30% of the CE-1+CE-2 blend.

| file | bytes | sha256 |
|---|---:|---|
| `model.safetensors` | 1135559698 | `64e4f958c3e248ad8828e8b6d205b0d1aa358e477726c7aff5f628da36b0ccb0` |
| `tokenizer.json` | 17098338 | `c4eb3c56fdc75b2990e1a823ea409c1de3c30cd7bcb56d07806059d643718281` |

Total: **2.96 GB**.

---

## The one file that is not what you would guess

`models/ce-e5-base/tokenizer_config.json` in the graded archive (574 bytes,
sha256 `14b147f2a4f939d9…`) is **not** the copy the CE-1 training run wrote
(682 bytes). The trainer's copy carries four extra keys — `max_length`,
`stride`, `truncation_side`, `truncation_strategy` — which would give the
tokenizer a default `max_length`; the archive deliberately keeps the base
copy, which does not, and sets `is_local: false`.

This is not fussiness. Shipping a checkpoint's own config over the base one
silently reverted an environment fix once and cost 0.00087 on the board.
`evaluation/build_swap.py` aborts on exactly this mismatch unless the caller
passes `--keep-base-tokenizer` and the tokenizer has been proven equivalent
first. It is the reason the small config files live in this repo rather than
travelling with the weights.

`ce-2/config.json` and `ce-3/config.json` are byte-identical — same
architecture, and the CE-3 checkpoint was shipped with the base archive's
config for the same reason. `ce-3/` ships no `sentencepiece.bpe.model` or
`special_tokens_map.json`: its tokenizer vocabulary (all 250,002 entries),
normalizer, pre_tokenizer and post_processor were verified equal to the CE-2
slot's before the base copy was kept.
