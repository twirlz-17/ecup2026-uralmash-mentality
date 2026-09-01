# Weight manifest

The cross-encoder weights are 2.9 GB and are not in git. They are shipped as a
separate bundle; verify every file against the sha256 below before building an
archive with `inference/build_submission.py`.

The GBDT is the exception: it is small enough to live in the repo, at
`inference/models/gbdt/` (25 files, 286 MB).

## `ce-1/` — t176full-ep1

mmBERT-base (ModernBERT): 22 layers, hidden 768, vocab 256k. CE-1, MAX_LEN=1024, scores every pair. Archive slot models/ce-e5-base/.

| file | bytes | sha256 |
|---|---:|---|
| `config.json` | 2023 | `95bf8ad64111004425d2fbf89b69dcd07e2a83df0a9ea41334b87c97e8882d9f` |
| `model.safetensors` | 615076194 | `eb51ac28237686daef674fe3c7336e762aea54265571281ebd27cfc79e7bf9dc` |
| `tokenizer.json` | 34363287 | `aebee76d0312011b0d73b08a255d4d11d25c83e9639324870c7b916ea102e13e` |
| `tokenizer_config.json` | 682 | `80fd3e43697c68f25c40c6a6a7573709de6e2032edf0bcf5bd4b042185c39afc` |

## `ce-2/` — t120loss-pw0.134-ep1

bge-reranker-v2-m3 (XLM-R-large): 24 layers, hidden 1024, vocab 250,002. CE-2, MAX_LEN_2=256, top 30% of CE-1. Archive slot models/ce-2/.

| file | bytes | sha256 |
|---|---:|---|
| `config.json` | 869 | `af1735f663acb75fdad07a6b9015b783b08f06bfa242ec42107bd1ff61e747d6` |
| `model.safetensors` | 1135559698 | `4695498b8f98d04310a46ea5ec08f51f9e2ecace6c5c174cec6dc734f80b13e3` |
| `sentencepiece.bpe.model` | 5069051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `special_tokens_map.json` | 1015 | `66715ae6a0dd4aff4fe228bcaaccac6e52c83fd0ba80992c2be7d0e43b362307` |
| `tokenizer.json` | 17098085 | `5df1f55d60c9705a501ab9a75550728625740741fe4be308dac4806c16b7d51d` |
| `tokenizer_config.json` | 409 | `9dc74fea52f2666f23d5f5d52d9e2983adc82c00c9a35dadc39346ecbc9aa53a` |

## `ce-3/` — alexbge (teammate's Stage-B bge)

bge-reranker-v2-m3 (XLM-R-large), same shape as CE-2. CE-3, MAX_LEN_3=256, top 30% of the CE-1+CE-2 blend. Archive slot models/ce-3/.

| file | bytes | sha256 |
|---|---:|---|
| `config.json` | 869 | `af1735f663acb75fdad07a6b9015b783b08f06bfa242ec42107bd1ff61e747d6` |
| `model.safetensors` | 1135559698 | `64e4f958c3e248ad8828e8b6d205b0d1aa358e477726c7aff5f628da36b0ccb0` |
| `tokenizer.json` | 17098338 | `c4eb3c56fdc75b2990e1a823ea409c1de3c30cd7bcb56d07806059d643718281` |
| `tokenizer_config.json` | 599 | `cd07cc54a45a56fdac7983e4a098caba14e0f0e6404424ed1096f3d478dfb020` |

---

Total: **2.96 GB** across the three checkpoints.

Note that `ce-2/config.json` and `ce-3/config.json` are byte-identical —
same architecture, and the CE-3 checkpoint was deliberately shipped with the
base archive's own config rather than its trainer's copy. Swapping in a
checkpoint's own `config.json` silently reverted an environment fix once and
cost 0.00087 on the board; `evaluation/build_swap.py` refuses the swap unless
the config and tokenizer are byte-equal or the caller passes
`--keep-base-tokenizer` explicitly.

`ce-3/` ships no `sentencepiece.bpe.model` or `special_tokens_map.json`: its
tokenizer vocabulary (all 250,002 entries), normalizer, pre_tokenizer and
post_processor were verified equal to the CE-2 slot's before the base copy
was kept.
