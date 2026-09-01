# Training

**The clean, minimal commands are in `docs/REPRODUCE.md` §B.** Read those first.

The files here are the *literal* artefacts: `train_ce.py` is the trainer as it
ran, and `recipes/` holds the launch scripts exactly as they were executed on
the GPU box, operational noise included — GPU-busy waits, Kaggle checkpoint
pushes, resume bundles, guards against a half-written checkpoint. They are kept
verbatim rather than tidied because several of them encode a failure that cost
real time, and the comment explaining it is worth more than the tidiness.

| file | what it produced |
|---|---|
| `recipes/queue10.sh` | `t119vol2x` — the Stage A that CE-2 was built on |
| `recipes/queue13.sh` | `t120loss-pw*` — the loss sweep whose 0.134 arm **is CE-2** |
| `recipes/queue94.sh` | `t176full` — **CE-1**, original launch |
| `recipes/queue96.sh` | the same run restarted at `--num-workers 2` (see below) |
| `recipes/queue98.sh` | the same run resumed on a fresh box after a sandbox died |
| `gbdt/train_gbdt_v8.py` | `inference/models/gbdt/` — the shipped GBDT |
| `build_llm_subset_box.py` | `items_llm_subset.parquet`, which **must** be rebuilt for the draw you train on |

`train_ce.py` reads its data from `$ECUP_STOR` (default `$ECUP_ROOT/storage`,
`ECUP_ROOT=/marimo`); set both to point at wherever the competition parquets
live. `push_ckpt.py`, `nommap.py` and `score_ckpt.py` are support scripts the
recipes call: off-box checkpoint upload, a loader for fp32 checkpoints too large
for `mmap`, and the scorer that produces the dumps the evaluation tools read.

## Two things in these scripts that are not obvious

**`--num-workers 2` at large draws.** DataLoader workers are forks, and
CPython's refcounter writes to every object header it touches, so copy-on-write
degrades toward a real copy of the ~32 GB tokenised set. At 8 workers (the
default, and correct for every earlier recipe) the 11.18M-pair draw took the box
to 899 MiB available with no swap; at 2 workers, 141 GiB — same pairs, same
batch size, same window. The GPU was already at 97% utilisation, so the loader
was never the bottleneck. Anything above ~4M pairs needs the flag.

**Resume is not free, and it was quadratic.** Rebuilding the batch list from
`batch_order.npz` used to re-decompress the whole index array once per batch —
`NpzFile.__getitem__` re-reads its ZIP member every time, so a slice that looks
like cheap numpy indexing is a full inflate. At 87,359 batches that is 9.4 hours
of a silent no-op with the GPU at 0%, and there is no log line in the gap where
it happens. Fixed by hoisting `idx, sizes = z['idx'], z['sizes']` out of the
comprehension. The cost is quadratic in the draw, which is why it went unnoticed
for months at 2.2M pairs and became fatal at 11.2M.
