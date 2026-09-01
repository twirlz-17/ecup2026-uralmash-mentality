# Local validation tooling

These are the instruments the decisions in `docs/LEDGER.csv` were made on. They
are included so a reviewer can see *how* a number was produced, not only what it
was. They are lifted from the team's working tree and expect its layout — score
dumps under `outputs/dumps/ce_scores_<tag>.npz` and the validation views under
`outputs/` — so they will not run unmodified against this repository alone.

| file | what it does |
|---|---|
| `cascade_read.py` | **the instrument that works.** Prices a candidate checkpoint in the container's own blend arithmetic, ported verbatim from `run.py`. 3-for-3 on board sign. |
| `arm_read_local.py` | paired bootstrap over the held-out set; solo reads, for training attribution only |
| `third_member.py` | prices a candidate as an additional ensemble member on the current stack |
| `grader_run.sh` | runs a built archive inside the grader image — the gate that a container must pass before it is submitted |
| `sim_ce3.py` | executes a new stage's source **lifted out of the built zip** against stubs, through every branch (normal / skip / raises / model-dir absent), asserting that every degraded path leaves the previous result untouched |
| `check_ce3_scope.py` | static check (AST) that every name a newly inserted block reads is actually bound at that point in `main()` |
| `build_swap.py` | one-entry weights swap; CRC-verified, and aborts if a checkpoint's `config.json` or tokenizer differs from the base archive's |
| `build_ce3.py`, `build_ce4.py` | add a cascaded stage to an existing archive; CRC-verified single-entry change |

## Why the reads are split SELECT / REPORT

The 36,542-row validation set is split **item-disjoint** by a hash of `pid1`.
Candidates are chosen on the SELECT half and the number that gets written down
comes from REPORT. A knob whose SELECT and REPORT deltas disagree in *sign* is
noise, and several did — see `docs/EXPERIMENTS.md` §3.3.

## Why `cascade_read.py` and not a solo comparison

A solo read asks how good a checkpoint is alone. The container asks how good it
is as the ranking half of a blend. On the change that produced the champion,
those two answers differed by 34× and a bar built on the solo read would have
rejected it.
