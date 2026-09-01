#!/bin/bash
# t119vol2x -- Stage A at 2x volume, mix UNCHANGED. LEDGER stageavol2x.
#
# THE ONE LARGE LEVER NOBODY HAS PULLED. Stage A trains on 2.2M of the 11.19M
# available LLM pairs (19.7%). This doubles it to 4.4M with the 12:5:5 class
# RATIO held fixed, so it is a pure VOLUME axis and not a mix change -- the two
# are judged by different rules (stageavol2x vs mixaxisclosed) and conflating
# them is what made me call this unreadable in the first place.
#
# WHY 256 AND NOT 320. t117 is deciding the window right now, but windowlbonly
# means its local val CANNOT tell us whether 320 is better, and no board read
# will exist today. Running this at 320 would compound an unproven change with
# a new one and neither could be attributed. 256 is the window PROVEN on the
# board -- it is what our champion v14 (0.48855) runs -- so this is a single
# variable against t116bge256: same backbone, same window, same batches, same
# Stage B, same seed. Only --llm-scale changes.
#
# HOW IT MAY BE READ -- ONE-SIDED, pre-registered in LEDGER stageavol2x.
# The ruler's real slice IS matches.parquet, i.e. the human distribution, so it
# flatters whichever arm leans human and penalises whichever leans LLM. More
# Stage A leans LLM. Therefore:
#   * E_real@0.045 says 2x is BETTER -> credible, the gain had to overcome a
#     bias working against it. Bar: > +0.006 and positive in >= 12/16 splits.
#   * E_real says WORSE -> UNINFORMATIVE. Decides nothing, and must NOT be
#     written up as a rejection.
# A one-sided instrument is enough to act on a win and not enough to reject a
# loss.
#
# WHY BEFORE t118 (LEDGER stageapriority): the board has five receipts on
# Stage-A interventions -- v10-lb, v13s, v13, t105, t96 -- and every one is a
# LOSING SUBTRACTION. t118 is another subtraction. This is the opposite sign
# and untried.
#
# COST: ~4.4M pairs at 256 is roughly 4 h of Stage A plus ~35 min of Stage B.
# Checkpoints are pushed the moment they exist -- the box died idle once today
# and took t116's Stage A with it.
set -u
while pgrep -f "[t]rain_ce\.py|[b]ench_infer|[s]core_swap|[q]ueue8\.sh|[q]ueue9\.sh|[q]ueue11\.sh" > /dev/null; do sleep 60; done
sleep 10

python -m pip install -q kaggle 2>/dev/null || true

echo "=== t119vol2x: Stage A 4.4M pairs @256, ratio unchanged ==="
python /marimo/train_ce.py --tag t119vol2x   --model BAAI/bge-reranker-v2-m3 --max-len 256   --llm-scale 2.0   --stage-a 1 --stage-b 2   --batch-llm 48 --batch-human 32   --ckpt-every 4000   > /marimo/train_t119vol2x.log 2>&1

for EP in stageA ep0 ep1; do
  D=/marimo/ckpt-t119vol2x/$EP
  if [ -d "$D" ]; then
    python /marimo/push_ckpt.py "$D" "ecup26-t119vol2x-$EP"       >> /marimo/push_t119.log 2>&1 && echo "pushed $EP" || echo "PUSH FAILED $EP"
  fi
done

# NO explicit universe scoring here: train_ce.py already scores the universe
# after every epoch and writes ce_scores_<tag>-ep<n>.npz itself (train_ce.py:362),
# which is exactly the filename score_ckpt.py would produce -- so calling it would
# spend ~10 GPU-minutes rewriting the same file. It also matters that t116's
# baseline scores were written by the trainer the same way, so t119 vs t116 is
# like-for-like with no code path to argue about.
echo QUEUE10_DONE
