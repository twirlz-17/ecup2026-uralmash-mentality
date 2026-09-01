#!/bin/bash
# t119vol2x Stage B + t120loss, off the SAME recovered Stage-A checkpoint.
#
# WHY ONE QUEUE AND NOT TWO. t120loss's pre-registered CONTROL is a pw=1.0
# Stage B resumed from t119vol2x's Stage A -- which is exactly what t119vol2x's
# own Stage B is. Running them separately would burn 40 GPU-minutes to produce
# two runs that differ only in RNG, and would then invite comparing a treatment
# against the wrong one. The pw=1.0 arm IS both, and it is stated here so no one
# later reads a shared control as a coincidence.
#
# THE CHECKPOINT. /marimo/ckpt-t119vol2x/llm-ep0-partial is Stage A at step
# 76,000 of 91,666 (83%), loss 0.4917, written by the rolling mid-epoch save and
# recovered intact after the box death that was reported as having destroyed it.
# It is now also on Kaggle as ecup26-t119vol2x-stagea-p76k. It is fp32 and 2.27
# GB, so it can only be loaded through the no-mmap path (nommap.py) -- normal
# from_pretrained SIGSEGVs on it.
#
# WHAT THE 83% COSTS, said before the numbers exist. Stage A never annealed:
# the LR at step 76,000 is ~17% of peak and still falling, so this checkpoint is
# caught mid-schedule. That makes the Stage-A read CONSERVATIVE (a full epoch
# should be at least as good) and it makes the Stage-B arms internally valid
# (all three start from the identical weights) but NOT directly comparable to
# t116bge256's Stage B, which resumed from an annealed Stage A. Read the
# pos_weight contrast against the pw=1.0 arm; read the volume question against
# t116 only with this caveat attached.
set -u
while pgrep -f "[t]rain_ce\.py|[b]ench_infer|[s]core_ckpt|[s]core_swap" > /dev/null; do sleep 60; done
sleep 10

SA=/marimo/ckpt-t119vol2x/llm-ep0-partial
if [ ! -f "$SA/model.safetensors" ]; then
  echo "NO STAGE-A CHECKPOINT at $SA -- queue13 aborts."; exit 1
fi
SZ=$(stat -c%s "$SA/model.safetensors")
if [ "$SZ" -lt 1000000000 ]; then
  echo "Stage-A checkpoint is $SZ bytes, under 1GB -- the truncation signature"
  echo "from the 2026-08-21 refresh. queue13 aborts."; exit 1
fi
echo "Stage-A checkpoint ok: $SZ bytes"

# THE PRE-REGISTERED FAILING BRANCH of stageavol2x, which never got a reading:
# Stage-A human-val MACRO against t116bge256's 0.6819 on the identical split.
echo "=== stageavol2x pre-registered Stage-A read ==="
python /marimo/train_ce.py --tag t119sa76k --model "$SA" --max-len 256 \
  --val-only --stage-a 0 --stage-b 0 \
  > /marimo/val_t119sa76k.log 2>&1
grep -a "VAL-ONLY\|MACRO" /marimo/val_t119sa76k.log | tail -3
echo "   t116bge256 Stage A on the same split read 0.6819"

for PW in 1.0 0.134 3.0; do
  TAG="t120loss-pw${PW}"
  echo "=== $TAG ==="
  python /marimo/train_ce.py --tag "$TAG" \
    --model "$SA" --max-len 256 \
    --stage-a 0 --stage-b 2 --resume-epoch 0 \
    --pos-weight "$PW" \
    --batch-human 32 --ckpt-every 4000 \
    > "/marimo/train_${TAG}.log" 2>&1
  grep -a "VAL after human" "/marimo/train_${TAG}.log" | tail -2
  for EP in ep0 ep1; do
    D="/marimo/ckpt-${TAG}/${EP}"
    if [ -d "$D" ]; then
      python /marimo/push_ckpt.py "$D" "ecup26-${TAG}-${EP}" \
        >> /marimo/push_t120.log 2>&1 && echo "pushed $TAG/$EP" || echo "PUSH FAILED $TAG/$EP"
    fi
  done
done
echo QUEUE13_DONE
