#!/bin/bash
# t176full RESUME with fewer dataloader workers -- the run was going to be
# OOM-killed and a naive resume would have died in the same place.
#
# WHAT WAS MEASURED, not guessed. At step 5000/87359 Box B read:
#     free -h        160Gi total, 159Gi used, AVAILABLE 899Mi, swap 0B
#     trainer RSS    36,360,916 kB parent + 32,234,312 kB worker (+6 more)
# DataLoader is constructed with num_workers=8, persistent_workers=True. Each
# worker is a FORK, and Python refcounting touches the shared pages, so
# copy-on-write degrades toward a real copy of the tokenised set. RAM therefore
# scales with PAIRS x WORKERS. At the champion's 2.2M draw that is free; at
# t176full's 11,182,000 it consumes the machine. This is `oompattern` in a new
# place -- that row was about fields held twice, this is the same object held
# nine times.
#
# WHY A RESTART RATHER THAN LETTING IT FALL OVER: the OOM killer would take the
# trainer, --resume-partial would restore it, and the resumed process would
# rebuild the SAME eight workers and die again. An OOM loop makes no progress and
# burns the box. The config has to change, so the restart is the cheap option.
#
# COST: about 35 minutes. The step-5000 partial is on Kaggle, so no training is
# lost; what is repaid is the text build and the length-bucketing pre-pass, which
# --resume-partial does not cache.
#
# WORKERS DO NOT COST THROUGHPUT HERE. nvidia-smi read 97% GPU utilisation at
# step 5000, i.e. the run is compute-bound and the loader is already ahead of the
# GPU. Dropping 8 -> 2 removes six copies of a 32 GB object and should not move
# steps/min. If it does, that is itself worth knowing and the log will show it --
# the pre-restart rate is 143 steps/min (500 steps per 3.5 min, steps 4000-5000).
#
# NOTHING ELSE MOVES. Same tag, same draw (pos 2,616,000 / mid 1,382,500 / zero
# 7,183,500 = 11,182,000 at 23.4% positive), same seed, same window, same
# recipe. `poolfull`'s pre-registration is untouched: >= +0.006 E_real at the
# matched ep0 comparison -> ACCEPT, one-sided, E_mix/E_mined descriptive.

set -o pipefail
cd /marimo || exit 1
log() { echo "[queue96] $(date -u +%H:%M:%S) $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POS=2.180
MID=2.765
ZERO=14.367
WORKERS=2

mkdir -p ~/.kaggle
[ -s /marimo/storage/access_token ] || { log "FATAL: no access_token"; exit 1; }
cp -f /marimo/storage/access_token ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN=$(cat /marimo/storage/access_token)
python -m kaggle --version >/dev/null 2>&1 || pip install -q kaggle

grep -q "num-workers" /marimo/train_ce.py || { log "FATAL: trainer lacks --num-workers"; exit 1; }
log "trainer carries --num-workers"

log "=== stopping the doomed t176full ==="
pkill -f "train_ce.py --tag t176full" 2>/dev/null
pkill -f "queue94.sh" 2>/dev/null
sleep 15
pgrep -af "train_ce.py --tag t176full" && { log "FATAL: old trainer still alive"; exit 1; }
log "old trainer cleared; free -h now:"
free -h | sed -n 2p

P=/marimo/ckpt-t176full-llm-ep0-partial
if [ ! -s "$P/model.safetensors" ]; then
  log "=== local partial missing, restoring from Kaggle ==="
  mkdir -p "$P"
  for i in 1 2 3; do
    python -m kaggle datasets download -d gordeevmax/ecup26-t176full-llm-ep0-partial -p "$P" --unzip && break
    log "retry $i"; sleep 20
  done
fi
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  [ -s "$P/$f" ] || { log "ABORT: $P/$f missing -- kaggle exits 0 on a 403"; exit 1; }
done
# The 6-file resume bundle is what makes this a RESUME and not a restart from
# scratch at the partial's weights. ckptguard: proven by readback, not reported.
# train_state.json is the file train_ce.py:570 actually requires -- it
# SystemExits without it rather than falling back, so this is a hard gate,
# not a note. (First version of this guard named resume_meta.json, which
# does not exist, and warned about a healthy bundle.)
for f in train_state.json batch_order.npz rng_state.pt; do
  [ -s "$P/$f" ] || { log "ABORT: $P/$f missing -- train_ce.py will refuse to resume"; exit 1; }
done
log "partial present ($(du -sh $P | cut -f1))"

[ -s /marimo/items_llm_subset.parquet ] || { log "FATAL: subset gone -- rerun queue94"; exit 1; }
log "subset present ($(stat -c%s /marimo/items_llm_subset.parquet) B)"

CMD="python train_ce.py --tag t176full --model $P --resume-partial $P --stage-a 1 --stage-b 2 --max-len 1024 --train-seed 0 --group-by-length 50 --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO --num-workers $WORKERS --ckpt-every 1000"
case "$CMD" in
  *"--llm-zero-scale 14.367"*"--num-workers 2"*)
      log "flag guard OK: full-pool draw AND the worker fix both present" ;;
  *)  log "FATAL: draw or worker flag missing from CMD"; exit 1 ;;
esac
log "=== t176full resumed: 11,182,000 pairs, ${WORKERS} dataloader workers ==="
$CMD > /marimo/t176full.log 2>&1
log "t176full rc=$?"
grep -aE "llm subsample|pairs with texts|MACRO=|done," /marimo/t176full.log | tail -8

PAIRS=""
for e in 0 1; do
  Dd=/marimo/ckpt-t176full-ep$e
  [ -s "$Dd/model.safetensors" ] && PAIRS="$PAIRS t176full-ep$e $Dd"
done
[ -z "$PAIRS" ] && { log "ABORT: no t176full checkpoint to score"; exit 1; }
log "scoring:$PAIRS"
python score_ckpt.py --max-len 1024 $PAIRS > /marimo/score_t176.log 2>&1
log "score_ckpt rc=$?"
grep -a "scored ->" /marimo/score_t176.log | tail -4
echo QUEUE96_DONE
