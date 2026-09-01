#!/bin/bash
# queue98 -- t176full RESUME ON A FRESH BOX (death #11).
#
# WHY THIS EXISTS RATHER THAN JUST RERUNNING queue96. queue96 resumes, but its
# first act is `[ -s /marimo/items_llm_subset.parquet ] || FATAL`. That file is
# 1.97 GB, it is NOT on Kaggle, and a fresh sandbox has none -- so queue96 alone
# aborts in ten seconds. The subset has to be REBUILT before the resume, which
# is queue94's job. This queue is those two halves in the right order.
#
# WHAT SURVIVED: ecup26-t176full-llm-ep0-partial, pushed 00:19:59, train_state
# step 57000/87359 -- 65% of Stage A. ~30,359 steps remain, ~3.2 h at the
# measured 157 steps/min. The rolling off-box push kept up this time, unlike
# the 4,000-step lag corrected in `workermem`.
#
# THE REBUILD IS SAFE TO REDO because build_llm_subset_box.py is deterministic:
# SEED = 42 and three .sample(random_state=SEED) calls in a fixed order. And it
# is CHECKED rather than trusted -- train_ce.py compares df_len and df_sig
# against the bundle and SystemExits on mismatch rather than training saved
# batch indices against different rows. If the rebuild drifts, this queue dies
# loudly instead of silently corrupting the arm.
#
# `poolfull`'s pre-registration is untouched: >= +0.006 E_real at the matched
# ep0 comparison against t149baseA-ep0 -> ACCEPT, one-sided, E_mix/E_mined
# descriptive. Nothing here moves a baseline.

set -o pipefail
cd /marimo || exit 1
log() { echo "[queue98] $(date -u +%H:%M:%S) $*"; }
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

# ---- 1. data ---------------------------------------------------------------
M=gordeevmax/ecup26-matching
D=gordeevmax/ecup26-derived
kdl() {
  for i in 1 2 3; do
    python -m kaggle datasets download -d "$1" -f "$2" -p "$3" && break
    log "retry $i on $1/$2"; sleep 20
  done
  ( cd "$3" && for z in *.zip; do [ -e "$z" ] && unzip -o -q "$z" && rm -f "$z"; done ) 2>/dev/null
  [ -s "$3/$2" ] || { log "ABORT: $3/$2 missing after download"; return 1; }
}
mkdir -p /marimo/storage
[ -s /marimo/storage/items.parquet ]         || kdl $M items.parquet         /marimo/storage || exit 1
[ -s /marimo/storage/matches.parquet ]       || kdl $M matches.parquet       /marimo/storage || exit 1
[ -s /marimo/storage/matches_llm.parquet ]   || kdl $M matches_llm.parquet   /marimo/storage || exit 1
[ -s /marimo/storage/items_human.parquet ]   || kdl $M items_human.parquet   /marimo/storage || exit 1
[ -s /marimo/storage/universe_view.parquet ] || kdl $D universe_view.parquet /marimo/storage || exit 1
log "storage ready"

# ---- 2. tooling ------------------------------------------------------------
for f in /marimo/train_ce.py /marimo/build_llm_subset_box.py /marimo/preflight_draw.py \
         /marimo/score_ckpt.py /marimo/push_ckpt.py; do
  [ -s "$f" ] || { log "ABORT: missing $f"; exit 1; }
done
grep -q "llm-pos-scale" /marimo/train_ce.py             || { log "ABORT: trainer lacks the class knobs"; exit 1; }
grep -q "llm-pos-scale" /marimo/build_llm_subset_box.py || { log "ABORT: builder lacks the class knobs"; exit 1; }
grep -q "num-workers"   /marimo/train_ce.py             || { log "ABORT: trainer lacks --num-workers (workermem)"; exit 1; }
grep -q "idx, sizes = " /marimo/train_ce.py             || { log "ABORT: trainer lacks the npzquadratic fix"; exit 1; }
log "tooling verified (push_ckpt, workers, npz fix all present)"

# ---- 3. the partial --------------------------------------------------------
P=/marimo/ckpt-t176full-llm-ep0-partial
if [ ! -s "$P/model.safetensors" ]; then
  log "=== restoring the step-57000 partial from Kaggle ==="
  mkdir -p "$P"
  for i in 1 2 3; do
    python -m kaggle datasets download -d gordeevmax/ecup26-t176full-llm-ep0-partial -p "$P" --unzip && break
    log "retry $i"; sleep 20
  done
fi
for f in config.json model.safetensors tokenizer.json tokenizer_config.json \
         train_state.json batch_order.npz rng_state.pt; do
  [ -s "$P/$f" ] || { log "ABORT: $P/$f missing -- kaggle exits 0 on a 403"; exit 1; }
done
log "partial present ($(du -sh $P | cut -f1)), state: $(cat $P/train_state.json)"

# ---- 4. rebuild the subset the resume needs --------------------------------
# PRESENCE IS NOT VALIDITY, and this bit me on the first run of this queue.
# /marimo is a PERSISTENT VOLUME: a fresh sandbox came back carrying a STALE
# items_llm_subset.parquet (539,488,868 B from an older, smaller draw; the
# full-pool draw needs 1,968,262,613 B). The original guard here was
# `[ ! -s file ]`, so the mere PRESENCE of a file skipped the rebuild and the
# trainer would have been handed a subset covering 3,304,398 of the 12,379,530
# items it needs. The preflight caught it -- COVERAGE_BAD, ABORT -- which is
# `molabvolume`'s second face: survival silently swapping a run's inputs.
# So: preflight FIRST, rebuild only if it fails, then preflight again and mean
# it. Self-healing instead of trusting the filesystem.
pf() {
  python /marimo/preflight_draw.py --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO \
    > /marimo/preflight_full.log 2>&1
  cat /marimo/preflight_full.log
  grep -q COVERAGE_OK /marimo/preflight_full.log
}

log "=== PREFLIGHT #1: does the subset on disk cover the draw? ==="
if pf; then
  log "preflight PASSED on the existing subset -- no rebuild needed"
else
  log "subset does not cover the draw (stale or absent) -- REBUILDING"
  rm -f /marimo/items_llm_subset.parquet
  python build_llm_subset_box.py --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO \
    > /marimo/build_subset_full.log 2>&1
  RC=$?
  log "builder rc=$RC"
  tail -6 /marimo/build_subset_full.log
  [ "$RC" = "0" ] || { log "ABORT: subset rebuild failed"; exit 1; }
  ls -la /marimo/items_llm_subset.parquet
  log "=== PREFLIGHT #2: after the rebuild, and this one is binding ==="
  pf || { log "ABORT: rebuilt subset STILL does not cover the draw -- refusing a dose-confounded arm"; exit 1; }
  log "preflight PASSED after rebuild"
fi
ls -la /marimo/items_llm_subset.parquet

# ---- 5. resume -------------------------------------------------------------
CMD="python train_ce.py --tag t176full --model $P --resume-partial $P --stage-a 1 --stage-b 2 --max-len 1024 --train-seed 0 --group-by-length 50 --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO --num-workers $WORKERS --ckpt-every 1000"
case "$CMD" in
  *"--llm-zero-scale 14.367"*"--num-workers 2"*)
      log "flag guard OK: full-pool draw AND the worker fix both present" ;;
  *)  log "FATAL: draw or worker flag missing from CMD"; exit 1 ;;
esac
log "=== t176full resumed from step 57000/87359, 2 dataloader workers ==="
$CMD > /marimo/t176full.log 2>&1
log "t176full rc=$?"
grep -aE "RESUME at step|llm subsample|MACRO=|done," /marimo/t176full.log | tail -8

# ---- 6. score whatever finished --------------------------------------------
PAIRS=""
for e in 0 1; do
  Dd=/marimo/ckpt-t176full-ep$e
  [ -s "$Dd/model.safetensors" ] && PAIRS="$PAIRS t176full-ep$e $Dd"
done
if [ -z "$PAIRS" ]; then
  log "no t176full epoch checkpoint to score"
else
  log "scoring:$PAIRS"
  python score_ckpt.py --max-len 1024 $PAIRS > /marimo/score_t176.log 2>&1
  log "score_ckpt rc=$?"
  grep -a "scored ->" /marimo/score_t176.log | tail -4
  mkdir -p /marimo/t176dumps
  cp -f /marimo/ce_scores_t176full-ep*.npz /marimo/t176dumps/ 2>/dev/null
  python push_ckpt.py /marimo/t176dumps ecup26-t176full-dumps \
    || log "NOTE: dump push failed -- pull them before the next death"
fi
echo QUEUE98_DONE
