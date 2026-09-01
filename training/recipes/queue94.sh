#!/bin/bash
# t176full -- THE ENTIRE LLM POOL AT ITS NATURAL MIX. 11,182,000 of the
# 11,187,780 available pairs (99.9%), on the champion backbone, window and
# recipe.
#
# WHY THIS AND NOT A BACKBONE SWAP. mmbertread pre-registered a length stratum
# to split v24's +0.02255 into its two moving parts, and the split came out
# lopsided: the BACKBONE was worth +0.00229, the rest was the WINDOW. That row's
# scope line reads "kills mmBERT AS A BACKBONE REPLACEMENT for the CE-1 slot".
# The window is now spent -- windowsat measures 0.66% of pairs truncating at
# 1024 against 61.9% at 256 -- so another backbone at the same window buys about
# +0.002. mmbertbench adds that mmBERT is 5% SLOWER than bge at equal window;
# the advertised 2-4x does not apply at our 340-token average. Capacity is not
# the lever it looked like, and bge in particular is mined by all three of us.
#
# WHAT THE LEDGER SAYS IS THE LEVER, in stageavol's own words: "SO THE
# INTERESTING AXIS IS MIX, NOT VOLUME." Stage A runs at 55% positive;
# matches_llm's own distribution is 23/12/64 and the board runs at ~4.5%. The
# ABUNDANT class is the under-used one -- we take 7.0% of the available zeros.
#
# AND MIX IS WHAT UNLOCKS VOLUME. The pure-volume ceiling is only 2.18x because
# POSITIVES are the scarce class and the sampler holds the 12:5:5 ratio; dose2x
# already spent 2.0x of that, so volume-at-fixed-mix is 91% exhausted. Drop the
# positive share to the pool's own and the whole 11.19M becomes reachable:
#     pos  1,200,000 x 2.180  = 2,616,000   (99.9% of the 2,619,567 available)
#     mid    500,000 x 2.765  = 1,382,500   (99.9% of 1,383,550)
#     zero   500,000 x 14.367 = 7,183,500   (100.0% of 7,184,663)
#                       total = 11,182,000  = 5.08x the champion's 2,200,000
#     positive share 54.5% -> 23.4%, toward the board's 4.5%
# The scales sit 0.002 UNDER the exact ratios on purpose: the sampler ASSERTS on
# an over-draw rather than silently shrinking, and an assert at minute 40 is a
# dead box life.
#
# TWO AXES MOVE AT ONCE AND THAT IS DELIBERATE, stated so nobody reads this as a
# controlled arm. shiponevariable is about SLOTS; this is a training arm whose
# whole hypothesis is that the two axes are COUPLED -- the mix change is what
# makes the volume reachable at all, so separating them means not testing the
# thing. If it wins, stageavol2x (2x at FIXED mix, still QUEUED) is the
# follow-up that splits them.
#
# READ: NO SYMMETRIC RULER, BUT A ONE-SIDED ONE, AND THAT IS ENOUGH TO ACT ON A
# WIN. mixaxisclosed (CLOSED-CONFIRMED) shows E_real gets the training-mix sign
# wrong at every prevalence. stageavol2x worked out WHY, and why the error is
# asymmetric: E_real's real slice IS matches.parquet, the HUMAN distribution, so
# it flatters whichever arm leans human and penalises whichever leans LLM. More
# Stage A leans LLM. The bias therefore points in a KNOWN direction:
#     E_real says BETTER -> credible; the gain had to beat a bias against it.
#     E_real says WORSE  -> UNINFORMATIVE; it decides nothing.
# PRE-REGISTERED HERE, BEFORE ANY NUMBER EXISTS: >= +0.006 E_real at the matched
# ep0 comparison against t149baseA-ep0 (which IS v39's predecessor v30) ->
# ACCEPT, and it earns a slot. Anything else -> NO DECISION, and the checkpoint
# is kept as a CE-1 artifact for the container comparison, which is artifact
# SELECTION and carries no mix-axis inference. E_mix and E_mined are DESCRIPTIVE
# only: eminediso showed E_mix's delta is its mined-negative component, and this
# arm changes negative composition more than any arm we have ever run.
#
# COST: about 5.08x the champion's Stage A. t174conf did 2.2M in 17,185 steps in
# roughly 70 minutes, so expect ~6 h Stage A plus ~1 h Stage B -- LONGER THAN
# ONE BOX LIFE. Survivable rather than reckless: --ckpt-every 1000 writes a
# rolling partial and push_ckpt ships Stage A to Kaggle the moment it completes,
# which is exactly what saved t175bal's 70-minute Stage A through box death #9.

set -o pipefail
cd /marimo || exit 1
log() { echo "[queue94] $(date -u +%H:%M:%S) $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POS=2.180
MID=2.765
ZERO=14.367

mkdir -p ~/.kaggle
[ -s /marimo/storage/access_token ] || { log "FATAL: no access_token"; exit 1; }
cp -f /marimo/storage/access_token ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN=$(cat /marimo/storage/access_token)
python -m kaggle --version >/dev/null 2>&1 || pip install -q kaggle

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

for f in /marimo/train_ce.py /marimo/build_llm_subset_box.py /marimo/preflight_draw.py \
         /marimo/score_ckpt.py /marimo/push_ckpt.py; do
  [ -s "$f" ] || { log "ABORT: missing $f"; exit 1; }
done
grep -q "llm-pos-scale" /marimo/train_ce.py             || { log "ABORT: trainer lacks the class knobs"; exit 1; }
grep -q "llm-pos-scale" /marimo/build_llm_subset_box.py || { log "ABORT: builder lacks the class knobs"; exit 1; }
log "tooling verified (push_ckpt present -- deathnine)"

while pgrep -af "train_ce.py --tag|score_ckpt.py" | grep -v t176full | grep -q .; do
  log "GPU busy: $(pgrep -af 'train_ce.py --tag|score_ckpt.py' | head -1 | cut -c1-90)"
  sleep 120
done
log "GPU free"

log "=== rebuilding items_llm_subset.parquet for the FULL-POOL draw ==="
python build_llm_subset_box.py --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO \
  > /marimo/build_subset_full.log 2>&1
RC=$?
log "builder rc=$RC"
tail -6 /marimo/build_subset_full.log
[ "$RC" = "0" ] || { log "ABORT: subset rebuild failed"; exit 1; }
ls -la /marimo/items_llm_subset.parquet

log "=== PREFLIGHT: item coverage of the draw the trainer will really make ==="
python /marimo/preflight_draw.py --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO \
  > /marimo/preflight_full.log 2>&1
log "preflight rc=$?"
cat /marimo/preflight_full.log
grep -q COVERAGE_OK /marimo/preflight_full.log || {
  log "ABORT: subset does not cover the draw -- refusing a dose-confounded arm"; exit 1; }
log "preflight PASSED: 100% item coverage"

CMD="python train_ce.py --tag t176full --model jhu-clsp/mmBERT-base --stage-a 1 --stage-b 2 --max-len 1024 --train-seed 0 --group-by-length 50 --llm-pos-scale $POS --llm-mid-scale $MID --llm-zero-scale $ZERO --ckpt-every 1000"
case "$CMD" in
  *"--llm-zero-scale 14.367"*) log "flag guard OK: full-pool draw present" ;;
  *) log "FATAL: zero-scale missing -- this would silently rerun the champion"; exit 1 ;;
esac
log "=== t176full: 11,182,000 LLM pairs, 23.4% positive, 5.08x champion dose ==="
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
echo QUEUE94_DONE
