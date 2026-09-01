#!/bin/bash
# grader_run.sh -- run a submission archive inside the REAL grader image.
#
#   bash tools/grader_run.sh <archive.zip> <workdir> [extra docker env...]
#
# WHY THIS IS NOT OPTIONAL FOR A run.py CHANGE. v37 scored 0.3611536 because a
# container was shipped unrun. SUBMISSIONS.md records that neither the v39 nor
# the v40 archive was ever executed in the grader image. v41 was the first one
# that was, and that run PROVED the full path executes -- CE-1 bf16, CE-2
# cascade on the top 30%, w_ce blend, done inside the stage budget.
#
# WHAT IT IS AND IS NOT EVIDENCE FOR. `v41lb` is emphatic: the macro AP this
# harness computes on 11,000 labelled pairs was SIGN-WRONG against the board
# (+0.00419 predicted, -0.00111 delivered) because ~59 positives per category
# is not enough. So the AP printed here is DESCRIPTIVE ONLY. What the run is
# trustworthy for is the thing it actually observes: does the container load,
# does every stage engage, does it finish inside the budget, does anything
# throw. For a change that edits run.py, that is exactly the question.
set -o pipefail
# Git-bash rewrites container-side absolute paths: `-w /app` reached docker as
# 'C:/Program Files/Git/app' and the daemon rejected it. Every path after this
# point that belongs to the CONTAINER must survive verbatim.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
ZIP="$1"; WORK="$2"; shift 2
[ -s "$ZIP" ] || { echo "FATAL: no archive at $ZIP"; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D="$REPO/outputs/graderepro"
for f in items.parquet matches.parquet; do
  [ -s "$D/$f" ] || { echo "FATAL: missing $D/$f"; exit 1; }
done

rm -rf "$WORK"; mkdir -p "$WORK/sub" "$WORK/data" "$WORK/out"
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$WORK/sub"
cp "$D/items.parquet" "$D/matches.parquet" "$WORK/data/"
echo "== extracted $(du -sh "$WORK/sub" | cut -f1); models: $(ls "$WORK/sub/models" 2>/dev/null | tr '\n' ' ')"

docker run --rm --gpus all \
  -v "$WORK/sub:/app" -v "$WORK/data:/data" -v "$WORK/out:/out" \
  -w /app "$@" \
  twirlz/ecup26-matching:1.0 \
  python -u run.py --items_path /data/items.parquet \
                   --matches_path /data/matches.parquet \
                   --output_path /out/submit.csv 2>&1 | tee "$WORK/run.log"
RC=$?
echo "== docker rc=$RC"
echo "== stage log =="
grep -aE "CE-1|CE-2|CE-3|ensemble|blend|wrote|done|SKIP|ERROR|Traceback|failed" "$WORK/run.log" | tail -25
[ -s "$WORK/out/submit.csv" ] || { echo "FATAL: no submit.csv produced"; exit 1; }
echo "== rows: $(python -c "import csv,sys;print(sum(1 for _ in open(sys.argv[1]))-1)" "$WORK/out/submit.csv")"
python "$REPO/tools/score_graderepro.py" "$WORK/out/submit.csv" 2>&1 | tail -6
