"""Reconstruct the src/ce.py that the graded archive executed.

    python provenance/rebuild_ce.py [out.py]      # default: ../inference/src/ce.py

WHY THIS EXISTS. The graded archives were built as a chain of one-variable
patches over previous archives, and the local .zip copies were deleted after the
deadline. The repo's own working copy of ce.py is OLDER than what shipped -- it
predates three separate edits. Rather than hand-editing a file and asking anyone
to trust it, this replays the three build scripts that actually made it, on the
inputs they actually consumed, asserting at every anchor:

    inputs/ce_v21.py                     the byte-for-byte v21 archive copy
      + tokenizer warmup                 (inputs/build_v24.py, WARMUP)
      + TEXT_CHAR_CAP 900 -> 2000        (inputs/build_v25_charcap.py)
      + _score -> token-sorted _score     (inputs/build_v26_sortkey.py)
    = the ce.py in v26 ... v47, i.e. the champion's.

No build script after v26 touches src/ce.py (checked by grep over every
submission/build_*.py and tools/*.py in the working repo), and ce.py is
byte-identical across the v18 and v21 staged copies, so the base is stable.

NOTE ON STEP 1. The v24 warmup is inserted inside the OLD _score, which step 3
then replaces wholesale -- so it does not survive into the shipped file. That is
faithful to what the real v26 build did: the token-sorted _score closes the same
'Already borrowed' race by construction, with a single tokenizer call from a
single thread, so the warmup became unnecessary rather than lost.
"""
import hashlib
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
IN = HERE / "inputs"
TARGET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 \
    else HERE.parent / "inference" / "src" / "ce.py"

# --- step 0: the v21 base -------------------------------------------------
ce = (IN / "ce_v21.py").read_text(encoding="utf-8")

# --- step 1: v24's tokenizer warmup ---------------------------------------
spec = importlib.util.spec_from_file_location("b24", IN / "build_v24.py")
b24 = importlib.util.module_from_spec(spec)
sys.argv = ["b24"]
try:
    spec.loader.exec_module(b24)
except SystemExit:
    pass                      # the module runs argparse at import; we want WARMUP
anchor = '    n_prod = max(1, int(os.environ.get("TOKENIZER_THREADS", "4")))'
assert ce.count(anchor) == 1, "warmup anchor drift"
ce = ce.replace(anchor, b24.WARMUP + anchor, 1)

# --- step 2: v25's character cap ------------------------------------------
OLD = 'os.environ.get("TEXT_CHAR_CAP", "900")'
NEW = 'os.environ.get("TEXT_CHAR_CAP", "2000")'
assert ce.count(OLD) == 2, "cap occurrences != 2 (the cap itself + the cache key)"
ce = ce.replace(OLD, NEW)

# --- step 3: v26's token-sorted _score ------------------------------------
src = (IN / "ce_sortkey.py").read_text(encoding="utf-8")
fn = src[src.index("def _score_sorted("):src.index("def verify_sortkey(")].rstrip() + "\n"
fn = fn.replace("def _score_sorted(", "def _score(", 1)
old_tail = ("            except queue.Empty:\n                pass\n"
            "    return scores, complete\n")
new_tail = ("            except queue.Empty:\n                pass\n"
            "    for th in threads:\n        th.join(timeout=5)\n"
            '    log("ce: scored %d/%d complete=%s" % (done, n, complete))\n'
            "    return scores, complete\n")
assert fn.count(old_tail) == 1, "sortkey tail anchor drift"
fn = fn.replace(old_tail, new_tail)
i0, i1 = ce.index("def _score("), ce.index("def ce_scores(")
ce = ce[:i0] + fn + "\n\n" + ce[i1:]

# --- post-conditions the original build scripts also asserted -------------
compile(ce, "ce.py", "exec")
for kept in ('"TEXT_CHAR_CAP", "2000"', "def ce_scores(", "_PAIR_TEXT_CACHE",
             "subset=None"):
    assert kept in ce, "lost a load-bearing piece of ce.py: %s" % kept

b = ce.encode("utf-8")
TARGET.write_bytes(b)
print("wrote %s\n  %d bytes  sha256 %s"
      % (TARGET, len(b), hashlib.sha256(b).hexdigest()))
