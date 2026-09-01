"""v25 = v24 with ONE thing changed: the inference-time character cap.

WHY. LEDGER `charcap`. src/ce.py truncates each side to TEXT_CHAR_CAP=900
CHARACTERS before tokenizing -- a leftover from the 256-token era, whose own
comment says "max_len=256 for the PAIR means no side survives past ~625 chars of
Russian". The window has been 1024 since v24 and the cap was never revisited.
box/train_ce.py has never applied it, and neither has any local ruler, so the
board has been reading a shorter text than every number we own was measured on.

MEASURED 2026-08-25, both arms on one Kaggle T4 in fp16 at batch 128, same
checkpoint, same split, the cap the only difference:

    cap  900   0.79969
    cap 2000   0.80296        +0.00327   against a 0.0007 run-to-run null

The uncapped arm reproduces the champion's box-measured 0.8028 to within
0.00016 -- a T4 in fp16 agreeing with an RTX PRO 6000 in bf16 -- which is what
licenses reading the difference as the cap rather than as the hardware. The gain
lands where the mechanism predicts: Ювелирные изделия +0.0467, fourteen times
the macro average, and jewellery carries the longest attribute text in the
catalogue.

2000 is chosen because no item in the catalogue exceeds it (11.40% exceed 900,
2.06% exceed 1500, 0% exceed 2000), so it is "off" rather than "looser" -- one
threshold to defend instead of two.

COST. Uncapped text is ~7% more tokens. I first wrote this file gating the
change on the padding fix, on the grounds that +7% pushes v24 from 806s to 863s
against a 756s watchdog. The human corrected two errors in that, and both are
load-bearing:

  * 806s is the H100-EQUALS-OUR-CARD scenario, which AUDIT A2 treats as a floor,
    not an estimate -- the grader H100 is ~2x our card's bf16 compute, putting
    the real figure near 430s of a 780s budget.
  * There is no longer any time-based decline path to trip. v22 deleted the
    silent gate, and on every SCORED stage force_ce is on, so run.py sets
    deadline_ts=None and ce.py's in-flight check (`if deadline_ts is not None`)
    never runs. The cross-encoder goes to completion. What remains is a hard
    SIGALRM at 0.97x budget whose failure is LOUD by design, and it has never
    fired: no submission of ours has ever landed near the ~0.29 GBDT-only
    fingerprint.

So the padding fix is worth shipping on its own merits and is not a
precondition for this one. Gating a confirmed +0.00327 on a risk that has never
materialised is exactly `gates-that-only-kill-lost-gold`.

BUILD SHAPE. This does not rebuild anything. It opens the archive the board
SCORED at 0.519802 and rewrites one file, changing two identical string literals
inside it -- the cap itself and the text-cache key that must agree with it.
Every other entry is copied with its CRC verified on the way through, the entry
list must come out identical, and the diff of src/ce.py must be exactly those
two substitutions or the script refuses to write.

    python submission/build_v25_charcap.py
"""
import argparse
import hashlib
import pathlib
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
CE = "src/ce.py"
OLD = 'os.environ.get("TEXT_CHAR_CAP", "900")'
NEW = 'os.environ.get("TEXT_CHAR_CAP", "2000")'
N_EXPECTED = 2          # the cap itself, and the pair-text cache key


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "dist" / "submission_v24.zip"))
    ap.add_argument("--out", default=str(HERE / "dist" / "submission_v25.zip"))
    a = ap.parse_args()
    src, out = pathlib.Path(a.base), pathlib.Path(a.out)
    if not src.exists():
        raise SystemExit("base archive missing: %s" % src)
    if out.exists():
        raise SystemExit("refusing to overwrite %s -- pick another --out" % out)

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        ce_old = z.read(CE)
    if CE not in names:
        raise SystemExit("base archive has no %s -- wrong base?" % CE)

    text = ce_old.decode("utf-8")
    n = text.count(OLD)
    if n != N_EXPECTED:
        raise SystemExit("found %d occurrences of the cap default, expected %d -- "
                         "ce.py has drifted, read it before patching" % (n, N_EXPECTED))
    ce_new = text.replace(OLD, NEW).encode("utf-8")

    # The edit must be EXACTLY those substitutions and nothing else.
    if ce_new.decode("utf-8").replace(NEW, OLD) != text:
        raise SystemExit("the patch is not reversible -- it changed more than the cap")

    changed, copied = [], 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for info in zin.infolist():
            if info.filename == CE:
                zout.writestr(info, ce_new)
                changed.append(info.filename)
                continue
            zout.writestr(info, zin.read(info.filename))   # read() verifies the CRC
            copied += 1

    with zipfile.ZipFile(out) as z:
        if z.namelist() != names:
            raise SystemExit("entry list changed -- refusing to ship")
        bad = z.testzip()
        if bad:
            raise SystemExit("corrupt entry in the new archive: %s" % bad)
        back = z.read(CE)
        if back != ce_new:
            raise SystemExit("src/ce.py did not survive the write")
        if b'"TEXT_CHAR_CAP", "2000"' not in back:
            raise SystemExit("the new cap is not in the shipped file")
    if changed != [CE]:
        raise SystemExit("changed the wrong set of entries: %s" % changed)

    print("base    %s" % src)
    print("out     %s  (%.2f GB)" % (out, out.stat().st_size / 2 ** 30))
    print("src/ce.py sha256 %s -> %s" % (sha(ce_old)[:16], sha(ce_new)[:16]))
    print("changed %d entry (%s), copied %d unchanged with CRCs verified"
          % (len(changed), CE, copied))
    print()
    print("ONE VARIABLE: TEXT_CHAR_CAP 900 -> 2000, worth +0.00327 human-val")
    print("measured on a held split. run.py, both GBDT sets, both cross-encoder")
    print("slots and both tokenizers are the bytes the board scored at 0.519802.")


if __name__ == "__main__":
    main()
