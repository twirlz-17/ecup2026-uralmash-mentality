"""v26 = v25 with ONE thing changed: the inference batch sort key.

WHY. LEDGER `sortkey`. The shipped src/ce.py sorts pairs by CHARACTER length
and pads each batch to its longest member in TOKENS; characters are a noisy
proxy for tokens in a mixed-script catalogue (pearson r 0.9603), and the
residual is padding the GPU processes for nothing -- 41.6% of padded tokens at
the 1024 window, ~30% of the CE's wall clock. The replacement tokenizes ONCE
up front, sorts by true token length, and has producer threads only pad -- so
the 'Already borrowed' race that cerace closed with a warmup cannot occur at
all (one tokenizer call, one thread).

THE GATE THAT CLEARED IT (2026-08-25, kernel ecup26-sortinv, T4): the champion
scored the same 73,131-pair split under token-sorted and completely UNSORTED
batch compositions -- MACRO 0.80296 vs 0.80296, delta +0.00000, every category
+0.0000. Char-sort is itself a composition between those two, so the macro is
invariant to this change a fortiori. Timing side-product on the same card:
13.4 min sorted vs 40.1 min unsorted, the ~3x the padding model predicts.

WHY THE 2-KEY BATCHES ARE SAFE. The patched batches carry input_ids and
attention_mask only. Verified against the archive itself: models/ce-2 is
XLMRobertaTokenizer (class default model_input_names = input_ids,
attention_mask; XLM-R has no token_type_ids) and models/ce-e5-base declares
model_input_names = [input_ids, attention_mask] in its tokenizer_config.json.
This script re-asserts both at build time and refuses to build otherwise.

BUILD SHAPE, same discipline as build_v25_charcap.py: open the archive the
board scored at 0.5213854, rewrite exactly one file (src/ce.py), and inside it
replace exactly one function (_score) with the token-sort implementation from
submission/patches/ce_sortkey.py -- renamed, with the shipped thread-join tail
preserved. Every other entry is CRC-verified on the way through, the entry
list must come out identical, the new ce.py must compile, and reverting the
function must reproduce v25's ce.py byte-exactly or the script refuses to
write.

    python submission/build_v26_sortkey.py
"""
import argparse
import hashlib
import json
import pathlib
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
CE = "src/ce.py"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def patched_score_source():
    """_score_sorted from patches/ce_sortkey.py, renamed to _score, with the
    shipped join+log tail restored before the return."""
    src = (HERE / "patches" / "ce_sortkey.py").read_text(encoding="utf-8")
    start = src.index("def _score_sorted(")
    end = src.index("def verify_sortkey(")
    fn = src[start:end].rstrip() + "\n"
    fn = fn.replace("def _score_sorted(", "def _score(", 1)
    old_tail = """            except queue.Empty:
                pass
    return scores, complete
"""
    new_tail = """            except queue.Empty:
                pass
    for th in threads:
        th.join(timeout=5)
    log("ce: scored %d/%d complete=%s" % (done, n, complete))
    return scores, complete
"""
    if fn.count(old_tail) != 1:
        raise SystemExit("patch tail anchor found %d times -- ce_sortkey.py "
                         "has drifted, read it before building" % fn.count(old_tail))
    return fn.replace(old_tail, new_tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "dist" / "submission_v25.zip"))
    ap.add_argument("--out", default=str(HERE / "dist" / "submission_v26.zip"))
    a = ap.parse_args()
    src, out = pathlib.Path(a.base), pathlib.Path(a.out)
    if not src.exists():
        raise SystemExit("base archive missing: %s" % src)
    if out.exists():
        raise SystemExit("refusing to overwrite %s -- pick another --out" % out)

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        ce_old = z.read(CE).decode("utf-8")
        # the 2-key-batch safety assertion, from the archive's own configs
        for slot, want in (("models/ce-2/tokenizer_config.json",
                            ("XLMRobertaTokenizer", None)),
                           ("models/ce-e5-base/tokenizer_config.json",
                            (None, ["input_ids", "attention_mask"]))):
            cfg = json.loads(z.read(slot))
            klass, minames = want
            if klass is not None and cfg.get("tokenizer_class") != klass:
                raise SystemExit("%s class drifted: %s" % (slot, cfg.get("tokenizer_class")))
            if minames is not None and cfg.get("model_input_names") != minames:
                raise SystemExit("%s model_input_names drifted: %s"
                                 % (slot, cfg.get("model_input_names")))
            if "token_type" in json.dumps(cfg.get("model_input_names")):
                raise SystemExit("%s wants token_type_ids -- the patch cannot ship" % slot)

    i0 = ce_old.index("def _score(")
    i1 = ce_old.index("def ce_scores(")
    old_fn = ce_old[i0:i1]
    new_fn = patched_score_source() + "\n\n"
    ce_new = ce_old[:i0] + new_fn + ce_old[i1:]

    compile(ce_new, CE, "exec")
    if ce_new.replace(new_fn, old_fn) != ce_old:
        raise SystemExit("the edit is not revertible -- it changed more than _score")
    for kept in ('"TEXT_CHAR_CAP", "2000"', "def ce_scores(", "_PAIR_TEXT_CACHE"):
        if kept not in ce_new:
            raise SystemExit("lost a load-bearing piece of ce.py: %s" % kept)
    ce_new_b = ce_new.encode("utf-8")

    changed, copied = [], 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for info in zin.infolist():
            if info.filename == CE:
                zout.writestr(info, ce_new_b)
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
        if z.read(CE) != ce_new_b:
            raise SystemExit("src/ce.py did not survive the write")
        if b"sorted by TOKEN length" not in z.read(CE):
            raise SystemExit("the token-sort marker is not in the shipped file")
    if changed != [CE]:
        raise SystemExit("changed the wrong set of entries: %s" % changed)

    print("base    %s" % src)
    print("out     %s  (%.2f GB)" % (out, out.stat().st_size / 2 ** 30))
    print("src/ce.py sha256 %s -> %s"
          % (sha(ce_old.encode())[:16], sha(ce_new_b)[:16]))
    print("changed 1 entry (%s), copied %d unchanged with CRCs verified" % (CE, copied))
    print()
    print("ONE VARIABLE: _score's batch sort key, chars -> true tokens.")
    print("Macro-invariance PASSED bit-exact on the champion (ecup26-sortinv);")
    print("-41.6%% padded tokens at 1024, ~-30%% CE wall clock. Every other byte")
    print("is the archive the board scored at 0.5213854.")


if __name__ == "__main__":
    main()
