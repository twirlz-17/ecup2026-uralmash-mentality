"""Add a FOURTH cross-encoder, cascaded, on top of a v46-class archive.

    python tools/build_ce4.py --base submission/dist/submission_v46.zip \
        --out submission/dist/submission_v47.zip --ce4 models_local/t167llm2x-ep1

WHY. v46 = 0.5414129346430998 is champion, +0.0082 over v45, and the CE-3 stage
COMPLETED inside the grader's budget -- so a further stage is not obviously
unaffordable. On the v46 blend, `stack4` measured a fourth member as the only
lever left with real size: CE3_COVER is exhausted (+0.00059 at 0.50, +0.00063 at
0.70) and w3 is noise (SELECT and REPORT disagree on its direction). The best
fourth member on the honest chooser is t167llm2x-ep1: SELECT +0.00452, REPORT
+0.00312. It is also the CE-1 that scored 0.5230454881 on this board as v39, so
it is a known-good checkpoint loading through the SAME mmBERT code path CE-1
already uses.

MAX_LEN_4 IS 1024, NOT 256. t167llm2x-ep1 is mmBERT trained at 1024;
`t109probe` measured -0.067 at 384 and -0.090 at 512 for a checkpoint run off
its own window. That makes CE-4 the EXPENSIVE stage -- roughly 2.3x CE-3's cost
at equal coverage (q15timing: mmBERT@1024 9.44 vs bge@256 4.06 CE-min) -- which
is exactly why it sits behind the same real budget guard.

THE GUARD IS THE ONE THAT ACTUALLY FIRES. It reads T0 + total_budget directly
and never touches deadline_ts, because FORCE_CE=1 sets deadline_ts=None on every
scored stage and would make the check inert (`watchdogkills`). Verified by
tools/sim_ce3.py on the CE-3 stage: SKIP, RAISES and NO-DIR all leave the score
bit-identical to the previous champion. Same structure here, so the worst case
of this archive is exactly v46.
"""
import argparse
import hashlib
import json
import os
import py_compile
import shutil
import sys
import tempfile
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANCHOR = (
    '                                log(f"CE-3 failed ({exc3!r}), two-CE kept")\n'
    '                                _alarm(0)\n'
)

CE4 = ANCHOR + (
    '                    # ---- FOURTH cross-encoder, cascaded on the 3-CE blend ----\n'
    '                    # `stack4` on the v46 champion: SELECT +0.00452, REPORT\n'
    '                    # +0.00312, the best of seven candidates and the only\n'
    '                    # lever left with size (CE3_COVER is exhausted at\n'
    '                    # +0.0006, w3 is noise). This checkpoint IS v39\'s CE-1,\n'
    '                    # which scored 0.5230454881, so it is board-proven and\n'
    '                    # loads through the same mmBERT path as CE-1.\n'
    '                    # MAX_LEN_4=1024 is its training window, not a choice.\n'
    '                    ce4_dir = os.path.join(HERE, "models/ce-4")\n'
    '                    if os.environ.get("SKIP_CE4", "") == "1":\n'
    '                        log("CE-4: skipped (SKIP_CE4=1)")\n'
    '                    elif not os.path.isdir(ce4_dir):\n'
    '                        log("CE-4: absent, three-CE behaviour")\n'
    '                    else:\n'
    '                        _cov4 = float(os.environ.get("CE4_COVER", "0.20"))\n'
    '                        # mmBERT@1024 costs ~1.0x CE-1 per pair, so no 0.43\n'
    '                        # discount here -- CE-4 is the expensive stage.\n'
    '                        _need4 = _ce1_secs * 1.0 * 1.6 * _cov4\n'
    '                        # deadline_ts is None under FORCE_CE=1; read the\n'
    '                        # wall clock so this guard can actually fire.\n'
    '                        _left4 = T0 + total_budget * 0.88 - time.time()\n'
    '                        if _left4 < _need4:\n'
    '                            log(f"CE-4: SKIP, need ~{_need4:.0f}s, "\n'
    '                                f"{_left4:.0f}s left")\n'
    '                        else:\n'
    '                            try:\n'
    '                                _alarm(max(10, int(T0 + total_budget * 0.97\n'
    '                                                   - time.time())))\n'
    '                                _k4 = max(1, int(round(_cov4 * len(ce))))\n'
    '                                _sel4 = np.argsort(-ce, kind="stable")[:_k4]\n'
    '                                log(f"CE-4: cascade to top {_k4}/{len(ce)} "\n'
    '                                    f"({_cov4:.0%}) of the 3-CE ranking")\n'
    '                                ce4_raw, complete4 = ce_scores(\n'
    '                                    args.items_path, args.matches_path, ce4_dir,\n'
    '                                    batch_size=int(os.environ.get(\n'
    '                                        "BATCH_SIZE_4", "256")),\n'
    '                                    max_len=int(os.environ.get(\n'
    '                                        "MAX_LEN_4", "1024")),\n'
    '                                    deadline_ts=deadline_ts, log=log,\n'
    '                                    subset=_sel4)\n'
    '                                _alarm(0)\n'
    '                                if complete4 and np.isfinite(\n'
    '                                        ce4_raw[_sel4]).all():\n'
    '                                    _w4 = float(os.environ.get("W_CE4", "0.2"))\n'
    '                                    ce = ((1.0 - _w4) * ce\n'
    '                                          + _w4 * _band_rank(ce4_raw, _sel4,\n'
    '                                                             len(ce)))\n'
    '                                    log(f"CE-4 ok, w4={_w4}")\n'
    '                                else:\n'
    '                                    log("CE-4 incomplete, three-CE result kept")\n'
    '                            except Exception as exc4:\n'
    '                                import traceback\n'
    '                                traceback.print_exc()\n'
    '                                log(f"CE-4 failed ({exc4!r}), three-CE kept")\n'
    '                                _alarm(0)\n'
)


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ce4", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        raise SystemExit("REFUSING: %s exists (never reuse a version number)" % a.out)

    need = ["config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json"]
    for f in need:
        if not os.path.exists(os.path.join(a.ce4, f)):
            raise SystemExit("ABORT: %s missing from --ce4" % f)
    cfg = json.load(open(os.path.join(a.ce4, "config.json"), encoding="utf-8"))
    print("CE-4: %s, hidden %s, %s layers, %.2f GB"
          % (cfg.get("model_type"), cfg.get("hidden_size"),
             cfg.get("num_hidden_layers"),
             os.path.getsize(os.path.join(a.ce4, "model.safetensors")) / 1e9))

    zin = zipfile.ZipFile(a.base)
    names = zin.namelist()
    base_crc = {n: zin.getinfo(n).CRC for n in names}
    src = zin.read("run.py").decode("utf-8")
    nl = "\r\n" if "\r\n" in src else "\n"
    anchor, repl = ANCHOR.replace("\n", nl), CE4.replace("\n", nl)
    if anchor not in src:
        raise SystemExit("ABORT: CE-3 tail anchor not found -- is --base a v46-class archive?")
    if "models/ce-4" in src:
        raise SystemExit("ABORT: base already references a CE-4")
    new_run = src.replace(anchor, repl).encode("utf-8")

    t = os.path.join(tempfile.gettempdir(), "ce4_run_check.py")
    open(t, "wb").write(new_run)
    py_compile.compile(t, doraise=True)
    print("patched run.py compiles (line ending %r)" % nl)

    tmp = a.out + ".tmp"
    added = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zout:
        for it in zin.infolist():
            data = new_run if it.filename == "run.py" else zin.read(it.filename)
            zout.writestr(it, data)
        for f in need + ["special_tokens_map.json"]:
            p = os.path.join(a.ce4, f)
            if os.path.exists(p):
                arc = "models/ce-4/" + f
                zout.write(p, arc, zipfile.ZIP_STORED)
                added.append(arc)
    shutil.move(tmp, a.out)

    zo = zipfile.ZipFile(a.out)
    diff = [n for n in names if base_crc[n] != zo.getinfo(n).CRC]
    blob = open(a.out, "rb").read()
    print("\nwrote %s: %d B  sha256 %s" % (a.out, len(blob), sha16(blob)))
    print("changed existing entries: %s" % diff)
    print("added entries: %s" % added)
    if diff != ["run.py"]:
        raise SystemExit("ABORT: run.py must be the ONLY modified entry, got %s" % diff)
    print("VERIFIED: %s + a CE-4 stage; every other entry byte-identical"
          % os.path.basename(a.base))


if __name__ == "__main__":
    main()
