"""Build a submission that ADDS a third cross-encoder, cascaded, to v39.

    python tools/build_ce3.py --base submission/dist/submission_v39.zip \
        --out submission/dist/submission_v42.zip --ce3 models_local/alexbge-fp16

WHAT THIS SHIPS AND WHY. `ce2worth` measured the CE-2 slot at +0.01434 inside
the cascade and showed nine of ten owned checkpoints LOSE when swapped into it,
so the slot is not improvable by substitution. `alexboot` then measured the two
shippable shapes for Alexander's Stage-B bge: REPLACING CE-2 is +0.00161 with a
90% CI through zero (11/20 categories, P=0.752 -- not real), while ADDING it as
a third member at cov 0.30 with w3=0.25 is +0.00565, 90% CI [+0.00169,
+0.00907], 15/20 categories, P(delta>0)=0.988. So it enters as CE-3, not as a
replacement. cov3 and w3 were chosen on the SELECT half and quoted on REPORT
(`ce3tune`); cov3 is held at CE-2's proven 0.30 rather than SELECT's preferred
0.50 because coverage costs inference seconds and w3 does not.

THE CASCADE MIRRORS CE-2 EXACTLY. CE-3 re-scores only the top CE3_COVER of the
CE-1+CE-2 BLENDED ranking, at its own 256 window (its tokenizer_config pins
max_length 256, and `t109probe` measured -0.067 for running a checkpoint off its
training window). It is band-ranked with the SAME _band_rank the shipped
container already uses, so no new ranking semantics are introduced. 90% of its
full-coverage value lands by cov 0.30, which is why 0.30 and not more.

FAILS SAFE, WHICH IS THE POINT. The new stage sits behind the same deadline
arithmetic CE-2 uses: if the remaining budget is short it logs a SKIP and the
container produces EXACTLY the v39 result. An absent models/ce-3 does the same,
and any exception is caught and the two-CE result kept. So the downside is
bounded at parity rather than at the v37 class of failure (0.3611536 from a
container shipped unrun) -- and this one WILL be run in the grader image before
it is offered.

ONE VARIABLE. Nothing else changes: CE-1 weights, CE-2 weights, W_CE=0.7,
CE_COVER=0.30, W_GBDT=0.1 and the rope setting are all left exactly as the
0.5230454881 archive ships them. Proven entry-by-entry by CRC, not asserted.
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
    '                    if _parts:\n'
    '                        w_ce = float(os.environ.get("W_CE", "0.7"))\n'
    '                        ce = w_ce * _rank01(ce) + (1.0 - w_ce) * _parts[0]\n'
    '                        log(f"CE ensemble: cascaded partner, w_ce={w_ce}")\n'
    '                    else:\n'
    '                        log("CE ensemble: partner missing, CE-1 alone")\n'
)

CE3 = ANCHOR + (
    '                    # ---- THIRD cross-encoder, cascaded over the BLEND ----\n'
    '                    # `alexboot`: as a CE-2 REPLACEMENT his bge is +0.00161\n'
    '                    # with a CI through zero (11/20, P=0.752). ADDED here it\n'
    '                    # is +0.00548, 90% CI [+0.00215, +0.00835], 15/20\n'
    '                    # categories, P(delta>0)=1.000. It re-scores only the\n'
    '                    # top CE3_COVER of the CE-1+CE-2 BLENDED ranking (90% of\n'
    '                    # its full-coverage value lands by 0.30) at its own 256\n'
    '                    # window. Same _band_rank as CE-2. On a short budget it\n'
    '                    # SKIPS and the container returns exactly the v39 result.\n'
    '                    ce3_dir = os.path.join(HERE, "models/ce-3")\n'
    '                    if os.environ.get("SKIP_CE3", "") == "1":\n'
    '                        log("CE-3: skipped (SKIP_CE3=1)")\n'
    '                    elif not os.path.isdir(ce3_dir):\n'
    '                        log("CE-3: absent, two-CE behaviour")\n'
    '                    else:\n'
    '                        _cov3 = float(os.environ.get("CE3_COVER", "0.30"))\n'
    '                        _need3 = _ce1_secs * 0.43 * 1.6 * _cov3\n'
    '                        # DO NOT use deadline_ts here. FORCE_CE defaults to\n'
    '                        # 1 on every scored stage, which sets deadline_ts =\n'
    '                        # None, which makes CE-2\'s identical check INERT --\n'
    '                        # it can never skip. The wall clock and the stage\n'
    '                        # budget are known unconditionally, so this guard\n'
    '                        # reads them directly and therefore actually fires.\n'
    '                        # Reserve 12% of the budget for the blend + write.\n'
    '                        # Worst case is then a SKIP, i.e. exactly the v39\n'
    '                        # two-CE result -- not the global watchdog, which\n'
    '                        # discards EVERY CE score and ships the GBDT alone\n'
    '                        # (`watchdogkills`).\n'
    '                        _left3 = T0 + total_budget * 0.88 - time.time()\n'
    '                        if _left3 < _need3:\n'
    '                            log(f"CE-3: SKIP, need ~{_need3:.0f}s, "\n'
    '                                f"{_left3:.0f}s left")\n'
    '                        else:\n'
    '                            try:\n'
    '                                _alarm(max(10, int(T0 + total_budget * 0.97\n'
    '                                                   - time.time())))\n'
    '                                _k3 = max(1, int(round(_cov3 * len(ce))))\n'
    '                                _sel3 = np.argsort(-ce, kind="stable")[:_k3]\n'
    '                                log(f"CE-3: cascade to top {_k3}/{len(ce)} "\n'
    '                                    f"({_cov3:.0%}) of the BLENDED ranking")\n'
    '                                ce3_raw, complete3 = ce_scores(\n'
    '                                    args.items_path, args.matches_path, ce3_dir,\n'
    '                                    batch_size=int(os.environ.get(\n'
    '                                        "BATCH_SIZE_3", "512")),\n'
    '                                    max_len=int(os.environ.get(\n'
    '                                        "MAX_LEN_3", "256")),\n'
    '                                    deadline_ts=deadline_ts, log=log,\n'
    '                                    subset=_sel3)\n'
    '                                _alarm(0)\n'
    '                                if complete3 and np.isfinite(\n'
    '                                        ce3_raw[_sel3]).all():\n'
    '                                    _w3 = float(os.environ.get("W_CE3", "0.25"))\n'
    '                                    ce = ((1.0 - _w3) * ce\n'
    '                                          + _w3 * _band_rank(ce3_raw, _sel3,\n'
    '                                                             len(ce)))\n'
    '                                    log(f"CE-3 ok, w3={_w3}")\n'
    '                                else:\n'
    '                                    log("CE-3 incomplete, two-CE result kept")\n'
    '                            except Exception as exc3:\n'
    '                                import traceback\n'
    '                                traceback.print_exc()\n'
    '                                log(f"CE-3 failed ({exc3!r}), two-CE kept")\n'
    '                                _alarm(0)\n'
)


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ce3", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        raise SystemExit("REFUSING: %s exists (never reuse a version number)" % a.out)

    need = ["config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json"]
    for f in need:
        p = os.path.join(a.ce3, f)
        if not os.path.exists(p):
            raise SystemExit("ABORT: %s missing from --ce3" % p)
    cfg = json.load(open(os.path.join(a.ce3, "config.json"), encoding="utf-8"))
    if cfg.get("model_type") != "xlm-roberta":
        raise SystemExit("ABORT: CE-3 is %r, expected xlm-roberta"
                         % cfg.get("model_type"))
    print("CE-3 verified: %s, hidden %s, %s layers, %.2f GB"
          % (cfg["model_type"], cfg["hidden_size"], cfg["num_hidden_layers"],
             os.path.getsize(os.path.join(a.ce3, "model.safetensors")) / 1e9))

    zin = zipfile.ZipFile(a.base)
    names = zin.namelist()
    # snapshot BEFORE writing: writestr MUTATES the ZipInfo it is handed, and
    # these come from zin.infolist(), so a later comparison would see no change.
    base_crc = {n: zin.getinfo(n).CRC for n in names}
    src = zin.read("run.py").decode("utf-8")
    # The SHIPPED run.py is CRLF. Matching it with LF anchors silently finds
    # nothing, which the stand-in dry run caught before it could waste the real
    # build. Match and insert in the file's OWN line ending so the diff stays
    # surgical instead of rewriting every line's terminator.
    nl = "\r\n" if "\r\n" in src else "\n"
    anchor = ANCHOR.replace("\n", nl)
    repl = CE3.replace("\n", nl)
    if anchor not in src:
        raise SystemExit("ABORT: could not find the CE blend anchor in run.py")
    if "models/ce-3" in src:
        raise SystemExit("ABORT: base already references a CE-3")
    print("run.py line ending: %r" % nl)
    new_run = src.replace(anchor, repl).encode("utf-8")

    t = os.path.join(tempfile.gettempdir(), "ce3_run_check.py")
    open(t, "wb").write(new_run)
    py_compile.compile(t, doraise=True)
    print("patched run.py compiles")

    tmp = a.out + ".tmp"
    added = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zout:
        for it in zin.infolist():
            data = new_run if it.filename == "run.py" else zin.read(it.filename)
            zout.writestr(it, data)
        for f in need + ["special_tokens_map.json"]:
            p = os.path.join(a.ce3, f)
            if os.path.exists(p):
                arc = "models/ce-3/" + f
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
        raise SystemExit("ABORT: expected run.py to be the ONLY modified entry, "
                         "got %s" % diff)
    print("VERIFIED: v39 + a CE-3 stage; every other entry byte-identical")


if __name__ == "__main__":
    main()
