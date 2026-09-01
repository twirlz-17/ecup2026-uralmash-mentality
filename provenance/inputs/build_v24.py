"""v24 -- the FULL mmBERT swap the human asked for: mmBERT@1024 becomes the
MAIN cross-encoder, the reigning bge CE-1 (t120pw0.134-ep1) is demoted to a
30%-cascade partner. e5-base leaves the container entirely.

MEASURED (tools/v24_mmswap.py, arm C, container shape, leak-free E_real):

    +0.01634 vs v22, 16/16 half-splits -- the largest container-shape number
    this project has ever measured.

PRICED HONESTLY, ALL THREE GRADER-SPEED ASSUMPTIONS (q15timing costs):
    13.44 raw min on our card ->  524s (H100 1.5x faster)   fits, minutes spare
                                  806s (H100 == our card)   watchdog at 756s
                                 1048s (fossil 1.30x)       watchdog
The 1.30x slow-grader constant is a fossil from the CPU-only-grader era; on
paper the H100 is ~2x our RTX PRO 6000's bf16 compute. If the watchdog does
fire during the PARTNER pass, the score degrades to mmBERT-alone (arm A,
+0.00645 local) -- still a local champion -- and only an mmBERT-pass overrun
drops to the loud GBDT-only ~0.29. The public run doubles as the grader-speed
measurement we cannot make locally.

WINDOW EXPOSURE IS THE POINT, NOT A BUG. `rulergrade` proved no local ruler
can price a window-crossing container -- this slot IS the measurement, by the
human's explicit call, and `windowrecord` established the board has never
actually judged a clean window change.

BUILD SHAPE (every ported piece is board-executed v21 code, as in v23):
  * src/ce.py <- ref_v21 (cascade subset) + the tokenizer-warmup race
    fix the smoke forced (see build_v23.py)
  * run.py: _band_rank + the cascaded CE-2 block; CE-1 env defaults move to
    MAX_LEN 1024 / BATCH_SIZE 256 (mmBERT's window, 80GB headroom); the
    partner runs at MAX_LEN_2 256 (bge's own window -- t109probe measured
    -0.067 for running a checkpoint at a window it was not trained for);
    need ratio 0.43 (bge@256 4.06 / mmBERT@1024 9.44), inert under FORCE_CE
  * models/ce-e5-base (CE-1 slot) <- mmBERT t126mmB-ep1, 4 files; the bge
    sentencepiece + special_tokens_map are REMOVED (different family)
  * models/ce-2 <- t120a weights+config, tokenizer set HARVESTED from the
    stage's own CE-1 slot BEFORE the mmBERT swap (that slot holds exactly the
    bge tokenizer that t120a needs)

    python submission/build_v24.py --ce1-dir models_local/t126mmB-ep1 \\
                                   --ce2-dir models_local/t120a
"""
import argparse
import hashlib
import pathlib
import shutil
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
V21_CE = HERE / "ref_v21" / "src__ce.py"
MM_FILES = ("config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json")
MM_BYTES = 615076194
BGE_BYTES = 1135559698
BGE_TOK = ("sentencepiece.bpe.model", "special_tokens_map.json",
           "tokenizer.json", "tokenizer_config.json")

WARMUP = "    # Fast-tokenizer thread-safety (found by the CPU smoke of THIS archive\n    # inside the shipping image): set_truncation_and_padding takes a MUTABLE\n    # borrow of the Rust tokenizer whenever the target settings differ from\n    # the current ones -- exactly the situation on the first wave of\n    # concurrent batches, where 4 producer threads collide and one dies with\n    # 'RuntimeError: Already borrowed'. In run.py that exception costs the\n    # whole cross-encoder. One synchronous warmup call pins the settings, so\n    # every later call takes the read-only path and the race is closed.\n    tokenize(0)\n\n"

BAND_RANK = '''
def _band_rank(raw, sel, n):
    """Rank a partner INSIDE the selected band, mapped onto the band's own
    slice of the global rank interval.

    The band is the top-|sel| of CE-1, so it owns global ranks [1-k/n, 1]. A
    partner that only scored the band has no global rank -- computing one would
    need the scores we deliberately skipped. Ranking within the band and mapping
    it onto the interval the band already occupies keeps the two regions
    ordered without a seam: off-band values are <= 1-k/n by construction and
    on-band values are >= 1-k/n.

    Positions off the band get 1-k/n exactly; the caller has already blended
    CE-1's own rank there, and the tail ordering is preserved because it is
    carried entirely by the w*r1 term.
    """
    from scipy.stats import rankdata

    k = len(sel)
    lo = 1.0 - k / float(n)
    out = np.full(n, lo, dtype=np.float64)
    if k > 1:
        out[sel] = lo + (1.0 - lo) * (rankdata(raw[sel]) / k)
    return out

'''

CASCADE_BLOCK = '''                    # ---- partner cross-encoder, CASCADED (v24, ----
                    # v24_mmswap arm C, cascade_impl). CE-1 is mmBERT@1024; the
                    # demoted bge champion re-scores only the top CE_COVER of
                    # mmBERT's ranking, at ITS OWN 256 window (t109probe:
                    # running a checkpoint at a window it was not trained for
                    # costs -0.067). Ranked INSIDE the band via _band_rank.
                    # This machinery ran on the board in v21.
                    _parts = []
                    ce2_dir = os.path.join(HERE, "models/ce-2")
                    if os.environ.get("SKIP_CE2", "") == "1":
                        log("CE-2: skipped (SKIP_CE2=1)")
                    elif not os.path.isdir(ce2_dir):
                        log("CE-2: absent, single-CE behaviour")
                    else:
                        # bge@256 costs ~0.43x mmBERT@1024 on identical pairs
                        # (q15timing: 4.06 vs 9.44 CE-min), 1.6x safety, scaled
                        # by the band. INERT under FORCE_CE (deadline_ts=None
                        # makes left=1e9); kept correct in case the gate is
                        # ever re-armed.
                        need = (_ce1_secs * 0.43 * 1.6
                                * float(os.environ.get("CE_COVER", "0.30")))
                        left = ((deadline_ts - time.time())
                                if deadline_ts else 1e9)
                        if left < need:
                            log(f"CE-2: SKIP, need ~{need:.0f}s, "
                                f"{left:.0f}s left (CE-1 took {_ce1_secs:.0f}s)")
                        else:
                            try:
                                _alarm(max(10, int(T0 + total_budget * 0.97
                                                   - time.time())))
                                _cov = float(os.environ.get("CE_COVER", "0.30"))
                                _k = max(1, int(round(_cov * len(ce))))
                                _sel2 = (np.argsort(-ce, kind="stable")[:_k]
                                         if _cov < 1.0 else
                                         np.arange(len(ce)))
                                log(f"CE-2: cascade to top {_k}/{len(ce)} "
                                    f"({_cov:.0%}) of the CE-1 ranking")
                                ce2_raw, complete2 = ce_scores(
                                    args.items_path, args.matches_path, ce2_dir,
                                    batch_size=int(os.environ.get(
                                        "BATCH_SIZE_2", "512")),
                                    max_len=int(os.environ.get(
                                        "MAX_LEN_2", "256")),
                                    deadline_ts=deadline_ts, log=log,
                                    subset=_sel2)
                                _alarm(0)
                                if complete2 and np.isfinite(
                                        ce2_raw[_sel2]).all():
                                    _parts.append(_band_rank(
                                        ce2_raw, _sel2, len(ce)))
                                    log(f"CE-2 ok (CE-1 {_ce1_secs:.0f}s)")
                                else:
                                    log("CE-2 incomplete, CE-1 alone")
                            except Exception as exc2:
                                import traceback
                                traceback.print_exc()
                                log(f"CE-2 failed ({exc2!r}), CE-1 alone")
                                _alarm(0)
                    if _parts:
                        w_ce = float(os.environ.get("W_CE", "0.7"))
                        ce = w_ce * _rank01(ce) + (1.0 - w_ce) * _parts[0]
                        log(f"CE ensemble: cascaded partner, w_ce={w_ce}")
                    else:
                        log("CE ensemble: partner missing, CE-1 alone")
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "dist" / "submission_v22.zip"))
    ap.add_argument("--out", default=str(HERE / "dist" / "submission_v24.zip"))
    ap.add_argument("--ce1-dir", required=True, help="t126mmB-ep1 (mmBERT@1024)")
    ap.add_argument("--ce2-dir", required=True,
                    help="t120a weights+config (bge, the demoted champion)")
    args = ap.parse_args()

    base = pathlib.Path(args.base)
    if not base.exists():
        raise SystemExit("base %s not found" % base)
    if not V21_CE.exists():
        raise SystemExit("ref_v21/src__ce.py missing")
    stage = HERE / "stage_v24"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    print("unpacking %s -> %s" % (base.name, stage.name))
    with zipfile.ZipFile(base) as z:
        z.extractall(stage)

    runp = stage / "run.py"
    run = runp.read_text(encoding="utf-8")
    for needed in ('force_ce = os.environ.get("FORCE_CE", "1") == "1" and n > 10_000',
                   "models/ce-2"):
        if needed not in run:
            raise SystemExit("base run.py lacks %r -- not the v22 archive" % needed)
    for forbidden in ("models/ce-3", "RUN_GBDT", "_band_rank"):
        if forbidden in run:
            raise SystemExit("base run.py already contains %s" % forbidden)

    # --- src/ce.py: board-executed v21 copy ---------------------------------
    ce_bytes = V21_CE.read_bytes()
    if b"subset=None" not in ce_bytes:
        raise SystemExit("ref ce.py lacks subset feature?!")
    (stage / "src" / "ce.py").write_bytes(ce_bytes)
    print("src/ce.py <- ref_v21, byte-for-byte (%d bytes)" % len(ce_bytes))

    # --- the one deliberate ce.py delta vs the board-executed v21 copy ------
    ce_path = stage / "src" / "ce.py"
    ce_txt = ce_path.read_text(encoding="utf-8")
    anchor = '    n_prod = max(1, int(os.environ.get("TOKENIZER_THREADS", "4")))'
    if anchor not in ce_txt:
        raise SystemExit("ce.py producer anchor not found -- refusing")
    ce_txt = ce_txt.replace(anchor, WARMUP + anchor, 1)
    compile(ce_txt, "ce.py", "exec")
    ce_path.write_text(ce_txt, encoding="utf-8")
    print("src/ce.py: tokenizer warmup inserted (closes the Already-borrowed "
          "race the smoke reproduced in the shipping image)")


    # --- run.py: CE-1 env defaults -> mmBERT's window ------------------------
    OLD_L = 'max_len=int(os.environ.get("MAX_LEN", "256")),'
    NEW_L = 'max_len=int(os.environ.get("MAX_LEN", "1024")),'
    OLD_B = 'batch_size=int(os.environ.get("BATCH_SIZE", "512")),'
    NEW_B = 'batch_size=int(os.environ.get("BATCH_SIZE", "256")),'
    if run.count(OLD_L) != 1 or run.count(OLD_B) != 1:
        raise SystemExit("CE-1 env default lines not unique -- refusing")
    run = run.replace(OLD_L, NEW_L, 1).replace(OLD_B, NEW_B, 1)
    print("run.py: CE-1 defaults MAX_LEN 256->1024, BATCH_SIZE 512->256")

    # --- run.py: _band_rank + cascade block ---------------------------------
    i = run.index("def main() -> None:")
    run = run[:i] + BAND_RANK.lstrip("\n") + "\n" + run[i:]
    start_m = "                    # ---- second cross-encoder (LEDGER ensboard) ----"
    end_m = "\n                else:\n                    log(\"cross-encoder incomplete"
    s0 = run.index(start_m)
    s1 = run.index(end_m, s0)
    for must in ("ce2_dir", "MAX_LEN_2", "exc2"):
        if must not in run[s0:s1]:
            raise SystemExit("CE-2 region lacks %r -- anchors drifted" % must)
    run = run[:s0] + CASCADE_BLOCK.rstrip("\n") + run[s1:]
    print("run.py: CE-2 block -> v21 cascade (cover 0.30, partner at 256, "
          "need ratio 0.43)")
    compile(run, "run.py", "exec")
    runp.write_text(run, encoding="utf-8")

    # --- models/ce-2 <- t120a + bge tokenizer HARVESTED from the CE-1 slot --
    ce1_slot = stage / "models" / "ce-e5-base"
    ce2_slot = stage / "models" / "ce-2"
    src2 = pathlib.Path(args.ce2_dir)
    if (src2 / "model.safetensors").stat().st_size != BGE_BYTES:
        raise SystemExit("ce2 weights wrong size -- expected t120a")
    if (ce1_slot / "model.safetensors").stat().st_size != BGE_BYTES:
        raise SystemExit("CE-1 slot does not hold the t120a bge -- not v22?")
    for p in sorted(ce2_slot.iterdir()):
        p.unlink()
    for f in ("model.safetensors", "config.json"):
        shutil.copyfile(src2 / f, ce2_slot / f)
    for f in BGE_TOK:
        shutil.copyfile(ce1_slot / f, ce2_slot / f)   # bge tokenizer, harvested
    print("models/ce-2 <- t120a weights+config + bge tokenizer (harvested "
          "from the CE-1 slot before the swap)")

    # --- models/ce-e5-base <- mmBERT, stale files removed --------------------
    src1 = pathlib.Path(args.ce1_dir)
    if (src1 / "model.safetensors").stat().st_size != MM_BYTES:
        raise SystemExit("ce1 weights wrong size -- expected t126mmB-ep1")
    removed = []
    for p in sorted(ce1_slot.iterdir()):
        if p.name not in MM_FILES:
            removed.append(p.name)
        p.unlink()
    for f in MM_FILES:
        shutil.copyfile(src1 / f, ce1_slot / f)
    print("models/ce-e5-base <- %s (stale %s removed)" % (src1.name, removed))

    out = pathlib.Path(args.out)
    if out.exists():
        out.unlink()
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(stage).as_posix())
    print("\n%s  %.2f GB  %d files" % (out.name, out.stat().st_size / 1e9,
                                       len(files)))
    print("sha256 %s" % sha(out))


if __name__ == "__main__":
    main()
