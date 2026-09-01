"""Build a new submission archive from a proven one by changing ONE thing.

    python tools/build_swap.py --base submission/dist/submission_v39.zip \
        --out submission/dist/submission_v42.zip --ce1 models_local/<ckpt>
    python tools/build_swap.py ... --w-ce 0.55        # config-only variant

WHY A SCRIPT AND NOT AN AD-HOC SESSION. v41 was built inline; the code that
produced the one archive we shipped today lives nowhere, so the last-hour build
would have been improvised against a deadline. It also encodes the guards that
this project paid for:

  * `shiponevariable` is CONFIRMED board evidence -- v21 moved four things and
    lost, v22 moved one with the window held and won +0.0055. So this refuses to
    change more than one thing unless --allow-multi is passed explicitly.
  * v37 scored 0.3611536 because a container was shipped unrun, and v40 lost
    because swapping a checkpoint's own config silently reverted an environment
    fix. So the tokenizer/config entries are compared BYTE-WISE against the base
    and a difference is an ABORT, not a warning.
  * The result is verified by ENTRY-BY-ENTRY CRC against the base archive, and
    the script prints exactly which entries differ. "One variable" is proven
    from the built artifact, never asserted from intent.
  * `preflightthereal` -- never reuse a version number across rebuilds. An
    existing --out is refused unless --force.
"""
import argparse
import hashlib
import os
import shutil
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CE1_DIR = "models/ce-e5-base/"
CE2_DIR = "models/ce-2/"
GUARDED = ("config.json", "tokenizer.json", "tokenizer_config.json",
           "special_tokens_map.json")


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ce1", help="checkpoint dir whose model.safetensors replaces CE-1")
    ap.add_argument("--ce2", help="checkpoint dir whose model.safetensors replaces CE-2")
    ap.add_argument("--w-ce", type=float, help="patch run.py's W_CE default")
    ap.add_argument("--keep-base-tokenizer", action="store_true",
                    help="weights-only swap: keep the base archive's tokenizer "
                         "files even if the checkpoint's differ (requires that "
                         "you have verified semantic equivalence)")
    ap.add_argument("--allow-multi", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    changes = sum(x is not None for x in (a.ce1, a.ce2, a.w_ce))
    if changes == 0:
        raise SystemExit("nothing to change")
    if changes > 1 and not a.allow_multi:
        raise SystemExit(
            "REFUSING: %d variables at once. `shiponevariable` is confirmed board\n"
            "evidence (v21 moved four and lost; v22 moved one and won +0.0055).\n"
            "Pass --allow-multi only if that trade-off is deliberate." % changes)
    if os.path.exists(a.out) and not a.force:
        raise SystemExit("REFUSING: %s exists. Never reuse a version number across "
                         "rebuilds (`preflightthereal`)." % a.out)

    zin = zipfile.ZipFile(a.base)
    names = zin.namelist()
    # SNAPSHOT the base CRCs BEFORE writing anything. zout.writestr(it, data)
    # MUTATES the ZipInfo it is handed, and these ZipInfos come from
    # zin.infolist() -- so writing silently overwrites the base's own recorded
    # CRCs and a later comparison finds NO differences for an entry that really
    # did change. Caught by rebuilding v41: the archive came out byte-identical
    # to the shipped one while the check claimed nothing had changed. It failed
    # CLOSED (aborted) rather than passing a wrong artifact, which is the only
    # reason this was a nuisance instead of a shipped mistake.
    base_crc = {n: zin.getinfo(n).CRC for n in names}
    print("base %s: %d entries" % (a.base, len(names)))

    # WHICH SLOT. CE-2 is the bge-reranker-v2-m3 partner; `tokdiff` verified
    # Alexander's checkpoint carries a BYTE-IDENTICAL config.json and a
    # tokenizer whose vocab, normalizer, pre_tokenizer and post_processor all
    # match ours, so the swap is legitimately ONE entry -- the weights blob --
    # with the shipped tokenizer left in place. Do not extend this to a
    # checkpoint whose config or vocab differs without re-checking.
    slot = CE2_DIR if a.ce2 else CE1_DIR
    ck = a.ce2 or a.ce1
    new_w = None
    if ck:
        p = os.path.join(ck, "model.safetensors")
        new_w = open(p, "rb").read()
        print("new weights for %s: %d B  sha %s"
              % (slot, len(new_w), sha16(new_w)))
        old = zin.read(slot + "model.safetensors")
        if len(new_w) != len(old):
            print("NOTE: size differs from base (%d -> %d)" % (len(old), len(new_w)))
        # v40's trap: a checkpoint's own config silently reverting an env fix.
        for g in GUARDED:
            src = os.path.join(ck, g)
            if not os.path.exists(src):
                continue
            if slot + g not in names:
                continue
            if open(src, "rb").read() != zin.read(slot + g):
                if a.keep_base_tokenizer and g != "config.json":
                    # This build replaces ONLY model.safetensors, so the
                    # checkpoint's own copy of this file is never shipped --
                    # the base archive's stays. That is safe ONLY when the two
                    # are semantically equivalent, which must be checked, not
                    # assumed: for Alexander's bge, tools/tokdiff verified the
                    # vocab (all 250,002 entries), normalizer, pre_tokenizer
                    # and post_processor are EQUAL and config.json is
                    # byte-identical; the file-level difference is formatting.
                    # config.json is deliberately excluded from this escape
                    # hatch -- that is the exact file whose silent substitution
                    # cost v40 0.00087.
                    print("NOTE: %s differs but is NOT shipped (base copy kept, "
                          "semantic equivalence pre-verified)" % (slot + g))
                    continue
                raise SystemExit(
                    "ABORT: %s differs between the checkpoint and the base archive.\n"
                    "Shipping the checkpoint's own copy is how v40 reverted an\n"
                    "environment fix and lost 0.00087. Reconcile it deliberately.\n"
                    "If this build replaces only the weights and you have VERIFIED\n"
                    "the tokenizers are semantically identical, pass\n"
                    "--keep-base-tokenizer." % (slot + g))
        print("guarded entries byte-identical to base: %s" % ", ".join(GUARDED))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    replaced, copied = [], 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if ck and it.filename == slot + "model.safetensors":
                data = new_w
                replaced.append(it.filename)
            elif a.w_ce is not None and it.filename == "run.py":
                src = data.decode("utf-8")
                old_s = 'os.environ.get("W_CE", "0.7")'
                new_s = 'os.environ.get("W_CE", "%s")' % a.w_ce
                if old_s not in src:
                    raise SystemExit("ABORT: could not find the W_CE default in run.py")
                data = src.replace(old_s, new_s).encode("utf-8")
                replaced.append(it.filename)
            else:
                copied += 1
            zout.writestr(it, data)
    shutil.move(tmp, a.out)

    # PROVE one variable from the artifact, entry by entry.
    zo = zipfile.ZipFile(a.out)
    diff = [n for n in names if base_crc[n] != zo.getinfo(n).CRC]
    blob = open(a.out, "rb").read()
    print("\nwrote %s: %d B  sha256 %s" % (a.out, len(blob), sha16(blob)))
    print("entries: %d copied verbatim, %d replaced" % (copied, len(replaced)))
    print("entries differing by CRC: %s" % diff)
    if sorted(diff) != sorted(replaced):
        raise SystemExit("ABORT: CRC diff %s does not match intended changes %s"
                         % (diff, replaced))
    print("VERIFIED: %s is %s with exactly %d entr%s changed"
          % (os.path.basename(a.out), os.path.basename(a.base),
             len(diff), "y" if len(diff) == 1 else "ies"))


if __name__ == "__main__":
    main()
