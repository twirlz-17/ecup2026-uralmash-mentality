"""Assemble the graded archive from this repository plus the weight bundle.

    python inference/build_submission.py --weights weights/ --out submission_v46.zip
    python inference/build_submission.py --weights weights/ --out out.zip \
           --verify C:/path/to/submission_v46.zip

Everything except the four large weight/tokenizer blobs lives in this repo, and
every one of those repo files is **byte-identical to submission_v46.zip**, which
scored 0.5414129346430998. `--verify` re-checks that entry by entry against a
copy of the graded archive.

The weight bundle supplies only what is too large for git:

    ce-1/  model.safetensors  tokenizer.json
    ce-2/  model.safetensors  tokenizer.json  sentencepiece.bpe.model
    ce-3/  model.safetensors  tokenizer.json

`config.json`, `tokenizer_config.json` and `special_tokens_map.json` come from
`inference/models/ce-{1,2,3}/` instead, because they encode decisions rather
than weights. One of them is load-bearing and is the reason this split exists:

    models/ce-e5-base/tokenizer_config.json in the archive is NOT the copy the
    CE-1 training run wrote. The trainer's copy carries max_length, stride,
    truncation_side and truncation_strategy, which would give the tokenizer a
    default max_length; the archive deliberately keeps the base copy, which does
    not. Shipping a checkpoint's own config over the base one silently reverted
    an environment fix once and cost 0.00087 on the board.

SLOT NAMES. CE-1 lives at `models/ce-e5-base/` inside the archive. That name is
a fossil -- the slot held an e5-base in v12, and renaming it later would have
been a second variable in a build where exactly one thing changed per
submission. `run.py:179` reads it.

COMPRESSION IS NOT UNIFORM, AND THAT IS AUTHENTIC. 43 of the 47 entries are
DEFLATE; the four `models/ce-3/*` entries are STORED, because the CE-3 stage was
added to an already-built archive by a script that wrote its new entries with
ZIP_STORED. Reproducing that split is what makes the rebuilt file match the
graded one at 2.91 GB rather than 3.26 GB. It has no effect on scoring.
"""
import argparse
import hashlib
import os
import pathlib
import zipfile

HERE = pathlib.Path(__file__).resolve().parent

# archive path <- source. `None` means "from the weight bundle".
CODE = ["run.py", "metadata.json", "src/__init__.py", "src/ce.py",
        "src/features.py", "src/features2.py", "src/gbdt_v2.py",
        "src/gbdt_only.py"]
SLOTS = [("models/ce-e5-base", "ce-1", ["model.safetensors", "tokenizer.json"]),
         ("models/ce-2", "ce-2", ["model.safetensors", "tokenizer.json",
                                  "sentencepiece.bpe.model"]),
         ("models/ce-3", "ce-3", ["model.safetensors", "tokenizer.json"])]
# v45 predates the third stage: same everything, minus the ce-3 slot, with the
# run.py that has no CE-3 block. Its archive is uniformly DEFLATE, because
# nothing was appended to it afterwards.
V45_DROP = "models/ce-3"
CONFIGS = ["config.json", "tokenizer_config.json", "special_tokens_map.json"]
# see the header: the CE-3 slot was appended by a ZIP_STORED writer
STORED_PREFIX = "models/ce-3/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="submission_v46.zip")
    ap.add_argument("--verify", help="path to the graded archive to compare against")
    ap.add_argument("--variant", choices=["v46", "v45"], default="v46",
                    help="which selected submission to build (default v46)")
    a = ap.parse_args()
    w, out = pathlib.Path(a.weights), pathlib.Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)

    v45 = a.variant == "v45"
    plan = []                       # (archive name, source path)
    for rel in CODE:
        src = HERE / "variants" / "v45" / "run.py" if (v45 and rel == "run.py")             else HERE / rel
        plan.append((rel, src))
    for f in sorted(os.listdir(HERE / "models" / "gbdt")):
        plan.append(("models/gbdt/" + f, HERE / "models" / "gbdt" / f))
    for arc, sub, big in SLOTS:
        if v45 and arc == V45_DROP:
            continue
        for f in CONFIGS:
            p = HERE / "models" / sub / f
            if p.exists():
                plan.append((arc + "/" + f, p))
        for f in big:
            p = w / sub / f
            if not p.exists():
                raise SystemExit("missing %s -- see docs/WEIGHTS.md" % p)
            plan.append((arc + "/" + f, p))

    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, src in plan:
            z.write(src, name,
                    compress_type=(zipfile.ZIP_STORED
                                   if (not v45 and name.startswith(STORED_PREFIX))
                                   else zipfile.ZIP_DEFLATED))
    tmp.replace(out)

    blob_sha = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            blob_sha.update(b)
    print("wrote %s (%s)  %.2f GB  %d entries  sha256 %s"
          % (out, a.variant, out.stat().st_size / 1e9, len(plan), blob_sha.hexdigest()))

    if not a.verify:
        return
    ref = zipfile.ZipFile(a.verify)
    got = zipfile.ZipFile(out)
    rn, gn = set(ref.namelist()), set(got.namelist())
    bad = 0
    for n in sorted(rn - gn):
        print("MISSING from build : %s" % n); bad += 1
    for n in sorted(gn - rn):
        print("EXTRA in build     : %s" % n); bad += 1
    for n in sorted(rn & gn):
        ri, gi = ref.getinfo(n), got.getinfo(n)
        if ri.CRC != gi.CRC:
            print("CRC differs        : %s" % n); bad += 1
        elif ri.compress_type != gi.compress_type:
            print("compression differs: %s (%d vs %d)"
                  % (n, ri.compress_type, gi.compress_type)); bad += 1
    if bad:
        raise SystemExit("%d entry difference(s) against %s" % (bad, a.verify))
    print("VERIFIED: all %d entries CRC-identical to %s" % (len(rn), a.verify))


if __name__ == "__main__":
    main()
