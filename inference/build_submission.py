"""Assemble the graded archive (submission_v46.zip) from source + weights.

    python inference/build_submission.py --weights /path/to/weights --out submission_v46.zip

`--weights` must contain three HuggingFace directories, whose contents are
listed with sha256 in docs/WEIGHTS.md:

    ce-1/   t176full-ep1        mmBERT-base / ModernBERT, 22 layers, 768 hidden
    ce-2/   t120loss-pw0.134-ep1  bge-reranker-v2-m3 / XLM-R-large
    ce-3/   alexbge-fp16        bge-reranker-v2-m3 / XLM-R-large (teammate's)

NOTE ON SLOT NAMES. Inside the archive CE-1 lives at `models/ce-e5-base/`. That
name is a fossil: the slot held an e5-base in v12 and the directory name was
never changed, because renaming it would have been a second variable in a build
whose whole discipline was one variable per submission. run.py:179 reads it.

The archive is written with ZIP_STORED (no compression): safetensors do not
compress and the grader unpacks under a time budget.
"""
import argparse
import hashlib
import json
import os
import pathlib
import zipfile

HERE = pathlib.Path(__file__).resolve().parent

CODE = ["run.py", "metadata.json",
        "src/__init__.py", "src/ce.py", "src/gbdt_v2.py",
        "src/features.py", "src/features2.py"]

# archive slot  <- weights subdir; the files each slot needs
SLOTS = [("models/ce-e5-base", "ce-1"),
         ("models/ce-2",       "ce-2"),
         ("models/ce-3",       "ce-3")]
WEIGHT_FILES = ["config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json",
                "sentencepiece.bpe.model"]
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="submission_v46.zip")
    a = ap.parse_args()
    w = pathlib.Path(a.weights)
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)

    for _, sub in SLOTS:
        for f in REQUIRED:
            if not (w / sub / f).exists():
                raise SystemExit("missing %s" % (w / sub / f))

    entries = []
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for rel in CODE:
            z.write(HERE / rel, rel)
            entries.append(rel)
        for f in sorted(os.listdir(HERE / "models" / "gbdt")):
            z.write(HERE / "models" / "gbdt" / f, "models/gbdt/" + f)
            entries.append("models/gbdt/" + f)
        for slot, sub in SLOTS:
            for f in WEIGHT_FILES:
                p = w / sub / f
                if p.exists():
                    z.write(p, slot + "/" + f)
                    entries.append(slot + "/" + f)
    tmp.replace(out)

    blob = out.read_bytes()
    print("wrote %s  %.2f GB  %d entries  sha256 %s"
          % (out, len(blob) / 1e9, len(entries),
             hashlib.sha256(blob).hexdigest()))
    print("\n".join("  " + e for e in entries))


if __name__ == "__main__":
    main()
