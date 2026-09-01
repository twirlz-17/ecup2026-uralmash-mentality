"""Lay the three cross-encoder checkpoints out the way build_submission.py wants.

    python stage_weights.py --src <dir holding the checkpoint dirs> --out weights/

Verifies every file against docs/WEIGHTS.md before copying, so a truncated or
half-downloaded checkpoint is caught here rather than inside a 2.9 GB archive.
"""
import argparse
import hashlib
import os
import pathlib
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent
# checkpoint directory names as they exist in the team's working tree
DEFAULT_NAMES = {"ce-1": "t176full-ep1", "ce-2": "ce2-shipped", "ce-3": "alexbge-fp16"}


def manifest():
    """{slot: {filename: (size, sha256)}} parsed out of docs/WEIGHTS.md."""
    text = (HERE / "docs" / "WEIGHTS.md").read_text(encoding="utf-8")
    out, slot = {}, None
    for line in text.splitlines():
        m = re.match(r"^## `([^`]+)/`", line)
        if m:
            slot = m.group(1)
            out[slot] = {}
            continue
        m = re.match(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|", line)
        if m and slot:
            out[slot][m.group(1)] = (int(m.group(2)), m.group(3))
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory containing the checkpoint directories")
    ap.add_argument("--out", default="weights")
    for slot, name in DEFAULT_NAMES.items():
        ap.add_argument("--" + slot, default=name,
                        help="directory name for %s (default %s)" % (slot, name))
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    want = manifest()
    bad = 0
    for slot in ("ce-1", "ce-2", "ce-3"):
        d = src / getattr(a, slot.replace("-", "_"))
        (out / slot).mkdir(parents=True, exist_ok=True)
        for f, (size, want_sha) in sorted(want[slot].items()):
            p = d / f
            if not p.exists():
                print("MISSING  %s" % p); bad += 1; continue
            if p.stat().st_size != size:
                print("SIZE     %s: %d, want %d" % (p, p.stat().st_size, size))
                bad += 1; continue
            got = sha(p)
            if got != want_sha:
                print("SHA256   %s: %s" % (p, got)); bad += 1; continue
            shutil.copy2(p, out / slot / f)
            print("ok       %s/%s" % (slot, f))
    if bad:
        raise SystemExit("%d file(s) failed verification -- archive NOT staged" % bad)
    print("\nstaged %s -- ready for inference/build_submission.py --weights %s"
          % (out, out))


if __name__ == "__main__":
    main()
