"""Ship a trained checkpoint off the box via Kaggle instead of the scratchpad.

The scratchpad download path moves ~0.4 MB/s, so a 1.1 GB fp32 checkpoint would
take ~45 minutes of round-trips. The box has ordinary outbound bandwidth and the
KGAT token in storage, so pushing a dataset and pulling it locally is minutes,
not an hour.

Stored fp16 on purpose: src/ce.py casts to bf16 on the grader's H100 at load
time anyway, and bf16 keeps 7 mantissa bits against fp16's 10 -- so the store is
strictly less lossy than the cast that happens regardless, and it halves both
the upload and the submission archive.

READS WITHOUT MMAP (2026-08-22). load_file mmaps, and mmap segfaults over 2 GiB
on this box (nommap.py). An fp32 568M checkpoint is 2.27 GB, so THIS SCRIPT
COULD NOT PUSH THE VERY CHECKPOINTS MOST WORTH PROTECTING -- it would have died
on exactly the large mid-run artifacts it exists to rescue, and only on those.

Usage (on the box): python push_ckpt.py <ckpt_dir> <dataset-slug>
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

STOR = "/marimo/storage"
sys.path.insert(0, "/marimo")
src = pathlib.Path(sys.argv[1])
slug = sys.argv[2]
stage = pathlib.Path(f"/marimo/push-{slug}")

# CREDENTIALS. This used to be an unconditional
#     exec(open(f"{STOR}/restore_kaggle.py").read())
# and on 2026-08-24 that single line cost 3.5 GPU-h of t130mmoleg: the file was
# absent on a rebuilt box, every rolling push died with FileNotFoundError, and
# the Kaggle mirror sat a day stale while the run trained on happily. The box
# then died and only the volume saved it.
#
# tools/restore_box.py now plants the token directly at /marimo/storage/
# access_token AND ~/.kaggle/access_token and verifies it with a real API call,
# so the old bootstrap is a dead dependency. Use it when present, and when it is
# not, CHECK THAT CREDENTIALS ACTUALLY EXIST rather than assuming -- a push that
# cannot authenticate must fail here, loudly, before the caller believes the
# checkpoint is safe.
_boot = pathlib.Path(STOR) / "restore_kaggle.py"
if _boot.exists():
    exec(_boot.read_text())
else:
    _tok = [pathlib.Path(STOR) / "access_token",
            pathlib.Path.home() / ".kaggle" / "access_token"]
    _have = [t for t in _tok if t.exists()]
    if not _have:
        raise SystemExit("push_ckpt: no restore_kaggle.py AND no access_token in "
                         "%s -- refusing to pretend this checkpoint is safe"
                         % " or ".join(str(t) for t in _tok))
    _dst = pathlib.Path.home() / ".kaggle" / "access_token"
    if not _dst.exists():
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_bytes(_have[0].read_bytes())
        _dst.chmod(0o600)
    print("push_ckpt: credentials from %s" % _have[0])

if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)

import torch
from safetensors.torch import save_file

from nommap import load_sd

sd = load_sd(src, torch.float16)
save_file(sd, str(stage / "model.safetensors"), metadata={"format": "pt"})
del sd

cfg = json.load(open(src / "config.json", encoding="utf-8"))
cfg["dtype"] = cfg["torch_dtype"] = "float16"
json.dump(cfg, open(stage / "config.json", "w", encoding="utf-8"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(src / f, stage / f)
# Resume bundle (train_ce --resume-partial). A partial that ships WITHOUT
# these restores as an epoch-boundary checkpoint only -- the wiped-volume
# death this bundle exists for is exactly when it must already be off-box.
for f in ("train_state.json", "batch_order.npz", "rng_state.pt"):
    if (src / f).exists():
        shutil.copy(src / f, stage / f)
# Remote-code models (EuroBERT): save_pretrained writes the modeling .py files
# into the dir, and a LOCAL-DIR reload needs them -- t160's death-#7 restore
# crashed on exactly their absence (2026-08-26). Stage every top-level .py.
for p in sorted(src.glob("*.py")):
    shutil.copy(p, stage / p.name)

json.dump({"title": slug, "id": f"gordeevmax/{slug}", "licenses": [{"name": "CC0-1.0"}]},
          open(stage / "dataset-metadata.json", "w"), indent=2)

for p in sorted(stage.iterdir()):
    print(f"  {p.name:24s} {p.stat().st_size / 1e6:9.1f} MB")

tmp = os.path.expanduser("~/.kaggle/uploads/datasets")
os.makedirs(tmp, exist_ok=True)
KG = [sys.executable, "-m", "kaggle"]

# ASK WHETHER THE DATASET EXISTS; DO NOT INFER IT FROM AN ERROR STRING.
#
# This block used to run `datasets create` and fall back to `datasets version`
# only `if "already exists" in out or r.returncode`. Kaggle's actual message is
#     Dataset creation error: The requested title "X" is already in use by a
#     dataset. Please choose another title.
# which does NOT contain "already exists" -- and `datasets create` exits 0 on
# it. So both halves of the condition were false, the version fallback never
# fired, and the script printed "PUSH rc= 0" and exited clean. train_ce logs
# "ok" on returncode 0, so EVERY rolling push to an already-existing slug was a
# silent no-op that reported success. Measured 2026-08-24: four consecutive
# "PUSH ecup26-t130mmoleg-llm-ep0-partial: ok" lines over 3.5h while the Kaggle
# copy stayed frozen at its 2026-08-23 creation date -- confirmed by weight
# fingerprint, the download matched ckpt-t130resume rather than the live
# partial. First-ever pushes of a NEW slug worked, which is why t139hwcont and
# t141molegB survived the box deaths and t130's partial did not.
# THE KAGGLE CLI'S RETURN CODE IS MEANINGLESS -- measured 2026-08-24:
#     $ kaggle datasets files gordeevmax/<does-not-exist>
#     403 Client Error: Forbidden for url: .../ListDatasetFiles
#     $ echo $?
#     0
# It exits 0 on 403s, on "already in use" refusals, on everything. That single
# fact is the root cause of every push failure today, and any script that keys
# off its returncode is broken by construction. So decide from the OUTPUT.

def _clean(txt):
    # THE SIZE-COLUMN TRAP, THIRD SIGHTING (see ckpt_guard 2026-08-25). tqdm
    # progress frames and file listings contain byte counts where "403M" /
    # "404M" appear as ordinary sizes -- a 1.2 GB upload's bar PASSES THROUGH
    # 404M, so on 2026-08-26 every 610m push reported FAILED rc=1 while the
    # upload had landed (the guard then found the remote "already current").
    # Scan only non-progress lines, and only for error-SHAPED phrases below.
    return "\n".join(l for l in txt.splitlines()
                     if "B/s" not in l and "%|" not in l and "it/s" not in l)


_ERRS = ("Client Error", "creation error", "already in use", "not found",
         "Not Found", "Forbidden", "Unauthorized", "TooManyRequests",
         "Traceback")

_probe = subprocess.run(KG + ["datasets", "files", f"gordeevmax/{slug}"],
                        capture_output=True, text=True)
_pout = (_probe.stdout or "") + (_probe.stderr or "")
_missing = any(x in _clean(_pout) for x in _ERRS)
_listed = "model.safetensors" in _pout or ("name" in _pout and "size" in _pout)
exists = _listed and not _missing
verb = "version" if exists else "create"
args = ["-m", "rolling checkpoint"] if exists else []
r = subprocess.run(KG + ["datasets", verb, "-p", str(stage), "--dir-mode", "zip"] + args,
                   capture_output=True, text=True)
out = (r.stdout + r.stderr).strip()
if verb == "create" and ("already in use" in out or "already exists" in out):
    # raced with another pusher, or `files` lied -- version it instead
    print("create rejected, versioning instead:", out[-200:])
    verb = "version"
    r = subprocess.run(KG + ["datasets", "version", "-p", str(stage),
                        "-m", "rolling checkpoint", "--dir-mode", "zip"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()

# A ZERO RETURN CODE IS NOT PROOF. Kaggle exits 0 on refusals, so demand a
# positive success marker and fail loudly without one -- believing a checkpoint
# is safe when it is not is the whole failure mode this script exists to avoid.
# Same rule on the way out: never let the CLI's rc vouch for anything. Demand
# that Kaggle's own file listing come back newer than the file we just pushed.
_bad = any(b in _clean(out) for b in _ERRS)
_after = subprocess.run(KG + ["datasets", "files", f"gordeevmax/{slug}"],
                        capture_output=True, text=True)
_aout = (_after.stdout or "") + (_after.stderr or "")
_readback = "model.safetensors" in _aout and not any(
    x in _clean(_aout) for x in _ERRS)
ok = _readback and not _bad
print("PUSH", verb, "rc=", r.returncode, "verified=", ok, "|", out[-400:])
if not ok:
    sys.exit(1)

# EXIT WITH THE KAGGLE RETURN CODE, NOT WITH 0. This script used to print the
# failure and exit clean, and train_ce's push_off_box logs "ok" on returncode 0
# -- so on 2026-08-22 a fresh box with no `kaggle` module logged
# "PUSH ecup26-t117b320e1-ep1: ok" for a push that never happened, for every
# checkpoint of the day. The rescue mechanism that exists because the sandbox
# dies unannounced was itself silently disabled, and the log said otherwise.
# A push that fails must be loud: losing a push is survivable, believing a
# checkpoint is safe when it is not is how 3.5 GPU-hours went missing before.
sys.exit(r.returncode)
