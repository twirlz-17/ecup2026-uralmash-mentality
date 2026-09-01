"""Parameterised CE trainer for the box — one script, several pre-registered rounds.

Generalises box/train_t102.py so the curriculum, the backbone, and the LLM
pool's cleanliness are flags rather than forks of the file. Every run scores
the frozen 415k universe after each Stage-B epoch, so each epoch is priced on
the calibrated line (LB = 0.3385 * leak-free-E_real + 0.2674, R2 0.9995)
without spending a submission slot.

Rounds this file was written for:

  t104  more human epochs from t98b's ep1 checkpoint. The ep0 -> ep1 step was
        +0.014 ruler = +0.005 LB with no sign of saturation, and extra epochs
        cost nothing at inference. Pre-registered: ACCEPT the best epoch iff it
        beats ep1's 0.60657 by > +0.006 leak-free E_real (~ +0.002 LB, clear of
        the fit's 0.0008 residual). Prediction on record: ep2 gains, ep3+
        flattens or turns over.

  t105  cleaned Stage A. The llm distillation pool carries 138,276 PROVEN label
        contradictions (COOKBOOK "the labels contradict themselves"), and Stage A
        is 17k steps of learning an identity relation from it. Two arms, one
        variable — whether rows in self-contradicting components are eligible.
        Class counts are fixed by the sampler, so cleaning changes WHICH rows
        are drawn, never how many or their balance. Pre-registered: ACCEPT iff
        the cleaned arm beats the baseline arm by > +0.006 leak-free E_real at
        the same epoch index. Prediction on record: the gain appears already
        after Stage A (val MACRO), because the mechanism is distillation, and
        survives Stage B at roughly half the size.

Both arms of t105 are trained HERE rather than reusing t98b's Kaggle
checkpoint: that one ran on a T4 in fp16, which would make the comparison
two-variable.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MAX_LEN = 256
CKPT_EVERY = 4000
WARMUP_RATIO = 0.05
LLM_SAMPLE = {"pos": 1_200_000, "mid": 500_000, "zero": 500_000}
SEED = 42
# Overridable so the same file runs on a Kaggle kernel without a fork; the
# default is the box layout and nothing that does not set these changes.
ROOT = os.environ.get("ECUP_ROOT", "/marimo")
STOR = os.environ.get("ECUP_STOR", ROOT + "/storage")
# LEDGER `charcap`: the SHIPPED container does text[:900] before tokenizing and
# this trainer never has, so every local number is measured on text the board is
# not given. 0 = off, which is the historical behaviour of every checkpoint we
# own; set it only to reproduce the container's view under --val-only.
TEXT_CAP = 0
# `kaggle-t4-bf16-emulation-trap`: torch.cuda.is_bf16_supported() returns True on
# a T4 and then EMULATES it, costing ~20x. Every box we own is Ampere-or-later so
# bf16 stays the default and nothing about existing runs changes; set
# ECUP_AMP_DTYPE=fp16 on a T4 kernel.
AMP_DTYPE = {"fp16": "float16", "bf16": "bfloat16"}[
    os.environ.get("ECUP_AMP_DTYPE", "bf16")]
ATTR_CLEAN = re.compile(r'[{}\[\]"]')
WS = re.compile(r"\s+")

torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()


def log(m):
    print(f"[{(time.time() - t0) / 60:7.1f} min] {m}", flush=True)


# FEED TELLS -- attribute keys that mark which FEED an item came from, not what
# the item IS. Mined by oleg (LEDGER b3-c1gate, oleg/artifacts/feed_tells.json)
# over 4.19M indexed items / 2.78M resolvable llm pairs, by the rule
# mass_llm>=1e-4 AND mass_human==0 AND within-pair co-occurrence obs/exp<0.5.
# `#hashtagi` alone is 7.1% of llm mass at 2,381,428 one-sided against 179
# two-sided (126,765 expected). Roughly 29% of resolvable llm pairs carry at
# least one on EXACTLY ONE side, and the annotator shifts -0.0338 on #hashtagi
# and -0.0637 on the other 11 -- so Stage A can learn feed provenance instead of
# matching. mass_human==0 means these keys never appear in the human pool at
# all, so dropping them cannot remove human evidence.
FEED_TELLS = frozenset(["#хештеги", "oem номер",
    "вес в килограммах с учетом упаковки", "вид линз",
    "детальное описание", "категория товара", "место происхождения",
    "название группы", "номер модели", "особенности картин по номерам",
    "особенности модели обуви", "особенности окна",
    "особенности очков/линз", "партномер аналога",
    "полное наименование товара", "регистрационное удостоверение рф",
    "серия/линейка", "стеклопакет", "тип краски в наборе",
    "уход за обувью",
])


def make_text(name, category, attributes, tells="off"):
    """Build the pair side.

    tells='off'     legacy path, byte-identical to every shipped checkpoint.
    tells='control' parse -> re-serialise, NOTHING removed. This is the correct
                    A/B baseline, not 'off': re-serialisation alone perturbs the
                    model (oleg measured rho 0.9624 under a pure re-serialise,
                    and our own reorder probe lost 0.006 on what turned out to
                    be a content/serialisation change). Comparing 'drop' to
                    'off' would confound removal with re-serialisation.
    tells='drop'    same round-trip, minus the feed-tell keys.
    """
    if tells == "off":
        attrs = WS.sub(" ", ATTR_CLEAN.sub(" ", str(attributes)))
        t = f"{name} [SEP] {category} [SEP] {attrs[:1500]}"
        return t[:TEXT_CAP] if TEXT_CAP else t
    try:
        d = json.loads(attributes)
    except Exception:
        d = None
    if not isinstance(d, dict):
        attrs = WS.sub(" ", ATTR_CLEAN.sub(" ", str(attributes)))
        return f"{name} [SEP] {category} [SEP] {attrs[:1500]}"
    parts = []
    for k, v in d.items():
        if tells == "drop" and str(k).strip().lower() in FEED_TELLS:
            continue
        ck = WS.sub(" ", ATTR_CLEAN.sub(" ", str(k)))
        cv = WS.sub(" ", ATTR_CLEAN.sub(" ", str(v)))
        parts.append(f"{ck} : {cv}")
    attrs = " " + " , ".join(parts) + " "
    return f"{name} [SEP] {category} [SEP] {attrs[:1500]}"


# Ship a rolling checkpoint at least this often, in seconds, regardless of
# where the epoch boundary falls. 45 min bounds the loss from a sandbox
# death at ~45 min of GPU rather than at one epoch.
PUSH_EVERY_S = 45 * 60
_LAST_PUSH = 0.0


def preflight_push(log):
    """Verify the checkpoint-push path BEFORE training, not after a box death.

    2026-08-24 cost this exact check ~3.2 GPU-h on t130mmoleg and an entire
    t131sdpa run: push_ckpt.py was missing on both boxes, so every rolling push
    failed. push_off_box already logged FAILED loudly -- but a log nobody greps
    is not an alarm. This runs once at startup, says precisely which link is
    broken, and leaves a marker file the watcher can see.
    """
    import shutil
    import subprocess

    problems = []
    if not os.path.exists(f"{ROOT}/push_ckpt.py"):
        problems.append(f"MISSING {ROOT}/push_ckpt.py -- checkpoints CANNOT be "
                        f"pushed. Bootstrap with tools/box_bootstrap.py, which "
                        f"uploads every box/*.py, not a hand-picked subset.")
    tok = os.path.expanduser("~/.kaggle/access_token")
    if not os.path.exists(tok):
        if os.path.exists(f"{STOR}/access_token"):
            try:
                os.makedirs(os.path.dirname(tok), exist_ok=True)
                shutil.copy(f"{STOR}/access_token", tok)
                os.chmod(tok, 0o600)
                log("preflight: restored ~/.kaggle/access_token from storage")
            except Exception as exc:
                problems.append(f"token restore failed: {exc!r}")
        else:
            problems.append("MISSING ~/.kaggle/access_token and "
                            f"{STOR}/access_token -- kaggle cannot authenticate")
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "--version"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            problems.append("kaggle CLI not runnable: "
                            + (r.stderr or r.stdout or "")[-200:])
    except Exception as exc:
        problems.append(f"kaggle CLI not importable: {exc!r}")

    marker = f"{ROOT}/PUSH_BROKEN"
    if problems:
        bar = "!" * 72
        log(bar)
        log("PREFLIGHT: THE CHECKPOINT PUSH PATH IS BROKEN. This run's work is")
        log("NOT durable -- a box death loses everything since the last push.")
        for p in problems:
            log("  - " + p)
        log(bar)
        try:
            open(marker, "w").write("\n".join(problems))
        except Exception:
            pass
        return False
    if os.path.exists(marker):
        try:
            os.remove(marker)
        except Exception:
            pass
    log("preflight: push path OK (push_ckpt.py, kaggle CLI, token)")
    _mirror_ancestor(log)
    return True


def _mirror_ancestor(log):
    """Push the checkpoint this run RESUMES FROM, before training touches it.

    2026-08-24 post-mortem: t126mm1024 pushed rolling Stage-A checkpoints, the
    last landing at ~step 36k, then the box died at ~50k. The VOLUME survived, so
    queue25's `if [ ! -f "$SRC" ]` branch trained our champion from the local 50k
    checkpoint rather than the Kaggle mirror -- silently, since the branch leaves
    no log line. That volume is now gone, so the ancestor of the best model we
    have exists nowhere and the run cannot be reproduced. Every run had pushed
    its OWN checkpoints; none had pushed the one it started from.
    """
    import subprocess
    src = globals().get("_RESUME_SRC")
    if not src or not os.path.isdir(src):
        return
    slug = "ecup26-anc-" + os.path.basename(os.path.normpath(src)).replace("_", "-").lower()
    try:
        r = subprocess.run([sys.executable, f"{ROOT}/push_ckpt.py", src, slug],
                           capture_output=True, text=True, timeout=1800)
        log(f"ancestor mirror {src} -> {slug}: "
            + ("ok" if r.returncode == 0 else
               "FAILED rc=" + str(r.returncode) + " :: "
               + (r.stderr or r.stdout or "")[-200:]))
    except Exception as exc:
        log(f"ancestor mirror {src}: FAILED {exc!r} -- training continues")


def push_off_box(ckpt_dir, slug, log):
    """Ship a checkpoint to Kaggle THE MOMENT it exists.

    The molab sandbox terminates without warning: three times in two days, and
    on 2026-08-22 it took t117's COMPLETED Stage A (55,000 steps), its
    human-ep0 checkpoint and its universe scores -- roughly 3.5 GPU-hours --
    because the queue script only pushed after the WHOLE run finished. The
    trainer was already writing those checkpoints to local disk; nothing was
    carrying them off the box.

    So the push belongs here, next to the save, not in a queue script that runs
    at the end. Failures are logged and swallowed on purpose: losing a push is
    an inconvenience, but killing a training run that is otherwise fine because
    Kaggle rate-limited us would be much worse.
    """
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, f"{ROOT}/push_ckpt.py", ckpt_dir, slug],
            capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0
        log(f"PUSH {slug}: {'ok' if ok else 'FAILED rc=' + str(r.returncode)}"
            + ("" if ok else " :: " + (r.stderr or r.stdout)[-300:]))
    except Exception as exc:
        log(f"PUSH {slug}: FAILED {exc!r} -- training continues")


def _backbone_layers(model):
    """(embeddings, [encoder layers, shallowest first]) for the CE's backbone.

    Every backbone either track has trained exposes `base_model.embeddings` and
    `base_model.encoder.layer`: e5-base (BertModel), bge-reranker-v2-m3 and
    ruRoberta-large (XLM-R / RoBERTa). An unknown architecture returns
    (None, []) and the callers below REFUSE rather than quietly training a
    full fine-tune while the log claims layers were frozen -- a silent no-op
    here would look exactly like "freezing does nothing", which is the answer
    the arm exists to measure.
    """
    base = getattr(model, "base_model", model)
    enc = getattr(base, "encoder", None)
    if enc is not None and getattr(enc, "layer", None) is not None:
        return (getattr(base, "embeddings", None), list(enc.layer))
    # ModernBERT (mmBERT): there is no `encoder` wrapper -- the ModuleList
    # lives at base.layers, embeddings at base.embeddings, same shallowest-
    # first order. Without this branch the freeze REFUSES on mmBERT, which is
    # the loud failure the docstring promises but not the arm oleg asked for.
    if getattr(base, "layers", None) is not None:
        return (getattr(base, "embeddings", None), list(base.layers))
    return (getattr(base, "embeddings", None), [])


def set_frozen(model, n, log):
    """Freeze embeddings + the lowest n encoder layers; n<=0 unfreezes all.

    Called at every stage boundary, so Stage B can hand back the layers Stage A
    held fixed. Returns the number of frozen parameters for the log.
    """
    emb, layers = _backbone_layers(model)
    for p in model.parameters():
        p.requires_grad = True
    if n <= 0:
        return 0
    if emb is None or not layers:
        raise SystemExit(
            f"--freeze-layers {n}: no embeddings/encoder.layer on "
            f"{type(model).__name__}; refusing to report a freeze that did not "
            f"happen")
    n = min(n, len(layers))
    for mod in [emb] + layers[:n]:
        for p in mod.parameters():
            p.requires_grad = False
    tot = sum(p.numel() for p in model.parameters())
    fro = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(f"freeze: embeddings + lowest {n}/{len(layers)} layers -> "
        f"{fro / 1e6:.1f}M of {tot / 1e6:.1f}M params frozen ({fro / tot:.1%})")
    return fro


def llrd_groups(model, lr, decay, wd=0.01):
    """Per-layer lr groups: head at lr, layer below it lr*decay, ..., embeddings
    deepest. Frozen params are dropped, so --llrd composes with --freeze-layers.
    """
    emb, layers = _backbone_layers(model)
    if not layers:
        raise SystemExit(f"--llrd {decay}: no encoder.layer on "
                         f"{type(model).__name__}; refusing to guess a depth")
    seen, groups = set(), []

    def add(mod, depth):
        ps = [p for p in mod.parameters() if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in ps)
        if ps:
            groups.append({"params": ps, "lr": lr * (decay ** depth),
                           "weight_decay": wd})

    for depth, lyr in enumerate(reversed(layers), start=1):
        add(lyr, depth)
    if emb is not None:
        add(emb, len(layers) + 1)
    # the head (and pooler) is whatever the loops above did not claim; it is
    # randomly initialised on a cold start, so it gets the full lr
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if head:
        groups.insert(0, {"params": head, "lr": lr, "weight_decay": wd})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--stage-a", type=int, default=1, help="LLM distillation epochs")
    ap.add_argument("--stage-b", type=int, default=2, help="human fine-tune epochs")
    ap.add_argument("--clean", action="store_true",
                    help="drop llm rows in self-contradicting MATCH components")
    ap.add_argument("--clean-mode", default="contra", choices=["contra", "margin", "both"],
                    help="WHICH definition of a bad label to drop. 'contra' = rows in a "
                         "MATCH component that also contains an internal NON-match "
                         "(threshold-free, provable, 9.26%% of rows). 'margin' = rows whose "
                         "llm probability sits in the undecided band (18.75%% of rows). The "
                         "two overlap only 13.5%%, so they are nearly DISJOINT definitions "
                         "of 'the worst 10%%' and must be tested separately")
    ap.add_argument("--tells", default="off",
                    choices=["off", "control", "drop"],
                    help="feed-tell handling (LEDGER b3-c1gate). off = legacy "
                         "byte-identical text. control = parse+re-serialise, "
                         "nothing removed -- THE A/B BASELINE, because "
                         "re-serialisation alone moves the model. drop = same "
                         "round-trip minus the 20 feed-tell keys. Applied to "
                         "ALL frames so Stage A, Stage B and universe scoring "
                         "share one serialisation; the tells have mass_human=0 "
                         "so only the llm frame changes in substance, but the "
                         "shipped run.py must use the SAME round-trip.")
    ap.add_argument("--no-push-epochs", dest="push_epochs",
                    action="store_false",
                    help="do NOT ship each checkpoint to Kaggle as it is "
                         "written. Default is to push, because the sandbox has "
                         "terminated without warning three times in two days "
                         "and on 2026-08-22 that cost a completed Stage A, a "
                         "human epoch and its universe scores -- all of which "
                         "were sitting on local box disk waiting for a queue "
                         "script to push them at the END of the run.")
    ap.add_argument("--llm-scale", type=float, default=1.0,
                    help="multiply every LLM_SAMPLE class count by this. The "
                         "12:5:5 RATIO is preserved, so this is a pure VOLUME "
                         "axis and not a mix change -- which matters, because "
                         "the two are judged by different rules (LEDGER "
                         "stageavol2x vs mixaxisclosed). 1.0 = the shipped "
                         "2.2M pairs; 2.0 = 4.4M. The availability assertion "
                         "below still fires, so an over-large scale fails "
                         "loudly instead of silently shrinking an arm.")
    ap.add_argument("--llm-zero-scale", type=float, default=1.0,
                    help="multiply ONLY the zero class, leaving pos and mid at "
                         "their --llm-scale counts. This is deliberately a MIX "
                         "change, the complement of --llm-scale's pure dose, and "
                         "it exists because of a measured prevalence mismatch: "
                         "Stage A trains at 54.9% positive MASS, Stage B at "
                         "25.7%, and the board scores at 4.5%. Per-category AP "
                         "is almost entirely explained by the false-positive "
                         "rate at the head (rho(AP, hardFP) = -0.893 across all "
                         "20 categories, LEDGER weakcatprofile), and a model "
                         "trained at 55% positives is under-penalised for "
                         "exactly that. 7,184,663 zeros exist and we draw "
                         "500,000 of them (7.0%). zero-scale 6.0 puts Stage A at "
                         "25.7% positive mass, matching Stage B. Read on E_mix, "
                         "not E_real: this moves the training MIX.")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers. Each one is a FORK that "
                         "copy-on-writes its way toward a full copy of the "
                         "tokenised set, so RAM scales with pairs x workers. At "
                         "the champion's 2.2M draw 8 workers are free; at "
                         "t176full's 11.18M they took the box to 899 MiB "
                         "available with no swap (LEDGER oompattern, workermem). "
                         "Lower it for large draws -- the GPU is compute-bound at "
                         "97% util, so workers are not the bottleneck.")
    ap.add_argument("--ckpt-every", type=int, default=4000,
                    help="write a mid-epoch checkpoint every N steps (0 disables). "
                         "Stage A is a single long epoch and the sandbox dies without "
                         "warning, so without this a kill late in Stage A loses hours")
    ap.add_argument("--max-len", type=int, default=None,
                    help="override the tokenised pair length. Oleg's bge track runs 192 "
                         "and his board anchor o-bge was measured there, so a bge arm must "
                         "match it or the comparison confounds backbone with length - and "
                         "length is NOT free (o-len384 = -0.0095 LB)")
    ap.add_argument("--llm-pos-scale", type=float, default=1.0,
                    help="multiply ONLY the pos class. The third of the three "
                         "class knobs, so Stage-A CLASS BALANCE can move as a "
                         "single variable. LEDGER stageamix (a): our Stage A is "
                         "54.5%% positive against alex 14.6%% and a BOARD "
                         "prevalence of 2.5-4.5%% (boardprev) -- 12-22x off, and "
                         "ranked the cheapest and most surprising of the four "
                         "differences. Never run.")
    ap.add_argument("--llm-mid-scale", type=float, default=1.0,
                    help="multiply ONLY the mid (soft-label) class; the exact "
                         "complement of --llm-zero-scale. 0 DELETES the soft "
                         "middle of the curriculum, which is what 'confident "
                         "answers only' means -- alex ran that on bge and got "
                         "LB 0.5143446765. LEDGER t105margin declined the arm "
                         "because --clean-mode margin and this class are THE "
                         "SAME ROWS (mid 1,383,550 -> 1,434), so the size-match "
                         "assert fires. This flag keeps the arm DOSE-MATCHED by "
                         "scaling pos/zero up: the honest way to run it.")
    ap.add_argument("--batch-llm", type=int, default=128)
    ap.add_argument("--batch-human", type=int, default=96)
    ap.add_argument("--lr-llm", type=float, default=2e-5)
    ap.add_argument("--lr-human", type=float, default=1e-5)
    ap.add_argument("--mined-neg", action="store_true",
                    help="add LSH-mined near-duplicate negatives to Stage B")
    ap.add_argument("--optim", choices=["adamw", "nadam"], default="adamw",
                    help="optimizer family; nadam = torch.optim.NAdam with "
                         "decoupled weight decay (LEDGER optblock)")
    ap.add_argument("--sched-epochs", type=int, default=0,
                    help="build ONE optimizer + ONE warmup->0 schedule spanning "
                         "N Stage-B epochs instead of restarting per epoch "
                         "(LEDGER epochreset: every epoch is a warm restart "
                         "annealed to zero; convaudit: never annealed at its "
                         "best point). 0 = shipped behaviour")
    ap.add_argument("--replay-file", type=str, default="",
                    help="parquet with id1,id2,target: --replay draws from THIS pool instead of the Stage-A llm subsample. Items must exist in the llm item set. Built for oleg's qwen-judge corrected rows (LEDGER qylabel)")
    ap.add_argument("--replay", type=int, default=0,
                    help="mix N llm-pool rows into EACH Stage-B epoch. alex's curriculum handoff: 100k replay (3:1 alternation) held his bge's Stage-A knowledge through Stage B (LB 0.4953). Here row-level mixing at the same ratio; 0 = shipped behaviour")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="BCEWithLogitsLoss pos_weight. None/1.0 = the shipped "
                         "plain BCE. 0.134 re-weights Stage B's ~26%% positives "
                         "to the board's ~4.5%%; >1 pushes positives up the "
                         "ranking instead. Single variable, one line.")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze the embeddings and the lowest N of the "
                         "backbone's encoder layers. 0 = the shipped full "
                         "fine-tune (the control). LEDGER freezeab: protect "
                         "pretrained representations from 11M noisy LLM "
                         "labels in Stage A, then unfreeze for Stage B.")
    ap.add_argument("--stage-c", type=int, default=0,
                    help="oleg's curriculum stage (team/oleg/molab/"
                         "molab_train_stagec.py, TG 2026-08-23): epochs of "
                         "fine-tuning on the model's OWN hard examples, mined "
                         "from the human TRAIN half. 0 = off.")
    ap.add_argument("--hard-k", type=int, default=30_000,
                    help="Stage C mines the top K scored negatives + bottom K "
                         "scored positives + 2K random ballast (oleg's "
                         "30k/30k/60k shape)")
    ap.add_argument("--stagec-top", type=int, default=4,
                    help="Stage C trains ONLY the top N encoder layers "
                         "(oleg: 'потом только последние слои тюнить на "
                         "сложные примеры'); everything below is frozen")
    ap.add_argument("--lr-hard", type=float, default=5e-6)
    ap.add_argument("--freeze-stage", default="a", choices=["a", "b", "ab"],
                    help="which stages --freeze-layers applies to. Default 'a' "
                         "is the registered freezeab design: freeze through "
                         "Stage A, full fine-tune in Stage B.")
    ap.add_argument("--llrd", type=float, default=None,
                    help="layer-wise lr decay. Layer i gets lr*DECAY^(depth "
                         "from the top), embeddings deepest, head at full lr. "
                         "0.9 is the usual setting. None = one lr for every "
                         "parameter, which is what every run to date has used "
                         "-- grep says LLRD has never been mentioned, let "
                         "alone tried, on either track.")
    ap.add_argument("--hard-weight", type=float, default=0.0,
                    help="Stage-B weight multiplier for hard pairs (identical-name negs, disagreeing-name pos); 0=off")
    ap.add_argument("--text-cap", type=int, default=0,
                    help="truncate each side to N chars, as the shipped "
                         "container does (LEDGER charcap). Diagnostic only: "
                         "requires --val-only, so no checkpoint is ever trained "
                         "on text a previous one did not see.")
    ap.add_argument("--val-only", action="store_true",
                    help="score --model on the human-val split and exit; no training")
    ap.add_argument("--resume-epoch", type=int, default=0,
                    help="epoch index the resumed checkpoint already reached")
    ap.add_argument("--group-by-length", type=int, default=0,
                    help="length-grouped training batches: sort by token length "
                         "inside a buffer of N batches, then shuffle batch order. "
                         "0=off. 50 cuts padded tokens 2.77x (65%% of training is "
                         "padding at batch 96/max_len 1024). TRAINING CHANGE -- "
                         "batches become length-homogeneous; needs its own A/B.")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="re-seed TRAINING stochasticity (dropout, batch order) "
                         "after the split is drawn; the val split, llm sampling "
                         "and Stage-C ballast all stay on SEED so arms stay "
                         "comparable. Use to measure the run-to-run null.")
    ap.add_argument("--cats", default="",
                    help="';'-separated category names: restrict BOTH stages to "
                         "these categories, filtering AFTER the global split and "
                         "AFTER the global SEED-42 llm draw -- so the "
                         "specialist's rows are an exact subset of the pooled "
                         "arms' rows and its per-category val rows stay "
                         "bit-identical to every pooled checkpoint's "
                         "(percatft). Val macro is over the kept categories "
                         "only. Inert when empty.")
    ap.add_argument("--category-idx", type=int, default=None,
                    help="percat: train and validate on ONE category -- index "
                         "into sorted(ih.category.unique()), passed as an int "
                         "so no Cyrillic crosses a shell (the kaggle-push "
                         "mangling trap). Filters tr_h AND va_h after the "
                         "split, so the val read IS that category's AP and "
                         "the split stays byte-identical to every other arm.")
    ap.add_argument("--no-save", action="store_true",
                    help="percat probes: skip the epoch checkpoint save, its "
                         "push and score_universe -- the read lives in "
                         "metrics_<tag>.json; 20 specialist saves would be "
                         "25GB of writes and a guard-push storm.")
    ap.add_argument("--resume-partial", default="",
                    help="resume INSIDE a Stage-A epoch from a -partial dir "
                         "written by --ckpt-every (euro610stage: a 610m llm-ep "
                         "is ~15h against 5-7h box lives). Implies --model "
                         "<dir>. Needs the dir's resume bundle "
                         "(train_state.json + batch_order.npz + rng_state.pt); "
                         "refuses without it, and refuses if the rebuilt data "
                         "does not match the bundle's signature. The seam is a "
                         "FRESH optimizer + scheduler fast-forward + 100-step "
                         "lr re-warmup (no optimizer state is shipped off-box: "
                         "that would triple every push; optblock bounds the "
                         "restart effect inside the 0.0019 band). Stage-A only, "
                         "--group-by-length only.")
    args = ap.parse_args()
    if AMP_DTYPE == "float16" and not args.val_only:
        # fp16 TRAINING needs a GradScaler this script does not have; without
        # one the loss silently becomes NaN. Inference in fp16 is fine, which is
        # the only reason the override exists.
        raise SystemExit("ECUP_AMP_DTYPE=fp16 is inference-only: pass --val-only")
    if args.text_cap:
        # FAIL LOUDLY rather than quietly train on a shorter text than every
        # existing checkpoint saw -- that would move a second variable in a
        # place nobody would think to look for it.
        if not args.val_only:
            raise SystemExit("--text-cap is a diagnostic: pass --val-only too")
        globals()["TEXT_CAP"] = args.text_cap
        print(f"TEXT_CAP={args.text_cap} (reproducing the container's view)",
              flush=True)
    _rp = None
    if args.resume_partial:
        # Validate the bundle BEFORE the ~10-min data build, and fail loudly:
        # a partial without a bundle predates this patch (or came from a
        # no-group-by-length run) and can only resume at epoch boundaries.
        _ts = os.path.join(args.resume_partial, "train_state.json")
        if not os.path.exists(_ts):
            raise SystemExit(f"--resume-partial: {_ts} missing -- no resume "
                             "bundle; use the epoch-boundary path instead")
        for _f in ("batch_order.npz", "rng_state.pt"):
            if not os.path.exists(os.path.join(args.resume_partial, _f)):
                raise SystemExit(f"--resume-partial: {_f} missing from "
                                 f"{args.resume_partial}")
        _rp = json.load(open(_ts))
        if not _rp["tag"].startswith("llm-ep"):
            raise SystemExit("--resume-partial is Stage-A-scoped "
                             f"(bundle tag {_rp['tag']!r}); Stage-B epochs "
                             "resume at boundaries via --model + --resume-epoch")
        if not args.group_by_length:
            raise SystemExit("--resume-partial requires --group-by-length "
                             "(the saved batch order IS the resume)")
        args.model = args.resume_partial

    ckpt = f"{ROOT}/ckpt-{args.tag}"
    os.makedirs(ckpt, exist_ok=True)

    global MAX_LEN, CKPT_EVERY
    CKPT_EVERY = args.ckpt_every
    if args.max_len:
        MAX_LEN = args.max_len
    log(f"max_len={MAX_LEN}")
    globals()["_RESUME_SRC"] = args.model if os.path.isdir(args.model) else None
    preflight_push(log)
    log(f"round {args.tag}: model={args.model} stageA={args.stage_a} "
        f"stageB={args.stage_b} clean={args.clean}")

    m = pd.read_parquet(f"{STOR}/matches.parquet")
    ih = pd.read_parquet(f"{STOR}/items_human.parquet")

    llm_sub = None
    if args.stage_a > 0 or args.replay > 0:
        mllm = pd.read_parquet(f"{STOR}/matches_llm.parquet")
        if args.clean:
            fl = np.load(f"{ROOT}/llm_clean_flags.npz")
            n = int(fl["n"])
            assert n == len(mllm), f"flag file has {n} rows, matches_llm has {len(mllm)}"
            ypk = np.unpackbits(fl["ypack"])[:n].astype(bool)
            assert (ypk == (mllm.target.to_numpy() > 0.5)).all(), \
                "flag file is not row-aligned with matches_llm"
            drop = np.zeros(n, dtype=bool)
            if args.clean_mode in ("contra", "both"):
                drop |= np.unpackbits(fl["contra"])[:n].astype(bool)
            if args.clean_mode in ("margin", "both"):
                drop |= np.unpackbits(fl["margin"])[:n].astype(bool)
            log(f"clean[{args.clean_mode}]: dropping {int(drop.sum())} rows "
                f"({drop.mean():.2%})")
            mllm = mllm[~drop]
        # the arms are only comparable while the sampler can still fill every
        # class to the SAME count -- cleaning must change which rows are drawn,
        # never how many. Fail loudly rather than silently shrinking an arm.
        avail = {"pos": int((mllm.target >= 0.5).sum()),
                 "mid": int(((mllm.target > 0) & (mllm.target < 0.5)).sum()),
                 "zero": int((mllm.target == 0).sum())}
        _want = {k: int(round(v * args.llm_scale
                                 * (args.llm_zero_scale if k == 'zero' else
                                    args.llm_mid_scale if k == 'mid' else
                                    args.llm_pos_scale)))
                  for k, v in LLM_SAMPLE.items()}
        short = {k: (avail[k], v) for k, v in _want.items() if avail[k] < v}
        assert not short, (
            f"clean[{args.clean_mode}] left too few rows to keep the arms "
            f"size-matched: {short} (have, need). Lower LLM_SAMPLE for BOTH arms "
            f"or the comparison confounds cleaning with training volume.")
        want = {k: int(round(v * args.llm_scale
                                 * (args.llm_zero_scale if k == 'zero' else
                                    args.llm_mid_scale if k == 'mid' else
                                    args.llm_pos_scale)))
                  for k, v in LLM_SAMPLE.items()}
        if args.llm_scale != 1.0 or args.llm_zero_scale != 1.0:
            log(f"llm-scale {args.llm_scale} zero-scale {args.llm_zero_scale}: {sum(LLM_SAMPLE.values())} -> "
                f"{sum(want.values())} pairs, ratio "
                f"{'unchanged' if args.llm_zero_scale == 1.0 else 'CHANGED (mix arm)'} {want}")
        pos_n = want["pos"]
        pos = mllm[mllm.target >= 0.5].sample(n=pos_n, random_state=SEED)
        mid = mllm[(mllm.target > 0) & (mllm.target < 0.5)].sample(
            n=want["mid"], random_state=SEED)
        zero = mllm[mllm.target == 0].sample(n=want["zero"], random_state=SEED)
        llm_sub = pd.concat([pos, mid, zero]).sample(
            frac=1.0, random_state=SEED).reset_index(drop=True)
        del pos, mid, zero, mllm
        log(f"llm subsample {len(llm_sub)} (pos {pos_n})")

    id2text = {}
    frames = [ih]
    if args.stage_a > 0 or args.replay > 0:
        frames.append(pd.read_parquet(f"{ROOT}/items_llm_subset.parquet"))
    # The treatment applies to the LLM frame ONLY. Feed tells have
    # mass_human=0, so human items are unaffected in substance -- and
    # keeping them on the legacy path means Stage B and inference stay
    # byte-identical to every shipped checkpoint and the container needs
    # no change. Stage A is where the provenance shortcut is learned.
    for fi, df in enumerate(frames):
        mode = args.tells if fi == 1 else "off"
        for row in df.itertuples(index=False):
            id2text[row.id] = make_text(row.name, row.category,
                                        row.attributes, mode)
    id2cat = dict(zip(ih["id"], ih["category"]))
    del frames
    log(f"texts for {len(id2text)} items")
    if llm_sub is not None:
        have = llm_sub.id1.isin(id2text.keys()) & llm_sub.id2.isin(id2text.keys())
        log(f"llm pairs with texts: {int(have.sum())}/{len(llm_sub)}")
        llm_sub = llm_sub[have].reset_index(drop=True)

    # ECUP_TRUST_REMOTE=1: EuroBERT ships its architecture as remote code
    # (eurolineage). Env-gated so every existing recipe is byte-identical.
    _trc = {"trust_remote_code": True} if os.environ.get("ECUP_TRUST_REMOTE") == "1" else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **_trc)
    device = torch.device("cuda")
    is_resume = os.path.isdir(args.model)
    if is_resume and os.path.getsize(
            os.path.join(args.model, "model.safetensors")) > 2 ** 31:
        # from_pretrained MMAPS, and mmap SIGSEGVs past 2 GiB on this box
        # (nommap.py). An fp32 568M checkpoint is 2.27 GB, so every mid-run
        # Stage-A checkpoint worth resuming from is exactly the file this API
        # cannot open. Reading the bytes ourselves is the whole fix.
        sys.path.insert(0, "/marimo")
        from nommap import load_model
        # _trc must ride along or a >2GiB EuroBERT checkpoint (fp32/bf16
        # stageA save) refuses its own remote code (t165euroA2 rc=1).
        model = load_model(args.model, num_labels=1, **_trc).to(device)
        log(f"loaded {args.model} WITHOUT mmap (>2 GiB)")
    else:
        # ATTN_IMPL env override (Box B FA2 validation; inert when unset).
        _attn = os.environ.get('ATTN_IMPL', '')
        _kw = {'attn_implementation': _attn} if _attn else {}
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1, ignore_mismatched_sizes=not is_resume,
            **_kw, **_trc).to(device)
        if _attn:
            log(f'attn_implementation={model.config._attn_implementation}')
    log(f"GPU={torch.cuda.get_device_name(0)}  resume={is_resume}")
    # ECUP_GRAD_CKPT=1: activation checkpointing for large backbones -- 610m
    # saves ~3x the activations of a BERT-shaped encoder and OOMs a 96GB card
    # at bs48@1024 without it (euro610price probes, 2026-08-26). Env-gated so
    # every existing recipe stays byte-identical. use_reentrant=False, and it
    # only engages in train mode (the probe bug: enabling on an eval-mode
    # model is a silent no-op that cost four bench rounds).
    if os.environ.get("ECUP_GRAD_CKPT") == "1":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        log("gradient checkpointing ON (use_reentrant=False)")
    if not args.val_only and next(model.parameters()).dtype != torch.float32:
        # Kaggle-restored checkpoints are fp16 (push_ckpt casts on push).
        # fp16 MASTER WEIGHTS NaN under AdamW after ONE step (nan_probe
        # 2026-08-24: sdpa and FA2 identically; bf16 autocast does not
        # protect the update). Inference in fp16 is fine -- val-only skips.
        model = model.float()
        log("fp16 checkpoint upcast to fp32 for training")
    # THE LOSS HAS NEVER BEEN AN EXPERIMENTAL VARIABLE. Every CE run on either
    # track uses plain BCE: grepping LEDGER + COOKBOOK + DIGEST for
    # pos_weight|focal|listwise|lambdarank|smooth-AP returns zero hits on any
    # training row. Stage B trains at ~26% positives; the board scores at ~4.5%
    # (boardprev). --pos-weight makes that a knob.
    #
    # WHAT IT CAN AND CANNOT DO, so the row is not over-read: PR-AUC is
    # invariant to any monotone rescaling of scores WITHIN a category, so
    # reweighting cannot move the Bayes-optimal ranking. It is a CAPACITY-
    # ALLOCATION intervention -- it changes what a finite model spends itself
    # on -- not a calibration fix. Aim it at where the damage actually is:
    # precision@top-1% is already 0.9888 (featexhausted), and pr_shape puts the
    # loss at recall 0.1-0.5.
    pw = None
    if args.pos_weight is not None and abs(args.pos_weight - 1.0) > 1e-9:
        pw = torch.tensor(args.pos_weight, device=device)
        log(f"pos_weight={args.pos_weight} "
            f"(1.0 = plain BCE, the shipped default; "
            f"0.134 matches board prevalence 0.045 from Stage B's ~0.26)")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    class PairDS(Dataset):
        def __init__(self, df):
            self.t1 = [id2text.get(i, "") for i in df.id1]
            self.t2 = [id2text.get(i, "") for i in df.id2]
            self.y = df.target.astype(np.float32).values
            self.w = (df["w"].astype(np.float32).values
                      if "w" in df.columns else None)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            if self.w is None:
                return self.t1[i], self.t2[i], self.y[i]
            return self.t1[i], self.t2[i], self.y[i], self.w[i]

    def collate(b):
        cols = list(zip(*b))
        t1, t2, y = cols[0], cols[1], cols[2]
        enc = tokenizer(list(t1), list(t2), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        if len(cols) == 4:
            enc["w"] = torch.tensor(cols[3], dtype=torch.float32)
        return enc

    def collate_nolabel(b):
        # zip(*b) yields 3 fields normally and 4 once --hard-weight adds 'w';
        # the old 't1, t2, _ = zip(*b)' raised ValueError on the second case,
        # which is any --hard-weight run that also reaches Stage C.
        cols = list(zip(*b))
        t1, t2 = cols[0], cols[1]
        return tokenizer(list(t1), list(t2), padding=True, truncation=True,
                         max_length=MAX_LEN, return_tensors="pt")

    def length_grouped_batches(ds, batch_size, buf_batches):
        """Permute -> sort by token length inside buffers -> shuffle batch order."""
        n = len(ds.y)
        lens = np.empty(n, dtype=np.int32)
        for s0 in range(0, n, 4096):
            e = tokenizer(list(ds.t1[s0:s0 + 4096]), list(ds.t2[s0:s0 + 4096]),
                          padding=False, truncation=True, max_length=MAX_LEN)
            lens[s0:s0 + 4096] = [len(x) for x in e["input_ids"]]
        perm = np.random.permutation(n)
        buf = batch_size * buf_batches
        order = np.concatenate([
            chunk[np.argsort(lens[chunk], kind="stable")]
            for chunk in (perm[i:i + buf] for i in range(0, n, buf))])
        batches = [order[i:i + batch_size].tolist()
                   for i in range(0, n - batch_size + 1, batch_size)]
        np.random.shuffle(batches)
        pad = sum(len(b) * int(lens[b].max()) for b in batches)
        log(f"group-by-length buf={buf_batches}: {len(batches)} batches, "
            f"padded tokens {pad/1e6:.1f}M vs {lens.sum()/1e6:.1f}M true "
            f"({100*(1-lens.sum()/pad):.0f}% padding)")
        return batches

    def _df_sig(df):
        # Cheap content signature over the epoch's dataframe. A resume replays
        # SAVED INDICES into a REBUILT dataframe -- if the deterministic rebuild
        # drifted (the silently-swapped-inputs trap, molab-volume memory), the
        # indices would silently train on the wrong rows. Refuse instead.
        import hashlib
        h = hashlib.sha256()
        h.update(str(len(df)).encode())
        for col in ("id1", "id2"):
            v = df[col].to_numpy()
            for sl in (v[:200], v[-200:]):
                h.update(sl.tobytes() if sl.dtype != object else
                         "|".join(map(str, sl)).encode())
        t = np.asarray(df.target.to_numpy(), dtype=np.float64)
        h.update(t[:200].tobytes() + t[-200:].tobytes())
        return h.hexdigest()[:16]

    def _save_resume_bundle(d, tag, gstep, full, batches, df):
        # Everything a seam-resume needs BESIDE the weights: the exact batch
        # order (so the epoch continues, not redraws), the RNG streams (dropout
        # continuity), and the position + data signature. NO optimizer state:
        # off-box that triples every push, and optblock measured the restart
        # axis dead at this dose (+-0.0019 band).
        if not os.path.exists(f"{d}/batch_order.npz"):
            np.savez_compressed(
                f"{d}/batch_order.npz",
                idx=np.concatenate([np.asarray(b, dtype=np.int64)
                                    for b in batches]),
                sizes=np.asarray([len(b) for b in batches], dtype=np.int32))
        torch.save({"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()}, f"{d}/rng_state.pt")
        json.dump({"tag": tag, "step": gstep, "n_batches": full,
                   "df_len": len(df), "df_sig": _df_sig(df),
                   "max_len": MAX_LEN, "train_seed": args.train_seed},
                  open(f"{d}/train_state.json", "w"))

    def train_epoch(df, batch_size, lr, tag, carry=None, resume=None):
        ds = PairDS(df)
        start, batches = 0, None
        if resume is not None:
            z = np.load(os.path.join(args.resume_partial, "batch_order.npz"))
            # np.load returns a LAZY NpzFile: every z[key] re-reads and
            # re-DECOMPRESSES the whole array out of the zip, and nothing is
            # cached (`z["idx"] is z["idx"]` is False). Indexing z["idx"]
            # inside the comprehension therefore paid one full 89 MB
            # decompression PER BATCH. Measured on box B, 2026-08-29:
            # 0.39 s per access x 87,359 batches = 9.4 HOURS to build a list
            # that takes seconds. Cost is quadratic in pairs and completely
            # silent -- no log line sits between here and the RESUME line --
            # so small draws only ever looked "a slow resume". Materialise
            # both arrays ONCE. See LEDGER `npzquadratic`.
            idx, sizes = z["idx"], z["sizes"]
            off = np.concatenate([[0], np.cumsum(sizes)])
            batches = [idx[off[k]:off[k + 1]].tolist()
                       for k in range(len(sizes))]
            start = int(resume["step"])
            if resume["df_len"] != len(df) or resume["df_sig"] != _df_sig(df):
                raise SystemExit(
                    f"{tag}: REFUSING resume -- rebuilt data does not match the "
                    f"bundle (len {len(df)} vs {resume['df_len']}, sig "
                    f"{_df_sig(df)} vs {resume['df_sig']}). The deterministic "
                    "rebuild drifted; do not train saved indices on it.")
            log(f"{tag}: RESUME at step {start}/{len(batches)} -- replaying the "
                "saved batch order, fresh optimizer, 100-step lr re-warmup")
        elif args.group_by_length:
            batches = length_grouped_batches(ds, batch_size,
                                             args.group_by_length)
        if batches is not None:
            dl = DataLoader(ds, batch_sampler=batches[start:],
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                            persistent_workers=True)
        else:
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=collate, drop_last=True, pin_memory=True,
                            persistent_workers=True)
        full = len(batches) if batches is not None else len(dl)
        def make_opt(params_or_groups):
            if args.optim == "nadam":
                return torch.optim.NAdam(params_or_groups, lr=lr,
                                         weight_decay=0.01,
                                         decoupled_weight_decay=True)
            return torch.optim.AdamW(params_or_groups, lr=lr, weight_decay=0.01)

        if carry is not None and "opt" in carry:
            # LEDGER epochreset: continue the SAME optimizer state and the SAME
            # schedule across epochs instead of a warm restart annealed to zero.
            opt, sched = carry["opt"], carry["sched"]
            log(f"{tag}: continuing optimizer+schedule "
                f"(lr now {sched.get_last_lr()[0]:.2e})")
        else:
            if args.llrd:
                groups = llrd_groups(model, lr, args.llrd)
                opt = make_opt(groups)
                log(f"{tag} llrd={args.llrd}: {len(groups)} lr groups, "
                    f"{groups[0]['lr']:.2e} (head) .. {groups[-1]['lr']:.2e} (deepest)")
            else:
                opt = make_opt([p for p in model.parameters() if p.requires_grad])
            # `full`, not len(dl): a resumed dl holds only the REMAINING
            # batches, but the schedule must be the one the epoch started on.
            horizon = full * (args.sched_epochs if carry is not None else 1)
            sched = get_linear_schedule_with_warmup(
                opt, int(full * WARMUP_RATIO), horizon)
            if carry is not None:
                carry["opt"], carry["sched"] = opt, sched
                log(f"{tag}: ONE schedule spanning {args.sched_epochs} epochs "
                    f"({horizon} steps, warmup {int(full * WARMUP_RATIO)})")
        if start:
            for _ in range(start):
                sched.step()
            log(f"{tag}: scheduler fast-forwarded to step {start} "
                f"(lr {sched.get_last_lr()[0]:.2e})")
            rng = torch.load(os.path.join(args.resume_partial, "rng_state.pt"),
                             weights_only=False)
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            log(f"{tag}: RNG streams restored from the partial")
        model.train()
        tot, nb = 0.0, 0
        for i, batch in enumerate(dl):
            gstep = start + i + 1
            labels = batch.pop("labels").to(device, non_blocking=True)
            wts = batch.pop("w", None)
            if wts is not None:
                wts = wts.to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=getattr(torch, AMP_DTYPE)):
                logits = model(**batch).logits.squeeze(-1).float()
                if wts is None:
                    loss = loss_fn(logits, labels)
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, weight=wts, pos_weight=pw)
            loss.backward()
            if start and i < 100:
                # Fresh-optimizer seam: damp the first steps so cold Adam
                # moments cannot take a full-lr jolt. LambdaLR recomputes lr
                # from base_lrs each step, so this scaling does not compound.
                for g in opt.param_groups:
                    g["lr"] *= (i + 1) / 100.0
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            nb += 1
            if gstep % 500 == 0:
                log(f"{tag} step {gstep}/{full} loss={tot / nb:.4f}")
            # The molab sandbox terminates without warning -- twice in four
            # hours on 2026-08-20 -- and Stage A is ONE epoch of tens of
            # thousands of steps, so between-epoch checkpoints mean a kill at
            # 90% costs everything. Write a rolling mid-epoch checkpoint so the
            # worst case is CKPT_EVERY steps, not the whole stage. Cost is one
            # ~1.1 GB write per interval, a few seconds against ~40 min of
            # training per interval.
            if CKPT_EVERY and gstep % CKPT_EVERY == 0:
                model.save_pretrained(f"{ckpt}/{tag}-partial", safe_serialization=True)
                tokenizer.save_pretrained(f"{ckpt}/{tag}-partial")
                if batches is not None and tag.startswith("llm-ep"):
                    # the resume bundle rides in the partial dir and off-box
                    # with it -- a wiped volume restores a RESUMABLE state
                    _save_resume_bundle(f"{ckpt}/{tag}-partial", tag, gstep,
                                        full, batches, df)
                log(f"{tag} mid-epoch checkpoint at step {gstep} -> {tag}-partial")
                # AND SHIP IT. The rolling checkpoint above protected against a
                # crash; it did NOT protect against the box vanishing, because
                # nothing carried it off. push_off_box only fired at EPOCH
                # boundaries, so the unprotected window was one whole epoch --
                # about 2.3h at our usual volume. On 2026-08-22 t119vol2x
                # doubled the LLM pool to 4.4M pairs, which doubled that window
                # to ~4.5h, and the sandbox died 92 minutes in. Doubling the
                # data silently doubled the exposure and defeated the exact
                # protection added for this. Push on a WALL-CLOCK cadence, so
                # the window is a constant that does not scale with the
                # experiment.
                global _LAST_PUSH
                if args.push_epochs and time.time() - _LAST_PUSH > PUSH_EVERY_S:
                    _LAST_PUSH = time.time()
                    push_off_box(f"{ckpt}/{tag}-partial",
                                 f"ecup26-{args.tag}-{tag}-partial", log)
        log(f"{tag} done, {nb} steps, mean loss={tot / max(nb, 1):.4f}")

    @torch.no_grad()
    def run_model(ds, bs=int(os.environ.get("ECUP_EVAL_BS", "512"))):
        # 512 is right on a box with 80-96GB. A Kaggle T4 has 14.56GB and
        # OOMs inside apply_rotary_pos_emb at 1024 tokens with uncapped
        # text -- 14.4 minutes in, at the END of the pass. Batch size
        # changes which pairs share a batch and nothing else (padding is
        # masked), so it does not move a score; it must still be held
        # EQUAL across arms of one comparison, since bf16/fp16 reduction
        # order does shift the last digits.
        # LENGTH-SORTED EVAL (LEDGER sortkey). Unsorted batches pad to an
        # arbitrary longest member: 4.07x the ideal padded-token count at 1024,
        # i.e. 67% of the scoring pass is padding. Sorting by TRUE TOKEN length
        # brings it to 1.36x. Sort by tokens, NOT characters -- character length
        # is only r=0.9603 with token length and that residual is exactly the
        # bug sortkey found in the shipped container.
        #
        # This reorders BATCH COMPOSITION only. Every pair is still scored with
        # exactly the tokens it would have had, so it is unrelated to the
        # reordertrunc/windowsat order-specialisation kill. Padding is masked,
        # so scores are unchanged up to bf16 reduction order (~1e-5, far under
        # the 0.0006 E_real floor) -- but that is NOT bit-identical, so the
        # first run under this code should not be compared bit-for-bit with an
        # older number. Set SORTED_EVAL=0 to fall back.
        order = None
        if os.environ.get("SORTED_EVAL", "1") != "0" and hasattr(ds, "t1"):
            t1, t2 = ds.t1, ds.t2
            lens = np.empty(len(t1), dtype=np.int32)
            for s0 in range(0, len(t1), 4096):
                e = tokenizer(list(t1[s0:s0 + 4096]), list(t2[s0:s0 + 4096]),
                              padding=False, truncation=True, max_length=MAX_LEN)
                lens[s0:s0 + 4096] = [len(x) for x in e["input_ids"]]
            order = np.argsort(lens, kind="stable")
            ds = torch.utils.data.Subset(ds, order.tolist())
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_nolabel)
        model.eval()
        out = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=getattr(torch, AMP_DTYPE)):
                out.append(torch.sigmoid(model(**batch).logits.squeeze(-1).float()).cpu())
        res = torch.cat(out).numpy()
        if order is None:
            return res
        unsorted = np.empty_like(res)
        unsorted[order] = res
        return unsorted

    def macro(df, s):
        c = df.id1.map(id2cat).to_numpy()
        y_ = df.target.to_numpy() > 0.5
        per = {}
        for cc in np.unique(c):
            k = c == cc
            if k.sum() >= 2 and y_[k].min() != y_[k].max():
                per[str(cc)] = float(average_precision_score(y_[k], s[k]))
        per["MACRO"] = float(np.mean(list(per.values())))
        return per

    if args.val_only:
        # Every score_universe() call sits BELOW the --val-only return, so this
        # block is pure cost on a val-only read: two Python lists of N universe
        # texts, which is gigabytes of string objects. That is affordable on the
        # box and is NOT affordable on a Kaggle kernel, where enabling the GPU
        # halves host RAM to ~13GB and an overrun arrives as a bare "Killed".
        log("val-only: universe not loaded (nothing below here scores it)")

        def score_universe(name):
            raise RuntimeError("universe not loaded under --val-only")
    else:
        uv = pd.read_parquet(f"{STOR}/universe_view.parquet",
                             columns=["pid1", "pid2", "category", "name1", "name2",
                                      "attributes1", "attributes2"])
        # ALWAYS legacy: this is what the shipped container computes, and the
        # tells have mass_human=0 so a human-pool pair is byte-identical anyway.
        ut1 = [make_text(n, c, a) for n, c, a in zip(uv.name1, uv.category, uv.attributes1)]
        ut2 = [make_text(n, c, a) for n, c, a in zip(uv.name2, uv.category, uv.attributes2)]

        class UDS(Dataset):
            def __init__(self):
                self.t1, self.t2 = ut1, ut2

            def __len__(self):
                return len(ut1)

            def __getitem__(self, i):
                return ut1[i], ut2[i], 0.0

        def score_universe(name):
            s = run_model(UDS())
            np.savez_compressed(f"{ROOT}/ce_scores_{name}.npz", ce=s,
                                pid1=uv.pid1.to_numpy(), pid2=uv.pid2.to_numpy())
            log(f"universe scored -> ce_scores_{name}.npz (mean {s.mean():.4f})")

    m["cat"] = m.id1.map(id2cat)
    m["strat"] = m["cat"].astype(str) + "_" + m.target.astype(int).astype(str)
    tr_h, va_h = train_test_split(m, test_size=0.2, random_state=SEED, stratify=m["strat"])
    tr_h, va_h = tr_h.reset_index(drop=True), va_h.reset_index(drop=True)
    log(f"human train={len(tr_h)} val={len(va_h)}")

    if args.cats:
        # percatft: the specialist arm must differ from the pooled control by
        # DATA RESTRICTION ONLY. Filtering sits here on purpose -- AFTER the
        # global split (each category's val rows stay bit-identical to every
        # pooled checkpoint's, so the within-category read is paired) and
        # AFTER the global SEED-42 draw (Stage-A rows are an exact subset of
        # the pooled arms'). llm categories come from items_llm_subset, the
        # same file the text builder reads.
        keep = {c.strip() for c in args.cats.split(";") if c.strip()}
        unknown = keep - set(m.cat.unique())
        assert not unknown, f"--cats: unknown categories {sorted(unknown)}"
        tr_h = tr_h[tr_h.cat.isin(keep)].reset_index(drop=True)
        va_h = va_h[va_h.cat.isin(keep)].reset_index(drop=True)
        if llm_sub is not None:
            ilc = pd.read_parquet(f"{ROOT}/items_llm_subset.parquet",
                                  columns=["id", "category"])
            lcat = llm_sub.id1.map(dict(zip(ilc.id, ilc.category)))
            del ilc
            llm_sub = llm_sub[lcat.isin(keep).values].reset_index(drop=True)
        log(f"--cats {sorted(keep)}: human train={len(tr_h)} val={len(va_h)} "
            f"llm={0 if llm_sub is None else len(llm_sub)}")
        assert len(tr_h) and len(va_h), "--cats filtered everything out"

    if args.train_seed is not None:
        # AFTER the split on purpose: the split, the llm subsample and the
        # Stage-C ballast must stay identical across arms or the comparison is
        # meaningless. This varies dropout and batch order only -- the actual
        # run-to-run noise for a resume-based arm.
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed_all(args.train_seed)
        np.random.seed(args.train_seed)
        random.seed(args.train_seed)
        log(f"train-seed={args.train_seed} (split/sampling stay on SEED={SEED})")

    if args.category_idx is not None:
        # percat: one specialist per category. AFTER the split and train-seed
        # on purpose -- the split must stay identical across all 21 arms
        # (control included) or the comparison confounds selection with
        # specialization.
        _cats = sorted(ih.category.unique())
        assert 0 <= args.category_idx < len(_cats), \
            f"category-idx {args.category_idx} outside 0..{len(_cats) - 1}"
        _c = _cats[args.category_idx]
        tr_h = tr_h[tr_h["cat"] == _c].reset_index(drop=True)
        va_h = va_h[va_h["cat"] == _c].reset_index(drop=True)
        log(f"category-idx {args.category_idx}: {_c!r} -- "
            f"train {len(tr_h)} val {len(va_h)}")

    if args.mined_neg:
        # The board's negatives are near-duplicates; ours are whatever the
        # organiser sampled. v11 scores 0.0844 macro-AP on the mined slice
        # against v9's 0.0837 — three board points of improvement moved that
        # number by nothing, so the hard regime has never been trained on.
        # The split is ITEM-disjoint (outputs/out-t102/mined_split.npz): no item
        # in this training half appears in the half we evaluate on.
        sp = np.load(f"{ROOT}/mined_split.npz")["split"]
        hard = uv[sp == 1][["pid1", "pid2"]].rename(columns={"pid1": "id1", "pid2": "id2"})
        hard["target"] = 0.0
        miss = (~hard.id1.isin(id2text.keys()) | ~hard.id2.isin(id2text.keys())).sum()
        log(f"hard negatives: {len(hard)} mined pairs added to Stage B "
            f"({len(hard) / len(tr_h):.1%} of the human half); missing texts {miss}")
        tr_h = pd.concat([tr_h[["id1", "id2", "target"]], hard], ignore_index=True)
        tr_h = tr_h.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        log(f"Stage B pool now {len(tr_h)} rows, positive rate {(tr_h.target > 0.5).mean():.4f}")

    if args.hard_weight and args.hard_weight > 0:
        # LEDGER hardpairw: x-weight the slice the board grades hardest --
        # identical-name negatives and disagreeing-name positives.
        #
        # PROVENANCE CORRECTED 2026-08-24. This comment used to say team04-2024
        # "wrote exactly this and rejected it on their LEAKY random pair split
        # (19/35 delta-sign agreement)". That rejection never happened. Checked
        # against their raw log: `dать веса объектам` sits at row 71 of
        # experiments_log.csv in a TRAILING BACKLOG (rows 71-75), after their
        # last logged run at row 65, with no date, no score and no result --
        # they never ran it. The 19/35 is a GLOBAL statistic about their whole
        # log (FINDINGS.md:105): across 36 submitted runs their local-vs-board
        # delta-sign agreement was a coin flip. I welded a global number onto a
        # specific experiment and manufactured a prior-art rejection out of it.
        # The 2024 material says NOTHING about this idea's outcome, only that a
        # 4th-place team thought of it and ran out of time.
        #
        # WHAT THE FLAG ACTUALLY DOES, which matters more than its pedigree --
        # see LEDGER `hwmechanism`: it upweights 1992 identical-name negatives
        # and 67081 disagreeing-name positives, so 97.1% of the reweighted rows
        # are POSITIVES (89.3% of every positive in the pool) and the effective
        # pos:neg weight ratio is ~1.84. Weights touch Stage B ONLY.
        id2name = dict(zip(ih["id"], ih["name"]))
        _n1 = tr_h.id1.map(id2name).fillna("").values
        _n2 = tr_h.id2.map(id2name).fillna("").values

        def _iou(a, b):
            sa, sb = set(a.split()), set(b.split())
            u = len(sa | sb)
            return (len(sa & sb) / u) if u else 1.0

        _io = np.fromiter((_iou(a, b) for a, b in zip(_n1, _n2)),
                          dtype=np.float32, count=len(_n1))
        _neg_hard = (tr_h.target.values < 0.5) & (_n1 == _n2)
        _pos_hard = ((tr_h.target.values >= 0.5) & (_n1 != _n2)
                     & (_io < 0.7))
        tr_h = tr_h.copy()
        tr_h["w"] = np.where(_neg_hard | _pos_hard, args.hard_weight,
                             1.0).astype(np.float32)
        log(f"hard-weight {args.hard_weight}: "
            f"{int(_neg_hard.sum())} identical-name negs + "
            f"{int(_pos_hard.sum())} disagreeing-name pos upweighted "
            f"of {len(tr_h)} Stage-B rows")

    if args.val_only:
        # Evaluate a checkpoint on the SAME human-val split every training run
        # reports, by running the same code that builds it -- a standalone
        # re-implementation would drift from the split it is being compared to,
        # and the comparison IS the experiment.
        r = macro(va_h, run_model(PairDS(va_h)))
        log(f"VAL-ONLY {args.model}: MACRO={r['MACRO']:.4f}")
        for k in sorted(r):
            log(f"   {k:28s} {r[k]:.4f}")
        json.dump(r, open(f"{ROOT}/valonly_{args.tag}.json", "w"),
                  indent=2, ensure_ascii=False)
        log("DONE")
        return

    metrics = {"args": vars(args)}
    # LEDGER freezeab. The freeze is a STAGE property, not a run property: the
    # registered design holds the lower layers fixed while 11M noisy LLM labels
    # go past, then hands them back for the 292k human pairs. Applying it at
    # each boundary is what makes "a" separable from "ab" as one variable.
    set_frozen(model, args.freeze_layers if "a" in args.freeze_stage else 0, log)
    for ep in range(args.stage_a):
        if _rp is not None and ep < int(_rp["tag"][6:]):
            # this epoch's work is already inside the resumed weights
            log(f"llm-ep{ep}: skipped (resume-partial is inside "
                f"{_rp['tag']})")
            continue
        if _rp is not None and ep == int(_rp["tag"][6:]):
            train_epoch(llm_sub, args.batch_llm, args.lr_llm, f"llm-ep{ep}",
                        resume=_rp)
            _rp = None          # epochs after the seam run normally
        else:
            train_epoch(llm_sub, args.batch_llm, args.lr_llm, f"llm-ep{ep}")
        model.save_pretrained(f"{ckpt}/stageA", safe_serialization=True)
        tokenizer.save_pretrained(f"{ckpt}/stageA")
        if args.push_epochs:
            push_off_box(f"{ckpt}/stageA", f"ecup26-{args.tag}-stagea", log)
        r = macro(va_h, run_model(PairDS(va_h)))
        metrics[f"val_after_llm_ep{ep}"] = r
        log(f"VAL after Stage A ep{ep}: MACRO={r['MACRO']:.4f}")
        with open(f"{ROOT}/metrics_{args.tag}.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    set_frozen(model, args.freeze_layers if "b" in args.freeze_stage else 0, log)
    _sched_carry = {} if args.sched_epochs > 0 else None
    for ep in range(args.stage_b):
        idx = args.resume_epoch + ep
        df_b = tr_h
        if args.replay > 0:
            # LEDGER `replayb`: fresh draw per epoch so each replay row is seen
            # once, matching alex's each-selected-row-seen-once. Same soft
            # targets Stage A trained on -- replay preserves, never re-labels.
            # With --replay-file the pool is the given parquet instead
            # (qylabel: judge-corrected rows, where re-labelling IS the point).
            pool = (pd.read_parquet(args.replay_file)
                    if args.replay_file else llm_sub)
            rep = pool.sample(n=min(args.replay, len(pool)),
                              random_state=SEED + 1000 + idx)
            df_b = pd.concat([tr_h[["id1", "id2", "target"]],
                              rep[["id1", "id2", "target"]]],
                             ignore_index=True)
            log(f"human-ep{idx}: +{len(rep)} llm replay rows "
                f"({len(rep)/len(df_b):.1%} of the epoch)")
        train_epoch(df_b, args.batch_human, args.lr_human, f"human-ep{idx}",
                    carry=_sched_carry)
        r = macro(va_h, run_model(PairDS(va_h)))
        metrics[f"val_after_human_ep{idx}"] = r
        log(f"VAL after human ep{idx}: MACRO={r['MACRO']:.4f}")
        if not args.no_save:
            model.save_pretrained(f"{ckpt}/ep{idx}", safe_serialization=True)
            tokenizer.save_pretrained(f"{ckpt}/ep{idx}")
            if args.push_epochs:
                push_off_box(f"{ckpt}/ep{idx}", f"ecup26-{args.tag}-ep{idx}", log)
            score_universe(f"{args.tag}-ep{idx}")
        with open(f"{ROOT}/metrics_{args.tag}.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    # ------------------------------------------------------------- Stage C
    # oleg's curriculum stage, ported from molab_train_stagec.py + his TG spec
    # of 2026-08-23: mine the model's OWN hard pairs from the human TRAIN half
    # (val is never touched, so the read stays comparable to every other run),
    # freeze everything except the top --stagec-top layers, fine-tune at a
    # small lr. MONOTONIC BY CONSTRUCTION: the baseline val is measured first
    # and every Stage C epoch is checkpointed and scored separately, so a
    # Stage C that hurts leaves the Stage B checkpoint as the deliverable and
    # costs nothing but GPU time.
    if args.stage_c > 0:
        r0 = macro(va_h, run_model(PairDS(va_h)))
        log(f"Stage C baseline (pre-C) val: MACRO={r0['MACRO']:.4f}")
        metrics["val_before_hard"] = r0

        log(f"Stage C: scoring the human train half ({len(tr_h)} pairs) with "
            f"the current model to mine hard examples")
        t = tr_h[["id1", "id2", "target"]].copy()
        t["s"] = run_model(PairDS(tr_h))
        hard_neg = t[t.target <= 0.5].nlargest(args.hard_k, "s")
        hard_pos = t[t.target > 0.5].nsmallest(min(args.hard_k,
                                                   int((t.target > 0.5).sum())),
                                               "s")
        ballast = t.sample(n=min(2 * args.hard_k, len(t)), random_state=SEED)
        hc = pd.concat([hard_neg, hard_pos, ballast])[["id1", "id2", "target"]]
        hc = hc.drop_duplicates().sample(frac=1.0, random_state=SEED)
        hc = hc.reset_index(drop=True)
        log(f"Stage C pool: {len(hard_neg)} hard_neg (score >= "
            f"{hard_neg.s.min():.3f}) + {len(hard_pos)} hard_pos (score <= "
            f"{hard_pos.s.max():.3f}) + {len(ballast)} ballast = {len(hc)} "
            f"rows after dedup, positive rate {(hc.target > 0.5).mean():.3f}")

        emb, enc_layers = _backbone_layers(model)
        keep = max(1, min(args.stagec_top, len(enc_layers)))
        set_frozen(model, len(enc_layers) - keep, log)
        log(f"Stage C: only the top {keep}/{len(enc_layers)} layers train")

        best_m, best_ep = r0["MACRO"], None
        for ep in range(args.stage_c):
            train_epoch(hc, args.batch_human, args.lr_hard, f"hard-ep{ep}")
            rc = macro(va_h, run_model(PairDS(va_h)))
            metrics[f"val_after_hard_ep{ep}"] = rc
            log(f"VAL after Stage C ep{ep}: MACRO={rc['MACRO']:.4f} "
                f"(pre-C baseline {r0['MACRO']:.4f})")
            model.save_pretrained(f"{ckpt}/hardep{ep}", safe_serialization=True)
            tokenizer.save_pretrained(f"{ckpt}/hardep{ep}")
            if args.push_epochs:
                push_off_box(f"{ckpt}/hardep{ep}",
                             f"ecup26-{args.tag}-hardep{ep}", log)
            score_universe(f"{args.tag}-hardep{ep}")
            with open(f"{ROOT}/metrics_{args.tag}.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            if rc["MACRO"] > best_m:
                best_m, best_ep = rc["MACRO"], ep
        log("Stage C best: %s MACRO=%.4f" % (
            "hard-ep%d" % best_ep if best_ep is not None
            else "the PRE-C checkpoint (Stage C did not improve val)", best_m))
    log("DONE")


if __name__ == "__main__":
    main()
