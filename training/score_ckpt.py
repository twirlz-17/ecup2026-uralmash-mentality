"""Score the frozen 415k ruler universe with one or more CE checkpoints.

Byte-identical text construction and inference path to train_t102.py's own
universe pass, so t98b's e5-base scores and t102's e5-large scores are
comparable numbers rather than two different measurements.

Usage: python score_ckpt.py [--max-len N] <tag> <ckpt_dir> [<tag> <ckpt_dir> ...]

--max-len must MATCH the length the checkpoint was trained at. Running a
checkpoint off its training length is not a free knob: the e5-base ep1 model
measured -0.067 at 384 and -0.090 at 512 (LEDGER t109probe).
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STOR = "/marimo/storage"
ROOT = "/marimo"
ARGV = sys.argv[1:]
MAX_LEN = 256
if ARGV and ARGV[0] == "--max-len":
    MAX_LEN = int(ARGV[1])
    ARGV = ARGV[2:]
ATTR_CLEAN = re.compile(r'[{}\[\]"]')
t0 = time.time()


def log(m):
    print(f"[{(time.time() - t0) / 60:7.1f} min] {m}", flush=True)


uv = pd.read_parquet(f"{STOR}/universe_view.parquet",
                     columns=["pid1", "pid2", "y", "category", "name1", "name2",
                              "attributes1", "attributes2"])
texts1 = [f"{n} [SEP] {c} [SEP] " + re.sub(r"\s+", " ", ATTR_CLEAN.sub(" ", str(a)))[:1500]
          for n, c, a in zip(uv.name1, uv.category, uv.attributes1)]
texts2 = [f"{n} [SEP] {c} [SEP] " + re.sub(r"\s+", " ", ATTR_CLEAN.sub(" ", str(a)))[:1500]
          for n, c, a in zip(uv.name2, uv.category, uv.attributes2)]
log(f"universe {len(uv)} rows")
device = "cuda"

args = ARGV
for tag, path in zip(args[::2], args[1::2]):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    class TDS(Dataset):
        def __len__(self):
            return len(texts1)

        def __getitem__(self, i):
            return texts1[i], texts2[i], 0.0

    def colT(b):
        a_, b_, _ = zip(*b)
        return tokenizer(list(a_), list(b_), padding=True, truncation=True,
                         max_length=MAX_LEN, return_tensors="pt")

    dl = DataLoader(TDS(), batch_size=512, shuffle=False, num_workers=8, collate_fn=colT)
    out = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.squeeze(-1)
            out.append(torch.sigmoid(logits.float()).cpu())
    scores = torch.cat(out).numpy()
    np.savez_compressed(f"{ROOT}/ce_scores_{tag}.npz", ce=scores,
                        pid1=uv.pid1.to_numpy(), pid2=uv.pid2.to_numpy())
    log(f"{tag}: scored -> ce_scores_{tag}.npz  (mean {scores.mean():.4f})")
    del model
    torch.cuda.empty_cache()
log("DONE")
