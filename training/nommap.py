"""Load a safetensors checkpoint WITHOUT mmap, because mmap is broken here.

THE BUG, measured 2026-08-22 on the replacement molab box. Every checkpoint
over 2 GiB segfaults the interpreter on read; every checkpoint under it reads
fine. The split is exactly 2^31 bytes and it is nothing to do with the files:

    dd if=<2271071852-byte ckpt> of=/dev/null   ->  all 2271071852 bytes, 3.0 GB/s
    safetensors.load_file(same path)            ->  SIGSEGV
    safetensors.torch.load(open(path,'rb').read()) -> loads, checksum sane

So the storage is healthy and the MMAP path is not. load_file mmaps, and so
does transformers' from_pretrained, which means an fp32 568M checkpoint cannot
be loaded, scored, or resumed by the normal API on this box.

WHAT THIS COST BEFORE IT WAS UNDERSTOOD: three separate crashes read as three
different things -- a bad statistic, a bad tensor, a corrupt volume -- and one
of them killed the marimo kernel outright (a native crash inside the kernel
leaves /api/status 'healthy' while every execute silently returns nothing).
The checkpoints declared lost to the fourth box death were never lost; they
were unreadable through one API.

TWO WAYS AROUND IT, both used here:
  * read the bytes yourself (this module), for anything that must stay fp32;
  * store fp16 (1.14 GB, under the limit), which is what push_ckpt.py already
    ships and what the grader casts to bf16 anyway -- so for everything on the
    inference path the fix costs nothing.
"""
import pathlib

import torch
from safetensors.torch import load


def load_sd(d, dtype=None):
    """State dict from a checkpoint directory, bypassing mmap."""
    f = pathlib.Path(d)
    if f.is_dir():
        f = f / "model.safetensors"
    sd = load(open(f, "rb").read())
    if dtype is not None:
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    return sd


def load_model(d, cls=None, **kw):
    """from_pretrained equivalent that never mmaps the weights."""
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cls = cls or AutoModelForSequenceClassification
    # remote-code checkpoints (EuroBERT) need the flag at BOTH steps: config
    # resolution and model-class resolution from auto_map.
    _trc = ({"trust_remote_code": kw["trust_remote_code"]}
            if "trust_remote_code" in kw else {})
    cfg = AutoConfig.from_pretrained(d, **kw)
    try:
        model = cls.from_config(cfg, **_trc)
    except TypeError:  # older transformers: from_config takes no trc
        model = cls.from_config(cfg)
    missing, unexpected = model.load_state_dict(load_sd(d, torch.float32), strict=False)
    if missing or unexpected:
        raise SystemExit(f"{d}: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return model
