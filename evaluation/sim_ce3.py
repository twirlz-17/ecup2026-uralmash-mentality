"""Execute the CE-3 block FROM THE BUILT ARCHIVE against stubs, all three paths.

    python tools/sim_ce3.py submission/dist/submission_v46.zip

WHY. The grader-image run is the real gate and Docker will not start on this
machine. v37 scored 0.3611536 from a container shipped unrun, so shipping this
blind is exactly the failure this project has already paid for once. This does
not replace that gate -- it cannot see transformers 5.14.1, CUDA, or real
timings. What it CAN do is run the ACTUAL SOURCE TEXT lifted out of the built
zip (not a re-implementation of it) through every branch, with numpy real and
everything else stubbed, and check the arithmetic and control flow.

The three paths that matter, and what each must produce:

  NORMAL   -- budget is ample, ce_scores returns finite scores on the band.
              `ce` must change, stay finite, stay in [0,1]-ish rank space, and
              the top of the ranking must be perturbed but not scrambled.
  SKIP     -- budget is short. `ce` must be returned COMPLETELY UNTOUCHED, i.e.
              exactly the v45 two-CE result. This is the branch the fixed guard
              exists for: FORCE_CE=1 sets deadline_ts=None, so the original
              deadline_ts-based check could never fire (`watchdogkills`).
  RAISES   -- ce_scores throws. The except must swallow it and leave `ce`
              untouched, again exactly v45.

A change to `ce` in SKIP or RAISES is a FAIL: it would mean a degraded run
silently ships something that is neither the three-CE nor the two-CE answer.
"""
import ast
import sys
import types
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def extract_block(src, var="ce3_dir"):
    """Pull the CE-3 statements verbatim out of run.py."""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == var:
                out.append(node)
    if not out:
        raise SystemExit("ABORT: no %s assignment found" % var)
    assign = out[0]
    # the if/elif/else immediately following it, at the same indent
    follow = None
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and node.lineno > assign.lineno \
                and node.col_offset == assign.col_offset:
            if follow is None or node.lineno < follow.lineno:
                follow = node
    if follow is None:
        raise SystemExit("ABORT: no if-block after ce3_dir")
    # get_source_segment strips the leading indent of the FIRST line only, so
    # rebuild from raw lines and dedent the whole span together -- otherwise the
    # nested `elif` keeps its original 20-space indent and will not compile.
    lines = src.splitlines()
    start = assign.lineno - 1
    end = follow.end_lineno
    span = lines[start:end]
    import textwrap
    seg = textwrap.dedent("\n".join(span)) + "\n"
    if seg.lstrip() != seg.lstrip(" "):
        raise SystemExit("ABORT: dedent left mixed indentation")
    return seg


def run_case(seg, name, *, budget_ok, raises, missing_dir=False):
    n = 4000
    rng = np.random.default_rng(0)
    ce_before = np.sort(rng.random(n))[::-1].copy()   # a plausible blended rank vector
    logs = []

    def log(m):
        logs.append(str(m))

    def ce_scores(items_path, matches_path, model_dir, batch_size, max_len,
                  deadline_ts, log, subset=None):
        if raises:
            raise RuntimeError("simulated CE-3 failure")
        out = np.full(n, np.nan)
        out[subset] = rng.random(len(subset))     # only the band is scored
        return out, True

    def _band_rank(raw, sel, nn):
        from scipy.stats import rankdata
        k = len(sel)
        lo = 1.0 - k / float(nn)
        o = np.full(nn, lo, dtype=np.float64)
        if k > 1:
            o[sel] = lo + (1.0 - lo) * (rankdata(raw[sel]) / k)
        return o

    import os as _os
    import time as _time

    class _OS:
        environ = {}
        path = _os.path
    fake_os = _OS()
    fake_os.environ = {}

    T0 = _time.time() - (10 if budget_ok else 355)   # 355s in on a 360s budget
    g = {
        "os": fake_os, "time": _time, "np": np,
        "HERE": "/app" if not missing_dir else "/nonexistent",
        "log": log, "ce": ce_before.copy(), "ce_scores": ce_scores,
        "_band_rank": _band_rank, "_alarm": lambda *a: None,
        "T0": T0, "total_budget": 360.0, "deadline_ts": None,
        "_ce1_secs": 120.0,
        "args": types.SimpleNamespace(items_path="i", matches_path="m"),
    }
    # make os.path.isdir answer for the CE-3 dir
    real_isdir = _os.path.isdir
    fake_os.path = types.SimpleNamespace(
        join=_os.path.join, isdir=lambda p: not missing_dir)

    exec(compile(seg, "<ce3-block>", "exec"), g)
    ce_after = g["ce"]
    changed = not np.array_equal(ce_before, ce_after)
    finite = bool(np.isfinite(ce_after).all())
    print("  %-8s changed=%-5s finite=%-5s | %s"
          % (name, changed, finite, " / ".join(logs) or "(no log)"))
    return changed, finite, logs


def main():
    var = sys.argv[2] if len(sys.argv) > 2 else "ce3_dir"
    src = zipfile.ZipFile(sys.argv[1]).read("run.py").decode("utf-8")
    seg = extract_block(src, var)
    print("extracted CE-3 block: %d lines\n" % len(seg.splitlines()))

    ok = True
    ch, fin, _ = run_case(seg, "NORMAL", budget_ok=True, raises=False)
    if not (ch and fin):
        print("  FAIL: normal path must modify ce and stay finite"); ok = False
    ch, fin, lg = run_case(seg, "SKIP", budget_ok=False, raises=False)
    if ch:
        print("  FAIL: short-budget path MUST leave ce untouched"); ok = False
    if not any("SKIP" in s for s in lg):
        print("  FAIL: short-budget path did not log a SKIP"); ok = False
    ch, fin, lg = run_case(seg, "RAISES", budget_ok=True, raises=True)
    if ch:
        print("  FAIL: exception path MUST leave ce untouched"); ok = False
    ch, fin, lg = run_case(seg, "NO-DIR", budget_ok=True, raises=False,
                           missing_dir=True)
    if ch:
        print("  FAIL: absent models/ce-3 MUST leave ce untouched"); ok = False

    print()
    if not ok:
        raise SystemExit("SIMULATION FAILED")
    print("PASS: normal path blends; SKIP, RAISES and NO-DIR all degrade to the "
          "untouched two-CE result (i.e. exactly v45)")


if __name__ == "__main__":
    main()
