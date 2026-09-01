"""Static scope check of the CE-3 block inside a built archive's run.py.

WHY THIS EXISTS. The grader-image run is the real gate, and it has NOT passed:
the one container test that ran died inside CE-1 (unchanged v39 code, slow local
GPU) before CE-3 was ever reached, and Docker then wedged. So the new stage has
zero execution evidence.

This does not replace that gate. What it DOES catch is the failure class a
never-executed code path is most likely to carry: a name that does not exist in
scope. `run.py` builds its CE section inside deeply nested try/else blocks, so a
variable I assumed was in scope (`_ce1_secs`, `ce_scores`, `_band_rank`, `HERE`,
`deadline_ts`, `total_budget`, `T0`, `_alarm`) might simply not be bound where I
inserted the block -- and that would surface as a NameError only at runtime, on
the grader, once, with a submission spent.

Method: parse run.py, walk main()'s body, collect every name bound BEFORE the
CE-3 block (assignments, imports, function args, comprehension targets) plus
module-level names and builtins, then assert every name the CE-3 block LOADS is
in that set.
"""
import ast
import builtins
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "models/ce-3"


def bound_names(node, out):
    """Every name this statement binds."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)


def main():
    src = zipfile.ZipFile(sys.argv[1]).read("run.py").decode("utf-8")
    tree = ast.parse(src)

    scope = set(dir(builtins))
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.add(node.name)
        else:
            bound_names(node, scope)
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            fn = node
    if fn is None:
        raise SystemExit("ABORT: no main() in run.py")
    bound_names(fn.args, scope)

    # find the CE-3 block: the statement subtree that mentions "models/ce-3"
    target, lineno = None, None
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and MARK in node.value:
            lineno = node.lineno
    if lineno is None:
        raise SystemExit("ABORT: no CE-3 block found (is this a v42-class archive?)")

    # everything textually before the block binds into scope
    for node in ast.walk(fn):
        if isinstance(node, ast.stmt) and node.lineno < lineno:
            bound_names(node, scope)

    # collect the block: all statements in main() at/after lineno
    loads, block_lines = set(), 0
    for node in ast.walk(fn):
        if isinstance(node, ast.stmt) and node.lineno >= lineno:
            block_lines += 1
            bound_names(node, scope)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.lineno >= lineno:
            loads.add(node.id)

    missing = sorted(n for n in loads if n not in scope)
    print("CE-3 block starts at run.py line %d (%d statements at/after it)"
          % (lineno, block_lines))
    print("names loaded by the block: %d" % len(loads))
    key = ["ce_scores", "_band_rank", "_rank01", "_alarm", "np", "os", "time",
           "HERE", "args", "deadline_ts", "total_budget", "T0", "_ce1_secs",
           "log", "ce"]
    for k in key:
        if k in loads:
            print("   %-14s %s" % (k, "BOUND" if k in scope else "*** NOT BOUND ***"))
    if missing:
        raise SystemExit("FAIL: names used but never bound: %s" % missing)
    print("PASS: every name the CE-3 block loads is bound in scope")


if __name__ == "__main__":
    main()
