"""Assemble kaggle-t94/exp.py -- the v8 export -- by patching t70's v7 export.

v8 = v7 + `bbig`: the per-category B models go from 31 leaves / 800 rounds to
63 leaves / 2000 rounds. That is the whole change. Everything else -- the
big/mc25 name table, the gating, the 0.5/0.5 rank blend, the phase-1
leak-free measurement -- is v7's and is left alone.

WHY min_data_in_leaf 2000 IS NOT IN THIS EXPORT. Round 89 measured it at
+0.00647 on human-labelled pairs standing alone, but round 91 measured the
whole ladder at SHIPPED DEPTH and it contributes **+0.00078** there -- the
ensemble and the B models already capture nearly all of it. That is 3/3 but far
under this project's standing bar of 0.004, and the bar is applied here the
same way it has been applied to everything else. Including it would add roughly
+0.0008 and a parameter to be wrong about later.

Round 91's honest prices, on ruler E's human half, at shipped depth, 3/3 each:

    + big name table, mc 25     +0.00307   (5.7x inflated on the mined half)
    + B models 63/2000          +0.00249   (3.7x)
    + min_data_in_leaf 2000     +0.00078   (37.2x)  <- excluded
    v8 vs v6, total             +0.00556

Also fixes a real bug inherited from t70: `json` is USED at the meta dump and
never imported, which is what raised NameError after every model had been saved
and forced v7's meta.json to be rebuilt by hand. compile() does not catch it
because Python does not resolve names at compile time, so there is an explicit
assert below.
"""
import pathlib

here = pathlib.Path(__file__).parent
src = (here / "kaggle-t70" / "exp.py").read_text(encoding="utf-8")

# ---- the bug that cost v7 its meta.json -------------------------------------
assert "json.dump(meta" in src, "the meta dump moved -- re-derive the import fix"
assert "\nimport json\n" not in src, "already imported; check whether t70 changed"
A = "import gc\n"
assert A in src
src = src.replace(A, "import gc\nimport json\n", 1)

# ---- the one change: B model capacity ---------------------------------------
OLD_NB = "N_A, N_B = 2000, 800\n"
assert OLD_NB in src, "the B round count moved"
src = src.replace(OLD_NB, "N_A, N_B = 2000, 2000\n", 1)
assert src.count("dict(PAR, num_leaves=31)") == 2, \
    "expected exactly two B-model constructions (phase 1 and phase 2)"
src = src.replace("dict(PAR, num_leaves=31)", "dict(PAR, num_leaves=63)")

# ---- provenance --------------------------------------------------------------
OLD_META = '        "change_vs_v6": ('
assert OLD_META in src
NEW_META = (
    '        "change_vs_v7": ('
    '"per-category B models go from 31 leaves/800 rounds to 63/2000 -- they '
    'had been at round 34\'s settings for ~50 rounds with nobody asking why. '
    'Round 86 (A trained once per draw and SHARED, so the delta isolates B): '
    '+0.01088 on ruler E 3/3 and +0.00381 on ruler G 3/3. Round 88 confirmed '
    'it survives SHIPPED DEPTH at +0.00950 3/3 and is additive with the big '
    'table (+0.01861 together, vs +0.01940 if perfectly additive). Round 91 '
    'priced it on ruler E\'s HUMAN half at +0.00249 3/3. Weight 0.5 was '
    'checked at the same time and is right: 0.25 and 0.75 both lose. '
    'Inference cost is more trees in a pass that already runs, and round 50 '
    'measured trees as close to free here -- featurisation dominates the '
    'container. The cost is ARCHIVE SIZE: B models are 5.1x more leaves."),\n'
    '        "excluded_min_data_in_leaf_2000": ('
    '"+0.00647 on human-labelled pairs standing alone (round 89, 4/4, sd '
    '0.00092) but only +0.00078 at shipped depth (round 91, 3/3) -- the '
    'ensemble and B models already capture it. Under the 0.004 bar, so left '
    'out."),\n'
    '        "ruler_E_delta_inflation": ('
    '"ROUND 89: ruler E is 182,827 human validation rows PLUS 232,350 '
    'LSH-mined negatives, and its DELTAS run 3.7x to 37x hot depending on the '
    'change -- its LEVEL is fine, which is why round 19\'s calibration did not '
    'catch this. Round 85 read min_data_in_leaf 20000 at +0.067 on the full '
    'ruler; on the human half it LOSES 0.0126. Price every future change on '
    'the human half. The v4->v6 leaderboard step (+0.00358) against ruler E\'s '
    '+0.03 to +0.05 for the same change is independent support."),\n'
    '        "expected_LB": ('
    '"v6 scored 0.35502. Round 91 puts v8 at +0.00556 over v6 on '
    'human-labelled pairs at shipped depth (3/3, sd 0.00072), so expect '
    '~0.3606. The old +0.019 figure was ruler E\'s mined half talking."),\n'
    '        "change_vs_v6": (')
src = src.replace(OLD_META, NEW_META, 1)

OLD_SLOPE = ('"competence; the E->LB slope is between 0.54 and 1.22, so "\n'
             '                 "quote a range, not a point."')
assert OLD_SLOPE in src, "the slope note moved -- it must be corrected, not left"
src = src.replace(OLD_SLOPE,
                  '"competence. The old \'E->LB slope 0.54 to 1.22\' claim is "\n'
                  '                 "WITHDRAWN: see ruler_E_delta_inflation."', 1)

src = src.replace('print(f"\\n=== ROUND 43: THE CORRECTED EXPORT ===")',
                  'print(f"\\n=== ROUND 94: THE v8 EXPORT ===")', 1)

for produced in ("N_A, N_B = 2000, 2000", "dict(PAR, num_leaves=63)",
                 "import json", "change_vs_v7", "ruler_E_delta_inflation",
                 "excluded_min_data_in_leaf_2000"):
    assert produced in src, f"patch produced nothing for {produced!r}"
assert "num_leaves=31" not in src, "a 31-leaf B model survived the patch"
compile(src, "exp.py", "exec")

# compile() does not resolve names; t70 shipped a `json` that was never bound.
import ast

names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
tree = ast.parse(src)
bound = set(names)
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
        bound.add(n.id)
    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(n.name)
    elif isinstance(n, ast.alias):
        bound.add((n.asname or n.name).split(".")[0])
    elif isinstance(n, ast.arg):
        bound.add(n.arg)
    elif isinstance(n, ast.comprehension):
        for t in ast.walk(n.target):
            if isinstance(t, ast.Name):
                bound.add(t.id)
used = {n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
missing = sorted(used - bound - {"__builtins__"})
assert not missing, f"names used but never bound: {missing}"
print("patched, compiles, and every name it uses is bound")

out = here / "kaggle-t94" / "exp.py"
out.parent.mkdir(exist_ok=True)
out.write_text(src, encoding="utf-8")
print(f"wrote {out}  ({len(src)} chars)")
