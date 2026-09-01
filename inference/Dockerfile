# Custom image for the E_CUP 2026 matching solution.
#
# Why this exists: the baseline image has neither lightgbm nor rapidfuzz. The
# workaround of vendoring unpacked wheels into the archive and putting them on
# sys.path made the container fail to finish three times; removing vendor/ made
# it succeed. Building our own image is the route the rules explicitly provide,
# and it also lets us PIN versions so model artifacts trained on Kaggle load
# here without a pickle-version mismatch.
#
# Model format note: we ship LightGBM .txt boosters, not pickles. The text
# format is stable across versions, so even a version drift cannot silently
# break scoring the way joblib did (that failure scored 0.1113 = pure
# prevalence, because the constant fallback was retained).
FROM odsai/ecup26-matching-baseline:1.0

# The baseline image has NO libgomp: the build failed with
#   OSError: libgomp.so.1: cannot open shared object file
# scikit-learn works there only because its manylinux wheel bundles a private
# copy under sklearn/.libs/ with a hashed SONAME, which cannot satisfy
# lightgbm's DT_NEEDED on plain "libgomp.so.1". Installing it properly at the
# OS level is the fix; preloading a vendored copy with ctypes at runtime (what
# the earlier archives did) was treating the symptom.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --break-system-packages: the base is Ubuntu 24.04, whose system Python is
# marked externally-managed (PEP 668), so pip refuses to install into it. That
# guard exists to protect an OS's own package manager; here the container IS
# the environment and the organizers themselves populate site-packages directly
# (COPY /install/local/lib), so there is nothing to protect.
RUN pip install --no-cache-dir --break-system-packages \
        lightgbm==4.6.0 \
        rapidfuzz==3.14.4

# Fail the BUILD rather than the submission if anything is missing or broken.
RUN python -c "\
import lightgbm, rapidfuzz, numpy, scipy, sklearn, pandas, pyarrow, joblib; \
from rapidfuzz import fuzz, process; \
from rapidfuzz.distance import JaroWinkler; \
assert fuzz.ratio('abc','abd') > 50; \
print('lightgbm', lightgbm.__version__); \
print('rapidfuzz', rapidfuzz.__version__); \
print('sklearn', sklearn.__version__); \
print('numpy', numpy.__version__); \
print('scipy', scipy.__version__); \
print('pandas', pandas.__version__); \
print('pyarrow', pyarrow.__version__); \
print('joblib', joblib.__version__)"

# Smoke-test an actual LightGBM round-trip through the .txt format.
RUN python -c "\
import numpy as np, lightgbm as lgb; \
X = np.random.RandomState(0).rand(200, 5); y = (X[:,0] > 0.5).astype(int); \
m = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y); \
m.booster_.save_model('/tmp/t.txt'); \
p = lgb.Booster(model_file='/tmp/t.txt').predict(X); \
assert p.shape == (200,) and np.isfinite(p).all(); \
print('lightgbm txt round-trip OK')"
