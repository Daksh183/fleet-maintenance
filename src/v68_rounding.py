"""v68: after realignment the histogram bins keep the RESOLUTION of the column they came from (test s055 arrives rounded to 10, s057 to
100, s062 to 10, ...). Does that change predictions? Train the classifier on (a) raw train, (b) train rounded to the test resolution;
compare test predictions with the change a different random seed makes."""
import numpy as np, pandas as pd, xgboost as xgb
from common import ROOT, load
from v2_params import PARAMS
tr, te, S = load(); H = [f"s{i:03d}" for i in range(54, 63)]
def step(v):                                      # coarsest of 1, 2, 10, 100, 1000 that (almost) all values are multiples of
    v = v.dropna().to_numpy(); v = v[v > 0]
    for s in (1000, 100, 10, 2, 1):
        if np.mean(np.isclose(v / s, np.round(v / s))) > 0.99: return s
    return 0
res = pd.DataFrame({"train_step": [step(tr[c]) for c in H], "test_step": [step(te[c]) for c in H]}, index=H); print(res.T.to_string())
trr = tr.copy()
for c in H:
    s = res.test_step[c]
    if s > res.train_step[c]: trr[c] = np.round(tr[c] / s) * s
fit = lambda D, seed=42: xgb.XGBClassifier(**{**PARAMS, "random_state": seed}).fit(D[S], D.failure).predict_proba(te[S])[:, 1]
a, b, a2 = fit(tr), fit(trr), fit(tr, 7)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
top = lambda p: set(np.argsort(-p)[:1433])
for nm, q in (("rounded training data", b), ("other random seed", a2)):
    print(f"{nm:22s}: top-1433 overlap with base {len(top(a) & top(q))}; rank corr {pd.Series(a).corr(pd.Series(q), method='spearman'):.4f}; "
          f"mean |logit diff| in base top-1433 {np.abs(lg(q) - lg(a))[list(top(a))].mean():.3f}")
