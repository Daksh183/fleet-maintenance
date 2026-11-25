"""v26: is failure risk a simple (simulated) function? Logistic regression on log1p sensors (+ missing indicators), 3x5 CV OOF
into reports/v18_search/ for v18_score.py. Also a blend with base_cens (probability average)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from common import ROOT, load, split_holdout
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
S2 = [s for s in S if s not in ("s063",)]
def X(d): return np.log1p(d[S2].clip(lower=0))
for nm, C in (("lr_c01", 0.1), ("lr_c1", 1.0)):
    P = np.zeros((3, len(y)))
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            P[r, b] = m.fit(X(dev.loc[a]), y[a]).predict_proba(X(dev.loc[b]))[:, 1]
    np.save(ROOT / f"reports/v18_search/{nm}.npy", P); print("done", nm, flush=True)
