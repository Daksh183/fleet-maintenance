"""v41: test probabilities of the 'slow' no-s120/s122 classifier (1000 trees, lr .02, colsample .5), Platt on 5-fold dev OOF."""
import numpy as np, xgboost as xgb
from calib import make
from common import ROOT, folds, load, split_holdout
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); NO = [s for s in S if s not in ("s120", "s122")]
prm = {**PARAMS, "n_estimators": 1000, "learning_rate": 0.02, "colsample_bytree": 0.5}
oof = np.zeros(len(y))
for a, b in folds(y): oof[b] = xgb.XGBClassifier(**prm).fit(dev.loc[a, NO], y[a]).predict_proba(dev.loc[b, NO])[:, 1]
q = make("platt").fit(oof, y).predict(xgb.XGBClassifier(**prm).fit(tr[NO], tr.failure).predict_proba(te[NO])[:, 1])
q22 = np.load(ROOT / "reports/v22_q.npy"); from scipy.stats import spearmanr
print(f"mean q {q.mean():.4f} (nos122 {q22.mean():.4f}); Spearman vs nos122 {spearmanr(q, q22)[0]:.3f}")
np.save(ROOT / "reports/v41_nosslow_q.npy", q)
