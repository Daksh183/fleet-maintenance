"""Is the shape cost model's gain consistent? Per-fold paired R2 (15 folds) and budget coverage."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from common import SEED, load, split_holdout
from decide import Q_STAR
from features import shape
from score import true_cost
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev)
X = {"base": dev[S], "shape": pd.concat([dev[S], shape(dev)], axis=1)}
r2 = {k: [] for k in X}; cov = {k: [] for k in X}; pin = {k: [] for k in X}
for fs in (SEED, 1, 2):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y):
        fa, fb = a[y[a] == 1], b[y[b] == 1]; lc, lb = np.log(c[fa]), np.log(c[fb])
        for k in X:
            resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), X[k].iloc[fa], lc, cv=KFold(5, shuffle=True, random_state=SEED))
            m = xgb.XGBRegressor(**CM).fit(X[k].iloc[fa], lc).predict(X[k].iloc[fb])
            r2[k].append(r2_score(lb, m)); cov[k].append(np.mean(lb <= m + np.quantile(resid, Q_STAR)))
            # pinball loss at q* on the cost scale = the budget's own loss for a sure failure
            bq = np.exp(m + np.quantile(resid, Q_STAR)); cf = c[fb]
            pin[k].append(np.mean(0.2 * np.maximum(cf - bq, 0) + 0.02 * np.maximum(bq - cf, 0)))
d = np.array(r2["shape"]) - np.array(r2["base"]); dp = np.array(pin["shape"]) - np.array(pin["base"])
print(f"R2 base {np.mean(r2['base']):.3f}  shape {np.mean(r2['shape']):.3f}  paired +{d.mean():.3f}  better {int((d>0).sum())}/15  min gain {d.min():+.3f}")
print(f"budget coverage (target {Q_STAR:.3f}): base {np.mean(cov['base']):.3f}  shape {np.mean(cov['shape']):.3f}")
print(f"budget loss per failing vehicle: base {np.mean(pin['base']):.2f}  shape {np.mean(pin['shape']):.2f}  paired {dp.mean():+.2f}  better {int((dp<0).sum())}/15")
