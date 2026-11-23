"""v11b: budget = conditional quantile. Compare, on failures only (3x5 CV), pinball loss at levels a in A (cost scale):
  RES  log-cost model (A1) + out-of-fold residual quantile, back-transformed  (current)
  QLOG XGB quantile regression on log cost at level a, then exp
  QRAW XGB quantile regression on raw cost at level a
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from common import SEED, load, split_holdout, log_result
from features import shape
from score import true_cost
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr)
f = dev[dev.failure == 1].reset_index(drop=True); X = pd.concat([f[S], shape(f)], axis=1); c = true_cost(f); lc = np.log(c)
CM_A1 = {**CM, "n_estimators": 1000, "max_depth": 2}
QP = {**CM, "n_estimators": 400, "max_depth": 2}
A = (0.5, 0.8, 0.91, 0.97)
pin = lambda y, b, a: np.mean(np.maximum(a * (y - b), (a - 1) * (y - b)))
L = {k: {a: [] for a in A} for k in ("RES", "QLOG", "QRAW")}
for rs in (42, 1, 2):
    for a_i, b_i in KFold(5, shuffle=True, random_state=rs).split(X):
        Xa, Xb = X.iloc[a_i], X.iloc[b_i]
        mi = cross_val_predict(xgb.XGBRegressor(**CM_A1), Xa, lc[a_i], cv=KFold(5, shuffle=True, random_state=42)); r = lc[a_i] - mi
        mb = xgb.XGBRegressor(**CM_A1).fit(Xa, lc[a_i]).predict(Xb)
        for a in A:
            L["RES"][a].append(pin(c[b_i], np.exp(mb + np.quantile(r, a)), a))
            ql = xgb.XGBRegressor(**QP, objective="reg:quantileerror", quantile_alpha=a).fit(Xa, lc[a_i]).predict(Xb)
            L["QLOG"][a].append(pin(c[b_i], np.exp(ql), a))
            qr = xgb.XGBRegressor(**QP, objective="reg:quantileerror", quantile_alpha=a).fit(Xa, c[a_i]).predict(Xb)
            L["QRAW"][a].append(pin(c[b_i], np.maximum(qr, 0), a))
    print("repeat", rs, flush=True)
for a in A:
    base = np.array(L["RES"][a])
    for k in L:
        x = np.array(L[k][a]); d = x - base
        print(f"level {a}: {k:5s} pinball {x.mean():8.2f}  paired vs RES {d.mean():+7.2f} ± {d.std(ddof=1)/np.sqrt(15):.2f}  better {int((d<0).sum())}/15")
