"""A1: cost-model capacity. Repeated 3x5 KFold over DEV failures only (hold-out untouched).
Metrics: R2 (log), calibration slope (lc ~ a + b*m; b>1 = over-shrunk), extremes bias, and budget
pinball loss at q* on the cost scale (the budget's own loss for a sure failure)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from common import SEED, load, split_holdout
from decide import Q_STAR
from features import shape
tr, te, S = load(); dev, _ = split_holdout(tr)
f = dev[dev.failure == 1].reset_index(drop=True)
X = pd.concat([f[S], shape(f)], axis=1); lc = np.log(f.repair_cost.to_numpy()); cst = f.repair_cost.to_numpy()
base = dict(subsample=0.8, colsample_bytree=0.5, min_child_weight=5, n_jobs=-1, random_state=SEED)
grid = {"v8 (200 tr, lr.03, d3)": dict(n_estimators=200, learning_rate=0.03, max_depth=3),
        "500 tr d3": dict(n_estimators=500, learning_rate=0.03, max_depth=3),
        "1000 tr d3": dict(n_estimators=1000, learning_rate=0.03, max_depth=3),
        "500 tr d4": dict(n_estimators=500, learning_rate=0.03, max_depth=4),
        "1000 tr d2": dict(n_estimators=1000, learning_rate=0.03, max_depth=2),
        "2000 tr d2": dict(n_estimators=2000, learning_rate=0.03, max_depth=2)}
res = {}
for name, g in grid.items():
    r2s, slopes, lo, hi, pins = [], [], [], [], []
    for rep in range(3):
        m = np.zeros(len(f))
        for a, b in KFold(5, shuffle=True, random_state=rep).split(X):
            m[b] = xgb.XGBRegressor(**base, **g).fit(X.iloc[a], lc[a]).predict(X.iloc[b])
        r = lc - m; qq = pd.qcut(m, 5, labels=False)
        r2s.append(r2_score(lc, m)); slopes.append(np.polyfit(m, lc, 1)[0])
        lo.append(r[qq == 0].mean()); hi.append(r[qq == 4].mean())
        bq = np.exp(m + np.quantile(r, Q_STAR))
        pins.append(np.mean(0.2 * np.maximum(cst - bq, 0) + 0.02 * np.maximum(bq - cst, 0)))
    res[name] = pins
    print(f"{name:24s} R2 {np.mean(r2s):.3f}  slope {np.mean(slopes):.2f}  bias low/high quintile {np.mean(lo):+.2f}/{np.mean(hi):+.2f}  budget loss {np.mean(pins):.2f}", flush=True)
b0 = np.array(res["v8 (200 tr, lr.03, d3)"])
for k, v in res.items(): print(f"  paired budget-loss vs v8: {k:24s} {np.mean(np.array(v) - b0):+.2f}  per repeat {np.round(np.array(v) - b0, 2)}")
