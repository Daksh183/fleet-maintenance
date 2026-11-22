"""Is the v10->v12 change in test cost predictions bigger than plain seed-to-seed variation of the same model?"""
import numpy as np, pandas as pd, xgboost as xgb
from common import load
from features import shape
from v5_percost_cfg import CM
tr, te, S = load(); f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
A1 = {**CM, "n_estimators": 1000, "max_depth": 2}; keep = [s for s in S if s not in ["s120", "s122", "s126", "s128"]]
Xa, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
Xk, Xtk = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
P = {s: xgb.XGBRegressor(**{**A1, "random_state": s}).fit(Xa, lc).predict(Xt) for s in (42, 1, 2, 3, 4)}
K = {s: xgb.XGBRegressor(**{**A1, "random_state": s}).fit(Xk, lc).predict(Xtk) for s in (42, 1, 2, 3, 4)}
d = lambda a, b: float(np.abs(a - b).mean())
print("all sensors, seed vs seed (mean abs log diff):", round(np.mean([d(P[a], P[b]) for a in P for b in P if a < b]), 3))
print("drop-4,      seed vs seed:                    ", round(np.mean([d(K[a], K[b]) for a in K for b in K if a < b]), 3))
print("all vs drop-4, same seed:                     ", round(np.mean([d(P[s], K[s]) for s in P]), 3))
bagA, bagK = np.mean(list(P.values()), 0), np.mean(list(K.values()), 0)
print("5-seed average all vs 5-seed average drop-4:  ", round(d(bagA, bagK), 3))
print("single seed-42 'all' vs 5-seed bag 'all':     ", round(d(P[42], bagA), 3), "  seed-42 drop-4 vs bag 'drop-4':", round(d(K[42], bagK), 3))
