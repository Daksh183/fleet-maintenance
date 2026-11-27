"""v84: can other learners improve the repair-cost model? OOF log-cost predictions on the 397 training failures from ridge, SVR, kNN and
regularised XGBoost (5 folds x 5 seeds), alone and averaged with the v51 GP OOF. Metrics: R2 (log), and newsvendor budget loss at the
quantile levels the flagged test vehicles need (empirical residual quantiles, cross-fitted)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import ROOT, load
from score import UNDER, OVER
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy())
z = np.load(ROOT / "reports/v51c_gp.npz"); gp = z["oof"]; ok = np.isfinite(gp); print("GP eligible", ok.sum(), "of", len(y), "| z y matches", np.allclose(z["y"][ok] if len(z["y"]) == len(y) else y[ok], y[ok]))
X = np.log1p(np.maximum(f[[s for s in S if s not in ("s120", "s122", "s126", "s128", "s063")]].to_numpy(float), 0)); X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
X = X[:, X.std(0) > 0]
M = {"ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 20))),
     "svr": lambda: make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.1, gamma="scale")),
     "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(15, weights="distance")),
     "xgb": lambda: xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, reg_lambda=5, n_jobs=4)}
O = {}
for nm, mk in M.items():
    o = np.zeros((5, len(y)))
    for s in range(5):
        for a, b in KFold(5, shuffle=True, random_state=s).split(y): o[s, b] = mk().fit(X[a], y[a]).predict(X[b])
    O[nm] = o.mean(0)
lev = np.load(ROOT / "reports/v51_levels.npy") if (ROOT / "reports/v51_levels.npy").exists() else None
Qs = np.array([0.5, 0.65, 0.75, 0.82, 0.86, 0.89])          # typical newsvendor levels 1-(1/11)/p for p from ~0.18 to ~0.83
def bloss(pred):
    r = y[ok] - pred[ok]; out = 0
    for a, b in KFold(10, shuffle=True, random_state=0).split(r):
        for q in Qs:
            bud = np.exp(pred[ok][b] + np.quantile(r[a], q)); c = np.exp(y[ok][b])
            out += (UNDER * np.maximum(c - bud, 0) + OVER * np.maximum(bud - c, 0)).sum()
    return out / len(r) / len(Qs)
R2 = lambda p: 1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
b0 = bloss(gp); print(f"{'model':18s} {'R2':>6s} {'budget loss':>12s} {'vs GP':>7s}")
rows = [("GP (v51)", gp)] + [(k, v) for k, v in O.items()] + [(f"GP+{k} 50/50", (gp + v) / 2) for k, v in O.items()] + [(f"GP 75 + {k} 25", 0.75 * gp + 0.25 * v) for k, v in O.items()] + [("GP+svr+xgb", (gp + O["svr"] + O["xgb"]) / 3), ("GP+all", (2 * gp + sum(O.values())) / 6)]
for nm, p in rows: bl = bloss(p); print(f"{nm:18s} {R2(p):6.3f} {bl:12.2f} {100 * (bl / b0 - 1):+6.1f}%")
