"""v18 cost-model search with the classifier FIXED (IPCW 4-XGB ensemble OOF). Per repeat/fold: cost model fit on the training
fold's failures (residuals from inner 5-fold OOF), pipeline cost on complete cases of the validation fold (IPCW-weighted)."""
import sys, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
from v18_censor import km_censor
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
c = dev.repair_cost.fillna(0).to_numpy() * y; complete = (y == 1) | (t >= 365)
G = km_censor(t, y); W = np.zeros(len(y)); W[complete] = 1 / G(np.where(y == 1, np.minimum(t, 365) - 1, 364)[complete])
CLS = ["ipcw", "ipcw_nodrift", "ipcw_d6_mcw5", "ipcw_slow"]
P = np.mean([np.load(ROOT / f"reports/v18_search/{k}.npy") for k in CLS], axis=0)
DRIFT = ["s120", "s122", "s126", "s128"]; keep = [s for s in S if s not in DRIFT]
SH = shape(dev); XK = pd.concat([dev[keep], SH], axis=1); XA = pd.concat([dev[S], SH], axis=1)
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}
xg = lambda **k: (lambda s=0: xgb.XGBRegressor(**{**CM9, **k, "random_state": s}))
VARS = {
    "cm_base": [(xg(), XK, 0)],
    "cm_bag5": [(xg(), XK, s) for s in range(5)],
    "cm_mix4": [(xg(), XK, 0), (xg(), XA, 1), (xg(max_depth=3, n_estimators=600), XK, 2), (xg(learning_rate=0.015, n_estimators=2000), XK, 3)],
    "cm_d3": [(xg(max_depth=3, n_estimators=600), XK, 0)],
    "cm_lgb": [(lambda s=0: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=4, min_child_samples=5, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.5, verbose=-1, random_state=s), XK, 0)],
    "cm_mix4_bag": [(xg(), XK, s) for s in range(3)] + [(xg(), XA, 10 + s) for s in range(3)] + [(xg(max_depth=3, n_estimators=600), XK, 20 + s) for s in range(3)],
}
names = sys.argv[1:] or list(VARS); R = {}
for nm in names:
    out = []
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            fa = a[y[a] == 1]; lc = np.log(c[fa]); mb = np.zeros(len(b)); mo = np.zeros(len(fa))
            for mk, X, s in VARS[nm]:
                mb += mk(s).fit(X.iloc[fa], lc).predict(X.iloc[b]) / len(VARS[nm])
                for ia, ib in KFold(5, shuffle=True, random_state=7).split(fa):
                    mo[ib] += mk(s).fit(X.iloc[fa[ia]], lc[ia]).predict(X.iloc[fa[ib]]) / len(VARS[nm])
            resid = lc - mo; ca = a[complete[a]]; cbm = complete[b]; cb = b[cbm]
            qq = make("platt").fit(P[r][ca], y[ca]).predict(P[r][cb]); fl, bu = decide_pv(qq, mb[cbm], resid); wb = W[cb]
            cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[cb] - bu, 0) + OVER * np.maximum(bu - c[cb], 0), c[cb])
            out.append((cost * wb).sum() / wb.sum())
    R[nm] = np.array(out); np.save(ROOT / f"reports/v18_search/cost_{nm}.npy", R[nm])
    d = R[nm] - (R["cm_base"] if "cm_base" in R else np.load(ROOT / "reports/v18_search/cost_cm_base.npy"))
    print(f"{nm:14s} cost {R[nm].mean():.4f}  vs cm_base {d.mean():+.4f}±{d.std()/np.sqrt(15):.4f}  better {(d<0).sum()}/15", flush=True)
