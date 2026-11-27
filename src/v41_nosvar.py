"""v41: variants of the no-s120/s122 failure model, judged in the shift-aware CV (wear-reweighted validation, Platt from other folds,
each scored at its own best level). The no-s122 model does not read the shifted sensors, so one scenario covers both stories.
Variants: v12 params, depth 3 / 600 trees, depth 6 + min_child_weight 5, slow (1000 trees, lr .02, colsample .5), L2/gamma regularised,
5-seed bag, CatBoost, LightGBM, and + GP predicted log repair cost (GP fitted on the training fold's failures only)."""
import os, sys
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); c = dev.repair_cost.fillna(0).to_numpy() * y
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
h = lambda x: np.histogram(np.log1p(x[np.isfinite(x)]), e)[0] / np.isfinite(x).sum()
ratio = np.clip(h(te.s126.to_numpy()) / np.maximum(h(tr.s126.to_numpy()), 1e-9), 0, 5)
W = ratio[np.clip(np.searchsorted(e, np.log1p(dev.s126.fillna(dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)]
NO = [s for s in S if s not in ("s120", "s122")]
X_ = lambda **k: (lambda: xgb.XGBClassifier(**{**PARAMS, **k}))
V = {"base": X_(), "d3_600": X_(max_depth=3, n_estimators=600), "d6_mcw5": X_(max_depth=6, min_child_weight=5),
     "slow": X_(n_estimators=1000, learning_rate=0.02, colsample_bytree=0.5), "reg": X_(reg_lambda=5.0, gamma=1.0, min_child_weight=3),
     "cat": lambda: CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, verbose=0, thread_count=8, random_seed=0),
     "lgb": lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1)}
names = sys.argv[1:] or list(V) + ["bag5", "gpfeat"]
G = extract(dev); FAIL = np.flatnonzero(y == 1)
P = {k: np.zeros((3, len(y))) for k in names}
for r in range(3):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
        Xa, Xb = dev.loc[a, NO], dev.loc[b, NO]
        for nm in names:
            if nm in V: P[nm][r, b] = V[nm]().fit(Xa, y[a]).predict_proba(Xb)[:, 1]
            elif nm == "bag5": P[nm][r, b] = np.mean([xgb.XGBClassifier(**{**PARAMS, "random_state": 42 + s}).fit(Xa, y[a]).predict_proba(Xb)[:, 1] for s in range(5)], 0)
            elif nm == "gpfeat":
                fa = np.intersect1d(a, FAIL); gp = GaussianSeverity().fit(select(G, fa), np.log(dev.repair_cost.to_numpy()[fa]))
                ga = gp.predict(select(G, a)); gb = gp.predict(select(G, b))
                P[nm][r, b] = xgb.XGBClassifier(**PARAMS).fit(Xa.assign(gp=ga), y[a]).predict_proba(Xb.assign(gp=gb))[:, 1]
    print("repeat", r, "done", flush=True)
def score(Pm):
    out = []
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            q = make("platt").fit(Pm[r, a], y[a]).predict(Pm[r, b]); w = W[b]; best = np.inf
            for lam in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
                qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, md[b], resid)
                cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[b] - bu, 0) + OVER * np.maximum(bu - c[b], 0), c[b]); best = min(best, (cost * w).sum() / w.sum())
            out.append(best)
    return np.array(out)
R = {k: score(v) for k, v in P.items()}; base = R.get("base")
for k, v in sorted(R.items(), key=lambda kv: kv[1].mean()):
    d = v - base; print(f"{k:9s} cost/veh {v.mean():.4f}  vs base no-s122 {d.mean():+.4f}±{d.std() / np.sqrt(15):.4f}  better {(d < 0).sum():2d}/15", flush=True)
for k, v in P.items(): np.save(ROOT / f"reports/v18_search/nos_{k}.npy", v)
