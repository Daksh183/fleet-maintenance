"""v8 feature experiment, judged on CV cost with 3 x 5-fold repeated CV (15 paired differences vs A).
  A  base classifier features / base cost-model features (= v5 Platt)
  B  classifier + histogram shape        / base cost model
  C  classifier + shape + usage ratios   / base cost model
  E  base classifier                     / cost model + shape
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout
from decide import decide_pv
from features import ratios, shape
from score import per_vehicle, true_cost
from v2_params import PARAMS
from v5_percost_cfg import CM

tr, te, S = load(); dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev)
SH, RA = shape(dev), ratios(dev, S)
X = {"base": dev[S], "shape": pd.concat([dev[S], SH], axis=1), "ratio": pd.concat([dev[S], SH, RA], axis=1)}
VAR = {"A": ("base", "base"), "B": ("shape", "base"), "C": ("ratio", "base"), "E": ("base", "shape")}
cost = {v: [] for v in VAR}; r2 = {"base": [], "shape": []}; oofs = {k: np.zeros((3, len(dev))) for k in ("base", "shape", "ratio")}
for r, fs in enumerate((SEED, 1, 2)):
    F = list(StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y))
    oof = {k: np.zeros(len(dev)) for k in ("base", "shape", "ratio")}
    for a, b in F:
        for k in oof:
            oof[k][b] = xgb.XGBClassifier(**PARAMS).fit(X[k].iloc[a], y[a]).predict_proba(X[k].iloc[b])[:, 1]
    for k in oof: oofs[k][r] = oof[k]
    for a, b in F:
        fa = a[y[a] == 1]; lc = np.log(c[fa]); fb = b[y[b] == 1]
        cm = {}
        for k in ("base", "shape"):
            Xc = X[k].iloc[fa]
            resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), Xc, lc, cv=KFold(5, shuffle=True, random_state=SEED))
            mdl = xgb.XGBRegressor(**CM).fit(Xc, lc)
            cm[k] = (mdl.predict(X[k].iloc[b]), resid)
            r2[k].append(r2_score(np.log(c[fb]), mdl.predict(X[k].iloc[fb])))
        for v, (fk, ck) in VAR.items():
            q = make("platt").fit(oof[fk][a], y[a]).predict(oof[fk][b])
            fl, bu = decide_pv(q, *cm[ck])
            cost[v].append(per_vehicle(c[b], fl, bu))
    print(f"repeat {r} done", flush=True)

A = np.array(cost["A"])
print(f"\ncost model R2 (log): base {np.mean(r2['base']):.3f}  shape {np.mean(r2['shape']):.3f}")
for k in oofs:
    print(f"classifier {k:6s}: log-loss {np.mean([log_loss(y, o) for o in oofs[k]]):.5f}  AUC {np.mean([roc_auc_score(y, o) for o in oofs[k]]):.4f}")
print("\nvariant  CV cost (15 folds)   paired Δ mean ± se     better   first-5 (fixed folds)")
for v in VAR:
    x = np.array(cost[v]); d = x - A
    print(f"{v}        {x.mean():.4f}±{x.std():.3f}     {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)) if v!='A' else 0:.4f}    {int((d<0).sum())}/15    {x[:5].mean():.4f}")
    if v != "A":
        log_result("v8", f"features variant {v} ({VAR[v][0]} clf / {VAR[v][1]} cost model), 3x5 CV", x.mean(), x.std(),
                   notes=f"paired vs A {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} (se), better {int((d<0).sum())}/15")
np.save(ROOT / "reports/v8_feat_costs.npy", np.array([cost[v] for v in VAR]))
