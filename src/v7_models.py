"""Classifier candidates judged DIRECTLY on CV cost (v5 pipeline, Platt, k=1), paired vs v2.
  v2x3   : v2 params, 3 seeds averaged
  lgb    : LightGBM (num_leaves 15, lr .03, 400 trees)
  blend  : mean of v2 and lgb raw p (then Platt)
  blend3 : mean of v2x3 and lgb
"""
import numpy as np, xgboost as xgb, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, folds, load, log_result, split_holdout
from decide import decide_pv
from score import per_vehicle, true_cost
from v2_params import PARAMS
from v5_percost_cfg import CM

tr, te, S = load(); dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev); F = folds(y)
P0 = {k: v for k, v in PARAMS.items() if k != "random_state"}
LG = dict(n_estimators=400, num_leaves=15, learning_rate=0.03, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, min_child_samples=20, verbose=-1, n_jobs=-1)
o = {"v2": np.load(ROOT / "reports/p_oof_v2.npy"), "v2x3": np.zeros(len(dev)), "lgb": np.zeros(len(dev))}
for a, b in F:
    for s in (42, 7, 2024):
        o["v2x3"][b] += xgb.XGBClassifier(**P0, random_state=s).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1] / 3
    o["lgb"][b] = lgb.LGBMClassifier(**LG, random_state=SEED).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
o["blend"] = (o["v2"] + o["lgb"]) / 2; o["blend3"] = (o["v2x3"] + o["lgb"]) / 2
np.save(ROOT / "reports/p_oof_v7models.npy", np.vstack([o[k] for k in o]))
cost = {k: [] for k in o}
for a, b in F:
    fa = a[y[a] == 1]; lc = np.log(c[fa])
    resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), dev.loc[fa, S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
    m_b = xgb.XGBRegressor(**CM).fit(dev.loc[fa, S], lc).predict(dev.loc[b, S])
    for k in o:
        fl, bu = decide_pv(make("platt").fit(o[k][a], y[a]).predict(o[k][b]), m_b, resid)
        cost[k].append(per_vehicle(c[b], fl, bu))
A = np.array(cost["v2"])
for k in o:
    x = np.array(cost[k]); d = x - A
    print(f"{k:7s} logloss {log_loss(y, o[k]):.5f} AUC {roc_auc_score(y, o[k]):.4f}  CV cost {x.mean():.4f}±{x.std():.3f}  "
          f"paired {np.round(d, 3)} mean {d.mean():+.3f}  better {int((d < 0).sum())}/5")
    if k != "v2":
        log_result("v7", f"classifier candidate {k}, Platt, v5 cost, k=1", x.mean(), x.std(), notes=f"paired vs v2 {d.mean():+.3f}, better {int((d<0).sum())}/5")
