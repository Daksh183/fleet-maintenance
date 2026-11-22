"""v6: failure-probability model variants (follow-up handling, missingness features).

Each variant changes only how the CLASSIFIER is trained; the v5 pipeline (cost model, Platt fitted on
the other folds' OOF, per-vehicle decision) is held fixed, and every variant is scored on ALL
validation vehicles. Paired by fold vs A (= v5 Platt, 0.904).
  B  + missingness features (NaN count excluding s063, NaN count in blocks of 20 columns)
  C  drop healthy vehicles with followup < 180 d from classifier training
  D  weight healthy vehicles by min(followup,365)/365
  E  B + D
followup_days is used only to build training rows/weights, never as a feature.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

from calib import make
from common import SEED, folds, load, log_result, split_holdout
from decide import Q_STAR, decide_pv
from score import per_vehicle, true_cost
from v5_percost_cfg import CM

tr, te, S = load()
dev, _ = split_holdout(tr)
y, c = dev["failure"].to_numpy(), true_cost(dev)
fu = dev["followup_days"].to_numpy()
F = folds(y)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=1, n_jobs=-1, random_state=SEED)


def with_miss(df):
    X = df[S].copy()
    S2 = [s for s in S if s != "s063"]            # s063 is NaN by construction in test
    X["n_nan"] = df[S2].isna().sum(axis=1)
    for i in range(0, len(S2), 20):
        X[f"nan_blk{i//20}"] = df[S2[i:i + 20]].isna().sum(axis=1)
    return X


Xb_dev, Xb_te = with_miss(dev), with_miss(te)
W = lambda a: np.where(y[a] == 1, 1.0, np.minimum(fu[a], 365) / 365)

variants = {
    "A": (dev[S], None, None),
    "B": (Xb_dev, None, None),
    "C": (dev[S], lambda a: a[(y[a] == 1) | (fu[a] >= 180)], None),
    "D": (dev[S], None, W),
    "E": (Xb_dev, None, W),
}
oofs = {}
for k, (X, sel, wf) in variants.items():
    o = np.zeros(len(dev))
    for a, b in F:
        aa = sel(a) if sel else a
        o[b] = xgb.XGBClassifier(**PARAMS).fit(X.loc[aa], y[aa], sample_weight=None if wf is None else wf(aa)
                                               ).predict_proba(X.loc[b])[:, 1]
    oofs[k] = o
    print(f"{k}: OOF AUC {roc_auc_score(y, o):.4f}  PR-AUC {average_precision_score(y, o):.4f}  mean p {o.mean():.4f}")
np.save("../reports/p_oof_v6.npy", np.vstack([oofs[k] for k in variants]))

cost = {k: [] for k in variants}
for a, b in F:                                     # same nested cost model as v5
    fa = a[y[a] == 1]
    Xc, lc = dev.loc[fa, S], np.log(c[fa])
    inner = cross_val_predict(xgb.XGBRegressor(**CM), Xc, lc, cv=KFold(5, shuffle=True, random_state=SEED))
    resid = lc - inner
    m_b = xgb.XGBRegressor(**CM).fit(Xc, lc).predict(dev.loc[b, S])
    for k in variants:
        p = make("platt").fit(oofs[k][a], y[a]).predict(oofs[k][b])
        fl, bu = decide_pv(p, m_b, resid)
        cost[k].append(per_vehicle(c[b], fl, bu))

A = np.array(cost["A"])
print("\nvariant  CV cost/veh        paired Δ vs A per fold                 improved")
for k in variants:
    x = np.array(cost[k]); d = x - A
    print(f"{k}       {x.mean():.4f}±{x.std():.3f}   {np.round(d, 3)}   {int((d < 0).sum())}/5")
    if k != "A":
        log_result("v6", f"failure-p variant {k}, Platt, v5 cost model", x.mean(), x.std(),
                   notes=f"A {A.mean():.3f}; paired {np.mean(d):+.3f}; AUC {roc_auc_score(y, oofs[k]):.4f}")
