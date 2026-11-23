"""v11: ideas from the structure of the problem, judged on full-pipeline CV cost (3x5 repeated, paired vs baseline).

The decision is flag iff p*mu is large, i.e. ranking by EXPECTED COST, not by failure probability. Expensive failures
dominate both the loss and the decision, so:
  W05 / W10  failure classifier trained with weight (c/mean c)^alpha on failures (alpha .5 / 1), then Platt-recalibrated
  TW         direct expected-cost model (XGB, Tweedie objective on c incl. zeros): p_T = clip(e/mu_i), Platt-recalibrated
  BL         average of logit(q) and logit(p_T)
Cost model (A1 + shape features) is shared and cached per fold. Classifier baseline = v2, Platt, k=1.
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from calib import make
from common import SEED, load, log_result, split_holdout
from decide import decide_pv
from features import shape
from score import per_vehicle, true_cost
from v2_params import PARAMS
from v5_percost_cfg import CM

tr, te, S = load(); dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev)
X = pd.concat([dev[S], shape(dev)], axis=1)
CM_A1 = {**CM, "n_estimators": 1000, "max_depth": 2}
inner = KFold(5, shuffle=True, random_state=SEED)
TW = {**PARAMS, "objective": "reg:tweedie", "tweedie_variance_power": 1.5}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
expit = lambda z: 1 / (1 + np.exp(-z))

V = ("base", "W05", "W10", "TW", "BL"); cost = {v: [] for v in V}
for fs in (SEED, 1, 2):
    F = list(StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y))
    oof = {v: np.zeros(len(dev)) for v in ("base", "W05", "W10", "TW")}
    for a, b in F:
        Xa, Xb = dev.loc[a, S], dev.loc[b, S]
        oof["base"][b] = xgb.XGBClassifier(**PARAMS).fit(Xa, y[a]).predict_proba(Xb)[:, 1]
        for nm, al in (("W05", .5), ("W10", 1.0)):
            w = np.ones(len(a)); f = y[a] == 1; r = (c[a][f] / c[a][f].mean()) ** al; w[f] = r / r.mean()
            oof[nm][b] = xgb.XGBClassifier(**PARAMS).fit(Xa, y[a], sample_weight=w).predict_proba(Xb)[:, 1]
        oof["TW"][b] = xgb.XGBRegressor(**TW).fit(Xa, c[a]).predict(Xb)
    for a, b in F:
        fa = a[y[a] == 1]; lc = np.log(c[fa])
        mi = cross_val_predict(xgb.XGBRegressor(**CM_A1), X.iloc[fa], lc, cv=inner); r = lc - mi
        mb = xgb.XGBRegressor(**CM_A1).fit(X.iloc[fa], lc).predict(X.iloc[b])
        Mb = np.exp(mb[:, None] + np.quantile(r, (np.arange(99) + .5) / 99)[None, :]).mean(1)
        # TW: convert expected cost to a failure probability with the vehicle's own mean cost; same for the train side (calibration)
        Xfull = X.iloc[a]; m_a = xgb.XGBRegressor(**CM_A1).fit(X.iloc[fa], lc).predict(Xfull)
        Ma = np.exp(m_a[:, None] + np.quantile(r, (np.arange(99) + .5) / 99)[None, :]).mean(1)
        pT_a, pT_b = np.clip(oof["TW"][a] / Ma, 1e-5, .99), np.clip(oof["TW"][b] / Mb, 1e-5, .99)
        q = {}
        for nm in ("base", "W05", "W10"):
            q[nm] = make("platt").fit(oof[nm][a], y[a]).predict(oof[nm][b])
        calT = make("platt").fit(pT_a, y[a]); q["TW"] = calT.predict(pT_b)
        z = .5 * (logit(q["base"]) + logit(q["TW"])); q["BL"] = expit(z)
        for nm in V:
            fl, bu = decide_pv(q[nm], mb, r); cost[nm].append(per_vehicle(c[b], fl, bu))
    print(f"repeat {fs} done", flush=True)
B0 = np.array(cost["base"])
for v in V:
    x = np.array(cost[v]); d = x - B0
    print(f"{v:5s} CV cost {x.mean():.4f}±{x.std():.3f}  paired vs base {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(15):.4f}  better {int((d<0).sum())}/15")
    if v != "base":
        log_result("v11", f"idea {v} (3x5 CV, k=1, A1 cost model)", x.mean(), x.std(),
                   notes=f"paired vs base {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(15):.4f}, better {int((d<0).sum())}/15")
