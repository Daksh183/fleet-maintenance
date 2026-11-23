"""Full-pipeline CV cost (3x5 repeated, paired) for cost-model variants; classifier = v2, Platt, k=1.
  E   v8 cost model (200 trees, depth 3)
  A1  1000 trees depth 2 (less shrinkage)
  A2  A1 + per-vehicle spread: scale model on |inner-OOF residual|, residuals standardised
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from calib import make
from common import SEED, load, log_result, split_holdout
from decide import Q_STAR, decide_pv
from features import shape
from score import per_vehicle, true_cost
from v2_params import PARAMS
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev)
X = pd.concat([dev[S], shape(dev)], axis=1)
CM_A1 = {**CM, "n_estimators": 1000, "max_depth": 2}
SC = dict(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=10, random_state=SEED)
inner = KFold(5, shuffle=True, random_state=SEED)


def cost_model(params, Xa, lc, Xb, hetero=False):
    m_in = cross_val_predict(xgb.XGBRegressor(**params), Xa, lc, cv=inner); r = lc - m_in
    mdl = xgb.XGBRegressor(**params).fit(Xa, lc); mb = mdl.predict(Xb)
    if not hetero:
        return mb, r, None
    Za, Zb = Xa.assign(m=m_in), Xb.assign(m=mb)
    s_in = np.maximum(cross_val_predict(xgb.XGBRegressor(**SC), Za, np.abs(r), cv=inner), 0.05)
    z = r / s_in; z = z / z.std() * r.std()                 # standardised residuals, same overall spread
    sb = np.maximum(xgb.XGBRegressor(**SC).fit(Za, np.abs(r)).predict(Zb), 0.05) / np.mean(s_in)
    z = z * 1.0; return mb, z, sb


V = ("E", "A1", "A2"); cost = {v: [] for v in V}; cov = {v: [] for v in V}
for fs in (SEED, 1, 2):
    F = list(StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y))
    oof = np.zeros(len(dev))
    for a, b in F:
        oof[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
    for a, b in F:
        fa = a[y[a] == 1]; lc = np.log(c[fa]); q = make("platt").fit(oof[a], y[a]).predict(oof[b])
        fb_mask = y[b] == 1
        for v, (params, het) in {"E": (CM, False), "A1": (CM_A1, False), "A2": (CM_A1, True)}.items():
            mb, r, sb = cost_model(params, X.iloc[fa], lc, X.iloc[b], het)
            fl, bu = decide_pv(q, mb, r, scale=sb)
            cost[v].append(per_vehicle(c[b], fl, bu))
            s = np.ones(len(b)) if sb is None else sb
            cov[v].append(np.mean(np.log(c[b][fb_mask]) <= mb[fb_mask] + s[fb_mask] * np.quantile(r, Q_STAR)))
    print(f"repeat {fs} done", flush=True)
E0 = np.array(cost["E"])
for v in V:
    x = np.array(cost[v]); d = x - E0
    print(f"{v:3s} CV cost {x.mean():.4f}±{x.std():.3f}  paired vs E {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(15):.4f}  better {int((d<0).sum())}/15  coverage {np.mean(cov[v]):.3f}")
    if v != "E":
        log_result("v9", f"cost model variant {v} (3x5 CV, k=1)", x.mean(), x.std(), notes=f"paired vs v8 cost model {d.mean():+.4f}, better {int((d<0).sum())}/15, coverage {np.mean(cov[v]):.3f}")
