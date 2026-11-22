"""Validate prior-shift correction on SIMULATED label shift inside dev (labels known).

For each outer fold: Platt fitted on the other folds' OOF; nested cost model as in v5.
Shift the validation fold to odds x k_true by keeping all failures and subsampling healthy
vehicles at 1/k_true (pure label shift: p(x|y) unchanged). Then compare decisions made with
  none   : Platt p as is (v5)
  EM     : prior estimated by EM from the fold's p only (no labels)
  oracle : true k
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict

from calib import make
from common import ROOT, SEED, folds, load, split_holdout
from decide import decide_pv
from score import per_vehicle, true_cost
from v5_percost_cfg import CM

tr, te, S = load()
dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev)
F = folds(y)
oof = np.load(ROOT / "reports/p_oof_v2.npy")
rng = np.random.default_rng(SEED)


def shift(p, k):
    o = k * p / (1 - p)
    return o / (1 + o)


def em_k(p, pi_tr, it=300):
    pi = pi_tr
    for _ in range(it):
        a = pi / pi_tr * p
        pi = (a / (a + (1 - pi) / (1 - pi_tr) * (1 - p))).mean()
    return (pi / (1 - pi)) / (pi_tr / (1 - pi_tr))


fold_parts = []
for a, b in F:
    fa = a[y[a] == 1]; lc = np.log(c[fa])
    resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), dev.loc[fa, S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
    m_b = xgb.XGBRegressor(**CM).fit(dev.loc[fa, S], lc).predict(dev.loc[b, S])
    q_b = make("platt").fit(oof[a], y[a]).predict(oof[b])
    fold_parts.append((b, q_b, m_b, resid, y[a].mean()))

print("k_true  EM k (mean, range over folds x reps)    cost/veh: none     EM      oracle   EM gain vs none (folds better)")
for k_true in (1, 2, 4, 6):
    est, res = [], {"none": [], "EM": [], "oracle": []}
    for rep in range(4):
        for b, q_b, m_b, resid, pi_tr in fold_parts:
            keep = (y[b] == 1) | (rng.random(len(b)) < 1 / k_true)
            q, m, cc = q_b[keep], m_b[keep], c[b][keep]
            k_hat = em_k(q, pi_tr); est.append(k_hat)
            for name, kk in (("none", 1), ("EM", k_hat), ("oracle", k_true)):
                fl, bu = decide_pv(shift(q, kk), m, resid)
                res[name].append(per_vehicle(cc, fl, bu))
    r = {n: np.array(v) for n, v in res.items()}
    g = r["EM"] - r["none"]
    print(f"{k_true:>4}    {np.mean(est):5.2f}  [{np.min(est):.2f}, {np.max(est):.2f}]"
          f"                {r['none'].mean():.3f}   {r['EM'].mean():.3f}   {r['oracle'].mean():.3f}    "
          f"{g.mean():+.3f}  ({int((g < 0).sum())}/{len(g)})")
