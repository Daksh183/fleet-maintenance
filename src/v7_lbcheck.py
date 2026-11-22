"""Which belief about the test depot explains the public scores we already have?

For each belief p_i (raw, Platt, prior-shifted Platt) and the v5 per-vehicle cost model, compute the
EXPECTED competition cost of each submitted file on a random 30% of test (0.3 x sum over test).
Used only as a consistency check of a single scalar (test failure level), not to fit anything per vehicle.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict

from calib import make
from common import ROOT, SEED, load, split_holdout
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM

tr, te, S = load()
dev, _ = split_holdout(tr)
y = dev.failure.to_numpy()
oof = np.load(ROOT / "reports/p_oof_v2.npy"); p_te = np.load(ROOT / "reports/p_te_v2.npy")
q_te = make("platt").fit(oof, y).predict(p_te)

f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), f[S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
m_te = xgb.XGBRegressor(**CM).fit(f[S], lc).predict(te[S])
M = np.exp(m_te[:, None] + np.quantile(resid, (np.arange(99) + .5) / 99)[None, :])
mu = M.mean(1)


def shift(p, k):           # prior-shift p by odds factor k
    o = k * p / (1 - p)
    return o / (1 + o)


def expected(sub, p):
    fl, b = sub.flag.to_numpy(), sub.budget.to_numpy()
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    e = np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu)
    return 0.3 * e.sum()


obs = {"v2": 74933, "v3": 27760, "v3b": 29530, "v4_platt": 23876, "v5_raw": 26061, "v5_platt": 22969}
subs = {k: pd.read_csv(ROOT / f"submissions/submission_{k}.csv") for k in obs}
subs["nobody"] = subs["v3"].assign(flag=0, budget=0.0)
beliefs = {"raw": p_te, "platt": q_te}
for k in (2, 4, 6.5, 10, 15):
    beliefs[f"platt x{k}"] = shift(q_te, k)
rows = {b: {s: expected(subs[s], p) for s in subs} for b, p in beliefs.items()}
tab = pd.DataFrame(rows).round(0)
tab.insert(0, "observed", pd.Series(obs))
print(tab.to_string())
o = pd.Series(obs)
print("\nRMS log error vs observed (levels):  ", {b: round(float(np.sqrt(np.mean(np.log(tab.loc[o.index, b] / o) ** 2))), 3) for b in beliefs})
d = o - o["v5_platt"]
print("RMS error of DIFFERENCES vs v5_platt:", {b: round(float(np.sqrt(np.mean(((tab.loc[o.index, b] - tab.loc['v5_platt', b]) - d) ** 2))), 0) for b in beliefs})
print("mean test failure rate implied by each belief:", {b: round(float(p.mean()), 4) for b, p in beliefs.items()})
# scale-free check (does not assume the public set is 30%): demeaned log error across submissions
L = np.log(tab.loc[o.index, list(beliefs)].astype(float)).sub(np.log(o), axis=0)
print("scale-free RMS (demeaned log error):", (L - L.mean()).pow(2).mean().pow(.5).round(3).to_dict())
print("implied public-set scale factor (1 = 30% of test):", np.exp(-L.mean()).round(2).to_dict())
