"""Regret of choosing the prior-shift factor k wrongly, on TEST, using the model's own expected costs.
Rows: k used to build the submission. Columns: k assumed true. Entry: expected cost per vehicle."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict

from calib import make
from common import ROOT, SEED, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM

tr, te, S = load()
dev, _ = split_holdout(tr)
y = dev.failure.to_numpy()
oof = np.load(ROOT / "reports/p_oof_v2.npy"); p_te = np.load(ROOT / "reports/p_te_v2.npy")
q = make("platt").fit(oof, y).predict(p_te)
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), f[S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM).fit(f[S], lc).predict(te[S])
M = np.exp(m[:, None] + np.quantile(resid, (np.arange(99) + .5) / 99)[None, :]); mu = M.mean(1)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))


def exp_cost(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).mean()


K = [1, 2, 3, 4, 5, 6.5, 8, 10]
T = pd.DataFrame(index=[f"build k={k}" for k in K], columns=[f"true {k}" for k in K], dtype=float)
rate = {}
for kb in K:
    fl, b = decide_pv(shift(q, kb), m, resid); rate[kb] = fl.mean()
    for kt in K:
        T.loc[f"build k={kb}", f"true {kt}"] = exp_cost(fl, b, shift(q, kt))
R = T - T.min(axis=0)          # regret vs the best build for that truth
print("expected cost / vehicle\n", T.round(3).to_string())
print("\nregret / vehicle\n", R.round(3).to_string())
print("\nmax regret over true k in [3, 10]:", R.iloc[:, 2:].max(axis=1).round(3).to_dict())
print("max regret over true k in [1, 10]:", R.max(axis=1).round(3).to_dict())
print("test flag rate by build k:", {k: round(v, 3) for k, v in rate.items()})
