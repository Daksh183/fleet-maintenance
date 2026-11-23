"""Expected public score of each file, anchored on the observed v5_platt score (22,969):
score(file) ~= 22,969 * E[cost(file)] / E[cost(v5_platt)] under a belief about the true k.
Belief model: Platt p shifted by k_true, shape cost model (v8)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xf, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), Xf, lc, cv=KFold(5, shuffle=True, random_state=SEED))
M = np.exp(xgb.XGBRegressor(**CM).fit(Xf, lc).predict(Xt)[:, None] + np.quantile(resid, (np.arange(99) + .5) / 99)[None, :]); mu = M.mean(1)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
def ec(sub, p):
    fl, b = sub.flag.to_numpy(), sub.budget.to_numpy()
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).sum()
files = ["v5_platt", "v4_platt", "v5_raw", "v8_k3", "v8_k4", "v8_em", "v8_k8"]
subs = {n: pd.read_csv(ROOT / f"submissions/submission_{n}.csv") for n in files}
rows = {}
for kt in (2, 3, 4, 6.5, 8):
    p = shift(q, kt); base = ec(subs["v5_platt"], p)
    rows[f"true k={kt}"] = {n: 22969 * ec(subs[n], p) / base for n in files}
t = pd.DataFrame(rows).round(-1).astype(int)
t["observed"] = pd.Series({"v5_platt": 22969, "v4_platt": 23876, "v5_raw": 26061})
print(t.to_string())
