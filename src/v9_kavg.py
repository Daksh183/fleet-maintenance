"""B: decide with p averaged over a belief about k (expected cost is linear in p, so this is the Bayes decision).
Belief (current evidence): EM 5.7-6.5 (likely biased high), public-level fit ~4, scale-free fit 2-4."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout
from decide import decide_pv
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xf, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), Xf, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM).fit(Xf, lc).predict(Xt)
M = np.exp(m[:, None] + np.quantile(resid, (np.arange(99) + .5) / 99)[None, :]); mu = M.mean(1)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
def ec(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).mean()
belief = {2: .05, 3: .20, 4: .30, 5: .20, 6.5: .20, 8: .05}
pbar = sum(w * shift(q, k) for k, w in belief.items())
# effective single k with the same mean p
ks = np.linspace(1, 10, 181); keff = ks[np.argmin([abs(shift(q, k).mean() - pbar.mean()) for k in ks])]
builds = {f"k={k}": decide_pv(shift(q, k), m, resid) for k in (3, 4, 5, 6.5)}
builds["avg belief"] = decide_pv(pbar, m, resid)
truths = (2, 3, 4, 5, 6.5, 8)
T = pd.DataFrame({f"true {kt}": {n: ec(*fb, shift(q, kt)) for n, fb in builds.items()} for kt in truths})
R = T - T.min(axis=0)
T["belief-avg"] = sum(belief[kt] * T[f"true {kt}"] for kt in truths)
print(f"mean p: avg belief {pbar.mean():.4f}  (= single k {keff:.2f});  flag rate {builds['avg belief'][0].mean():.3%}")
print("expected cost / vehicle\n", T.round(4).to_string())
print("\nregret / vehicle\n", R.round(4).to_string())
print("max regret over k in [2,8]:", R.max(axis=1).round(4).to_dict())
