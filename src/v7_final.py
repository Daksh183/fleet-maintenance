"""v7 final submissions: v2 classifier (tuned one rejected: worse cost 5/5 folds), Platt, v5 cost model,
prior-shift factor k bracketed by the evidence: 3, 4 (consensus), 6.53 (EM, written by v7_build as v7a_em)."""
import numpy as np, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout, write_submission
from decide import decide_pv
from v5_percost_cfg import CM

tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), f[S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM).fit(f[S], lc).predict(te[S])
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
for k in (2, 3, 4, 5, 8, 10):
    fl, bu = decide_pv(shift(q, k), m, resid)
    print(f"k={k}: ", end=""); write_submission(te, fl, bu, f"v7_k{k}")
    if k in (3, 4): log_result("v7", f"v2 classifier, Platt, prior shift k={k} on test", np.nan, np.nan, notes=f"test flag rate {fl.mean():.3%}")
