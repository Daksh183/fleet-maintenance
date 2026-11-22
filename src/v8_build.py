"""v8 submissions: v2 classifier + Platt (unchanged), cost model on sensors + histogram shape features
(R2 0.66 -> 0.79, 15/15 folds), prior shift k on test."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout, write_submission
from decide import decide_pv
from features import shape
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xf, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), Xf, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM).fit(Xf, lc).predict(Xt)
print(f"test predicted failure cost: median {np.median(np.exp(m)):.0f}  (train failures median {f.repair_cost.median():.0f})")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
for k, name in ((1, "v8_k1"), (3, "v8_k3"), (4, "v8_k4"), (5, "v8_k5"), (6.53, "v8_em"), (8, "v8_k8"), (10, "v8_k10")):
    fl, bu = decide_pv(shift(q, k), m, resid)
    print(f"k={k}: ", end=""); write_submission(te, fl, bu, name)
    log_result("v8", f"shape cost model + prior shift k={k} ({name})", np.nan, np.nan, notes=f"test flag rate {fl.mean():.3%}")
