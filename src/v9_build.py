"""v9 submissions: as v8, but the cost model has more capacity (1000 trees, depth 2: R2 .798 -> .810,
budget loss -3.5%, less shrinkage at the extremes). Same probe design: k1 (cost model only), k4, EM."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout, write_submission
from decide import decide_pv
from features import shape
from v5_percost_cfg import CM
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xf, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xf, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xf, lc).predict(Xt)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
for k, name in ((1, "v9_k1"), (3, "v9_k3"), (4, "v9_k4"), (5, "v9_k5"), (6.53, "v9_em"), (8, "v9_k8")):
    fl, bu = decide_pv(shift(q, k), m, resid)
    print(f"k={k}: ", end=""); write_submission(te, fl, bu, name)
    log_result("v9", f"v9 cost model (1000 tr d2) + prior shift k={k} ({name})", np.nan, np.nan, notes=f"test flag rate {fl.mean():.3%}")
v8 = pd.read_csv(ROOT / "submissions/submission_v8_k4.csv"); v9 = pd.read_csv(ROOT / "submissions/submission_v9_k4.csv")
print(f"v8_k4 vs v9_k4: flags differ {int((v8.flag != v9.flag).sum())}, median |budget change| among budgeted {np.median(np.abs(v8.budget - v9.budget)[v9.budget > 0]):.1f}")
