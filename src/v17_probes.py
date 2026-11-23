"""v17: the two level probes on the v12_robust base (the mapping showed no reliable effect, so it is dropped).
 A) v17_bud125: v12_robust flags, flagged budgets x1.25.   B) v17_k135: v12_robust with shift scale x1.35 (more flags)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout, write_submission, log_result
from decide import decide_pv
from features import shape
from v5_percost_cfg import CM
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; DRIFT = ["s120", "s122", "s126", "s128"]
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda lam: sum(w * shift(q, k * lam) for w, k in zip(W, GRID))
fl, bu = decide_pv(pavg(1.0), m, resid); ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv")
print("v12_robust reproduced:", bool((ref.flag.to_numpy() == fl).all()), "max budget diff", np.abs(ref.budget.to_numpy() - bu).max())
write_submission(te, fl, np.where(fl == 1, bu * 1.25, bu), "v17_bud125")
fl2, bu2 = decide_pv(pavg(1.35), m, resid); write_submission(te, fl2, bu2, "v17_k135")
log_result("v17", "v17_bud125: v12_robust with flagged budgets x1.25", np.nan, np.nan, notes=f"{int(fl.sum())} flags")
log_result("v17", "v17_k135: v12_robust with shift scale x1.35", np.nan, np.nan, notes=f"{int(fl2.sum())} flags")
