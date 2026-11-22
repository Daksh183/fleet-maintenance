"""v12: v10_bayes with ONE change: the repair-cost model does not use the four drifted sensors (s120/s122/s126/s128).
Same classifier, same Platt, same posterior-averaged shift as v10_bayes, so the public-score difference isolates the cost-model inputs.
CV: dropping them costs nothing (R2 0.801 vs 0.802); test cost predictions move by ~12% (mean abs log diff 0.118)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout, write_submission
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
GRID, W = np.load(ROOT / "reports/v10_posterior.npy")
pbar = sum(w * shift(q, k) for w, k in zip(W, GRID))
fl, bu = decide_pv(pbar, m, resid)
ref = pd.read_csv(ROOT / "submissions/submission_v10_bayes.csv")
dis = int((ref.flag.to_numpy() != fl).sum()); both = (ref.flag.to_numpy() == 1) & (fl == 1)
print(f"flag rate {fl.mean():.3%} (v10_bayes {ref.flag.mean():.3%}); flags differing {dis}; "
      f"budget change on commonly flagged: median {np.median(bu[both] - ref.budget.to_numpy()[both]):+.1f}, mean abs {np.abs(bu[both] - ref.budget.to_numpy()[both]).mean():.1f}")
write_submission(te, fl, bu, "v12_robust")
log_result("v12", "v12_robust: v10_bayes with a cost model that excludes the four drifted sensors", np.nan, np.nan,
           notes=f"test flag rate {fl.mean():.3%}; {dis} flags differ from v10_bayes; CV R2 of cost model 0.801 vs 0.802")
