"""v16: two one-change level probes on the v15 ranking (the biggest public-score lever so far has been LEVEL, not ranking).
 A) v16_bud125: v15 flags, flagged budgets x1.25 (is the test repair-cost level above the training model's?).
 B) v16_k135: v15 with the shift scale lambda x1.35 (more flags; is the true depot shift above the posterior mean?)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout, write_submission, log_result
from decide import decide_pv
from features import shape
from v2_params import PARAMS
from v5_percost_cfg import CM
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; DRIFT = ["s120", "s122", "s126", "s128"]; MAP = ["s120", "s122"]
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
te_map = te.copy()
for col in MAP:
    x = te[col]; r = x.rank(pct=True); ok = x.notna(); o = x.copy()
    o[ok] = np.quantile(tr[col].dropna(), np.clip((r[ok] * ok.sum() - .5) / ok.sum(), 0, 1)); te_map[col] = o
p_map = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te_map[S])[:, 1]
qm = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(p_map)
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda lam: sum(w * shift(qm, k * lam) for w, k in zip(W, GRID))
LAM = 1.769
fl, bu = decide_pv(pavg(LAM), m, resid); v15 = pd.read_csv(ROOT / "submissions/submission_v15_map_v12.csv")
print("v15 reproduced:", bool((v15.flag.to_numpy() == fl).all()), "max budget diff", np.abs(v15.budget.to_numpy() - bu).max())
write_submission(te, fl, np.where(fl == 1, bu * 1.25, bu), "v16_bud125")
fl2, bu2 = decide_pv(pavg(LAM * 1.35), m, resid); write_submission(te, fl2, bu2, "v16_k135")
log_result("v16", "v16_bud125: v15 with flagged budgets x1.25", np.nan, np.nan, notes=f"{int(fl.sum())} flags")
log_result("v16", "v16_k135: v15 with shift scale x1.35", np.nan, np.nan, notes=f"{int(fl2.sum())} flags")
