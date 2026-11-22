"""v13: variance reduction. Same pipeline as v12_robust, but the classifier (5 seeds) and the repair-cost model (8 seeds) are
seed-averaged, so budgets and flags no longer depend on one random column-sampling draw (seed-to-seed change in test cost
predictions of a single fit: ~0.135 mean abs log difference, as large as the v10->v12 change)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold
from calib import make
from common import ROOT, SEED, folds, load, log_result, split_holdout, write_submission
from decide import decide_pv
from features import shape
from v2_params import PARAMS
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); F = folds(y)
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; DRIFT = ["s120", "s122", "s126", "s128"]; NC, NK = 5, 8
# --- classifier bag: OOF on dev (for Platt) and test probabilities from a fit on all of train, averaged over seeds
oof = np.zeros(len(dev)); pte = np.zeros(len(te))
for s in range(NC):
    prm = {**PARAMS, "random_state": SEED + s}
    for a, b in F:
        oof[b] += xgb.XGBClassifier(**prm).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1] / NC
    pte += xgb.XGBClassifier(**prm).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1] / NC
    print("classifier seed", s, flush=True)
q = make("platt").fit(oof, y).predict(pte)
q1 = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
print(f"mean calibrated test p: bag {q.mean():.4f} vs single {q1.mean():.4f}; correlation {np.corrcoef(q, q1)[0, 1]:.4f}", flush=True)
# --- cost model bag (robust inputs), residuals from the bagged inner out-of-fold predictions
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
inner = KFold(5, shuffle=True, random_state=SEED); m = np.zeros(len(te)); mo = np.zeros(len(f))
for s in range(NK):
    prm = {**CM9, "random_state": SEED + s}
    for a, b in inner.split(Xa):
        mo[b] += xgb.XGBRegressor(**prm).fit(Xa.iloc[a], lc[a]).predict(Xa.iloc[b]) / NK
    m += xgb.XGBRegressor(**prm).fit(Xa, lc).predict(Xt) / NK
    print("cost seed", s, flush=True)
resid = lc - mo
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy")
pbar = sum(w * shift(q, k) for w, k in zip(W, GRID))
fl, bu = decide_pv(pbar, m, resid)
for nm in ("v12_robust", "v10_bayes", "v9_k4"):
    r = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); both = (r.flag.to_numpy() == 1) & (fl == 1)
    print(f"vs {nm:10s}: flags differ {int((r.flag.to_numpy() != fl).sum()):4d}; budget change on common flags: median {np.median(bu[both] - r.budget.to_numpy()[both]):+6.1f} mean abs {np.abs(bu[both] - r.budget.to_numpy()[both]).mean():6.1f}")
print(f"flag rate {fl.mean():.3%}")
write_submission(te, fl, bu, "v13_bag")
log_result("v13", "v13_bag: seed-averaged classifier (5) and cost model (8, drifted sensors excluded), posterior-averaged shift", np.nan, np.nan,
           notes=f"test flag rate {fl.mean():.3%}")
