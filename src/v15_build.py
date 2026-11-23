"""v15: v12_robust (single-seed v2 classifier, cost model without drifted sensors) with ONE change: test s120/s122 quantile-mapped onto the
training distribution for the classifier. Flag count matched to v12_robust, so the comparison v12_robust -> v15 isolates the mapping on the
same base (v13_bag -> v14_map gave -599)."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout, write_submission
from decide import decide_pv
from features import shape
from v2_params import PARAMS
from v5_percost_cfg import CM
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; DRIFT = ["s120", "s122", "s126", "s128"]; MAP = ["s120", "s122"]
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
def qmap(col):
    x = te[col]; r = x.rank(pct=True, method="average"); out = x.copy(); ok = x.notna()
    out[ok] = np.quantile(tr[col].dropna(), np.clip((r[ok] * ok.sum() - 0.5) / ok.sum(), 0, 1)); return out
te_map = te.copy()
for s in MAP: te_map[s] = qmap(s)
mdl = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
p_raw, p_map = mdl.predict_proba(te[S])[:, 1], mdl.predict_proba(te_map[S])[:, 1]
saved = np.load(ROOT / "reports/p_te_v2.npy"); print("refit reproduces saved p_te_v2: max abs diff", np.abs(p_raw - saved).max())
pl = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y); q, qm = pl.predict(saved), pl.predict(p_map)
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy")
pavg = lambda qq, lam: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))
ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv"); target = int(ref.flag.sum())
lo, hi = 0.5, 4.0
for _ in range(25):
    lam = (lo + hi) / 2
    if decide_pv(pavg(qm, lam), m, resid)[0].sum() < target: lo = lam
    else: hi = lam
fl, bu = decide_pv(pavg(qm, lam), m, resid); both = (ref.flag.to_numpy() == 1) & (fl == 1)
print(f"lambda {lam:.3f}; mean p {pavg(qm, lam).mean():.4f} vs v12 {pavg(q, 1.0).mean():.4f}; flags {fl.sum()} (v12 {target}); differing {int((ref.flag.to_numpy() != fl).sum())}; "
      f"budget mean abs change {np.abs(bu[both] - ref.budget.to_numpy()[both]).mean():.1f}")
write_submission(te, fl, bu, "v15_map_v12")
log_result("v15", "v15_map_v12: v12_robust with test s120/s122 quantile-mapped for the classifier (flag count matched)", np.nan, np.nan,
           notes=f"{int(fl.sum())} flags; {int((ref.flag.to_numpy() != fl).sum())} differ from v12_robust; lambda {lam:.3f}")
