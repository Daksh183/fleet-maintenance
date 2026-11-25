"""v22_nos122: v12_robust with the classifier refit WITHOUT the two mis-scaled depot sensors s120 (offset) and s122 (scale x2.55).
s122 carries 20% of the v12 classifier's gain and has KS 0.31 between depots; dropping it costs nothing in CV (v22_advsel:
-0.024/vehicle vs all sensors, 11/15 folds). v3b's public rejection of dropping drifted sensors (20 Nov) was at k=1, where the
inflated s122 partly compensated for under-flagging; here the flag count is matched to v12_robust (1,761) so only the ranking changes.
Platt is fitted on dev OOF of the same feature set; cost model and budgets as v12 (v18_qm / v18_resid)."""
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from calib import make
from common import ROOT, folds, load, log_result, split_holdout, write_submission
from decide import decide_pv
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); F = folds(y)
cols = [s for s in S if s not in ("s063", "s120", "s122")]
oof = np.zeros(len(dev))
for a, b in F: oof[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, cols], y[a]).predict_proba(dev.loc[b, cols])[:, 1]
raw = xgb.XGBClassifier(**PARAMS).fit(tr[cols], tr.failure).predict_proba(te[cols])[:, 1]
q = make("platt").fit(oof, y).predict(raw)
qv12, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda qq, lam=1.0: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))
fl0, bu0 = decide_pv(pavg(qv12), m, resid); ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv")
assert (ref.flag.to_numpy() == fl0).all(); target = int(fl0.sum())
print(f"mean calibrated test q: no-s122 {q.mean():.4f} vs v12 {qv12.mean():.4f}; Spearman {spearmanr(q, qv12)[0]:.3f}; "
      f"flags at lambda=1: {int(decide_pv(pavg(q), m, resid)[0].sum())} (v12 {target})")
lo, hi = 0.2, 8.0
for _ in range(40):
    lam = (lo + hi) / 2
    if decide_pv(pavg(q, lam), m, resid)[0].sum() < target: lo = lam
    else: hi = lam
fl, bu = decide_pv(pavg(q, hi), m, resid); bu = np.minimum(bu, 8000)
print(f"matched lambda {hi:.3f}: {int(fl.sum())} flags, {int((fl != fl0).sum())} differ from v12 ({int(((fl==1)&(fl0==0)).sum())} added, {int(((fl==0)&(fl0==1)).sum())} removed)")
r12 = pd.read_csv(ROOT / "data/test.csv", usecols=["s122", "s126"])
for nm, f_ in (("added", (fl == 1) & (fl0 == 0)), ("removed", (fl == 0) & (fl0 == 1)), ("kept", (fl == 1) & (fl0 == 1))):
    print(f"  {nm:8s} median test s122 {np.nanmedian(te.s122[f_]):10.0f}  median s126 {np.nanmedian(te.s126[f_]):10.0f}  mean q {q[f_].mean():.3f} / v12 q {qv12[f_].mean():.3f}")
np.save(ROOT / "reports/v22_q.npy", q)
write_submission(te, fl, bu, "v22_nos122")
log_result("v22", "v22_nos122: v12_robust with classifier without s120/s122 (mis-scaled at the test depot), flag count matched, budgets capped 8000",
           np.nan, np.nan, notes=f"{int(fl.sum())} flags; lambda {hi:.3f}; {int((fl != fl0).sum())} differ from v12")
