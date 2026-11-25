"""v18_ipcw: v12_robust with ONE change: the failure classifier is trained censoring-aware (IPCW: censored healthy rows
dropped, complete rows weighted 1/G(t)), and Platt-calibrated on complete cases only. Flag count matched to v12_robust
(shift scale lambda by bisection), so the public comparison isolates ranking."""
import numpy as np, pandas as pd, xgboost as xgb
from calib import make
from common import ROOT, load, split_holdout, write_submission, log_result
from decide import decide_pv
from v2_params import PARAMS
from v18_censor import km_censor
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
complete = (y == 1) | (t >= 365)
Y, T = tr.failure.to_numpy(), tr.followup_days.to_numpy(); G = km_censor(T, Y)
cm = (Y == 1) | (T >= 365); w = 1 / G(np.where(Y == 1, np.minimum(T, 365) - 1, 364)[cm])
pte = xgb.XGBClassifier(**PARAMS).fit(tr.loc[cm, S], Y[cm], sample_weight=w).predict_proba(te[S])[:, 1]
pi = np.load(ROOT / "reports/v18_p_oof_ipcw.npy")
q = make("platt").fit(pi[complete], y[complete]).predict(pte)
q0, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda qq, lam=1.0: sum(w_ * shift(qq, k * lam) for w_, k in zip(W, GRID))
ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv"); target = int(ref.flag.sum())
fl0, bu0 = decide_pv(pavg(q0), m, resid); assert (fl0 == ref.flag.to_numpy()).all(), "v12_robust not reproduced"
lo, hi = 0.2, 4.0
for _ in range(30):
    lam = (lo + hi) / 2
    if decide_pv(pavg(q, lam), m, resid)[0].sum() < target: lo = lam
    else: hi = lam
fl, bu = decide_pv(pavg(q, lam), m, resid); both = (fl == 1) & (fl0 == 1)
from scipy.stats import spearmanr
print(f"mean q: v12 {q0.mean():.4f} ipcw {q.mean():.4f}; Spearman {spearmanr(q0, q)[0]:.4f}; lambda {lam:.3f}")
print(f"flags {int(fl.sum())} (v12 {target}); differ {int((fl != fl0).sum())}; budget mean abs change on common flags {np.abs(bu[both]-bu0[both]).mean():.1f}")
write_submission(te, fl, bu, "v18_ipcw")
log_result("v18", "v18_ipcw: v12_robust with censoring-aware (IPCW) classifier, flag count matched", np.nan, np.nan, notes=f"{int(fl.sum())} flags; lambda {lam:.3f}")
