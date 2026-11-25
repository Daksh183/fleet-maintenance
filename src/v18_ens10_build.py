"""v18_ens: v12_robust with the classifier replaced by a censoring-aware (IPCW) ensemble: probability average of ten XGBoost
variants (v2 params; v2 without the four drifted sensors; depth 6 / min_child_weight 5; 1000 trees lr 0.02 colsample 0.5).
3x5 CV on certain labels: cost -0.063/vehicle vs the v12 classifier (12/15 folds), -0.025 vs a single IPCW model (14/15).
Platt on the ensemble's out-of-fold predictions (complete cases); flag count matched to v12_robust (ranking-only change)."""
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from calib import make
from common import ROOT, load, split_holdout, write_submission, log_result
from decide import decide_pv
from v2_params import PARAMS
from v18_censor import km_censor
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
complete = (y == 1) | (t >= 365); SNO = [s for s in S if s not in ["s120", "s122", "s126", "s128"]]
D6 = {"max_depth": 6, "min_child_weight": 5}; SL = {"n_estimators": 1000, "learning_rate": 0.02, "colsample_bytree": 0.5}
VAR = {"ipcw": ({}, S), "ipcw_nodrift": ({}, SNO), "ipcw_d6_mcw5": (D6, S), "ipcw_slow": (SL, S), "ipcw_nodrift_d6": (D6, SNO),
       "ipcw_nodrift_slow": (SL, SNO), "ipcw_d6_slow": ({**D6, **SL}, S), "ipcw_nodrift_d6_slow": ({**D6, **SL}, SNO),
       "ipcw_s1": ({"random_state": 1}, S), "ipcw_nodrift_s1": ({"random_state": 1}, SNO)}
Y, T = tr.failure.to_numpy(), tr.followup_days.to_numpy(); G = km_censor(T, Y); cm = (Y == 1) | (T >= 365)
w = 1 / G(np.where(Y == 1, np.minimum(T, 365) - 1, 364)[cm])
pte = np.mean([xgb.XGBClassifier(**{**PARAMS, **kw}).fit(tr.loc[cm, cols], Y[cm], sample_weight=w).predict_proba(te[cols])[:, 1]
               for kw, cols in VAR.values()], axis=0)
oof = np.mean([np.load(ROOT / f"reports/v18_search/{k}.npy")[0] for k in VAR], axis=0)        # repeat-0 out-of-fold ensemble
q = make("platt").fit(oof[complete], y[complete]).predict(pte)
q0, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda qq, lam=1.0: sum(w_ * shift(qq, k * lam) for w_, k in zip(W, GRID))
ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv"); target = int(ref.flag.sum())
fl0, bu0 = decide_pv(pavg(q0), m, resid); assert (fl0 == ref.flag.to_numpy()).all()
lo, hi = 0.2, 4.0
for _ in range(30):
    lam = (lo + hi) / 2
    if decide_pv(pavg(q, lam), m, resid)[0].sum() < target: lo = lam
    else: hi = lam
fl, bu = decide_pv(pavg(q, lam), m, resid); both = (fl == 1) & (fl0 == 1)
ip = pd.read_csv(ROOT / "submissions/submission_v18_ipcw.csv").flag.to_numpy()
print(f"mean q {q.mean():.4f} (v12 {q0.mean():.4f}); Spearman with v12 q {spearmanr(q0, q)[0]:.4f}; lambda {lam:.3f}")
print(f"flags {int(fl.sum())}; differ from v12_robust {int((fl != fl0).sum())}, from v18_ens {int((fl != ip).sum())}; budget mean abs change {np.abs(bu[both]-bu0[both]).mean():.1f}")
np.save(ROOT / "reports/v18_ens10_q.npy", q)
write_submission(te, fl, bu, "v18_ens10")
log_result("v18", "v18_ens10: v12_robust with IPCW 10-XGB ensemble classifier, flag count matched", np.nan, np.nan, notes=f"{int(fl.sum())} flags; lambda {lam:.3f}")
