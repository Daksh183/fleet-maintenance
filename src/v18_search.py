"""v18 search: classifier families/settings, judged on CERTAIN labels (complete cases, IPCW-weighted), 3 repeats x 5 folds.
Metrics per fold: weighted AP, AUC, cost-capture@8.8% (share of failure cost in the top 8.8%), and pipeline cost/vehicle
(Platt on complete cases of the other folds, v12 decision rule, fixed cost model). OOF predictions saved per candidate."""
import sys, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from common import ROOT, load, split_holdout
from v2_params import PARAMS
from v18_censor import km_censor
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
complete = (y == 1) | (t >= 365); DRIFT = ["s120", "s122", "s126", "s128"]; SNO = [s for s in S if s not in DRIFT]
def ipcw(idx):
    G = km_censor(t[idx], y[idx]); w = np.zeros(len(idx)); cm = complete[idx]
    w[cm] = 1 / G(np.where(y[idx] == 1, np.minimum(t[idx], 365) - 1, 364)[cm]); return w
X_ = lambda **k: (lambda: xgb.XGBClassifier(**{**PARAMS, **k}))
CANDS = {
    "base_cens": (X_(), S, False), "ipcw": (X_(), S, True),
    "ipcw_d3_600": (X_(max_depth=3, n_estimators=600), S, True),
    "ipcw_d6_mcw5": (X_(max_depth=6, min_child_weight=5), S, True),
    "ipcw_slow": (X_(n_estimators=1000, learning_rate=0.02, colsample_bytree=0.5), S, True),
    "ipcw_nodrift": (X_(), SNO, True),
    "ipcw_lgb": (lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                                            subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=8), S, True),
    "ipcw_cat": (lambda: CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, verbose=0, thread_count=8, random_seed=0), S, True),
}
D6 = dict(max_depth=6, min_child_weight=5); SL = dict(n_estimators=1000, learning_rate=0.02, colsample_bytree=0.5)
CANDS.update({
    "ipcw_nodrift_d6": (X_(**D6), SNO, True), "ipcw_nodrift_slow": (X_(**SL), SNO, True), "ipcw_d6_slow": (X_(**D6, **SL), S, True),
    "ipcw_nodrift_d6_slow": (X_(**D6, **SL), SNO, True), "ipcw_d5_cs3": (X_(max_depth=5, colsample_bytree=0.3, n_estimators=500), S, True),
    "ipcw_s1": (X_(random_state=1), S, True), "ipcw_nodrift_s1": (X_(random_state=1), SNO, True),
    "cens_nodrift": (X_(), SNO, False), "cens_d6_mcw5": (X_(**D6), S, False), "cens_slow": (X_(**SL), S, False),
})
from features import shape as _shape
_SH = _shape(dev); dev = pd.concat([dev, _SH], axis=1); dev["nan_cnt"] = dev[[c for c in S if c != "s063"]].isna().sum(1)
SHC = list(_SH.columns)
CANDS.update({"ipcw_nd_nan": (X_(), SNO + ["nan_cnt"], True), "ipcw_nd_shape": (X_(), SNO + SHC, True),
              "partial_nd": (X_(), SNO, "partial")})
N = ["nan_cnt"]
CANDS.update({"ipcw_nan": (X_(), S + N, True), "ipcw_d6_nan": (X_(**D6), S + N, True), "ipcw_slow_nan": (X_(**SL), S + N, True),
              "ipcw_nd_slow_nan": (X_(**SL), SNO + N, True), "ipcw_nd_d6_nan": (X_(**D6), SNO + N, True), "ipcw_nd_nan_s1": (X_(random_state=1), SNO + N, True)})
names = sys.argv[1:] or list(CANDS)
for nm in names:
    make, cols, w_on = CANDS[nm]; P = np.zeros((3, len(y)))
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            if w_on == "partial": w = ipcw(a); cz = (w == 0); w[cz] = t[a][cz] / 365; k = w > 0; P[r, b] = make().fit(dev.loc[a[k], cols], y[a[k]], sample_weight=w[k]).predict_proba(dev.loc[b, cols])[:, 1]
            elif w_on: w = ipcw(a); k = w > 0; P[r, b] = make().fit(dev.loc[a[k], cols], y[a[k]], sample_weight=w[k]).predict_proba(dev.loc[b, cols])[:, 1]
            else: P[r, b] = make().fit(dev.loc[a, cols], y[a]).predict_proba(dev.loc[b, cols])[:, 1]
    np.save(ROOT / f"reports/v18_search/{nm}.npy", P); print("done", nm, flush=True)
