"""v63: survival objectives for the failure ranking (untried family). followup_days is used only as the training target, never as input.
 cox   XGBoost Cox partial likelihood          aft   XGBoost accelerated failure time (censored vehicles: lower bound only)
Same two judges as v60 (wear-reweighted CV; extrapolation to the top 20% of s126), same inputs (no s120/s122), plus logit blends with base."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from calib import make
import v60_feats as F
dev, y, NO = F.dev, F.y, F.NO; t = np.maximum(dev.followup_days.to_numpy(float), 0.5)
P0 = dict(max_depth=4, eta=0.02, subsample=0.8, colsample_bytree=0.5, min_child_weight=1, nthread=2, seed=42)
def fit(nm, a, b):
    Xa, Xb = dev[NO].iloc[a], dev[NO].iloc[b]
    if nm == "cox":
        d = xgb.DMatrix(Xa, label=np.where(y[a] == 1, t[a], -t[a])); m = xgb.train({**P0, "objective": "survival:cox", "base_score": 1.0}, d, 1000)
        r = m.predict(xgb.DMatrix(Xb), output_margin=True)
    else:
        d = xgb.DMatrix(Xa); d.set_float_info("label_lower_bound", t[a]); d.set_float_info("label_upper_bound", np.where(y[a] == 1, t[a], np.inf))
        m = xgb.train({**P0, "objective": "survival:aft", "aft_loss_distribution": nm.split("_")[1], "aft_loss_distribution_scale": 1.0, "base_score": 365.0}, d, 1000)
        r = -m.predict(xgb.DMatrix(Xb), output_margin=True)                       # longer survival = lower risk
    return 1 / (1 + np.exp(-np.clip(r - (0 if nm == "cox" else -6), -15, 15)))   # monotone squashing; Platt recalibrates
NAMES = ["cox", "aft_normal", "aft_logistic"]
if __name__ == "__main__":
    P = {"base": np.load(F.ROOT / "reports/v18_search/v60_base.npy")}
    for n in NAMES:
        R = Parallel(n_jobs=4)(delayed(fit)(n, a, b) for r in range(3) for a, b in F.FOLDS[r]); P[n] = np.zeros((3, len(y))); i = 0
        for r in range(3):
            for a, b in F.FOLDS[r]: P[n][r, b] = R[i]; i += 1
        np.save(F.ROOT / f"reports/v18_search/v63_{n}.npy", P[n]); print("done", n, flush=True)
    L = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    def platt_all(p):                                                       # per-repeat calibrated probabilities (for blending)
        return np.stack([np.concatenate([[0.0]])] ) if False else p
    V = {}
    for n in P: V[n] = np.array([F.cost_best(make("platt").fit(P[n][r, a], y[a]).predict(P[n][r, b]), b, F.W[b]) for r in range(3) for a, b in F.FOLDS[r]])
    for n in NAMES:
        out = []
        for r in range(3):
            for a, b in F.FOLDS[r]:
                m1 = make("platt").fit(P["base"][r, a], y[a]); m2 = make("platt").fit(P[n][r, a], y[a])
                blend = lambda i: 1 / (1 + np.exp(-(L(m1.predict(P["base"][r, i])) + L(m2.predict(P[n][r, i]))) / 2))
                out.append(F.cost_best(make("platt").fit(blend(a), y[a]).predict(blend(b)), b, F.W[b]))
        V["base+" + n] = np.array(out)
    lo = np.where(~F.HI)[0]; hi = np.where(F.HI)[0]; ex = {}
    inner = list(StratifiedKFold(5, shuffle=True, random_state=1).split(lo, y[lo]))
    for n in ["base"] + NAMES:
        f_ = (lambda a, b: F.fit("base", a, b)) if n == "base" else (lambda a, b, n=n: fit(n, a, b))
        o = np.zeros(len(lo))
        for a, b in inner: o[b] = f_(lo[a], lo[b])
        q = make("platt").fit(o, y[lo]).predict(f_(lo, hi)); ex[n] = q
    for n in NAMES: ex["base+" + n] = 1 / (1 + np.exp(-(L(ex["base"]) + L(ex[n])) / 2))
    rows = []
    for n in V:
        d = V[n] - V["base"]; q = ex[n]
        rows.append(dict(model=n, wear_cv_cost=V[n].mean(), d_vs_base=d.mean(), se=d.std() / np.sqrt(15), better=int((d < 0).sum()),
                         extrap_cost=F.cost_best(q, hi, np.ones(len(hi))), extrap_AP=average_precision_score(y[hi], q), extrap_AUC=roc_auc_score(y[hi], q)))
    O = pd.DataFrame(rows).sort_values("wear_cv_cost"); O.to_csv(F.ROOT / "reports/v63_survival.csv", index=False); print(O.round(4).to_string(index=False))
