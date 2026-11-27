"""v64: confirm v63's AFT result on FRESH folds (seeds 200-204, 5 x 5) and vary one setting at a time. Same judges as v60/v63."""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from calib import make
import v60_feats as F, v63_survival as V6
dev, y, NO, S = F.dev, F.y, F.NO, F.S; t = V6.t
AFT = {"aft s1": {}, "aft s0.5": dict(scale=0.5), "aft s2": dict(scale=2.0), "aft s1 d3": dict(max_depth=3), "aft s1 d6": dict(max_depth=6, min_child_weight=5),
       "aft s1 all sensors": dict(cols=S), "aft s1 600 trees": dict(rounds=600), "aft s1 2000 trees": dict(rounds=2000), "aft s1 seed 7": dict(seed=7)}
def aft(a, b, scale=1.0, cols=None, rounds=1000, **kw):
    cols = cols or NO
    d = xgb.DMatrix(dev[cols].iloc[a]); d.set_float_info("label_lower_bound", t[a]); d.set_float_info("label_upper_bound", np.where(y[a] == 1, t[a], np.inf))
    m = xgb.train({**V6.P0, "objective": "survival:aft", "aft_loss_distribution": "normal", "aft_loss_distribution_scale": scale, "base_score": 365.0, **kw}, d, rounds)
    r = -m.predict(xgb.DMatrix(dev[cols].iloc[b]), output_margin=True) / scale
    return 1 / (1 + np.exp(-np.clip(r + 6, -13, 13)))
def fit(nm, a, b): return F.fit(nm.split()[1], a, b) if nm.startswith("clf") else aft(a, b, **AFT[nm])
NAMES = ["clf base", "clf all"] + list(AFT)
L = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
BL = {"clf base + aft s1": ("clf base", "aft s1"), "clf all + aft s1": ("clf all", "aft s1"), "clf base + clf all + aft s1 + aft all": ("clf base", "clf all", "aft s1", "aft s1 all sensors"),
      "aft s1 + aft s0.5 + aft s2": ("aft s1", "aft s0.5", "aft s2")}
if __name__ == "__main__":
    FOLDS = [list(StratifiedKFold(5, shuffle=True, random_state=200 + r).split(np.zeros(len(y)), y)) for r in range(5)]
    jobs = [(n, r, k) for n in NAMES for r in range(5) for k in range(5)]
    R = Parallel(n_jobs=4)(delayed(fit)(n, *FOLDS[r][k]) for n, r, k in jobs); P = {n: np.zeros((5, len(y))) for n in NAMES}
    for (n, r, k), p in zip(jobs, R): P[n][r, FOLDS[r][k][1]] = p
    for n in NAMES: np.save(F.ROOT / f"reports/v18_search/v64_{n.replace(' ', '_')}.npy", P[n])
    print("cv fitted", flush=True)
    V = {}
    for n in NAMES: V[n] = np.array([F.cost_best(make("platt").fit(P[n][r, a], y[a]).predict(P[n][r, b]), b, F.W[b]) for r in range(5) for a, b in FOLDS[r]])
    for n, parts in BL.items():
        out = []
        for r in range(5):
            for a, b in FOLDS[r]:
                ms = [make("platt").fit(P[q][r, a], y[a]) for q in parts]
                blend = lambda i: 1 / (1 + np.exp(-np.mean([L(m.predict(P[q][r, i])) for m, q in zip(ms, parts)], 0)))
                out.append(F.cost_best(make("platt").fit(blend(a), y[a]).predict(blend(b)), b, F.W[b]))
        V[n] = np.array(out)
    lo = np.where(~F.HI)[0]; hi = np.where(F.HI)[0]; inner = list(StratifiedKFold(5, shuffle=True, random_state=2).split(lo, y[lo])); ex = {}
    jobs = [(n, i) for n in NAMES for i in range(6)]
    R = Parallel(n_jobs=4)(delayed(fit)(n, *((lo[inner[i][0]], lo[inner[i][1]]) if i < 5 else (lo, hi))) for n, i in jobs)
    for n in NAMES:
        o = np.zeros(len(lo))
        for (m, i), p in zip(jobs, R):
            if m == n and i < 5: o[inner[i][1]] = p
            elif m == n: ph = p
        ex[n] = make("platt").fit(o, y[lo]).predict(ph)
    for n, parts in BL.items(): ex[n] = 1 / (1 + np.exp(-np.mean([L(ex[q]) for q in parts], 0)))
    rows = []
    for n in V:
        d = V[n] - V["clf base"]; q = ex[n]; per = d.reshape(5, 5).mean(1)
        rows.append(dict(model=n, wear_cv=V[n].mean(), d_vs_base=d.mean(), se=d.std() / np.sqrt(25), folds=int((d < 0).sum()), repeats=int((per < 0).sum()),
                         extrap_cost=F.cost_best(q, hi, np.ones(len(hi))), AP=average_precision_score(y[hi], q), AUC=roc_auc_score(y[hi], q)))
    O = pd.DataFrame(rows).sort_values("wear_cv"); O.to_csv(F.ROOT / "reports/v64_aft.csv", index=False); print(O.round(4).to_string(index=False))
