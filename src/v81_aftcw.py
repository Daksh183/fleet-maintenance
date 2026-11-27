"""v81: survival (AFT) members trained with failures weighted by sqrt(repair cost / mean). Dev OOF -> Platt; locked hold-out check against
the unweighted AFT ensemble and the v12-style classifier (AP, and cost per vehicle at the best flag level); test q -> reports/v81_q.npz."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
import v65_aft_holdout as V
import v63_survival as V6
tr, te, S = load(); NO = [s for s in S if s not in ("s120", "s122")]; dev, ho = split_holdout(tr); y = dev.failure.to_numpy(); yh = ho.failure.to_numpy()
MEM = {"s1": {}, "s2": dict(scale=2.0), "d6": dict(max_depth=6, min_child_weight=5)}
def aftw(D, E, scale=1.0, seed=42, pw=0.5, **kw):
    t = np.maximum(D.followup_days.to_numpy(float), 0.5); e = D.failure.to_numpy(); c = D.repair_cost.fillna(0).to_numpy()
    w = np.where(e == 1, (np.maximum(c, 5) / c[e == 1].mean()) ** pw, 1.0)
    d = xgb.DMatrix(D[NO]); d.set_float_info("label_lower_bound", t); d.set_float_info("label_upper_bound", np.where(e == 1, t, np.inf)); d.set_weight(w)
    m = xgb.train({**V6.P0, "seed": seed, "objective": "survival:aft", "aft_loss_distribution": "normal", "aft_loss_distribution_scale": scale, "base_score": 365.0, **kw}, d, 1000)
    return 1 / (1 + np.exp(-np.clip(-m.predict(xgb.DMatrix(E[NO]), output_margin=True) / scale + 6, -13, 13)))
folds = list(StratifiedKFold(5, shuffle=True, random_state=11).split(dev, y))
jobs = [(m, "oof", k) for m in MEM for k in range(5)] + [(m, w, 0) for m in MEM for w in ("ho", "te")] + [(m, "ho_u", 0) for m in MEM]
def run(m, kind, k):
    kw = MEM[m]
    if kind == "oof": a, b = folds[k]; return aftw(dev.iloc[a], dev.iloc[b], **kw)
    if kind == "ho": return aftw(dev, ho, **kw)
    if kind == "ho_u": kw2 = dict(kw); return V.aft(dev, ho, **kw2)          # unweighted member on the hold-out, same params
    return np.mean([aftw(tr, te, seed=42 + 10 * i, **kw) for i in range(3)], 0)
R = dict(zip(jobs, Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs)))
L, sig = V.L, V.sig; Q = {}
for m in MEM:
    o = np.zeros(len(y))
    for k, (a, b) in enumerate(folds): o[b] = R[(m, "oof", k)]
    cal = make("platt").fit(o, y); Q[m] = (cal.predict(R[(m, "ho", 0)]), cal.predict(R[(m, "te", 0)]))
FILES = {"s1": "aft_s1", "s2": "aft_s2", "d6": "aft_s1_d6"}
qu = sig(np.mean([L(make("platt").fit(np.load(ROOT / f"reports/v18_search/v64_{FILES[m]}.npy")[0], y).predict(R[(m, "ho_u", 0)])) for m in MEM], 0))
qw = sig(np.mean([L(Q[m][0]) for m in MEM], 0))
clfo = np.load(ROOT / "reports/v18_search/v64_clf_all.npy")[0]; qc = make("platt").fit(clfo, y).predict(V.clf(dev, ho, S))
# hold-out cost at each candidate's best level (repair-cost model as in v65: per-vehicle log-mean from a regressor on dev failures)
md_ho = np.load(ROOT / "reports/v18_qm.npy") if False else None
fa = dev[dev.failure == 1]; ly = np.log(fa.repair_cost.to_numpy()); cm = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
NODR = [s for s in S if s not in ("s120", "s122", "s126", "s128")]
from sklearn.model_selection import KFold
oo = np.zeros(len(ly))
for a, b in KFold(5, shuffle=True, random_state=42).split(ly): oo[b] = xgb.XGBRegressor(**cm).fit(fa[NODR].iloc[a], ly[a]).predict(fa[NODR].iloc[b])
mh = xgb.XGBRegressor(**cm).fit(fa[NODR], ly).predict(ho[NODR]); res = ly - oo; ch = ho.repair_cost.fillna(0).to_numpy() * yh
def best_cost(q):
    out = []
    for lam in (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0):
        qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, mh, res)
        out.append(np.where(fl == 1, INSPECT + UNDER * np.maximum(ch - bu, 0) + OVER * np.maximum(bu - ch, 0), ch).mean())
    return min(out)
rng = np.random.default_rng(0); B = rng.integers(0, len(yh), (500, len(yh)))
base = best_cost(qc)
for nm, q in (("v12-style classifier", qc), ("AFT unweighted", qu), ("AFT sqrt-cost weighted", qw), ("both AFT (logit avg)", sig((L(qu) + L(qw)) / 2))):
    print(f"hold-out {nm:24s} AP {average_precision_score(yh, q):.4f}  cost/veh {best_cost(q):.4f} ({100 * (best_cost(q) / base - 1):+.1f}% vs classifier)")
q65 = np.load(ROOT / "reports/v65_q.npz"); qte = sig(np.mean([L(Q[m][1]) for m in MEM], 0))
np.savez(ROOT / "reports/v81_q.npz", aftcw=qte, aftcw_aft=sig((L(qte) + L(q65["aft"])) / 2))
from scipy.stats import spearmanr
print("test spearman aftcw vs aft", round(spearmanr(qte, q65["aft"]).correlation, 3), "| vs v12", round(spearmanr(qte, np.load(ROOT / "reports/v18_qm.npy")[0]).correlation, 3))
