"""v45: does the GP repair-cost model improve FLAGS (not just budgets)? Shift-aware CV (wear-reweighted validation, each candidate at its
own best level), failure model = slow no-s122 XGB (best ranking in v41). Expected repair cost per vehicle from:
 tree (v12 cost model, dev OOF), GP (v39 inputs, refit on each training fold's failures), and 50/50 blend in log space.
Budgets for every candidate use the same rule (newsvendor with that candidate's cost distribution) so the comparison is end to end."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "8")
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); c = dev.repair_cost.fillna(0).to_numpy() * y
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
h = lambda x: np.histogram(np.log1p(x[np.isfinite(x)]), e)[0] / np.isfinite(x).sum()
ratio = np.clip(h(te.s126.to_numpy()) / np.maximum(h(tr.s126.to_numpy()), 1e-9), 0, 5)
W = ratio[np.clip(np.searchsorted(e, np.log1p(dev.s126.fillna(dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)]
NO = [s for s in S if s not in ("s120", "s122")]; prm = {**PARAMS, "n_estimators": 1000, "learning_rate": 0.02, "colsample_bytree": 0.5}
WEAR = ["s126", "s149", "s123"]; SENS = ["s116", "s142", "s105", "s104", "s078"]
def feats(d, ref):
    out = extract(d)
    for cols in (WEAR, SENS):
        R = np.log1p(np.maximum(ref[cols].to_numpy(float), 0)); mu_, sd_ = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
        out.append((np.log1p(np.maximum(d[cols].to_numpy(float), 0)) - mu_) / sd_)
    return out
res = {k: [] for k in ("tree", "gp", "blend")}
for r in range(3):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
        Da, Db = dev.loc[a], dev.loc[b]
        oofa = np.zeros(len(a)); 
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=1).split(np.zeros(len(a)), y[a]):
            oofa[ib] = xgb.XGBClassifier(**prm).fit(Da.iloc[ia][NO], y[a][ia]).predict_proba(Da.iloc[ib][NO])[:, 1]
        q = make("platt").fit(oofa, y[a]).predict(xgb.XGBClassifier(**prm).fit(Da[NO], y[a]).predict_proba(Db[NO])[:, 1])
        fa = Da[Da.failure == 1]; ly = np.log(fa.repair_cost.to_numpy()); Fa = feats(fa, fa)
        go = np.full(len(fa), np.nan)
        for ia, ib in KFold(5, shuffle=True, random_state=42).split(ly): go[ib] = GaussianSeverity().fit(select(Fa, ia), ly[ia]).predict(select(Fa, ib))
        gres = (ly - go)[np.isfinite(go)]; gb = GaussianSeverity().fit(Fa, ly).predict(feats(Db, fa)); sup = np.isfinite(gb)
        mt = md[b]
        cands = {"tree": (mt, resid, None), "gp": (np.where(sup, gb, mt), gres, sup), "blend": (np.where(sup, 0.5 * gb + 0.5 * mt, mt), 0.5 * (np.sort(gres)[np.linspace(0, len(gres) - 1, 200).astype(int)] + np.sort(resid)[np.linspace(0, len(resid) - 1, 200).astype(int)]), sup)}
        w = W[b]
        for nm, (mm, rr, _) in cands.items():
            best = np.inf
            for lam in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
                qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, mm, rr)
                cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[b] - bu, 0) + OVER * np.maximum(bu - c[b], 0), c[b]); best = min(best, (cost * w).sum() / w.sum())
            res[nm].append(best)
    print("repeat", r, flush=True)
R = {k: np.array(v) for k, v in res.items()}
for k, v in R.items():
    d = v - R["tree"]; print(f"{k:6s} cost/veh {v.mean():.4f}  vs tree {d.mean():+.4f}±{d.std() / np.sqrt(15):.4f}  better {(d < 0).sum()}/15")
