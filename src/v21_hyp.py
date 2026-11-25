"""v21: which risk ranking do the 17 public scores support? Model comparison, not a new submission.
Each hypothesis H gives raw test probabilities r_H. Truth model: p = sigmoid(a + b*logit(r_H)); repair cost as in v18 (alpha level).
For every H, maximise the exact Gaussian likelihood of the 17 public scores (v18 machinery) over (a, b, alpha).
A hypothesis whose ranking is closer to the truth should explain the scores of all 17 files better.
 H base   : v12 classifier (all sensors), test as loaded
 H wear   : same model, but the test depot's s120 offset is treated as REAL wear: s126/s128/s012 raised by the same amount
 H nodrift: classifier refit without the wear counters s012/s120/s122/s126/s128
 H nocnt  : classifier refit without any of the five counters AND with s126-free shape features (pure usage histograms)"""
import numpy as np, pandas as pd, xgboost as xgb
from common import ROOT, load
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
tr, te, S = load()
qv12, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
x = te[["s120", "s126"]].dropna(); sl, off = np.linalg.lstsq(np.c_[x.s126, np.ones(len(x))], x.s120, rcond=None)[0]
d126 = off / sl
te_w = te.copy(); te_w["s126"] += d126; te_w["s128"] += 1.117 * d126; te_w["s012"] += 3.406 * d126
CNT = ["s012", "s120", "s122", "s126", "s128"]; S2 = [s for s in S if s not in CNT]
mdl = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
H = {"base": mdl.predict_proba(te[S])[:, 1], "wear": mdl.predict_proba(te_w[S])[:, 1],
     "nodrift": xgb.XGBClassifier(**PARAMS).fit(tr[S2], tr.failure).predict_proba(te[S2])[:, 1]}
np.save(ROOT / "reports/v21_H.npy", np.vstack(list(H.values())))
for k, v in H.items(): print(f"{k:8s} mean raw p {v.mean():.4f}  spearman vs base {pd.Series(v).corr(pd.Series(H['base']), method='spearman'):.3f}", flush=True)

OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718, "v5_platt": 22968.5508,
       "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558, "v9_k4": 17013.8646, "v10_bayes": 17348.7454, "v12_robust": 16839.8576,
       "v13_bag": 17974.5456, "v14_map": 17375.6526, "v15_map_v12": 17058.2816, "v17_bud125": 16826.0, "v17_k135": 16908.0,
       "v18_ens10": 17624.0}
names = list(OBS); obs = np.array([OBS[k] for k in names])
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 33; base = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :])
U0 = Fl * (INSPECT + OVER * Bu); U00 = np.einsum("nj,nl->njl", U0, U0)
pre = {}
for al in (0.9, 1.0, 1.1, 1.25):
    C = base * al
    U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
    pre[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)
sig = lambda z: 1 / (1 + np.exp(-z))
for EPS in (0.02, 0.04):
    print(f"\nEPS {EPS}")
    for hn, r in H.items():
        lg = np.log(r / (1 - r)); best = (-1e18,)
        for al, (E1, G) in pre.items():
            for b in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5):
                for a in np.linspace(-3, 3, 41):
                    p = sig(a + b * lg)
                    mi = (1 - p)[:, None] * U0 + p[:, None] * E1; mean = 0.3 * mi.sum(0)
                    cov = 0.3 * (np.einsum("n,njl->jl", 1 - p, U00) + np.einsum("n,njl->jl", p, G)) - 0.09 * mi.T @ mi + np.diag((EPS * mean) ** 2)
                    L = np.linalg.cholesky(cov); z = np.linalg.solve(L, obs - mean); ll = -0.5 * z @ z - np.log(np.diag(L)).sum()
                    if ll > best[0]: best = (ll, al, b, a, p.mean(), mean)
        ll, al, b, a, mp, mean = best
        print(f"{hn:8s} max ll {ll:8.2f}  alpha {al} b {b} a {a:+.2f} mean p {mp:.4f}  worst |z| file: "
              f"{names[int(np.argmax(np.abs(obs - mean)))]} ({(obs - mean)[int(np.argmax(np.abs(obs - mean)))]:+.0f})", flush=True)
        if EPS == 0.02: print("   resid:", dict(zip(names[6:], (obs - mean)[6:].round(0))))
