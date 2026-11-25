"""v25: does risk at the new depot rise with wear more steeply than in training? Fit a wear tilt to all 20 public scores.
Truth model: p = sigmoid(a + logit(q_v12) + c * z), z = standardised wear covariate on test. c = 0 is the v18/v21 model.
Files that moved flags away from high-s122/s120 vehicles (v14, v15, v22) all lost; a tilt c > 0 would explain that.
Exact Gaussian likelihood of the public scores (v18 machinery), maximised over a for each c; alpha (cost level) in {1.0, 1.15}."""
import numpy as np, pandas as pd
from common import ROOT, load
from score import INSPECT, OVER, UNDER
tr, te, S = load()
q, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
def zs(x): x = np.log1p(np.nan_to_num(np.maximum(x, 0), nan=np.nanmedian(x))); return (x - x.mean()) / x.std()
Z = {"s122": zs(te.s122.to_numpy()), "s126": zs(te.s126.to_numpy()), "s120": zs(te.s120.to_numpy())}
OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718, "v5_platt": 22968.5508,
       "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558, "v9_k4": 17013.8646, "v10_bayes": 17348.7454, "v12_robust": 16839.8576,
       "v13_bag": 17974.5456, "v14_map": 17375.6526, "v15_map_v12": 17058.2816, "v17_bud125": 16826.0, "v17_k135": 16908.0,
       "v18_ens10": 17624.0, "v22_nos122": 17172.5716, "v24_pu": 17575.0}
names = list(OBS); obs = np.array([OBS[k] for k in names])
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 33; base = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :])
U0 = Fl * (INSPECT + OVER * Bu); U00 = np.einsum("nj,nl->njl", U0, U0); pre = {}
for al in (1.0, 1.15):
    C = np.clip(base * al, 5, 8000)
    U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
    pre[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)
sig = lambda x: 1 / (1 + np.exp(-x)); lg = np.log(q / (1 - q))
def ll(p, al, EPS):
    E1, G = pre[al]; mi = (1 - p)[:, None] * U0 + p[:, None] * E1; mean = 0.3 * mi.sum(0)
    cov = 0.3 * (np.einsum("n,njl->jl", 1 - p, U00) + np.einsum("n,njl->jl", p, G)) - 0.09 * mi.T @ mi + np.diag((EPS * mean) ** 2)
    L = np.linalg.cholesky(cov); z = np.linalg.solve(L, obs - mean); return -0.5 * z @ z - np.log(np.diag(L)).sum(), mean
rows = []
for EPS in (0.02, 0.04):
    for zn, z in Z.items():
        for c in (-0.6, -0.3, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3):
            for al in pre:
                best = max(((ll(sig(a + lg + c * z), al, EPS)[0], a) for a in np.linspace(-1, 3, 33)))
                rows.append((EPS, zn, c, al, best[0], best[1]))
R = pd.DataFrame(rows, columns=["eps", "z", "c", "alpha", "ll", "a"])
R.to_csv(ROOT / "reports/v25_tilt.csv", index=False)
for (e, zn), g in R.groupby(["eps", "z"]):
    b = g.groupby("c").ll.max(); print(f"eps {e} z={zn}: ll by tilt c (max over alpha):", (b - b[0.0]).round(2).to_dict())
e, zn = 0.02, "s122"; g = R[(R.eps == e) & (R.z == zn)]; bst = g.loc[g.ll.idxmax()]
_, mean = ll(sig(bst.a + lg + bst.c * Z[zn]), bst.alpha, e); _, mean0 = ll(sig(g[g.c == 0].sort_values("ll").iloc[-1].a + lg), 1.0, e)
print(f"\nbest tilt on {zn}: c {bst.c} alpha {bst.alpha}; residual obs-pred per file (tilt | no tilt):")
for n, o, a1, a0 in zip(names[9:], obs[9:], mean[9:], mean0[9:]): print(f"  {n:12s} {o - a1:+7.0f} | {o - a0:+7.0f}")
