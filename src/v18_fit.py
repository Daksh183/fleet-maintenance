"""v18: what differs at the new depot? Fit a 3-parameter shift model to ALL 17 public scores.
   p_i = sigmoid(a + b*logit(q_i))   (a = log k: failure odds; b: classifier confidence at the new depot; b=1 is the v10 prior shift)
   log c | fail ~ m_i + resid + log(alpha)   (alpha: repair-cost level at the new depot)
Exact Gaussian likelihood: mean and covariance of the 17 public scores are computed analytically
(random 30% subset, Bernoulli failures, per-vehicle cost distribution), plus a relative model-error term."""
import sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; DRIFT = ["s120", "s122", "s126", "s128"]
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
np.save(ROOT / "reports/v18_qm.npy", np.vstack([q, m])); np.save(ROOT / "reports/v18_resid.npy", resid)
OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718, "v5_platt": 22968.5508,
       "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558, "v9_k4": 17013.8646, "v10_bayes": 17348.7454, "v12_robust": 16839.8576,
       "v13_bag": 17974.5456, "v14_map": 17375.6526, "v15_map_v12": 17058.2816, "v17_bud125": 16826.0, "v17_k135": 16908.0}
names = list(OBS); obs = np.array([OBS[k] for k in names]); J = len(names)
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 49; base = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :])          # n x K cost draws (alpha = 1)
ALPHA = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0]
U0 = Fl * (INSPECT + OVER * Bu)                                                                   # cost if no failure (n x J)
pre = {}
for al in ALPHA:
    C = base * al
    U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
    pre[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)                  # E[U1], E[U1 U1^T] per vehicle
    print("precomputed alpha", al, flush=True)
U00 = np.einsum("nj,nl->njl", U0, U0)
lg = np.log(q / (1 - q)); sig = lambda z: 1 / (1 + np.exp(-z))
A = np.log([1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 11]); B = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5]
rows = []
for al in ALPHA:
    E1, G = pre[al]
    for b in B:
        for a in A:
            p = sig(a + b * lg)[:, None]
            mi = (1 - p) * U0 + p * E1                                                            # n x J expected cost
            mean = 0.3 * mi.sum(0)
            cov = 0.3 * (np.einsum("n,njl->jl", 1 - p[:, 0], U00) + np.einsum("n,njl->jl", p[:, 0], G)) - 0.09 * mi.T @ mi
            cov += np.diag((EPS * mean) ** 2)
            d = obs - mean; L = np.linalg.cholesky(cov)
            z = np.linalg.solve(L, d); ll = -0.5 * z @ z - np.log(np.diag(L)).sum()
            rows.append((al, b, np.exp(a), ll, p.mean(), *mean))
R = pd.DataFrame(rows, columns=["alpha", "b", "k", "ll", "mean_p", *names]); R["w"] = np.exp(R.ll - R.ll.max()); R.w /= R.w.sum()
R.to_csv(ROOT / f"reports/v18_fit_eps{EPS}.csv", index=False)
print(f"\nEPS {EPS}: marginal posteriors")
for c in ("alpha", "b", "k"): print(c, R.groupby(c).w.sum().round(3).to_dict())
best = R.loc[R.ll.idxmax()]; b1 = R[(R.b == 1.0) & (R.alpha == 1.0)].sort_values("ll").iloc[-1]
print(f"MAP: alpha {best.alpha} b {best.b} k {best.k:.2f} ll {best.ll:.1f} | best with b=1, alpha=1: k {b1.k:.2f} ll {b1.ll:.1f}")
print(pd.DataFrame({"obs": obs, "MAP": best[names].astype(float).round(0).values, "b1a1": b1[names].astype(float).round(0).values}, index=names))
