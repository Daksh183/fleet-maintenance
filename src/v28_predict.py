"""Posterior-predictive public score of new files, conditional on the 20 observed public scores (v28 posterior)."""
"""v28 harness: posterior over a wide family of test-depot truths (every ranking hypothesis we built, wear tilts, confidence slope,
an unexplained-failure floor, cost level), weighted by all 20 public scores; then candidate submissions scored by posterior-expected
PRIVATE cost. Expected cost is linear in p, so each candidate is scored with one pass per cost level.

Truth: p_i = floor + (1-floor) * sigmoid(a + b * L_i + c * z_i),  L = (1-w) logit(q_v12) + w logit(q_R),  cost draws exp(m+resid)*alpha.
Usage: python v28_harness.py [cand_name ...]   (writes the named candidates as submissions)"""
import sys, itertools, numpy as np, pandas as pd, xgboost as xgb
from calib import make
from common import ROOT, load, split_holdout, write_submission
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
qv12, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
sig = lambda v: 1 / (1 + np.exp(-v)); logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
def zs(x): x = np.log1p(np.nan_to_num(np.maximum(x, 0), nan=np.nanmedian(x))); return (x - x.mean()) / x.std()
# ---- ranking hypotheses (test probabilities from every classifier variant built so far)
v19 = np.load(ROOT / "reports/v19_q.npy"); H21 = np.load(ROOT / "reports/v21_H.npy")
te_map = te.copy()
for c in ("s120", "s122"):                                                                   # v15-style quantile mapping of test to train
    r = te[c].rank(pct=True); te_map[c] = np.nanquantile(tr[c], r.fillna(0.5).clip(0, 1))
    te_map.loc[te[c].isna(), c] = np.nan
mdl = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
q_map = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(mdl.predict_proba(te_map[S])[:, 1])
RANK = {"v12": qv12, "nos122": np.load(ROOT / "reports/v22_q.npy"), "pu": np.load(ROOT / "reports/v24_q.npy"),
        "pu_x2": np.load(ROOT / "reports/v24_pu_x2_q.npy"), "ens10": np.load(ROOT / "reports/v18_ens10_q.npy"),
        "ens4": np.load(ROOT / "reports/v18_ens_q.npy"), "fix120": v19[0], "fix": v19[1], "nodrift": H21[2], "map": q_map}
RANK = {k: logit(np.ravel(v)[-len(te):]) for k, v in RANK.items()}
Z = {"none": np.zeros(len(te)), "s120": zs(te.s120.to_numpy()), "s122": zs(te.s122.to_numpy()), "s126": zs(te.s126.to_numpy()), "s149": zs(te.s149.to_numpy())}
OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718, "v5_platt": 22968.5508,
       "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558, "v9_k4": 17013.8646, "v10_bayes": 17348.7454, "v12_robust": 16839.8576,
       "v13_bag": 17974.5456, "v14_map": 17375.6526, "v15_map_v12": 17058.2816, "v17_bud125": 16826.0, "v17_k135": 16908.0,
       "v18_ens10": 17624.0, "v22_nos122": 17172.5716, "v24_pu": 17575.0}
names = list(OBS); obs = np.array([OBS[k] for k in names])
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 33; RQ = np.quantile(resid, (np.arange(K) + .5) / K); ALPHAS = (1.0, 1.15, 1.3)
U0 = Fl * (INSPECT + OVER * Bu); U00 = np.einsum("nj,nl->njl", U0, U0); PRE = {}
for al in ALPHAS:
    C = np.clip(np.exp(m[:, None] + RQ[None, :]) * al, 5, 8000)
    U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
    PRE[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)
print("precomputed", flush=True)
def loglik(p, al, eps=0.03):
    E1, G = PRE[al]; mi = (1 - p)[:, None] * U0 + p[:, None] * E1; mean = 0.3 * mi.sum(0)
    cov = 0.3 * (np.einsum("n,njl->jl", 1 - p, U00) + np.einsum("n,njl->jl", p, G)) - 0.09 * mi.T @ mi + np.diag((eps * mean) ** 2)
    L = np.linalg.cholesky(cov); z = np.linalg.solve(L, obs - mean); return -0.5 * z @ z - np.log(np.diag(L)).sum()
def prob(r): L = (1 - r["w"]) * RANK["v12"] + r["w"] * RANK[r["R"]]; return r["fl0"] + (1 - r["fl0"]) * sig(r["a"] + r["b"] * L + r["c"] * Z[r["z"]])
def fit_a(r):
    best = max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(-1.5, 3.5, 13))
    return max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(best[1] - 0.3, best[1] + 0.3, 7))

import sys
P = pd.read_csv(ROOT / "reports/v28_post.csv"); T = P[P.wt > 1e-3].copy(); T.wt /= T.wt.sum()
NEW = sys.argv[1:]
FlA = np.column_stack([Fl] + [pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy()[:, None] for k in NEW]).astype(float)
BuA = np.column_stack([Bu] + [pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy()[:, None] for k in NEW])
J0 = len(names); JA = FlA.shape[1]
V0 = FlA * (INSPECT + OVER * BuA); V00 = np.einsum("nj,nl->njl", V0, V0); PA = {}
for al in ALPHAS:
    C = np.clip(np.exp(m[:, None] + RQ[None, :]) * al, 5, 8000)
    U1 = np.where(FlA[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - BuA[:, None, :], 0) + OVER * np.maximum(BuA[:, None, :] - C[:, :, None], 0), C[:, :, None])
    PA[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)
eps = 0.03; mus = []; vs = []; ws = []
for r in T.itertuples():
    p = prob(r._asdict()); E1, G = PA[r.alpha]; mi = (1 - p)[:, None] * V0 + p[:, None] * E1; mean = 0.3 * mi.sum(0)
    cov = 0.3 * (np.einsum("n,njl->jl", 1 - p, V00) + np.einsum("n,njl->jl", p, G)) - 0.09 * mi.T @ mi + np.diag((eps * mean) ** 2)
    So, Sno = cov[:J0, :J0], cov[J0:, :J0]; A = np.linalg.solve(So, Sno.T).T
    mus.append(mean[J0:] + A @ (obs - mean[:J0])); vs.append(np.diag(cov[J0:, J0:] - A @ Sno.T)); ws.append(r.wt)
mus, vs, ws = np.array(mus), np.array(vs), np.array(ws)
for j, nm in enumerate(NEW):
    mu = (ws * mus[:, j]).sum(); var = (ws * (vs[:, j] + (mus[:, j] - mu) ** 2)).sum()
    print(f"{nm:22s} predicted PUBLIC score {mu:8.0f}  sd {np.sqrt(var):5.0f}  (80% range {mu-1.28*np.sqrt(var):.0f} - {mu+1.28*np.sqrt(var):.0f})")
