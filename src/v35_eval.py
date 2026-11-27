"""v35: harness-expected score of candidate files under tree, GP and mixed repair-cost truths."""
"""v32: v30 harness plus Bayes decisions that respect the shift-aware CV (rankings including nos122)."""
"""v30 harness (round 3): v29 truths re-weighted with the 21st public score (v29_bayes 17,196); a re-fitted locally.
v29 harness (round 2): v28 plus 3-way ranking blends, two-sensor wear tilts, finer tilt/slope/cost grids; posterior over the
union of round-1 and round-2 truths.
v28 harness: posterior over a wide family of test-depot truths (every ranking hypothesis we built, wear tilts, confidence slope,
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
       "v18_ens10": 17624.0, "v22_nos122": 17172.5716, "v24_pu": 17575.0, "v29_bayes": 17196.0}
names = list(OBS); obs = np.array([OBS[k] for k in names])
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 33; RQ = np.quantile(resid, (np.arange(K) + .5) / K); ALPHAS = (1.0, 1.15, 1.3, 1.45)
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
def prob(r):
    g = lambda k, d: r[k] if k in r and isinstance(r[k], (int, float, str)) and r[k] == r[k] else d
    w2, R2, c2, z2 = g("w2", 0.0), g("R2", "v12"), g("c2", 0.0), g("z2", "none")
    L = (1 - r["w"] - w2) * RANK["v12"] + r["w"] * RANK[r["R"]] + w2 * RANK[R2]
    return r["fl0"] + (1 - r["fl0"]) * sig(r["a"] + r["b"] * L + r["c"] * Z[r["z"]] + c2 * Z[z2])
def fit_a(r):
    best = max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(-1.5, 3.5, 13))
    return max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(best[1] - 0.3, best[1] + 0.3, 7))
# ---- 1. posterior
TILT = [("none", 0.0)] + [(z, c) for z in ("s120", "s122", "s126", "s149") for c in (0.4, 0.8, 1.2)]
GRID = [dict(R="v12", w=0.0)] + [dict(R=R, w=w) for R in RANK if R != "v12" for w in (0.5, 1.0)]

z = np.load(ROOT / "reports/v35_gp.npz"); sev, gres = z["sev"], z["res"]; sup = np.isfinite(sev)
P = pd.read_csv(ROOT / "reports/v30_post.csv"); T = P[P.wt > 1e-4].copy(); T.wt /= T.wt.sum()
PB = {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}; W = {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
Kd = 99; RDt = np.quantile(resid, (np.arange(Kd) + .5) / Kd); RDg = np.quantile(gres, (np.arange(Kd) + .5) / Kd)
for truth in ("tree", "gp", "mix"):
    base = np.exp(m[:, None] + RDt[None, :])
    if truth != "tree":
        g = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + RDg[None, :]), base)
        base = g if truth == "gp" else np.concatenate([base, g], 1)
    MXs = {al: np.clip(base * al, 5, 8000) for al in ALPHAS}
    def tot(fl, b):
        t = 0.0
        for al in ALPHAS:
            if W[al] == 0: continue
            Mx = MXs[al]; fail = (UNDER * np.maximum(Mx - b[:, None], 0) + OVER * np.maximum(b[:, None] - Mx, 0)).mean(1)
            t += np.where(fl == 1, W[al] * (INSPECT + OVER * b) + PB[al] * (fail - OVER * b), PB[al] * Mx.mean(1)).sum()
        return 0.3 * t
    out = {}
    for nm in ("v20_cap", "v29_bayes", "v32_bayes_nos", "v33_gp_budget", "v35_v33cap", "v35_gpflags", "v35_combo", "v34_budget"):
        d = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); out[nm] = tot(d.flag.to_numpy(), d.budget.to_numpy())
    print(f"cost truth = {truth}: expected score on a random 30% (lower is better)")
    for nm, v in sorted(out.items(), key=lambda kv: kv[1]): print(f"   {nm:16s} {v:8.0f}   vs v33 {v - out['v33_gp_budget']:+7.0f}")
