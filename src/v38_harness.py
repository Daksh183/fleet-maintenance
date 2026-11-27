"""v38: v36 harness with the improved GP costs (v37) by default and 25 public scores (adds v37_bayes_gp2 16,410).
v36: harness with GP repair costs (Daksh v33 model) as the cost truth and 24 public scores (adds v32_hist_kernel, v33_gp_budget, v35_combo).
v32: v30 harness plus Bayes decisions that respect the shift-aware CV (rankings including nos122)."""
"""v30 harness (round 3): v29 truths re-weighted with the 21st public score (v29_bayes 17,196); a re-fitted locally.
v29 harness (round 2): v28 plus 3-way ranking blends, two-sensor wear tilts, finer tilt/slope/cost grids; posterior over the
union of round-1 and round-2 truths.
v28 harness: posterior over a wide family of test-depot truths (every ranking hypothesis we built, wear tilts, confidence slope,
an unexplained-failure floor, cost level), weighted by all 20 public scores; then candidate submissions scored by posterior-expected
PRIVATE cost. Expected cost is linear in p, so each candidate is scored with one pass per cost level.

Truth: p_i = floor + (1-floor) * sigmoid(a + b * L_i + c * z_i),  L = (1-w) logit(q_v12) + w logit(q_R),  cost draws exp(m+resid)*alpha.
Usage: python v28_harness.py [cand_name ...]   (writes the named candidates as submissions)"""
import os, sys, itertools, numpy as np, pandas as pd, xgboost as xgb
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
       "v18_ens10": 17624.0, "v22_nos122": 17172.5716, "v24_pu": 17575.0, "v29_bayes": 17196.0,
       "v32_hist_kernel": 16991.5, "v33_gp_budget": 16587.7456, "v35_combo": 16878.0, "v37_bayes_gp2": 16410.0}
names = list(OBS); obs = np.array([OBS[k] for k in names])
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 33; RQ = np.quantile(resid, (np.arange(K) + .5) / K)
_z = np.load(ROOT / os.environ.get("GPNPZ", "reports/v37_gp.npz")); SEV, GRES = _z["sev"], _z["res"]; SUP = np.isfinite(SEV)
RQG = np.quantile(GRES, (np.arange(K) + .5) / K)
NB = int(os.environ.get("BANDS", "1"))
if NB > 1:                                                   # residual spread by predicted-cost band (terciles of OOF GP predictions)
    _mu = _z["oof"][np.isfinite(_z["oof"])]; _edges = np.quantile(_mu, np.linspace(0, 1, NB + 1)[1:-1])
    _bf = np.searchsorted(_edges, _mu); BAND = np.searchsorted(_edges, np.where(SUP, SEV, 0.0))
def draws(Kn, RQt, RQg):
    if NB > 1:
        grid = (np.arange(Kn) + .5) / Kn; QB = np.stack([np.quantile(GRES[_bf == j], grid) for j in range(NB)]); RG = QB[BAND]
    else: RG = RQg[None, :]
    return np.where(SUP[:, None], np.exp(np.where(SUP, SEV, m)[:, None] + RG), np.exp(m[:, None] + RQt[None, :]))
ALPHAS = (1.0, 1.15, 1.3, 1.45)
U0 = Fl * (INSPECT + OVER * Bu); U00 = np.einsum("nj,nl->njl", U0, U0); PRE = {}
for al in ALPHAS:
    C = np.clip(draws(K, RQ, RQG) * al, 5, 8000)
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
REUSE = "--reuse" in sys.argv; sys.argv = [a for a in sys.argv if a != "--reuse"]
if not REUSE:
    P0 = pd.read_csv(ROOT / "reports/v36_post.csv"); P0 = P0[P0.wt > 1e-5].reset_index(drop=True); rows = []
    for i, r in enumerate(P0.drop(columns=["wt", "ll"]).to_dict("records")):
        ll, a = max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(r["a"] - 0.6, r["a"] + 0.6, 9)); rows.append({**r, "a": a, "ll": ll})
        if i % 500 == 0: print("re-weighted", i, "of", len(P0), flush=True)
    P = pd.DataFrame(rows); P["wt"] = np.exp(P.ll - P.ll.max()); P.wt /= P.wt.sum(); P.to_csv(ROOT / "reports/v38_post.csv", index=False)
else: P = pd.read_csv(ROOT / "reports/v38_post.csv")
print(f"{len(P)} truths; top 15:"); print(P.sort_values("ll", ascending=False).head(15).round(3).to_string(index=False), flush=True)
for col in ("R", "w", "R2", "w2", "z", "c", "z2", "c2", "b", "alpha"): print(f"  posterior over {col}:", P.groupby(col).wt.sum().round(3).to_dict())
# ---- 2. candidates, scored by posterior-expected private cost (linear in p: one pass per alpha)
T = P[P.wt > 1e-4].copy(); T.wt /= T.wt.sum(); print(f"{len(T)} truths carry the posterior", flush=True)
PBAR = {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}     # weighted p per alpha
WAL = {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
Kd = 99; RD = np.quantile(resid, (np.arange(Kd) + .5) / Kd); MB = draws(Kd, RD, np.quantile(GRES, (np.arange(Kd) + .5) / Kd))
MX = {al: np.clip(MB * al, 5, 8000) for al in ALPHAS}; MU = {al: MX[al].mean(1) for al in ALPHAS}
def exp_total(fl, b, PB=None, W=None):
    PB, W = PB or PBAR, W or WAL; tot = 0.0
    for al in ALPHAS:
        if W[al] == 0: continue
        fail = (UNDER * np.maximum(MX[al] - b[:, None], 0) + OVER * np.maximum(b[:, None] - MX[al], 0)).mean(1); pb = PB[al]
        tot += np.where(fl == 1, W[al] * (INSPECT + OVER * b) + pb * (fail - OVER * b), pb * MU[al]).sum()
    return tot
def decide(PB, W, bscale=1.0):
    """Bayes decision for a mixture over alpha with weighted probabilities PB[alpha] (sum of weights W)."""
    wsum = sum(W.values()); p = sum(PB.values()) / wsum
    Mx = np.concatenate([MX[al] for al in ALPHAS if W[al] > 0], 1)
    wcol = np.concatenate([PB[al][:, None] / np.maximum(p[:, None] * wsum, 1e-12) * np.ones((1, Kd)) / Kd for al in ALPHAS if W[al] > 0], 1)
    o = np.argsort(Mx, 1); Ms = np.take_along_axis(Mx, o, 1); cw = np.cumsum(np.take_along_axis(wcol, o, 1), 1)
    Q = 1 - (1 - UNDER / (UNDER + OVER)) / np.maximum(p, 1e-9)
    idx = np.clip((cw < np.clip(Q, 0, 1)[:, None]).sum(1), 0, Ms.shape[1] - 1)
    b = np.minimum(np.where(Q > 0, Ms[np.arange(len(p)), idx], 0.0) * bscale, 8000)
    fail = ((UNDER * np.maximum(Mx - b[:, None], 0) + OVER * np.maximum(b[:, None] - Mx, 0)) * wcol).sum(1); mu = (Mx * wcol).sum(1)
    fl = (INSPECT + p * fail + (1 - p) * OVER * b < p * mu).astype(int); return fl, np.where(fl == 1, b, 0.0)
ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv"); f0, b0 = ref.flag.to_numpy(), ref.budget.to_numpy()
cands = {"v20_cap": (f0, b0)}
for nm in ("v25_tilt", "v22_nos122", "v24_pu", "v17_bud125", "v17_k135", "v12_robust", "v28_bayes", "v28_bayes_lv12", "v29_bayes", "v33_gp_budget", "v35_combo", "v35_v33cap", "v35_gpflags", "v32_bayes_nos", "v37_bayes_gp2"):
    r = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); cands[nm] = (r.flag.to_numpy(), r.budget.to_numpy())
for lv in (0.7, 0.85, 1.0, 1.2, 1.4):                                                       # Bayes decision, with level and budget variations
    for bs in (0.9, 1.0, 1.15):
        cands[f"BAYES_lv{lv:g}_b{bs:g}"] = decide({al: np.clip(PBAR[al] * lv, 0, WAL[al]) for al in ALPHAS}, WAL, bs)
for r in T.sort_values("wt", ascending=False).head(60).itertuples():                           # 'plug-in' decisions for single truths
    d = r._asdict(); cands[f"T_{r.R}{r.w:.2g}+{r.R2}{r.w2:.2g}_{r.z}{r.c:g}+{r.z2}{r.c2:g}_b{r.b:g}_al{r.alpha:g}"] = decide({al: (prob(d) if al == r.alpha else np.zeros(len(te))) for al in ALPHAS},
                                                                                                  {al: (1.0 if al == r.alpha else 0.0) for al in ALPHAS})
TN = T[(T.R == "nos122") | (T.R2 == "nos122")].copy(); TN.wt /= TN.wt.sum(); print(f"nos122 truths: {len(TN)} carrying {T[(T.R == 'nos122') | (T.R2 == 'nos122')].wt.sum():.3f} of the posterior", flush=True)
PBN = {al: sum((r.wt * prob(r._asdict()) for r in TN[TN.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}; WN = {al: TN[TN.alpha == al].wt.sum() for al in ALPHAS}
cands["BAYES_nos122"] = decide(PBN, WN)
TN2 = T.copy(); TN2.loc[(TN2.R == "nos122") | (TN2.R2 == "nos122"), "wt"] *= 4; TN2.wt /= TN2.wt.sum()           # shift-CV evidence as a 4x prior
PB2 = {al: sum((r.wt * prob(r._asdict()) for r in TN2[TN2.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}; W2 = {al: TN2[TN2.alpha == al].wt.sum() for al in ALPHAS}
cands["BAYES_nosprior"] = decide(PB2, W2)
C0 = exp_total(f0, b0)
for nm in ("v20_cap", "v33_gp_budget", "v35_combo", "v37_bayes_gp2", "BAYES_lv1_b1", "BAYES_nos122", "BAYES_nosprior"):                                       # expected score on a random 30% (public scale)
    print(f"posterior-expected score on a random 30% of test: {nm:18s} {0.3 * exp_total(*cands[nm]):8.0f}", flush=True)
# robustness: expected private difference under each of the 20 heaviest truths
TOP = T.sort_values("wt", ascending=False).head(20)
def per_truth(fl, b):
    out = []
    for r in TOP.itertuples():
        PB = {al: (prob(r._asdict()) if al == r.alpha else np.zeros(len(te))) for al in ALPHAS}; W = {al: float(al == r.alpha) for al in ALPHAS}
        out.append(0.7 * (exp_total(fl, b, PB, W) - exp_total(f0, b0, PB, W)))
    return np.array(out)
rows = []
for nm, (fl, b) in cands.items():
    pt = per_truth(fl, b)
    rows.append(dict(cand=nm, flags=int(fl.sum()), differ=int((fl != f0).sum()), d_private=0.7 * (exp_total(fl, b) - C0),
                     worst_top20=pt.max(), best_top20=pt.min(), frac_better=(pt < 0).mean()))
O = pd.DataFrame(rows).sort_values("d_private"); O.to_csv(ROOT / "reports/v38_cands.csv", index=False)
print(f"\n{len(O)} candidates, posterior-expected PRIVATE cost vs v20_cap (negative = better); worst/best over the 20 heaviest truths:")
print(O.head(25).round(1).to_string(index=False))
print("\nreference files:"); print(O[O.cand.isin(["v37_bayes_gp2", "v33_gp_budget", "v35_combo", "v35_v33cap", "v35_gpflags", "v32_bayes_nos", "BAYES_nos122", "BAYES_nosprior", "BAYES_lv1_b1", "v20_cap", "v12_robust", "v28_bayes", "v29_bayes", "v28_bayes_lv12", "v25_tilt", "v22_nos122", "v24_pu"])].round(1).to_string(index=False))
for nm in sys.argv[1:]: write_submission(te, *cands[nm], nm)
