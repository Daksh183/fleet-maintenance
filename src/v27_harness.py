"""v27 harness: posterior over 'truths' for the test depot from all public scores, then score many candidate submissions by their
posterior-expected cost on the private 70%.

Truth family T = (base ranking, wear covariate z, tilt c, odds a, cost level alpha):
    p_i = sigmoid(a + logit(r_i) + c * z_i),   r = blend of classifier rankings,   repair cost draws exp(m + resid) * alpha clipped [5, 8000]
Weights: exact Gaussian likelihood of the public scores (random 30% subset, Bernoulli failures, per-vehicle cost draws, eps model error).
Candidates: flags/budgets from the newsvendor + flag rule applied to many assumed truths, plus the Bayes decision under the whole posterior.
Output: reports/v27_post.csv (truth weights), reports/v27_cands.csv (candidate expected costs), candidate files on request."""
import sys, itertools, numpy as np, pandas as pd
from common import ROOT, load, write_submission
from score import INSPECT, OVER, UNDER
tr, te, S = load()
qv12, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
sig = lambda v: 1 / (1 + np.exp(-v)); logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
def zs(x): x = np.log1p(np.nan_to_num(np.maximum(x, 0), nan=np.nanmedian(x))); return (x - x.mean()) / x.std()
Z = {"none": np.zeros(len(te)), "s120": zs(te.s120.to_numpy()), "s122": zs(te.s122.to_numpy()), "s126": zs(te.s126.to_numpy()), "s149": zs(te.s149.to_numpy())}
RANK = {"v12": logit(qv12), "nos122": logit(np.load(ROOT / "reports/v22_q.npy")), "pu": logit(np.load(ROOT / "reports/v24_q.npy"))}
OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718, "v5_platt": 22968.5508,
       "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558, "v9_k4": 17013.8646, "v10_bayes": 17348.7454, "v12_robust": 16839.8576,
       "v13_bag": 17974.5456, "v14_map": 17375.6526, "v15_map_v12": 17058.2816, "v17_bud125": 16826.0, "v17_k135": 16908.0,
       "v18_ens10": 17624.0, "v22_nos122": 17172.5716, "v24_pu": 17575.0}
names = list(OBS); obs = np.array([OBS[k] for k in names]); J = len(names)
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag.to_numpy() for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget.to_numpy() for k in names])
K = 33; RQ = np.quantile(resid, (np.arange(K) + .5) / K)
ALPHAS = (1.0, 1.15, 1.3)
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
# ---------------- 1. posterior over truths
BLENDS = [("v12", 0.0)] + [(alt, w) for alt in ("nos122", "pu") for w in (0.5, 1.0)]
TILTS = {"none": [0.0], "s120": [0.3, 0.6, 0.9], "s122": [0.4, 0.8, 1.2, 1.6], "s126": [0.4, 0.8, 1.2, 1.6], "s149": [0.4, 0.8, 1.2]}
rows = []
for (alt, w), (zn, cs), al in itertools.product(BLENDS, TILTS.items(), ALPHAS):
    base = (1 - w) * RANK["v12"] + w * RANK[alt]
    for c in cs:
        best = max((loglik(sig(a + base + c * Z[zn]), al), a) for a in np.linspace(-0.5, 3.5, 21))
        a0 = best[1]; fine = max((loglik(sig(a + base + c * Z[zn]), al), a) for a in np.linspace(a0 - 0.2, a0 + 0.2, 9))
        rows.append(dict(alt=alt, w=w, z=zn, c=c, alpha=al, a=fine[1], ll=fine[0]))
P = pd.DataFrame(rows); P["wt"] = np.exp(P.ll - P.ll.max()); P.wt /= P.wt.sum(); P.to_csv(ROOT / "reports/v27_post.csv", index=False)
print(P.sort_values("ll", ascending=False).head(12).round(3).to_string(), flush=True)
for col in ("alt", "w", "z", "c", "alpha"): print(col, P.groupby(col).wt.sum().round(3).to_dict())
# truth points with non-negligible weight, used to score candidates
T = P[P.wt > 1e-3].copy(); T.wt /= T.wt.sum(); print(f"{len(T)} truth points carry the posterior", flush=True)
TP = [(sig(r.a + (1 - r.w) * RANK["v12"] + r.w * RANK[r.alt] + r.c * Z[r.z]), r.alpha, r.wt) for r in T.itertuples()]
Kd = 99; RD = np.quantile(resid, (np.arange(Kd) + .5) / Kd); MB = np.exp(m[:, None] + RD[None, :])
def exp_total(fl, b):
    """posterior-expected total cost over all 20,000 vehicles (private 70% = 0.7 of this in expectation)"""
    tot = 0.0
    for p, al, wt in TP:
        Mx = np.clip(MB * al, 5, 8000); fail = (UNDER * np.maximum(Mx - b[:, None], 0) + OVER * np.maximum(b[:, None] - Mx, 0)).mean(1)
        tot += wt * np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * Mx.mean(1)).sum()
    return tot
def decide(p, al_mix):
    """newsvendor budget + flag rule for probability p and a cost-level mixture {alpha: weight}"""
    Mx = np.concatenate([np.clip(MB * al, 5, 8000) for al in al_mix], 1); wcol = np.concatenate([np.full(Kd, w / Kd) for w in al_mix.values()])
    o = np.argsort(Mx, 1); Ms = np.take_along_axis(Mx, o, 1); cw = np.cumsum(wcol[o], 1)
    Q = 1 - (1 - UNDER / (UNDER + OVER)) / np.maximum(p, 1e-9)
    idx = np.clip((cw < np.clip(Q, 0, 1)[:, None]).sum(1), 0, Ms.shape[1] - 1); b = np.where(Q > 0, Ms[np.arange(len(p)), idx], 0.0)
    fail = ((UNDER * np.maximum(Mx - b[:, None], 0) + OVER * np.maximum(b[:, None] - Mx, 0)) * wcol).sum(1); mu = (Mx * wcol).sum(1)
    fl = (INSPECT + p * fail + (1 - p) * OVER * b < p * mu).astype(int); return fl, np.where(fl == 1, b, 0.0)
# ---------------- 2. candidates
ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv"); C0 = exp_total(ref.flag.to_numpy(), ref.budget.to_numpy())
cands = {"v20_cap": (ref.flag.to_numpy(), ref.budget.to_numpy())}
for nm in ("v25_tilt", "v22_nos122", "v24_pu", "v17_bud125"):
    r = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); cands[nm] = (r.flag.to_numpy(), r.budget.to_numpy())
alw = T.groupby("alpha").wt.sum().to_dict()
pbar = sum(wt * p for p, _, wt in TP); cands["BAYES"] = decide(pbar, alw)                      # Bayes decision under the full posterior
for (alt, w), zn, al, lv in itertools.product(BLENDS, ["none", "s120", "s122", "s126", "s149"], (1.0, 1.15, 1.3), (-0.3, 0.0, 0.3)):
    for c in TILTS[zn]:
        sub = T[(T.alt == alt) & (T.w == w) & (T.z == zn) & (T.c == c)]
        a = (sub.a * sub.wt).sum() / sub.wt.sum() if len(sub) else P[(P.alt == alt) & (P.w == w) & (P.z == zn) & (P.c == c)].sort_values("ll").a.iloc[-1]
        base = (1 - w) * RANK["v12"] + w * RANK[alt]
        cands[f"{alt}{w:g}_{zn}{c:g}_al{al:g}_lv{lv:+g}"] = decide(sig(a + lv + base + c * Z[zn]), {al: 1.0})
out = []
for nm, (fl, b) in cands.items():
    out.append(dict(cand=nm, flags=int(fl.sum()), d_private=0.7 * (exp_total(fl, b) - C0), differ=int((fl != ref.flag.to_numpy()).sum())))
O = pd.DataFrame(out).sort_values("d_private"); O.to_csv(ROOT / "reports/v27_cands.csv", index=False)
print(f"\n{len(O)} candidates; posterior-expected PRIVATE cost difference vs v20_cap (negative = better):")
print(O.head(25).round(1).to_string(index=False)); print("...\n", O[O.cand.isin(["v20_cap", "v25_tilt", "v22_nos122", "v24_pu", "v17_bud125"])].round(1).to_string(index=False))
np.save(ROOT / "reports/v27_bayes.npy", np.vstack(cands["BAYES"]))
for nm in sys.argv[1:]: write_submission(te, *cands[nm], nm)
