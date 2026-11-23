"""v10: posterior over the depot-shift odds factor k from ALL observed public scores, and the next uploads.

Generative model of the public score: failures ~ Bernoulli(shift(q_i, k)), repair cost ~ v9 per-vehicle cost
distribution, public set = random 30% of test. For each k we simulate the vector of public scores of every
uploaded file (they share the same subset and failures, so they are strongly correlated), fit a Gaussian
(mean, covariance) and evaluate the observed vector. A relative model-error term (eps) is added to the
covariance: our probabilities / cost model are not exact.
"""
import sys
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold, cross_val_predict

from calib import make
from common import ROOT, SEED, load, split_holdout
from decide import decide_pv
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xa, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
K = 99
M = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :]); mu = M.mean(1)
n = len(te)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))

OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718,
       "v5_platt": 22968.5508, "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558}
files = {k: pd.read_csv(ROOT / f"submissions/submission_{k}.csv") for k in OBS}
CAND_K = [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10]
cand = {f"c{k}": decide_pv(shift(q, k), m, resid) for k in CAND_K}
names = list(OBS) + list(cand)
Fm = np.column_stack([files[k].flag.to_numpy() for k in OBS] + [cand[k][0] for k in cand])
Bm = np.column_stack([files[k].budget.to_numpy() for k in OBS] + [cand[k][1] for k in cand])
no = len(OBS); obs = np.array(list(OBS.values()))


def exp_cost_full(fl, b, p):                       # exact expected cost per vehicle on the WHOLE test
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return float(np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).mean())


rng = np.random.default_rng(SEED)
GRID = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12]
mean, cov = {}, {}
for k in GRID:
    p = shift(q, k); T = np.zeros((R, len(names)))
    for r in range(R):
        fail = rng.random(n) < p
        c = np.where(fail, M[np.arange(n), rng.integers(0, K, n)], 0.0)
        pub = rng.random(n) < 0.3
        cc = c[pub][:, None]; Fp, Bp = Fm[pub], Bm[pub]
        T[r] = np.where(Fp == 1, INSPECT + UNDER * np.maximum(cc - Bp, 0) + OVER * np.maximum(Bp - cc, 0), cc).sum(0)
    mean[k], cov[k] = T.mean(0), np.cov(T.T)
    print(f"k={k}: simulated mean public scores of the uploads:", np.round(mean[k][:no], -2).astype(int).tolist(), flush=True)


def posterior(eps):
    ll = {}
    for k in GRID:
        C = cov[k][:no, :no] + np.diag((eps * mean[k][:no]) ** 2)
        ll[k] = multivariate_normal(mean=mean[k][:no], cov=C, allow_singular=True).logpdf(obs)
    L = np.array([ll[k] for k in GRID]); w = np.exp(L - L.max()); return w / w.sum(), L


print(f"\nobserved: {dict((k, round(v)) for k, v in OBS.items())}")
post = {}
for eps in (0.03, 0.06, 0.10, 0.15):
    w, L = posterior(eps); post[eps] = w
    mk = float((w * np.array(GRID)).sum()); cdf = np.cumsum(w)
    lo, hi = GRID[int(np.searchsorted(cdf, .05))], GRID[min(int(np.searchsorted(cdf, .95)), len(GRID) - 1)]
    print(f"eps {eps:.2f}: posterior mean k {mk:.2f}  90% [{lo}, {hi}]  mode {GRID[int(w.argmax())]}   max loglik {L.max():.1f}")
w = post[0.06]
print("\nposterior over k (eps 0.06):", {k: round(float(x), 3) for k, x in zip(GRID, w)})
kb = GRID[int(w.argmax())]; zres = (obs - mean[kb][:no]) / np.sqrt(np.diag(cov[kb][:no, :no]) + (0.06 * mean[kb][:no]) ** 2)
print(f"standardised residuals at k={kb}:", dict(zip(OBS, np.round(zres, 2))))

# ---- which candidate files are best to upload?  (a) expected full-test cost under the posterior, (b) information
W = np.array(post[0.06])
exp_full = {}
for nm, (fl, b) in cand.items():
    exp_full[nm] = sum(wk * exp_cost_full(fl, b, shift(q, k)) for wk, k in zip(W, GRID))
pbar = sum(wk * shift(q, k) for wk, k in zip(W, GRID))
bay = decide_pv(pbar, m, resid); cand["bayes"] = bay
exp_full["bayes"] = sum(wk * exp_cost_full(*bay, shift(q, k)) for wk, k in zip(W, GRID))
print("\nexpected cost per vehicle on the full test under the posterior:")
for nm, v in sorted(exp_full.items(), key=lambda t: t[1]): print(f"  {nm:8s} {v:.4f}   flag rate {cand[nm][0].mean():.3%}")
np.save(ROOT / "reports/v10_posterior.npy", np.vstack([GRID, W]))

# discrimination: conditional on the observed scores, how different is the predicted score of a candidate across k?
print("\ninformation of each candidate: spread of its conditional prediction across k (posterior-weighted) vs conditional sd")
ci = [names.index(f"c{k}") for k in CAND_K]
rows = []
for nm in [f"c{k}" for k in CAND_K]:
    j = names.index(nm); mus, sds = [], []
    for k in GRID:
        So = cov[k][:no, :no] + np.diag((0.06 * mean[k][:no]) ** 2); Sco = cov[k][j, :no]
        g = np.linalg.solve(So, obs - mean[k][:no]); mus.append(mean[k][j] + Sco @ g)
        sds.append(np.sqrt(max(cov[k][j, j] - Sco @ np.linalg.solve(So, Sco), 1.0) + (0.06 * mean[k][j]) ** 2))
    mus, sds = np.array(mus), np.array(sds); mbar = (W * mus).sum()
    spread = np.sqrt((W * (mus - mbar) ** 2).sum())
    rows.append((nm, mbar, spread, np.sqrt((W * sds ** 2).sum())))
print(pd.DataFrame(rows, columns=["file", "expected public score", "spread across k", "noise sd"]).round(0).to_string(index=False))
