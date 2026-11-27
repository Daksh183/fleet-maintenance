"""v54: GP fitting and spread settings on the v51 inputs (A and B averaged), one change at a time against v51 C.
Same honest cross-fitted budget loss as v49 (flagged-test level mix), fold seeds 501-508, paired per seed."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import v47_ablation as A, v49_round2 as R2
FS = [R2.F(), R2.F(wear=["s126", "s149"])]
class GPS(A.GP):
    """the v33 fit with its three settings exposed: optimiser iterations, prior variance on log-parameters, amplitude bound"""
    def __init__(self, it=35, pv=4.0, hi=5.0, loss="gauss"): self.it, self.pv, self.hi, self.loss = it, pv, hi, loss
    def fit(self, features, y):
        self.training = features; ind = np.triu_indices(len(y), 1); self.scale = []
        for x in features:
            ds = A.cdist(x, x, "sqeuclidean")[ind]; ds = ds[np.isfinite(ds)]; self.scale.append(max(np.median(ds) if len(ds) else 1., 1e-6))
        blocks = self.blocks(features); unit = np.eye(len(y)); w = np.ones(len(y))
        for _ in range(4 if self.loss == "t" else 1):                     # "t": reweight noise of outlying repairs (Student-t, nu=4)
            self.center = np.average(y, weights=w); z = y - self.center
            def value(lt, final=False):
                th = np.exp(lt); cov = np.einsum("i,ijk->jk", th[:-1], blocks) + th[-1] * np.diag(1 / w); fac = cho_factor(cov, lower=True, check_finite=False)
                al = cho_solve(fac, z, check_finite=False)
                if final: return al
                pv = np.r_[np.full(len(blocks), self.pv[0]), self.pv[1]] if isinstance(self.pv, tuple) else self.pv   # (amplitudes, noise)
                L = .5 * z @ al + np.log(np.diag(fac[0])).sum() + .5 * np.sum(lt ** 2 / pv)
                inv = cho_solve(fac, unit, check_finite=False) - np.outer(al, al)
                return L, .5 * th * np.r_[np.einsum("ij,kij->k", inv, blocks), np.sum(np.diag(inv) / w)] + lt / pv
            res = minimize(value, np.zeros(len(blocks) + 1), jac=True, method="L-BFGS-B", bounds=[(-5., self.hi)] * len(blocks) + [(-4., 3.)], options={"maxiter": self.it})
            self.amplitude = np.exp(res.x[:-1]); self.alpha = value(res.x, True); noise = np.exp(res.x[-1])
            r = noise * self.alpha / w; w = np.clip(5.0 / (4.0 + r ** 2 / noise), 0.05, 1.25)
        self.bounds = (min(y), max(y)); return self
# name: (GP settings, spread settings)
FIT = {"base": {}, "iter 200": dict(it=200), "iter 10": dict(it=10), "prior var 1": dict(pv=1.0), "prior var 16": dict(pv=16.0),
       "prior var 1e6": dict(pv=1e6), "amp bound 8": dict(hi=8.0), "student-t noise": dict(loss="t")}
SPREAD = {"bands 2": dict(nb=2), "bands 4": dict(nb=4), "bands 5": dict(nb=5), "resid x0.9": dict(sc=0.9), "resid x1.1": dict(sc=1.1),
          "overlapping bands (kernel)": dict(smooth=True), "centre on median resid": dict(med=True)}
def oof(kw, s):
    o = []
    for F in FS:
        p = np.full(len(A.y), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=s).split(A.y): p[b] = GPS(**kw).fit(A.select(F, a), A.y[a]).predict(A.select(F, b))
        o.append(p)
    return (o[0] + o[1]) / 2
def loss(o, s, nb=3, sc=1.0, smooth=False, med=False):
    mu, cc, t = o[A.valid], A.c[A.valid], A.y[A.valid]; r = t - mu; fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s + 1000).split(mu)): fold[b] = j
    band = np.searchsorted(np.quantile(mu, np.linspace(0, 1, nb + 1)[1:-1]), mu); Q = np.zeros((len(A.lev), len(mu)))
    rank = np.argsort(np.argsort(mu)) / len(mu)
    for j in range(5):
        if smooth:                      # residual quantiles from a Gaussian window over predicted-cost rank
            src = fold != j; order = np.argsort(r[src]); rs = r[src][order]; rk = rank[src][order]
            for i in np.where(fold == j)[0]:
                w = np.exp(-0.5 * ((rk - rank[i]) / 0.2) ** 2); cw = np.cumsum(w) / w.sum()
                Q[:, i] = rs[np.minimum(np.searchsorted(cw, A.lev), len(rs) - 1)]
        else:
            for k in range(nb):
                src = (fold != j) & (band == k); idx = (fold == j) & (band == k)
                if idx.any(): Q[:, idx] = np.quantile(r[src], A.lev)[:, None]
    if med:
        md = np.median(r); Q = md + sc * (Q - md)
    else: Q = sc * Q
    B = np.minimum(np.exp(mu[None, :] + Q), 8000)
    return (A.p[:, None] * (A.UNDER * np.maximum(cc[None, :] - B, 0) + A.OVER * np.maximum(B - cc[None, :], 0))).sum(0)
def run(s):
    out = {}; base = None
    for n, kw in FIT.items():
        o = oof(kw, s); out[n] = loss(o, s)
        if n == "base": base = o
    for n, kw in SPREAD.items(): out[n] = loss(base, s, **kw)
    return out
if __name__ == "__main__":
    seeds = list(range(501, 509)); R = Parallel(n_jobs=4)(delayed(run)(s) for s in seeds); rows = []
    for n in R[0]:
        d = np.array([100 * (r[n].sum() / r["base"].sum() - 1) for r in R]); L = np.mean([r[n] for r in R], 0); L0 = np.mean([r["base"] for r in R], 0)
        rng = np.random.default_rng(0); i = rng.integers(0, len(L), (4000, len(L))); boot = 100 * (L - L0)[i].sum(1) / L0[i].sum(1)
        rows.append(dict(variant=n, mean=d.mean(), seeds_better=int((d < 0).sum()), boot_sd=boot.std(), p_better=(boot < 0).mean()))
    O = pd.DataFrame(rows).sort_values("mean"); O.to_csv(A.ROOT / "reports/v54_gpsettings.csv", index=False); print(O.round(2).to_string(index=False))
