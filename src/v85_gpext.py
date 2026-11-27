"""v85: GP repair-cost refinements never tested before, on the v51 inputs (A and B averaged), against v51 with the same honest
cross-fitted budget loss as v54 (flagged-test level mix, fold seeds 501-508, paired per seed).
  ls       lengthscale multiplier per input group fitted with the amplitudes (v33 fixes it at the median pairwise distance)
  lin      extra linear kernel on the wear and sensor z-scores (a linear mean with a Gaussian prior)
  pvar     budgets from the GP's own predictive sd per vehicle: residual quantiles of r/sd, times each vehicle's sd
  pvar+bands  the same within the three predicted-cost bands"""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
import v47_ablation as A, v54_gpsettings as V
FS = V.FS
class GPX:
    def __init__(self, ls=False, lin=False, it=35): self.ls, self.lin, self.it = ls, lin, it
    def _D(self, X, Y): return [cdist(x, y, "sqeuclidean") for x, y in zip(X, Y)]
    def _K(self, D, m):
        out = [np.nan_to_num(np.exp(-d / (2 * s * np.exp(mm))), nan=0.) for d, s, mm in zip(D, self.scale, m)]
        return out
    def _lin(self, X, Y): return np.nan_to_num(X) @ np.nan_to_num(Y).T / X.shape[1]
    def fit(self, features, y):
        self.training = features; n = len(y); ind = np.triu_indices(n, 1); self.scale = []
        D = self._D(features, features)
        for d in D: ds = d[ind]; ds = ds[np.isfinite(ds)]; self.scale.append(max(np.median(ds) if len(ds) else 1., 1e-6))
        G = len(features); self.LX = np.column_stack([features[-2], features[-1]]) if self.lin else None
        KL = self._lin(self.LX, self.LX) if self.lin else None
        self.center = y.mean(); z = y - self.center; I = np.eye(n); nb = G + (1 if self.lin else 0)
        def unpack(p): return np.exp(p[:nb]), np.exp(p[nb]), (p[nb + 1:] if self.ls else np.zeros(G))
        def value(p, final=False):
            th, noise, m = unpack(p); K = self._K(D, m); blocks = K + ([KL] if self.lin else [])
            cov = sum(t * k for t, k in zip(th, blocks)) + noise * I; fac = cho_factor(cov, lower=True, check_finite=False); al = cho_solve(fac, z, check_finite=False)
            if final: return al, fac
            L = .5 * z @ al + np.log(np.diag(fac[0])).sum() + .5 * (p @ p) / 4
            inv = cho_solve(fac, I, check_finite=False) - np.outer(al, al)
            g = [.5 * t * np.einsum("ij,ij->", inv, k) for t, k in zip(th, blocks)] + [.5 * noise * np.trace(inv)]
            if self.ls: g += [.5 * th[i] * np.einsum("ij,ij->", inv, K[i] * np.nan_to_num(D[i]) / (2 * self.scale[i] * np.exp(m[i]))) for i in range(G)]
            return L, np.array(g) + p / 4
        k0 = nb + 1 + (G if self.ls else 0)
        bnd = [(-5., 5.)] * nb + [(-4., 3.)] + ([(-2.5, 2.5)] * G if self.ls else [])
        res = minimize(value, np.zeros(k0), jac=True, method="L-BFGS-B", bounds=bnd, options={"maxiter": self.it if not self.ls else 100})
        self.p = res.x; self.th, self.noise, self.m = unpack(res.x); self.alpha, self.fac = value(res.x, True); self.bounds = (y.min(), y.max()); return self
    def predict(self, features, var=False):
        K = self._K(self._D(features, self.training), self.m) + ([self._lin(np.column_stack([features[-2], features[-1]]), self.LX)] if self.lin else [])
        Ks = sum(t * k for t, k in zip(self.th, K)); mu = np.clip(Ks @ self.alpha + self.center, *self.bounds)
        if not var: return mu
        prior = sum(self.th[:len(features)]) + (self.th[-1] * (np.nan_to_num(np.column_stack([features[-2], features[-1]])) ** 2).mean(1) if self.lin else 0)
        v = prior - np.einsum("ij,ji->i", Ks, cho_solve(self.fac, Ks.T, check_finite=False)) + self.noise
        return mu, np.sqrt(np.maximum(v, 1e-4))
def oof(kw, s):
    o, sd = [], []
    for F in FS:
        p = np.full(len(A.y), np.nan); q = np.full(len(A.y), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=s).split(A.y): p[b], q[b] = GPX(**kw).fit(A.select(F, a), A.y[a]).predict(A.select(F, b), var=True)
        o.append(p); sd.append(q)
    return (o[0] + o[1]) / 2, (sd[0] + sd[1]) / 2
def loss_pv(o, sd, s, bands):
    mu, cc, t, sg = o[A.valid], A.c[A.valid], A.y[A.valid], sd[A.valid]; r = (t - mu) / sg; fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s + 1000).split(mu)): fold[b] = j
    nb = 3 if bands else 1; band = np.searchsorted(np.quantile(mu, np.linspace(0, 1, nb + 1)[1:-1]), mu); Q = np.zeros((len(A.lev), len(mu)))
    for j in range(5):
        for k in range(nb):
            src = (fold != j) & (band == k); idx = (fold == j) & (band == k)
            if idx.any(): Q[:, idx] = np.quantile(r[src], A.lev)[:, None] * sg[idx][None, :]
    B = np.minimum(np.exp(mu[None, :] + Q), 8000)
    return (A.p[:, None] * (A.UNDER * np.maximum(cc[None, :] - B, 0) + A.OVER * np.maximum(B - cc[None, :], 0))).sum(0)
def run(s):
    out = {"base": V.loss(V.oof({}, s), s)}
    for nm, kw in {"gpx (same model)": {}, "ls": dict(ls=True), "lin": dict(lin=True), "ls+lin": dict(ls=True, lin=True)}.items():
        o, sd = oof(kw, s); out[nm] = V.loss(o, s); out[nm + " | pvar"] = loss_pv(o, sd, s, False); out[nm + " | pvar+bands"] = loss_pv(o, sd, s, True)
        if nm == "gpx (same model)": out["_r2"] = o
    return out
if __name__ == "__main__":
    seeds = list(range(501, 509)); R = Parallel(n_jobs=4)(delayed(run)(s) for s in seeds); rows = []
    for n in R[0]:
        if n.startswith("_"): continue
        d = np.array([100 * (r[n].sum() / r["base"].sum() - 1) for r in R]); L = np.mean([r[n] for r in R], 0); L0 = np.mean([r["base"] for r in R], 0)
        rng = np.random.default_rng(0); i = rng.integers(0, len(L), (4000, len(L))); boot = 100 * (L - L0)[i].sum(1) / L0[i].sum(1)
        rows.append(dict(variant=n, mean=d.mean(), seeds_better=int((d < 0).sum()), boot_sd=boot.std(), p_better=(boot < 0).mean()))
    O = pd.DataFrame(rows).sort_values("mean"); O.to_csv(A.ROOT / "reports/v85_gpext.csv", index=False); print(O.round(2).to_string(index=False))
