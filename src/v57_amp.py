"""v57: amplitude prior of the GP kernels. v54/v56: a tighter prior on log-amplitudes helps; where should it be centred, and does the gain
survive at the budget level the submission actually uses (about x1.19)? Fresh seeds 801-810. Reported at level x1.0 and x1.19."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import v54_gpsettings as G, v55_confirm as C
A = G.A
class GPM(A.GP):
    """amplitude prior N(mu, v) on each log-amplitude; noise prior N(0, 4) as before; v=0 fixes amplitudes at exp(mu)"""
    def __init__(self, mu=0.0, v=4.0): self.mu, self.v = mu, v
    def fit(self, features, y):
        self.training = features; ind = np.triu_indices(len(y), 1); self.scale = []
        for x in features:
            ds = A.cdist(x, x, "sqeuclidean")[ind]; ds = ds[np.isfinite(ds)]; self.scale.append(max(np.median(ds) if len(ds) else 1., 1e-6))
        blocks = self.blocks(features); unit = np.eye(len(y)); self.center = np.mean(y); z = y - self.center; k = len(blocks)
        pm = np.r_[np.full(k, self.mu), 0.0]; pv = np.r_[np.full(k, max(self.v, 1e-8)), 4.0]
        def value(lt, final=False):
            th = np.exp(lt); fac = cho_factor(np.einsum("i,ijk->jk", th[:-1], blocks) + th[-1] * unit, lower=True, check_finite=False)
            al = cho_solve(fac, z, check_finite=False)
            if final: return al
            L = .5 * z @ al + np.log(np.diag(fac[0])).sum() + .5 * np.sum((lt - pm) ** 2 / pv)
            inv = cho_solve(fac, unit, check_finite=False) - np.outer(al, al)
            return L, .5 * th * np.r_[np.einsum("ij,kij->k", inv, blocks), np.trace(inv)] + (lt - pm) / pv
        bnd = [((self.mu, self.mu) if self.v == 0 else (-5., 5.))] * k + [(-4., 3.)]
        res = minimize(value, pm.copy(), jac=True, method="L-BFGS-B", bounds=bnd, options={"maxiter": 100})
        self.amplitude = np.exp(res.x[:-1]); self.alpha = value(res.x, True); self.noise = np.exp(res.x[-1]); self.bounds = (min(y), max(y)); return self
V = {"base": (0.0, 4.0)}
for mu in (-2.0, -1.0, 0.0, 0.7):
    for v in (0.05, 0.0): V[f"mu {mu:+g} var {v:g}"] = (mu, v)
V["mu -1 var 0.5"] = (-1.0, 0.5)
def oof(mu, v, s):
    out = []
    for F in G.FS:
        p = np.full(len(A.y), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=s).split(A.y): p[b] = GPM(mu, v).fit(A.select(F, a), A.y[a]).predict(A.select(F, b))
        out.append(p)
    return (out[0] + out[1]) / 2
def run(s):
    out = {}
    for n, (mu, v) in V.items():
        o = oof(mu, v, s); out[(n, 1.0)] = G.loss(o, s); out[(n, 1.19)] = C.level(o, s, 1.19)
    return out
if __name__ == "__main__":
    R = Parallel(n_jobs=5)(delayed(run)(s) for s in range(801, 811)); rows = []
    for n, lv in R[0]:
        base = ("base", lv); d = np.array([100 * (r[(n, lv)].sum() / r[base].sum() - 1) for r in R])
        L = np.mean([r[(n, lv)] for r in R], 0); L0 = np.mean([r[base] for r in R], 0)
        rng = np.random.default_rng(0); i = rng.integers(0, len(L), (4000, len(L))); boot = 100 * (L - L0)[i].sum(1) / L0[i].sum(1)
        rows.append(dict(variant=n, level=lv, mean=d.mean(), seeds_better=int((d < 0).sum()), boot_sd=boot.std(), p_better=(boot < 0).mean()))
    O = pd.DataFrame(rows).sort_values(["level", "mean"]); O.to_csv(A.ROOT / "reports/v57_amp.csv", index=False); print(O.round(2).to_string(index=False))
