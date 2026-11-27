"""v38: per-vehicle GP uncertainty for budgets (Daksh's v34 idea) on the improved v37 GP (histograms + wear + sensors).
Budget = exp(mu + sigma_i * quantile(standardised OOF residuals, level)) vs the homoscedastic exp(mu + quantile(OOF residuals, level)).
Judged by OOF budget loss on the 397 failures at the budget levels our flagged test vehicles use (v35 harness p)."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np, pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, extract, select
class GPVar(GaussianSeverity):
    def fit(self, features, y):
        super().fit(features, y)
        blocks = self.blocks(features); unit = np.eye(len(y))
        # recover the noise by maximising the same penalised marginal likelihood with amplitudes fixed
        z = y - self.center
        def nll(ln):
            cov = np.einsum("i,ijk->jk", self.amplitude, blocks) + np.exp(ln[0]) * unit; F = cho_factor(cov, lower=True)
            return .5 * z @ cho_solve(F, z) + np.log(np.diag(F[0])).sum() + .5 * ln[0] ** 2 / 4
        self.noise = float(np.exp(minimize(nll, [0.0], bounds=[(-4., 3.)], method="L-BFGS-B").x[0]))
        self.factor = cho_factor(np.einsum("i,ijk->jk", self.amplitude, blocks) + self.noise * unit, lower=True); return self
    def predict_var(self, features):
        out = np.full(len(features[0]), np.nan)
        for st in range(0, len(out), 800):
            rows = slice(st, st + 800); vals = select(features, rows); kb = self.blocks(vals)
            k = np.einsum("i,ijk->jk", self.amplitude, kb)
            valid_groups = np.stack([np.isfinite(x).all(1) for x in vals], 0)
            kxx = (self.amplitude[:, None] * valid_groups).sum(0)
            out[rows] = np.maximum(self.noise, self.noise + kxx - np.einsum("ij,ji->i", k, cho_solve(self.factor, k.T)))
        return np.sqrt(out)
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy())
WEAR = ["s126", "s149", "s123"]; SENS = ["s049", "s116", "s142", "s105", "s010", "s104"]
def zfit(cols): X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
ZW, ZS = zfit(WEAR), zfit(SENS)
feats = lambda d: extract(d) + [(np.log1p(np.maximum(d[WEAR].to_numpy(float), 0)) - ZW[0]) / ZW[1], (np.log1p(np.maximum(d[SENS].to_numpy(float), 0)) - ZS[0]) / ZS[1]]
F = feats(f); mu = np.full(len(y), np.nan); sd = np.full(len(y), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(y):
    g = GPVar().fit(select(F, a), y[a]); mu[b] = g.predict(select(F, b)); sd[b] = g.predict_var(select(F, b))
ok = np.isfinite(mu); r = y[ok] - mu[ok]; zr = r / sd[ok]; c = np.exp(y[ok])
print(f"OOF sigma: median {np.median(sd[ok]):.3f}, 10-90% {np.quantile(sd[ok], [.1, .9]).round(3)}; corr(|resid|, sigma) {np.corrcoef(np.abs(r), sd[ok])[0, 1]:.3f}")
ph = np.load(ROOT / "reports/v35_harness_p.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v37_bayes_gp2.csv").flag.to_numpy() == 1
p = ph[fl]; lev = np.clip(1 - (1 - UNDER / (UNDER + OVER)) / p, 0, 1)
def wloss(budget_fn):
    tot = 0.0
    for pi, q in zip(p, lev):
        if q <= 0: continue
        b = budget_fn(q); tot += pi * (UNDER * np.maximum(c - b, 0) + OVER * np.maximum(b - c, 0)).mean()
    return tot
homo = wloss(lambda q: np.exp(mu[ok] + np.quantile(r, q)))
for shrink in (1.0, 0.5):
    s_ = sd[ok] ** shrink * np.median(sd[ok]) ** (1 - shrink)
    het = wloss(lambda q: np.exp(mu[ok] + s_ * np.quantile(r / s_, q)))
    print(f"per-vehicle sigma (shrink {shrink}): expected budget loss {het:9.1f} vs homoscedastic {homo:9.1f}  ({100 * (het / homo - 1):+.1f}%)")
