"""v44: GP kernel-width multiplier (all groups) and bootstrap bagging, on the v39 inputs with v42's 3-band spread.
Honest cross-fitted budget loss over the flagged-test level mix (v42 flags), three fold seeds, vs v42 (width x1, no bagging)."""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "2")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, extract, select
class GPW(GaussianSeverity):
    def __init__(self, mult=1.0): self.mult = mult
    def blocks(self, features):
        if not hasattr(self, "_scaled"): self.scale = [s * self.mult for s in self.scale]; self._scaled = True
        return super().blocks(features)
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy()); c = np.exp(y)
WEAR = ["s126", "s149", "s123"]; SENS = ["s116", "s142", "s105", "s104", "s078"]
def zfit(cols): X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
ZW, ZS = zfit(WEAR), zfit(SENS)
F = extract(f) + [(np.log1p(np.maximum(f[WEAR].to_numpy(float), 0)) - ZW[0]) / ZW[1], (np.log1p(np.maximum(f[SENS].to_numpy(float), 0)) - ZS[0]) / ZS[1]]
ph = np.load(ROOT / "reports/v35_harness_p.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v42_gp_bands.csv").flag.to_numpy() == 1
p = ph[fl]; lev = np.clip(1 - (1 - UNDER / (UNDER + OVER)) / p, 0, 1); k = lev > 0; p, lev = p[k], lev[k]
def oof(seed, mult=1.0, bag=0):
    o = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(y):
        if bag == 0: o[b] = GPW(mult).fit(select(F, a), y[a]).predict(select(F, b))
        else:
            rng = np.random.default_rng(seed); preds = []
            for _ in range(bag):
                s = rng.choice(a, len(a), replace=True); preds.append(GPW(mult).fit(select(F, s), y[s]).predict(select(F, b)))
            o[b] = np.nanmean(preds, 0)
    return o
def honest(o, seed):
    ok = np.isfinite(o); mu, r, cc = o[ok], y[ok] - o[ok], c[ok]; fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed + 1000).split(mu)): fold[b] = j
    band = np.searchsorted(np.quantile(mu, [1 / 3, 2 / 3]), mu); Q = np.zeros((len(lev), len(mu)))
    for j in range(5):
        for kk in range(3):
            src = (fold != j) & (band == kk); idx = (fold == j) & (band == kk)
            if idx.any(): Q[:, idx] = np.quantile(r[src], lev)[:, None]
    B = np.exp(mu[None, :] + Q); return float((p * (UNDER * np.maximum(cc[None, :] - B, 0) + OVER * np.maximum(B - cc[None, :], 0)).mean(1)).sum())
VARS = [("x1 (v42)", 1.0, 0), ("x0.5", 0.5, 0), ("x0.75", 0.75, 0), ("x1.5", 1.5, 0), ("x2", 2.0, 0), ("bag10", 1.0, 10)]
for seed in (42, 7, 123):
    res = dict(Parallel(n_jobs=6)(delayed(lambda nm, mu_, bg: (nm, honest(oof(seed, mu_, bg), seed)))(nm, mu_, bg) for nm, mu_, bg in VARS))
    base = res["x1 (v42)"]; print(f"seed {seed}: " + " | ".join(f"{k} {100 * (v / base - 1):+.1f}%" for k, v in res.items()), flush=True)
