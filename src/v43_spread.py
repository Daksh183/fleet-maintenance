"""v43: residual-spread models for budgets on the v39 GP, honest cross-fitted (spread estimated on the other folds' failures).
 pooled | 3 bands by predicted cost (v42) | smooth location-scale: log|r| ~ a + b*mu, quantiles of r/s(mu) | 3 bands by wear (log s126)
 | 3x2 bands (predicted cost x wear). Budget loss over the flagged-test level mix, three fold seeds."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy()); c = np.exp(y)
WEAR = ["s126", "s149", "s123"]; SENS = ["s116", "s142", "s105", "s104", "s078"]
def zfit(cols): X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
ZW, ZS = zfit(WEAR), zfit(SENS)
F = extract(f) + [(np.log1p(np.maximum(f[WEAR].to_numpy(float), 0)) - ZW[0]) / ZW[1], (np.log1p(np.maximum(f[SENS].to_numpy(float), 0)) - ZS[0]) / ZS[1]]
wear = np.log1p(f.s126.fillna(f.s126.median()).to_numpy())
ph = np.load(ROOT / "reports/v35_harness_p.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v42_gp_bands.csv").flag.to_numpy() == 1
p = ph[fl]; lev = np.clip(1 - (1 - UNDER / (UNDER + OVER)) / p, 0, 1); k = lev > 0; p, lev = p[k], lev[k]
for seed in (42, 7, 123):
    o = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(y): o[b] = GaussianSeverity().fit(select(F, a), y[a]).predict(select(F, b))
    ok = np.isfinite(o); mu, r, cc, w = o[ok], y[ok] - o[ok], c[ok], wear[ok]
    fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed + 1000).split(mu)): fold[b] = j
    def loss(Q):
        B = np.exp(mu[None, :] + Q); return float((p * (UNDER * np.maximum(cc[None, :] - B, 0) + OVER * np.maximum(B - cc[None, :], 0)).mean(1)).sum())
    def banded(keys):
        Q = np.zeros((len(lev), len(mu)))
        for j in range(5):
            tst = fold == j
            for kk in np.unique(keys):
                src = (fold != j) & (keys == kk); idx = tst & (keys == kk)
                if idx.any(): Q[:, idx] = np.quantile(r[src] if src.sum() >= 15 else r[fold != j], lev)[:, None]
        return loss(Q)
    def tercile(x, train_mask=None): return np.searchsorted(np.quantile(x, [1 / 3, 2 / 3]), x)
    def smooth():
        Q = np.zeros((len(lev), len(mu)))
        for j in range(5):
            src = fold != j; tst = fold == j
            A = np.c_[np.ones(src.sum()), mu[src]]; coef = np.linalg.lstsq(A, np.log(np.abs(r[src]) + 0.05), rcond=None)[0]
            s = lambda x: np.exp(coef[0] + coef[1] * x); e = r[src] / s(mu[src])
            Q[:, tst] = np.quantile(e, lev)[:, None] * s(mu[tst])[None, :]
        return loss(Q)
    res = {"pooled": banded(np.zeros(len(mu), int)), "3 bands cost (v42)": banded(tercile(mu)), "smooth loc-scale": smooth(),
           "3 bands wear": banded(tercile(w)), "3x2 cost x wear": banded(tercile(mu) * 2 + (w > np.median(w)))}
    base = res["3 bands cost (v42)"]
    print(f"seed {seed}: " + " | ".join(f"{k} {100 * (v / base - 1):+.1f}%" for k, v in res.items()), flush=True)
