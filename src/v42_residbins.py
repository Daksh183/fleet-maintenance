"""v42: residual spread by predicted-cost level. Budget quantiles taken from OOF residuals of failures in the same predicted log-cost
band (2 or 3 bands, by OOF GP prediction) instead of one pooled set. Judged on v37 GP OOF budget loss over the flagged-test level mix,
on two fold seeds (v37 seed 42 and an independent refit with seed 7)."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy()); c = np.exp(y)
WEAR = ["s126", "s149", "s123"]; SENS = ["s049", "s116", "s142", "s105", "s010", "s104"]
def zfit(cols): X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
ZW, ZS = zfit(WEAR), zfit(SENS)
F = extract(f) + [(np.log1p(np.maximum(f[WEAR].to_numpy(float), 0)) - ZW[0]) / ZW[1], (np.log1p(np.maximum(f[SENS].to_numpy(float), 0)) - ZS[0]) / ZS[1]]
ph = np.load(ROOT / "reports/v35_harness_p.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v37_bayes_gp2.csv").flag.to_numpy() == 1
p = ph[fl]; lev = np.clip(1 - (1 - UNDER / (UNDER + OVER)) / p, 0, 1); k = lev > 0; p, lev = p[k], lev[k]
def run(seed):
    o = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(y): o[b] = GaussianSeverity().fit(select(F, a), y[a]).predict(select(F, b))
    ok = np.isfinite(o); mu, r, cc = o[ok], y[ok] - o[ok], c[ok]
    def loss(qfun):
        B = np.exp(mu[None, :] + qfun(lev)); return float((p * (UNDER * np.maximum(cc[None, :] - B, 0) + OVER * np.maximum(B - cc[None, :], 0)).mean(1)).sum())
    fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed + 1000).split(mu)): fold[b] = j
    def honest(nb):
        edges = np.quantile(mu, np.linspace(0, 1, nb + 1)[1:-1]) if nb > 1 else np.array([]); band = np.searchsorted(edges, mu)
        Q = np.zeros((len(lev), len(mu)))
        for j in range(5):
            tst = fold == j
            for bb in range(nb):
                src = (fold != j) & (band == bb); idx = tst & (band == bb)
                if idx.any(): Q[:, idx] = np.quantile(r[src], lev)[:, None]
        return loss(lambda L: Q)
    out = {"pooled": honest(1), "2 bands": honest(2), "3 bands": honest(3), "4 bands": honest(4)}
    return out
for seed in (42, 7):
    o = run(seed); print(f"fold seed {seed}: " + "  ".join(f"{k} {v:9.1f} ({100 * (v / o['pooled'] - 1):+.1f}%)" for k, v in o.items()), flush=True)
