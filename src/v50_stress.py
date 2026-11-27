"""v50: stress tests of the reduced GP (v42 minus exposure, s023, s129 histograms).
(a) different flagged sets / level mixes: v37 flags, v12 flags, all test levels at p=ph; (b) 10-fold instead of 5-fold;
(c) test-wear weighting of failures (density ratio of log s126, test failures-proxy: flagged test vehicles);
(d) forward-in-wear: train on low-wear failures, score the high-wear third (the depot is shifted toward high wear);
(e) also drop wear s123 (v49's best single move)."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import v47_ablation as A, v49_round2 as R2
V = {"v42": (A.BASE, A.GP, "log"), "reduced": (R2.F(), A.GP, "log"), "reduced - s123": (R2.F(wear=["s126", "s149"]), A.GP, "log")}
ph = np.load(A.ROOT / "reports/v35_harness_p.npy"); te = A.te
def mix(flags):
    p = ph[flags]; lev = np.clip(1 - (1 - A.UNDER / (A.UNDER + A.OVER)) / p, 0, 1); k = lev > 0; return p[k], lev[k]
flag = lambda n: pd.read_csv(A.ROOT / f"submissions/submission_{n}.csv").flag.to_numpy() == 1
MIX = {"v42 flags": mix(flag("v42_gp_bands")), "v37 flags": mix(flag("v37_bayes_gp2")), "v12 flags": mix(flag("v12_robust"))}
f = A.f; x = np.log1p(f.s126.fillna(f.s126.median()).to_numpy()); xt = np.log1p(te.s126.fillna(te.s126.median()).to_numpy())[flag("v42_gp_bands")]
e = np.quantile(x, np.linspace(0, 1, 9)); e[0], e[-1] = -np.inf, np.inf
WT = np.clip(np.histogram(xt, e)[0] / len(xt) / (np.histogram(x, e)[0] / len(x)), 0, 5)[np.clip(np.searchsorted(e, x, side="right") - 1, 0, 7)]
def oof(n, s, k=5):
    F, cls, _ = V[n]; o = np.full(len(A.y), np.nan)
    for a, b in KFold(k, shuffle=True, random_state=s).split(A.y): o[b] = cls().fit(A.select(F, a), A.y[a]).predict(A.select(F, b))
    return o
def loss(o, s, p, lev, w=None, sel=None, pool=None):
    ok = A.valid; mu, cc, t = o[ok], A.c[ok], A.y[ok]; r = t - mu; fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s + 1000).split(mu)): fold[b] = j
    band = np.searchsorted(np.quantile(mu, [1 / 3, 2 / 3]), mu); Q = np.zeros((len(lev), len(mu)))
    for j in range(5):
        for kk in range(3):
            src = (fold != j) & (band == kk); idx = (fold == j) & (band == kk)
            if idx.any(): Q[:, idx] = np.quantile(r[src], lev)[:, None]
    B = np.minimum(np.exp(mu[None, :] + Q), 8000)
    v = (p[:, None] * (A.UNDER * np.maximum(cc[None, :] - B, 0) + A.OVER * np.maximum(B - cc[None, :], 0))).sum(0)
    return float((v * (1 if w is None else w[ok])).sum())
def one(n, s):
    o5, o10 = oof(n, s), oof(n, s, 10); out = {}
    for m, (p, lev) in MIX.items(): out[m] = loss(o5, s, p, lev)
    out["10-fold"] = loss(o10, s, *MIX["v42 flags"]); out["test-wear weighted"] = loss(o5, s, *MIX["v42 flags"], w=WT)
    return n, s, out
if __name__ == "__main__":
    seeds = list(range(301, 309))
    R = Parallel(n_jobs=6)(delayed(one)(n, s) for s in seeds for n in V)
    D = pd.DataFrame([dict(variant=n, seed=s, **o) for n, s, o in R]).groupby("variant").mean().drop(columns="seed")
    print((100 * (D / D.loc["v42"] - 1)).round(2).T.to_string())
    # (d) extrapolation in wear: train on the lower two thirds of s126, predict the top third; residual quantiles from train OOF
    q = np.quantile(x, 2 / 3); a = np.where(x <= q)[0]; b = np.where((x > q) & A.valid)[0]; p, lev = MIX["v42 flags"]
    for n, (F, cls, _) in V.items():
        o = np.full(len(A.y), np.nan)
        for i, j in KFold(5, shuffle=True, random_state=1).split(a): o[a[j]] = cls().fit(A.select(F, a[i]), A.y[a[i]]).predict(A.select(F, a[j]))
        r = A.y[a] - o[a]; r = r[A.valid[a]]
        pred = cls().fit(A.select(F, a), A.y[a]).predict(A.select(F, b)); B = np.minimum(np.exp(pred[None, :] + np.quantile(r, lev)[:, None]), 8000); cc = A.c[b]
        L = (p[:, None] * (A.UNDER * np.maximum(cc[None, :] - B, 0) + A.OVER * np.maximum(B - cc[None, :], 0))).sum()
        if n == "v42": L0 = L
        print(f"extrapolate to top-wear third: {n:16s} {100 * (L / L0 - 1):+.2f}%  rmse {np.sqrt(np.mean((pred - A.y[b]) ** 2)):.4f}")
