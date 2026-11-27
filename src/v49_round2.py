"""v49: second backward/forward round starting from the reduced GP inputs (v42 minus exposure, s023 and s129 histograms).
One change at a time, five fold seeds. Reports the paired standard error over VEHICLES too (data noise, not just fold noise)."""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import v47_ablation as A
ST = [37, 54, 80, 154]; HG = dict(zip(A.START, A.HG)); W, SE = A.WEAR, A.SENS
def F(st=ST, wear=W, sens=SE, extra=()):
    return [HG[a] for a in st] + [A.z(list(wear)), A.z(list(sens))] + list(extra)
V = {"v42": (A.BASE, A.GP, "log"), "reduced": (F(), A.GP, "log")}
for a in ST: V[f"- hist s{a:03d}"] = (F(st=[x for x in ST if x != a]), A.GP, "log")
for s in W: V[f"- wear {s}"] = (F(wear=[x for x in W if x != s]), A.GP, "log")
for s in SE: V[f"- sensor {s}"] = (F(sens=[x for x in SE if x != s]), A.GP, "log")
for s in ("s049", "s010", "s120", "s122"): V[f"+ sensor {s}"] = (F(sens=SE + [s]), A.GP, "log")
V["+ hist s023 back"] = (F(st=[23] + ST), A.GP, "log"); V["+ hist s129 back"] = (F(st=ST + [129]), A.GP, "log")
V["+ missing count"] = (F(extra=[A.MISS]), A.GP, "log"); V["+ all other sensors"] = (F(extra=[A.Zo]), A.GP, "log")
V["+ other hists as 1 raw block"] = (F(extra=[np.column_stack(A.RAW)]), A.GP, "log")
V["Matern 3/2"] = (F(), A.Matern, "log"); V["rational quadratic"] = (F(), A.Cauchy, "log")
V["blend 0.9"] = (F(), A.GP, "blend1")
def run(n, s, v):
    """per-vehicle loss vector (summed over the flagged-test level mix)"""
    A.V[n] = v; Fm, cls, tt = v; y = A.y; c = A.c; o = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=s).split(y):
        o[b] = cls().fit(A.select(Fm, a), y[a]).predict(A.select(Fm, b))
        if tt.startswith("blend"):
            w = int(tt[-1]) / 10
            tree = A.xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.02, subsample=0.8, colsample_bytree=0.5, n_jobs=1,
                                      random_state=42).fit(A.f[A.S].iloc[a], y[a])
            o[b] = (1 - w) * o[b] + w * tree.predict(A.f[A.S].iloc[b])
    mu, cc, t = o[A.valid], c[A.valid], y[A.valid]; r = t - mu; fold = np.zeros(len(mu), int)
    for j, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s + 1000).split(mu)): fold[b] = j
    band = np.searchsorted(np.quantile(mu, [1 / 3, 2 / 3]), mu); Q = np.zeros((len(A.lev), len(mu)))
    for j in range(5):
        for kk in range(3):
            src = (fold != j) & (band == kk); idx = (fold == j) & (band == kk)
            if idx.any(): Q[:, idx] = np.quantile(r[src], A.lev)[:, None]
    B = np.minimum(np.exp(mu[None, :] + Q), 8000)
    return n, s, (A.p[:, None] * (A.UNDER * np.maximum(cc[None, :] - B, 0) + A.OVER * np.maximum(B - cc[None, :], 0))).sum(0) / len(A.lev)
def table(V, seeds, ref):
    R = Parallel(n_jobs=6)(delayed(run)(n, s, v) for s in seeds for n, v in V.items())
    L = {n: np.mean([l for m, s, l in R if m == n], 0) for n in V}; out = []
    per = {n: np.array([next(l for m, ss, l in R if m == n and ss == s).sum() for s in seeds]) for n in V}
    for n in V:
        d = L[n] - L[ref]; rng = np.random.default_rng(0); idx = rng.integers(0, len(d), (4000, len(d)))
        boot = 100 * d[idx].sum(1) / L[ref][idx].sum(1)
        out.append(dict(variant=n, mean=100 * (L[n].sum() / L[ref].sum() - 1), seeds_better=int((per[n] < per[ref]).sum()),
                        boot_sd=boot.std(), p_better=(boot < 0).mean()))
    return pd.DataFrame(out).sort_values("mean")
if __name__ == "__main__":
    O = table(V, [42, 7, 123, 11, 99], "reduced"); O.to_csv(A.ROOT / "reports/v49_round2.csv", index=False); print(O.round(2).to_string(index=False))
    print("\nreduced and v47 candidates vs v42, fresh seeds 201-210, with vehicle bootstrap:")
    V2 = {"v42": V["v42"], "reduced": V["reduced"], "drop exp+s129": (F(st=[23] + ST), A.GP, "log")}
    print(table(V2, list(range(201, 211)), "v42").round(3).to_string(index=False))
