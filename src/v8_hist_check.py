import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from common import load
tr, te, S = load()
G = {n: [f"s{i:03d}" for i in range(a, a + 10)] for n, a in (("h023", 23), ("h037", 37), ("h054", 54), ("h080", 80), ("h129", 129), ("h154", 154))}
T = tr[G["h054"]].sum(axis=1, min_count=10)
for n, g in G.items():
    r = (tr[g].sum(axis=1, min_count=10) / T).dropna(); print(f"   {n} sum/T quantiles", np.quantile(r, [.05,.25,.5,.75,.95]).round(3))
    shares_tr = tr[g].div(tr[g].sum(axis=1), axis=0); shares_te = te[g].div(te[g].sum(axis=1), axis=0)
    ks0 = [ks_2samp(shares_tr[g[i]].dropna(), shares_te[g[i]].dropna()).statistic for i in range(10) if shares_te[g[i]].notna().any()]
    ksp = [ks_2samp(shares_tr[g[i]].dropna(), shares_te[g[i + 1]].dropna()).statistic for i in range(9) if shares_te[g[i + 1]].notna().any()]
    print(f"{n}: sum/T median {r.median():.4f} within1% {np.mean(abs(r-1)<.01):.3f} | NaN tr {tr[g].isna().mean().mean():.3f} te {te[g].isna().mean().mean():.3f}"
          f" | KS same-bin max {max(ks0):.3f}  shift+1 max {max(ksp):.3f}")
    print("    mean share per bin  train", shares_tr.mean().round(3).tolist())
    print("                        test ", shares_te.mean().round(3).tolist())
