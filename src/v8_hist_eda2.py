"""Non-consecutive histogram search: non-negative least squares of the known histogram total T on all
other columns (rows without NaN). A hidden histogram would show up as ~10 columns with coefficient ~1."""
import numpy as np, pandas as pd
from scipy.optimize import nnls
from common import load
tr, te, S = load()
H = [f"s{i:03d}" for i in range(54, 64)]
T = tr[H].sum(axis=1, min_count=10)
cand = [c for c in S if c not in H]
# a bin can never exceed the total: keep columns with value <= T on 99.9% of rows
ratio = tr[cand].div(T.replace(0, np.nan), axis=0)
cand = [c for c in cand if (ratio[c].dropna() <= 1.0001).mean() > 0.999 and tr[c].notna().mean() > 0.8]
print(f"columns that never exceed the total: {len(cand)}")
d = pd.concat([tr[cand], T.rename("T")], axis=1).dropna()
d = d.sample(min(len(d), 15000), random_state=0)
coef, rn = nnls(d[cand].to_numpy(float), d["T"].to_numpy(float))
s = pd.Series(coef, index=cand).sort_values(ascending=False)
print(f"rows {len(d)}, relative residual {rn / np.linalg.norm(d['T']):.4f}")
print("largest coefficients:", s.head(25).round(3).to_dict())
grp = s[(s > 0.8) & (s < 1.2)].index.tolist()
if grp:
    r = d[grp].sum(axis=1) / d["T"]
    print(f"{len(grp)} columns with coef ~1: {grp}; their sum / T: median {r.median():.4f}, within 1%: {np.mean(abs(r-1)<.01):.3f}")
