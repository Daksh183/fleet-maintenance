import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import ks_2samp
from common import load
tr, te, S = load(); trr, ter, _ = load(raw=True)
H = [f"s{i:03d}" for i in range(54, 64)]
T = tr[H].sum(axis=1, min_count=10)
# all distinct coefficient groups from the NNLS fit
cand = [c for c in S if c not in H]
ratio = tr[cand].div(T.replace(0, np.nan), axis=0)
cand = [c for c in cand if (ratio[c].dropna() <= 1.0001).mean() > 0.999 and tr[c].notna().mean() > 0.8]
d = pd.concat([tr[cand], T.rename("T")], axis=1).dropna().sample(15000, random_state=0)
coef, _ = nnls(d[cand].to_numpy(float), d["T"].to_numpy(float))
s = pd.Series(coef, index=cand); s = s[s > 1e-3].round(3)
for v, g in s.groupby(s): print(f"coef {v}: {list(g.index)}")
