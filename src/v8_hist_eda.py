"""Find 10-bin histogram groups: consecutive windows of 10 columns whose row sums equal the known
histogram's (s054-s063) row sum. Also report the columns most correlated with that total (usage proxies)."""
import numpy as np, pandas as pd
from common import load
tr, te, S = load()
X = tr[S]
T = tr[[f"s{i:03d}" for i in range(54, 64)]].sum(axis=1, min_count=10)
print(f"known histogram total: median {T.median():.0f}, NaN {T.isna().mean():.3f}")
cols = [c for c in tr.columns if c.startswith("s")]
idx = {c: int(c[1:]) for c in cols}
out = []
for start in range(0, 161):
    w = [f"s{i:03d}" for i in range(start, start + 10)]
    if not all(c in X for c in w): continue
    s = X[w].sum(axis=1, min_count=10)
    ok = s.notna() & T.notna() & (T > 0)
    r = (s[ok] / T[ok])
    out.append((start, np.median(np.abs(r - 1)), np.mean(np.abs(r - 1) < 0.01)))
o = pd.DataFrame(out, columns=["start", "med_abs_dev", "share_equal_1pct"]).sort_values("med_abs_dev")
print(o.head(12).to_string(index=False))
c = X.corrwith(T, method="spearman").abs().sort_values(ascending=False)
print("most correlated with histogram total (spearman):", c.head(15).round(3).to_dict())
