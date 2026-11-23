import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from common import load
from features import ratios, shape
tr, te, S = load()
def ks(a, b): a, b = a.dropna(), b.dropna(); return ks_2samp(a, b).statistic if len(a) and len(b) else np.nan
raw = pd.Series({c: ks(tr[c], te[c]) for c in S})
sh = pd.Series({c: ks(x, shape(te)[c]) for c, x in shape(tr).items()})
rt, rte = ratios(tr, S), ratios(te, S)
ra = pd.Series({c: ks(rt[c], rte[c]) for c in rt})
print(f"KS raw sensors: median {raw.median():.3f}  max {raw.max():.3f}  (n {len(raw)})")
print(f"KS shape feats: median {sh.median():.3f}  max {sh.max():.3f}  top {sh.sort_values().tail(3).round(3).to_dict()}")
print(f"KS ratio feats: median {ra.median():.3f}  max {ra.max():.3f}  top {ra.sort_values().tail(5).round(3).to_dict()}")
for c in ("s120", "s122", "s126", "s128"):
    print(f"  {c}: raw KS {raw[c]:.3f}  per-usage KS {ra[c + '_perU']:.3f}")
print(f"usage logU: train median {np.expm1(shape(tr).logU.median()):.3g}  test {np.expm1(shape(te).logU.median()):.3g}  KS {sh['logU']:.3f}")
