"""v8 engineered features (from v8_hist_eda*: six 10-bin histograms over a shared operating-time base).

Each histogram's total is a fixed fraction of the reference total U (h054), so U is 'usage'.
- shape(): per-histogram bin shares, mean bin, std bin, entropy, low/high tail shares, log U.
- ratios(): non-histogram sensors divided by U (usage-normalised rates).
None of the histograms except h054 is shifted in test (KS <= 0.022); h054 is realigned in load().
In test, h054 bin 10 (s063) is NaN after realignment, so h054 shares use bins 0-8 only.
"""
import numpy as np
import pandas as pd

HIST = {a: [f"s{i:03d}" for i in range(a, a + 10)] for a in (23, 37, 54, 80, 129, 154)}
HCOLS = {c for g in HIST.values() for c in g}


def shape(df):
    out = {}
    U = df[HIST[54][:9]].sum(axis=1, min_count=9)
    out["logU"] = np.log1p(U)
    for a, g in HIST.items():
        g = g[:9] if a == 54 else g
        x = df[g].to_numpy(float)
        tot = x.sum(1, keepdims=True)
        sh = np.where(tot > 0, x / np.where(tot > 0, tot, 1), np.nan)
        k = np.arange(len(g))
        mean = (sh * k).sum(1)
        out.update({f"h{a}_sh{i}": sh[:, i] for i in range(len(g))})
        out[f"h{a}_mean"] = np.where(tot[:, 0] > 0, mean, np.nan)
        out[f"h{a}_std"] = np.where(tot[:, 0] > 0, np.sqrt(np.maximum((sh * k ** 2).sum(1) - mean ** 2, 0)), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"h{a}_ent"] = np.where(tot[:, 0] > 0, -np.nansum(np.where(sh > 0, sh * np.log(sh), 0), 1), np.nan)
        out[f"h{a}_lo3"] = sh[:, :3].sum(1)
        out[f"h{a}_hi3"] = sh[:, -3:].sum(1)
    return pd.DataFrame(out, index=df.index)


def ratios(df, S):
    U = df[HIST[54][:9]].sum(axis=1, min_count=9).replace(0, np.nan)
    cols = [c for c in S if c not in HCOLS]
    return df[cols].div(U, axis=0).add_suffix("_perU")
