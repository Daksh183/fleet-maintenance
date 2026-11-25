"""Why did v12_robust score best? Compare the vehicles each file flags against v12, and relate per-file flag statistics to public scores."""
import numpy as np, pandas as pd
from common import ROOT, load
tr, te, S = load()
OBS = {"v9_k4": 17013.86, "v10_bayes": 17348.75, "v12_robust": 16839.86, "v13_bag": 17974.55, "v14_map": 17375.65, "v15_map_v12": 17058.28,
       "v17_bud125": 16826.0, "v17_k135": 16908.0, "v18_ens10": 17624.0, "v9_em": 18058.56, "v7_k4": 19067.17, "v9_k1": 19600.79}
F = {k: pd.read_csv(ROOT / f"submissions/submission_{k}.csv") for k in OBS}
q, m = np.load(ROOT / "reports/v18_qm.npy")
DR = ["s120", "s122", "s126", "s128"]
hi = {s: (te[s] > np.nanquantile(tr[s], 0.99)).to_numpy() for s in DR}             # test values beyond the training 99th pct
nan = te[[c for c in S if c != "s063"]].isna().sum(1).to_numpy()
v12 = F["v12_robust"].flag.to_numpy()
rows = []
for k, f in F.items():
    fl = f.flag.to_numpy(); only12, onlyk = (v12 == 1) & (fl == 0), (fl == 1) & (v12 == 0)
    r = {"file": k, "public": OBS[k], "flags": fl.sum(), "v12only": only12.sum(), "fileonly": onlyk.sum()}
    for s in DR: r[f"{s}_hi"] = hi[s][fl == 1].mean()
    r["nan_mean"] = nan[fl == 1].mean(); r["exp_m"] = np.exp(m[fl == 1]).mean(); r["budget_mean"] = f.budget[fl == 1].mean()
    if onlyk.sum(): r["diff_v12only_s126hi"] = hi["s126"][only12].mean() if only12.sum() else np.nan; r["diff_fileonly_s126hi"] = hi["s126"][onlyk].mean()
    rows.append(r)
D = pd.DataFrame(rows).sort_values("public"); pd.set_option("display.width", 250); print(D.round(3).to_string(index=False))
print("\nshare of test beyond train 99th pct:", {s: round(hi[s].mean(), 3) for s in DR})
