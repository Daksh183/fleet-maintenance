"""How much of a public score is luck? Exact sd of public totals (and of differences between files) under the fitted shift model."""
import numpy as np, pandas as pd
from common import ROOT
from score import INSPECT, OVER, UNDER
q, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
names = ["v12_robust", "v13_bag", "v15_map_v12", "v17_bud125", "v17_k135", "v9_k4", "v10_bayes"]
Fl = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").flag for k in names]).astype(float)
Bu = np.column_stack([pd.read_csv(ROOT / f"submissions/submission_{k}.csv").budget for k in names])
K = 49; C = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :])
U0 = Fl * (INSPECT + OVER * Bu)
U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
E1 = U1.mean(1); G = np.einsum("nkj,nkl->njl", U1, U1) / K; U00 = np.einsum("nj,nl->njl", U0, U0)
for k, frac in ((3.0, 0.3), (3.0, 0.7)):
    p = (k * q / (1 - q)) / (1 + k * q / (1 - q))
    mi = (1 - p[:, None]) * U0 + p[:, None] * E1
    cov = frac * (np.einsum("n,njl->jl", 1 - p, U00) + np.einsum("n,njl->jl", p, G)) - frac**2 * mi.T @ mi
    sd = np.sqrt(np.diag(cov)); print(f"k={k}, subset {frac:.0%}: expected v12_robust {frac*mi[:,0].sum():,.0f}, sd {sd[0]:,.0f} ({sd[0]/(frac*mi[:,0].sum()):.1%})")
    for j in range(1, len(names)):
        d = np.sqrt(cov[0, 0] + cov[j, j] - 2 * cov[0, j]); print(f"    sd of (v12_robust - {names[j]}) {d:6.0f}; expected diff {frac*(mi[:,0]-mi[:,j]).sum():+7.0f}")
