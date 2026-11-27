"""v35: refit Daksh's v33 Gaussian-process repair-cost model (histogram kernels) on the 397 training failures.
Saves test log-cost predictions (NaN where < 3 valid histograms) and the honest 5-fold out-of-fold residuals."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import ROOT, load
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load()
f = tr[tr.failure == 1]; y = np.log(f.repair_cost.to_numpy()); F = extract(f)
oof = np.full(len(y), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(y):
    oof[b] = GaussianSeverity().fit(select(F, a), y[a]).predict(select(F, b))
ok = np.isfinite(oof); res = y[ok] - oof[ok]
gp = GaussianSeverity().fit(F, y); sev = gp.predict(extract(te))
qv12, m = np.load(ROOT / "reports/v18_qm.npy")
print(f"OOF R2 GP {1 - ((y[ok]-oof[ok])**2).sum()/((y[ok]-y[ok].mean())**2).sum():.3f} on {ok.sum()} failures; resid sd {res.std():.3f}")
print(f"test supported {np.isfinite(sev).sum()}; corr GP vs tree log-cost {np.corrcoef(sev[np.isfinite(sev)], m[np.isfinite(sev)])[0,1]:.3f}")
np.savez(ROOT / "reports/v35_gp.npz", sev=sev, res=res, oof=oof, y=y)
