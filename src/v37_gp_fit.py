"""v37: final fit of the improved GP repair-cost model (v33 histogram kernels + wear group + top-6 non-histogram cost sensors).
Saves test log-cost predictions and honest 5-fold OOF residuals (same split as v33) to reports/v37_gp.npz."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np
from sklearn.model_selection import KFold
from common import ROOT, load
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy())
WEAR = ["s126", "s149", "s123"]; SENS = ["s049", "s116", "s142", "s105", "s010", "s104"]
def zfit(cols):
    X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
ZW, ZS = zfit(WEAR), zfit(SENS)
def feats(d):
    zw = (np.log1p(np.maximum(d[WEAR].to_numpy(float), 0)) - ZW[0]) / ZW[1]; zs = (np.log1p(np.maximum(d[SENS].to_numpy(float), 0)) - ZS[0]) / ZS[1]
    return extract(d) + [zw, zs]
F = feats(f); oof = np.full(len(y), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(y):
    oof[b] = GaussianSeverity().fit(select(F, a), y[a]).predict(select(F, b))
ok = np.isfinite(oof); res = y[ok] - oof[ok]
sev = GaussianSeverity().fit(F, y).predict(feats(te))
old = np.load(ROOT / "reports/v35_gp.npz")["sev"]; both = np.isfinite(sev) & np.isfinite(old)
print(f"OOF R2 {1 - (res**2).sum() / ((y[ok] - y[ok].mean())**2).sum():.4f}; resid sd {res.std():.3f}; test supported {np.isfinite(sev).sum()}; "
      f"corr with v33 GP on test {np.corrcoef(sev[both], old[both])[0, 1]:.3f}; mean log-cost shift {np.mean(sev[both] - old[both]):+.3f}")
np.savez(ROOT / "reports/v37_gp.npz", sev=sev, res=res, oof=oof, y=y)
