"""v37: which added kernel group shifts test repair-cost predictions? Compare test means of tree, v33 GP, GP+wear, GP+sens, GP+wear+sens,
overall and on flagged (v35_combo) vehicles, and against the training-failure mean."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np, pandas as pd
from common import ROOT, load
from v33_hist_gp import GaussianSeverity, extract
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy())
WEAR = ["s126", "s149", "s123"]; SENS = ["s049", "s116", "s142", "s105", "s010", "s104"]
def zfit(cols): X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
Z = {k: zfit(c) for k, c in (("w", WEAR), ("s", SENS))}
zc = lambda d, cols, k: (np.log1p(np.maximum(d[cols].to_numpy(float), 0)) - Z[k][0]) / Z[k][1]
def feats(d, w, s): return extract(d) + ([zc(d, WEAR, "w")] if w else []) + ([zc(d, SENS, "s")] if s else [])
_, m = np.load(ROOT / "reports/v18_qm.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v35_combo.csv").flag.to_numpy() == 1
out = {"tree": m}
for nm, w, s in (("gp_v33", 0, 0), ("gp_wear", 1, 0), ("gp_sens", 0, 1), ("gp_wear_sens", 1, 1)):
    out[nm] = GaussianSeverity().fit(feats(f, w, s), y).predict(feats(te, w, s))
print(f"training failures: mean log cost {y.mean():.3f}")
for nm, v in out.items():
    ok = np.isfinite(v); print(f"{nm:13s} test mean {np.nanmean(v):.3f}   flagged mean {np.nanmean(v[fl & ok]):.3f}   wear-z>2 share of flagged {np.mean(zc(te, WEAR, 'w')[fl][:,0] > 2):.2f}")
zt = zc(te, WEAR, "w")[:, 0]; zf = zc(f, WEAR, "w")[:, 0]
print(f"wear z (s126): train failures max {np.nanmax(zf):.2f}; flagged test above that: {np.mean(zt[fl] > np.nanmax(zf)):.3f}")
np.savez(ROOT / "reports/v37_shiftcheck.npz", **out)
