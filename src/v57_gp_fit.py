"""v57: v51 C inputs with a tight prior on the kernel amplitudes (log-amplitude ~ N(0, 0.05)); average of the fits with and without s123.
Saves reports/v57_gp.npz in the v39 format (test log-cost, 5-fold seed-42 OOF residuals)."""
import numpy as np
from sklearn.model_selection import KFold
import v47_ablation as A, v49_round2 as R2, v57_amp as M
from v51_gp_fit import testfeats
ref = np.load(A.ROOT / "reports/v51c_gp.npz"); sup = np.isfinite(ref["sev"]); oofs, sevs = [], []
for wear in (A.WEAR, ["s126", "s149"]):
    F = R2.F(wear=wear); o = np.full(len(A.y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=42).split(A.y): o[b] = M.GPM(0.0, 0.05).fit(A.select(F, a), A.y[a]).predict(A.select(F, b))
    o[~A.valid] = np.nan; g = M.GPM(0.0, 0.05).fit(F, A.y); s = g.predict(testfeats((R2.ST, wear))); s[~sup] = np.nan; oofs.append(o); sevs.append(s)
    print("amplitudes", g.amplitude.round(2), "noise", round(float(g.noise), 3))
oof, sev = np.mean(oofs, 0), np.mean(sevs, 0); ok = np.isfinite(oof); res = A.y[ok] - oof[ok]; d = sev[sup] - ref["sev"][sup]
print(f"OOF R2 {1 - (res**2).sum() / ((A.y[ok] - A.y[ok].mean())**2).sum():.4f}; resid sd {res.std():.3f}; vs v51 on test: corr {np.corrcoef(sev[sup], ref['sev'][sup])[0, 1]:.4f}, mean shift {d.mean():+.4f}, sd {d.std():.3f}")
np.savez(A.ROOT / "reports/v57_gp.npz", sev=sev, res=res, oof=oof, y=A.y)
