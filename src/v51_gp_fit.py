"""v51: reduced GP repair-cost model (v39 inputs minus exposure and the s023 and s129 histograms; v47-v50), saved in the same
format as v39_gp.npz. Variant B also drops wear sensor s123; C is the average of A and B in log cost. Prints honest checks for A, B, C
on fresh seeds and writes reports/v51{a,b,c}_gp.npz (test log-cost, 5-fold seed-42 OOF residuals)."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import v47_ablation as A, v49_round2 as R2, v50_stress as T
V = {"v42": T.V["v42"], "A": T.V["reduced"], "B": T.V["reduced - s123"]}; T.V.update(V)
def one(s):
    T.V.update(V); o = {n: T.oof(n, s) for n in V}; o["C"] = (o["A"] + o["B"]) / 2
    return {n: (T.loss(v, s, *T.MIX["v42 flags"]), T.loss(v, s, *T.MIX["v42 flags"], w=T.WT)) for n, v in o.items()}
def testfeats(F_cols):
    te, f = A.te, A.f; st, wear = F_cols; ht = A.extract(te)
    def zt(cols):
        Xf = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); X = np.log1p(np.maximum(te[cols].to_numpy(float), 0))
        return (X - np.nanmean(Xf, 0)) / (np.nanstd(Xf, 0) + 1e-9)
    return [ht[A.START.index(a)] for a in st] + [zt(wear), zt(A.SENS)]
if __name__ == "__main__":
    R = Parallel(n_jobs=6)(delayed(one)(s) for s in range(401, 411))
    for n in ("A", "B", "C"):
        d = np.array([[100 * (r[n][i] / r["v42"][i] - 1) for i in (0, 1)] for r in R])
        print(f"{n}: v42-flag loss {d[:, 0].mean():+.2f}% ({(d[:, 0] < 0).sum()}/10); wear-weighted {d[:, 1].mean():+.2f}% ({(d[:, 1] < 0).sum()}/10)")
    ref = np.load(A.ROOT / "reports/v39_gp.npz"); sup = np.isfinite(ref["sev"]); out = {}
    for n, wear in (("A", A.WEAR), ("B", ["s126", "s149"])):
        F, cls, _ = T.V["reduced" if n == "A" else "reduced - s123"]; oof = np.full(len(A.y), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=42).split(A.y): oof[b] = cls().fit(A.select(F, a), A.y[a]).predict(A.select(F, b))
        oof[~A.valid] = np.nan; sev = cls().fit(F, A.y).predict(testfeats((R2.ST, wear))); sev[~sup] = np.nan; out[n] = (sev, oof)
    out["C"] = ((out["A"][0] + out["B"][0]) / 2, (out["A"][1] + out["B"][1]) / 2)
    for n, (sev, oof) in out.items():
        ok = np.isfinite(oof); res = A.y[ok] - oof[ok]; d = sev[sup] - ref["sev"][sup]
        print(f"{n}: seed-42 OOF R2 {1 - (res**2).sum() / ((A.y[ok] - A.y[ok].mean())**2).sum():.4f} (v42 {1 - (ref['res']**2).sum() / ((A.y[ok] - A.y[ok].mean())**2).sum():.4f}); "
              f"resid sd {res.std():.3f}; test: corr with v39 {np.corrcoef(sev[sup], ref['sev'][sup])[0, 1]:.4f}, mean shift {d.mean():+.4f}, sd {d.std():.3f}")
        np.savez(A.ROOT / f"reports/v51{n.lower()}_gp.npz", sev=sev, res=res, oof=oof, y=A.y)
