"""v40: H1, failure model trained with wear weights (training rows weighted by the same test/train wear density ratio).
v32: shift-aware CV. Validation folds made to look like the test depot, to judge RANKINGS (which vehicles to flag).
 - wear reweighting: validation vehicles weighted by the test/train density ratio of log(s126) (20-bin histogram ratio, clipped)
 - scenario 'real'      : no sensor change in validation (s122's higher test readings are genuine)
 - scenario 'instrument': validation s122 x 2.55 and s120 + 16.0M (the test depot's quirks are instrument faults)
Calibration (Platt) is fitted on UNSHIFTED out-of-fold predictions of the other folds, as in the real pipeline.
Each candidate is scored at its own best level (odds multiplier scanned), so differences are about ranking only.
Candidates: all sensors (v12), no s120/s122 (v22), quantile-mapped s120/s122 (v14/15), oracle-fixed (instrument undone),
and 50/50 logit blends."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
c = dev.repair_cost.fillna(0).to_numpy() * y
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
# wear weights
e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
h = lambda x: np.histogram(np.log1p(x[np.isfinite(x)]), e)[0] / np.isfinite(x).sum()
ratio = np.clip(h(te.s126.to_numpy()) / np.maximum(h(tr.s126.to_numpy()), 1e-9), 0, 5)
bins = np.clip(np.searchsorted(e, np.log1p(dev.s126.fillna(dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)
W = ratio[bins]; print(f"wear weights: ESS {W.sum()**2 / (W**2).sum():,.0f} of {len(W):,}; weighted failure rate {np.average(y, weights=W):.4f} vs {y.mean():.4f}", flush=True)
NO = [s for s in S if s not in ("s120", "s122")]
def shifted(D, scen):
    D = D.copy()
    if scen == "instrument": D["s122"] = D.s122 * 2.55; D["s120"] = D.s120 + 16.0e6
    return D
def qmap(Dv, Dt):
    Dv = Dv.copy()
    for col in ("s120", "s122"):
        r = Dv[col].rank(pct=True); Dv[col] = np.where(Dv[col].isna(), np.nan, np.nanquantile(Dt[col], r.fillna(.5)))
    return Dv
RES = {}
for scen in ("real", "instrument"):
    P = {k: np.zeros((3, len(y))) for k in ("all", "nos122", "all_w", "nos122_w")}; P0 = {k: np.zeros((3, len(y))) for k in ("all", "nos122", "all_w", "nos122_w")}
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            Da, Db = dev.loc[a], dev.loc[b]; Dbs = shifted(Db, scen)
            for nm, cols, wt in (("all", S, None), ("nos122", NO, None), ("all_w", S, W[a]), ("nos122_w", NO, W[a])):
                mdl = xgb.XGBClassifier(**PARAMS).fit(Da[cols], y[a], sample_weight=wt)
                P0[nm][r, b] = mdl.predict_proba(Db[cols])[:, 1]; P[nm][r, b] = mdl.predict_proba(Dbs[cols])[:, 1]
    print("fitted", scen, flush=True)
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    CANDS = {k: (P[k], P0[k]) for k in P}
    for nm, (Ps, Pu) in CANDS.items():
        out = []
        for r in range(3):
            for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
                q = make("platt").fit(Pu[r, a], y[a]).predict(Ps[r, b]); w = W[b]
                best = np.inf
                for lam in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
                    qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, md[b], resid)
                    cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[b] - bu, 0) + OVER * np.maximum(bu - c[b], 0), c[b])
                    best = min(best, (cost * w).sum() / w.sum())
                out.append(best)
        RES[(scen, nm)] = np.array(out)
base = {s: RES[(s, "all")] for s in ("real", "instrument")}
print(f"\n{'scenario':11s} {'candidate':14s} {'cost/veh':>9s} {'vs all (v12)':>16s} {'better':>7s}")
for (s, nm), v in sorted(RES.items(), key=lambda kv: (kv[0][0], kv[1].mean())):
    d = v - base[s]; print(f"{s:11s} {nm:14s} {v.mean():9.4f} {d.mean():+8.4f}±{d.std() / np.sqrt(15):.4f} {(d < 0).sum():3d}/15")
pd.DataFrame({f"{s}|{n}": v for (s, n), v in RES.items()}).to_csv(ROOT / "reports/v40_wearw.csv", index=False)
