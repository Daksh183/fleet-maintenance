"""v58: end-to-end check of the v57 amplitude prior (candidate "amp") against v51 C (candidate "v51"). Otherwise as v52:
v52: end-to-end check of the reduced GP. Same shift-aware CV as v45 (slow no-s122 failure model, nested Platt, validation weighted to
test wear, each candidate at its own best level), 5 repeats x 5 folds. The GP is refit on each training fold's failures only and uses
v42's 3-band residual spread. Candidates: v42 inputs, and v51 A / B / C (reduced inputs)."""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, KFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
from v33_hist_gp import GaussianSeverity, START, extract, select
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); c = dev.repair_cost.fillna(0).to_numpy() * y
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
h = lambda x: np.histogram(np.log1p(x[np.isfinite(x)]), e)[0] / np.isfinite(x).sum()
ratio = np.clip(h(te.s126.to_numpy()) / np.maximum(h(tr.s126.to_numpy()), 1e-9), 0, 5)
W = ratio[np.clip(np.searchsorted(e, np.log1p(dev.s126.fillna(dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)]
NO = [s for s in S if s not in ("s120", "s122")]; prm = {**PARAMS, "n_estimators": 1000, "learning_rate": 0.02, "colsample_bytree": 0.5, "n_jobs": 1}
SENS = ["s116", "s142", "s105", "s104", "s078"]; W3 = ["s126", "s149", "s123"]; ST = (37, 54, 80, 154)
SPEC = {"A": (ST, W3, False, False), "B": (ST, W3[:2], False, False), "A2": (ST, W3, False, True), "B2": (ST, W3[:2], False, True)}
from v57_amp import GPM
def feats(d, ref, st, wear, exp):
    H = extract(d); out = [H[START.index(a)] for a in st] + ([H[6]] if exp else [])
    for cols in (wear, SENS):
        R = np.log1p(np.maximum(ref[cols].to_numpy(float), 0)); mu_, sd_ = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
        out.append((np.log1p(np.maximum(d[cols].to_numpy(float), 0)) - mu_) / sd_)
    return out
def eligible(d): return sum(np.isfinite(x).all(1).astype(int) for x in extract(d)[:6]) >= 3
class GP(GaussianSeverity):
    def predict(self, F):
        out = np.full(len(F[0]), np.nan)
        for s in range(0, len(out), 800):
            rows = slice(s, s + 800); out[rows] = np.clip(np.einsum("i,ijk->jk", self.amplitude, self.blocks(select(F, rows))) @ self.alpha + self.center, *self.bounds)
        return out
class GPAmp(GPM):
    predict = GP.predict
def fold(r, a, b):
    Da, Db = dev.loc[a], dev.loc[b]; oofa = np.zeros(len(a))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=1).split(np.zeros(len(a)), y[a]):
        oofa[ib] = xgb.XGBClassifier(**prm).fit(Da.iloc[ia][NO], y[a][ia]).predict_proba(Da.iloc[ib][NO])[:, 1]
    q = make("platt").fit(oofa, y[a]).predict(xgb.XGBClassifier(**prm).fit(Da[NO], y[a]).predict_proba(Db[NO])[:, 1])
    fa = Da[Da.failure == 1].reset_index(drop=True); ly = np.log(fa.repair_cost.to_numpy()); ea, eb = eligible(fa), eligible(Db)
    pred = {}
    for nm, full in SPEC.items():
        spec = full[:3]; make_gp = (lambda: GPAmp(0.0, 0.05)) if full[3] else GP
        Fa = feats(fa, fa, *spec); go = np.full(len(fa), np.nan)
        for ia, ib in KFold(5, shuffle=True, random_state=42).split(ly): go[ib] = make_gp().fit(select(Fa, ia), ly[ia]).predict(select(Fa, ib))
        go[~ea] = np.nan; gb = make_gp().fit(Fa, ly).predict(feats(Db, fa, *spec)); gb[~eb] = np.nan; pred[nm] = (go, gb)
    pred = {"v51": tuple((pred["A"][i] + pred["B"][i]) / 2 for i in (0, 1)), "amp": tuple((pred["A2"][i] + pred["B2"][i]) / 2 for i in (0, 1))}
    out = {}; w = W[b]; mt = md[b]
    for nm, (go, gb) in pred.items():
        ok = np.isfinite(go); mu, res = go[ok], (ly - go)[ok]; edges = np.quantile(mu, [1 / 3, 2 / 3]); bf = np.searchsorted(edges, mu)
        sup = np.isfinite(gb); band = np.searchsorted(edges, np.where(sup, gb, 0.0)); best = np.inf
        for lam in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
            qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, mt, resid)           # tree where the GP is unsupported
            for k in range(3):
                j = sup & (band == k)
                if j.any(): fl[j], bu[j] = decide_pv(qq[j], gb[j], res[bf == k])
            bu = np.minimum(bu, 8000)
            cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[b] - bu, 0) + OVER * np.maximum(bu - c[b], 0), c[b]); best = min(best, (cost * w).sum() / w.sum())
        out[nm] = best
    print("fold done", r, flush=True); return out
if __name__ == "__main__":
    jobs = [(r, a, b) for r in range(5) for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y)]
    R = pd.DataFrame(Parallel(n_jobs=5)(delayed(fold)(*j) for j in jobs)); R.to_csv(ROOT / "reports/v58_flagcv.csv", index=False)
    for k in R:
        d = R[k] - R["v51"]; print(f"{k:4s} cost/veh {R[k].mean():.4f}  vs v51 {d.mean():+.4f} ± {d.std() / np.sqrt(len(R)):.4f} ({100 * d.mean() / R['v51'].mean():+.2f}%)  better {(d < 0).sum()}/{len(R)}")
