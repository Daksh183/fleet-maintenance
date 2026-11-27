"""v60: failure-model ideas not yet tried under a shift-aware test. All XGBoost "slow" settings; one change each.
 base      all raw sensors except s120/s122          all      all raw sensors
 mono      base + risk forced non-decreasing in usage and in s126, s149 (extrapolates sensibly to high wear)
 shares    base + histogram shares and log usage      sharesOnly  histogram counts replaced by shares + usage
 rates     base + non-histogram sensors per unit usage          ratesOnly   those sensors replaced by their rates
 linear+   logistic regression on log1p sensors + shares (a smooth model that extrapolates)
Two judges: (1) wear-reweighted CV (validation weighted to test wear), cost at each candidate's own best level; (2) extrapolation:
train on vehicles below the 80th percentile of s126, validate on the top 20%."""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); c = dev.repair_cost.fillna(0).to_numpy() * y
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
NO = [s for s in S if s not in ("s120", "s122")]
HG = {a: [f"s{i:03d}" for i in range(a, a + (9 if a == 54 else 10))] for a in (23, 37, 54, 80, 129, 154)}
HC = [x for g in HG.values() for x in g]; OTHER = [s for s in NO if s not in HC and s != "s063"]
def derived(d):
    U = d[HG[54]].sum(axis=1, min_count=9)
    sh = {}
    for a, g in HG.items():
        tot = d[g].sum(axis=1, min_count=len(g)).replace(0, np.nan)
        for col in g: sh["sh_" + col] = d[col] / tot
    sh["logU"] = np.log1p(U)
    rt = d[OTHER].div(U.replace(0, np.nan), axis=0).add_suffix("_perU")
    return pd.DataFrame(sh, index=d.index), rt
SH, RT = derived(dev); D = pd.concat([dev, SH, RT], axis=1)
SHC, RTC = list(SH.columns), list(RT.columns)
SLOW = {**PARAMS, "n_estimators": 1000, "learning_rate": 0.02, "colsample_bytree": 0.5, "n_jobs": 2}
def mono(cols): return "(" + ",".join("1" if x in ("s126", "s149", "logU") else "0" for x in cols) + ")"
SPEC = {"base": (NO, {}), "all": (S, {}), "mono": (NO + ["logU"], {"monotone_constraints": mono(NO + ["logU"])}), "base+logU": (NO + ["logU"], {}),
        "shares": (NO + SHC, {}), "sharesOnly": ([s for s in NO if s not in HC] + SHC, {}), "rates": (NO + RTC + ["logU"], {}),
        "ratesOnly": (HC + RTC + ["logU"], {}), "linear+": (None, {})}
LIN = [s for s in NO if s != "s063"]
def fit(nm, a, b):
    cols, kw = SPEC[nm]
    if cols is None:
        Z = pd.concat([np.log1p(D[LIN].clip(lower=0)), D[SHC]], axis=1); mu, sd = Z.iloc[a].mean(), Z.iloc[a].std() + 1e-9
        Z = ((Z - mu) / sd).clip(-6, 6); Z = pd.concat([Z.fillna(0), Z.isna().add_suffix("_na").astype(float)], axis=1).to_numpy()
        return LogisticRegression(C=0.01, max_iter=3000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
    return xgb.XGBClassifier(**{**SLOW, **kw}).fit(D[cols].iloc[a], y[a]).predict_proba(D[cols].iloc[b])[:, 1]
x = dev.s126.to_numpy(); e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
h = lambda v: np.histogram(np.log1p(v[np.isfinite(v)]), e)[0] / np.isfinite(v).sum()
ratio = np.clip(h(te.s126.to_numpy()) / np.maximum(h(tr.s126.to_numpy()), 1e-9), 0, 5)
W = ratio[np.clip(np.searchsorted(e, np.log1p(dev.s126.fillna(dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)]
HI = np.nan_to_num(x, nan=np.nanmedian(x)) >= np.nanquantile(x, 0.8)
FOLDS = [list(StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y)) for r in range(3)]
def cost_best(q, idx, w):
    best = np.inf
    for lam in (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6):
        qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, md[idx], resid)
        cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[idx] - bu, 0) + OVER * np.maximum(bu - c[idx], 0), c[idx]); best = min(best, (cost * w).sum() / w.sum())
    return best
if __name__ == "__main__":
    names = list(SPEC)
    jobs = [(n, r, k) for n in names for r in range(3) for k in range(5)]
    R = Parallel(n_jobs=4)(delayed(lambda n, r, k: fit(n, *FOLDS[r][k]))(*j) for j in jobs)
    P = {n: np.zeros((3, len(y))) for n in names}
    for (n, r, k), p in zip(jobs, R): P[n][r, FOLDS[r][k][1]] = p
    for n in names: np.save(ROOT / f"reports/v18_search/v60_{n}.npy", P[n])
    for nm, parts in {"base+mono+shares": ("base", "mono", "shares"), "base+rates": ("base", "rates"), "base+linear": ("base", "linear+"), "all+base": ("all", "base")}.items():
        P[nm] = 1 / (1 + np.exp(-np.mean([np.log(np.clip(P[q], 1e-7, 1 - 1e-7) / (1 - np.clip(P[q], 1e-7, 1 - 1e-7))) for q in parts], 0)))
    lo = np.where(~HI)[0]; hi = np.where(HI)[0]; rows = []
    inner = list(StratifiedKFold(5, shuffle=True, random_state=1).split(lo, y[lo]))
    ex = {}
    for n in names:
        o = np.zeros(len(lo)); res = Parallel(n_jobs=4)(delayed(fit)(n, lo[a], lo[b]) for a, b in inner)
        for (a, b), p in zip(inner, res): o[b] = p
        ex[n] = (o, np.mean([fit(n, lo, hi)], 0))
    for nm, parts in {"base+mono+shares": ("base", "mono", "shares"), "base+rates": ("base", "rates"), "base+linear": ("base", "linear+"), "all+base": ("all", "base")}.items():
        avg = lambda i: 1 / (1 + np.exp(-np.mean([np.log(np.clip(ex[q][i], 1e-7, 1 - 1e-7) / (1 - np.clip(ex[q][i], 1e-7, 1 - 1e-7))) for q in parts], 0)))
        ex[nm] = (avg(0), avg(1))
    V = {}
    for n in P:
        V[n] = np.array([cost_best(make("platt").fit(P[n][r, a], y[a]).predict(P[n][r, b]), b, W[b]) for r in range(3) for a, b in FOLDS[r]])
    print(f"extrapolation set: {len(hi)} vehicles, {int(y[hi].sum())} failures (train: {int(y[lo].sum())} failures)")
    for n in P:
        o, ph = ex[n]; q = make("platt").fit(o, y[lo]).predict(ph); d = V[n] - V["base"]
        rows.append(dict(model=n, wear_cv_cost=V[n].mean(), d_vs_base=d.mean(), se=d.std() / np.sqrt(15), better=int((d < 0).sum()),
                         extrap_cost=cost_best(q, hi, np.ones(len(hi))), extrap_AP=average_precision_score(y[hi], ph), extrap_AUC=roc_auc_score(y[hi], ph)))
    O = pd.DataFrame(rows).sort_values("wear_cv_cost"); O.to_csv(ROOT / "reports/v60_feats.csv", index=False); print(O.round(4).to_string(index=False))
