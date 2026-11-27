"""v37: Daksh-style hypotheses on the repair-cost side. Variants of the additive histogram-kernel GP, judged on the 397 training
failures by honest 5-fold OOF (same split as v33): R2 and budget loss at the newsvendor level q=0.909 (and q=0.5)."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "4")
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy())
base = extract(f)
def zcols(cols):
    X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    return (X - mu) / sd
WEAR = zcols(["s126", "s149", "s123"])
# non-histogram sensors with most gain in a tree cost model
HC = {f"s{i:03d}" for a in (23, 37, 54, 80, 129, 154) for i in range(a, a + 10)}
cm = xgb.XGBRegressor(n_estimators=600, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5).fit(f[S], y)
g = pd.Series(cm.get_booster().get_score(importance_type="total_gain")); g = g[[c for c in g.index if c not in HC]].sort_values(ascending=False)
TOP = list(g.index[:6]); print("top non-histogram cost sensors:", TOP)
SENS = zcols(TOP)
VARS = {"gp_v33": base, "gp+wear": base + [WEAR], "gp+sens": base + [SENS], "gp+wear+sens": base + [WEAR, SENS]}
tree = y - np.load(ROOT / "reports/v18_resid.npy")
def evaluate(oof, nm):
    ok = np.isfinite(oof); rr = y[ok] - oof[ok]
    def pin(q): b = np.exp(oof[ok] + np.quantile(rr, q)); c = np.exp(y[ok]); return (UNDER * np.maximum(c - b, 0) + OVER * np.maximum(b - c, 0)).mean()
    print(f"{nm:16s} n {ok.sum():3d}  R2 {1 - (rr**2).sum() / ((y[ok] - y[ok].mean())**2).sum():.4f}  budget loss q.909 {pin(0.909):6.2f}  q.5 {pin(0.5):6.2f}", flush=True)
    return oof
OOF = {}
for nm, F in VARS.items():
    oof = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=42).split(y):
        oof[b] = GaussianSeverity().fit(select(F, a), y[a]).predict(select(F, b))
    OOF[nm] = evaluate(oof, nm)
# kNN on histogram shape features (all groups concatenated, NaN groups -> 0)
Xk = np.nan_to_num(np.hstack(base[:-1])); knn = np.full(len(y), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(y):
    knn[b] = KNeighborsRegressor(15, weights="distance").fit(Xk[a], y[a]).predict(Xk[b])
evaluate(knn, "knn15")
best = min(OOF, key=lambda k: np.nanmean((y - OOF[k]) ** 2))
for w in (0.15, 0.3):
    evaluate(np.where(np.isfinite(OOF[best]), (1 - w) * OOF[best] + w * knn, np.nan), f"{best}+knn{w}")
    evaluate(np.where(np.isfinite(OOF[best]), (1 - w) * OOF[best] + w * tree, np.nan), f"{best}+tree{w}")
np.savez(ROOT / "reports/v37_costvar.npz", **{k.replace("+", "_"): v for k, v in OOF.items()}, knn=knn, y=y)
