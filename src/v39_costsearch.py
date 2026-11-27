"""v39: explore the GP repair-cost model further. A: greedy forward search over non-histogram sensors added to the 'sensor' kernel group.
Criterion: honest 5-fold OOF budget loss on the 397 failures, weighted by the budget levels our flagged test vehicles use (v37 flags,
harness p). An addition is kept only if it cuts that loss by > 1%. Results re-checked on a second fold seed (C)."""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "2")
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from common import ROOT, load
from score import UNDER, OVER
from v33_hist_gp import GaussianSeverity, extract, select
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True); y = np.log(f.repair_cost.to_numpy()); c = np.exp(y)
HC = {f"s{i:03d}" for a in (23, 37, 54, 80, 129, 154) for i in range(a, a + 10)}
WEAR = ["s126", "s149", "s123"]
cm = xgb.XGBRegressor(n_estimators=600, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5).fit(f[S], y)
g = pd.Series(cm.get_booster().get_score(importance_type="total_gain"))
POOL = [s for s in g.sort_values(ascending=False).index if s not in HC and s not in WEAR and s not in ("s120", "s122")][:20]
BASE = extract(f)
def z(cols):
    X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
ph = np.load(ROOT / "reports/v35_harness_p.npy"); fl = pd.read_csv(ROOT / "submissions/submission_v37_bayes_gp2.csv").flag.to_numpy() == 1
p = ph[fl]; lev = np.clip(1 - (1 - UNDER / (UNDER + OVER)) / p, 0, 1); keep = lev > 0; p, lev = p[keep], lev[keep]
def oof(F, seed=42):
    o = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(y): o[b] = GaussianSeverity().fit(select(F, a), y[a]).predict(select(F, b))
    return o
def wloss(o):
    ok = np.isfinite(o); r = y[ok] - o[ok]; cc = c[ok]
    Q = np.quantile(r, lev); B = np.exp(o[ok][None, :] + Q[:, None])
    return float((p * (UNDER * np.maximum(cc[None, :] - B, 0) + OVER * np.maximum(B - cc[None, :], 0)).mean(1)).sum())
def feats(sens): return BASE + [z(WEAR)] + ([z(sens)] if sens else [])
cur = ["s049", "s116", "s142", "s105", "s010", "s104"]; best = wloss(oof(feats(cur)))
print(f"pool: {POOL}\nstart (v37 sensors {cur}): weighted budget loss {best:.1f}", flush=True)
for rnd in range(4):
    cand = [s for s in POOL if s not in cur]
    res = Parallel(n_jobs=8)(delayed(lambda s: (s, wloss(oof(feats(cur + [s])))))(s) for s in cand)
    # also try removing each current sensor
    rem = Parallel(n_jobs=8)(delayed(lambda s: ("-" + s, wloss(oof(feats([x for x in cur if x != s])))))(s) for s in cur)
    allr = sorted(res + rem, key=lambda t: t[1]); top = allr[0]
    print(f"round {rnd}: best move {top[0]} -> {top[1]:.1f} ({100 * (top[1] / best - 1):+.1f}%); next: {[(a, round(b, 1)) for a, b in allr[1:4]]}", flush=True)
    if top[1] < best * 0.99:
        cur = [x for x in cur if x != top[0][1:]] if top[0].startswith("-") else cur + [top[0]]; best = top[1]
    else: break
print("selected sensors:", cur, f"weighted loss {best:.1f}")
v37 = wloss(oof(feats(["s049", "s116", "s142", "s105", "s010", "s104"]), seed=7)); new = wloss(oof(feats(cur), seed=7))
print(f"C: second fold seed: v37 sensors {v37:.1f} vs selected {new:.1f} ({100 * (new / v37 - 1):+.1f}%)")
pd.Series(cur).to_csv(ROOT / "reports/v39_sensors.csv", index=False)
