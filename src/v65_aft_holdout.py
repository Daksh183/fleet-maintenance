"""v65: (1) independent check on the locked 20% hold-out (not used by v60-v64): fit on dev, score the hold-out. AFT ensemble =
logit average of the Platt-calibrated members (scale 1, scale 2, depth 6, second seed), each also bagged over 3 seeds.
(2) test-set probabilities from fits on all train -> reports/v65_q.npz."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.model_selection import KFold
from calib import make
from common import ROOT, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
import v60_feats as F, v63_survival as V6
tr, te, S, NO = F.tr, F.te, F.S, F.NO; dev, ho = split_holdout(tr); y = dev.failure.to_numpy(); yh = ho.failure.to_numpy()
MEM = {"aft s1": {}, "aft s2": dict(scale=2.0), "aft d6": dict(max_depth=6, min_child_weight=5), "aft seed7": dict(seed=7)}
FILES = {"aft s1": "aft_s1", "aft s2": "aft_s2", "aft d6": "aft_s1_d6", "aft seed7": "aft_s1_seed_7"}
def aft(D, E, scale=1.0, seed=42, **kw):
    t = np.maximum(D.followup_days.to_numpy(float), 0.5); e = D.failure.to_numpy()
    d = xgb.DMatrix(D[NO]); d.set_float_info("label_lower_bound", t); d.set_float_info("label_upper_bound", np.where(e == 1, t, np.inf))
    m = xgb.train({**V6.P0, "seed": seed, "objective": "survival:aft", "aft_loss_distribution": "normal", "aft_loss_distribution_scale": scale, "base_score": 365.0, **kw}, d, 1000)
    return 1 / (1 + np.exp(-np.clip(-m.predict(xgb.DMatrix(E[NO]), output_margin=True) / scale + 6, -13, 13)))
def clf(D, E, cols, seed=42): return xgb.XGBClassifier(**{**F.SLOW, "random_state": seed}).fit(D[cols], D.failure).predict_proba(E[cols])[:, 1]
SEEDS = (42, 1, 2)
def run(name, which):
    D, E = (dev, ho) if which == "ho" else (tr, te)
    if name.startswith("clf"): return np.mean([clf(D, E, NO if name == "clf base" else S, s) for s in SEEDS], 0)
    kw = dict(MEM[name]); base = kw.pop("seed", 42)
    return np.mean([aft(D, E, seed=base + 10 * i, **kw) for i in range(3)], 0)
L = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
sig = lambda v: 1 / (1 + np.exp(-v))
if __name__ == "__main__":
    names = ["clf base", "clf all"] + list(MEM); jobs = [(n, w) for w in ("ho", "te") for n in names]
    R = dict(zip(jobs, Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs)))
    oof = {n: np.load(ROOT / f"reports/v18_search/v64_{n.replace(' ', '_') if n.startswith('clf') else FILES[n]}.npy")[0] for n in names}
    cal = {n: make("platt").fit(oof[n], y) for n in names}
    Q = {w: {n: cal[n].predict(R[(n, w)]) for n in names} for w in ("ho", "te")}
    for w in Q:
        Q[w]["aft ens"] = sig(np.mean([L(Q[w][n]) for n in MEM], 0))
        Q[w]["clf + aft"] = sig((L(Q[w]["clf base"]) + L(Q[w]["aft ens"])) / 2)
    # hold-out judge: v12-style tree cost model fitted on dev failures; flags at each model's own best level
    NODR = [s for s in S if s not in ("s120", "s122", "s126", "s128")]; fa = dev[y == 1]; ly = np.log(fa.repair_cost.to_numpy())
    cm = dict(n_estimators=1000, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, random_state=42, n_jobs=4)
    o = np.zeros(len(ly))
    for a, b in KFold(5, shuffle=True, random_state=42).split(ly): o[b] = xgb.XGBRegressor(**cm).fit(fa[NODR].iloc[a], ly[a]).predict(fa[NODR].iloc[b])
    mh = xgb.XGBRegressor(**cm).fit(fa[NODR], ly).predict(ho[NODR]); res = ly - o; ch = ho.repair_cost.fillna(0).to_numpy() * yh
    e = np.nanquantile(np.log1p(tr.s126), np.linspace(0, 1, 21)); e[0], e[-1] = -np.inf, np.inf
    h = lambda v: np.histogram(np.log1p(v[np.isfinite(v)]), e)[0] / np.isfinite(v).sum()
    ratio = np.clip(h(te.s126.to_numpy()) / np.maximum(h(tr.s126.to_numpy()), 1e-9), 0, 5)
    W = ratio[np.clip(np.searchsorted(e, np.log1p(ho.s126.fillna(dev.s126.median()).to_numpy()), side="right") - 1, 0, 19)]
    LAM = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6)
    def per(q, lam):
        qq = lam * q / (1 - q) / (1 + lam * q / (1 - q)); fl, bu = decide_pv(qq, mh, res)
        return np.where(fl == 1, INSPECT + UNDER * np.maximum(ch - bu, 0) + OVER * np.maximum(bu - ch, 0), ch)
    rng = np.random.default_rng(0); I = rng.integers(0, len(yh), (1000, len(yh))); rows = []; V = {}
    for n, q in Q["ho"].items():
        costs = [per(q, l) for l in LAM]; wsum = [(c * W).sum() / W.sum() for c in costs]; k = int(np.argmin(wsum)); V[n] = costs[k] * W
        rows.append(dict(model=n, AP=average_precision_score(yh, q), AUC=roc_auc_score(yh, q), wAP=average_precision_score(yh, q, sample_weight=W),
                         logloss=log_loss(yh, np.clip(q, 1e-7, 1 - 1e-7)), wear_cost=min(wsum), plain_cost=min(c.mean() for c in costs), cost_lam1=per(q, 1.0).mean()))
    O = pd.DataFrame(rows); print(f"hold-out: {len(ho)} vehicles, {int(yh.sum())} failures"); print(O.round(4).to_string(index=False))
    base = V["clf base"]
    for n in V:
        d = V[n] - base; b = d[I].sum(1) / base[I].sum(1); print(f"{n:10s} wear-cost vs clf base {100 * d.sum() / base.sum():+.2f}%  bootstrap p(better) {(b < 0).mean():.2f}")
    np.savez(ROOT / "reports/v65_q.npz", **{"aft": Q["te"]["aft ens"], "aftd6": Q["te"]["aft d6"], "clfaft": Q["te"]["clf + aft"]})
    qv12, _ = np.load(ROOT / "reports/v18_qm.npy")
    for n in ("aft ens", "aft d6", "clf + aft", "clf base"):
        q = Q["te"][n]; print(f"test {n:10s} mean q {q.mean():.4f}; Spearman with v12 {pd.Series(q).corr(pd.Series(qv12), method='spearman'):.3f}; top-1433 overlap with v12 {len(set(np.argsort(-q)[:1433]) & set(np.argsort(-qv12)[:1433]))}")
