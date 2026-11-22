"""v7 build: tuned + seed-averaged classifier, prior-shift (EM) correction, three submissions (2x2 with v5).

              classifier v2      classifier tuned
  k = 1       v5_platt (22,969)  (CV only)
  k = EM      v7a_em             v7b_tuned_em
  k = 4       -                  v7c_tuned_k4   (minimax-regret hedge over k in [1, 10])
"""
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

from calib import make
from common import ROOT, SEED, folds, load, log_result, split_holdout, write_submission
from decide import decide_pv
from score import per_vehicle, true_cost
from v2_params import PARAMS
from v5_percost_cfg import CM

tr, te, S = load()
dev, _ = split_holdout(tr)
y, c = dev.failure.to_numpy(), true_cost(dev); F = folds(y)
SEEDS = (42, 7, 2024)
best = min(json.load(open(ROOT / "reports/v7_tune.json")), key=lambda r: r["ll"])
TUNED = dict(n_estimators=best["n"], max_depth=best["depth"], learning_rate=0.02, subsample=0.8,
             colsample_bytree=best["col"], min_child_weight=best["mcw"], n_jobs=-1)
print("tuned params:", TUNED)

shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))


def em_k(p, pi_tr, it=300):
    pi = pi_tr
    for _ in range(it):
        a = pi / pi_tr * p
        pi = (a / (a + (1 - pi) / (1 - pi_tr) * (1 - p))).mean()
    return (pi / (1 - pi)) / (pi_tr / (1 - pi_tr))


def oof_and_test(params, cols, seeds):
    o, t = np.zeros(len(dev)), np.zeros(len(te))
    for s in seeds:
        for a, b in F:
            o[b] += xgb.XGBClassifier(**params, random_state=s).fit(dev.loc[a, cols], y[a]).predict_proba(dev.loc[b, cols])[:, 1] / len(seeds)
        t += xgb.XGBClassifier(**params, random_state=s).fit(tr[cols], tr.failure).predict_proba(te[cols])[:, 1] / len(seeds)
    return o, t


P0 = {k: v for k, v in PARAMS.items() if k != "random_state"}
oof_v2, te_v2 = np.load(ROOT / "reports/p_oof_v2.npy"), np.load(ROOT / "reports/p_te_v2.npy")
oof_t1, te_t1 = oof_and_test(TUNED, S, (42,))
oof_t3, te_t3 = oof_and_test(TUNED, S, SEEDS)
np.save(ROOT / "reports/p_oof_v7.npy", oof_t3); np.save(ROOT / "reports/p_te_v7.npy", te_t3)
drift = ["s120", "s122", "s126", "s128"]
oof_nd, te_nd = oof_and_test(P0, [s for s in S if s not in drift], (42,))
for n, o in (("v2", oof_v2), ("tuned 1 seed", oof_t1), ("tuned 3 seeds", oof_t3), ("v2 no-drift", oof_nd)):
    print(f"{n:14s} OOF log-loss {log_loss(y, o):.5f}  AUC {roc_auc_score(y, o):.4f}")

# ---- dev CV cost, k = 1 (no shift on dev), v5 pipeline, paired vs v2 classifier
cost = {n: [] for n in ("v2", "tuned1", "tuned3")}
for a, b in F:
    fa = a[y[a] == 1]; lc = np.log(c[fa])
    resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), dev.loc[fa, S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
    m_b = xgb.XGBRegressor(**CM).fit(dev.loc[fa, S], lc).predict(dev.loc[b, S])
    for n, o in (("v2", oof_v2), ("tuned1", oof_t1), ("tuned3", oof_t3)):
        fl, bu = decide_pv(make("platt").fit(o[a], y[a]).predict(o[b]), m_b, resid)
        cost[n].append(per_vehicle(c[b], fl, bu))
A = np.array(cost["v2"])
for n in cost:
    x = np.array(cost[n]); d = x - A
    print(f"CV cost {n:7s} {x.mean():.4f}±{x.std():.3f}  paired vs v2 {np.round(d, 3)}  better {int((d < 0).sum())}/5")
log_result("v7", f"tuned XGB (depth {best['depth']}, {best['n']} trees, lr .02) x3 seeds, Platt, v5 cost, k=1",
           np.mean(cost["tuned3"]), np.std(cost["tuned3"]),
           notes=f"paired vs v5 platt {np.mean(np.array(cost['tuned3']) - A):+.3f}; log-loss {log_loss(y, oof_t3):.5f} vs {log_loss(y, oof_v2):.5f}")

# ---- test: EM prior for each classifier (robustness), then submissions
pi = y.mean()
q = {}
for n, o, t in (("v2", oof_v2, te_v2), ("tuned", oof_t3, te_t3), ("no-drift", oof_nd, te_nd)):
    q[n] = make("platt").fit(o, y).predict(t)
    print(f"EM odds factor on test, {n:8s} classifier: k = {em_k(q[n], pi):.2f}   (mean Platt p {q[n].mean():.4f})")
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), f[S], lc, cv=KFold(5, shuffle=True, random_state=SEED))
m_te = xgb.XGBRegressor(**CM).fit(f[S], lc).predict(te[S])
k_v2, k_t = em_k(q["v2"], pi), em_k(q["tuned"], pi)
for name, qq, k in (("v7a_em", q["v2"], k_v2), ("v7b_tuned_em", q["tuned"], k_t), ("v7c_tuned_k4", q["tuned"], 4.0)):
    fl, bu = decide_pv(shift(qq, k), m_te, resid)
    print(f"{name} (k={k:.2f}): ", end=""); write_submission(te, fl, bu, name)
    log_result("v7", f"{name}: prior shift k={k:.2f} on test", np.nan, np.nan,
               notes=f"test flag rate {fl.mean():.3%}; CV not applicable (shift is test-only)")
