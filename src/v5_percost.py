"""v5: per-vehicle repair-cost model (XGB on log cost, failures only) -> per-vehicle budget + flag.

Ablation per probability variant (raw / platt / iso), all paired by fold against global-F:
  A  global F for flag and budget          (v3 / v4)
  B  per-vehicle F for the flag decision, global-F budget
  D  per-vehicle F for flag and budget     (full v5)
Nested: for each outer fold the cost model and its residuals use only that fold's training failures
(residuals come from inner cross-validation, so they are out-of-sample).
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

from calib import make
from common import ROOT, SEED, folds, load, log_result, split_holdout, write_submission
from decide import Q_STAR, decide_global, decide_pv
from score import per_vehicle, true_cost

tr, te, S = load()
dev, _ = split_holdout(tr)
y, c = dev["failure"].to_numpy(), true_cost(dev)
F = folds(y)
oof = np.load(ROOT / "reports" / "p_oof_v2.npy")
p_te_raw = np.load(ROOT / "reports" / "p_te_v2.npy")

CM = dict(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
          min_child_weight=5, n_jobs=-1, random_state=SEED)


def fit_cost_model(X, logc):
    """Returns fitted model and OUT-OF-FOLD log residuals from 5-fold CV on these failures."""
    inner = cross_val_predict(xgb.XGBRegressor(**CM), X, logc, cv=KFold(5, shuffle=True, random_state=SEED))
    return xgb.XGBRegressor(**CM).fit(X, logc), logc - inner


# ---- nested CV
variants = ("raw", "platt", "iso")
cost = {(v, m): [] for v in variants for m in "ABD"}
cover, r2s = [], []
for a, b in F:
    fa = a[y[a] == 1]
    model, resid = fit_cost_model(dev.loc[fa, S], np.log(c[fa]))
    m_b = model.predict(dev.loc[b, S])
    fc = c[a][c[a] > 0]
    fb = b[y[b] == 1]
    if len(fb):
        mf = model.predict(dev.loc[fb, S])
        r2s.append(r2_score(np.log(c[fb]), mf))
        cover.append(np.mean(np.log(c[fb]) <= mf + np.quantile(resid, Q_STAR)))   # should be ~0.909
    for v in variants:
        p = oof[b] if v == "raw" else make(v).fit(oof[a], y[a]).predict(oof[b])
        fl, bu = decide_global(p, fc);               cost[(v, "A")].append(per_vehicle(c[b], fl, bu))
        fl, bu = decide_pv(p, m_b, resid, fc);       cost[(v, "B")].append(per_vehicle(c[b], fl, bu))
        fl, bu = decide_pv(p, m_b, resid);           cost[(v, "D")].append(per_vehicle(c[b], fl, bu))

print(f"cost model held-out-fold R2 (log scale): {np.round(r2s, 2)}  mean {np.mean(r2s):.2f}")
print(f"budget-quantile coverage on held-out failing vehicles (target {Q_STAR:.3f}): {np.round(cover, 2)}  mean {np.mean(cover):.3f}")
print("\nvariant  rule   CV cost/veh        paired Δ vs A (per fold)                 improved")
for v in variants:
    A = np.array(cost[(v, "A")])
    for m in "ABD":
        x = np.array(cost[(v, m)]); d = x - A
        print(f"{v:6s}   {m}     {x.mean():.4f}±{x.std():.3f}   {np.round(d, 3)}   {int((d < 0).sum())}/5")
    log_result("v5", f"per-vehicle cost model, p={v} (full: flag+budget)", np.mean(cost[(v, 'D')]), np.std(cost[(v, 'D')]),
               notes=(f"A(global) {np.mean(A):.3f}; B(pv flag only) {np.mean(cost[(v,'B')]):.3f}; "
                      f"D {np.mean(cost[(v,'D')]):.3f}; paired D-A {np.mean(np.array(cost[(v,'D')])-A):+.3f}"))

# ---- test submissions: cost model on ALL training failures; calibrators on all dev OOF
f_all = tr[tr.failure == 1]
model, resid = fit_cost_model(f_all[S], np.log(f_all.repair_cost.to_numpy()))
m_te = model.predict(te[S])
print(f"\ntest predicted failure cost: median {np.median(np.exp(m_te)):.0f}  mean {np.exp(m_te).mean():.0f}"
      f"   (train failures: median {f_all.repair_cost.median():.0f}  mean {f_all.repair_cost.mean():.0f})")
for v in ("raw", "platt", "iso"):
    pt = p_te_raw if v == "raw" else make(v).fit(oof, y).predict(p_te_raw)
    fl, bu = decide_pv(pt, m_te, resid)
    print(f"  {v}: ", end="")
    write_submission(te, fl, bu, f"v5_{v}")
