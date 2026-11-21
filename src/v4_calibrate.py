"""v4: probability calibration (Platt vs isotonic) before the cost-based decision rule.

CV is cross-fitted: the calibrator for fold k is fit on the OOF predictions of the OTHER folds,
so no fold is calibrated on itself. Differences vs v3 are paired per fold.
First hold-out check (hold-out is touched only at v0, v4, v8).
"""
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from common import ROOT, SEED, folds, load, log_result, split_holdout, write_submission
from decide import decide_global
from score import per_vehicle, true_cost

tr, te, S = load()
dev, ho = split_holdout(tr)
y, c = dev["failure"].to_numpy(), true_cost(dev)
c_ho, y_ho = true_cost(ho), ho["failure"].to_numpy()
F = folds(y)
oof = np.load(ROOT / "reports" / "p_oof_v2.npy")      # same model/folds as v2/v3
p_te = np.load(ROOT / "reports" / "p_te_v2.npy")
EPS = 1e-6
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def make(kind):
    if kind == "platt":
        class Platt:
            def fit(self, p, y):
                self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y); return self
            def predict(self, p):
                return self.m.predict_proba(logit(p)[:, None])[:, 1]
        return Platt()
    class Iso:
        def fit(self, p, y):
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y); return self
        def predict(self, p):
            return self.m.predict(p)
    return Iso()


def cv_costs(kind):
    """Per-fold cost/vehicle + calibrated OOF probabilities (cross-fitted)."""
    pc, costs = oof.copy(), []
    for k, (a, b) in enumerate(F):
        if kind != "raw":
            pc[b] = make(kind).fit(oof[a], y[a]).predict(oof[b])
        fl, bu = decide_global(pc[b], c[a][c[a] > 0])
        costs.append(per_vehicle(c[b], fl, bu))
    return np.array(costs), pc


res = {k: cv_costs(k) for k in ("raw", "platt", "iso")}
base = res["raw"][0]
print("\nreliability (OOF, quantile bins of raw p): mean predicted vs observed failure rate")
edges = np.quantile(oof, [0, .5, .8, .9, .95, .98, .99, 1.0])
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (oof >= lo) & (oof <= hi)
    print(f"  p in [{lo:.4f},{hi:.4f}]  n={m.sum():5d}  pred {oof[m].mean():.4f}  obs {y[m].mean():.4f}  "
          f"platt {res['platt'][1][m].mean():.4f}  iso {res['iso'][1][m].mean():.4f}")

print("\nmethod   CV cost    paired Δ vs raw (per fold)                  folds improved  Brier   logloss")
for k, (costs, pc) in res.items():
    d = costs - base
    print(f"{k:6s}  {costs.mean():.4f}±{costs.std():.3f}  {np.round(d, 3)}  {int((d < 0).sum())}/5"
          f"  {brier_score_loss(y, pc):.5f}  {log_loss(y, np.clip(pc, EPS, 1 - EPS)):.5f}")
    if k != "raw":
        log_result("v4", f"{k} calibration before cost rule", costs.mean(), costs.std(),
                   notes=f"paired delta vs v3 {d.mean():+.3f} ± {d.std():.3f}; improved {(d<0).sum()}/5 folds")

# ---- hold-out check: model fit on dev only, calibrator fit on all dev OOF, costs from dev failures
P = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=1, n_jobs=-1, random_state=SEED)
p_ho = xgb.XGBClassifier(**P).fit(dev[S], y).predict_proba(ho[S])[:, 1]
fc_dev = c[c > 0]
print(f"\nHOLD-OUT (n={len(ho)}, failures={y_ho.sum()}): cost per vehicle")
ho_cost = {}
for k in ("raw", "platt", "iso"):
    pc = p_ho if k == "raw" else make(k).fit(oof, y).predict(p_ho)
    fl, bu = decide_global(pc, fc_dev)
    ho_cost[k] = per_vehicle(c_ho, fl, bu)
    print(f"  {k:6s} {ho_cost[k]:.4f}   flagged {fl.mean():.3%}")
log_result("v4", "hold-out check: raw / platt / iso", ho_cost["raw"], 0.0,
           holdout=min(ho_cost["platt"], ho_cost["iso"]),
           notes=f"holdout raw {ho_cost['raw']:.3f} platt {ho_cost['platt']:.3f} iso {ho_cost['iso']:.3f}")

# ---- submissions: calibrator fit on all dev OOF, applied to the test probabilities
fc_all = true_cost(tr)[true_cost(tr) > 0]
print("\ntest: mean p raw %.4f" % p_te.mean())
for k in ("platt", "iso"):
    pt = make(k).fit(oof, y).predict(p_te)
    fl, bu = decide_global(pt, fc_all)
    print(f"  {k}: mean p {pt.mean():.4f}", end="  ")
    write_submission(te, fl, bu, f"v4_{k}")
