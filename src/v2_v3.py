"""v2: plain XGBoost, flag if p>0.5.  v3: same model, cost-based decision rule.

OOF predictions on the 80% dev split (5 fixed folds); test predictions from a model fit on
the full training file. The hold-out is not touched here (reserved for v0, v4, v8).
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from common import SEED, folds, load, log_result, split_holdout, write_submission, fold_costs
from decide import decide_global
from score import true_cost

tr, te, S = load()
dev, _ = split_holdout(tr)
y, c = dev["failure"].to_numpy(), true_cost(dev)
F = folds(y)

PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=1, n_jobs=-1, random_state=SEED)

oof = np.zeros(len(dev))
for k, (a, b) in enumerate(F):
    m = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, S], y[a])
    oof[b] = m.predict_proba(dev.loc[b, S])[:, 1]
print(f"OOF AUC {roc_auc_score(y, oof):.4f}  PR-AUC {average_precision_score(y, oof):.4f}")

full = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr["failure"])
p_te = full.predict_proba(te[S])[:, 1]
np.save("../reports/p_te_v2.npy", p_te); np.save("../reports/p_oof_v2.npy", oof)

# ---- v2: naive threshold, budget = mean failure cost
mean_cost = float(c[c > 0].mean())
flag2, b2 = (oof > 0.5).astype(int), np.full(len(oof), mean_cost)
m2, s2 = fold_costs(c, flag2, b2, F)
log_result("v2", "XGBoost, flag if p>0.5, budget=mean cost", m2, s2,
           notes=f"AUC {roc_auc_score(y, oof):.3f}; flagged {flag2.mean():.3%}")
write_submission(te, (p_te > 0.5), np.full(len(te), mean_cost), "v2")

# ---- v3: cost-based rule with global failure-cost distribution (fold-safe: use train-fold costs)
flag3, b3 = np.zeros(len(oof), int), np.zeros(len(oof))
for a, b in F:
    fc = c[a][c[a] > 0]
    flag3[b], b3[b] = decide_global(oof[b], fc)
m3, s3 = fold_costs(c, flag3, b3, F)
log_result("v3", "same model, cost-based flag + newsvendor budget (global F)", m3, s3,
           notes=f"flagged {flag3.mean():.3%}")
fc_all = true_cost(tr)[true_cost(tr) > 0]
f_te, b_te = decide_global(p_te, fc_all)
write_submission(te, f_te, b_te, "v3")

# threshold sweep for reference (not used to pick anything yet)
for t in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5):
    mt, st = fold_costs(c, (oof > t).astype(int), np.zeros(len(oof)), F)
    print(f"  sweep p>{t}: {mt:.3f} ± {st:.3f}")
