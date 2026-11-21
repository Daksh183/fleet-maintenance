"""v3b: v3 with the four most-drifted sensors removed (s120, s122, s126, s128). Diagnostic for depot shift."""
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from common import SEED, folds, load, log_result, split_holdout, write_submission, fold_costs
from decide import decide_global
from score import true_cost
tr, te, S = load()
S = [s for s in S if s not in ("s120", "s122", "s126", "s128")]
dev, _ = split_holdout(tr); y, c = dev["failure"].to_numpy(), true_cost(dev); F = folds(y)
P = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
oof = np.zeros(len(dev))
for a, b in F:
    oof[b] = xgb.XGBClassifier(**P).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
p_te = xgb.XGBClassifier(**P).fit(tr[S], tr["failure"]).predict_proba(te[S])[:, 1]
f3, b3 = np.zeros(len(oof), int), np.zeros(len(oof))
for a, b in F:
    f3[b], b3[b] = decide_global(oof[b], c[a][c[a] > 0])
m, s = fold_costs(c, f3, b3, F)
print(f"AUC {roc_auc_score(y, oof):.4f}; mean p oof {oof.mean():.4f} test {p_te.mean():.4f}")
log_result("v3b", "v3 minus drifted sensors s120/s122/s126/s128", m, s, notes=f"mean p test {p_te.mean():.4f}")
fc = true_cost(tr)[true_cost(tr) > 0]; ft, bt = decide_global(p_te, fc); write_submission(te, ft, bt, "v3b")
