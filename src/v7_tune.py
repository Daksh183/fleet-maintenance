"""Classifier tuning on OOF LOG-LOSS (smooth, uses every vehicle) rather than on cost (noisy, driven
by a few large failures). lr 0.02, up to 1500 trees, log-loss read at several tree counts."""
import itertools, json, time
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from common import ROOT, SEED, folds, load, split_holdout
from v2_params import PARAMS

tr, te, S = load()
dev, _ = split_holdout(tr)
y = dev.failure.to_numpy(); F = folds(y)
ref = np.load(ROOT / "reports/p_oof_v2.npy")
print(f"reference (v2 params) OOF log-loss {log_loss(y, ref):.5f}  AUC {roc_auc_score(y, ref):.4f}", flush=True)
NT = [300, 500, 800, 1100, 1500]
res = []
for depth, mcw, col in itertools.product((3, 5, 7), (1, 5), (0.5,)):
    t = time.time(); P = {n: np.zeros(len(dev)) for n in NT}
    for a, b in F:
        m = xgb.XGBClassifier(n_estimators=max(NT), max_depth=depth, learning_rate=0.02, subsample=0.8,
                              colsample_bytree=col, min_child_weight=mcw, n_jobs=-1, random_state=SEED
                              ).fit(dev.loc[a, S], y[a])
        for n in NT:
            P[n][b] = m.predict_proba(dev.loc[b, S], iteration_range=(0, n))[:, 1]
    for n in NT:
        res.append(dict(depth=depth, mcw=mcw, col=col, n=n, ll=log_loss(y, P[n]), auc=roc_auc_score(y, P[n])))
    best = min((r for r in res if r["depth"] == depth and r["mcw"] == mcw), key=lambda r: r["ll"])
    print(f"depth {depth} mcw {mcw} col {col}: best log-loss {best['ll']:.5f} at {best['n']} trees, AUC {best['auc']:.4f}  ({time.time()-t:.0f}s)", flush=True)
json.dump(res, open(ROOT / "reports/v7_tune.json", "w"), indent=1)
b = min(res, key=lambda r: r["ll"]); print("BEST", b)
