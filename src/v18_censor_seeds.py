import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import v18_censor as V          # reuses data, weights() and the fold-0 results printed on import
from v2_params import PARAMS
dev, y, S, complete = V.dev, V.y, V.S, V.complete
for seed in (1, 2, 3):
    Fs = list(StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(np.zeros(len(y)), y))
    pb, pi = np.zeros(len(y)), np.zeros(len(y)); prm = {**PARAMS, "random_state": seed}
    for a, b in Fs:
        pb[b] = xgb.XGBClassifier(**prm).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
        w = V.weights(a); k = w > 0
        pi[b] = xgb.XGBClassifier(**prm).fit(dev.loc[a[k], S], y[a[k]], sample_weight=w[k]).predict_proba(dev.loc[b, S])[:, 1]
    r = []
    for a, b in Fs:
        cb = b[complete[b]]; wb = V.wall[cb]
        r.append([average_precision_score(y[cb], pb[cb], sample_weight=wb), average_precision_score(y[cb], pi[cb], sample_weight=wb),
                  roc_auc_score(y[cb], pb[cb], sample_weight=wb), roc_auc_score(y[cb], pi[cb], sample_weight=wb)])
    r = np.array(r); print(f"seed {seed}: wAP base {r[:,0].mean():.4f} ipcw {r[:,1].mean():.4f} (better {int((r[:,1]>r[:,0]).sum())}/5) | wAUC base {r[:,2].mean():.4f} ipcw {r[:,3].mean():.4f} (better {int((r[:,3]>r[:,2]).sum())}/5)", flush=True)
