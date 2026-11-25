"""v18: censoring-aware classifier. Healthy vehicles followed < 365 days have uncertain labels; that censoring is
predictable from sensors (AUC 0.72), so it biases the ranking. Fix: inverse-probability-of-censoring weights (IPCW):
drop censored healthy rows, weight every complete row by 1/G(t), G = Kaplan-Meier survival of the censoring time.
Evaluation must NOT use censored labels: validation = complete cases only (failures + healthy followed 365 d), IPCW-weighted."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from calib import make
from common import ROOT, load, split_holdout, folds
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
c = dev.repair_cost.fillna(0).to_numpy() * y

def km_censor(t, y, H=365):
    """G(s) = P(not censored before s); censoring event = healthy with follow-up < H."""
    cens = (y == 0) & (t < H); ts = np.unique(t[cens]); G = 1.0; out = {}
    for s in ts:
        G *= 1 - ((t == s) & cens).sum() / (t >= s).sum(); out[s] = G
    keys = np.array(list(out)); vals = np.array(list(out.values()))
    return lambda u: np.where(u <= keys[0], 1.0, vals[np.searchsorted(keys, u, side="right") - 1].clip(min=0) if len(keys) else 1.0)

complete = (y == 1) | (t >= 365)
def weights(idx):
    G = km_censor(t[idx], y[idx]); w = np.zeros(len(idx)); cm = complete[idx]
    tt = np.where(y[idx] == 1, np.minimum(t[idx], 365) - 1, 364)        # G just before the event / end of horizon
    w[cm] = 1 / G(tt[cm]); return w
po = np.load(ROOT / "reports/p_oof_v2.npy"); qd, md = np.load(ROOT / "reports/v18_dev_q.npy")
resid = np.load(ROOT / "reports/v18_resid.npy")
F = folds(y); pi = np.zeros(len(dev))
for a, b in F:
    w = weights(a); keep = w > 0
    pi[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a[keep], S], y[a[keep]], sample_weight=w[keep]).predict_proba(dev.loc[b, S])[:, 1]
np.save(ROOT / "reports/v18_p_oof_ipcw.npy", pi)
wall = weights(np.arange(len(dev)))
print(f"complete cases {complete.sum()} of {len(dev)}; IPCW weights: failures mean {wall[y==1].mean():.3f}, healthy {wall[(y==0)&complete].mean():.3f}")
res = []
for nm, raw in (("baseline (censored labels)", po), ("IPCW", pi)):
    out = []
    for a, b in F:
        ca, cb = a[complete[a]], b[complete[b]]
        qq = make("platt").fit(raw[ca], y[ca]).predict(raw[cb])          # calibrate on complete cases of the other folds
        fl, bu = decide_pv(qq, md[cb], resid); wb = wall[cb]
        cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[cb] - bu, 0) + OVER * np.maximum(bu - c[cb], 0), c[cb])
        out.append(((cost * wb).sum() / wb.sum(), roc_auc_score(y[cb], raw[cb], sample_weight=wb), average_precision_score(y[cb], raw[cb], sample_weight=wb)))
    res.append(np.array(out)); o = res[-1]
    print(f"{nm:28s} cost/vehicle {o[:,0].mean():.4f}  wAUC {o[:,1].mean():.4f}  wAP {o[:,2].mean():.4f}  folds {np.round(o[:,0],3)}")
d = res[1][:, 0] - res[0][:, 0]; print(f"IPCW - baseline per fold: {np.round(d,4)}  mean {d.mean():+.4f}  better in {(d<0).sum()}/5")
