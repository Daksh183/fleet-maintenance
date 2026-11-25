import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v18_censor import km_censor
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
c = dev.repair_cost.fillna(0).to_numpy() * y; complete = (y == 1) | (t >= 365)
G = km_censor(t, y); W = np.zeros(len(y)); W[complete] = 1 / G(np.where(y == 1, np.minimum(t, 365) - 1, 364)[complete])
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
P = np.mean([np.load(ROOT / f"reports/v18_search/{k}.npy") for k in ["ipcw", "ipcw_nodrift", "ipcw_d6_mcw5", "ipcw_slow"]], axis=0)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
def cal(kind, pa, ya, wa, pb):
    if kind == "platt": return LogisticRegression(C=1e6, max_iter=1000).fit(lg(pa), ya).predict_proba(lg(pb))[:, 1]
    if kind == "platt_w": return LogisticRegression(C=1e6, max_iter=1000).fit(lg(pa), ya, sample_weight=wa).predict_proba(lg(pb))[:, 1]
    if kind == "iso_w": return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pa, ya, sample_weight=wa).predict(pb)
    if kind == "platt2_w":                                    # quadratic in logit (more flexible, still smooth)
        f = lambda p: np.hstack([lg(p), lg(p) ** 2]); return LogisticRegression(C=1e4, max_iter=2000).fit(f(pa), ya, sample_weight=wa).predict_proba(f(pb))[:, 1]
R = {}
for kind in ("platt", "platt_w", "iso_w", "platt2_w"):
    out = []
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            ca, cb = a[complete[a]], b[complete[b]]
            qq = cal(kind, P[r][ca], y[ca], W[ca], P[r][cb]); fl, bu = decide_pv(qq, md[cb], resid); wb = W[cb]
            cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[cb] - bu, 0) + OVER * np.maximum(bu - c[cb], 0), c[cb])
            out.append((cost * wb).sum() / wb.sum())
    R[kind] = np.array(out); d = R[kind] - R["platt"]
    print(f"{kind:9s} {R[kind].mean():.4f}  vs platt {d.mean():+.4f}±{d.std()/np.sqrt(15):.4f}  better {(d<0).sum()}/15")
