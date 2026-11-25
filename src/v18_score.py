import sys, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from calib import make
from common import ROOT, load, split_holdout
from decide import decide_pv
from score import INSPECT, OVER, UNDER
from v18_censor import km_censor
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); t = dev.followup_days.to_numpy()
c = dev.repair_cost.fillna(0).to_numpy() * y; complete = (y == 1) | (t >= 365)
G = km_censor(t, y); W = np.zeros(len(y)); W[complete] = 1 / G(np.where(y == 1, np.minimum(t, 365) - 1, 364)[complete])
qd, md = np.load(ROOT / "reports/v18_dev_q.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
D = ROOT / "reports/v18_search"; P = {p.stem: np.load(p) for p in sorted(D.glob("*.npy")) if not p.stem.startswith("cost_")}
rk = lambda A: np.vstack([rankdata(a) / len(a) for a in A])
for spec in sys.argv[1:]:                                          # blends: name=a+b+c (rank average)
    if "~" in spec: nm, parts = spec.split("~"); P[nm] = np.mean([P[p] for p in parts.split("+")], axis=0)     # probability average
    else: nm, parts = spec.split("="); P[nm] = np.mean([rk(P[p]) for p in parts.split("+")], axis=0)
def metrics(Pm):
    out = []
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            ca, cb = a[complete[a]], b[complete[b]]; wb = W[cb]; p = Pm[r]
            qq = make("platt").fit(p[ca], y[ca]).predict(p[cb]); fl, bu = decide_pv(qq, md[cb], resid)
            cost = np.where(fl == 1, INSPECT + UNDER * np.maximum(c[cb] - bu, 0) + OVER * np.maximum(bu - c[cb], 0), c[cb])
            top = p[cb] >= np.quantile(p[cb], 1 - 0.088)
            out.append([average_precision_score(y[cb], p[cb], sample_weight=wb), roc_auc_score(y[cb], p[cb], sample_weight=wb),
                        (c[cb] * wb)[top].sum() / (c[cb] * wb).sum(), (cost * wb).sum() / wb.sum()])
    return np.array(out)
R = {k: metrics(v) for k, v in P.items()}; ref = R.get("ipcw"); base = R.get("base_cens")
print(f"{'candidate':24s} {'wAP':>7s} {'wAUC':>7s} {'capt':>6s} {'cost':>7s}  {'dcost vs base':>14s} {'better':>6s}  {'dcost vs ipcw':>14s} {'better':>6s}")
for k, v in sorted(R.items(), key=lambda kv: kv[1][:, 3].mean()):
    d1 = v[:, 3] - base[:, 3]; d2 = v[:, 3] - ref[:, 3]
    print(f"{k:24s} {v[:,0].mean():7.4f} {v[:,1].mean():7.4f} {v[:,2].mean():6.3f} {v[:,3].mean():7.4f}  {d1.mean():+7.4f}±{d1.std()/np.sqrt(15):.4f} {(d1<0).sum():3d}/15  {d2.mean():+7.4f}±{d2.std()/np.sqrt(15):.4f} {(d2<0).sum():3d}/15")
