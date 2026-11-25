"""v18 diagnosis: where does v12_robust lose cost? (a) model-expected decomposition on test, (b) realised decomposition in dev CV."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout, folds
from decide import decide_pv
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; DRIFT = ["s120", "s122", "s126", "s128"]
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
po = np.load(ROOT / "reports/p_oof_v2.npy"); q = make("platt").fit(po, y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); p = sum(w * shift(q, k) for w, k in zip(W, GRID))
fl, bu = decide_pv(p, m, resid)
K = 99; M = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :]); mu = M.mean(1)
short = (UNDER * np.maximum(M - bu[:, None], 0)).mean(1); over_f = (OVER * np.maximum(bu[:, None] - M, 0)).mean(1)
F = fl == 1
parts = {"inspection (10 x flagged)": INSPECT * F.sum(), "budget wasted on flagged non-failures": ((1 - p) * OVER * bu)[F].sum(),
         "shortage on flagged failures": (p * short)[F].sum(), "overage on flagged failures": (p * over_f)[F].sum(),
         "missed failures (not flagged)": (p * mu)[~F].sum()}
tot = sum(parts.values())
print(f"TEST, model-expected (posterior p), scaled to the public 30%: total {0.3*tot:,.0f}  (actual public 16,840)")
for k_, v in parts.items(): print(f"   {k_:40s} {0.3*v:9,.0f}  {v/tot:6.1%}")
print(f"   expected failures in test {p.sum():.0f}; among flagged {p[F].sum():.0f} ({p[F].sum()/p.sum():.1%}); flags {F.sum()}")
print(f"   expected cost of failures: all {(p*mu).sum()*0.3:,.0f} (flag-nobody), oracle ~{0.3*p.sum()*INSPECT:,.0f}")
# missed-failure cost by p band
bands = [0, .005, .01, .02, .03, .05, .1, 1]
for a, b in zip(bands[:-1], bands[1:]):
    s = (~F) & (p >= a) & (p < b); s2 = F & (p >= a) & (p < b)
    print(f"   p in [{a:.3f},{b:.3f}): unflagged {s.sum():6d} exp missed cost {0.3*(p*mu)[s].sum():7,.0f} | flagged {s2.sum():5d}")
# --- dev CV realised decomposition (no shift): same pipeline, Platt on OOF
c = dev.repair_cost.fillna(0).to_numpy() * y
qd = np.zeros(len(dev)); md = np.zeros(len(dev)); Fo = folds(y)
for a, b in Fo:
    qd[b] = make("platt").fit(po[a], y[a]).predict(po[b])
    fa = dev.iloc[a][y[a] == 1]; Xf = pd.concat([fa[keep], shape(fa)], axis=1)
    md[b] = xgb.XGBRegressor(**CM9).fit(Xf, np.log(fa.repair_cost.to_numpy())).predict(pd.concat([dev.iloc[b][keep], shape(dev.iloc[b])], axis=1))
fd, bd = decide_pv(qd, md, resid)
Fd = fd == 1; fail = y == 1
cd = {"inspection": INSPECT * Fd.sum(), "budget on flagged non-failures": (OVER * bd)[Fd & ~fail].sum(),
      "shortage flagged failures": (UNDER * np.maximum(c - bd, 0))[Fd & fail].sum(), "overage flagged failures": (OVER * np.maximum(bd - c, 0))[Fd & fail].sum(),
      "missed failures": c[~Fd & fail].sum()}
T = sum(cd.values()); print(f"\nDEV CV realised: {T/len(dev):.3f}/vehicle; failures {fail.sum()}, caught {int((Fd&fail).sum())}, flags {Fd.sum()}")
for k_, v in cd.items(): print(f"   {k_:35s} {v/len(dev):.3f}  {v/T:6.1%}")
mf = ~Fd & fail
print("   missed failures: n", mf.sum(), " their p:", np.round(np.sort(qd[mf])[::-1][:15], 3), " costs:", np.round(np.sort(c[mf])[::-1][:10]))
print(f"   missed cost share from top-10 missed: {np.sort(c[mf])[::-1][:10].sum()/c[mf].sum():.1%}; median p of missed {np.median(qd[mf]):.4f}")
np.save(ROOT / "reports/v18_dev_q.npy", np.vstack([qd, md]))
