"""Design of tomorrow's uploads: expected public-score DIFFERENCES between files under each true k, and
their noise (Monte Carlo over the random public 30% subset and over which vehicles fail).
Belief: Platt p shifted by k_true; per-vehicle cost draws from the v8 shape cost model."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, split_holdout
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xf, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM), Xf, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM).fit(Xf, lc).predict(Xt)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
sub = {n: pd.read_csv(ROOT / f"submissions/submission_{n}.csv") for n in ("v5_platt", "v8_k1", "v8_k3", "v8_k4", "v8_em")}
F = {n: s.flag.to_numpy() for n, s in sub.items()}; B = {n: s.budget.to_numpy() for n, s in sub.items()}
pairs = [("v5_platt", "v8_k1"), ("v8_k1", "v8_k4"), ("v8_k4", "v8_em"), ("v8_k3", "v8_k4")]
for a, b in pairs:
    print(f"{a} -> {b}: vehicles that differ: flag {int((F[a] != F[b]).sum())}, budget {int((np.abs(B[a] - B[b]) > 0.5).sum())}")
rng = np.random.default_rng(0); R = 2000; n = len(te)
def veh_cost(fl, b, c): return np.where(fl == 1, INSPECT + UNDER * np.maximum(c - b, 0) + OVER * np.maximum(b - c, 0), c)
rows = []
for kt in (1, 2, 3, 4, 6.5, 8):
    p = shift(q, kt); D = {pr: [] for pr in pairs}
    for _ in range(R):
        fail = rng.random(n) < p
        c = np.where(fail, np.exp(m + rng.choice(resid, n)), 0.0)
        pub = rng.random(n) < 0.3
        tot = {k: veh_cost(F[k][pub], B[k][pub], c[pub]).sum() for k in F}
        for a, b in pairs: D[(a, b)].append(tot[b] - tot[a])
    for pr in pairs:
        d = np.array(D[pr]); rows.append(dict(k_true=kt, pair=f"{pr[0]}->{pr[1]}", mean=d.mean(), sd=d.std(), p_improves=np.mean(d < 0)))
t = pd.DataFrame(rows)
print("\nexpected change in public score (negative = better), mean ± sd, P(improves):")
print(t.pivot(index="pair", columns="k_true", values="mean").round(-1).astype(int).to_string())
print(t.pivot(index="pair", columns="k_true", values="sd").round(-1).astype(int).add_prefix("sd k=").to_string())
print(t.pivot(index="pair", columns="k_true", values="p_improves").round(2).add_prefix("P k=").to_string())
