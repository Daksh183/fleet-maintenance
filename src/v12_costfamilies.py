"""v12b: other tree families / seed-averaging for the repair-cost model (failures only, 2x5 CV).
Metrics: R2 of log cost, budget (pinball @ q*) loss on cost scale, paired vs XGB_A1."""
import numpy as np, pandas as pd, xgboost as xgb, warnings; warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from common import SEED, load, split_holdout
from features import shape
from score import true_cost
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr)
f = dev[dev.failure == 1].reset_index(drop=True); X = pd.concat([f[S], shape(f)], axis=1); c = true_cost(f); lc = np.log(c)
X = X.loc[:, X.notna().any()]; Q = 0.2 / 0.22
pin = lambda y, b, a: np.mean(np.maximum(a * (y - b), (a - 1) * (y - b)))
A1 = {**CM, "n_estimators": 1000, "max_depth": 2}
class Bag:                                   # XGB A1 averaged over seeds (subsample/colsample differ per seed)
    def __init__(self, n=5): self.n = n
    def get_params(self, deep=True): return {"n": self.n}
    def set_params(self, **p): self.n = p.get("n", self.n); return self
    def fit(self, X, y): self.m = [xgb.XGBRegressor(**{**A1, "random_state": s}).fit(X, y) for s in range(self.n)]; return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.m], 0)
imp = lambda est: make_pipeline(SimpleImputer(strategy="median", add_indicator=False), est)
M = {"XGB_A1": lambda: xgb.XGBRegressor(**A1),
     "XGB_bag5": lambda: Bag(5),
     "CatBoost": lambda: CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05, verbose=0, random_seed=SEED, thread_count=4),
     "RF": lambda: imp(RandomForestRegressor(300, min_samples_leaf=2, max_features=0.33, n_jobs=-1, random_state=SEED)),
     "ExtraTrees": lambda: imp(ExtraTreesRegressor(300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=SEED))}
names = list(M) + ["Avg(XGB,Cat)", "Avg(XGB,Cat,ET)"]
res = {k: {"r2": [], "pin": []} for k in names}
for rs in (42, 1):
    for ia, ib in KFold(5, shuffle=True, random_state=rs).split(X):
        Xa, Xb = X.iloc[ia], X.iloc[ib]; P, R = {}, {}
        for k, mk in M.items():
            P[k] = mk().fit(Xa, lc[ia]).predict(Xb)
            R[k] = lc[ia] - cross_val_predict(mk(), Xa, lc[ia], cv=KFold(5, shuffle=True, random_state=SEED))
        for nm, parts in (("Avg(XGB,Cat)", ("XGB_A1", "CatBoost")), ("Avg(XGB,Cat,ET)", ("XGB_A1", "CatBoost", "ExtraTrees"))):
            P[nm] = np.mean([P[p] for p in parts], 0); R[nm] = np.mean([R[p] for p in parts], 0)
        for k in P:
            res[k]["r2"].append(r2_score(lc[ib], P[k])); res[k]["pin"].append(pin(c[ib], np.exp(P[k] + np.quantile(R[k], Q)), Q))
        print("fold done", flush=True)
b = np.array(res["XGB_A1"]["pin"])
for k, v in res.items():
    p = np.array(v["pin"]); d = p - b
    print(f"{k:16s} R2 {np.mean(v['r2']):.3f}  budget loss {p.mean():7.2f}  paired vs XGB_A1 {d.mean():+6.2f} ± {d.std(ddof=1)/np.sqrt(10):.2f}  better {int((d<0).sum())}/10", flush=True)
