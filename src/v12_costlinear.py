"""v12a: is the repair-cost function simple enough that a sparse/additive model beats boosted trees with ~318 failures?
Failures only, 3x5 CV. Metrics: R2 of log cost (out-of-fold), pinball loss at q*=.909 of the budget (cost scale), paired vs XGB A1."""
import numpy as np, pandas as pd, xgboost as xgb, warnings; warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
lin = lambda est: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)
M = {"XGB_A1": lambda: xgb.XGBRegressor(**{**CM, "n_estimators": 1000, "max_depth": 2}),
     "XGB_d1": lambda: xgb.XGBRegressor(**{**CM, "n_estimators": 1500, "max_depth": 1}),
     "Lasso": lambda: lin(LassoCV(cv=5, n_alphas=30, eps=1e-2, max_iter=3000, tol=1e-3, random_state=SEED)),
     "Ridge": lambda: lin(RidgeCV(alphas=np.logspace(0, 4, 20)))}
res = {k: {"r2": [], "pin": []} for k in list(M) + ["Avg(XGB,Lasso)"]}
for rs in (42, 1):
    for ia, ib in KFold(5, shuffle=True, random_state=rs).split(X):
        Xa, Xb = X.iloc[ia], X.iloc[ib]; P, R = {}, {}
        for k, mk in M.items():
            P[k] = mk().fit(Xa, lc[ia]).predict(Xb)
            R[k] = lc[ia] - cross_val_predict(mk(), Xa, lc[ia], cv=KFold(5, shuffle=True, random_state=SEED))
        P["Avg(XGB,Lasso)"] = (P["XGB_A1"] + P["Lasso"]) / 2; R["Avg(XGB,Lasso)"] = (R["XGB_A1"] + R["Lasso"]) / 2
        for k in P:
            res[k]["r2"].append(r2_score(lc[ib], P[k])); res[k]["pin"].append(pin(c[ib], np.exp(P[k] + np.quantile(R[k], Q)), Q))
        print("fold done", flush=True)
    print("repeat", rs, flush=True)
b = np.array(res["XGB_A1"]["pin"])
for k, v in res.items():
    p = np.array(v["pin"]); d = p - b
    print(f"{k:15s} R2 {np.mean(v['r2']):.3f}  budget loss {p.mean():7.2f}  paired vs XGB_A1 {d.mean():+6.2f} ± {d.std(ddof=1)/np.sqrt(10):.2f}  better {int((d<0).sum())}/10")
print("n features", X.shape[1], " n failures", len(f))
