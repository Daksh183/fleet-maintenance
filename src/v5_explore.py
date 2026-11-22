"""v5 exploration: is repair cost (given failure) predictable from the sensors?"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, SEED, load, split_holdout

tr, te, S = load()
dev, ho = split_holdout(tr)
f = dev[dev.failure == 1].reset_index(drop=True)
y = np.log(f.repair_cost.to_numpy())
print(f"failures in dev: {len(f)}   log-cost std {y.std():.3f}   corr(log cost, followup) {np.corrcoef(f.followup_days, y)[0,1]:.3f}")
X, Xl = f[S], np.log1p(f[S].clip(lower=0))
cv = KFold(5, shuffle=True, random_state=SEED)


def rep(name, model, X):
    p = cross_val_predict(model, X, y, cv=cv)
    print(f"{name:28s} OOF R2 {r2_score(y, p):+.3f}   RMSE {np.sqrt(np.mean((y - p) ** 2)):.3f}")


rep("mean (baseline)", make_pipeline(SimpleImputer(), DummyRegressor()), Xl)
rep("ridge on log1p", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    RidgeCV(alphas=np.logspace(0, 4, 20))), Xl)
for d in (2, 3):
    rep(f"xgb depth{d}", xgb.XGBRegressor(n_estimators=200, max_depth=d, learning_rate=0.03, subsample=0.8,
                                         colsample_bytree=0.5, min_child_weight=5, n_jobs=-1, random_state=SEED), X)
c = Xl.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
print("top |corr| log1p(sensor) vs log cost:", c.head(8).round(3).to_dict())
p = np.load(ROOT / "reports" / "p_oof_v2.npy")[np.where(dev.failure.to_numpy() == 1)[0]]
print(f"corr(log cost, log OOF failure-p) among failures: {np.corrcoef(y, np.log(p + 1e-6))[0,1]:.3f}")
