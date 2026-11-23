"""Does the cost model's error size vary predictably by vehicle? OOF residuals of the v8 shape cost model."""
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_predict
from common import SEED, load
from features import shape
from v5_percost_cfg import CM
tr, te, S = load(); f = tr[tr.failure == 1].reset_index(drop=True)
X = pd.concat([f[S], shape(f)], axis=1); lc = np.log(f.repair_cost.to_numpy())
cv = KFold(5, shuffle=True, random_state=SEED)
m = cross_val_predict(xgb.XGBRegressor(**CM), X, lc, cv=cv); r = lc - m; a = np.abs(r)
print(f"OOF residual sd {r.std():.3f}; log cost range {lc.min():.2f}..{lc.max():.2f}")
print(f"spearman |resid| vs predicted log cost: {spearmanr(a, m)[0]:+.3f}")
qs = pd.qcut(m, 5, labels=False)
print("residual sd by predicted-cost quintile:", [round(r[qs == i].std(), 3) for i in range(5)])
print("mean residual by quintile (bias):     ", [round(r[qs == i].mean(), 3) for i in range(5)])
# can the features predict |resid| out of sample?
SC = dict(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=10, random_state=SEED)
ah = cross_val_predict(xgb.XGBRegressor(**SC), X.assign(m=m), a, cv=cv)
print(f"OOF prediction of |resid| from features: spearman {spearmanr(a, ah)[0]:+.3f}")
lo = ah < np.median(ah)
print(f"residual sd where predicted spread is low {r[lo].std():.3f} vs high {r[~lo].std():.3f}")
