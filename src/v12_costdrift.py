"""v12c: does the repair-cost model rely on sensors that are drifted at the new depot? (failures only, 2x5 CV)"""
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from common import load, split_holdout
from features import shape
from score import true_cost
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr)
f = dev[dev.failure == 1].reset_index(drop=True); c = true_cost(f); lc = np.log(c)
Xf = pd.concat([f[S], shape(f)], axis=1); Xt = pd.concat([te[S], shape(te)], axis=1); Xf = Xf.loc[:, Xf.notna().any() & Xt.reindex(columns=Xf.columns).notna().any()]; Xt = Xt[Xf.columns]
A1 = {**CM, "n_estimators": 1000, "max_depth": 2}
m = xgb.XGBRegressor(**A1).fit(Xf, lc)
imp = pd.Series(m.get_booster().get_score(importance_type="total_gain")).sort_values(ascending=False); imp /= imp.sum()
ks = pd.Series({k: ks_2samp(tr[k].dropna() if k in tr else shape(tr)[k].dropna(), te[k].dropna() if k in te else shape(te)[k].dropna()).statistic for k in imp.index[:12]})
print("top cost-model features (share of total gain) and their train-vs-test KS drift:")
print(pd.DataFrame({"gain": imp.head(12).round(3), "KS": ks.round(3)}).to_string())
Z = Xf.copy(); 
KS_all = pd.Series({k: ks_2samp(shape(tr)[k].dropna() if k not in tr else tr[k].dropna(), Xt[k].dropna()).statistic for k in Xf.columns})
print(f"\ngain-weighted mean KS drift of the cost model's inputs: {(imp.reindex(KS_all.index).fillna(0) * KS_all).sum() / imp.reindex(KS_all.index).fillna(0).sum():.3f}")
sets = {"all": [], "drop drifted 4": ["s120", "s122", "s126", "s128"], "drop KS>0.5": list(KS_all[KS_all > 0.5].index), "drop KS>0.3": list(KS_all[KS_all > 0.3].index)}
print("columns dropped:", {k: len(v) for k, v in sets.items()})
for nm, drop in sets.items():
    cols = [x for x in Xf.columns if x not in drop]; r2 = []
    for rs in (42, 1):
        for ia, ib in KFold(5, shuffle=True, random_state=rs).split(Xf):
            r2.append(r2_score(lc[ib], xgb.XGBRegressor(**A1).fit(Xf.iloc[ia][cols], lc[ia]).predict(Xf.iloc[ib][cols])))
    pt = xgb.XGBRegressor(**A1).fit(Xf[cols], lc).predict(Xt[cols])
    if nm == "all": base = pt
    print(f"{nm:16s} CV R2 {np.mean(r2):.3f}   test predicted log-cost mean {pt.mean():.3f}   corr with 'all' on test {np.corrcoef(pt, base)[0, 1]:.3f}   mean abs diff vs all {np.abs(pt - base).mean():.3f}")
print(f"train failures: mean log-cost {lc.mean():.3f}; model prediction on all test vehicles: mean {base.mean():.3f}")
