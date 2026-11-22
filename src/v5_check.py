"""Sanity: do the sensors driving repair cost drift between depots? Compare predicted cost for likely failures."""
import numpy as np, xgboost as xgb
from scipy.stats import ks_2samp
from common import ROOT, SEED, load
from v5_percost_cfg import CM
tr, te, S = load()
for c in ["s080","s081","s082","s083","s084","s085","s022","s056"]:
    a, b = tr[c].dropna(), te[c].dropna()
    print(f"{c}: KS {ks_2samp(a,b).statistic:.3f}  median tr {a.median():.0f} te {b.median():.0f}  NaN tr {tr[c].isna().mean():.3f} te {te[c].isna().mean():.3f}")
f = tr[tr.failure == 1]
m = xgb.XGBRegressor(**CM).fit(f[S], np.log(f.repair_cost))
p_tr = np.load(ROOT/"reports"/"p_oof_v2.npy"); p_te = np.load(ROOT/"reports"/"p_te_v2.npy")
# train OOF probs are for the dev split only; compare likely-failure vehicles by p>0.1 on each side
from common import split_holdout
dev, _ = split_holdout(tr)
hi_tr, hi_te = p_tr > 0.1, p_te > 0.1
print(f"\nvehicles with p>0.1: dev {hi_tr.sum()} ({hi_tr.mean():.2%}) | test {hi_te.sum()} ({hi_te.mean():.2%})")
print("predicted failure cost among them (median/mean): dev %.0f/%.0f  test %.0f/%.0f" % (
    np.median(np.exp(m.predict(dev.loc[hi_tr, S]))), np.exp(m.predict(dev.loc[hi_tr, S])).mean(),
    np.median(np.exp(m.predict(te.loc[hi_te, S]))), np.exp(m.predict(te.loc[hi_te, S])).mean()))
print("actual failure cost in dev failures: median %.0f mean %.0f" % (dev[dev.failure==1].repair_cost.median(), dev[dev.failure==1].repair_cost.mean()))
