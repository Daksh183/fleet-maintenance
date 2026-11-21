"""v1: EDA findings (no model change). Writes reports/v1_eda.txt.

Runs adversarial validation twice: on the raw files, and after the v1 fixes in common.load().
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import HIST, ROOT, SEED, load

out = []
say = lambda *a: (print(*a), out.append(" ".join(map(str, a))))


def adversarial(tr, te, cols, tag):
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method="predict_proba")[:, 1]
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    say(f"[{tag}] adversarial AUC = {roc_auc_score(y, p):.4f}; top:", imp.head(6).round(3).to_dict())
    return p


def top_ks(tr, te, cols, n=8):
    ks = pd.Series({c: ks_2samp(tr[c].dropna(), te[c].dropna()).statistic
                    for c in cols if tr[c].notna().sum() > 100 and te[c].notna().sum() > 100})
    return ks.sort_values(ascending=False).head(n).round(3).to_dict()


# ---------------- raw data ----------------
tr, te, S = load(raw=True)
f = tr[tr.failure == 1]
say(f"failures: {len(f)} / {len(tr)} = {tr.failure.mean():.4%}")
say("repair cost quantiles 5/25/50/75/90/95/99:",
    f.repair_cost.quantile([.05, .25, .5, .75, .9, .95, .99]).round(0).tolist())
top = f.repair_cost.sort_values(ascending=False)
say(f"top 10% of failures = {top.iloc[:len(top)//10].sum()/top.sum():.1%} of total repair cost")

bins = pd.cut(tr.followup_days, [0, 90, 180, 270, 364, 365])
say("failure rate by followup_days:\n"
    + tr.groupby(bins, observed=True).failure.agg(["mean", "size"]).round(4).to_string())

nan_tr, nan_te = tr[S].isna().mean(axis=1), te[S].isna().mean(axis=1)
say(f"mean row-NaN fraction: train {nan_tr.mean():.3f}  test {nan_te.mean():.3f}")
q = pd.qcut(nan_tr.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
say("failure rate by row-NaN quartile:", tr.groupby(q, observed=True).failure.mean().round(4).to_dict())
say("constant columns in train:", [c for c in S if tr[c].nunique(dropna=True) <= 1])
say("columns fully NaN in test:", [c for c in S if te[c].isna().all()])

say("histogram medians train:", tr[HIST].median().round(0).tolist())
say("histogram medians test: ", te[HIST].median().round(0).tolist())
say("KS(train bin k, test bin k+1):",
    [round(ks_2samp(tr[HIST[i]].dropna(), te[HIST[i + 1]].dropna()).statistic, 3) for i in range(9)])
say("top KS (raw):", top_ks(tr, te, S))
adversarial(tr, te, S, "raw")

# ---------------- after v1 fixes ----------------
tr, te, S = load()
cols = [c for c in S if c != HIST[-1]]
say("top KS (aligned):", top_ks(tr, te, cols))
for c in ["s126", "s128"]:
    a, b = tr[c].dropna(), te[c].dropna()
    say(f"{c}: share divisible by 1000  train {(a % 1000 == 0).mean():.3f}  test {(b % 1000 == 0).mean():.3f}")
p = adversarial(tr, te, cols, "aligned")
np.save(ROOT / "reports" / "adv_p_train.npy", p[: len(tr)])  # reused in v7
noshift = [c for c in cols if c not in ("s120", "s122", "s126", "s128")]
adversarial(tr, te, noshift, "aligned, minus s120/s122/s126/s128")

(ROOT / "reports" / "v1_eda.txt").write_text("\n".join(out))
