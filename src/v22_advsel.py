"""v22: adversarial feature selection. Drop the features whose distribution shifted most between depots, as long as CV cost holds.
Drift per feature = KS(train, test) and the gain share in a train-vs-test classifier (drifted counters included).
Candidates saved as OOF in reports/v18_search/ (same 3x5 folds, censored labels like base_cens) and scored with v18_score.py."""
import sys, numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from common import ROOT, load, split_holdout
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
ks = pd.Series({s: ks_2samp(tr[s].dropna(), te[s].dropna()).statistic if te[s].notna().any() else 1.0 for s in S}).sort_values(ascending=False)
Xa = pd.concat([tr[S], te[S]]); ya = np.r_[np.zeros(len(tr)), np.ones(len(te))]
adv = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=8, eval_metric="logloss").fit(Xa, ya)
g = pd.Series(adv.get_booster().get_score(importance_type="total_gain")).reindex(S).fillna(0); g /= g.sum()
clf = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
fg = pd.Series(clf.get_booster().get_score(importance_type="total_gain")).reindex(S).fillna(0); fg /= fg.sum()
T = pd.DataFrame({"ks": ks, "adv_gain": g, "fail_gain": fg}).sort_values("ks", ascending=False)
T.to_csv(ROOT / "reports/v22_drift.csv"); print(T.head(25).round(4).to_string(), flush=True)
print("failure-gain share carried by features with KS>0.1:", round(T.fail_gain[T.ks > 0.1].sum(), 3), " >0.05:", round(T.fail_gain[T.ks > 0.05].sum(), 3))
CANDS = {}
for th in (0.1, 0.06, 0.04, 0.03):
    CANDS[f"ks{int(th*100):02d}"] = [s for s in S if T.ks[s] <= th]
CANDS["adv10"] = [s for s in S if s not in g.sort_values(ascending=False).index[:10]]
CANDS["adv20"] = [s for s in S if s not in g.sort_values(ascending=False).index[:20]]
for nm, cols in CANDS.items():
    P = np.zeros((3, len(y)))
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            P[r, b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, cols], y[a]).predict_proba(dev.loc[b, cols])[:, 1]
    np.save(ROOT / f"reports/v18_search/sel_{nm}.npy", P); print(f"done sel_{nm}: {len(cols)} features (dropped {len(S)-len(cols)})", flush=True)
