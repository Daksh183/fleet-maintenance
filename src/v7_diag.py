"""v7 diagnostics: where does the test depot's higher predicted risk come from, and can we correct for it?"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from calib import make
from common import ROOT, SEED, load, split_holdout
from v2_params import PARAMS

tr, te, S = load()
dev, _ = split_holdout(tr)
y = dev.failure.to_numpy()
oof = np.load(ROOT / "reports/p_oof_v2.npy"); p_te = np.load(ROOT / "reports/p_te_v2.npy")
pl = make("platt").fit(oof, y)
q_dev, q_te = pl.predict(oof), pl.predict(p_te)
print(f"train failure rate {tr.failure.mean():.4f}")
print(f"mean p   raw: dev OOF {oof.mean():.4f}  test {p_te.mean():.4f}   platt: dev {q_dev.mean():.4f}  test {q_te.mean():.4f}")
for t in (0.025, 0.1, 0.5):
    print(f"  share p_platt > {t}: dev {np.mean(q_dev > t):.4f}  test {np.mean(q_te > t):.4f}")


# ---- 1. Saerens-Latinne-Decaestecker EM for the test prior (assumes p(x|y) unchanged = pure label shift)
def em_prior(p, pi_tr, it=200):
    pi = pi_tr
    for _ in range(it):
        a = pi / pi_tr * p
        post = a / (a + (1 - pi) / (1 - pi_tr) * (1 - p))
        pi = post.mean()
    return pi, post


pi_tr = y.mean()
pi_te, _ = em_prior(q_te, pi_tr)
print(f"\nEM test prior (from Platt p): {pi_te:.4f}   = {pi_te / pi_tr:.2f}x train ({pi_tr:.4f})")
pi_chk, _ = em_prior(q_dev, pi_tr)
print(f"EM on dev OOF itself (should return ~train rate): {pi_chk:.4f}")

# ---- 2. Which features raise the test risk? mean SHAP contribution, test minus dev
full = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
bst = full.get_booster()
cd = bst.predict(xgb.DMatrix(dev[S]), pred_contribs=True).mean(0)
ct = bst.predict(xgb.DMatrix(te[S]), pred_contribs=True).mean(0)
diff = pd.Series(ct - cd, index=S + ["bias"]).drop("bias").sort_values(ascending=False)
print(f"\nmean logit test - dev: {diff.sum():+.3f}  (from the 4 drifted sensors: "
      f"{diff[['s120', 's122', 's126', 's128']].sum():+.3f})")
print("top contributors to the test increase:", diff.head(8).round(3).to_dict())
print("top contributors pulling test down:", diff.tail(4).round(3).to_dict())

# ---- 3. importance weights: effective sample size
drift = ["s120", "s122", "s126", "s128"]
Sa = [s for s in S if s not in drift]
X = pd.concat([tr[Sa], te[Sa]]); d = np.r_[np.zeros(len(tr)), np.ones(len(te))]
adv = cross_val_predict(xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=SEED),
                        X, d, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:len(tr), 1]
for clip in (None, 20, 5):
    w = adv / (1 - adv + 1e-9)
    if clip: w = np.minimum(w / w.mean(), clip)
    ess = w.sum() ** 2 / (w ** 2).sum()
    wf = w[tr.failure.to_numpy() == 1]
    print(f"importance weights (no drifted sensors), clip {clip}: ESS {ess:,.0f} of {len(tr):,} ({ess/len(tr):.1%}); "
          f"failures ESS {wf.sum()**2/(wf**2).sum():.0f} of {len(wf)}; weighted failure rate {np.average(tr.failure, weights=w):.4f}")
np.save(ROOT / "reports/adv_p_train_nodrift.npy", adv)
