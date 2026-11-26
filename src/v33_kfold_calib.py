"""v33: is 5-fold OOF good enough for calibration? (practical stand-in for leave-one-out)
Platt fitted on 20-fold OOF (models see 95% of dev) vs on the 5-fold OOF used so far (80%). Compared on the test probabilities."""
import numpy as np, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from calib import make
from common import ROOT, load, split_holdout
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
oof20 = np.zeros(len(y))
for i, (a, b) in enumerate(StratifiedKFold(20, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    oof20[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
np.save(ROOT / "reports/v33_oof20.npy", oof20)
oof5 = np.load(ROOT / "reports/p_oof_v2.npy"); pte = np.load(ROOT / "reports/p_te_v2.npy")
P5, P20 = make("platt").fit(oof5, y), make("platt").fit(oof20, y)
q5, q20 = P5.predict(pte), P20.predict(pte)
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
sl, ic = np.polyfit(lg(q5), lg(q20), 1)
print(f"OOF mean raw p: 5-fold {oof5.mean():.5f}, 20-fold {oof20.mean():.5f}; corr {np.corrcoef(oof5, oof20)[0,1]:.4f}")
print(f"test mean calibrated q: 5-fold Platt {q5.mean():.5f}, 20-fold Platt {q20.mean():.5f}; Spearman {spearmanr(q5, q20)[0]:.6f}")
print(f"logit(q20) = {sl:.4f} * logit(q5) {ic:+.4f}   (an affine map in logit: absorbed by the harness's fitted level a and slope b)")
for t in (0.02, 0.05, 0.1, 0.3):
    print(f"  share of test with q >= {t}: 5-fold {np.mean(q5 >= t):.4f}  20-fold {np.mean(q20 >= t):.4f}")
