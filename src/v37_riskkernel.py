"""v37: Daksh-style hypothesis on the failure side. Histogram-kernel failure model: square-root shares + cumulative shares of the
six histograms (as in v33), log usage, wear group; Nystroem RBF features + logistic regression. OOF on dev with the v18 3x5 folds
-> reports/v18_search/ (scored with v18_score.py, alone and blended with XGBoost)."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "8")
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from common import ROOT, load, split_holdout
from v33_hist_gp import extract
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
G = extract(dev); W = np.log1p(np.maximum(dev[["s126", "s149", "s123", "s122"]].to_numpy(float), 0))
X = np.hstack(G + [W])
for nm, C, nc in (("kern_lr", 1.0, 600), ("kern_lr_c01", 0.1, 600)):
    P = np.zeros((3, len(y)))
    for r in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Nystroem(gamma=1.0 / X.shape[1], n_components=nc, random_state=r),
                                LogisticRegression(C=C, max_iter=3000, class_weight=None))
            P[r, b] = mdl.fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    np.save(ROOT / f"reports/v18_search/{nm}.npy", P); print("done", nm, flush=True)
