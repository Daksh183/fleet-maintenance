"""v24: informative censoring -> soft labels. Healthy-labelled vehicles with follow-up < 365 d look ~10x riskier than fully followed
healthy ones (OOF mean p 0.007-0.015 vs 0.0013), so they likely hide unrecorded failures. Instead of dropping them (v18 IPCW),
give each censored-healthy row a soft label pi = P(fail | x) from a model trained on COMPLETE cases only (failures + healthy with
365 d), and train the final classifier on all rows with censored rows duplicated (positive weight pi, negative weight 1-pi).
 mode 'cv'  : OOF on dev with the v18 3x5 folds -> reports/v18_search/pu*.npy (scored with v18_score.py)
 mode 'test': fit on all train, v12 pipeline, flag count matched -> submissions/submission_v24_pu.csv"""
import sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from common import ROOT, load, split_holdout
from v2_params import PARAMS
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
SC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; TAG = "v24_pu" if SC == 1.0 else f"v24_pu_x{SC:g}"
tr, te, S = load()
def fit_pu(D, scale=1.0):
    y = D.failure.to_numpy(); t = D.followup_days.to_numpy(); cens = (y == 0) & (t < 365)
    A = xgb.XGBClassifier(**PARAMS).fit(D.loc[~cens, S], y[~cens])
    pi = np.clip(A.predict_proba(D.loc[cens, S])[:, 1] * scale, 0, 1)
    X = pd.concat([D.loc[~cens, S], D.loc[cens, S], D.loc[cens, S]])
    Y = np.r_[y[~cens], np.ones(cens.sum()), np.zeros(cens.sum())]
    Wt = np.r_[np.ones((~cens).sum()), pi, 1 - pi]
    return xgb.XGBClassifier(**PARAMS).fit(X, Y, sample_weight=Wt), pi.sum()
if MODE == "cv":
    dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
    for scale in (1.0, 2.0):
        P = np.zeros((3, len(y))); hid = []
        for r in range(3):
            for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(len(y)), y):
                mdl, h = fit_pu(dev.loc[a].reset_index(drop=True), scale); hid.append(h)
                P[r, b] = mdl.predict_proba(dev.loc[b, S])[:, 1]
        nm = "pu" if scale == 1.0 else "pu_x2"
        np.save(ROOT / f"reports/v18_search/{nm}.npy", P); print(f"done {nm}: expected hidden failures per training fold {np.mean(hid):.1f}", flush=True)
else:
    from calib import make
    from common import folds, write_submission, log_result
    from decide import decide_pv
    from scipy.stats import spearmanr
    dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); F = folds(y)
    oof = np.zeros(len(dev))
    for a, b in F: oof[b] = fit_pu(dev.loc[a].reset_index(drop=True), SC)[0].predict_proba(dev.loc[b, S])[:, 1]
    mdl, h = fit_pu(tr, SC); print(f"expected hidden failures in train: {h:.1f} (recorded {int(tr.failure.sum())})")
    q = make("platt").fit(oof, y).predict(mdl.predict_proba(te[S])[:, 1])
    qv12, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
    shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
    GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda qq, lam=1.0: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))
    fl0, _ = decide_pv(pavg(qv12), m, resid); target = int(fl0.sum())
    lo, hi = 0.1, 8.0
    for _ in range(40):
        lam = (lo + hi) / 2
        if decide_pv(pavg(q, lam), m, resid)[0].sum() < target: lo = lam
        else: hi = lam
    fl, bu = decide_pv(pavg(q, hi), m, resid); bu = np.minimum(bu, 8000)
    print(f"mean q {q.mean():.4f} (v12 {qv12.mean():.4f}); Spearman {spearmanr(q, qv12)[0]:.3f}; lambda {hi:.3f}; {int(fl.sum())} flags; "
          f"{int((fl != fl0).sum())} differ from v12")
    np.save(ROOT / f"reports/{TAG}_q.npy", q); write_submission(te, fl, bu, TAG)
    log_result("v24", f"{TAG}: v12_robust with censored-healthy rows soft-labelled (pi x{SC:g}) from a complete-case model, flag count matched, budgets capped",
               np.nan, np.nan, notes=f"{int(fl.sum())} flags; lambda {hi:.3f}; {int((fl != fl0).sum())} differ from v12")
