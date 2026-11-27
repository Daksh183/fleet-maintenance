"""v73: CatBoost (no s120/s122) failure classifier, bagged over 3 seeds; Platt on dev OOF; test q and the catNO+aft blend.
Also a hold-out check (fit dev, score the locked 20%). Writes reports/v73_q.npz (catNO, catNOaft, catNOaft50)."""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from calib import make
from common import ROOT, load, split_holdout
import v65_aft_holdout as V
tr, te, S = load(); NO = [s for s in S if s not in ("s120", "s122")]; dev, ho = split_holdout(tr); y = dev.failure.to_numpy(); yh = ho.failure.to_numpy()
def cat(D, E, seed):
    m = CatBoostClassifier(iterations=800, learning_rate=0.04, depth=6, l2_leaf_reg=5, thread_count=2, random_seed=seed, verbose=0)
    return m.fit(D[NO], D.failure).predict_proba(E[NO])[:, 1]
folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(dev, y))
jobs = [("oof", k, s) for k in range(5) for s in (0, 1, 2)] + [("ho", 0, s) for s in (0, 1, 2)] + [("te", 0, s) for s in (0, 1, 2)]
def run(kind, k, s):
    if kind == "oof": a, b = folds[k]; return cat(dev.iloc[a], dev.iloc[b], s)
    return cat(dev, ho, s) if kind == "ho" else cat(tr, te, s)
R = dict(zip(jobs, Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs)))
oof = np.zeros(len(y))
for k, (a, b) in enumerate(folds): oof[b] = np.mean([R[("oof", k, s)] for s in (0, 1, 2)], 0)
cal = make("platt").fit(oof, y)
qho = cal.predict(np.mean([R[("ho", 0, s)] for s in (0, 1, 2)], 0)); qte = cal.predict(np.mean([R[("te", 0, s)] for s in (0, 1, 2)], 0))
L, sig = V.L, V.sig
# AFT ensemble on the hold-out (one seed per member, Platt from the saved dev OOF) for the blend check
aho = {}
for n, kw in V.MEM.items():
    kw = dict(kw); sd = kw.pop("seed", 42); raw = V.aft(dev, ho, seed=sd, **kw)
    o = np.load(ROOT / f"reports/v18_search/v64_{V.FILES[n]}.npy")[0]; aho[n] = make("platt").fit(o, y).predict(raw)
aens = sig(np.mean([L(v) for v in aho.values()], 0))
from v63_survival import P0  # noqa: F401  (same params as the CV fits)
clfo = np.load(ROOT / "reports/v18_search/v64_clf_all.npy")[0]; clfho = make("platt").fit(clfo, y).predict(V.clf(dev, ho, S))
for nm, q in (("v12-style classifier", clfho), ("catNO", qho), ("aft ens", aens), ("catNO+aft (1:3)", sig(0.25 * L(qho) + 0.75 * L(aens))), ("catNO+aft (1:1)", sig(0.5 * L(qho) + 0.5 * L(aens)))):
    top = np.argsort(-q)[:int(0.07 * len(q))]
    print(f"hold-out {nm:22s} AUC {roc_auc_score(yh, q):.4f}  AP {average_precision_score(yh, q):.4f}  failures in top 7% {int(yh[top].sum())}/{int(yh.sum())}")
q65 = np.load(ROOT / "reports/v65_q.npz")
np.savez(ROOT / "reports/v73_q.npz", catNO=qte, catNOaft=sig(0.25 * L(qte) + 0.75 * L(q65["aft"])), catNOaft50=sig(0.5 * L(qte) + 0.5 * L(q65["aft"])))
from scipy.stats import spearmanr
print("test: spearman catNO vs v12", round(spearmanr(qte, np.load(ROOT / "reports/v18_qm.npy")[0]).correlation, 3), "| catNOaft vs aft", round(spearmanr(sig(0.25 * L(qte) + 0.75 * L(q65["aft"])), q65["aft"]).correlation, 3))
