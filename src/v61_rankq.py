"""v61: test-set probabilities for the v60 failure-model variants (fit on all train, Platt on dev out-of-fold repeat 0)."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from calib import make
from common import ROOT
import v60_feats as F
NEW = ["mono", "rates", "ratesOnly", "shares", "sharesOnly", "linear+"]
SHa, RTa = F.derived(F.tr); SHb, RTb = F.derived(F.te)
D = pd.concat([pd.concat([F.tr, SHa, RTa], axis=1), pd.concat([F.te, SHb, RTb], axis=1)], ignore_index=True)
Y = np.r_[F.tr.failure.to_numpy(), np.zeros(len(F.te), int)]; a = np.arange(len(F.tr)); b = np.arange(len(F.tr), len(D))
def one(n):
    F.D, F.y = D, Y
    return F.fit(n, a, b)
if __name__ == "__main__":
    R = Parallel(n_jobs=4)(delayed(one)(n) for n in NEW); qv12, _ = np.load(ROOT / "reports/v18_qm.npy")
    out = {}
    for n, p in zip(NEW, R):
        q = make("platt").fit(np.load(ROOT / f"reports/v18_search/v60_{n}.npy")[0], F.y).predict(p); out[n.rstrip("+")] = q
        r = pd.Series(q).corr(pd.Series(qv12), method="spearman"); top = set(np.argsort(-q)[:1500]); t12 = set(np.argsort(-qv12)[:1500])
        print(f"{n:10s} mean q {q.mean():.4f}; Spearman with v12 {r:.3f}; top-1500 overlap with v12 {len(top & t12)}")
    np.savez(ROOT / "reports/v61_q.npz", **out)
