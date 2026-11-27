"""v77: extend the repair-cost level grid beyond 1.45 (posterior had 11% at the edge). Top v75 truth configurations x alpha in
1.0..2.2, level a refit. Writes reports/v78_post.csv."""
import os, sys, io, contextlib, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
from score import INSPECT, OVER, UNDER
ROOT, RANK, logit, PRE, Fl, Bu, draws, K, RQ, RQG, fit_a = (G[k] for k in "ROOT RANK logit PRE Fl Bu draws K RQ RQG fit_a".split())
z = np.load(ROOT / "reports/v73_q.npz"); RANK.update({k: logit(z[k]) for k in z.files})
NEWA = ()
for al in NEWA:
    C = np.clip(draws(K, RQ, RQG) * al, 5, 8000)
    U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
    PRE[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)
P = pd.read_csv(ROOT / "reports/v75_post.csv"); cfg = ["R", "w", "R2", "w2", "z", "c", "z2", "c2", "fl0"]
top = P.groupby(cfg, dropna=False).wt.sum().sort_values(ascending=False).head(25).reset_index()
rows = [{**{k: r[k] for k in cfg}, "alpha": al, "b": bb} for _, r in top.iterrows() for al in (1.0, 1.15, 1.3) for bb in (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6)]
for r in rows:
    if not isinstance(r["R2"], str): r["R2"] = "v12"
    if not isinstance(r["z2"], str): r["z2"] = "none"
    for k in ("w2", "c2", "fl0"):
        if r[k] != r[k]: r[k] = 0.0
print(len(rows), "truths", flush=True)
def one(r): ll, a = fit_a(r); return {**r, "a": a, "ll": ll}
Q = pd.DataFrame(Parallel(n_jobs=4, backend="threading")(delayed(one)(r) for r in rows))
Q["wt"] = np.exp(Q.ll - Q.ll.max()); Q.wt /= Q.wt.sum(); Q.to_csv(ROOT / "reports/v78_post.csv", index=False)
print("best log-lik by b:", Q.groupby("b").ll.max().round(2).to_dict()); print("posterior over b:", Q.groupby("b").wt.sum().round(3).to_dict())
print("mean fitted a by b:", Q.groupby("b").a.mean().round(2).to_dict())
