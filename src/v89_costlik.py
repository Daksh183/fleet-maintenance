"""v89: public-score log-likelihood of the v87 cost model against v51's, for the top 40 truth configurations (level a refit)."""
import os, sys, io, contextlib, numpy as np, pandas as pd
from joblib import Parallel, delayed
os.environ["GPNPZ"] = "reports/v51c_gp.npz"; os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
H = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, H)
from score import INSPECT, OVER, UNDER
ROOT, RANK, logit, Fl, Bu, m, resid, fit_a, PRE = (H[k] for k in "ROOT RANK logit Fl Bu m resid fit_a PRE".split())
for f in ("v73_q.npz", "v81_q.npz"):
    z = np.load(ROOT / "reports" / f); RANK.update({k: logit(z[k]) for k in z.files})
g = np.load(ROOT / "reports/v87_gp.npz"); sev, sd, oof, oofsd, res = g["sev"], g["sd"], g["oof"], g["oofsd"], g["res"]; sup = np.isfinite(sev); ok = np.isfinite(oof)
K = 33; grid = (np.arange(K) + .5) / K; edges = np.quantile(oof[ok], [1 / 3, 2 / 3]); bf = np.searchsorted(edges, oof[ok]); zr = res / oofsd[ok]
QB = np.stack([np.quantile(zr[bf == j], grid) for j in range(3)]); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
D87 = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + np.where(sup, sd, 1.0)[:, None] * QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
P = pd.read_csv(ROOT / "reports/v82all_post.csv"); cfg = ["R", "w", "R2", "w2", "z", "c", "z2", "c2", "b", "fl0"]
top = P.groupby(cfg, dropna=False).wt.sum().sort_values(ascending=False).head(40).reset_index()
rows = []
for _, r in top.iterrows():
    d = {k: r[k] for k in cfg}; d["R2"] = d["R2"] if isinstance(d["R2"], str) else "v12"; d["z2"] = d["z2"] if isinstance(d["z2"], str) else "none"
    for k in ("w2", "c2", "fl0"): d[k] = 0.0 if d[k] != d[k] else d[k]
    for al in (1.0, 1.15, 1.3): rows.append({**d, "alpha": al})
def one(r): ll, a = fit_a(r); return ll
ll51 = Parallel(n_jobs=4, backend="threading")(delayed(one)(r) for r in rows)
for al in (1.0, 1.15, 1.3):
    C = np.clip(D87 * al, 5, 8000)
    U1 = np.where(Fl[:, None, :] == 1, INSPECT + UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0) + OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0), C[:, :, None])
    PRE[al] = (U1.mean(1), np.einsum("nkj,nkl->njl", U1, U1, optimize=True) / K)
ll87 = Parallel(n_jobs=4, backend="threading")(delayed(one)(r) for r in rows)
from scipy.special import logsumexp
a, b = np.array(ll51), np.array(ll87)
print(f"best log-lik: v51 cost model {a.max():.2f}, v87 cost model {b.max():.2f}; log-evidence (same truths, equal prior) v51 {logsumexp(a):.2f}, v87 {logsumexp(b):.2f}")
print(f"posterior probability that the v87 cost model is right (equal prior): {1 / (1 + np.exp(logsumexp(a) - logsumexp(b))):.3f}")
