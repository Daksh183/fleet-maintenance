"""v70b: which cost component is under-predicted? GLS of observed public scores on per-file components under the best truths."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
from score import INSPECT, OVER, UNDER
ROOT, prob, PRE, U0, U00, obs, names, Fl, Bu, draws, resid, GRES = (G[k] for k in "ROOT prob PRE U0 U00 obs names Fl Bu draws resid GRES".split())
K = 33; C0 = draws(K, np.quantile(resid, (np.arange(K)+.5)/K), np.quantile(GRES, (np.arange(K)+.5)/K))
P = pd.read_csv(ROOT / "reports/v66_post.csv").sort_values("ll", ascending=False)
for _, r in P.head(2).iterrows():
    p = prob(r.to_dict()); C = np.clip(C0 * r.alpha, 5, 8000)
    insp = 0.3 * (Fl * INSPECT).sum(0)
    unused_ok = 0.3 * ((1 - p)[:, None] * Fl * OVER * Bu).sum(0)
    short = 0.3 * (p[:, None] * Fl * (UNDER * np.maximum(C[:, :, None] - Bu[:, None, :], 0)).mean(1)).sum(0)
    overf = 0.3 * (p[:, None] * Fl * (OVER * np.maximum(Bu[:, None, :] - C[:, :, None], 0)).mean(1)).sum(0)
    missed = 0.3 * (p[:, None] * (1 - Fl) * C.mean(1)[:, None]).sum(0)
    E1, Gm = PRE[r.alpha]; mi = (1 - p)[:, None] * U0 + p[:, None] * E1
    cov = 0.3 * (np.einsum("n,njl->jl", 1 - p, U00) + np.einsum("n,njl->jl", p, Gm)) - 0.09 * mi.T @ mi + np.diag((0.03 * 0.3 * mi.sum(0)) ** 2)
    Wi = np.linalg.inv(cov); fixed = insp + unused_ok
    X = np.column_stack([short, overf, missed]); yv = obs - fixed
    beta = np.linalg.solve(X.T @ Wi @ X, X.T @ Wi @ yv); se = np.sqrt(np.diag(np.linalg.inv(X.T @ Wi @ X)))
    r0 = yv - X.sum(1); r1 = yv - X @ beta
    print(f"truth {r.R} z={r.z} c={r.c} alpha={r.alpha}: betas shortfall {beta[0]:.2f}±{se[0]:.2f}, over-on-failures {beta[1]:.2f}±{se[1]:.2f}, missed {beta[2]:.2f}±{se[2]:.2f}; GLS chi2 {r0@Wi@r0:.1f} -> {r1@Wi@r1:.1f}")
    for X2, nm in ((missed[:, None], "missed only"), (short[:, None], "shortfall only"), (np.column_stack([short, missed]), "short+missed")):
        b2 = np.linalg.solve(X2.T @ Wi @ X2, X2.T @ Wi @ (yv - (X.sum(1) - X2.sum(1)))); rr = yv - (X.sum(1) - X2.sum(1)) - X2 @ b2
        print(f"   scale {nm:15s}: {np.round(b2, 2)} chi2 {rr @ Wi @ rr:.1f}")
    D = pd.DataFrame(dict(file=names, obs=obs, insp=insp, unused=unused_ok, short=short, overf=overf, missed=missed, fit=fixed + X @ beta)).round(0)
    print(D.to_string(index=False))
