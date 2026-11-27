"""v76: budget scale. Expected public-scale cost of v75_hedge with budgets x s, under the posterior as fitted (repair costs as modelled)
and under repair costs x1.15 / x1.27 (the GLS shortfall scale from v70b)."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
ROOT, te, RANK, logit, ALPHAS, prob, exp_total, MB = (G[k] for k in "ROOT te RANK logit ALPHAS prob exp_total MB".split())
z = np.load(ROOT / "reports/v73_q.npz"); RANK.update({k: logit(z[k]) for k in z.files})
P = pd.read_csv(ROOT / "reports/v75_post.csv"); T = P[P.wt > 1e-5].copy(); T.wt /= T.wt.sum()
PB = {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}; W = {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
s = pd.read_csv(ROOT / "submissions/submission_v75_hedge.csv"); f, b = s.flag.to_numpy(), s.budget.to_numpy()
print(f"{'scale':>6s} " + " ".join(f"{'costs x' + str(k):>14s}" for k in (1.0, 1.15, 1.27)))
base = {}
for sc in (0.9, 1.0, 1.1, 1.2, 1.3, 1.5):
    out = []
    for k in (1.0, 1.15, 1.27):
        G["MX"] = {al: np.clip(MB * al * k, 5, 8000) for al in ALPHAS}; G["MU"] = {al: G["MX"][al].mean(1) for al in ALPHAS}
        v = 0.3 * exp_total(f, np.minimum(b * sc, 8000), PB, W); base.setdefault(k, v) if sc == 1.0 else None; out.append(v)
    print(f"{sc:6.2f} " + " ".join(f"{v:14.0f}" for v in out))
