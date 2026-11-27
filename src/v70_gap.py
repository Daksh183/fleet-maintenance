"""v70: where does the posterior miss? Predicted public score of each uploaded file (best truths) vs observed."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
g = lambda k: G[k]
P = pd.read_csv(g("ROOT") / "reports/v66_post.csv").sort_values("ll", ascending=False)
PRE, U0, obs, names, prob = g("PRE"), g("U0"), g("obs"), g("names"), g("prob")
for _, r in P.head(3).iterrows():
    p = prob(r.to_dict()); E1, Gm = PRE[r.alpha]; mi = (1 - p)[:, None] * U0 + p[:, None] * E1; mean = 0.3 * mi.sum(0)
    print(f"truth R={r.R} w={r.w} z={r.z} c={r.c} alpha={r.alpha} a={r.a:.2f} b={r.b} ll={r.ll:.2f}; expected failures in test {p.sum():.0f} ({p.mean():.4f})")
    sd = np.sqrt(np.diag(0.3 * (np.einsum("n,njl->jl", 1 - p, g('U00')) + np.einsum("n,njl->jl", p, Gm)) - 0.09 * mi.T @ mi))
    for n, o, mu, s in zip(names, obs, mean, sd): print(f"   {n:16s} obs {o:9.0f} pred {mu:9.0f} diff {o - mu:+7.0f} ({(o - mu) / s:+.1f} sd)")
