"""v74: do the 27 public scores support the CatBoost rankings? Truths led by catNO / catNOaft / catNOaft50 (alone or 50/50 with v12),
with the v69 tilt grid; compared with the best v66 truth (-190.42). Writes reports/v74_post.csv (new truths only)."""
import os, sys, io, contextlib, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
ROOT, RANK, fit_a, logit = (G[k] for k in "ROOT RANK fit_a logit".split())
z = np.load(ROOT / "reports/v73_q.npz"); RANK.update({k: logit(z[k]) for k in z.files})
TL = [("none", 0.0, "none", 0.0)] + [("s122", c, z2, c2) for c in (0.4, 0.8, 1.2) for z2, c2 in (("none", 0.0), ("s149", 0.4), ("s126", 0.4))] + [("s126", 0.4, "none", 0.0), ("s149", 0.4, "none", 0.0)]
GR = [dict(R=R, w=w) for R in z.files for w in (0.5, 1.0)]
rows = [{**g, "R2": "v12", "w2": 0.0, "z": z1, "c": c1, "z2": z2, "c2": c2, "b": b, "alpha": al, "fl0": 0.0}
        for g, (z1, c1, z2, c2), b, al in itertools.product(GR, TL, (0.85, 1.0, 1.15), (1.0, 1.15, 1.3))]
print(len(rows), "truths", flush=True)
def one(r): ll, a = fit_a(r); return {**r, "a": a, "ll": ll}
Q = pd.DataFrame(Parallel(n_jobs=4, backend="threading")(delayed(one)(r) for r in rows)); Q.to_csv(ROOT / "reports/v74_post.csv", index=False)
old = pd.read_csv(ROOT / "reports/v66_post.csv")
print("best log-lik by ranking and weight:"); print(Q.groupby(["R", "w"]).ll.max().round(2).to_string())
print("best of v66 truths", round(old.ll.max(), 2), "| v12 alone", round(old[(old.w == 0)].ll.max(), 2), "| aft", round(old[old.R == "aft"].ll.max(), 2))
print(Q.sort_values("ll", ascending=False).head(8)[["R", "w", "z", "c", "z2", "c2", "b", "alpha", "a", "ll"]].round(3).to_string(index=False))
