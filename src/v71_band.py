"""v71: can the 27 public scores support extra failure odds for BORDERLINE vehicles (v12 rank band) and a separate failure-cost level?
Top v66 truths x band bump g on logit for vehicles ranked [lo, hi) by v12, refit level a. Writes reports/v71_post.csv."""
import os, sys, io, contextlib, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
ROOT, te, RANK, loglik, prob0 = (G[k] for k in "ROOT te RANK loglik prob".split())
rk = np.empty(len(te), int); rk[np.argsort(-RANK["v12"])] = np.arange(len(te))
BANDS = {"none": np.zeros(len(te), bool), "1300-2000": (rk >= 1300) & (rk < 2000), "1300-3000": (rk >= 1300) & (rk < 3000), "1800-4000": (rk >= 1800) & (rk < 4000), "1000-2000": (rk >= 1000) & (rk < 2000)}
def prob(r):
    p = prob0(r)
    if r.get("g", 0) and r.get("band", "none") != "none":
        L = np.log(p / (1 - p)); p = 1 / (1 + np.exp(-(L + r["g"] * BANDS[r["band"]])))
    return p
def fit_a(r):
    best = max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(r["a0"] - 1.2, r["a0"] + 0.6, 10))
    return max((loglik(prob({**r, "a": a}), r["alpha"]), a) for a in np.linspace(best[1] - 0.2, best[1] + 0.2, 5))
P0 = pd.read_csv(ROOT / "reports/v66_post.csv").sort_values("ll", ascending=False)
base = P0.drop_duplicates(["R", "w", "R2", "w2", "z", "c", "z2", "c2", "b"]).head(24)
rows = []
for _, r in base.iterrows():
    for band, g, al in itertools.product(BANDS, (0.0, 0.5, 1.0, 1.5), (1.0, 1.15, 1.3, 1.45)):
        if (band == "none") != (g == 0.0): continue
        rows.append({**{k: r[k] for k in ["R", "w", "R2", "w2", "z", "c", "z2", "c2", "b", "fl0"]}, "a0": r.a, "band": band, "g": g, "alpha": al})
print(len(rows), "truths", flush=True)
def one(r): ll, a = fit_a(r); return {**r, "a": a, "ll": ll}
Q = pd.DataFrame(Parallel(n_jobs=4, backend="threading")(delayed(one)(r) for r in rows))
Q["wt"] = np.exp(Q.ll - Q.ll.max()); Q.wt /= Q.wt.sum(); Q.to_csv(ROOT / "reports/v71_post.csv", index=False)
print("best ll by band/g:"); print(Q.groupby(["band", "g"]).ll.max().round(2).to_string())
print("posterior: band", Q.groupby("band").wt.sum().round(3).to_dict(), "g", Q.groupby("g").wt.sum().round(3).to_dict(), "alpha", Q.groupby("alpha").wt.sum().round(3).to_dict())
