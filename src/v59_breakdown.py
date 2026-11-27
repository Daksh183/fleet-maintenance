"""v59: where does the expected cost of a submission come from? Uses the harness posterior (v53_post, 27 scores) and v51 repair costs.
Prints expected PUBLIC-scale (30%) cost split into parts, and the value of removing each source of uncertainty (oracle bounds)."""
import sys, runpy, os, numpy as np, pandas as pd
os.environ.update(GPNPZ="reports/v51c_gp.npz", TAG="v53"); sys.argv = ["x", "--reuse"]
import io, contextlib
src = open(os.path.join(os.path.dirname(__file__), "v53_harness.py")).read()
src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]          # posterior and decision functions only
G = {"__name__": "h", "__file__": __file__}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
te, PBAR, WAL, MX, ALPHAS, decide = (G[k] for k in ("te", "PBAR", "WAL", "MX", "ALPHAS", "decide"))
s = pd.read_csv(G["ROOT"] / "submissions/submission_v51_gp_reduced.csv"); fl, b = s.flag.to_numpy(), s.budget.to_numpy()
p = sum(PBAR.values()); EC = sum(PBAR[a] * MX[a].mean(1) for a in ALPHAS)                        # E[p], E[p*c]
short = sum(PBAR[a] * np.maximum(MX[a] - b[:, None], 0).mean(1) for a in ALPHAS) * 0.2
over_f = sum(PBAR[a] * np.maximum(b[:, None] - MX[a], 0).mean(1) for a in ALPHAS) * 0.02
parts = {"inspections (10 each)": 10 * fl, "shortfall on flagged failures": fl * short, "unused budget, flagged failures": fl * over_f,
         "unused budget, flagged healthy": fl * 0.02 * b * (1 - p), "missed failures (unflagged)": (1 - fl) * EC}
tot = sum(v.sum() for v in parts.values())
for k, v in parts.items(): print(f"{k:34s} {0.3 * v.sum():8.0f}  {100 * v.sum() / tot:5.1f}%")
print(f"{'total (model expectation)':34s} {0.3 * tot:8.0f}; flags {int(fl.sum())}; expected failures {p.sum():.0f}, of which flagged {p[fl == 1].sum():.0f}")
# oracle bounds under the same model: draw failures and costs, then see what perfect knowledge of each would buy
rng = np.random.default_rng(0); res = {k: [] for k in ("now", "know which vehicles fail", "know every repair cost", "know both")}
W = np.array([WAL[a] for a in ALPHAS]); W = W / W.sum()
for it in range(40):
    a = ALPHAS[rng.choice(len(ALPHAS), p=W)]; y = rng.random(len(p)) < np.clip(PBAR[a] / WAL[a], 0, 1)
    c = MX[a][np.arange(len(p)), rng.integers(0, MX[a].shape[1], len(p))] * y
    cost = lambda f, bb: np.where(f == 1, 10 + 0.2 * np.maximum(c - bb, 0) + 0.02 * np.maximum(bb - c, 0), c).sum() * 0.3
    res["now"].append(cost(fl, b))
    f1 = (y & (c > 10)).astype(int)                                                             # know who fails, budget from cost distribution
    q = np.quantile(MX[a], 10 / 11, axis=1); res["know which vehicles fail"].append(cost(f1, np.where(f1 == 1, q, 0)))
    pbar = np.clip(p, 1e-9, 1)
    f2 = (10 + (1 - pbar) * 0.02 * (EC / pbar) < EC).astype(int); cfull = MX[a][np.arange(len(p)), rng.integers(0, MX[a].shape[1], len(p))]
    cfull = np.where(y, c, cfull); f2 = ((10 + (1 - pbar) * 0.02 * cfull < pbar * cfull) & (pbar > 1 / 11)).astype(int); res["know every repair cost"].append(cost(f2, np.where(f2 == 1, cfull, 0)))
    res["know both"].append(cost(f1, np.where(f1 == 1, c, 0)))
for k, v in res.items(): print(f"oracle: {k:28s} {np.mean(v):8.0f}")
r = fl == 1
for lo, hi in ((0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)):
    k = r & (p >= lo) & (p < hi); print(f"flagged with p in [{lo},{hi}): {k.sum():5d} vehicles, expected cost {0.3 * sum(v[k].sum() for v in parts.values()):6.0f}")
