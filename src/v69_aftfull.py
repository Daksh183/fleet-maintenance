"""v69: survival ranking FULLY replacing the classifier. Truths use only the AFT rankings (w = 1: no classifier blend), with level a,
confidence slope b, wear tilts and repair-cost level fitted to the 27 public scores (finer grid than v66). Decision = Bayes decision
under that posterior, repair costs as v51. Writes submissions/submission_v69_aftfull.csv.
Usage: GPNPZ=reports/v51c_gp.npz TAG=v66 python v69_aftfull.py"""
import os, sys, io, contextlib, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v66_harness.py")).read()
src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]      # posterior, likelihood and decision functions
G = {"__name__": "h", "__file__": __file__}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
ROOT, te, ALPHAS, fit_a, prob, decide, exp_total, write_submission = (G[k] for k in ("ROOT", "te", "ALPHAS", "fit_a", "prob", "decide", "exp_total", "write_submission"))
AFT = ["aft", "aftd6", "clfaft"]
TL = [(z1, c1, z2, c2) for z1, cs in (("s122", (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6)),) for c1 in cs for z2, c2 in (("none", 0.0), ("s149", 0.4), ("s149", 0.8), ("s126", 0.4), ("s126", 0.8), ("s120", 0.4))]
rows = [dict(R=R, w=1.0, R2="v12", w2=0.0, z=z1, c=c1, z2=z2, c2=c2, b=b, alpha=al, fl0=0.0)
        for R, (z1, c1, z2, c2), b, al in itertools.product(AFT, TL, (0.85, 1.0, 1.15), ALPHAS)]
print(len(rows), "truths", flush=True)
def one(r):
    ll, a = fit_a(r); return {**r, "a": a, "ll": ll}
Q = pd.DataFrame(Parallel(n_jobs=4, backend="threading")(delayed(one)(r) for r in rows))
Q["wt"] = np.exp(Q.ll - Q.ll.max()); Q.wt /= Q.wt.sum(); Q.to_csv(ROOT / "reports/v69_post.csv", index=False)
old = pd.read_csv(ROOT / "reports/v66_post.csv")
print(f"best public log-lik: AFT-only {Q.ll.max():.3f} vs best of all v66 truths {old.ll.max():.3f}")
for col in ("R", "c", "z2", "c2", "b", "alpha"): print(" ", col, Q.groupby(col).wt.sum().round(3).to_dict())
T = Q[Q.wt > 1e-4].copy(); T.wt /= T.wt.sum()
PB = {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}; W = {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
fl, bu = decide(PB, W); write_submission(te, fl, bu, "v69_aftfull")
for nm in ("v51_gp_reduced", "v66_aft", "v69_aftfull"):
    s = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); f, b = s.flag.to_numpy(), s.budget.to_numpy()
    print(f"expected 30% score  {nm:16s} AFT-only truths {0.3 * exp_total(f, b, PB, W):8.0f} | full v66 posterior (all rankings) {0.3 * exp_total(f, b):8.0f}")
v = pd.read_csv(ROOT / "submissions/submission_v51_gp_reduced.csv"); com = (fl == 1) & (v.flag == 1)
print(f"vs v51: flags {fl.sum()} ({int((fl != v.flag).sum())} differ, {int(((fl == 1) & (v.flag == 0)).sum())} added); budget change on common flags: median abs {np.median(np.abs(bu - v.budget)[com]):.1f}, 90th pct {np.quantile(np.abs(bu - v.budget)[com], .9):.0f}")
p = sum(PB.values()) / sum(W.values()); added = (fl == 1) & (v.flag == 0); removed = (fl == 0) & (v.flag == 1)
for nm, k in (("added", added), ("removed", removed), ("kept", com)):
    print(f"  {nm:8s} {int(k.sum()):5d} vehicles: median s126 {np.nanmedian(te.s126[k]):.0f}, median s122 {np.nanmedian(te.s122[k]):.0f}, mean p {p[k].mean():.3f}")
