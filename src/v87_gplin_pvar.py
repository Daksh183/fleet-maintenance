"""v87: repair-cost model with a linear mean (lin) and budgets from each vehicle's GP predictive sd within the three predicted-cost bands
(v85/v86: -2.4% total budget-related cost, 8/8 seeds). Fit A and B on all training failures, average; seed-42 OOF for the standardised
residuals. Decision: Bayes under the v82 posterior (truth weights from the v51 cost model, reused) with the new cost draws.
Writes reports/v87_gp.npz and submissions/submission_v87_gppvar.csv."""
import os, sys, io, contextlib, numpy as np, pandas as pd
from sklearn.model_selection import KFold
import v47_ablation as A, v49_round2 as R2, v51_gp_fit as G51, v85_gpext as X
y = A.y; ref = np.load(A.ROOT / "reports/v51c_gp.npz"); sup = np.isfinite(ref["sev"])
oo, os_, ts, tsd = [], [], [], []
for F, wear in zip(X.FS, (A.WEAR, ["s126", "s149"])):
    o = np.full(len(y), np.nan); q = np.full(len(y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=42).split(y): o[b], q[b] = X.GPX(lin=True).fit(A.select(F, a), y[a]).predict(A.select(F, b), var=True)
    m_, s_ = X.GPX(lin=True).fit(F, y).predict(G51.testfeats((R2.ST, wear)), var=True)
    oo.append(o); os_.append(q); ts.append(m_); tsd.append(s_)
oof, oofsd, sev, sd = (oo[0] + oo[1]) / 2, (os_[0] + os_[1]) / 2, (ts[0] + ts[1]) / 2, (tsd[0] + tsd[1]) / 2
oof[~A.valid] = np.nan; sev[~sup] = np.nan; ok = np.isfinite(oof); res = y[ok] - oof[ok]
print(f"OOF R2 {1 - (res ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum():.4f} (v51 C {1 - (ref['res'] ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum():.4f}); "
      f"test corr with v51 {np.corrcoef(sev[sup], ref['sev'][sup])[0, 1]:.4f}; predictive sd: train OOF median {np.median(oofsd[ok]):.3f}, test median {np.nanmedian(sd[sup]):.3f}, test 90th pct {np.nanquantile(sd[sup], .9):.3f}")
print(f"standardised residual: corr(|r|, sd) on OOF {np.corrcoef(np.abs(res), oofsd[ok])[0, 1]:.3f}")
np.savez(A.ROOT / "reports/v87_gp.npz", sev=sev, sd=sd, res=res, oof=oof, oofsd=oofsd, y=y)
# ---- decision with the new draws
os.environ["GPNPZ"] = "reports/v87_gp.npz"; os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
H = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, H)
ROOT, te, RANK, logit, ALPHAS, prob, decide, exp_total, write_submission, m, resid = (H[k] for k in "ROOT te RANK logit ALPHAS prob decide exp_total write_submission m resid".split())
for f in ("v73_q.npz", "v81_q.npz"):
    z = np.load(ROOT / "reports" / f); RANK.update({k: logit(z[k]) for k in z.files})
Kd = 99; grid = (np.arange(Kd) + .5) / Kd; edges = np.quantile(oof[ok], [1 / 3, 2 / 3]); bf = np.searchsorted(edges, oof[ok]); zr = res / oofsd[ok]
QB = np.stack([np.quantile(zr[bf == j], grid) for j in range(3)]); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
MBnew = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + np.where(sup, sd, 1.0)[:, None] * QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
P = pd.read_csv(ROOT / "reports/v82all_post.csv"); T = P[P.wt > 1e-5].copy(); T.wt /= T.wt.sum()
PB = {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}; W = {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
H["MX"] = {al: np.clip(MBnew * al, 5, 8000) for al in ALPHAS}; H["MU"] = {al: H["MX"][al].mean(1) for al in ALPHAS}
fl, bu = decide(PB, W); write_submission(te, fl, bu, "v87_gppvar")
v = pd.read_csv(ROOT / "submissions/submission_v51_gp_reduced.csv"); h = pd.read_csv(ROOT / "submissions/submission_v82_hedge.csv"); com = (fl == 1) & (v.flag == 1)
print(f"vs v51: {int((fl != v.flag).sum())} flags differ; common flags budget median ratio {np.median(bu[com] / np.maximum(v.budget[com], 1)):.3f}, mean {bu[com].mean():.0f} vs {v.budget[com].mean():.0f}; vs v82_hedge flags differ {int((fl != h.flag).sum())}")
for nm in ("v51_gp_reduced", "v82_hedge", "v87_gppvar"):
    s = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); print(f"expected public-scale cost under the NEW cost model: {nm:16s} {0.3 * exp_total(s.flag.to_numpy(), s.budget.to_numpy(), PB, W):8.0f}")
