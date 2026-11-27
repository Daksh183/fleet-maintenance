"""v88: v87 scored under BOTH cost models (v51 GP draws, v87 linear-mean + predictive-sd draws), full posterior and per ranking family."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ["GPNPZ"] = "reports/v51c_gp.npz"; os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
H = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, H)
ROOT, te, RANK, logit, ALPHAS, prob, exp_total, m, resid, MB = (H[k] for k in "ROOT te RANK logit ALPHAS prob exp_total m resid MB".split())
for f in ("v73_q.npz", "v81_q.npz"):
    z = np.load(ROOT / "reports" / f); RANK.update({k: logit(z[k]) for k in z.files})
g = np.load(ROOT / "reports/v87_gp.npz"); sev, sd, oof, oofsd, res = g["sev"], g["sd"], g["oof"], g["oofsd"], g["res"]; sup = np.isfinite(sev); ok = np.isfinite(oof)
Kd = 99; grid = (np.arange(Kd) + .5) / Kd; edges = np.quantile(oof[ok], [1 / 3, 2 / 3]); bf = np.searchsorted(edges, oof[ok]); zr = res / oofsd[ok]
QB = np.stack([np.quantile(zr[bf == j], grid) for j in range(3)]); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
MB87 = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + np.where(sup, sd, 1.0)[:, None] * QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
P = pd.read_csv(ROOT / "reports/v82all_post.csv"); CAT = ["catNO", "catNOaft", "catNOaft50"]; AFT = ["aft", "aftd6", "clfaft", "aftcw", "aftcw_aft"]
P["fam"] = np.where(P.R.isin(CAT), "catboost", np.where(P.R.isin(AFT) | P.R2.isin(AFT), "survival", np.where(P.w == 0, "v12", "other clf")))
def pb(T):
    T = T[T.wt > 1e-5].copy(); T.wt /= T.wt.sum()
    return {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}, {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
VIEWS = {"all": pb(P), **{f: pb(P[P.fam == f]) for f in ("other clf", "survival", "catboost")}}
files = ["v51_gp_reduced", "v66_aft", "v82_hedge", "v87_gppvar"]; S = {f: pd.read_csv(ROOT / f"submissions/submission_{f}.csv") for f in files}
for cm, mb in (("v51 cost model", MB), ("v87 cost model", MB87)):
    H["MX"] = {al: np.clip(mb * al, 5, 8000) for al in ALPHAS}; H["MU"] = {al: H["MX"][al].mean(1) for al in ALPHAS}
    D = pd.DataFrame({f: {k: 0.3 * exp_total(S[f].flag.to_numpy(), S[f].budget.to_numpy(), *v) for k, v in VIEWS.items()} for f in files}).T
    print(f"\n{cm}: expected public-scale cost, change vs v51"); print((D - D.loc["v51_gp_reduced"]).round(0).to_string())
