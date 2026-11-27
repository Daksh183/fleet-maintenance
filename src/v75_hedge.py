"""v75: Bayes decision under the union of v66 truths and the CatBoost-led truths (v74), and a cross-table: expected public-scale cost
of each file under each ranking family. Writes submissions/submission_v75_hedge.csv."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
ROOT, te, RANK, logit, ALPHAS, prob, decide, exp_total, write_submission = (G[k] for k in "ROOT te RANK logit ALPHAS prob decide exp_total write_submission".split())
z = np.load(ROOT / "reports/v73_q.npz"); RANK.update({k: logit(z[k]) for k in z.files})
P = pd.concat([pd.read_csv(ROOT / "reports/v66_post.csv").drop(columns="wt"), pd.read_csv(ROOT / "reports/v74_post.csv")], ignore_index=True)
P["wt"] = np.exp(P.ll - P.ll.max()); P.wt /= P.wt.sum(); P.to_csv(ROOT / "reports/v75_post.csv", index=False)
CAT = ["catNO", "catNOaft", "catNOaft50"]; AFT = ["aft", "aftd6", "clfaft"]
fam = np.where(P.R.isin(CAT), "catboost", np.where(P.R.isin(AFT) | P.R2.isin(AFT), "survival", np.where(P.w == 0, "v12", "other classifier")))
P["fam"] = fam; print("posterior by family:", P.groupby("fam").wt.sum().round(3).to_dict())
def pb(T):
    T = T[T.wt > 1e-5].copy(); T.wt /= T.wt.sum()
    return {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}, {al: T[T.alpha == al].wt.sum() for al in ALPHAS}
VIEWS = {"all (data weights)": P, "equal families": P.assign(wt=P.wt / P.groupby("fam").wt.transform("sum"))}
VIEWS.update({f"only {f}": P[P.fam == f] for f in ("v12", "other classifier", "survival", "catboost")})
PBS = {k: pb(v) for k, v in VIEWS.items()}
fl, bu = decide(*PBS["all (data weights)"]); write_submission(te, fl, bu, "v75_hedge")
fl2, bu2 = decide(*PBS["equal families"]); write_submission(te, fl2, bu2, "v75_eqfam")
files = ["v51_gp_reduced", "v66_aft", "v69_aftfull", "v75_hedge", "v75_eqfam"]
rows = []
for f in files:
    s = pd.read_csv(ROOT / f"submissions/submission_{f}.csv"); rows.append({"file": f, "flags": int(s.flag.sum()), **{k: 0.3 * exp_total(s.flag.to_numpy(), s.budget.to_numpy(), *PBS[k]) for k in VIEWS}})
D = pd.DataFrame(rows).set_index("file"); print((D - D.loc["v51_gp_reduced"]).round(0).to_string()); print(D.round(0).to_string())
v = pd.read_csv(ROOT / "submissions/submission_v51_gp_reduced.csv"); print("v75_hedge vs v51 flags differ", int((fl != v.flag).sum()), "| v75_eqfam", int((fl2 != v.flag).sum()))
