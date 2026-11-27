"""v70c: paired residuals. For pairs of uploads, (observed difference - predicted difference) / paired sd under the best truth."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
ROOT, prob, PRE, U0, U00, obs, names, Fl, Bu = (G[k] for k in "ROOT prob PRE U0 U00 obs names Fl Bu".split())
P = pd.read_csv(ROOT / "reports/v66_post.csv").sort_values("ll", ascending=False); r = P.iloc[0]
p = prob(r.to_dict()); E1, Gm = PRE[r.alpha]; mi = (1 - p)[:, None] * U0 + p[:, None] * E1; mean = 0.3 * mi.sum(0)
cov = 0.3 * (np.einsum("n,njl->jl", 1 - p, U00) + np.einsum("n,njl->jl", p, Gm)) - 0.09 * mi.T @ mi
J = len(names); rows = []
for i in range(J):
    for j in range(i + 1, J):
        sd = np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j]); od = obs[j] - obs[i]; pdif = mean[j] - mean[i]
        nflag = int((Fl[:, i] != Fl[:, j]).sum()); bratio = Bu[:, j].sum() / max(Bu[:, i].sum(), 1)
        rows.append(dict(a=names[i], b=names[j], obs_d=od, pred_d=pdif, sd=sd, z=(od - pdif) / sd, flags_differ=nflag, budget_ratio=bratio))
D = pd.DataFrame(rows); D.to_csv(ROOT / "reports/v70_pairs.csv", index=False)
late = D[D.a.isin(names[11:]) & D.b.isin(names[11:])]
print("pairs among v12 onwards, sorted by z:"); print(late.sort_values("z").round(2).to_string(index=False))
print("\nmean z over late pairs", late.z.mean().round(2), "; mean z where b has more total budget than a (ratio>1.05):", late[late.budget_ratio > 1.05].z.mean().round(2), "fewer:", late[late.budget_ratio < 0.95].z.mean().round(2))
