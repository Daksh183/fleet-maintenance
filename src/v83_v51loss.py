"""v83: v51's cost broken down by component on the public 30% scale, under the full posterior (v82all), with expected counts and
two oracles (perfect failure knowledge with v51's budgets; perfect repair-cost knowledge with v51's flags)."""
import os, sys, io, contextlib, numpy as np, pandas as pd
os.environ.setdefault("GPNPZ", "reports/v51c_gp.npz"); os.environ["TAG"] = "v66"; sys.argv = ["x", "--reuse"]
src = open("v66_harness.py").read(); src = src[:src.index('ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv")')]
G = {"__name__": "h"}
with contextlib.redirect_stdout(io.StringIO()): exec(src, G)
from score import INSPECT, OVER, UNDER
ROOT, te, RANK, logit, ALPHAS, prob, MB = (G[k] for k in "ROOT te RANK logit ALPHAS prob MB".split())
for f in ("v73_q.npz", "v81_q.npz"):
    z = np.load(ROOT / "reports" / f); RANK.update({k: logit(z[k]) for k in z.files})
P = pd.read_csv(ROOT / "reports/v82all_post.csv"); T = P[P.wt > 1e-5].copy(); T.wt /= T.wt.sum()
s = pd.read_csv(ROOT / "submissions/submission_v51_gp_reduced.csv"); fl, b = s.flag.to_numpy() == 1, s.budget.to_numpy()
acc = {}
def add(k, v): acc[k] = acc.get(k, 0) + v
for al in ALPHAS:
    Ta = T[T.alpha == al]
    if not len(Ta): continue
    w = Ta.wt.sum(); p = sum(r.wt * prob(r._asdict()) for r in Ta.itertuples()) / w
    C = np.clip(MB * al, 5, 8000); mu = C.mean(1)
    sh = (UNDER * np.maximum(C - b[:, None], 0)).mean(1); ov = (OVER * np.maximum(b[:, None] - C, 0)).mean(1)
    add("inspections", w * INSPECT * fl.sum()); add("unused budget, flagged vehicles that don't fail", w * (OVER * b * (1 - p))[fl].sum())
    add("shortfall, flagged failures (cost above budget)", w * (p * sh)[fl].sum()); add("unused budget, flagged failures (cost below budget)", w * (p * ov)[fl].sum())
    add("missed failures (not flagged)", w * (p * mu)[~fl].sum())
    add("_fail_flagged", w * p[fl].sum()); add("_fail_unflagged", w * p[~fl].sum()); add("_flag_nofail", w * (1 - p)[fl].sum())
    add("_oracle_fail", w * (p * (INSPECT + sh + ov)).sum())                      # flag exactly the failures, v51 budgets (b=0 if none)
    best = np.minimum(p[:, None] * UNDER * C, (1 - p)[:, None] * OVER * C).mean(1)            # know the cost IF it fails: budget 0 or exactly c
    add("_oracle_cost", w * (INSPECT * fl.sum() + (p * mu)[~fl].sum() + best[fl].sum()))
    add("_oracle_both", w * (p * INSPECT).sum())                                             # flag exactly the failures, budget = cost
tot = sum(v for k, v in acc.items() if not k.startswith("_"))
print(f"v51 on the public 30% scale (x0.3), expected under the full posterior; actual public score 16,388\n")
for k, v in acc.items():
    if not k.startswith("_"): print(f"  {k:52s} {0.3 * v:7.0f}  ({100 * v / tot:4.1f}%)")
print(f"  {'TOTAL expected':52s} {0.3 * tot:7.0f}")
print(f"\nexpected counts on the public 30%: flagged {0.3 * fl.sum():.0f}; failures among flagged {0.3 * acc['_fail_flagged']:.0f}; flagged that don't fail {0.3 * acc['_flag_nofail']:.0f}; "
      f"failures missed {0.3 * acc['_fail_unflagged']:.0f}  (catch rate {acc['_fail_flagged'] / (acc['_fail_flagged'] + acc['_fail_unflagged']):.0%})")
print(f"oracles: knowing which vehicles fail (v51-style budgets) {0.3 * acc['_oracle_fail']:.0f}; knowing each vehicle's repair cost if it fails (v51 flags, not knowing whether it fails) {0.3 * acc['_oracle_cost']:.0f}; knowing both {0.3 * acc['_oracle_both']:.0f}")
