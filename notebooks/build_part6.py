"""Builds the v17 notebook (the two level probes on the v12_robust base: budget x1.25 and shift x1.35)."""
from nbkit import NB, SHAPE

POSTERIOR = r'''
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}                      # v9 cost model
DRIFT = ["s120", "s122", "s126", "s128"]; MAP = ["s120", "s122"]
keep = [s for s in S if s not in DRIFT]
f_all = tr[tr.failure == 1]; lc_all = np.log(f_all.repair_cost.to_numpy())
Xf, Xt = pd.concat([f_all[keep], shape(f_all)], axis=1), pd.concat([te[keep], shape(te)], axis=1)   # cost model: no drifted sensors (v12)
# posterior over the depot shift k from the v10 notebook (nine public scores), full precision
GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0]
W = [0.008172761879954695, 0.13116481847950048, 0.26721015776942314, 0.267056932957599, 0.1710924552535038, 0.0956936236265262,
     0.03459275261241694, 0.016549303552639335, 0.005725443007721319, 0.002368204370941116, 0.00031775367779660887,
     5.437858042612226e-05, 1.3770078232241008e-06, 3.722372808678639e-08]
pavg = lambda qq, lam=1.0: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))   # posterior-averaged shifted probability
'''

EXPCOST = r'''
def expected_cost(p, m, resid, flag, b, K=99):
    """Model-expected cost per vehicle: p = failure probability, log c | fail ~ m + out-of-fold residual quantiles."""
    M = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + 0.5) / K)[None, :])
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(flag == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * M.mean(1))
'''

n = NB("v17: Level probes on the v12_robust base (budget size, number of flags)")
n.header("v17", "`v17_bud125` (#16) and `v17_k135` (#17) → `outputs/submission_v17_*.csv` (each a single change vs `v12_robust`)",
         "**Question:** ranking changes (seed averaging, sensor mapping) moved the public score by a few hundred points in both directions, "
         "inside the noise. Level changes (the shift k) moved it by thousands. Two levels are still unprobed on the best base: are flagged "
         "budgets too small (the test depot's repairs may be more expensive than training's), and do we flag too few vehicles? One file per "
         "question, each a single change vs `v12_robust`:\n\n"
         "- **`v17_bud125`:** same flags, every flagged budget ×1.25.\n"
         "- **`v17_k135`:** shift factor k scaled by 1.35 (≈ 7% more flags), budgets from the same newsvendor rule.")
n.code(SHAPE); n.code(POSTERIOR)
n.md("### 1. Reproduce `v12_robust` (single-seed classifier, cost model without the drifted sensors)")
n.code(r'''
q = Platt().fit(fit_oof(dev[S]), y).predict(xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1])
m, resid = cost_model(CM9, Xf, lc_all, Xt)
p0 = pavg(q); fl12, bu12 = decide_pv(p0, m, resid)
r12 = pd.read_csv("../submissions/submission_v12_robust.csv")
assert (r12.flag.to_numpy() == fl12).all() and np.allclose(r12.budget, bu12.round(2), atol=0.011), "v12_robust not reproduced"
print(f"v12_robust reproduced: {int(fl12.sum())} flags")
''')
n.md("### 2. The two probes")
n.code(r'''
bu_a = np.where(fl12 == 1, bu12 * 1.25, bu12)
write_submission(te.id, fl12, bu_a, "v17_bud125")
fl_b, bu_b = decide_pv(pavg(q, 1.35), m, resid)
write_submission(te.id, fl_b, bu_b, "v17_k135")
for nm, fl, bu in (("v17_bud125", fl12, bu_a), ("v17_k135", fl_b, bu_b)):
    ref = pd.read_csv(f"../submissions/submission_{nm}.csv")
    assert (ref.flag.to_numpy() == fl).all() and np.allclose(ref.budget, bu.round(2), atol=0.011), f"{nm} not reproduced"
print(f"v17_k135: {int(fl_b.sum())} flags ({int(fl_b.sum() - fl12.sum()):+d} vs v12_robust, all added, none removed: "
      f"{bool((fl_b >= fl12).all())})")
''')
n.md(r"""
### 3. What did our own model expect these changes to do?

Before reading the public scores: under the model (posterior-averaged p, per-vehicle cost distribution) we compute each file's expected total cost on a random 30% of test, the size of the public subset. If the model already says a change is worth only a few dozen points, a public result within noise is what we should expect, and it says nothing against the model.
""")
n.code(EXPCOST + r'''
base = expected_cost(p0, m, resid, fl12, bu12)
for nm, fl, bu in (("v17_bud125", fl12, bu_a), ("v17_k135", fl_b, bu_b)):
    d = expected_cost(p0, m, resid, fl, bu) - base
    print(f"{nm}: model-expected change on the public 30%: {0.3 * d.sum():+8.1f}")
for k in (2.0, 3.0, 4.0):                                       # if the true shift were one fixed k instead of the posterior
    pk = shift(q, k); b0 = expected_cost(pk, m, resid, fl12, bu12)
    print(f"  at k={k:.0f}: bud125 {0.3 * (expected_cost(pk, m, resid, fl12, bu_a) - b0).sum():+7.1f}   "
          f"k135 {0.3 * (expected_cost(pk, m, resid, fl_b, bu_b) - b0).sum():+7.1f}")
''')
n.md(r"""
## Results and decisions

| file | change vs v12_robust | public score | difference |
|---|---|---|---|
| `v12_robust` | n/a | 16,839.86 | n/a |
| `v17_bud125` | flagged budgets ×1.25 | 16,826 | −14 |
| `v17_k135` | k × 1.35 (≈ 7% more flags) | 16,908 | +68 |

- **Both are ties.** The differences (−14 and +68) are far inside the ≈ 600-point noise between sibling files.
- **The model predicted small changes, both slightly worse:** +275 for bud125 and +117 for k135 on the public 30% (posterior over k). k135 landed where predicted (+68). bud125 came out ≈ 290 better than predicted: a hint, well inside noise, that test repairs cost a bit more than the training-fitted cost model expects.
- **Budget size is not a lever we can read.** Only ~1% of vehicles fail, so only the few flagged *failures* feel the budget; over-budgeting costs 0.02 per unit on every flagged vehicle, under-budgeting 0.20 on the failures, and near the newsvendor quantile the two roughly cancel. We keep the newsvendor budget.
- **Flag count is at its optimum.** The posterior over k came from nine earlier public scores; ≈ 7% more flags on top of it gave the small loss the model expected, consistent with sitting near the flat bottom of the cost-vs-k curve.
- **Conclusion:** every lever that the level (k, budget size) controls has now been used. The remaining gap to the leader can only come from *ranking*: which vehicles get flagged, i.e. the classifier's quality at the new depot.

**Caveats:** one public comparison per probe; the 70% private part could react differently, but neither probe gives a reason to prefer it over `v12_robust`.
""")
n.save("v17_level_probes.ipynb")
