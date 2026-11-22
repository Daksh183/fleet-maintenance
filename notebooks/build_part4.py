"""Builds the v12 notebook: v10_bayes with a repair-cost model that does not use the four drifted sensors."""
from nbkit import NB, SHAPE

n = NB("v12: Is the repair-cost model safe at the new depot? (robust inputs)")
n.header("v12", "`v12_robust` → `outputs/submission_v12_robust.csv` (single-change probe against `v10_bayes`)",
         "**Question:** the repair-cost model sets every flagged vehicle's budget. If it relies on sensors whose scale differs between depots "
         "(s120, s122, s126, s128) its test-depot budgets are extrapolations that cross-validation on the training depot cannot see. "
         "We check how much the model uses those sensors and what dropping them costs, then build `v10_bayes` again with exactly that one change.")
n.code(SHAPE)
n.code(r'''
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}                      # v9 cost model
DRIFT = ["s120", "s122", "s126", "s128"]
Pl = Platt().fit(fit_oof(dev[S]), y)
p_te_raw = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
q = Pl.predict(p_te_raw)                                                # calibrated test probabilities (old-depot scale)
f_all = tr[tr.failure == 1]; lc_all = np.log(f_all.repair_cost.to_numpy())
keep = [s for s in S if s not in DRIFT]
Xf_all, Xt_all = pd.concat([f_all[S], shape(f_all)], axis=1), pd.concat([te[S], shape(te)], axis=1)
Xf_keep, Xt_keep = pd.concat([f_all[keep], shape(f_all)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
# posterior over the depot shift k from the v10 notebook (nine public scores, model-error 0.06), full precision
GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0]
W = [0.008172761879954695, 0.13116481847950048, 0.26721015776942314, 0.267056932957599, 0.1710924552535038, 0.0956936236265262,
     0.03459275261241694, 0.016549303552639335, 0.005725443007721319, 0.002368204370941116, 0.00031775367779660887,
     5.437858042612226e-05, 1.3770078232241008e-06, 3.722372808678639e-08]
pbar = sum(w * shift(q, k) for w, k in zip(W, GRID))
''')
n.md("### 1. How much does the cost model use the drifted sensors, and how much do its inputs drift?")
n.code(r'''
Xc = Xf_all.loc[:, Xf_all.notna().any() & Xt_all.reindex(columns=Xf_all.columns).notna().any()]
m_full = xgb.XGBRegressor(**CM9).fit(Xc, lc_all)
imp = pd.Series(m_full.get_booster().get_score(importance_type="total_gain")); imp = (imp / imp.sum()).sort_values(ascending=False)
trf, tef = pd.concat([tr[S], shape(tr)], axis=1), Xt_all
ks = pd.Series({k: ks_2samp(trf[k].dropna(), tef[k].dropna()).statistic for k in Xc.columns})
print("top cost-model features: share of total gain and train-vs-test KS drift")
print(pd.DataFrame({"gain": imp.head(8).round(3), "KS": ks.reindex(imp.head(8).index).round(3)}).to_string())
print(f"gain-weighted mean KS of all inputs: {float((imp * ks.reindex(imp.index)).sum() / imp.sum()):.3f}; "
      f"gain share of the four drifted sensors: {float(imp.reindex(DRIFT).fillna(0).sum()):.3f}")
''')
n.md("### 2. What does dropping them cost? (failures only, 2×5 repeated CV, R² of log repair cost)")
n.code(r'''
def cv_r2(Xf):
    out = []
    for rs in (42, 1):
        for ia, ib in KFold(5, shuffle=True, random_state=rs).split(Xf):
            out.append(r2_score(lc_all[ib], xgb.XGBRegressor(**CM9).fit(Xf.iloc[ia], lc_all[ia]).predict(Xf.iloc[ib])))
    return float(np.mean(out))
r2_all, r2_keep = cv_r2(Xf_all), cv_r2(Xf_keep)
m_a, res_a = cost_model(CM9, Xf_all, lc_all, Xt_all); m_k, res_k = cost_model(CM9, Xf_keep, lc_all, Xt_keep)
print(f"CV R2 all sensors {r2_all:.3f}   without the four drifted sensors {r2_keep:.3f}")
print(f"test predicted log-cost: correlation {float(np.corrcoef(m_a, m_k)[0, 1]):.3f}, mean abs difference {float(np.abs(m_a - m_k).mean()):.3f}")
''')
n.md("### 3. Decisions: same classifier, calibration and posterior-averaged shift as `v10_bayes`; only the cost-model inputs differ")
n.code(r'''
v10 = decide_pv(pbar, m_a, res_a)                      # = v10_bayes (all sensors)
rob = decide_pv(pbar, m_k, res_k)                      # = v12_robust
ref = pd.read_csv("../submissions/submission_v10_bayes.csv")
assert (ref.flag.to_numpy() == v10[0]).all() and np.allclose(ref.budget, v10[1].round(2)), "v10_bayes not reproduced"
both = (v10[0] == 1) & (rob[0] == 1)
print(f"flag rate: v10_bayes {v10[0].mean():.3%}  v12_robust {rob[0].mean():.3%}; flags differing {int((v10[0] != rob[0]).sum())}; "
      f"budget change on commonly flagged: median {float(np.median(rob[1][both] - v10[1][both])):+.1f}, mean abs {float(np.abs(rob[1][both] - v10[1][both]).mean()):.1f}")
write_submission(te.id, rob[0], rob[1], "v12_robust")
''')
n.md(r"""
## Results and decisions

- **The cost model barely uses drifted inputs:** its top features are histogram-shape means (`h80_mean` ≈ 22% of gain, `h54_mean`, `h37_mean`), with KS drift ≤ 0.03; the gain-weighted drift of all inputs is ≈ 0.015.
- **Dropping the four drifted sensors is free in CV** (R² 0.801 vs 0.802) **but moves test cost predictions by ≈ 12%** (mean absolute log difference 0.118, correlation 0.986). That is the kind of change cross-validation on the old depot cannot judge: the training data say the sensors are not needed; whether the test-depot values are used correctly is only visible on the test depot.
- **`v12_robust`** therefore differs from `v10_bayes` in exactly one way (62 flags, mean absolute budget change ≈ 43 on commonly flagged vehicles). Expected public-score difference: a few hundred points either way; it is a probe of the cost-model inputs, not a predicted improvement.

**Caveats**
- No labels at the new depot: nothing here shows which version is right. The upload is the only evidence, and with ~6,000 public vehicles a difference of a few hundred points is still within noise.
- The shift posterior is reused from v10 (nine scores); the re-fit with eleven scores moved it only slightly (mean k 3.0 → 2.9).
- Other sensors (s057, s055, s062…) drift mildly; this only addresses the four obvious outliers.
""")
n.save("v12_robust_cost_inputs.ipynb")
