"""Builds notebooks v0..v5 (v2, v3, v3b, v4, v5 each correspond to uploaded Kaggle submissions)."""
from nbkit import EM, NB, SHAPE

# ------------------------------------------------------------------ v0
n = NB("v0: Trivial baselines")
n.header("v0", "none (local baseline only)",
         "**Question:** what do policies that need no model cost? They set the floor to beat and the ceiling for improvement.")
n.code(r'''
med = float(np.median(c[c > 0]))
policies = {
    "flag nobody": lambda cc: (np.zeros(len(cc)), np.zeros(len(cc))),
    "flag all, budget 0": lambda cc: (np.ones(len(cc)), np.zeros(len(cc))),
    f"flag all, budget = median failure cost ({med:.0f})": lambda cc: (np.ones(len(cc)), np.full(len(cc), med)),
    "oracle (perfect foresight)": lambda cc: ((cc > 0).astype(int), cc),
}
rows = []
for name, pol in policies.items():
    cv = [per_vehicle(c[v], *pol(c[v])) for _, v in F]
    rows.append((name, np.mean(cv), np.std(cv), per_vehicle(c_ho, *pol(c_ho))))
print(pd.DataFrame(rows, columns=["policy", "CV cost/veh", "CV std", "hold-out"]).round(3).to_string(index=False))
''')
n.md(r"""
## Results and caveats

| Policy | CV cost / vehicle | Hold-out |
|---|---|---|
| Flag nobody | 3.487 ± 0.692 | 3.506 |
| Flag all, budget 0 | 10.697 ± 0.139 | 10.701 |
| Flag all, budget = median failure cost (176) | 14.031 ± 0.136 | 14.040 |
| Oracle | 0.071 | 0.071 |

- **The bar to beat is 3.49 (flag nobody).** Flagging everyone costs 3× more because 99.3% of inspections would be wasted. The oracle (0.07) shows the room a model has.
- **Fold noise is large** (±0.69 on 3.49, i.e. 20%): costs are heavy-tailed, so one 8,000-unit failure moves a fold. From here on, versions are compared with **paired fold differences** (same folds), not by overlapping mean ± std.
- These numbers use the *training* depot's failure rate and cost distribution; the test depot differs (v1, v7).
- The hold-out is touched only at v0, v4 and the final check.
""")
n.save("v0_baselines.ipynb")

# ------------------------------------------------------------------ v1
n = NB("v1: Exploratory data analysis and depot-shift diagnosis")
n.header("v1", "none (analysis; its fixes are used by every later version)",
         "**Questions:** what does the target look like, is the label reliable (censoring), how are missing values related to failure, and how do "
         "the train depot and the test depot differ? The fixes found here (histogram realignment, dropped columns) are in `load()` in every notebook.")
n.code(r'''
tr0, te0, S0 = load(raw=True); f = tr0[tr0.failure == 1]
print(f"failures: {len(f)} / {len(tr0)} = {tr0.failure.mean():.3%}")
print("repair cost quantiles 5/25/50/75/90/95/99:", f.repair_cost.quantile([.05, .25, .5, .75, .9, .95, .99]).round(0).tolist(), " max", f.repair_cost.max())
top = f.repair_cost.sort_values(ascending=False)
print(f"top 10% of failures carry {top.iloc[:len(top)//10].sum() / top.sum():.1%} of all repair cost")
print(f"flag threshold with mean cost mu={f.repair_cost.mean():.0f}: p*mu > 12.5  ->  p > {12.5 / f.repair_cost.mean():.2%}")
''')
n.md("### Censoring: `followup_days`")
n.code(r'''
bins = pd.cut(tr0.followup_days, [0, 90, 180, 270, 364, 365])
print(tr0.groupby(bins, observed=True).failure.agg(["mean", "size"]).round(4).to_string())
print(f"\nhealthy vehicles with < 365 days of follow-up: {((tr0.failure == 0) & (tr0.followup_days < 365)).sum() / (tr0.failure == 0).sum():.1%}")
''')
n.md("### Missing values")
n.code(r'''
nan_tr, nan_te = tr0[S0].isna().mean(axis=1), te0[S0].isna().mean(axis=1)
print(f"mean missing fraction: train {nan_tr.mean():.3f}  test {nan_te.mean():.3f}")
q = pd.qcut(nan_tr.rank(method="first"), 4, labels=["Q1 (least missing)", "Q2", "Q3", "Q4 (most missing)"])
print("failure rate by missingness quartile:", tr0.groupby(q, observed=True).failure.mean().round(4).to_dict())
print("constant columns in train:", [col for col in S0 if tr0[col].nunique(dropna=True) <= 1])
print("columns 100% missing in test:", [col for col in S0 if te0[col].isna().all()])
''')
n.md("### Depot shift: adversarial validation and the shifted histogram")
n.code(r'''
def adversarial(a, b, cols, tag):
    X = pd.concat([a[cols], b[cols]], ignore_index=True); z = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
    p = cross_val_predict(clf, X, z, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    imp = pd.Series(clf.fit(X, z).feature_importances_, index=cols).sort_values(ascending=False)
    print(f"[{tag}] adversarial AUC = {roc_auc_score(z, p):.4f}; most important:", imp.head(6).round(3).to_dict()); return p

adversarial(tr0, te0, S0, "raw files")
print("\nKS(train bin k, test bin k)   :", [round(float(ks_2samp(tr0[HISTB[i]].dropna(), te0[HISTB[i]].dropna()).statistic), 3) for i in range(9)])
print("KS(train bin k, test bin k+1) :", [round(float(ks_2samp(tr0[HISTB[i]].dropna(), te0[HISTB[i + 1]].dropna()).statistic), 3) for i in range(9)], " <- test histogram is shifted by one bin")
''')
n.code(r'''
cols = [col for col in S if col != HISTB[-1]]            # tr/te: after the fixes
ks = pd.Series({col: ks_2samp(tr[col].dropna(), te[col].dropna()).statistic for col in cols if tr[col].notna().sum() > 100 and te[col].notna().sum() > 100})
print("largest remaining KS after fixes:", ks.sort_values(ascending=False).head(8).round(3).to_dict())
adversarial(tr, te, cols, "after fixes")
adversarial(tr, te, [col for col in cols if col not in ("s120", "s122", "s126", "s128")], "after fixes, minus s120/s122/s126/s128")
''')
n.md(r"""
## Findings and caveats

- **Target:** 397 failures out of 56,000 (**0.71%**). Repair cost is heavy-tailed (median 176, mean 492, p99 6,847, max 8,000); the top 10% of failures carry 53% of the cost. With μ = 492 the flag rule is **p > ~2.5%**, about 3.6× the base rate, far below 0.5.
- **Censoring:** 22% of healthy vehicles were followed for < 365 days. The failure rate rises with follow-up (0.07% for ≤ 90 days → 1.9% for 271-364 days), so records of failing vehicles appear to stop near the failure: `followup_days` is partly determined *by* the outcome. It is **never used as a feature** (and it is not in test).
- **Missingness predicts failure:** the most-missing quartile fails at 2.06% vs 0.02-0.6% for the others. NaN stays NaN (XGBoost handles it natively).
- **Depot shift:** adversarial AUC on the raw files is **1.000**. Two causes are data errors and are fixed: `s047` is 100% missing in test (dropped; `s146` is constant, dropped), and the 10-bin histogram `s054-s063` is **shifted by one bin in test** (test bin k+1 matches train bin k, KS ≤ 0.07; realigned). After the fixes the AUC is still **0.9999**: `s120` and `s122` are level-shifted (a counter that runs higher, ~×2-2.7) and `s126`/`s128` differ in a way a rounding hypothesis did not explain; without the four, AUC is 0.937 (a diffuse shift remains).

**Caveats**
- The realignment is inferred from distributions, not from documentation. It is the simplest explanation (a shift of exactly one bin matches all 9 pairs; two does not), but it cannot be confirmed.
- Adversarial AUC shows the depots *differ*, not that the differences *matter for failure*. Whether to drop the drifted sensors is decided on cost (v3b), not on AUC.
- With 397 positives every per-group failure rate has wide confidence intervals (the ≤ 90-day group has only 2 failures).
""")
n.save("v1_eda.ipynb")

# ------------------------------------------------------------------ v2
n = NB("v2: Plain XGBoost, flag if p > 0.5")
n.header("v2", "**#1** `sub01-2026-11-19-v2` → `outputs/submission_v2.csv` (public score 74,933.17)",
         "**Hypothesis:** a gradient-boosted classifier that ranks failures well (AUC ~0.98) should beat the baselines, even with the naive "
         "decision rule (flag when p > 0.5, budget = mean repair cost). This is the first real model and the reference for what the decision rule adds in v3.")
n.code(r'''
oof = fit_oof(dev[S])
p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
print(f"OOF AUC {roc_auc_score(y, oof):.4f}   PR-AUC {average_precision_score(y, oof):.4f}   (base rate {y.mean():.4f})")
mean_cost = float(c[c > 0].mean())
naive = fold_cost((oof > 0.5).astype(int), np.full(len(y), mean_cost))
print(f"v2 (p > 0.5, budget = mean cost {mean_cost:.0f}): {naive.mean():.3f} ± {naive.std():.3f} per vehicle; flagged {(oof > 0.5).mean():.2%}")
for t in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5):          # threshold sweep (budget 0), for reference only
    s = fold_cost((oof > t).astype(int), np.zeros(len(y))); print(f"   threshold p > {t:<5}: {s.mean():.3f} ± {s.std():.3f}")
write_submission(te.id, p_te > 0.5, np.full(len(te), mean_cost), "v2")
''')
n.md(r"""
## Results and caveats

| | CV cost / vehicle | Public score |
|---|---|---|
| Flag nobody (v0) | 3.487 | n/a |
| **v2** | **2.501 ± 0.833** | **74,933** |

- The classifier ranks well (OOF **AUC 0.984, PR-AUC 0.526** at a 0.7% base rate) and beats flag-nobody, but only modestly: the threshold of 0.5 is far too high. The sweep shows lower thresholds are much cheaper; v3 derives the right threshold from the cost function.
- The 74,933 public score is the worst of our uploads: it is the baseline the later versions are measured against.

**Caveats:** the budget (mean cost, 492) is a placeholder; budgets are derived properly in v3. Fold noise is large (±0.83). The model is fitted on the old depot; its probabilities may not transfer to the new one (v7).
""")
n.save("v2_naive_xgboost.ipynb")

# ------------------------------------------------------------------ v3
n = NB("v3: Cost-derived flag rule and newsvendor budget")
n.header("v3", "**#2** `sub02-2026-11-19-v3` → `outputs/submission_v3.csv` (public score 27,760.14)",
         "**Hypothesis:** the model was fine, the decision rule was wrong. With costs this asymmetric the optimal threshold is ~2.5% (not 0.5) "
         "and the optimal budget is the 0.909 quantile of repair cost. Same classifier as v2, new decision layer.")
n.md(r"""
### The decision rule

- **Budget (newsvendor):** shortage costs 0.20/unit, unused budget 0.02/unit, so the optimal budget is the $q^*=0.20/0.22=0.909$ quantile. With failure probability $p$ the cost is 0 w.p. $1-p$, so $b^*=0$ if $p\le 0.0909$ else $b^*=F^{-1}(1-0.0909/p)$ ($F$ = repair-cost distribution *given* failure).
- **Flag** iff $E[\text{cost}\mid\text{flag},b^*]<p\,\mu$. With $b^*=0$ that is $p\mu>12.5$, i.e. $p\gtrsim 2.5\%$.
- $F$ here is **global**: the empirical cost distribution of the training failures (fold-safe: only the training folds' failures).
""")
n.code(r'''
oof = fit_oof(dev[S])
p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
naive = fold_cost((oof > 0.5).astype(int), np.full(len(y), c[c > 0].mean()))
fl, bu = np.zeros(len(y), int), np.zeros(len(y))
for a, b in F:
    fl[b], bu[b] = decide_global(oof[b], c[a][c[a] > 0])
v3 = fold_cost(fl, bu)
print(f"v2 naive : {naive.mean():.3f} ± {naive.std():.3f}")
print(f"v3       : {v3.mean():.3f} ± {v3.std():.3f}   paired gain vs v2 per fold {np.round(v3 - naive, 3)}   flagged {fl.mean():.2%}")
f3, b3 = decide_global(p_te, c_all[c_all > 0])
write_submission(te.id, f3, b3, "v3")
''')
n.md(r"""
## Results and caveats

| | CV cost / vehicle | Public score |
|---|---|---|
| v2 (p > 0.5) | 2.501 | 74,933 |
| **v3** | **1.242 ± 0.387** | **27,760** (−63%) |

- Same classifier, new decision rule: cost halves in CV (paired gain in all 5 folds) and the public score drops 63%. **The decision rule is the largest single lever** in the whole project.
- Test flag rate is 5.4% (vs 1.6% for v2), as expected from a ~2.5% threshold.

**Caveats / what v3 gets wrong (motivating v4, v5, v7):**
1. The decision depends on $p$ itself near 2.5%, so it needs **calibrated** probabilities (v4).
2. $F$ is the same for every vehicle, although repair cost turns out to be predictable from the sensors (v5).
3. Everything assumes the training depot's failure rate; the new depot appears to fail more often (v7).
4. The public score is a total over an unknown number of vehicles, so its absolute level can't be turned into a per-vehicle cost; only ratios between submissions are meaningful.
""")
n.save("v3_cost_rule.ipynb")

# ------------------------------------------------------------------ v3b
n = NB("v3b: v3 without the four most-drifted sensors (rejected)")
n.header("v3b", "**#3** `sub03-2026-11-19-v3b` → `outputs/submission_v3b.csv` (public score 29,529.75, **worse than v3, reverted**)",
         "**Hypothesis:** `s120`, `s122`, `s126`, `s128` drift between depots (v1); a model that relies on them may transfer badly to the new depot, "
         "so removing them should help on test. This is a diagnostic upload for whether the drift matters.")
n.code(r'''
DRIFT = ["s120", "s122", "s126", "s128"]; Sb = [col for col in S if col not in DRIFT]
oof3, oofb = fit_oof(dev[S]), fit_oof(dev[Sb])
p_te_b = xgb.XGBClassifier(**PARAMS).fit(tr[Sb], tr.failure).predict_proba(te[Sb])[:, 1]
p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
def v3_costs(o):
    fl, bu = np.zeros(len(y), int), np.zeros(len(y))
    for a, b in F: fl[b], bu[b] = decide_global(o[b], c[a][c[a] > 0])
    return fold_cost(fl, bu)
cv3, cvb = v3_costs(oof3), v3_costs(oofb)
print(f"AUC  v3 {roc_auc_score(y, oof3):.4f}   v3b {roc_auc_score(y, oofb):.4f}")
print(f"CV   v3 {cv3.mean():.3f} ± {cv3.std():.3f}   v3b {cvb.mean():.3f} ± {cvb.std():.3f}   paired v3b-v3 {np.round(cvb - cv3, 3)}")
print(f"mean p on test: v3 {p_te.mean():.4f}   v3b {p_te_b.mean():.4f}   (dev OOF mean {oof3.mean():.4f})")
f, b = decide_global(p_te_b, c_all[c_all > 0]); write_submission(te.id, f, b, "v3b")
''')
n.md(r"""
## Results and decision

| | CV cost / vehicle | Public score |
|---|---|---|
| v3 | 1.242 ± 0.387 | 27,760 |
| v3b (no drifted sensors) | 1.216 ± 0.410 | 29,530 (+6%) |

- CV says slightly better (−0.03, well inside fold noise); the public score says 6% worse. **No evidence that dropping the drifted sensors helps → reverted.** All sensors stay.
- The model without the four sensors still predicts the same elevated failure level on test (mean p 0.0175 vs 0.0183), so the extra predicted failures on the new depot are **not** an artifact of those sensors (this feeds into v7).

**Caveats:** a public difference of 6% on ~6,000 vehicles with few failures is itself within noise; the decision rests on "no benefit in either measure", not on the public score being proof of harm.
""")
n.save("v3b_drop_drifted_sensors.ipynb")

# ------------------------------------------------------------------ v4
n = NB("v4: Probability calibration")
n.header("v4", "**#4** `sub04-2026-11-20-v4_platt` → `outputs/submission_v4_platt.csv` (public score 23,875.67). Isotonic variant built but not uploaded.",
         "**Hypothesis:** the flag rule depends on $p$ itself (flag when $p\\mu>12.5$), so miscalibrated probabilities cost money. Calibrate the raw "
         "probabilities (Platt or isotonic) before applying the v3 decision rule.")
n.code(r'''
oof = fit_oof(dev[S])
p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
print("reliability on OOF predictions (quantile bins of the raw p): predicted vs observed failure rate")
edges = np.quantile(oof, [0, .5, .8, .9, .95, .98, .99, 1.0])
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (oof >= lo) & (oof <= hi); print(f"  p in [{lo:.4f}, {hi:.4f}]  n={m.sum():5d}  predicted {oof[m].mean():.4f}  observed {y[m].mean():.4f}")
''')
n.md("### Cross-fitted comparison (paired per fold)\nEach fold's calibrator is fitted on the **other** folds' out-of-fold predictions, so no fold is calibrated on itself.")
n.code(r'''
def cv_costs(kind):
    pc, costs = oof.copy(), []
    for a, b in F:
        if kind != "raw": pc[b] = make_cal(kind).fit(oof[a], y[a]).predict(oof[b])
        fl, bu = decide_global(pc[b], c[a][c[a] > 0]); costs.append(per_vehicle(c[b], fl, bu))
    return np.array(costs), pc
res = {k: cv_costs(k) for k in ("raw", "platt", "iso")}
print("method    CV cost/veh      paired delta vs raw (per fold)          folds better   Brier     log-loss")
for k, (costs, pc) in res.items():
    d = costs - res["raw"][0]
    print(f"{k:6s}  {costs.mean():.4f}±{costs.std():.3f}   {str(np.round(d, 3)):38s}  {int((d < 0).sum())}/5      {brier_score_loss(y, pc):.5f}   {log_loss(y, np.clip(pc, 1e-6, 1 - 1e-6)):.5f}")
''')
n.md("### Hold-out check (11,200 vehicles, 79 failures): the first of the three planned uses of the locked hold-out")
n.code(r'''
p_ho = xgb.XGBClassifier(**PARAMS).fit(dev[S], y).predict_proba(ho[S])[:, 1]
print(f"hold-out: n={len(ho)}, failures={y_ho.sum()}")
for k in ("raw", "platt", "iso"):
    pc = p_ho if k == "raw" else make_cal(k).fit(oof, y).predict(p_ho)
    fl, bu = decide_global(pc, c[c > 0]); print(f"  {k:6s} {per_vehicle(c_ho, fl, bu):.4f} per vehicle   flagged {fl.mean():.2%}")
''')
n.md("### Submissions: calibrator fitted on all dev OOF predictions, applied to the test probabilities")
n.code(r'''
for k in ("platt", "iso"):
    pt = make_cal(k).fit(oof, y).predict(p_te)
    fl, bu = decide_global(pt, c_all[c_all > 0]); print(f"{k}: mean test p {pt.mean():.4f} (raw {p_te.mean():.4f}); ", end="")
    write_submission(te.id, fl, bu, f"v4_{k}")
''')
n.md(r"""
## Results and caveats

The raw model is **under-confident exactly where decisions are made**: for raw p between 0.003 and 0.028 the observed failure rate is 2.5% against 1.0% predicted, and for 0.028-0.15 it is 13.8% against 6.7%. Above 0.15 it is well calibrated. So raw p under-flags vehicles near the 2.5% threshold.

| Method | CV cost / vehicle | Paired Δ vs raw | Folds better | Hold-out | Public |
|---|---|---|---|---|---|
| raw (v3) | 1.242 ± 0.387 | n/a | n/a | 0.906 | 27,760 |
| **Platt** | **1.107 ± 0.220** | −0.135 | 4/5 | 0.955 | **23,876** |
| isotonic | 1.052 ± 0.258 | −0.190 | 5/5 | 1.008 | not uploaded |

- **CV and hold-out disagreed** (the hold-out has only 79 failures: 0.05-0.10 is one or two large failures, so it is noise, but it is not support either). The **public score then favoured calibration** (27,760 → 23,876, −14%), and the same pattern repeated in v5 (raw 26,061 vs Platt 22,969). Isotonic is riskier with few positives; Platt is smoother and was kept.

**Caveats:** calibration is fitted to the *training* depot's failure rate. On test the model already predicts ~3× the failures; calibration raises that further (mean p 0.018 → 0.020) and test flag rate from 5.4% to 6.7% (Platt). That the public score improved suggests the test depot really does fail more often (the subject of v7). Test probabilities come from a model fit on all data while calibrators were fit on OOF predictions of 80%-data models, so test p is slightly more confident than what the calibrator saw. The hold-out must not drive tuning any more.
""")
n.save("v4_calibration.ipynb")

# ------------------------------------------------------------------ v5
n = NB("v5: Per-vehicle repair-cost model")
n.header("v5", "**#5** `sub05-2026-11-20-v5_raw` → `outputs/submission_v5_raw.csv` (26,060.57) and **#6** `sub06-2026-11-20-v5_platt` → "
               "`outputs/submission_v5_platt.csv` (22,968.55). Isotonic variant built, not uploaded.",
         "**Hypothesis:** v3/v4 use one repair-cost distribution for every vehicle. If repair cost is predictable from the sensors, a per-vehicle cost "
         "distribution improves **both** decisions: the budget (per-vehicle newsvendor quantile) and the flag (per-vehicle expected cost).")
n.md(r"""
### The cost model
XGBoost (depth 3, 200 trees) on **log(repair cost)**, trained on failing vehicles only. Each vehicle's cost distribution is `log c | fail ~ m_i + residuals`, where the residuals are **out-of-fold** (inner 5-fold CV) so the spread is honest, and the back-transform uses residual quantiles (no exp(mean) bias). Nested: inside each outer fold, the cost model and its residuals use only that fold's training failures.

Ablation per probability variant, paired by fold: **A** = global F for flag and budget (v3/v4); **B** = per-vehicle flag, global-F budget; **D** = per-vehicle flag and budget (full v5).
""")
n.code(r'''
oof = fit_oof(dev[S])
p_te_raw = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
variants = ("raw", "platt", "iso")
cost = {(v, m): [] for v in variants for m in "ABD"}; cover, r2s = [], []
for a, b in F:
    fa, fb = a[y[a] == 1], b[y[b] == 1]
    m_b, resid = cost_model(CM, dev.loc[fa, S], np.log(c[fa]), dev.loc[b, S])
    fc = c[a][c[a] > 0]
    mf = m_b[y[b] == 1]
    r2s.append(r2_score(np.log(c[fb]), mf)); cover.append(np.mean(np.log(c[fb]) <= mf + np.quantile(resid, Q_STAR)))
    for v in variants:
        p = oof[b] if v == "raw" else make_cal(v).fit(oof[a], y[a]).predict(oof[b])
        for m, (fl, bu) in (("A", decide_global(p, fc)), ("B", decide_pv(p, m_b, resid, fc)), ("D", decide_pv(p, m_b, resid))):
            cost[(v, m)].append(per_vehicle(c[b], fl, bu))
print(f"cost model R2 on held-out failures (log scale): {np.round(r2s, 2)}  mean {np.mean(r2s):.2f}")
print(f"budget quantile coverage (target {Q_STAR:.3f}): {np.round(cover, 2)}  mean {np.mean(cover):.3f}\n")
print("variant rule  CV cost/veh        paired delta vs A (per fold)             folds better")
for v in variants:
    A = np.array(cost[(v, "A")])
    for m in "ABD":
        x = np.array(cost[(v, m)]); d = x - A
        print(f"{v:6s}  {m}    {x.mean():.4f}±{x.std():.3f}   {str(np.round(d, 3)):40s} {int((d < 0).sum())}/5")
''')
n.md("### Submissions: cost model on all training failures; calibrators on all dev OOF predictions")
n.code(r'''
f_all = tr[tr.failure == 1]
m_te, resid_all = cost_model(CM, f_all[S], np.log(f_all.repair_cost.to_numpy()), te[S])
print(f"test predicted failure cost: median {np.median(np.exp(m_te)):.0f}   (train failures: median {f_all.repair_cost.median():.0f})")
for v in variants:
    pt = p_te_raw if v == "raw" else make_cal(v).fit(oof, y).predict(p_te_raw)
    fl, bu = decide_pv(pt, m_te, resid_all); print(f"{v}: ", end=""); write_submission(te.id, fl, bu, f"v5_{v}")
''')
n.md(r"""
## Results and caveats

| p | A (global F) | B (per-vehicle flag) | **D (full v5)** | D−A | folds better | Public |
|---|---|---|---|---|---|---|
| raw | 1.242 | 1.081 | 0.999 ± 0.316 | −0.243 | 5/5 | 26,061 (#5) |
| **Platt** | 1.107 | 0.994 | **0.904 ± 0.235** | −0.203 | 5/5 | **22,969 (#6)** |
| isotonic | 1.052 | 1.016 | 0.932 ± 0.230 | −0.120 | 5/5 | not uploaded |

- **Repair cost is predictable from the sensors:** R² **0.68** on log scale on held-out failures (0.60-0.71 per fold; RMSE 0.95 vs 1.62 for a constant), driven mostly by `s080-s085`. It is unrelated to the failure score (corr −0.08) and to follow-up (0.05).
- **The budget does what it should:** the 0.909 quantile covers **91.5%** of held-out failures (target 90.9%).
- **Most of the gain is the flag decision** (A→B), the rest the budget (B→D, ~0.08-0.09). Calibration still helps on top (raw 0.999 → Platt 0.904), and on the public set (26,061 vs 22,969).
- Cost-driving sensors do not drift between depots (KS 0.002-0.009), so the cost model transfers; the depot shift is in failure *probability*.

**Caveats:** only 318 failures train the cost model per fold; repair costs are **simulated**, which probably explains the high R² (a real process would be noisier); residuals are assumed equally spread for all vehicles (tested in v9: no gain from modelling it); the failure probability is still the larger source of error.
""")
n.save("v5_per_vehicle_cost.ipynb")
