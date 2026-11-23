"""Builds notebooks v6..v9 (experiments; v7-v9 files are prepared for upload, none uploaded yet)."""
from nbkit import EM, NB, SHAPE

PIPE = r'''
def pipeline_costs(oofs, base="A", cm=CM, X=None, repeats=None):
    """Paired CV cost for several failure-probability candidates through the v5 pipeline
    (nested cost model fitted once per fold, Platt fitted on the other folds' OOF)."""
    X = dev[S] if X is None else X
    cost = {k: [] for k in oofs}
    for a, b in F:
        fa = a[y[a] == 1]
        m_b, resid = cost_model(cm, X.iloc[fa], np.log(c[fa]), X.iloc[b])
        for k, o in oofs.items():
            fl, bu = decide_pv(make_cal("platt").fit(o[a], y[a]).predict(o[b]), m_b, resid)
            cost[k].append(per_vehicle(c[b], fl, bu))
    return {k: np.array(v) for k, v in cost.items()}

def report(cost, base):
    A = cost[base]
    for k, x in cost.items():
        d = x - A; print(f"{k:12s} CV cost {x.mean():.4f} ± {x.std():.3f}   paired vs {base}: {d.mean():+.4f}   per fold {np.round(d, 3)}   better {int((d < 0).sum())}/{len(d)}")
'''

# ------------------------------------------------------------------ v6
n = NB("v6: Follow-up handling and missingness features (no gain, reverted)")
n.header("v6", "none (experiment; reverted)",
         "**Hypotheses.** (B) the missing-value count predicts failure, so explicit NaN-count features might help the trees. (C/D) healthy vehicles with "
         "short follow-up have unreliable labels (v1): dropping them (C: follow-up < 180 d) or down-weighting them (D: weight min(followup,365)/365) "
         "should give cleaner training labels. (E) B + D. Only the **classifier** changes; the v5 pipeline is held fixed and every variant is scored on "
         "**all** validation vehicles. `followup_days` shapes training rows/weights only, never a feature.")
n.code(PIPE)
n.code(r'''
fu = dev.followup_days.to_numpy()
def with_miss(df):
    X = df[S].copy(); S2 = [s for s in S if s != "s063"]            # s063 is NaN by construction in test
    X["n_nan"] = df[S2].isna().sum(axis=1)
    for i in range(0, len(S2), 20): X[f"nan_blk{i // 20}"] = df[S2[i:i + 20]].isna().sum(axis=1)
    return X
Xb = with_miss(dev)
W = lambda a: np.where(y[a] == 1, 1.0, np.minimum(fu[a], 365) / 365)
variants = {"A": (dev[S], None, None), "B": (Xb, None, None),
            "C": (dev[S], lambda a: a[(y[a] == 1) | (fu[a] >= 180)], None),
            "D": (dev[S], None, W), "E": (Xb, None, W)}
oofs = {}
for k, (X, sel, wf) in variants.items():
    o = np.zeros(len(dev))
    for a, b in F:
        aa = sel(a) if sel else a
        o[b] = xgb.XGBClassifier(**PARAMS).fit(X.loc[aa], y[aa], sample_weight=None if wf is None else wf(aa)).predict_proba(X.loc[b])[:, 1]
    oofs[k] = o
    print(f"{k}: OOF AUC {roc_auc_score(y, o):.4f}  PR-AUC {average_precision_score(y, o):.4f}  mean p {o.mean():.4f}")
cost = pipeline_costs(oofs); print(); report(cost, "A")
''')
n.md(r"""
## Results and decision: revert to A

| Variant | OOF AUC | CV cost | Paired Δ vs A | Folds better |
|---|---|---|---|---|
| A (v5 Platt) | 0.9840 | 0.904 ± 0.235 | n/a | n/a |
| B: + NaN counts | 0.9846 | 0.915 | +0.011 | 2/5 |
| C: drop follow-up < 180 | 0.9843 | 0.888 ± 0.178 | −0.016 | 2/5 |
| D: weight by follow-up | 0.9841 | 0.881 ± 0.181 | −0.023 | 2/5 |
| E: B + D | 0.9841 | 0.923 | +0.019 | 1/5 |

- **No variant is a real improvement.** C and D look better on average, but that comes from **one fold** (−0.17 / −0.18 in fold 5) while they are worse in 3 of the other 4 folds: one lucky large failure, not a gain. B does not help because trees already see NaN natively.
- AUC is flat for all variants (0.984-0.985).

**Caveats:** with 397 failures, swings of 0.1-0.2 between folds are normal; a repeated-CV test would be needed to settle C/D, and the potential gain is too small to justify it. Dropping/weighting changes the training base rate (mean p 0.0058 → 0.0077 for C); Platt absorbs this, but only for the training depot. **Decision: revert; no submission.**
""")
n.save("v6_followup_missingness.ipynb")

# ------------------------------------------------------------------ v7
n = NB("v7: Depot shift: why test looks riskier, and how to correct for it")
n.header("v7", "**#8** `sub08-2026-11-21-v7_k4` → `outputs/submission_v7_k4.csv` (public score 19,067.17; uploaded by mistake instead of `v9_k4`, but informative: it isolates the shift on the v5 cost model). `v7_em` was not uploaded. v7 is superseded by v8/v9, which add a better cost model to the same correction",
         "**Question:** the model predicts ~3× more failures on the new depot (mean p 0.018 vs 0.006). Is that real, and how should decisions change? "
         "Tested: importance weighting, a prior-shift (EM) correction validated on a simulated shift, consistency with the public scores already "
         "obtained, the cost of guessing the shift wrongly, and whether a better classifier is the answer instead.")
n.code(EM)
n.code(PIPE)
n.md("### 1. The setting")
n.code(r'''
oof = fit_oof(dev[S]); p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
pl = Platt().fit(oof, y); q_dev, q_te = pl.predict(oof), pl.predict(p_te)
print(f"train failure rate {tr.failure.mean():.4f}")
print(f"mean p   raw: dev OOF {oof.mean():.4f}  test {p_te.mean():.4f}     Platt: dev {q_dev.mean():.4f}  test {q_te.mean():.4f}")
for t in (0.025, 0.1, 0.5): print(f"  share with Platt p > {t}: dev {np.mean(q_dev > t):.4f}   test {np.mean(q_te > t):.4f}")
''')
n.md("### 2. Where does the extra test risk come from? (SHAP contribution, test minus dev)")
n.code(r'''
bst = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).get_booster()
cd = bst.predict(xgb.DMatrix(dev[S]), pred_contribs=True).mean(0); ct = bst.predict(xgb.DMatrix(te[S]), pred_contribs=True).mean(0)
diff = pd.Series(ct - cd, index=S + ["bias"]).drop("bias").sort_values(ascending=False)
DR = ["s120", "s122", "s126", "s128"]
print(f"mean logit test - dev: {diff.sum():+.3f}   of which the 4 drifted sensors: {diff[DR].sum():+.3f}")
print("largest contributors:", diff.head(6).round(3).to_dict())
''')
n.md("### 3. Importance weighting (p(test|x) / p(train|x)): is there enough overlap?")
n.code(r'''
Sa = [s for s in S if s not in DR]
Xa = pd.concat([tr[Sa], te[Sa]]); z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
adv = cross_val_predict(xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=SEED), Xa, z,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:len(tr), 1]
for clip in (None, 20, 5):
    w = adv / (1 - adv + 1e-9)
    if clip: w = np.minimum(w / w.mean(), clip)
    wf = w[tr.failure.to_numpy() == 1]
    print(f"clip {clip}: effective sample size {w.sum() ** 2 / (w ** 2).sum():,.0f} of {len(tr):,}   failures {wf.sum() ** 2 / (wf ** 2).sum():.0f} of {len(wf)}")
''')
n.md(r"""
### 4. Prior-shift correction (Saerens-Latinne-Decaestecker EM)
Assumes only the failure rate changed ($p(x|y)$ fixed). EM estimates the test failure odds from the test predictions alone; each calibrated probability is then shifted by the odds factor $k$: $p'=kp/(1-p+kp)$.
""")
n.code(r'''
pi_tr = y.mean()
print(f"EM odds factor on test (v2 classifier): k = {em_odds_factor(q_te, pi_tr):.2f}   (sanity: on dev OOF itself k = {em_odds_factor(q_dev, pi_tr):.2f}, should be ~1)")
''')
n.md("**Validation on a simulated shift inside dev** (keep all failures, subsample healthy vehicles at 1/k_true: pure label shift). Compared: no correction, EM, and the oracle that knows k_true.")
n.code(r'''
rng = np.random.default_rng(SEED); parts = []
for a, b in F:
    fa = a[y[a] == 1]; m_b, resid = cost_model(CM, dev.loc[fa, S], np.log(c[fa]), dev.loc[b, S])
    parts.append((b, make_cal("platt").fit(oof[a], y[a]).predict(oof[b]), m_b, resid, y[a].mean()))
print("k_true   EM k (mean [min,max])      cost/veh: none    EM     oracle    EM gain vs none (folds x reps better)")
for kt in (1, 2, 4, 6):
    est, res = [], {"none": [], "EM": [], "oracle": []}
    for rep in range(4):
        for b, q_b, m_b, resid, pit in parts:
            keep = (y[b] == 1) | (rng.random(len(b)) < 1 / kt)
            q, mm, cc = q_b[keep], m_b[keep], c[b][keep]; kh = em_odds_factor(q, pit); est.append(kh)
            for name, kk in (("none", 1), ("EM", kh), ("oracle", kt)):
                fl, bu = decide_pv(shift(q, kk), mm, resid); res[name].append(per_vehicle(cc, fl, bu))
    r = {k: np.array(v) for k, v in res.items()}; g = r["EM"] - r["none"]
    print(f"{kt:>5}   {np.mean(est):5.2f} [{np.min(est):.2f}, {np.max(est):.2f}]            {r['none'].mean():.3f}   {r['EM'].mean():.3f}   {r['oracle'].mean():.3f}     {g.mean():+.3f} ({int((g < 0).sum())}/{len(g)})")
''')
n.md(r"""
### 5. Which shift is consistent with the public scores we already have?
Using our own failure probabilities and the v5 cost model, compute each already-submitted file's **expected** score on a random 30% of test under different assumed shifts k, and compare with the observed public scores. This uses the leaderboard only to estimate **one scalar** (the test failure level), not to fit anything per vehicle.
""")
n.code(r'''
f_all = tr[tr.failure == 1]; m5, resid5 = cost_model(CM, f_all[S], np.log(f_all.repair_cost.to_numpy()), te[S])
M = np.exp(m5[:, None] + np.quantile(resid5, (np.arange(99) + .5) / 99)[None, :]); mu = M.mean(1)
def exp_cost(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu)
obs = {"v2": 74933, "v3": 27760, "v3b": 29530, "v4_platt": 23876, "v5_raw": 26061, "v5_platt": 22969}
subs = {k: pd.read_csv(f"../submissions/submission_{k}.csv") for k in obs}
beliefs = {"k=1 (Platt)": q_te, **{f"k={k}": shift(q_te, k) for k in (2, 4, 6.5, 10)}}
tab = pd.DataFrame({b: {s: 0.3 * exp_cost(d.flag.to_numpy(), d.budget.to_numpy(), p).sum() for s, d in subs.items()} for b, p in beliefs.items()}).round(0)
tab.insert(0, "observed", pd.Series(obs)); print(tab.to_string())
o = pd.Series(obs); L = np.log(tab.drop(columns="observed").astype(float)).sub(np.log(o), axis=0)
print("\nRMS log error of the levels:      ", L.pow(2).mean().pow(.5).round(3).to_dict())
print("scale-free (demeaned) RMS:        ", (L - L.mean()).pow(2).mean().pow(.5).round(3).to_dict())
print("implied public-set size factor (1 = 30% of test):", np.exp(-L.mean()).round(2).to_dict())
''')
n.md("### 6. Regret of guessing k wrongly (expected cost / vehicle on test; rows = k used to build the decisions, columns = the true k)")
n.code(r'''
def build_cost(kb, kt):
    fl, bu = decide_pv(shift(q_te, kb), m5, resid5); return exp_cost(fl, bu, shift(q_te, kt)).mean()
K = [1, 2, 3, 4, 5, 6.5, 8, 10]
T = pd.DataFrame({f"true {kt}": {f"build {kb}": build_cost(kb, kt) for kb in K} for kt in K}); R = T - T.min(axis=0)
print(R.round(3).to_string()); print("\nmax regret over true k in [1, 10]:", R.max(axis=1).round(3).to_dict())
''')
n.md("### 7. Is a better classifier the answer instead? (tuned XGBoost, LightGBM, blend; judged on CV cost)")
n.code(r'''
import lightgbm as lgb
TUNED = dict(n_estimators=500, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.5, min_child_weight=1, n_jobs=-1)
LG = dict(n_estimators=400, num_leaves=15, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, verbose=-1, n_jobs=-1)
cand = {"v2": oof, "tuned x3 seeds": np.zeros(len(dev)), "lightgbm": np.zeros(len(dev))}
for a, b in F:
    for s in (42, 7, 2024):
        cand["tuned x3 seeds"][b] += xgb.XGBClassifier(**TUNED, random_state=s).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1] / 3
    cand["lightgbm"][b] = lgb.LGBMClassifier(**LG, random_state=SEED).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
cand["xgb + lgbm blend"] = (oof + cand["lightgbm"]) / 2
for k, o in cand.items(): print(f"{k:18s} log-loss {log_loss(y, o):.5f}   AUC {roc_auc_score(y, o):.4f}")
cost = pipeline_costs(cand); print(); report(cost, "v2")
''')
n.md("### 8. Corrected files (v2 classifier, Platt, v5 cost model, shift k)")
n.code(r'''
for k, name in ((4, "v7_k4"), (em_odds_factor(q_te, pi_tr), "v7_em")):      # EM estimate, unrounded
    fl, bu = decide_pv(shift(q_te, k), m5, resid5); print(f"k={k}: ", end=""); write_submission(te.id, fl, bu, name)
''')
n.md(r"""
## Results and decisions

- **Importance weighting is ruled out:** effective sample size is **0.3% of rows unclipped, 1.3% clipped; 1-3 of 397 failures** carry any weight. Train and test overlap too little.
- **The extra risk is not one artifact:** s122 contributes +0.13 of the +0.34 mean-logit increase, the rest is diffuse; a model *without* the drifted sensors still predicts the same elevated risk (v3b). This favours a genuinely higher failure rate at the new depot.
- **EM** gives k = **6.53** on test (6.19 with the tuned classifier and 5.67 without the drifted sensors, in `reports/findings.md`): a test failure rate of ~4.5% vs 0.71%. **Validated on simulated shift:** it recovers k (true 2/4/6 → 1.9/3.9/5.9), captures ~95% of the oracle's gain (k=6: none 3.91 → EM 3.07 → oracle 3.03) and costs +0.009 when there is no shift.
- **Public-score consistency:** if the public set is 30% of test, **k ≈ 4** explains the scores almost exactly (level fit 0.084, size factor 0.98); the unshifted model (k=1) fits worst and the scale-free pattern prefers k 2-4. EM's 6.5 sits higher, as expected: under extra covariate shift EM tends to over-estimate.
- **Regret:** under-correcting is expensive, over-correcting cheap (building with k=1 when the truth is 6.5 costs +1.13/vehicle; with 6.5 when the truth is 3 costs +0.13). **k = 4 minimises worst-case regret over k ∈ [1, 10]** (0.28).
- **A better classifier is not the answer:** tuned XGBoost, LightGBM and the blend all improve log-loss/AUC but **none improves CV cost** (tuned: 0.940 vs 0.904, worse in 5/5 folds; LightGBM 0.988; blend 0.914). Cost depends on the few vehicles near the 2.5% threshold, not on overall ranking.

**Caveats**
- EM assumes pure label shift; the test depot also has covariate shift, so the EM k is biased (probably upward).
- The public-score fit uses six public scores to estimate **one scalar**; the public subset is noisy (~6,000 vehicles, few failures).
- A single constant k assumes the extra failures are spread proportionally; if they are concentrated in some vehicle types, flags are misallocated.
- The cost model is assumed unchanged at the new depot (its sensors do not drift, v5).
- There is **no CV number for the shift itself** (dev has no shift by construction); its support is the simulation plus the public-score consistency.

**Next:** v8 and v9 improve the repair-cost model; the shift factor k is then fixed from the observed leaderboard differences between k = 1, 4 and the EM value.
""")
n.save("v7_depot_shift.ipynb")

# ------------------------------------------------------------------ v8
n = NB("v8: Hidden histograms and shape features")
n.header("v8", "none yet (files `v8_*` are superseded by v9, which only improves the cost model's capacity)",
         "**Hypothesis:** the 170 sensors have structure. Raw bin counts of a histogram mix 'how much the truck was used' with 'how its usage was "
         "distributed'; trees struggle to separate the two. If there are more histograms than the one we found in v1, **shape features** (bin shares, "
         "mean, entropy, tails) should add real signal, for the failure model and/or the repair-cost model.")
n.code(SHAPE)
n.code(PIPE)
n.md("### 1. Discovering the hidden histograms\nIf a column set is a histogram over shared operating time, its row total is a fixed fraction of the known histogram's total. A non-negative least-squares fit of that total on all other columns exposes them.")
n.code(r'''
from scipy.optimize import nnls
U = tr[HISTB].sum(axis=1, min_count=10); cand = [s for s in S if s not in HISTB]
ratio = tr[cand].div(U.replace(0, np.nan), axis=0)
cand = [s for s in cand if (ratio[s].dropna() <= 1.0001).mean() > 0.999 and tr[s].notna().mean() > 0.8]
d = pd.concat([tr[cand], U.rename("U")], axis=1).dropna().sample(15000, random_state=0)
coef, rn = nnls(d[cand].to_numpy(float), d.U.to_numpy(float))
print(f"relative residual of the fit: {rn / np.linalg.norm(d.U):.6f}  (exact)")
g = pd.Series(coef, index=cand).round(3); g = g[g > 0.05]
for v, grp in g.groupby(g): print(f"  equal coefficient {v}: {list(grp.index)}")
for a, cols in HIST.items():
    r = (tr[cols].sum(axis=1, min_count=10) / U).dropna()
    print(f"histogram s{a:03d}-s{a + 9:03d}: total / reference total = {r.median():.3f} (5-95%: {r.quantile(.05):.3f}-{r.quantile(.95):.3f})")
''')
n.md("### 2. Are the new features safe under the depot shift?")
n.code(r'''
def ks(x, z): x, z = x.dropna(), z.dropna(); return ks_2samp(x, z).statistic if len(x) and len(z) else np.nan
Str, Ste = shape(tr), shape(te)
print("same-bin KS of the five un-shifted histograms (max over bins):", {f"h{a}": round(float(max(ks(tr[col], te[col]) for col in cols)), 3) for a, cols in HIST.items() if a != 54})
sh = pd.Series({col: ks(Str[col], Ste[col]) for col in Str}); print(f"KS of the shape features: median {sh.median():.3f}  max {sh.max():.3f}")
usage = lambda df: df[HIST[54][:9]].sum(axis=1, min_count=9)
print(f"usage (operating-time total): train median {usage(tr).median():.3g}  test {usage(te).median():.3g}  KS {ks(usage(tr), usage(te)):.3f}")
Rtr, Rte = ratios(tr, S), ratios(te, S)
for col in ("s120", "s122"): print(f"  {col}: raw KS {ks(tr[col], te[col]):.3f}   per-usage KS {ks(Rtr[col + '_perU'], Rte[col + '_perU']):.3f}")
''')
n.md("### 3. Do they help? 3 × 5 repeated CV on total cost, paired vs A\nA = v5 (base features in both models). B = classifier + shape. C = classifier + shape + usage ratios. E = cost model + shape (classifier unchanged).")
n.code(r'''
SH, RA = shape(dev), ratios(dev, S)
X = {"base": dev[S], "shape": pd.concat([dev[S], SH], axis=1), "ratio": pd.concat([dev[S], SH, RA], axis=1)}
VAR = {"A": ("base", "base"), "B": ("shape", "base"), "C": ("ratio", "base"), "E": ("base", "shape")}
cost = {v: [] for v in VAR}; r2 = {"base": [], "shape": []}; ll = {k: [] for k in X}; auc = {k: [] for k in X}
for r, fs in enumerate((SEED, 1, 2)):
    Fr = list(StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y))
    oo = {k: np.zeros(len(dev)) for k in X}
    for a, b in Fr:
        for k in oo: oo[k][b] = xgb.XGBClassifier(**PARAMS).fit(X[k].iloc[a], y[a]).predict_proba(X[k].iloc[b])[:, 1]
    for k in oo: ll[k].append(log_loss(y, oo[k])); auc[k].append(roc_auc_score(y, oo[k]))
    for a, b in Fr:
        fa, fb = a[y[a] == 1], b[y[b] == 1]; cm = {}
        for k in ("base", "shape"):
            mb, resid = cost_model(CM, X[k].iloc[fa], np.log(c[fa]), X[k].iloc[b]); cm[k] = (mb, resid)
            r2[k].append(r2_score(np.log(c[fb]), mb[y[b] == 1]))
        for v, (fk, ck) in VAR.items():
            fl, bu = decide_pv(make_cal("platt").fit(oo[fk][a], y[a]).predict(oo[fk][b]), *cm[ck]); cost[v].append(per_vehicle(c[b], fl, bu))
A0 = np.array(cost["A"])
print(f"cost-model R2 (log): base {np.mean(r2['base']):.3f}   with shape {np.mean(r2['shape']):.3f}")
for k in X: print(f"classifier on {k:6s}: log-loss {np.mean(ll[k]):.5f}   AUC {np.mean(auc[k]):.4f}")
print("\nvariant  CV cost (15 folds)   paired delta ± se       folds better")
for v in VAR:
    x = np.array(cost[v]); d = x - A0
    print(f"{v}        {x.mean():.4f} ± {x.std():.3f}     {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f}     {int((d < 0).sum())}/15")
''')
n.md("The total-cost gain of E is within noise, so check the cost model on its **own** metrics, which are much less noisy: per-fold R² and the budget's own loss (pinball loss at the 0.909 quantile, i.e. the budget's loss for a vehicle that certainly fails).")
n.code(r'''
r2d, pin = {k: [] for k in X if k != "ratio"}, {k: [] for k in X if k != "ratio"}; dR = []
for fs in (SEED, 1, 2):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y):
        fa, fb = a[y[a] == 1], b[y[b] == 1]; lc = np.log(c[fa]); out = {}
        for k in ("base", "shape"):
            mb, resid = cost_model(CM, X[k].iloc[fa], lc, X[k].iloc[fb])
            out[k] = (r2_score(np.log(c[fb]), mb), np.exp(mb + np.quantile(resid, Q_STAR)))
            pin[k].append(np.mean(UNDER * np.maximum(c[fb] - out[k][1], 0) + OVER * np.maximum(out[k][1] - c[fb], 0)))
        dR.append(out["shape"][0] - out["base"][0])
dR, dp = np.array(dR), np.array(pin["shape"]) - np.array(pin["base"])
print(f"R2 improves in {int((dR > 0).sum())}/15 folds (mean {dR.mean():+.3f}, worst {dR.min():+.3f})")
print(f"budget loss per failing vehicle: base {np.mean(pin['base']):.2f}  shape {np.mean(pin['shape']):.2f}  (better in {int((dp < 0).sum())}/15 folds)")
''')
n.md("### 4. Files (v2 classifier, Platt, cost model with shape features, shift k=4)")
n.code(r'''
oof = fit_oof(dev[S]); p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
q_te = Platt().fit(oof, y).predict(p_te)
f_all = tr[tr.failure == 1]
m_te, resid = cost_model(CM, pd.concat([f_all[S], shape(f_all)], axis=1), np.log(f_all.repair_cost.to_numpy()), pd.concat([te[S], Ste], axis=1))
fl, bu = decide_pv(shift(q_te, 4), m_te, resid); write_submission(te.id, fl, bu, "v8_k4")
''')
n.md(r"""
## Results and decisions

**Discovery.** The competition files contain **six 10-bin histograms** (`s023-032, s037-046, s054-063, s080-089, s129-138, s154-163`). Each one's total is a fixed fraction of the reference total for every vehicle (0.296, 0.180, 1.000, 0.749, 0.695, 0.246; spread ±0.001): histograms over a shared **operating time ("usage")**. `s080-s089`, the cost-driving sensors of v5, is one of them. None except `s054-s063` is shifted in test (same-bin KS ≤ 0.03).

**Usage does not explain the depot shift:** it barely differs between depots (median 2.72M vs 2.83M, KS 0.029), and dividing `s120`/`s122` by usage does not remove their drift (KS 0.68 / 0.56).

| Variant (3×5 CV) | CV cost | Paired Δ vs A | Better |
|---|---|---|---|
| A: v5 | 0.887 | n/a | n/a |
| B: classifier + shape | 0.923 | +0.036 ± 0.019 | 2/15 |
| C: classifier + shape + usage ratios | 0.930 | +0.043 ± 0.025 | 4/15 |
| **E: cost model + shape** | **0.866** | **−0.021 ± 0.028** | **10/15** |

- **Cost model with shape features (E): R² 0.66 → 0.79, better in 15/15 folds** (worst gain +0.06); the budget's own loss falls 30.0 → 24.3 (−19%, 13/15). **Kept.**
- **Shape features in the classifier (B, C) hurt:** AUC rises slightly (0.985 → 0.986) but cost gets worse in 13/15 folds. **Rejected**: the third time an AUC/log-loss gain did not turn into a cost gain.

**Caveats:** the structure is inferred from the data (nothing in the course material documents it); no outside data is used. Repeated CV reuses the same 397 failures, so the 15 paired differences are correlated and the standard errors optimistic. Costs are simulated; the large R² jump may reflect how the simulator generates cost from these sensors. The total-cost gain of E alone is within noise; it is kept because its own metrics improve in every fold and the mechanism (repair cost depends on the shape of the s080-s089 histogram) is clear.
""")
n.save("v8_histogram_features.ipynb")

# ------------------------------------------------------------------ v9
n = NB("v9: Cost-model capacity, per-vehicle spread, and averaging over k")
n.header("v9", "**#7** `sub07-2026-11-21-v9_k1` (public 19,600.79) and **#9** `sub09-2026-11-21-v9_em` (public 18,058.56) → `outputs/submission_v9_*.csv`; `v9_k4` not uploaded yet",
         "**Hypothesis (A):** the cost model has never been tuned since v5 and may over-shrink; under-predicting expensive repairs is the costly "
         "direction (a unit of shortage costs 0.20, a unit of excess 0.02). Also test whether the spread of repair cost differs by vehicle. "
         "**(B):** with the shift k uncertain, choose the decision by averaging over a belief about k.")
n.code(SHAPE)
n.code(PIPE)
n.md("### A1. Diagnostics and capacity of the cost model (repeated 3 × 5 CV on dev failures only; hold-out untouched)\nMetrics: R², calibration slope of `log c ~ prediction` (1.0 = no shrinkage), bias in the lowest/highest predicted-cost quintile, and the **budget loss** (pinball loss at the 0.909 quantile).")
n.code(r'''
f = dev[dev.failure == 1].reset_index(drop=True); Xf = pd.concat([f[S], shape(f)], axis=1)
lcf, cst = np.log(f.repair_cost.to_numpy()), f.repair_cost.to_numpy()
base = dict(subsample=0.8, colsample_bytree=0.5, min_child_weight=5, n_jobs=-1, random_state=SEED)
grid = {"v8: 200 trees d3": dict(n_estimators=200, learning_rate=0.03, max_depth=3), "500 trees d3": dict(n_estimators=500, learning_rate=0.03, max_depth=3),
        "1000 trees d3": dict(n_estimators=1000, learning_rate=0.03, max_depth=3), "500 trees d4": dict(n_estimators=500, learning_rate=0.03, max_depth=4),
        "1000 trees d2": dict(n_estimators=1000, learning_rate=0.03, max_depth=2), "2000 trees d2": dict(n_estimators=2000, learning_rate=0.03, max_depth=2)}
pins = {}
for name, g in grid.items():
    r2s, sl, lo, hi, pn = [], [], [], [], []
    for rep in range(3):
        m = np.zeros(len(f))
        for a, b in KFold(5, shuffle=True, random_state=rep).split(Xf): m[b] = xgb.XGBRegressor(**base, **g).fit(Xf.iloc[a], lcf[a]).predict(Xf.iloc[b])
        r = lcf - m; qq = pd.qcut(m, 5, labels=False); bq = np.exp(m + np.quantile(r, Q_STAR))
        r2s.append(r2_score(lcf, m)); sl.append(np.polyfit(m, lcf, 1)[0]); lo.append(r[qq == 0].mean()); hi.append(r[qq == 4].mean())
        pn.append(np.mean(UNDER * np.maximum(cst - bq, 0) + OVER * np.maximum(bq - cst, 0)))
    pins[name] = np.array(pn)
    print(f"{name:18s} R2 {np.mean(r2s):.3f}  slope {np.mean(sl):.2f}  bias low/high quintile {np.mean(lo):+.2f}/{np.mean(hi):+.2f}  budget loss {np.mean(pn):.2f}  (paired vs v8: {np.mean(pn - pins['v8: 200 trees d3']):+.2f})")
''')
n.md("### Is the spread vehicle-specific? (diagnostic on the v8 cost model's out-of-fold residuals)")
n.code(r'''
from scipy.stats import spearmanr
m_oof = cross_val_predict(xgb.XGBRegressor(**CM), Xf, lcf, cv=inner); r_oof = lcf - m_oof
qq = pd.qcut(m_oof, 5, labels=False)
print("residual sd by predicted-cost quintile:", [round(float(r_oof[qq == i].std()), 3) for i in range(5)], "  overall", round(float(r_oof.std()), 3))
SC = dict(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=10, random_state=SEED)
ah = cross_val_predict(xgb.XGBRegressor(**SC), Xf.assign(m=m_oof), np.abs(r_oof), cv=inner)
print(f"can the features predict |residual| out of sample? Spearman {spearmanr(np.abs(r_oof), ah)[0]:+.3f}")
''')
n.md("### A2. Full pipeline, total cost (3 × 5 repeated CV, paired vs the v8 cost model)\nE = v8 cost model (200 trees, depth 3). A1 = 1000 trees, depth 2. A2 = A1 + a per-vehicle spread (a scale model on |out-of-fold residual|, residuals standardised).")
n.code(r'''
X = pd.concat([dev[S], shape(dev)], axis=1)
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}
def cost_model_h(params, Xa, lc, Xb):
    m_in = cross_val_predict(xgb.XGBRegressor(**params), Xa, lc, cv=inner); r = lc - m_in
    mb = xgb.XGBRegressor(**params).fit(Xa, lc).predict(Xb)
    Za, Zb = Xa.assign(m=m_in), Xb.assign(m=mb)
    s_in = np.maximum(cross_val_predict(xgb.XGBRegressor(**SC), Za, np.abs(r), cv=inner), 0.05)
    z = r / s_in; z = z / z.std() * r.std()
    return mb, z, np.maximum(xgb.XGBRegressor(**SC).fit(Za, np.abs(r)).predict(Zb), 0.05) / np.mean(s_in)
cost = {v: [] for v in ("E", "A1", "A2")}; cov = {v: [] for v in cost}
for fs in (SEED, 1, 2):
    Fr = list(StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y)); oo = np.zeros(len(dev))
    for a, b in Fr: oo[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
    for a, b in Fr:
        fa = a[y[a] == 1]; lc = np.log(c[fa]); q = make_cal("platt").fit(oo[a], y[a]).predict(oo[b]); fm = y[b] == 1
        for v, (prm, het) in {"E": (CM, False), "A1": (CM9, False), "A2": (CM9, True)}.items():
            if het: mb, r, sb = cost_model_h(prm, X.iloc[fa], lc, X.iloc[b])
            else: mb, r = cost_model(prm, X.iloc[fa], lc, X.iloc[b]); sb = None
            fl, bu = decide_pv(q, mb, r, scale=sb); cost[v].append(per_vehicle(c[b], fl, bu))
            s = np.ones(len(b)) if sb is None else sb
            cov[v].append(np.mean(np.log(c[b][fm]) <= mb[fm] + s[fm] * np.quantile(r, Q_STAR)))
E0 = np.array(cost["E"])
for v in cost:
    x = np.array(cost[v]); d = x - E0
    print(f"{v:3s} CV cost {x.mean():.4f} ± {x.std():.3f}   paired vs E {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(15):.4f}   better {int((d < 0).sum())}/15   coverage {np.mean(cov[v]):.3f}")
''')
n.md(r"""
### B. Averaging over the uncertain shift
For a fixed decision the expected cost is linear in $p$, so the Bayes decision under a belief about $k$ uses $\bar p=\sum_k w_k\,\text{shift}(p,k)$. The belief weights come from v7 (EM 5.7-6.5 likely biased up; public-score fit ~4; scale-free fit 2-4).
""")
n.code(EM)
n.code(r'''
oof = fit_oof(dev[S]); p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
q_te = Platt().fit(oof, y).predict(p_te)
f_all = tr[tr.failure == 1]; Xt = pd.concat([te[S], shape(te)], axis=1); Xa = pd.concat([f_all[S], shape(f_all)], axis=1)
m_te, resid = cost_model(CM9, Xa, np.log(f_all.repair_cost.to_numpy()), Xt)        # the v9 cost model
M = np.exp(m_te[:, None] + np.quantile(resid, (np.arange(99) + .5) / 99)[None, :]); mu = M.mean(1)
def exp_cost(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).mean()
belief = {2: .05, 3: .20, 4: .30, 5: .20, 6.5: .20, 8: .05}
pbar = sum(w * shift(q_te, k) for k, w in belief.items())
builds = {f"k={k}": decide_pv(shift(q_te, k), m_te, resid) for k in (3, 4, 5, 6.5)}; builds["avg belief"] = decide_pv(pbar, m_te, resid)
truths = (2, 3, 4, 5, 6.5, 8)
T = pd.DataFrame({f"true {kt}": {n: exp_cost(*fb, shift(q_te, kt)) for n, fb in builds.items()} for kt in truths}); R = T - T.min(axis=0)
print(f"belief-averaged p has mean {pbar.mean():.4f}; flag rate {builds['avg belief'][0].mean():.2%}")
print("regret / vehicle\n", R.round(4).to_string()); print("max regret over k in [2, 8]:", R.max(axis=1).round(3).to_dict())
''')
n.md("### Files (v2 classifier, Platt, **v9 cost model**, shift k). The upload design isolates one change per comparison: `v9_k1` vs the known v5-Platt score = the new cost model alone; `v9_k4` vs `v9_k1` = the shift alone; `v9_em` vs `v9_k4` = whether k should be higher.")
n.code(r'''
for k, name in ((1, "v9_k1"), (4, "v9_k4"), (6.53, "v9_em")):        # 6.53 = the EM estimate from v7
    fl, bu = decide_pv(shift(q_te, k), m_te, resid); print(f"k={k}: ", end=""); write_submission(te.id, fl, bu, name)
''')
n.md(r"""
## Results and decisions

**A. The cost model was over-shrunk:** it over-predicts cheap repairs (mean residual −0.21) and **under-predicts expensive ones (+0.15)**, the costly direction.

| Cost model | R² | Calibration slope | Budget loss |
|---|---|---|---|
| v8: 200 trees, depth 3 | 0.798 | 1.10 | 24.01 |
| 1000 trees, depth 3 | 0.805 | 1.06 | 23.21 |
| **1000 trees, depth 2** | **0.810** | **1.05** | **23.17** |
| 2000 trees, depth 2 | 0.807 | 1.04 | 23.24 |

- The budget loss improves in all 3 repeats (−0.84 for 1000 trees/depth 2). In **total cost** the gain is below what CV can resolve (A1 0.862 vs 0.866, −0.004 ± 0.009, 7/15 folds): its expected effect at the training failure rate is ~0.005/vehicle. On test, with 4-6× more failures, it should be worth ~150-250 public points. **A1 kept** (its own metrics improve consistently and the mechanism is clear).
- **A2 (per-vehicle spread) rejected:** residual sd varies only mildly across vehicles (0.58-0.86 by predicted-cost quintile, no trend) and the features predict |residual| only weakly (out-of-sample Spearman 0.09); modelling it made things worse (+0.012, 4/15) and pushed coverage below target (0.902).

**B. Averaging over k** behaves like an effective **k ≈ 4.4**: its belief-weighted cost is the lowest and its worst-case regret the smallest (0.113/vehicle over k in [2, 8], vs 0.141 for k=5 and 0.168 for k=4), but its average gain over k=4 or 5 is only ~0.003/vehicle. Its real use comes **after** the first probe uploads: the observed score differences update the belief about k (using the simulated likelihoods), and the final decision uses the posterior-averaged $p$.

**Caveats:** the capacity gain is invisible in total-cost CV (noise); all conclusions about k rest on our own model of the test depot, which has been off by 10-15% when predicting past public scores.
""")
n.md(r"""
## Public results (21 Nov 2026)

| Upload | File | What it isolates | Public score |
|---|---|---|---|
| #6 (20 Nov) | `v5_platt` | v5 cost model, no shift (k=1) | 22,968.55 |
| #7 | `v9_k1` | v9 cost model, no shift (k=1) | **19,600.79** (−3,368 vs #6: the new cost model alone) |
| #8 | `v7_k4` | v5 cost model, shift k=4 (uploaded by mistake instead of `v9_k4`) | **19,067.17** (−3,902 vs #6: the shift alone) |
| #9 | `v9_em` | v9 cost model, shift k=6.53 (EM) | **18,058.56** (best so far) |

- **Both changes help, and they add up:** the cost model alone is worth ~15% and the shift alone ~17%; together (`v9_em`) the score is 21% below v5.
- The cost-model gain (−3,368) is larger than the ~−700 to −2,200 predicted from our own model, so on the new depot the cost model matters more than our estimate suggested.
- The shift effect is in line with k ≈ 4 (predicted −3,300 at k=4); the three scores together do not pin k down tightly, because differences of 1,000-2,000 are within the noise of a ~6,000-vehicle public subset. The next step is to update the belief about k from all scores jointly, then upload `v9_k4` and neighbours.
""")
n.save("v9_cost_model_capacity.ipynb")
