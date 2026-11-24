# Fleet Maintenance: findings and caveats log

Metric everywhere: **cost per vehicle** (competition cost ÷ number of vehicles), computed with
`src/score.py`. Results are the mean ± std over the same 5 stratified folds of the 80% dev split.
The 20% hold-out (`common.split_holdout`) is locked: it's used only at v0, v4 and v8.
Raw numbers for every run are in `experiments.csv`.

---

## v0: Trivial baselines

| Policy | CV cost / vehicle | Hold-out |
|---|---|---|
| Flag nobody | 3.487 ± 0.692 | 3.506 |
| Flag all, budget 0 | 10.697 ± 0.139 | 10.701 |
| Flag all, budget = median failure cost (176) | 14.031 ± 0.136 | 14.040 |
| Oracle (perfect foresight) | 0.071 | 0.071 |

**Takeaways**
- **The bar to beat is 3.49 (flag nobody).** Flagging everyone costs 3× more, because 99.3% of inspections would be wasted.
- The gap between the bar and the oracle (3.49 → 0.07) is the room a model has to improve.

**Caveats**
- **Fold noise is large:** ±0.69 on 3.49, i.e. 20%. Failure costs are heavy-tailed, so one 8,000-unit failure moving between folds shifts a fold's cost noticeably. From now on, compare versions with **paired fold differences** (same folds, version A − version B), not by checking whether two means ± std overlap. Use repeated CV if differences are small.
- The baselines use the training depot's failure rate and cost distribution. The test depot may differ (see v1), so these numbers are only approximate for test.

---

## v1: Exploratory analysis

**Target**
- Only **397 failures out of 56,000 (0.71%)**.
- Repair cost is heavy-tailed: median 176, mean 492, 99th percentile 6,847, max 8,000. The **top 10% of failures account for 53% of all repair cost**.
- Sanity check of the decision rule using a global mean μ = 492: flag when p·μ > 12.5, i.e. **p > ~2.5%**. That's about 3.6× the base rate, far below 0.5.

**Censoring (`followup_days`)**
- 22% of healthy vehicles were followed for less than 365 days, so their "healthy" label is uncertain.
- Failure rate rises with follow-up: 0.07% for ≤90 days, then 0.48%, 0.85%, and **1.9% for 271–364 days**. The 365-day group is 0.66%.
- **Caveat:** the high rate in the 271–364 group suggests that records for some failing vehicles stop near the time of failure. `followup_days` is therefore partly determined *by* the outcome, which is one more reason it **must not be used as a feature**. It isn't in the test set anyway.

**Missing values**
- 8.3% of values are missing in train and 9.1% in test.
- Missingness predicts failure: vehicles in the most-missing quartile fail at **2.06%**, the rest at 0.02–0.6%. Keep missing values as NaN (XGBoost handles them natively) and add a missing-count feature in v6.
- s146 is constant, so it's dropped.

**Depot shift: the main finding**
- **Raw adversarial AUC = 1.000.** A classifier separates train from test perfectly. Causes:
  1. **s047 is 100% missing in test.** Dropped.
  2. **s054–s063 form a 10-bin histogram shifted by one bin in test.** Test bin k+1 matches train bin k (KS 0.015–0.067 for all 9 pairs), and the group totals agree (KS 0.03). Fixed in `common.load()`: test bins s055..s063 are mapped to s054..s062, and test s063 is set to NaN (it has no counterpart).
- **After the fixes the AUC is still 0.9999.** The remaining drift is in:
  - **s120**, which looks like a counter that runs higher at the new depot (+16M at the low end, about ×2 at the top). A genuine shift.
  - **s122**, about ×2.7 at every quantile. Possibly a units change, possibly genuine.
  - **s126 and s128**, which top the importance list even though their per-column KS is small. **Hypothesis tested and rejected:** a rounding-precision artifact (both depots have about 10% of values divisible by 1,000). The drift is probably in their joint relationship. Still open.
- Without s120, s122, s126 and s128: **AUC 0.937.** A real, diffuse shift remains (s057, s055, s062…), which is what v7 addresses.

**Caveats**
- The histogram realignment is inferred from matching distributions, not from documentation. It's the simplest explanation (9 bins all match after a shift of exactly 1, and a shift of 2 doesn't match), but it can't be confirmed without the source.
- Adversarial AUC shows that the depots *differ*. It doesn't show that the differences *matter for predicting failure*. Whether to drop s120/s122/s126/s128 gets decided in v7 using cost, not AUC.
- With 397 positives, every per-group failure rate above has wide confidence intervals. For example, the ≤90-day group has only 2 failures.

**Fixes carried into all later versions:** histogram realignment, dropping s047 and s146, never using `followup_days` as a feature.

---

## Next: v2 (plain XGBoost, 0.5 threshold), then v3 (threshold set by cost)

---

## v2 and v3: public leaderboard results

| Version | CV cost/veh | Public score (total) |
|---|---|---|
| v2: XGB, flag if p>0.5 | 2.50 | 74,933 |
| v3: cost-based flag + newsvendor budget | 1.24 | **27,760** |
| v3b: v3 minus s120/s122/s126/s128 | 1.22 | 29,530 |

- v2 → v3 cut the public score by 63% (CV predicted 50%). The decision rule is the largest single lever.
- **Decision on v3b: reverted.** CV said slightly better (-0.03, inside fold noise); the public score said 6% worse. No evidence that dropping the drifted sensors helps, so all sensors stay for now. Revisit in v7 with importance weighting.
- **Caveat:** the public score is a total over an unknown number of vehicles (the statement says 30% of test, about 6,000), so absolute per-vehicle levels can't be read off it. A flag-nobody submission would give the baseline. Public scores are on roughly 6,000 vehicles with few failures, so differences of a few percent are noise.

## v4: Probability calibration

**Hypothesis:** the cost rule depends on p itself (flag when p·μ > 12.5), so miscalibrated probabilities cost money.

**Reliability (OOF).** The raw model is **under-confident exactly where decisions are made**: for raw p between 0.003 and 0.028 the observed failure rate is 2.5% against 1.0% predicted; for p between 0.028 and 0.15, 13.8% observed against 6.7% predicted. Above 0.15 it is well calibrated. So raw p under-flags vehicles near the 2.5% threshold.

**CV (cross-fitted: each fold calibrated on the other folds' OOF; paired vs raw):**

| Method | CV cost/veh | Paired Δ vs raw | Folds improved | Brier |
|---|---|---|---|---|
| raw (v3) | 1.242 ± 0.387 | n/a | n/a | 0.00464 |
| Platt | 1.107 ± 0.220 | -0.135 | 4/5 | 0.00452 |
| Isotonic | 1.052 ± 0.258 | -0.190 | 5/5 | 0.00447 |

**Hold-out (11,200 vehicles, 79 failures): disagrees.** raw 0.906, Platt 0.955, isotonic 1.008. Calibration flagged more vehicles (3.3-3.9% vs 2.3%) and cost more.

**Reading the disagreement.**
- The hold-out has only 79 failures. A 0.05-0.10 difference is about 560-1,100 total cost on 11,200 vehicles, which is one or two large failures. It is within noise, but it is also not support for calibration.
- The CV result is pooled over 44,800 vehicles and 318 failures, and is consistent across folds for isotonic (5/5), but calibrator and evaluation come from the same OOF pool, so it can still be optimistic.
- **Net:** calibration probably helps a little; we can't claim it does. Isotonic is the riskiest choice with few positives; Platt is smoother and sat closer to raw on the hold-out.

**Caveats**
- Calibration is fitted to the **training depot's** failure rate. On test the model already predicts ~3× the failures (mean p 0.018); calibration raises that further (Platt 0.020, isotonic 0.020), and test flag rates rise to 6.7% and 7.7% (from 5.4%). If the extra predicted failures on test are a shift artifact, calibration amplifies it. The public scores for v4 are the first real evidence on this.
- Test predictions come from a model fit on all training data, while the calibrators were fit on OOF predictions from 80%-data models, so test probabilities are slightly more confident than what the calibrator saw.
- The hold-out was used here as planned (v0, v4, v8). It must not drive tuning any more.

**Decision:** keep v3 as the safe base; submit v4_platt and v4_iso to see which direction the public score moves, and choose by CV + hold-out together, not the public score.

---

## v5: Per-vehicle repair-cost model (budget + flag)

**Change:** XGBoost (depth 3, 200 trees) on log(repair_cost), trained on failing vehicles only. Each vehicle gets its own cost distribution, `log c | fail ~ m_i + residuals`, where the residuals are **out-of-fold** (inner 5-fold CV), so the spread is honest. The budget is the per-vehicle newsvendor quantile; the flag rule uses the per-vehicle mean. Back-transform uses residual quantiles, not exp(mean).

**Cost model quality** (nested: fit on training-fold failures only, scored on held-out-fold failures): R² on log scale **0.68** (0.60–0.71 per fold); RMSE 0.95 vs 1.62 for a constant. Sensors s080–s085 drive most of it (|corr| ~0.55). Repair cost is **not** related to the failure score (corr -0.08) or to follow-up (0.05).

**Budget calibration check:** the 0.909 budget quantile covers **91.5%** of held-out failing vehicles (target 90.9%; per fold 86–95%). The budget is doing what it should.

**Ablation** (CV cost/vehicle; A = global F, B = per-vehicle flag + global budget, D = per-vehicle flag + budget; paired vs A):

| p | A | B | D (full v5) | D−A | folds D better |
|---|---|---|---|---|---|
| raw | 1.242 | 1.081 | **0.999** | -0.243 | 5/5 |
| Platt | 1.107 | 0.994 | **0.904** | -0.203 | 5/5 |
| isotonic | 1.052 | 1.016 | **0.932** | -0.120 | 5/5 |

- Most of the gain is the **flag decision** (A→B, e.g. 1.24→1.08 for raw) and the rest the **budget** (B→D, ~0.08–0.09).
- Platt + per-vehicle costs is the best CV combination: **0.904** (-27% vs v3, -18% vs v4 Platt).
- Calibration still helps on top of the cost model in CV (raw 0.999 → Platt 0.904), but the v4 hold-out disagreement is unresolved, so raw-p versions are also kept as submissions.

**Depot-shift check on the cost model:** the cost-driving sensors do not drift (KS 0.002–0.009 for s080–s085). Among vehicles with p>0.1, predicted failure cost is similar on test (median/mean 207/296) and dev (171/271). The shift is in failure *probability* (3.5% of test vehicles have p>0.1 vs 1.2% in dev), not in cost.

**Caveats**
- Only 318 failures train the cost model in each fold (397 for the final model). Per-fold R² ranges 0.60–0.71, so it is stable but not precise.
- Repair costs are **simulated** for this course, which probably explains why they are so predictable (R² 0.68 from sensors). A real repair-cost process would be noisier; this result should not be generalised.
- Residuals are assumed to have the same spread for all vehicles (homoscedastic in log space). Not tested; a vehicle-dependent spread could shift the budget quantile.
- The flag rule uses a probability that is still uncertain on the test depot (see v4 caveats). v5 improves the cost half of the decision, not the failure-probability half, which is now the larger source of error.
- Hold-out not touched in v5 (reserved for v8). Test flag rates: raw 4.8%, Platt 6.3%, isotonic 6.8%.

**Decision:** keep. v5 supersedes v3/v4 as the base. Submit `v5_platt` (best CV) and `v5_raw` (hedge against the calibration doubt).

---

## Public leaderboard: v4 and v5 (3 uploads, 20 Nov)

| Submission | CV cost/veh | Public score (total) | vs v3 (27,760) |
|---|---|---|---|
| v3 | 1.242 | 27,760 | n/a |
| v4 Platt | 1.107 | 23,876 | -14% |
| v5 raw | 0.999 | 26,061 | -6% |
| **v5 Platt** | **0.904** | **22,969** | **-17%** |

- **Direction agrees with CV for the key comparison:** calibration (Platt) helped on test in both v4 and v5 (v5 raw 26,061 -> v5 Platt 22,969, -12%). That is the first test-side evidence for calibration, which the v4 hold-out had left open.
- **v5 Platt is the best on both CV and public.** Magnitudes differ: CV predicted -27% from v3, public shows -17%.
- v5 raw (CV 0.999) lands worse than v4 Platt (CV 1.107) on the public set, so the public ordering does not match CV exactly. A ~6,000-vehicle subset with few failures can't separate such rows with confidence.
- **Caveat:** public score is 30% of test only; do not pick the final model on it. It is one more noisy data point.

---

## v6: Follow-up handling and missingness features (failure-probability model)

**Hypotheses.** (B) the missing-value count is a known predictor of failure, so explicit NaN-count features might help trees. (C/D) healthy vehicles with short follow-up have unreliable "healthy" labels (v1), so dropping them (C: follow-up < 180 d) or down-weighting them (D: weight min(followup,365)/365) should give cleaner training labels. (E) B + D.

**Setup.** Only the classifier changes; the v5 pipeline (nested cost model, Platt fitted on the other folds' OOF, per-vehicle decision) is held fixed and scored on all validation vehicles. `followup_days` shapes training rows/weights only, never a feature. The NaN count excludes s063 (always NaN in test by construction).

| Variant | OOF AUC | PR-AUC | CV cost/veh | Paired Δ vs A | Folds better |
|---|---|---|---|---|---|
| A (v5 Platt) | 0.9840 | 0.526 | 0.904 ± 0.235 | n/a | n/a |
| B + NaN counts | 0.9846 | 0.526 | 0.915 ± 0.247 | +0.011 | 2/5 |
| C drop followup < 180 | 0.9843 | 0.517 | 0.888 ± 0.178 | -0.016 | 2/5 |
| D weight by followup | 0.9841 | 0.532 | 0.881 ± 0.181 | -0.023 | 2/5 |
| E B + D | 0.9841 | 0.518 | 0.923 ± 0.245 | +0.019 | 1/5 |

**Reading.** No variant is a real improvement. C and D look better on average, but that comes from a single fold (-0.17 / -0.18 in fold 5) while they are worse in 3 of the other 4 folds. That is exactly what one lucky large failure looks like, so it doesn't beat fold noise. B (missingness features) doesn't help: trees already see the NaNs natively. AUC is flat for all variants (0.984-0.985).

**Caveats**
- With 397 failures, fold-to-fold swings of 0.1-0.2 are normal; a 0.02 average gain cannot be told from noise. A repeated-CV test (several fold seeds) would be needed to settle C/D, and the cost of that is probably not worth it for such a small potential gain.
- Dropping/weighting changes the training base rate (mean p 0.0058 → 0.0077 for C); Platt recalibration absorbs this, but only for the training depot.
- Rows dropped in C are mostly unlabeled-outcome vehicles, which contain information about the "healthy" class; the data say that this information hardly matters.

**Decision: revert to A.** No v6 submission. The failure model stays as in v2/v5; the remaining gain is more likely in the depot shift (v7).

---

## v7: Depot shift, prior-shift correction (and a rejected tuning attempt)

**Question.** The model predicts ~3× more failures on test (mean p 0.018 vs 0.006). Is that real, and if so, how should decisions change?

**1. Importance weighting: ruled out.** Adversarial classifier (without the 4 drifted sensors) → weights p(test|x)/p(train|x). Effective sample size is **0.3% of rows unclipped, 1.3% clipped at 5×; failures ESS 1–3 of 397**. Train and test overlap too little for reweighting to work. Not used.

**2. Where the extra test risk comes from (SHAP, test − dev mean logit = +0.34).** s122 (the ×2.7 sensor) contributes +0.13; the rest is diffuse (s123, s149, s034, …). But a model **without** the drifted sensors still predicts the same elevated risk (mean p 0.0175; EM k 5.7). The elevation is not an artifact of one sensor; it's carried by many features. This favours a genuinely higher failure rate.

**3. Prior-shift correction (Saerens–Latinne–Decaestecker EM).** Estimates the test failure rate from the test p alone (no labels), then shifts every p by the odds factor k: p' = kp/(1−p+kp).
- **Validated on simulated label shift inside dev** (keep all failures, subsample healthy at 1/k): EM recovers k = 0.99 / 1.93 / 3.89 / 5.91 for true 1 / 2 / 4 / 6, and captures ~95% of the oracle's cost gain (k=6: none 3.91 → EM 3.07 → oracle 3.03 per vehicle). With no shift it costs +0.009 (negligible).
- **On test:** k = **6.53** (v2 classifier), 6.19 (tuned), 5.67 (no-drift). Test failure rate ≈ 4.5% vs 0.71% in train.

**4. Consistency with the public scores already observed.** Using our own p and cost model, compute each submitted file's expected score on 30% of test under different k:

| k (belief) | level fit (RMS log err) | implied public-set scale (1 = 30%) | pattern fit, scale-free |
|---|---|---|---|
| 1 (as is) | 0.735 | 2.07 | 0.117 |
| 2 | 0.381 | 1.45 | 0.079 |
| **4** | **0.084** | **0.98** | 0.080 |
| 6.5 | 0.334 | 0.73 | 0.096 |
| 10 | 0.598 | 0.56 | 0.117 |

If the public set is 30% of test (as stated), **k≈4** explains the scores almost exactly. The scale-free pattern prefers k 2–4. EM's 6.5 sits higher, as expected: under covariate shift EM tends to over-estimate. The uncorrected model (k=1) is the worst fit on every measure: our submissions have been under-flagging on the new depot.

**5. Cost of guessing k wrong (expected cost/vehicle on test, regret vs best).** Under-correcting is expensive, over-correcting is cheap: building with k=1 when the truth is 6.5 costs +1.13/vehicle; building with 6.5 when the truth is 3 costs +0.13. **k=4 minimises the worst-case regret over k∈[1,10] (0.28).**

**6. Rejected: classifier tuning.** Grid on OOF log-loss (depth 3/5/7, min_child_weight 1/5, up to 1,500 trees at lr 0.02). Best: depth 3, 500 trees, colsample 0.5; 3-seed average. Log-loss improved 0.01676 → 0.01589 (−5%), AUC 0.984 → 0.985, **but CV cost got worse in 5/5 folds (0.904 → 0.940)**. Log-loss rewards getting the bulk of healthy vehicles right; cost is decided by the few vehicles near the flag threshold and the big failures. Lesson: tune on the objective, or at least confirm on it. Reverted; the two files built on it are in `submissions/rejected/`.

**Submissions (all: v2 classifier, Platt, v5 per-vehicle cost model, prior shift k):**

| File | k | Why | Test flag rate |
|---|---|---|---|
| `submission_v7_k4.csv` | 4 | consensus: level fit, minimax regret, inside pattern fit | 9.5% |
| `submission_v7_em.csv` | 6.53 | pure unsupervised EM estimate | 10.9% |
| `submission_v7_k3.csv` | 3 | lower bracket | 8.8% |

Together with v5_platt (k=1, 22,969) these trace public score vs k. If the minimum is near 4, three independent lines of evidence agree.

**Caveats**
- EM assumes **pure label shift** (p(x|y) unchanged). The test depot also has covariate shift (adversarial AUC 0.94 even without the drifted sensors), so the EM k is biased; direction most likely upward.
- The public-score fit uses **six public scores** to estimate **one scalar** (the test failure level). That's mild use of the leaderboard, not per-vehicle fitting, but it is still the 30% public subset, with the noise that comes with it. The final k will be chosen by agreement between EM, regret and the score fit, not by the single best public score.
- Shifting by one constant k assumes the extra risk is spread proportionally over all vehicles. If the new depot's extra failures are concentrated in some kind of vehicle, a constant k misallocates flags.
- The cost model is assumed unchanged at the new depot (its sensors don't drift, v5), but repair-cost levels there are not observable. Higher costs would act like a higher k.
- No CV number exists for the shift itself: the dev set has no shift by construction. Its support is the simulation (the method works when the assumption holds) plus the public-score consistency (the assumption roughly holds here).

### v7 addendum: other allowed classifiers, judged on CV cost (k=1, paired vs v2)

| Candidate | Log-loss | AUC | CV cost | Paired Δ | Folds better |
|---|---|---|---|---|---|
| v2 (XGB, current) | 0.01676 | 0.9840 | **0.904** | n/a | n/a |
| v2 × 3 seeds | 0.01667 | 0.9836 | 0.912 | +0.008 | 2/5 |
| LightGBM | 0.02007 | 0.9843 | 0.988 | +0.084 | 1/5 |
| XGB + LGBM blend | 0.01752 | 0.9842 | 0.914 | +0.011 | 1/5 |
| XGB×3 + LGBM blend | 0.01747 | 0.9838 | 0.914 | +0.010 | 1/5 |

None beats v2. Together with the rejected tuning, this shows the classifier is not the bottleneck: AUC is ~0.984-0.985 for every model. The remaining error is the depot shift, so the remaining submissions go to estimating k.

**Public leaderboard (20 Nov):** 1st 18,561 · 2nd 19,225 · 3rd 20,002 · us 22,969 (sorted ascending: lower is better, so the earlier worry about sort direction was unfounded for the board itself).

**Submission plan for the k-search** (files `submission_v7_k{2,3,4,5,8,10}.csv`, `submission_v7_em.csv` = k 6.53):
- Day 1: k=3, k=4, k=6.53 (EM): brackets the prior evidence.
- Day 2+: refine on the side where the public score is falling (e.g. 5/8/10 if EM wins; 2/3.5/5 if k=3 wins).
- Final pick: fit a smooth curve of public score vs k (a single global parameter, so little risk of overfitting the 30% public set) and combine it with the EM/regret evidence. Select the curve minimum plus a neighbouring hedge as the two final submissions.

---

## v8: Histogram structure → features (judged on cost, 3×5 repeated CV)

**Discovery.** Non-negative least squares of the known histogram total (s054–s063) on all other columns fits **exactly** (residual 0.0000) and exposes **five more 10-bin histograms**: s023–032, s037–046, s080–089, s129–138, s154–163. Each one's total is a fixed fraction of the reference total for every vehicle (0.296, 0.180, 0.749, 0.695, 0.246; spread ±0.001), as histograms over a shared operating-time base should be. None of the five is bin-shifted in test (same-bin KS ≤ 0.022), and bin shapes are near-identical across depots. **s080–s089, the cost-driving sensors from v5, is one of these histograms.**

**Usage.** The shared total is operating time ("usage"). It barely differs between depots (median 2.72M vs 2.83M, KS 0.029). So the depot shift is **not** "the new depot's trucks are used more", and dividing s120/s122 by usage does not remove their drift (KS 0.68 / 0.56).

**Features** (`src/features.py`): per-histogram bin shares, mean bin, std, entropy, low/high-3 shares, log usage (66 features, max KS between depots 0.067); usage-normalised non-histogram sensors (`_perU`).

**Results** (3 repeats × 5 folds = 15 paired differences vs A; ± is the standard error of the paired mean):

| Variant | CV cost | Paired Δ | Better | Note |
|---|---|---|---|---|
| A: v5 (base features both models) | 0.887 | n/a | n/a | |
| B: classifier + shape | 0.923 | +0.036 ± 0.019 | 2/15 | AUC 0.985→0.986 but cost worse |
| C: classifier + shape + usage ratios | 0.930 | +0.043 ± 0.025 | 4/15 | |
| **E: cost model + shape** | **0.866** | **−0.021 ± 0.028** | **10/15** | **R² 0.66 → 0.79** |

**Cost-model check (less noisy than total cost):** R² better in **15/15 folds** (+0.13, minimum +0.06); the budget's own loss on failing vehicles 30.0 → 24.3 per vehicle (−19%, 13/15); quantile coverage 0.919 (target 0.909).

**Decisions**
- **Keep E.** The cost model now uses sensors + histogram shape. The total-cost gain alone is within noise, but the cost model's own metrics improve in every fold, and the mechanism is clear: repair cost depends on how the s080–s089 histogram is distributed, which ratios express directly.
- **Reject B and C.** Third time a ranking gain (AUC) did not turn into a cost gain (also: v7 tuning, v7 LightGBM/blends). The classifier stays as v2.

**Caveats**
- The histogram structure is inferred from the competition files alone (the exact least-squares fit above); nothing outside them is used, and all decisions rest on our own CV.
- Repeated CV reuses the same 397 failures, so the 15 paired differences are correlated; the standard errors are optimistic.
- Costs are simulated: the large R² jump may reflect how the simulator generates cost from these sensors.

**Submissions:** `submission_v8_k{1,3,4,5,8,10}.csv`, `submission_v8_em.csv` (k=6.53), all v2 classifier + Platt + shape cost model + prior shift k. These supersede the v7 files (same flags ±1%, better budgets).

### Upload design for 21 Nov (one change per comparison)

Each file differs from a known score in exactly one way. The public subset is fixed, so a score *difference* depends only on the vehicles where two files disagree, which makes differences much less noisy than levels. Expected differences (Monte Carlo over the 30% subset and failures; negative = better):

| Comparison | k=1 | k=2 | k=3 | k=4 | k=6.5 | sd |
|---|---|---|---|---|---|---|
| v5_platt → v8_k1 (cost model only) | −730 | −1,190 | −1,420 | −1,750 | −2,230 | 1.0–2.1k |
| v8_k1 → v8_k4 (shift only) | +1,780 | −130 | −1,820 | −3,320 | −6,660 | 1.1–2.5k |
| v8_k4 → v8_em (k above 4?) | +1,540 | +1,140 | +750 | +380 | −430 | 0.4–1.0k |
| v8_k3 → v8_k4 (dropped) | +610 | +330 | +90 | −130 | −650 | 0.4–0.8k |

Uploads: `v8_k1`, `v8_k4`, `v8_em`. k3 was dropped: its comparison with k4 is inside the noise for every k. Caveat: expectations come from our own model (off by 10–15% on past uploads).

---

## v9: Cost-model refinements (A) and averaging over k (B)

**A: diagnostic first.** OOF residuals of the v8 cost model:
- Spread barely depends on the vehicle: residual sd 0.67–0.82 across predicted-cost quintiles, no trend; features predict |residual| only weakly (Spearman 0.16).
- But the model is **biased at the extremes**: it over-predicts cheap repairs (mean residual −0.14) and **under-predicts expensive ones (+0.20)**: over-shrinkage, and the costly direction (each unit of budget shortfall costs 0.20). Cost-model settings had never been tuned since v5.

**A1: more capacity** (3×5 repeated CV on dev failures; budget loss = pinball loss at q* on the cost scale):

| Cost model | R² | Calib. slope | Bias low/high quintile | Budget loss |
|---|---|---|---|---|
| v8: 200 trees, depth 3 | 0.798 | 1.10 | −0.21 / +0.15 | 24.01 |
| 1000 trees, depth 3 | 0.805 | 1.06 | −0.14 / +0.12 | 23.21 |
| **1000 trees, depth 2** | **0.810** | **1.05** | **−0.13 / +0.08** | **23.17** |
| 2000 trees, depth 2 | 0.807 | 1.04 | −0.11 / +0.08 | 23.24 |

Budget loss improves in all 3 repeats (−0.84 for 1000/depth 2).

**Full-pipeline check** (3×5 CV, total cost, paired vs v8 cost model):

| Variant | CV cost | Paired Δ | Better | Coverage |
|---|---|---|---|---|
| v8 cost model | 0.866 | n/a | n/a | 0.919 |
| **A1: 1000 trees depth 2** | 0.862 | −0.004 ± 0.009 | 7/15 | 0.926 |
| A2: A1 + per-vehicle spread | 0.878 | +0.012 ± 0.010 | 4/15 | 0.902 |

- **Keep A1.** Total-cost CV can't resolve it (expected effect ~0.005/vehicle at the training failure rate, exactly what was measured); the cost model's own metrics improve consistently and the mechanism (less shrinkage on expensive repairs) is clear. On test, with 4–6× more failures, it's worth roughly 150–250 public points.
- **Reject A2.** The spread signal is too weak to estimate from ~318 failures; modelling it adds noise and pushes coverage below target.

**B: averaging over k.** For a fixed decision the expected cost is linear in p, so the Bayes decision under uncertainty about k is the decision made with p averaged over the belief about k. Current belief (k: weight) 2: .05, 3: .20, 4: .30, 5: .20, 6.5: .20, 8: .05 → equivalent single k ≈ 4.4.

| Build | Belief-weighted cost | Max regret (k 2–8) |
|---|---|---|
| k=3 | 3.412 | 0.296 |
| k=4 | 3.373 | 0.144 |
| k=5 | 3.371 | 0.128 |
| k=6.5 | 3.407 | 0.245 |
| **averaged belief** | **3.369** | **0.094** |

B is the principled way to set k, not a separate submission: its value comes after the 21 Nov probes, when the observed score differences update the belief (likelihoods from the probe simulation) and the averaged p is recomputed from that posterior.

**Submissions:** `submission_v9_{k1,k3,k4,k5,em,k8}.csv` (v9 cost model). Probe set for 21 Nov is unchanged in design: **v9_k1, v9_k4, v9_em**. v9_k4 differs from v8_k4 in 92 flags and a median budget change of 42.

---

## Public results of 21 Nov and v10: what shift do the scores support?

**Uploads (v9 cost model unless noted):** `v9_k1` 19,601 (no shift), `v7_k4` 19,067 (v5 cost model, k=4; uploaded by mistake instead of `v9_k4`), `v9_em` 18,059 (k=6.53). Against v5 Platt (22,969): the new cost model alone −3,368 (−15%), the shift alone −3,902 (−17%), both −4,911 (−21%). The cost-model gain was larger than predicted (−700 to −2,200).

**v10: joint posterior over the shift k from all nine public scores** (`notebooks/v10_shift_posterior.ipynb`). For each candidate k we simulate the public score of every uploaded file (same 30% subset, same failures, so scores are correlated) and evaluate the observed vector under a Gaussian with an added relative model-error term.
- **Posterior: k ≈ 3 (mean 3.0, 90% interval 2-4.5)**; 2.5-3.2 across model-error settings. Standardised residuals all within ~±1. **EM's 6.5 has ~0 posterior mass** (it assumes pure label shift; the covariate shift between depots biases it upward). k=4 has probability ~0.10.
- **Posterior-averaged decision** (`v10_bayes`, behaves like k≈3.0, flags 8.9%): expected cost/vehicle on the whole test set **5.1% below `v9_em`**, 4.4% for `v9_k4`, 3.9% for `v10_k2`; `v9_k1` is 4.9% above `v9_em`. In public points: roughly 17,100-17,300 vs 18,059 if the public set behaves like the model.
- **Public scores cannot separate the close candidates** (differences of 150-350 points vs a public noise of ~1,000+); uploading them gives the score-vs-k curve for the report, but the final pick should rest on the posterior, not on the public ranking.

**Caveats:** one constant k for all vehicles and the v9 cost model are assumed; an unobserved higher repair-cost level at the new depot would act like a higher k, so k is an *effective* value. The model-error term is a stand-in for everything the simulation misses. Remaining gains are more likely in the models (cost model, failure ranking) than in fine-tuning k.

**Next uploads (22 Nov):** `v10_bayes`, `v9_k4`, `v10_k2`.

---

## v11: Ideas from the structure of the problem (all rejected)

The decision is "flag iff p·μ is large" and the loss is dominated by expensive failures, so the ideas aim at the expected-cost side. Full-pipeline CV cost (3×5 repeated, paired vs baseline = v2 classifier + Platt + A1 cost model, k=1; baseline 0.8623).

| Idea | CV cost | Paired Δ | Better |
|---|---|---|---|
| Classifier weighted by (c/mean c)^0.5 on failures | 0.873 | +0.011 ± 0.017 | 5/15 |
| Same, exponent 1.0 | 0.887 | +0.025 ± 0.016 | 4/15 |
| Direct expected-cost model (Tweedie), p = e/μ | 1.162 | +0.300 ± 0.095 | 4/15 |
| Blend of baseline and Tweedie (logit average) | 0.911 | +0.048 ± 0.041 | 7/15 |

**Budget as a direct quantile regression** (failures only, 3×5 CV, pinball loss on the cost scale at levels 0.5/0.8/0.91/0.97): worse than the current residual-quantile budget at every level (+6 to +74, better in at most 5/15 folds; at the 0.91 level 105.9 vs 158.1 (log) and 146.7 (raw)). Only ~320 failures, so a separate model per quantile is much noisier than one mean model with pooled residuals.

**Conclusion:** the classifier and the budget method stay as they are. Fourth time a plausible-sounding change to the failure side does not beat fold noise; with ~320 training failures the pipeline is probably near what the data can support. Scripts: `src/v11_ideas.py`, `src/v11_quantile.py`.

**Not testable on our labels:** rescaling the drifted sensors s120/s122 to the training depot (genuine wear or instrument offset is unknowable from train alone). It could only be judged on a public score, which would cost an upload slot; not recommended while the posterior-based files are still unevaluated.

---

## Public results of 22 Nov and updated posterior

**Uploads:** `v9_k4` **17,013.86** (best so far), `v10_bayes` **17,348.75**. `v10_k2` was skipped (least informative).

- **Both inside the v10 prediction** (v10_bayes predicted 17,100-17,300 ± 1,500). `v9_k4` beat `v10_bayes` by 335 points where the model expected `v10_bayes` ~120 better; the two files differ on few vehicles and the gap is within noise (standardised residuals -0.29 and -0.07).
- **Posterior re-fitted with all 11 scores** (exact values): mean k 2.9 (2.4-3.1 across model-error settings), 90% interval [2, 4], mode 2.5-3. EM's 6.5 still has ~0 mass; k=4 has probability ~0.075. The posterior barely moved, so the Bayes decision (flags 8.8%) is essentially `v10_bayes` again.
- **Expected cost per vehicle on the whole test set under the posterior:** Bayes 3.065, k=3 3.066, k=3.5 3.074, k=4 3.091 (+0.9%), k=2 3.096. Differences between k=3 and k=4 are ~1%, far below what public scores can resolve (conditional noise sd ~1,300 on a level).
- **Final-file candidates:** `v10_bayes` (posterior-optimal) or `v9_k4` (best public score, 0.9% worse in expectation); choose by the posterior, not the public rank.

---

## v12: Where the remaining cost is, and what could shrink it

**Cost decomposition** (CV, k=1, v9 pipeline; 0.851 per vehicle): inspections 0.298 (2.98% flagged), **under-budget penalty 0.294**, missed failures 0.165, over-budget 0.094. 253 of 318 failures are flagged (80%). The ranking is already good; the biggest controllable piece is the budget, i.e. repair-cost prediction.

**Cost-model form** (failures only, 2×5 CV; budget loss = pinball at q*; paired vs XGB A1, R² 0.805, loss 108.7):

| Model | R² | Budget loss | Paired Δ |
|---|---|---|---|
| XGB depth 1 (additive) | 0.790 | 120.8 | +12.1 ± 2.7 (0/10 better) |
| Lasso | 0.695 | 11,400 | blows up (linear extrapolation on outliers) |
| Ridge | 0.563 | 15,996 | same |
| Seed-averaged XGB (5) | 0.806 | 109.9 | +1.2 ± 3.3 |
| CatBoost | 0.785 | 102.6 | −6.2 ± 6.2 (7/10) |
| Random forest | 0.757 | 117.0 | +8.3 ± 7.3 |
| Extra trees | 0.779 | 109.6 | +0.9 ± 5.8 |
| Avg(XGB, CatBoost) | 0.802 | 105.0 | −3.7 ± 3.9 (8/10) |
| Avg(XGB, CatBoost, ET) | 0.803 | 104.0 | −4.8 ± 4.3 (7/10) |

**Conclusion:** boosted trees of depth 2 are the right form (linear and purely additive models are clearly worse; there are real interactions). CatBoost and the averages lean slightly better on budget loss but are within one standard error and worse on R², so no change. The cost model looks close to what 318 failures support.

**Drift check on the cost model.** Its top features are the histogram means (h80_mean 22% of gain, h54_mean 11%, h37_mean 11%), which barely drift between depots (KS 0.005-0.024); gain-weighted KS of all inputs is 0.015. Dropping the four drifted sensors costs nothing (CV R² 0.801 vs 0.802) but moves test cost predictions by ~12% (mean abs log difference 0.118, correlation 0.986). So the cost model is mostly robust, and excluding those sensors is free insurance against extrapolation that CV cannot see.

**`submission_v12_robust.csv`:** `v10_bayes` with exactly that one change (62 flags differ, mean absolute budget change 43 on commonly flagged vehicles). Expected public difference vs `v10_bayes` is small; it is a single-change probe, not a large expected gain. Scripts: `src/v12_costlinear.py`, `v12_costfamilies.py`, `v12_costdrift.py`, `v12_build.py`.

### Public result of `v12_robust` (22 Nov): 16,839.86 (best so far), and what it does and does not show

`v12_robust` scored **16,839.86**: 509 below `v10_bayes` (17,348.75) and 174 below `v9_k4` (17,013.86). The only difference from `v10_bayes` is the cost-model inputs (four drifted sensors excluded).

**But the sensors are probably not the reason.** In the full cost model those four sensors carry 0.2% of the gain. Refitting the *same* cost model with a different random seed moves test cost predictions by a mean absolute log difference of 0.135 (0.138 for the drop-4 model), as much as dropping the sensors does (0.118). Averaging 5 seeds cuts the all-vs-drop-4 difference to 0.062. So the gain is most plausibly a cost-model fit that suited the public subset better, i.e. **model variance (random column sampling) is large at the new depot**, larger than the CV R² suggests. Part of the gain may also be public-subset noise.

**Consequence:** a single XGBoost fit is a noisy budget-setter. Next candidate: seed-averaged cost model (and classifier), which removes this variance by construction. Script: `src/v12_seedcheck.py`.

---

## v13 and v14: seed-averaging, and quantile-mapping the instrument-like drifted sensors

**v13_bag** (`submission_v13_bag.csv`): `v12_robust` with the classifier averaged over 5 seeds and the cost model over 8 (drifted sensors excluded from the cost model; shift posterior unchanged; mean test p 0.0203 either way). Differs from `v12_robust` in 86 flags (mean abs budget change 59 on commonly flagged vehicles), from `v10_bayes` in 94. Purpose: remove the random-draw variance found in v12 (seed-to-seed test cost predictions differ by 0.135 mean abs log). Flag rate 8.795%.

**What the four drifted sensors look like** (test/train quantile ratios at 5/25/50/75/95%):
- `s122`: 2.45, 2.57, 2.68, 2.81, 4.89 (≈ ×2.7: looks like a units/instrument change)
- `s120`: a counter with a ≈16M floor in test, ≈ ×2 at the top (offset + scale)
- `s126`, `s128`: 1.01, 1.01, 1.05, 1.09, 1.81 (identical in the bulk, heavier upper tail: looks like a real shift in wear, not an instrument change, so quantile-mapping them would erase genuine signal)

**The classifier does use `s120`/`s122`** (16.4% of its gain), unlike the cost model (0.2%). **v14_map** (`submission_v14_map.csv`) = `v13_bag` with the test values of `s120` and `s122` quantile-mapped onto the training distribution before the classifier scores them; `s126`/`s128` are left alone. Effects: mean test p 0.0203 → 0.0183, Spearman 0.986 with the unmapped scores, 1,670 of the top 1,760 vehicles shared. The shift factor k is rescaled (×1.72) so that the number of flagged vehicles equals `v13_bag` (1,759; 222 flags differ, mean abs budget change 75.5 on commonly flagged vehicles). Matching the average probability instead left 173 fewer flags, which would have confounded the comparison.

**Caveats:** the mapping assumes the two depots have the same distribution of these two sensors, i.e. that the whole difference is instrumentation; if part is real wear, mapping removes real signal. Matching the flag count also raised the average p by ~10%, so v14's budgets are slightly higher: the comparison isolates ranking only approximately. Judged only by a public score; no CV test exists for this change.

## Public results of 23 Nov (v13_bag, v14_map)
| file | public LB |
|---|---|
| v12_robust | 16,839.86 |
| v14_map | 17,375.65 |
| v13_bag | 17,974.55 |

- **v13_bag is 1,135 worse than v12_robust** although it only averages over seeds (5 classifier, 8 cost). Both use the same cost-model inputs and shift posterior. This supports the seed-variance caveat: v12_robust's lead over v10_bayes (-509) was mostly a favourable draw, not a robust gain. Public-LB differences of a few hundred between sibling files are therefore noise-level; v12_robust's level is not a reliable expectation for the private 70%.
- **v14_map is 599 better than v13_bag** (same seeds and cost model; only the classifier sees s120/s122 quantile-mapped, flag count matched). That is a gain beyond my pre-stated 300 threshold, in the direction predicted by the instrument-shift hypothesis, but one comparison on 30% of test cannot prove it.
- Caveat: v14_map is still 535 above v12_robust and 27 above v10_bayes; mapping helps relative to its own base, which is worse than v12's single draw.

## v15_map_v12 public result (23 Nov): 17,058.28
Same mapping as v14, applied on the v12_robust base (flag count matched): **+218 vs v12_robust**, whereas on the v13_bag base it gave -599. The two effects have opposite signs, so the s120/s122 mapping has no reliable effect on the public score: it is within the noise of sibling files (a few hundred). We do not claim the mapping helps. Combined with the v13_bag result, differences below ~600 between sibling files are not interpretable.

## v17 level probes, public results (24 Nov)
Both on the v12_robust base, each a single change:
| file | change | public | vs v12_robust | model-expected change (public 30%) |
|---|---|---|---|---|
| v17_bud125 | flagged budgets x1.25 | 16,826 | -14 | +275 |
| v17_k135 | k x1.35, 1,880 flags (+119, all added) | 16,908 | +68 | +117 |

- Both are ties, far inside the ~600 noise between sibling files. Neither level is worth changing: newsvendor budgets and the k posterior stay.
- k135 landed where the model predicted. bud125 was ~290 better than predicted, a weak hint (inside noise) that test repairs are somewhat more expensive than the cost model expects.
- Every level lever (k, flag count, budget size) is now exhausted. The remaining gap to the leader (~15k) has to come from ranking, i.e. which vehicles the classifier puts above the flag threshold at the new depot.
Notebook: `notebooks/v17_level_probes.ipynb`. Script: `src/v17_probes.py`.
