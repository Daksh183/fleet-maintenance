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

## v21: which risk ranking do the public scores support? (24 Nov)

Model comparison on all 18 public scores (v18 likelihood, maximised over failure odds a, slope b and cost level alpha for each hypothesis). `src/v21_hyp.py`.

| Hypothesis for the true risk ranking | max log-lik (eps 2%) | Spearman vs base |
|---|---|---|
| base: v12 classifier, test as loaded | −131.40 | 1.000 |
| wear: test s120 offset treated as real wear, added to s126/s128/s012 | −131.40 | 1.000 |
| nodrift: classifier without the five wear counters | −131.11 | 0.919 |

- **The wear hypothesis is inert.** The classifier's ranking does not change at all (Spearman 1.000), so neither "instrument fault" (v19) nor "real wear" moves anything.
- **The public scores cannot separate rankings with Spearman 0.92 (Δll 0.3).** Eighteen scalars over ~150 public failures carry far too little information to choose a ranking.
- **Every file scores ≈ 1,000–2,500 above what the best-fitting model predicts** (v12 by +1,310). Because all files share the same public subset, this is mainly one shared draw (more or dearer failures in the public 30% than expected) and applies equally to every team on that subset. It is not a lever.

## v22: adversarial feature selection, and the search for a structural gap (24–25 Nov)

**Leaderboard 25 Nov:** Group 7 went from 22,152 to **14,610** in one day; Garv Jain is at 15,092 and we are 5th at 16,827. Two teams below 15.1k point to a real gap of about 12%.

**s122 is the classifier's top feature and is mis-scaled at the test depot.** It carries 20% of the v12 classifier's gain, has KS 0.31 between depots, and its ratio to s126 is ×2.55 in test. Dropping s063/s120/s122 costs nothing in CV (−0.024 ± 0.011 per vehicle vs all sensors, better in 11/15 folds; `src/v22_advsel.py`, `reports/v22_drift.csv`). Dropping more of the shifted sensors (KS > 0.03) is clearly worse (+0.050). **`submission_v22_nos122.csv`** is v12_robust with the classifier refit without s120/s122, flag count matched (1,782; 221 flags differ), budgets capped at 8,000. v3b's public rejection of this (20 Nov) was at k=1, where the inflated s122 partly made up for under-flagging.

**Ruled out while looking for a structural edge:**
- **No id or row-order leak.** Failure rate, repair cost and sensors are flat across id order; lag-1 autocorrelation of failure is 0.008.
- **No exact repair-cost formula.** Repair costs are integers (27 at exactly 5, 2 at 8,000). Log-log ridge reaches R² 0.78, lasso 0.80, XGB 0.81: the same ceiling for every model family.
- **Apart from the counters, the depot difference is rounding.** Without s120/s122, adversarial AUC 0.94 comes almost entirely from s055 and s062, which the test depot records in multiples of 10 (train has finer values). Share-of-total KS is ≤ 0.04 for every bin of the realigned histogram, so the v1 realignment is right.
- **CV cost is decision-optimal given p and the cost distribution.** At k=1: inspections 0.290, under-budget 0.292, over-budget 0.093, missed 0.171 per vehicle. Missed failures are only 4.9% of failure cost; the under-budget cost comes mostly from vehicles flagged at p < 0.09 with zero budget (flagging turns a cost c into 10 + 0.2c), which is the intended trade-off.

## v22 public result and v23: is repair cost higher at the new depot? (25 Nov)

**v22_nos122 public: 17,172.57**, 333 worse than v12_robust. Every ranking change to v12 has now scored worse on public (v13 +1,135, v18_ens10 +784, v14 +536, v15 +218, v22 ~+200, mean ≈ +570), although each was as good or better in CV. The most likely reading is that v12's public score was about 500 lucky.

**Cost-level hypothesis (alpha = test repair costs / training).** Model-expected difference vs v12 under different true alphas, compared with what each file actually scored (`src/v23_alpha.py`):

| file | observed vs v12 | α=1 | α=1.25 | α=1.5 | α=2 |
|---|---|---|---|---|---|
| v17_bud125 | −14 | +333 | −157 | −575 | −1,474 |
| v17_k135 | +68 | +122 | −89 | −290 | −649 |
| v9_k4 | +174 | +222 | −36 | −264 | −685 |

All three point to **α ≈ 1.1–1.15**, not 1.5. At that level a consistent re-decision gains only about 100. `submission_v23_alpha15.csv` (α=1.5: 1,955 flags) would be about +300 if α ≈ 1.15. **Not recommended for upload.** The cost level does not explain the gap to the top teams either.

## v24: informative censoring, i.e. hidden failures in the training labels (25 Nov)

**Finding.** Healthy-labelled training vehicles with follow-up < 365 days look about 10× riskier to the OOF classifier than healthy vehicles followed for the full year: mean p 0.007–0.015 vs 0.0013, and 2.3–3.6% vs 0.5% above p = 0.05. Under random censoring the two groups would look the same. So censoring is informative: these records most likely stop because the vehicle failed, unrecorded. A complete-case model puts **≈ 211 expected hidden failures in train against 397 recorded (+53%)**. If test labels are complete, this alone explains part of the depot's higher failure odds (k).

**v24_pu.** Instead of dropping censored-healthy rows (v18 IPCW), each one gets a soft label π = P(fail | x) from a model trained on complete cases, and the final classifier trains on all rows (censored rows duplicated with weights π / 1−π). Complete-case CV: −0.029 ± 0.011 per vehicle vs base (12/15 folds); doubling π gives −0.041 (10/15). Complete-case CV cannot see the benefit on the hidden failures themselves, since those rows are excluded from evaluation. `submission_v24_pu.csv`: flag count matched (1,761), 172 flags differ from v12, Spearman 0.906, budgets capped. `src/v24_pu.py`.

**v24_pu public: 17,575** (+735 vs v12), close to v18_ens10 (+784). Both censoring "fixes" lost, so the test labels are most likely censored the same way as train. Sensors predict short follow-up among healthy vehicles (AUC 0.72), so a classifier trained on the raw labels already predicts what the test labels record. v24_pu_x2 was not uploaded.

## v25: risk rises more steeply with wear at the new depot (25 Nov)

Every file that reduced the model's reliance on wear sensors lost (v14 +536, v15 +218, v22 +333). Fit to all 20 public scores of p = sigmoid(a + logit(q_v12) + c·z), with z the standardised log wear counter on test (`src/v25_tilt.py`, `reports/v25_tilt.csv`):

| wear covariate | Δlog-lik vs c=0 at c = −0.6 / −0.3 / 0.3 / 0.6 / 1.0 |
|---|---|
| s120 | −10.7 / −4.0 / ≈+2.0 / **+2.73** / +0.3 (interior optimum c ≈ 0.6) |
| s122 | −4.7 / −1.8 / ≈+1.2 / +2.1 / +2.8 (still rising at 1.3: +2.9) |
| s126 | −4.2 / −1.5 / ≈+1.0 / +1.7 / +2.2 |

Positive tilt is favoured on every counter (likelihood ratio ≈ 15 for one parameter); negative tilt is clearly rejected. **`submission_v25_tilt.csv`**: q' = sigmoid(logit q + 0.6·z_s120), flag count matched (1,761), 144 flags differ from v20_cap (added flags median s126 27.9M vs removed 5.2M), budgets capped. Model-expected public difference vs v20_cap if the true tilt is 0 / 0.3 / 0.6 / 0.9: **+568 / −56 / −747 / −1,365**. Averaged over the tilt profile likelihood, the expected gain is about −500. `src/v25_build.py`.

## v26: back to fundamentals, the problem statement and what the sensors are (25 Nov)

**Problem statement (Classroom PDF, Problem 2), re-read in full.**
- The scoring rule matches `src/score.py` exactly.
- "Vehicles, columns and outcomes have been anonymised and **partly simulated** for this course; repair costs are simulated."
- `followup_days` is "how many days of records exist after the service visit", and `failure` is a failure *recorded* in that window. This fits the v24 finding that test labels are censored the same way as train.
- No hidden rule or quantity was found that we were not already using.

**Every sensor is a usage counter.** Rank correlation with s126 is 0.88–0.98 for every top sensor (s122, s123, s149, s053, s034, s058). Failure rate rises steeply and keeps rising at extreme usage (s126 train bins: ≤q50 0.04%, q90–95 6.1%, q95–98 18.2%, q98–99 22.7%, q99–99.9 24.4%, >q99.9 30.4%). The test fleet has about twice the share in the top bins (s126 > train q95: 3.7% of test vs 2.0% of train). The v12 classifier already predicts higher risk there (mean q 0.34–0.40 for test vehicles above train q98), so tree extrapolation is not what limits it.

**Failure risk is not a simple formula in the sensors.** Logistic regression on log1p sensors plus missing indicators: +0.108 per vehicle vs XGBoost in 3×5 CV (1/15 folds better); a 50/50 blend is also worse (+0.032). `src/v26_linear.py`.

## v28: posterior harness over ~2,200 test-depot hypotheses (25 Nov)

`src/v28_harness.py`. A truth is p = floor + (1−floor)·sigmoid(a + b·L + c·z), where L blends logit q_v12 with one of 9 other rankings we built (nos122, pu, pu_x2, ens10, ens4, fix120, fix, nodrift, v15-style map) at w ∈ {0, 0.5, 1}. z is a wear tilt on s120/s122/s126/s149 (c ∈ {0.4, 0.8, 1.2}), b ∈ {0.85, 1, 1.15} is the slope, and repair-cost level α ∈ {1, 1.15, 1.3}. Each of the 2,235 truths is weighted by the exact likelihood of all 20 public scores.

**Posterior.**
- No tilt: 5%. Weight is spread over tilt strengths (0.4: 30%, 0.8: 34%, 1.2: 31%) and sensors (s122 32%, s149 24%, s126 23%, s120 17%).
- Blended rankings dominate: w=0.5 52%, w=1 42%, v12 alone only 5.5%.
- Slope b ≥ 1 is favoured (1.15: 41%); α spread over 1.0–1.3; an unexplained-failure floor gets no weight.

**Candidates scored by posterior-expected private cost vs v20_cap:**
- **Bayes decision under the full posterior (`submission_v28_bayes.csv`):** −1,206, with 1,505 flags (fewer than v12's 1,761: 294 differ). It is better than v20_cap under every one of the 20 heaviest truths (from −475 to −2,150).
- With level ×1.2 and budgets ×0.9 (`submission_v28_bayes_lv12.csv`): −1,058 with 1,590 flags; it has the best worst case (−517).
- For reference: v12_robust +131 (the budget cap), v22 +46, v17_bud125 +196, v25_tilt +302, v24_pu +386, v17_k135 +677. These agree with how those files actually scored.

**Caveat.** These are model-based expectations. The posterior is fitted to the same 20 public scores, so part of the edge may be fitting the public subset. The next public score is the real test, and it goes back into the posterior.

## v29: harness round 2 (25 Nov)

`src/v29_harness.py` adds about 2,200 truths to round 1's (4,400 in all):
- three-way blends of v12 with pairs of fix / map / nodrift / nos122 / fix120,
- wear tilts on two sensors (s122 plus s149, s126 or s120),
- finer tilt values (0.6 / 1.0 / 1.6) and cost level ×1.45.

**Posterior.**
- Tilt on s122: 75%, centred on c = 0.8 (50%).
- Rankings: blends led by `fix` (29%) and `map` (23%); v12 alone 2%.
- Slope b = 1–1.15; cost level spread over 1.0–1.3 (×1.45: 13%).

**Bayes decision (`submission_v29_bayes.csv`, 1,459 flags):** posterior-expected private cost −1,687 vs v20_cap, against −1,593 for v28_bayes. It is better than v20_cap under all 20 heaviest truths (worst −865). Expected score on a random 30% of test: **14,696** (v20_cap 15,418). Round 2 adds only about 95 over round 1, so within this hypothesis family the decision has converged.

**v29_bayes public: 17,196** (predicted about 16,750, 80% range about 16,000–17,500). It is 357 worse than v12_robust and 450 worse than predicted, but inside the range. Round 3 (`src/v30_harness.py`) re-weights the ~4,000 plausible truths with this 21st score.

**Round 3 result.** The 21st score barely moves the posterior (tilt on s122 c≈0.8; fix/map-led blends; α 1.0–1.3). The round-3 Bayes decision is essentially v29_bayes (1,460 vs 1,459 flags). **v29_bayes: posterior-expected private −1,611 vs v20_cap, better under all 20 heaviest truths (worst −528, best −2,052).** Expected score on a random 30%: v29 14,635 vs v20_cap 15,326.

**The case against.** Every file that moved away from v12's flags has scored worse on public: v13 +1,135, v18 +784, v24 +735, v14 +536, v29 +357, v22 +333, v15 +218. The harness explains this as selection: v12 was the best of about a dozen public scores, so its flags fit the public 30% unusually well. That is plausible, but the public scores alone cannot confirm the private gain.

## v31: textbook label-shift estimators from the literature (25 Nov)

`src/v31_labelshift.py`, `src/v31_meanp.py`.
- **BCTS + EM/MLLS** (Alexandari et al.; Saerens et al.): binary BCTS fits T = 0.816, b = +0.036, effectively our Platt scaling. EM gives a test failure rate of **4.5% (k = 6.5)**, the same as the v7 EM estimate.
- **BBSE** (Lipton et al.), thresholds 0.02–0.5: **4.0–5.0% (k = 5.8–7.4)**.
- **Harness posterior (21 public scores):** mean test failure rate **3.2%** (90% range 2.8–3.6%). Truths with ≥ 4% get zero weight. The one direct test of k ≈ 6.5 (v9_em) scored 18,058, about 1,000 worse than k = 4.
- **Reading.** Both estimators assume pure label shift (p(x|y) unchanged), and the depot also has covariate shift (more worn fleet, rescaled s122). The literature notes these estimators are biased under such violations, so their agreement does not make them right.
- **Effect on the decision.** Scaling the v29 Bayes probabilities up toward a 4.5% rate (level ×1.4) costs about 480 in harness-expected private cost vs v29 and has a worse worst case. **No change: v29_bayes stays the pick.**

## v32: shift-aware CV (a new validation strategy) (25 Nov)

`src/v32_shiftcv.py`. Each validation fold is made to resemble the test depot:
- validation vehicles are weighted by the test/train density ratio of log s126 (ESS 44,104 of 44,800; weighted failure rate 1.02% vs 0.71%);
- two scenarios: 'real' (sensors unchanged) and 'instrument' (validation s122 ×2.55, s120 + 16.0M);
- Platt is fitted on unshifted OOF from the other folds, as in the real pipeline;
- each candidate is scored at its own best level, so only the ranking is compared.

| ranking (cost/veh vs all sensors, 3×5 folds) | instrument | real |
|---|---|---|
| no s120/s122 (v22) | **−0.061 ± 0.009 (13/15)** | **−0.019 ± 0.009 (12/15)** |
| map + no-s122 blend | −0.058 (14/15) | −0.016 (11/15) |
| all + no-s122 blend | −0.046 (15/15) | −0.017 (11/15) |
| oracle-fixed sensors | −0.042 (14/15) | 0 (identical) |
| quantile-mapped (v14/15) | −0.039 (12/15) | +0.003 (9/15) |

**Dropping s120/s122 is the best ranking under both stories.** Mapping and fixing help only if the quirks are instrument faults.

**Combined with the harness (`src/v32_harness.py`).** The Bayes decision restricted to harness truths whose ranking includes no-s122 (27% of the posterior) scores −1,585 vs v20_cap under the full posterior. That is within 26 of v29_bayes (−1,611), with a worst case of −450 over the 20 heaviest truths. Expected score on a random 30%: 14,647. **`submission_v32_bayes_nos.csv`** (1,456 flags) is the decision both validations agree on.

## v33: leave-one-out-style calibration check (25 Nov)

Literal leave-one-out (56,000 XGBoost fits) is impractical. Its practical stand-in: Platt fitted on 20-fold OOF (models see 95% of dev) vs the 5-fold OOF used so far (80%). `src/v33_kfold_calib.py`.
- Mean test q: 0.02015 vs 0.02032. The ranking is identical (Spearman 1.000000).
- logit q20 = 1.035 · logit q5 + 0.020: a small affine change in logit, which the harness's fitted level a and slope b (grid 0.85 / 1 / 1.15) absorb.
- Share of test above p = 0.05: 5.21% vs 5.37%.
- **Conclusion:** 5-fold calibration is adequate, and the v29 / v32 decisions are unaffected.

## v35: Daksh's Gaussian-process budgets combined with our flags (27 Nov, deadline day)

**Leaderboard 27 Nov:** we are 9th at 16,587.75 (Daksh's v33_gp_budget). Six teams sit between 15,750 and 15,930.

**Gaussian-process repair-cost model (Daksh's v33, refit in `src/v35_gp.py`).**
- Out-of-fold R² 0.814 vs 0.798 for the tree, on the same 383 failures. Correlation with the tree on test is only 0.84.
- Budget loss at the newsvendor level (q = 0.909, OOF): **GP 18.7 vs tree 24.1 (−22%)**. GP/tree blends land in between, so budgets use the GP only.
- v33's actual public gain over the v12-like files (16,588 vs 16,826) supports the GP budgets.

**Candidates (`src/v35_build.py`)**
- `v35_v33cap`: v33 with its 7 budgets above 8,000 capped.
- `v35_gpflags`: v33's k=3 rule with GP costs in the flag decision too (1,768 flags; 136 differ from v33).
- `v35_combo`: harness flags (v32_bayes_nos, 1,456) with GP newsvendor budgets from the harness's risk estimates, × the posterior-mean cost level 1.194, capped at 8,000.

**Harness-expected score on a random 30% (`src/v35_eval.py`), with three repair-cost truths (vs v33_gp_budget):**

| file | tree truth | GP truth | 50/50 mix |
|---|---|---|---|
| v35_combo | −677 | **−661 (best)** | **−669 (best)** |
| v32_bayes_nos | −1,168 | −122 | −645 |
| v35_gpflags | −46 | −115 | −80 |
| v35_v33cap | −72 | −72 | −72 |

Each truth favours budgets built from its own cost model. v35_combo is the only file that is strong under all three, and the public evidence (v33 beating the tree-budget files) favours the GP truth. **v35_combo is the recommended final.** Its public score is likely about 16,950 (v29's 17,196 minus about 240 from GP budgets), because the public 30% has favoured v12-like flags all along.

**v35_combo public: 16,878** (predicted about 16,950). That is 318 better than v29 (same flags, tree budgets: 17,196), so the GP budgets carry over to our flags, and 290 behind v33 (16,588) because of the flags.

## v36–v37: more hypotheses in the style of Daksh's v33, and the harness on GP costs (27 Nov)

**Repair-cost side (`src/v37_costvar.py`, honest 5-fold OOF on the 397 failures).**

| GP inputs | R² | budget loss q=.909 | q=.5 | expected budget loss over the flagged-test level mix |
|---|---|---|---|---|
| v33: 6 histograms + usage | 0.814 | 18.67 | 41.08 | baseline |
| + wear group (log s126, s149, s123) | 0.829 | 18.58 | 39.22 | −2.6% |
| + top-6 non-histogram cost sensors (s049, s116, s142, s105, s010, s104) | 0.816 | 17.38 | 39.45 | −5.8% |
| **+ wear + sensors (v37)** | **0.832** | 18.18 | **37.48** | **−4.6%** |
| kNN-15 on histogram shapes | 0.687 | 31.18 | 54.91 | n/a |

- Blends with kNN or the tree do not help.
- All added sensors are stable between depots (KS ≤ 0.032).
- On test, v37 predicts lower costs than v33's GP only in the low-wear bulk (mean log cost 5.48 vs 5.87, now in line with the tree's 5.47). On flagged vehicles they agree (5.83 vs 5.79), and only 0.1% of flagged test vehicles are more worn than any training failure.

**Failure side (`src/v37_riskkernel.py`).** A histogram-kernel failure model (Nyström RBF features + logistic regression) is far worse than XGBoost (+0.73 per vehicle alone), and every blend hurts (+0.11 to +0.28). XGBoost stays the failure model.

**v36 harness: GP cost truth, 24 public scores (`src/v36_harness.py`).** Adds v32_hist_kernel, v33_gp_budget and v35_combo. The posterior barely moves: s122 tilt c≈0.8, fix/map-led blends, α 1.15–1.3.
- With v33's GP costs: joint Bayes flags + budgets 13,673 on a random 30% vs v35_combo 13,746 and v33 14,433.
- **With v37's GP costs (`GPNPZ=reports/v37_gp.npz`): joint Bayes 13,393 vs v35_combo 13,644 and v33 14,417.** It is better than v33 under all 20 heaviest truths. These numbers judge budgets under the same cost model that built them, so they favour it somewhat.
- **`submission_v37_bayes_gp2.csv`**: 1,435 flags. Next upload to test.

**v37_bayes_gp2 public: 16,410**, our best so far: −178 vs v33 (16,588) and −468 vs v35_combo (16,878). The prediction was 16,600–16,700.

## v38: per-vehicle GP uncertainty, and the harness with 25 scores (27 Nov)

- **Per-vehicle GP uncertainty (Daksh's v34 idea) on the v37 GP (`src/v38_gpsigma.py`).** OOF sigma barely tracks the actual errors (corr(|resid|, sigma) 0.05). Budgets built on it lose +2.3% more over the flagged-test level mix (+0.5% with a damped version). Rejected, as in Daksh's holdout check.
- **Harness with 25 public scores and v37 GP costs (`src/v38_harness.py`).** The posterior is stable: s122 tilt c ≈ 0.8, fix/map blends, α shifted slightly up (1.3: 34%). The re-derived joint decision (1,445 flags) is within 8 of v37_bayes_gp2 in expected private cost (−2,966 vs −2,957 vs v20_cap). **The loop has converged on v37_bayes_gp2; a re-derived file would not be worth an upload.**

## v39: richer cost-model inputs (27 Nov)

- **A. Greedy sensor search (`src/v39_costsearch.py`).** 20-sensor pool; a move is kept only if it cuts the flagged-mix budget loss by > 1%. Selected sensors: **s116, s142, s105, s104, s078** (s078 added, s049 and s010 removed). Loss −4.6% on the search split, **−3.1% on an independent fold seed**. OOF R² 0.835 (v37: 0.832). s078 KS 0.023.
- **B. Cost-5 floor.** 27 of 397 failures cost exactly 5. The v37 GP already separates them (AUC 0.986; mean predicted log cost 2.0 vs 5.3), so their budgets are already near the floor. No two-part model needed.
- **Candidate `submission_v39_bayes_gp3.csv`** (v38 posterior, v39 GP costs; 28 flags differ from v37). Cross-evaluated under three GP cost truths, vs v37_bayes_gp2: **+1 (v33 GP), +47 (v37 GP), −58 (v39 GP)**. Each truth favours its own budgets, and the net is a wash. **The 3% budget-loss gain is too small to show up in expected cost; v37_bayes_gp2 stays our best file.** Cost-model tuning is exhausted at this level.
- **Consistency check.** Under the v33-GP truth, v35_combo would beat v37 by 173. On public, v37 beat v35_combo by 468, so the v37 cost model is closer to the truth than v33's.

## v40: three more hypotheses (27 Nov), all rejected

- **H1: failure model trained on a test-like fleet** (training rows weighted by the test/train density ratio of log s126; `src/v40_wearw.py`, shift-aware CV). It is worse than unweighted: no-s122 −0.061 / −0.019 vs no-s122 weighted −0.042 / −0.000 (instrument / real); all-sensors weighted −0.012 / +0.003.
- **H2: repair-cost level tilting with wear at the new depot** (cost × α × exp(d·z_s126); 150 heaviest truths, α ∈ {1.15, 1.3}; `src/v40_costtilt.py`). Posterior over d: −0.3 1.5%, −0.15 27%, **0 51%**, +0.15 20%, +0.3 0.7%. The best fit is at d = 0, so there is no wear-dependent cost shift.
- **H3: flag-count and budget-size scan** (harness level × budget-scale grid under GP costs, `reports/v38_cands.csv`). The joint decision at level 1, budget ×1 is best. Level 0.85–1.2 and budget 0.9–1.15 lose 100–1,200 in expected private cost, so v37 sits at the optimum on both.

## v41: variants of the no-s122 failure model (27 Nov)

Shift-aware CV, one scenario (the model does not read s120/s122). Cost per vehicle vs the no-s122 model with v12 params (`src/v41_nosvar.py`):

| variant | Δ cost/veh | better |
|---|---|---|
| **slow (1,000 trees, lr .02, colsample .5)** | **−0.0155 ± 0.0071** | 11/15 |
| 5-seed bag | −0.0155 ± 0.0103 | 8/15 |
| + GP predicted log cost as a feature | −0.0024 ± 0.0093 | 6/15 |
| LightGBM | −0.0003 | 7/15 |
| L2/gamma regularised | +0.0041 | 6/15 |
| depth 6, min_child_weight 5 | +0.0063 | 7/15 |
| depth 3, 600 trees | +0.0185 | 7/15 |
| CatBoost | +0.0218 | 5/15 |

**Slow no-s122 as a harness ranking (`src/v41_slow_q.py`, `src/v41_harness.py`).** Test Spearman vs no-s122 is 0.893. Each truth using no-s122 was duplicated with nos_slow and re-weighted on 25 public scores. The two get equal weight (20.0% vs 20.5%), so the public scores cannot tell them apart. The re-derived joint decision is within 9 of v37_bayes_gp2 (−2,997 vs −2,989). No new file is worth an upload.

## v42: repair-cost spread by predicted-cost band, plus the v39 sensor set (27 Nov)

**Course announcement (26 Nov 22:47):** do not retrain at the last minute, save artifacts, make inference reproducible. Our competition takes CSV uploads, so nothing runs on Kaggle at the deadline. Every v37/v42 input is saved in `reports/` (GP predictions and residuals, harness posterior, ranking probabilities), so the files rebuild without retraining.

**Banded residual spread (`src/v42_residbins.py`).** Budget quantiles come from OOF residuals of failures in the same predicted-log-cost band, instead of one pooled set. Honest cross-fitted check (band quantiles estimated on the other folds' failures), budget loss over the flagged-test level mix vs pooled:

| | 2 bands | **3 bands** | 4 bands |
|---|---|---|---|
| fold seed 42 | −0.1% | **−3.5%** | −1.7% |
| fold seed 7 | −1.2% | **−3.1%** | −2.2% |

**Combined with the v39 sensor set (`src/v42b_combined.py`)**, vs the v37 GP with pooled residuals, honest:

| fold seed | v37 + 3 bands | v39 pooled | **v39 + 3 bands** |
|---|---|---|---|
| 42 | −3.5% | −4.0% | **−7.9%** |
| 7 | −3.1% | −3.1% | **−6.6%** |
| 123 (unused for any choice) | −2.2% | −4.8% | **−6.2%** |

The two gains stack. Budget loss is about 12,300 over the full test, so −6 to −8% is roughly −850 over the full test: about −250 on a 30% scale and −600 on private.

**`submission_v42_gp_bands.csv`.** Harness joint decision under v39-GP costs with 3-band spread (`BANDS=3 GPNPZ=reports/v39_gp.npz python src/v38_harness.py --reuse BAYES_lv1_b1`, from the saved v38 posterior). It has 1,452 flags (35 differ from v37) and mean budget on common flags 670 vs 644. Under its own cost truth it scores −137 vs v37 on a random 30%. The honest OOF budget check is the better evidence and points to about −250. Next upload to test.

**v42_gp_bands public: 16,487** (+77 vs v37's 16,410). Under the model, the public difference between v42 and v37 has sd ≈ 453 (private ≈ 690), because the files differ mainly in budgets, which only the few failing public vehicles feel. So +77 is z ≈ 0.1: public cannot distinguish them. The honest OOF budget evidence (−6 to −8% on 3 seeds) still favours v42 for private.

## v43: other residual-spread models (27 Nov)

Honest cross-fitted budget loss vs v42's 3 bands by predicted cost, three fold seeds (`src/v43_spread.py`):

| spread model | seed 42 | seed 7 | seed 123 |
|---|---|---|---|
| pooled (v37-style) | +4.2% | +3.7% | +1.6% |
| smooth location-scale (log\|r\| linear in predicted cost) | +2.2% | −0.1% | −2.0% |
| 3 bands by wear (log s126) | −0.8% | +0.7% | −4.5% |
| 3×2 bands (predicted cost × wear) | −1.1% | +1.3% | −1.3% |

None beats the v42 bands consistently; differences are within ±2–4% and change sign across seeds. **v42's spread model stays.**

## v44: GP kernel width and bagging (27 Nov)

Honest budget loss vs v42 (v39 inputs, 3 bands), `src/v44_gpkernel.py`:
- width ×0.5: +7.5 / +5.3 / −0.5%; ×0.75: +3.1 / +3.2 / −0.2%;
- ×1.5: −0.1 / +0.2 / +0.1%; ×2: +0.3 / +0.4 / +0.1%;
- 10-bootstrap bag: +30.6 / +11.0 / +6.3% (resampled duplicates distort the GP noise estimate).

The median-heuristic width is already optimal, and bagging hurts. v42 is unchanged.

## v45: GP repair costs in the flag decision, end-to-end check (27 Nov)

Shift-aware CV with the slow no-s122 failure model (nested Platt). Flags and budgets come from each candidate's cost distribution, and the GP is refit on each training fold's failures only (`src/v45_flagmu.py`). Cost per vehicle:

| cost model used for flags + budgets | cost/veh | vs tree | better |
|---|---|---|---|
| tree (v12) | 1.1735 | n/a | n/a |
| **GP (v39 inputs)** | **1.1348** | **−0.039 ± 0.015 (−3.3%)** | 11/15 |
| 50/50 log blend | 1.1466 | −0.027 ± 0.009 | 12/15 |

This confirms in CV what the public scores showed: the GP cost model improves the whole decision, not just budgets. That is the direction v37 and v42 take. The blend is more consistent but smaller on average, so the GP alone stays.

## v46: harness with 26 public scores (27 Nov)

`src/v46_harness.py` (v38 harness, v42 repair-cost model, v42's 16,487 added). The posterior barely moves (heaviest rankings `fix`, `map`,
`nos122`; s122 wear tilt 71%). Expected public-scale score: v42 13,697, new Bayes decision 13,693, v37 13,810. The new Bayes decision is
v42 to within 4 points, so no new file from this step. The harness's absolute expectations run about 2,800 below the actual public scores of
the recent files; only its differences have been of some use.

## v47–v50: ablation of the GP repair-cost inputs (27 Nov)

One change at a time against v42, honest cross-fitted budget loss on the same 383 failures (`src/v47_ablation.py`, 26 variants, 5 seeds):
- necessary groups (removing them costs a lot): histograms s080 (+38%), s054 (+20%), s037 (+10%), s154 (+7%), the sensor group (+6%);
- no better: Matern or rational-quadratic kernels, raw bin levels instead of shares, other target transforms (cost^0.25, cost^0.5), all
  other sensors, their principal components, a missing-value count, one kernel per sensor;
- small gains from REMOVING inputs: the usage (exposure) group, the s023 histogram, the s129 histogram.

Confirmation on ten fresh fold seeds (`src/v48_combo.py`): removing all three gives **−4.9% (9/10)**; vehicle bootstrap sd 1.9%,
97–99% of resamples better (`src/v49_round2.py`). A second backward round found one more candidate, wear sensor s123 (−2.2%, within noise).

Stress tests (`src/v50_stress.py`, reduced inputs vs v42): same gain under the v37 and v12 flag mixes and 10-fold CV (−5.5%); −4.2% when
failures are weighted to the test depot's wear; **−8.9% when trained on low-wear failures and scored on the top-wear third**.

Caveat: the groups to remove were chosen using these same 383 failures. Fresh fold seeds guard against fold luck, not against selection on
this sample, so the real gain is probably smaller than the measured one.

## v51: reduced-input GP (27 Nov)

`src/v51_gp_fit.py`, ten more fold seeds: A (without exposure, s023, s129) −6.1% (10/10), wear-weighted −4.7%; B (also without s123)
−7.6% / −6.9%; **C (average of A and B in log cost) −7.4% / −6.4% (10/10)**. C is used (less dependence on the last, weakest choice).
On test it correlates 0.994 with v42's costs (mean log shift +0.017, sd 0.10). `src/v51_harness.py` makes the joint decision under C.

## v51 harness, v52 end-to-end check (27 Nov)

**`submission_v51_gp_reduced.csv`**: harness joint decision under the v51 C costs (`src/v51_harness.py BAYES_lv1_b1`). 1,433 flags, 27
different from v42 (4 added, 23 removed); median budget change on common flags 11.

End-to-end shift-aware CV (`src/v52_flagcv.py`, 5 repeats x 5 folds, GP refit inside each fold, 3-band spread):

| GP inputs | cost/veh | vs v42 inputs | folds better |
|---|---|---|---|
| v42 | 1.1365 | n/a | n/a |
| A (no exposure, s023, s129) | 1.1153 | −1.9% | 19/25 |
| B (A without s123) | 1.1132 | −2.0% | 17/25 |
| **C (average, used)** | **1.1121** | **−2.15%** | **18/25** |

The fold standard errors (±0.008) ignore the overlap between folds and the choice of inputs on the same failures, so they overstate certainty.

Public scores cannot choose between GP variants: log-evidence −177.35 (v51 C), −177.38 (v42 inputs), −176.95 (v37 inputs), against
−179.16 for tree costs. Posterior-expected public-scale score, new file against v42: −79 if v51's costs are true, +72 if v42's are.
Expected public difference −94 with noise sd about 290, so one public score will not settle it either.

**v51_gp_reduced public: 16,388**, the best public score so far (−99 vs v42, −22 vs v37; the model expected −94 vs v42, noise sd about 290).
The sign agrees with both CV checks, but one score of this size is not evidence on its own.

## v53–v58: after the v51 upload (27 Nov)

- **v53 harness, 27 public scores** (`src/v53_harness.py`): its decision matches v51 (5 flags differ, median budget change 0; expected score
  equal to the point). Not a new file; kept in `submissions/rejected/`.
- **v54–v56, GP settings** (`src/v54_gpsettings.py`, `v55_confirm.py`, `v56_prior.py`): optimiser iterations, amplitude bound, number of
  bands, kernel-smoothed spread and Student-t noise give nothing or hurt. A tighter prior on the kernel amplitudes helped budget loss by
  2 to 4%. Budgets above the honest quantile helped on training failures (about −7% at ×1.19, flat to ×1.3); the submissions already sit
  at about ×1.14 through the harness's cost-level mixture, so no level change was made.
- **v57, amplitude prior N(0, 0.05)** on fresh seeds (`src/v57_amp.py`): −2.9% at level ×1.0, −2.3% at ×1.19 (9/10 seeds, vehicle
  bootstrap p ≈ 0.9).
- **v58, end-to-end check** (`src/v58_flagcv.py`): **no gain** (+0.14% cost per vehicle, better in 9/25 folds). **v57 rejected**; v51 stays.

## v59: where the cost comes from (27 Nov)

Model expectation for v51 on the public 30% (13,546): inspections 32%, shortfall on flagged failures 26%, unused budget on healthy flags
16%, missed failures 15%, unused budget on failures 12%. Under the same model, knowing every repair cost would save about 650; knowing
which vehicles fail would save about 8,200. The cost model has little left to give; the failure ranking is the large uncertainty.

## v60–v62: new failure-model features, and new rankings in the harness (all rejected)

Monotone wear constraints, histogram shares, per-usage rates, rates only and logistic regression (`src/v60_feats.py`): none beats the
base classifier in wear-reweighted CV (best alternative +0.003 per vehicle). Added to the harness as ranking hypotheses (`v61`, `v62`),
they take modest weight and move the decision by 20 flags and 8 expected points. Not uploaded.

## v63–v65: survival models for the failure ranking (27 Nov)

XGBoost accelerated failure time (normal errors), `followup_days` used only as the training target; censored vehicles give a lower
bound. Cox was much worse.

| check | classifier (base) | AFT result |
|---|---|---|
| v63, wear CV 3x5 | 1.1653 | scale 1: 1.1479 (−1.5%, 11/15) |
| v64, fresh folds 5x5 | 1.1888 | depth 6: −2.2% and scale 2: −2.0% (18/25, 5/5 repeats); scale 1: −0.6% |
| v64, train low wear, score top 20% of s126 | 6.35 (AP 0.34) | all variants 5.8–6.2 (AP 0.35–0.38) |
| v65, locked 20% hold-out (79 failures), wear-weighted | 1.0348 | ensemble −2.7% (bootstrap p 0.85); depth 6 −6.6%; scale 2 +0.2% |

The hold-out was touched here once, outside the v0/v4/v8 convention, because it was the only data not used to pick the variants.
Warning: earlier censoring-aware methods (v18 IPCW, v24 soft labels) looked good offline and lost on the public board.

## v66: survival rankings in the harness (27 Nov)

`src/v66_harness.py`: AFT rankings get 0.20 posterior weight from the public scores alone. `submission_v66_aft.csv` gives AFT hypotheses
4x prior weight (0.50): 1,427 flags, 38 different from v51. Expected public-scale score: 13,594 against v51's 13,611 under that mixture,
13,584 against 13,617 under AFT-only hypotheses. The noise of the public difference is about 350, far above the expected gain.

## v67: seven validation setups, backtested against public results (27 Nov)

`src/v67_valmethods.py`, 80% dev split, cost per vehicle at each candidate's own best flag level; change against the v12 classifier
(negative is better), bootstrap probability of being better in brackets. Setups: plain 3x5 CV; wear-weighted; complete cases with
inverse-censoring weights; test-depot sensor readings applied to validation (instrument); leave-vehicle-cluster-out; train on the lower
80% of wear (ho-wear) or of total usage (ho-usage), score the rest.

| candidate | plain | wear | complete | instrument | cluster | ho-wear | ho-usage |
|---|---|---|---|---|---|---|---|
| nos122 | +2.1% (0.10) | +2.1% (0.11) | -0.0% (0.47) | -0.2% (0.56) | -1.2% (0.66) | -3.3% (0.93) | +0.3% (0.40) |
| ipcw | -0.5% (0.57) | -0.4% (0.57) | -4.3% (0.93) | -1.0% (0.66) | -6.8% (1.00) | +8.1% (0.30) | +6.5% (0.03) |
| pu | -0.8% (0.69) | -0.5% (0.64) | -1.9% (0.71) | +0.4% (0.40) | +0.3% (0.42) | -4.4% (0.98) | +3.5% (0.10) |
| bag3 | -0.2% (0.62) | -0.2% (0.66) | +0.4% (0.38) | +0.8% (0.26) | -3.2% (0.95) | -2.1% (0.86) | -1.0% (0.59) |
| drop4 | -0.8% (0.69) | -1.1% (0.76) | +0.6% (0.39) | -3.1% (0.99) | -3.6% (0.96) | -3.8% (0.92) | -4.1% (0.98) |
| aft | -3.2% (0.90) | -3.3% (0.90) | -2.4% (0.87) | -5.4% (0.97) | -4.2% (0.95) | -11.8% (1.00) | -5.1% (0.99) |
| aftd6 | -1.9% (0.85) | -2.2% (0.87) | -2.4% (0.86) | -4.2% (0.96) | -7.8% (1.00) | -6.9% (0.98) | -6.6% (1.00) |
| clf+aft | -2.8% (1.00) | -2.9% (1.00) | -2.1% (0.84) | -5.1% (1.00) | -2.8% (0.89) | -7.1% (0.99) | -3.5% (0.96) |

Backtest on the five changes known to score worse than v12 publicly (nos122, ipcw, pu, bag3, drop4): each setup calls only 1 to 3 of the
5 worse (ho-usage 3; complete and instrument 2). Correlation with the size of the public loss runs from -0.79 to +0.47 on four points.
No offline setup is a reliable guide to public ranking results. The survival ensemble is ahead under all seven.

Adversarial validation was dropped: after removing s120, s122 and the columns that differ only in rounding, train against test AUC is
0.51.

## v68: histogram column resolution (27 Nov)

The v1 histogram realignment is right, but each column keeps its own recording resolution, so test bins arrive rounded differently
(this alone lets a classifier tell the depots apart). Training on data rounded the same way changes the test top-1,433 less
(1,390 shared) than a different random seed does (1,369). No change made.

## v69: survival ranking replaces the classifier (27 Nov)

`src/v69_aftfull.py`: 1,512 hypotheses using only survival rankings, fitted to the 27 public scores. Best public log-likelihood
-190.425 against -190.422 for the best of all v66 hypotheses, so the public scores do not separate the two rankings.
`submission_v69_aftfull.csv`: 1,365 flags, 120 different from v51 (26 added, 94 removed).

| expected score, public 30% scale | v51 | v66 | v69 |
|---|---|---|---|
| survival-only hypotheses | 13,688 | 13,629 | 13,529 |
| full v66 posterior (all rankings) | 13,607 | 13,601 | 13,741 |

The model's expectations run about 2,800 below actual public scores; only the differences mean anything, and the noise of a public
difference is a few hundred. Rebuilt exactly by `notebooks/v69_aftfull.ipynb`. Not uploaded.

## v70–v71: why the public scores sit above the model, and borderline vehicles (27 Nov)

`src/v70_gap.py`: under the best-fitting hypotheses all 27 uploads score above their prediction, by about +2,000 to +3,500 each (v2
+12,000). The residuals are almost perfectly correlated (GLS chi2 12.7 on 27 scores), so they amount to one shock: a few expensive
failures in the public 30% that every file pays for. Predictive sd of one file's public score is about 1,850; v51 is expected at about
13,450 and scored 16,388 (+1.6 sd).

`src/v70_parts.py`: scaling cost components by GLS gives shortfall x1.27 and missed failures x1.23 (both ±0.2), chi2 12.7 -> 10.5.

`src/v70_pairs.py` (reports/v70_pairs.csv): paired differences, which share most of the noise. v12-family files (v12, v17, v32, v33)
beat their predictions relative to harness files by about 800; v33 vs v35 (same budget model, harness drops ~350 borderline vehicles
with v12 p ~ 0.02 that v12 flagged at near-zero budget): predicted -885, observed +290 (1.4 sd). Files with more total budget beat
predictions (mean z -0.41), files with less fall short (+0.71). v13_bag vs v12 is a +3.4 sd outlier.

`src/v71_band.py`: hypotheses with extra failure odds for vehicles in v12 rank bands 1000–4000, fitted to all 27 scores: best
log-lik -190.60 against -190.42 without. The public scores as a whole do not support the borderline effect; the pairs above look like a
few failures. Train: 1,171 groups of exactly duplicated rows (2,345 rows) fail at 0.17% against 0.73%; 338 test rows are in duplicate
groups. No id or row-order signal.

## v72–v75: new model families, CatBoost rankings in the harness, hedged decision (27 Nov)

`src/v72_newmodels.py` (v67 setups; cost change vs v12 classifier, %): CatBoost, LightGBM and sklearn HistGB alone all lose in most
setups (HistGB +9 to +55%); missing-count and duplicate-row features do not help. CatBoost without s120/s122 averaged with the survival
ensemble (catNO+aft) beats v12 in all seven: plain -4.3, wear -4.8, complete -2.4, instrument -6.5, cluster -10.6, ho-wear -9.3,
ho-usage -4.8 (survival alone: -3.2, -3.3, -2.4, -5.4, -4.2, -11.8, -5.1).

`src/v73_catboost.py`, locked hold-out (79 failures): AP v12-style 0.569, catNO 0.525, survival 0.578, catNO+aft 0.557. Not confirmed.
On test, catNO ranks very differently from v12 (Spearman 0.58).

`src/v74_harness_cat.py`: public log-lik of CatBoost-led truths -190.65 (catNO+aft) to -191.38 (catNO alone), against -190.42 best
and -191.10 for v12 alone. The public scores cannot separate rankings that disagree this much on test.

`src/v75_hedge.py`: Bayes decision over v66 and CatBoost truths (`submission_v75_hedge.csv`, 1,440 flags, 31 different from v51) and
with equal weight per ranking family (`v75_eqfam`). Expected public-scale change against v51 under each family: v66 -7 to +21 (-34 under
survival), v69 +38 to +326 (-5 under CatBoost), v75_hedge -41 to +9, v75_eqfam -58 to +7. All candidates are within tens of points.

## v76–v77: budget scale and repair-cost level (27 Nov)

`src/v76_budscale.py`: if repair costs run x1.27 above the model with failure counts unchanged, the model's expected public score for
v75_hedge becomes 16,263 (close to actual) and budgets x1.2–1.3 save about 400; at modelled costs x1.2 loses 198.
`src/v77_alpha.py`: extending the cost-level grid to 2.2 with the failure level refit, the public scores prefer low cost levels: best
log-lik 1.0 -190.42, 1.3 -190.85, 1.6 -192.45, 2.0 -195.55; posterior beyond 1.45 is 5%. The data explain the level with more failures,
not costlier ones, so the budget lever is not supported. No budget scaling.

## v78–v82: audit of slope, regularisation and loss (27 Nov)

- Scoring rule and labels re-checked against the problem statement: only failures carry a repair cost (max 8,000). Test vehicles are
  not near-copies of train vehicles (the zero nearest-neighbour distances are low-usage all-zero vehicles, equally common in train).
- `src/v78_slope.py`: confidence slope b extended to 0.4–1.6. Best public log-lik at 1.0–1.3 (-190.3 to -190.5); 0.7 -193.6, 0.55
  -200.0. Flatter probability curves (more failures among borderline vehicles) are rejected.
- `src/v79_regular.py` (v67 setups): 11 XGBoost regularisation variants (depth 2/3/6, slow learning, min_child_weight 10,
  colsample 0.3, L1/L2, max_delta_step, scale_pos_weight, subsampling, gamma). None beats the current settings outside the cluster
  setup, where every variant wins. A five-member regularised survival ensemble beats v12 in all seven (p >= 0.92), slightly ahead
  of the three-member one in five.
- `src/v80_costweight.py`: failures weighted by repair cost. Classifier: sqrt weighting about neutral, linear weighting worse. Survival
  model with sqrt(cost) weights beats v12 in all seven (-4.2 to -7.2%; ho-wear -5.3) and the unweighted model in five.
  `src/v81_aftcw.py`, locked hold-out: unweighted -1.8%, weighted -0.7%, both -2.1% (AP 0.574, 0.552, 0.566). Test Spearman with the
  unweighted model 0.97.
- `src/v82_harness_aftcw.py`, `v82_hedge.py`: public log-lik -190.63 to -190.88 (best overall -190.42). Hedged decisions over all
  families (`submission_v82_hedge.csv`, `v82_eqfam`) differ from v75 by a few points; every candidate is within -59 to +21 of v51
  under every ranking family.

## v83–v84: v51's cost by component, and the repair-cost model against other learners (27 Nov)

`src/v83_v51loss.py`, expected on the public 30% under the full posterior (actual 16,388):

| component | expected | share |
|---|---|---|
| inspections (430 flags) | 4,299 | 31.5% |
| unused budget on flagged vehicles that don't fail (~260) | 2,083 | 15.3% |
| shortfall on flagged failures | 3,310 | 24.3% |
| unused budget on flagged failures | 1,685 | 12.3% |
| missed failures (~19 of ~189) | 2,269 | 16.6% |
| total | 13,646 | |

Oracles: knowing which vehicles fail 7,338; knowing each vehicle's repair cost if it fails (v51 flags) 9,681; both 1,889. (The
earlier "~650 from knowing costs" referred to a narrower comparison.) Repair-cost uncertainty is worth about 4,000 on the public 30%.

`src/v84_costens.py`, OOF on the 383 eligible training failures: v51 GP R2 0.838, budget loss 23.8; ridge 0.644 (+68%), SVR 0.642
(+79%), kNN 0.473 (+113%), regularised XGBoost 0.716 (+56%). Every blend with the GP is worse (best: 75% GP + 25% XGBoost, +16%).

## v85–v87: untested GP repair-cost refinements (27 Nov)

`src/v85_gpext.py` (v54 budget loss, seeds 501–508): lengthscale multipliers fitted per input group +4.8% (worse); linear mean on the
wear and sensor z-scores -0.4%; budgets from each vehicle's GP predictive sd within the three cost bands -3.4% (7/8), with the linear
mean -4.7% (8/8). v54's loss leaves out the unused budget on flagged vehicles that do not fail. `src/v86_gpext_full.py` adds it:
linear mean + predictive sd -2.4% (8/8, bootstrap p 0.78), predictive sd alone -1.5% (7/8), linear mean alone -0.3%.

`src/v87_gplin_pvar.py`: refit on all failures. OOF R2 0.845 (v51 0.839), but test predictions correlate only 0.86 with v51's: the
linear wear terms extrapolate into the test depot's higher wear. The predictive sd does not track error size (corr(|resid|, sd) on
OOF -0.02) and is larger on test (median 1.00 against 0.87), so the CV gain has no clear mechanism and the test-side widening is
unvalidated. `submission_v87_gppvar.csv` (1,422 flags, 53 different from v51) is expected 254 below v51 only under its own cost model.
Not recommended.
