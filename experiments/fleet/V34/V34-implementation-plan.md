# V34 implementation plan | Adjust parts budgets on the vehicles V33 already inspects

**Build this as an experiment. Keep the exact V33 submission as the benchmark.** Your reported leaderboard improvement makes V33 the relevant comparison. Its exact new public score is unspecified. The proposed policy reduced simulated depot cost by 1.95% versus V33 in repeated development validation. A high-usage extrapolation check reversed the gain. The evidence supports building and evaluating the model, with V33 still the better-supported final choice.

## What the experiments established

I scored 23 policy variants against the same V33 predictions. The comparisons used 44,800 development vehicles, five outer folds per repeat, three repeat seeds, and inner out-of-fold predictions for calibration. Reconstructed V33 costs matched the saved V33 experiment in all 150 fold/scenario/odds comparisons. Every comparison used the official inspection, missed-repair, shortage and excess charges.

Two effects were worth pursuing. First, V33 already knows when it has seen similar failures, through the uncertainty of its Gaussian-process repair estimate. Its submitted budget used the same distribution of log-cost errors for every supported vehicle. Scaling that error distribution to the individual vehicle saved **1.54%** in the target-depot stress test by itself. Second, a small recalibration using wear, usage and two sensor readings added information about failure probability. Since the budget depends on that probability, this can change the amount ordered even with the same inspection decision.

Several plausible changes did worse on the same simulated depot comparison. Learning a separate length scale for each histogram increased cost by 1.39%. Adding pairwise histogram interactions increased it by 1.15%. Giving extra weight to rare histogram bins through log-ratio kernels increased it by 1.71%. Removing the 25% budget multiplier increased it by 1.65%. These results narrow the implementation to the two changes below.

A candidate using both changes for inspection and budget saved 2.48% in development. I froze it, then checked it on the historical 11,200-row holdout. It cost **51.08 more** under ordinary holdout scoring and **1.19% more** under simulated depot conditions. In all three simulated scenarios it skipped a vehicle with a repair bill of 185 that V33 inspected. This is the reason to preserve the submitted V33 flags in the proposed build.

The budget-only policy was already one of the development comparisons. Its mean ordinary cost fell from **0.82879 to 0.82463** per vehicle. Its mean simulated depot cost fell from **1.98065 to 1.94212**. The three repeat savings were **1.96%, 1.65% and 2.24%**. All three survived removing the single vehicle whose deletion was least favorable. The partial-scale scenario explains the mechanism: shortage costs fell by 0.0435 per weighted vehicle while excess costs rose by 0.0062. Inspection and missed-repair costs were identical. The simulated depot uses three times the ordinary failure weight, target wear proportions, and three plausible sensor acquisition shifts.

I checked the budget-only policy separately after the full-policy holdout reversal. Two refits reserved the entire top decile of one sensor measurement plus a random 10% of the remaining development vehicles. On the **high-wear split**, simulated cost improved **0.14%**. On the **high-usage split**, it worsened **0.42%**. Ordinary cost on the latter worsened **2.62%**. These are severe extrapolation checks: they left just 22 and 30 failures, respectively, for training. The two evaluation populations overlapped by 4,446 vehicles. The fixed-flag design is exploratory, and these reused source-depot vehicles cannot establish its private score.

A third diagnostic more closely approximated deployment. Using only official test sensor readings, I measured the new depot's proportions in nine wear-by-usage groups. I drew 8,960 labeled development vehicles to match those proportions, without using their outcomes to choose them, and retrained on the remainder. The reserved set contained 99 failures and the training portion retained 219. V34 saved **0.41%** on ordinary cost and **0.76%** on the wear-weighted simulated depot cost. Removing the most favorable single vehicle turned the latter into a **0.21% increase**. This is a better approximation of the observed sensor mix than an entire missing tail, but its single-vehicle sensitivity is a good reason to treat V34 as an experimental trial.

## Build the model

1. **Anchor the output to the submitted file.** Start with `V33/submission_v33_gp_budget.csv`. Verify its SHA-256 is `eed2288b3309d637fab1eae2975ceee67a07d6bfda5c39f1d24c9ebabe981583`. Join on official IDs and copy its 1,780 flags. The V34 directory contains the implementation. It reuses `V33/model/build_v33.py`, `v33_hist_gp.py`, `baseline.npz`, the bundled `fleet_runtime`, and the official competition ZIP. The cache supplies source-calibrated test probabilities, original tree log-cost predictions, and tree residuals. The builder has a baseline switch that reproduces the archived V33 output to the cent when both changes are disabled.

2. **Fit a small correction to the failure odds.** Recreate the original 44,800-row development ordering with `common.split_holdout(train)`. Read the team's `fleet_runtime/reports/p_oof_v2.npy` in that order. Fit the original Platt calibrator on those raw out-of-fold predictions and recorded failures, then obtain a source probability q for each development row. The correction uses six sensor-derived inputs:

   - log(1 + s126), measuring the wear counter;
   - log(1 + sum of s080 through s089), requiring all ten usage bins;
   - the difference between the two log readings;
   - log(1 + s123);
   - log(1 + s149);
   - the fraction of missing readings across s000–s169, excluding s047, s063, s120, s122 and s146.

   On the fitting rows only, calculate each input's 1st percentile, median and 99th percentile. Median-impute, clip to the percentile bounds, then center and scale using the mean and population standard deviation of those transformed fitting values. Floor each standard deviation at 0.001. Append a constant and two raw missingness indicators for wear and usage. Use the identical column order at prediction time. Apply the official test histogram alignment once, as in V33.

   Fit the nine coefficients with logistic regression using logit(q) as a fixed offset. Optimize the **sum** of binary log losses plus `8 * sum(coefficients**2)`, including the intercept and indicator coefficients, with SciPy L-BFGS-B starting at zero. Clip q to [0.000001, 0.999999] when computing its logit. Apply the fitted correction to each production test row as `eta = clip(X @ coef, -1.5, 1.5)`. For budgeting, start with V33's depot-adjusted probability `p = 3*q/(1-q+3*q)` and use `p_budget = expit(logit(clip(p,1e-9,1-1e-9)) + eta)`. Use this probability only in the budget calculation. Fit every transform and coefficient inside the training portion when validating.

3. **Return a predictive standard deviation from the existing GP.** Retain its seven input groups, learned amplitudes, noise, length scales, clipped mean prediction, and requirement for at least three valid histograms out of six. Store the Cholesky factor already computed for the final fitted covariance `K_train + noise*I`, and store the fitted **noise variance**. For a row x, let k be its weighted kernel vector against the training failures. Its latent self-covariance kxx is the sum of the learned amplitudes for input groups observed on x. Use the GP's exact zero-contribution convention for a missing group.

   ```python
   solved = scipy.linalg.cho_solve(factor, k)
   variance = max(noise, noise + kxx - k @ solved)
   sigma = np.sqrt(variance)
   ```

   These are log-repair-cost units. Work in batches and reuse the factor. During production fitting, use the same five shuffled failure-only folds as V33, seed 42, on the 397 failures in official file order. On each supported held-out failure save the **signed** standardized residual `(log(actual_cost) - gp_mean) / sigma`. Pool these values and fit the final GP on all failures. The support mask should still give 383 eligible out-of-fold failures. An uncertainty estimate fitted on the vehicle's own repair outcome would make the budget quantiles too optimistic.

4. **Write new parts budgets.** For every V33-inspected vehicle, including ones whose old budget was zero, compute the budget using the corrected probability. For supported rows with `p_budget > 1/11`, let `level = 1 - 1/(11*p_budget)`. The raw budget is `exp(gp_mean + sigma * np.quantile(standardized_residuals, level))`. Otherwise the raw budget is zero when the probability is at or below the threshold. For an unsupported inspected row, use the original tree mean and tree residual quantile at the same corrected probability. Multiply the resulting inspected budget by **1.25**, then round once to two decimals. Copy the archived value for uninspected rows. Training/evaluation and final inference use only competition records and the team's own models. The scientific references supply methodology.

5. **Verify before exporting `submission_v34_budget.csv`.** Match all 20,000 unique official IDs in sample order, require finite nonnegative budgets, and compare flags by ID to the submitted V33. The expected flag difference is **zero**. Save counts of increased, reduced, zero-to-positive and positive-to-zero inspected budgets, total parts allocation, and a SHA-256 hash. Check the scorer gives 1,241 on the assignment example and verify that inspection and missed-repair charges match V33 exactly on labeled validation rows. Keep the upload decision separate from building the file.

## Decision and sources

Use the fold comparisons to verify the implementation. A public improvement of another 200 points or a score of 16,000 is unverified. For a final entry, V33 has actual leaderboard evidence. An extra trial of this budget-only candidate would answer a separate question about whether the improved development budgeting transfers to the real depot. Comparing models on the reused holdout after changing the design adds selection bias.

The predictive-variance formula is standard Gaussian-process regression. A logistic offset supplies the existing failure log-odds with coefficient one. The regularization, sensor inputs, empirical residual quantile, and decision to preserve inspection flags are project choices evaluated on the team's data. [Rasmussen and Williams, Gaussian Processes for Machine Learning, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf) · [Statsmodels GLM offset documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html) · [Cawley and Talbot on selection bias](https://jmlr.org/papers/v11/cawley10a.html)

## Rebuild the submitted candidate

Run from the V34 directory with the project Python environment. Supply the official ZIP path for your machine.

```powershell
python -B .\model\build_v34.py --zip "C:/path/to/mlprojectfleet2.zip" --v33 ..\V33 --runtime ..\fleet_runtime --out .\rebuilt
```

Use `--baseline` to verify the archived V33 file.
