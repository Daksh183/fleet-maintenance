# Fleet maintenance V33 | Implementation and trial decision

**Trial file** — `submission_v33_gp_budget.csv`. This candidate has been built and validated. V17's recorded public cost is 16,826. The team reported 16,991.5 for v32. Reaching 16,000 requires saving 826 from v17, or 4.91%. The team reported a public improvement of about 200 points. The exact public score and the private score are unspecified.

## What changes

Keep the team's locally rebuilt inspection policy. Change the parts budget for supported vehicles it flags with a positive budget. V32 combined a tree and a fixed histogram kernel and allowed the combined estimate to change inspections. V33 gives a supervised Gaussian process full responsibility for these budgets. It learns a separate weight for each of six sensor histograms and for exposure. This lets failures with different operating patterns have different repair estimates.

The test file contains 20,000 official IDs in sample order and 1,780 inspection flags. V33 changes 1,076 charged budgets versus its matched local control, with zero flags changed. The local control itself adds 39 inspections and drops 20 relative to the archived 16,826 submission. A leaderboard comparison with that archived file therefore also includes this known baseline difference.

## Build recipe

1. Use only the official competition ZIP and the team's own intermediate model outputs derived from it. The baseline failure model is the seed-42 v2 XGBoost classifier with Platt calibration. Use its calibrated source probability q and the existing fixed odds adjustment p = 3q/(1 − q + 3q). Run the original v12 repair-cost tree and cost rule to get the local control flags and budgets. The enclosed cache holds IDs, q, tree predictions and the tree's out-of-fold log residuals. Public scores and the old leaderboard-fitted odds mixture are excluded from candidate fitting and inference.
2. Fit the new estimator on the 397 official training failures, with natural log repair cost as the response. Use the ten-bin histograms starting at s023, s037, s080, s129 and s154, plus the first nine aligned bins starting at s054. On raw test data, shift s055–s063 into s054–s062 exactly once. For each complete nonnegative histogram with positive total, concatenate square-root bin shares and cumulative shares excluding the last bin. Add log(1 + the s080 histogram total) as a seventh input group.
3. For each group use a Gaussian similarity exp(−d²/(2s)), where d² is squared Euclidean distance and s is the median finite squared distance between distinct training-row pairs, floored at 0.000001. A missing group contributes zero similarity. Sum the seven kernels with positive learned amplitudes and add diagonal noise. Center log costs at their training mean. Fit amplitudes and noise by penalized marginal likelihood with the bounds and optimizer in `src/v33_hist_gp.py`. Predict on rows with at least three valid histograms. Clip predicted log cost to the range observed in that fit.
4. Obtain honest budget residuals with shuffled five-fold splitting of the 397 failures in original file order, seed 42. Refit all kernel parameters inside each fold. Subtract each supported held-out prediction from its observed log cost. The final fit has 383 usable residuals. Refit the model on all failures. For supported, inspected vehicles with positive baseline budgets, use exp(predicted log cost + residual quantile at 1 − 1/(11p)). Multiply by 1.25 and round to cents. Keep baseline values elsewhere. The cost threshold comes from the competition's 0.20 shortage and 0.02 excess charges.
5. Verify exactly 20,000 unique official IDs, binary flags, finite nonnegative budgets, the unchanged local flags, and identical budget values on fallback rows. The build script reopens the CSV and checks it against the model output. It also checks the official four-vehicle scoring example.

## Evidence for the choice

On 44,800 development vehicles, three repeats of five outer folds each compared policies on the same vehicles. Inner folds produced the calibration and residual inputs. Ordinary source-depot cost fell **3.40%** on average. Under triple failure odds, target wear proportions and three simulated acquisition shifts, mean cost fell **6.19%**. Savings on the three repeats were 5.31%, 6.45% and 6.81%. Both shortage and excess charges fell on average. Inspection and missed-repair costs were identical by construction. The repeats reuse vehicles.

The research also tested failure-risk corrections, a discriminant model, an undersampled forest blend, risk-band budget multipliers and an elastic-net histogram model. A half-weight Gaussian-process budget blend saved about 3.3% under the same simulated shifts. The full-weight, budget-only candidate was frozen before its supporting holdout check.

On the historical **11,200-row holdout**, ordinary total cost fell from **8,623.18 to 8,523.47**, saving 99.71 or **1.16%**. Simulated depot cost fell **1.75%**, with gains in all three acquisition scenarios. The most adverse single-vehicle deletion erased the ordinary holdout gain and left a small simulated-depot gain. A second check standardizing by wear and histogram usage gave a **1.93%** saving, with few observations in some groups. This holdout has been used in earlier project versions. The new depot's labels remain hidden.

**Verdict:** supported for a Kaggle trial. The separate holdout showed a smaller gain than the 4.91% needed for 16,000. Use the actual public result to record how the trial performed. Neither percentage predicts the private ranking.

## Rebuild from the repository root

Install requirements.txt and supply the official competition ZIP.

```powershell
python src/v33_build.py --zip "C:/path/to/mlprojectfleet2.zip" --base reports/v33_baseline.npz --runtime . --out run/v33
```

The research used primary sources for methods and competition files for data. The learned sum of kernels follows the additive Gaussian-process approach. The held-out comparison addresses the risk of selecting the best-looking model after trying several alternatives. [Duvenaud, Nickisch and Rasmussen, Additive Gaussian Processes](https://papers.neurips.cc/paper_files/paper/2011/hash/4c5bde74a8f110656874902f07378009-Abstract.html) · [Cawley and Talbot, On Over-fitting in Model Selection](https://jmlr.org/papers/v11/cawley10a.html)
