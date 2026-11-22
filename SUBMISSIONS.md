# Kaggle submission log

Every Kaggle submission has an annotated git tag `subNN-YYYY-MM-DD-<version>` (unique number + submission date), created with `tools/tag_submission.sh`. Public scores are total cost on the public 30% of test (lower is better). **Only the last submission before the 27 Nov deadline is evaluated**, so the final upload must be the chosen model.

Submissions 1–6 were uploaded on the dates shown, before their notebooks existed. Each tag was created on 20 Nov 2026, after the fact, at the commit that adds the per-version notebooks, and points to the notebook that regenerates that exact file (verified: identical flags and budgets, see `notebooks/verify_outputs.py`).

| # | Date | Tag | File | Notebook | What changed | Public score |
|---|---|---|---|---|---|---|
| 01 | 2026-11-19 | `sub01-2026-11-19-v2` | `submissions/submission_v2.csv` | `notebooks/v2_naive_xgboost.ipynb` | XGBoost, flag if p > 0.5 | 74,933.17 |
| 02 | 2026-11-19 | `sub02-2026-11-19-v3` | `submissions/submission_v3.csv` | `notebooks/v3_cost_rule.ipynb` | cost-based flag + newsvendor budget (= Model 1 of the final report) | 27,760.14 |
| 03 | 2026-11-19 | `sub03-2026-11-19-v3b` | `submissions/submission_v3b.csv` | `notebooks/v3b_drop_drifted_sensors.ipynb` | v3 without the four drifting sensors; reverted | 29,529.75 |
| 04 | 2026-11-20 | `sub04-2026-11-20-v4_platt` | `submissions/submission_v4_platt.csv` | `notebooks/v4_calibration.ipynb` | + Platt calibration | 23,875.67 |
| 05 | 2026-11-20 | `sub05-2026-11-20-v5_raw` | `submissions/submission_v5_raw.csv` | `notebooks/v5_per_vehicle_cost.ipynb` | per-vehicle repair-cost model, raw p | 26,060.57 |
| 06 | 2026-11-20 | `sub06-2026-11-20-v5_platt` | `submissions/submission_v5_platt.csv` | `notebooks/v5_per_vehicle_cost.ipynb` | per-vehicle cost model + Platt (= Model 2 of the final report) | 22,968.55 |
| 07 | 2026-11-21 | `sub07-2026-11-21-v9_k1` | `submissions/submission_v9_k1.csv` | `notebooks/v9_cost_model_capacity.ipynb` | v9 cost model (1000 trees d2) + shape features, no shift (k=1): isolates the new cost model | 19,600.79 |
| 08 | 2026-11-21 | `sub08-2026-11-21-v7_k4` | `submissions/submission_v7_k4.csv` | `notebooks/v7_depot_shift.ipynb` | v5 cost model + prior shift k=4 (uploaded by mistake instead of v9_k4): isolates the shift | 19,067.17 |
| 09 | 2026-11-21 | `sub09-2026-11-21-v9_em` | `submissions/submission_v9_em.csv` | `notebooks/v9_cost_model_capacity.ipynb` | v9 cost model + prior shift k=6.53 (EM estimate) | 18,058.56 |
| 10 | 2026-11-22 | `sub10-2026-11-22-v9_k4` | `submissions/submission_v9_k4.csv` | `notebooks/v9_cost_model_capacity.ipynb` | v9 cost model + prior shift k=4 (the comparison that was meant for 21 Nov) | 17,013.86 |
| 11 | 2026-11-22 | `sub11-2026-11-22-v10_bayes` | `submissions/submission_v10_bayes.csv` | `notebooks/v10_shift_posterior.ipynb` | v10: v9 cost model + shift averaged over the posterior of k (behaves like k~3) | 17,348.75 |
| 12 | 2026-11-22 | `sub12-2026-11-22-v12_robust` | `submissions/submission_v12_robust.csv` | `notebooks/v12_robust_cost_inputs.ipynb` | v12: v10_bayes with the repair-cost model excluding the four drifted sensors (single change) | 16,839.86 |
