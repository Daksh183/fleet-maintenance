# Kaggle submission log

Every Kaggle submission has an annotated git tag `subNN-YYYY-MM-DD-<version>` (unique number + submission date), created with `tools/tag_submission.sh`. Public scores are total cost on the public 30% of test (lower is better). **Only the last submission before the 27 Nov deadline is evaluated**, so the final upload must be the chosen model.

Submissions 22 and 23 were uploaded by Daksh183; their dates are taken from the commits and should be confirmed on Kaggle's submissions page. Submissions 18–27 were logged and their notebooks for 18–21 and 24–27 written on 27 Nov, after upload; those tags are to be created at the commit that adds the notebooks.

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
| 13 | 2026-11-23 | `sub13-2026-11-23-v13_bag` | `submissions/submission_v13_bag.csv` | `notebooks/v13_seed_averaging.ipynb` | v13: v12_robust recipe with the classifier (5 seeds) and cost model (8 seeds) seed-averaged | 17,974.55 |
| 14 | 2026-11-23 | `sub14-2026-11-23-v14_map` | `submissions/submission_v14_map.csv` | `notebooks/v14_quantile_map.ipynb` | v14: v13_bag with test s120/s122 quantile-mapped for the classifier, flag count matched | 17,375.65 |
| 15 | 2026-11-23 | `sub15-2026-11-23-v15_map_v12` | `submissions/submission_v15_map_v12.csv` | `notebooks/v15_quantile_map_v12_base.ipynb` | v15: v12_robust with the same s120/s122 mapping (replication of v14), flag count matched | 17,058.28 |
| 16 | 2026-11-24 | `sub16-2026-11-24-v17_bud125` | `submissions/submission_v17_bud125.csv` | `notebooks/v17_level_probes.ipynb` | v17: v12_robust with every flagged budget x1.25 (level probe) | 16,826 |
| 17 | 2026-11-24 | `sub17-2026-11-24-v17_k135` | `submissions/submission_v17_k135.csv` | `notebooks/v17_level_probes.ipynb` | v17: v12_robust with the shift factor k scaled x1.35, 1,880 flags (level probe) | 16,908 |
| 18 | 2026-11-24 | `sub18-2026-11-24-v18_ens10` | `submissions/submission_v18_ens10.csv` | `notebooks/v18_censoring_ensemble.ipynb` | censoring-weighted ensemble of ten XGBoost classifiers on the v12 base, flag count matched | 17,624 |
| 19 | 2026-11-25 | `sub19-2026-11-25-v22_nos122` | `submissions/submission_v22_nos122.csv` | `notebooks/v22_drop_depot_sensors.ipynb` | classifier without the mis-scaled sensors s120 and s122, flag count matched | 17,172.57 |
| 20 | 2026-11-25 | `sub20-2026-11-25-v24_pu` | `submissions/submission_v24_pu.csv` | `notebooks/v24_soft_labels.ipynb` | soft labels for censored healthy vehicles, flag count matched | 17,575 |
| 21 | 2026-11-25 | `sub21-2026-11-25-v29_bayes` | `submissions/submission_v29_bayes.csv` | `notebooks/v29_harness_bayes.ipynb` | Bayes flags and budgets under a posterior over test-depot hypotheses fitted to the public scores (tree repair costs) | 17,196 |
| 22 | 2026-11-26 | `sub22-2026-11-26-v32_hist_kernel` | `submissions/submission_v32_hist_kernel.csv` | `notebooks/v32_hist_kernel.ipynb` | histogram-kernel repair-cost blend | 16,991.5 |
| 23 | 2026-11-26 | `sub23-2026-11-26-v33_gp_budget` | `submissions/submission_v33_gp_budget.csv` | `notebooks/v33_gp_budget.ipynb` | Gaussian-process (sensor histogram) repair budgets | 16,587.75 |
| 24 | 2026-11-27 | `sub24-2026-11-27-v35_combo` | `submissions/submission_v35_combo.csv` | `notebooks/v35_gp_budget_combo.ipynb` | harness flags with Gaussian-process budgets | 16,878 |
| 25 | 2026-11-27 | `sub25-2026-11-27-v37_bayes_gp2` | `submissions/submission_v37_bayes_gp2.csv` | `notebooks/v37_gp_sensor_costs.ipynb` | Gaussian-process costs with wear and sensor groups, used for both flags and budgets | 16,410 |
| 26 | 2026-11-27 | `sub26-2026-11-27-v42_gp_bands` | `submissions/submission_v42_gp_bands.csv` | `notebooks/v42_gp_banded_spread.ipynb` | re-selected sensor group and cost spread by predicted-cost band | 16,487 |
| 27 | 2026-11-27 | `sub27-2026-11-27-v51_gp_reduced` | `submissions/submission_v51_gp_reduced.csv` | `notebooks/v51_gp_reduced_inputs.ipynb` | repair-cost model without the exposure group and the s023 and s129 histograms, averaged over with/without s123 | 16,388 |
| 28 | 2026-11-27 | `sub28-2026-11-27-v82_hedge` | `submissions/submission_v82_hedge.csv` | `notebooks/v82_hedge.ipynb` | Bayes decision over every ranking family (classifiers, survival, CatBoost, cost-weighted survival), weighted by the public scores | pending |
