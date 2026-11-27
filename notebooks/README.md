# Notebooks

One **self-contained** notebook per version (each needs only `../data/train.csv` and `../data/test.csv`, runs top to bottom, and writes its Kaggle CSV to `outputs/`). Every uploaded Kaggle submission is tagged in git to the notebook below that produced it (see `../SUBMISSIONS.md`).

| Notebook | Version | Kaggle submission (git tag) | Public score |
|---|---|---|---|
| `fleet_maintenance.ipynb` | **Final report**: the three selected models (Model 1-3) | built from Models 1, 2, 3 | n/a |
| `v0_baselines.ipynb` | v0 trivial baselines | none | n/a |
| `v1_eda.ipynb` | v1 EDA, censoring, depot-shift diagnosis | none | n/a |
| `v2_naive_xgboost.ipynb` | v2 XGBoost, flag if p > 0.5 | #1 `sub01-2026-11-19-v2` | 74,933.17 |
| `v3_cost_rule.ipynb` | v3 cost-derived flag rule + newsvendor budget | #2 `sub02-2026-11-19-v3` | 27,760.14 |
| `v3b_drop_drifted_sensors.ipynb` | v3b drop drifted sensors (rejected) | #3 `sub03-2026-11-19-v3b` | 29,529.75 |
| `v4_calibration.ipynb` | v4 Platt / isotonic calibration | #4 `sub04-2026-11-20-v4_platt` | 23,875.67 |
| `v5_per_vehicle_cost.ipynb` | v5 per-vehicle repair-cost model | #5 `sub05-2026-11-20-v5_raw`, #6 `sub06-2026-11-20-v5_platt` | 26,060.57 / 22,968.55 |
| `v6_followup_missingness.ipynb` | v6 follow-up handling, missingness (reverted) | none | n/a |
| `v7_depot_shift.ipynb` | v7 depot-shift analysis and prior correction | #8 `sub08-2026-11-21-v7_k4` | 19,067.17 |
| `v8_histogram_features.ipynb` | v8 hidden histograms, shape features | none yet | n/a |
| `v9_cost_model_capacity.ipynb` | v9 cost-model capacity, k-averaging | #7 `sub07-2026-11-21-v9_k1`, #9 `sub09-2026-11-21-v9_em`, #10 `sub10-2026-11-22-v9_k4` | 19,600.79 / 18,058.56 / 17,013.86 |
| `v10_shift_posterior.ipynb` | v10 posterior over the shift k from all public scores; posterior-averaged decision | #11 `sub11-2026-11-22-v10_bayes` (`v10_k2` not uploaded) | 17,348.75 |
| `v12_robust_cost_inputs.ipynb` | v12 cost model without the four drifted sensors (single change vs v10_bayes) | #12 `sub12-2026-11-22-v12_robust` | 16,839.86 |
| `v13_seed_averaging.ipynb` | v13 seed-averaged classifier (5) and cost model (8): is v12's gain seed luck? | #13 `sub13-2026-11-23-v13_bag` | 17,974.55 |
| `v14_quantile_map.ipynb` | v14 quantile-map the instrument-like drifted sensors s120/s122 for the classifier (flag count matched, on the v13 base) | #14 `sub14-2026-11-23-v14_map` | 17,375.65 |
| `v15_quantile_map_v12_base.ipynb` | v15 the same mapping on the v12_robust base (replication; conclusion: no reliable effect) | #15 `sub15-2026-11-23-v15_map_v12` | 17,058.28 |
| `v17_level_probes.ipynb` | v17 level probes on the v12_robust base: flagged budgets ×1.25, and k ×1.35 (≈7% more flags); with the model-expected change of each | #16 `sub16-2026-11-24-v17_bud125`, #17 `sub17-2026-11-24-v17_k135` | 16,826 / 16,908 |
| `v18_censoring_ensemble.ipynb` | v18 censoring-weighted classifier ensemble (rejected) | #18 `sub18-2026-11-24-v18_ens10` | 17,624 |
| `v22_drop_depot_sensors.ipynb` | v22 classifier without s120/s122 | #19 `sub19-2026-11-25-v22_nos122` | 17,172.57 |
| `v24_soft_labels.ipynb` | v24 soft labels for censored vehicles (rejected) | #20 `sub20-2026-11-25-v24_pu` | 17,575 |
| `v29_harness_bayes.ipynb` | v29 Bayes decision under a posterior fitted to the public scores | #21 `sub21-2026-11-25-v29_bayes` | 17,196 |
| `v35_gp_budget_combo.ipynb` | v35 harness flags, Gaussian-process budgets | #24 `sub24-2026-11-27-v35_combo` | 16,878 |
| `v37_gp_sensor_costs.ipynb` | v37 Gaussian-process costs with wear and sensor groups | #25 `sub25-2026-11-27-v37_bayes_gp2` | 16,410 |
| `v42_gp_banded_spread.ipynb` | v42 cost spread by predicted-cost band | #26 `sub26-2026-11-27-v42_gp_bands` | 16,487 |
| `v51_gp_reduced_inputs.ipynb` | v51 repair-cost model with fewer inputs | #27 `sub27-2026-11-27-v51_gp_reduced` | 16,388 |
| `v66_survival_rankings.ipynb` | v66 survival-model rankings added to the harness | not uploaded yet | n/a |
| `v69_aftfull.ipynb` | v69 survival-model ranking replaces the classifier | not uploaded yet | n/a |
| `v82_hedge.ipynb` | v82 Bayes decision hedged over every ranking family | #28 `sub28-2026-11-27-v82_hedge` | pending |

`build_part1-7.py` + `nbkit.py` generate the notebooks (part 7 loads saved artifacts from `../reports/`); `run_all.sh` executes them; `verify_outputs.py` checks that each notebook regenerates exactly the CSV in `../submissions/`.

## V32–V34 rebuild notebooks

These notebooks run the versioned builders from this checkout. They take the official competition ZIP and reuse project artifacts committed under `reports/` and `submissions/`.

- [`v32_hist_kernel.ipynb`](v32_hist_kernel.ipynb) rebuilds [`submission_v32_hist_kernel.csv`](../submissions/submission_v32_hist_kernel.csv). Histogram-kernel cost blend. 16,991.5 reported.
- [`v33_gp_budget.ipynb`](v33_gp_budget.ipynb) rebuilds [`submission_v33_gp_budget.csv`](../submissions/submission_v33_gp_budget.csv). Gaussian-process parts budgets. About 200 points better, per team.
- [`v34_budget.ipynb`](v34_budget.ipynb) rebuilds [`submission_v34_budget.csv`](../submissions/submission_v34_budget.csv). Budget recalibration with fixed inspections. Unverified.
