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
| `v9_cost_model_capacity.ipynb` | v9 cost-model capacity, k-averaging | #7 `sub07-2026-11-21-v9_k1`, #9 `sub09-2026-11-21-v9_em` (`v9_k4` not uploaded yet) | 19,600.79 / 18,058.56 |
| `v10_shift_posterior.ipynb` | v10 posterior over the shift k from all public scores; posterior-averaged decision | ready to upload (`v10_bayes`, `v10_k2`) | pending |

`build_*.py` + `nbkit.py` generate the notebooks; `run_all.sh` executes them; `verify_outputs.py` checks that each notebook regenerates exactly the CSV in `../submissions/`.
