# Fleet Maintenance Planning (Kaggle `mlprojectfleet2`)

ML course project. For 20,000 vehicles from a new depot, decide whether to **flag** each one for inspection and what spare-parts **budget** to set, minimising total cost:

- not flagged: cost = repair cost if it fails, else 0
- flagged: 10 + 0.20·max(c−b, 0) + 0.02·max(b−c, 0)

Course rules: classical ML only (XGBoost / LightGBM / sklearn), no deep learning, AutoML, pretrained models, LLM predictions or external data.

## Pipeline (current best)

1. **Failure probability:** XGBoost classifier (300 trees, depth 4, lr 0.05) on all sensors, after the v1 data fixes (s054–s063 histogram realigned in test; s047, s146 dropped). `followup_days` is never a feature.
2. **Calibration:** Platt scaling fitted on out-of-fold predictions.
3. **Depot-shift correction:** the test depot fails ~4–6× more often. Probabilities are shifted by an odds factor k (EM estimate 6.5; public-score fit and minimax regret point to ~4).
4. **Repair-cost model:** XGBoost on log(repair cost), failures only, using sensors + histogram-shape features (six hidden 10-bin histograms). Per-vehicle cost distribution = prediction + out-of-fold residuals.
5. **Decision:** budget = newsvendor quantile (critical ratio 0.20/0.22 = 0.909, adjusted for p); flag when the expected cost of flagging is below p × expected repair cost (≈ p > 2.5%).

## Results (CV cost per vehicle, 80% dev split; lower is better)

| Version | Change | CV cost | Public |
|---|---|---|---|
| v0 | flag nobody | 3.487 | n/a |
| v2 | XGBoost, flag if p > 0.5 | 2.501 | 74,933 |
| v3 | cost-based flag + newsvendor budget | 1.242 | 27,760 |
| v4 | + Platt calibration | 1.107 | 23,876 |
| v5 | + per-vehicle cost model | 0.904 | 22,969 |
| v6 | follow-up handling, missingness features | no gain (reverted) | n/a |
| v7 | prior-shift correction for the new depot | n/a (test-only) | pending |
| v8 | histogram-shape features in the cost model (R² 0.66 → 0.79) | 0.866 (3×5 CV) | pending |

Full per-version findings, caveats and rejected experiments: [`reports/findings.md`](reports/findings.md). Every run: [`reports/experiments.csv`](reports/experiments.csv).

## Course submission requirements

- **One notebook per version** (including every uploaded submission): see [`notebooks/README.md`](notebooks/README.md).
- **Report notebook:** [`notebooks/fleet_maintenance.ipynb`](notebooks/fleet_maintenance.ipynb) is the running codebase for the three selected models (it writes `notebooks/outputs/submission_model{1,2,3}.csv`) with the analysis in markdown cells. Rebuild it from `notebooks/build_notebook.py`.
- **Every Kaggle submission is tagged:** run `tools/tag_submission.sh <csv> <notebook> "<what changed>"` right after uploading; the log is [`SUBMISSIONS.md`](SUBMISSIONS.md).
- **Individual contributions:** each member commits their own work from their own account; summary in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
- **Only the last submission before 27 Nov counts**, so the final upload must be the chosen model.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Download `train.csv`, `test.csv`, `sample_submission.csv` from the Kaggle competition into `data/`. Scripts run from `src/`:

```bash
cd src && ../.venv/bin/python v2_v3.py     # base classifier (writes reports/p_oof_v2.npy, p_te_v2.npy)
../.venv/bin/python v8_build.py            # current submissions -> submissions/submission_v8_*.csv
```

## Layout

- `src/score.py`: exact competition cost. `src/common.py`: data loading + fixes, locked 20% hold-out, fixed 5 folds, submission writer, experiment logger.
- `src/decide.py`: flag/budget decision rules. `src/calib.py`: Platt / isotonic. `src/features.py`: histogram-shape features.
- `src/vN_*.py`: one script per version/experiment.
- `notebooks/`: report notebook + its generator. `tools/`: submission tagging.
- `submissions/`: ready-to-upload files (`rejected/` holds files built on rejected models).

## Working rules

- Judge every change on **paired fold differences in cost**, not AUC or log-loss (several AUC gains made cost worse).
- The 20% hold-out is locked (used only at v0, v4, final).
- Choose final submissions by CV + hold-out + evidence agreement, not by the public leaderboard alone (30% of test, few failures).
