# Fleet Maintenance Planning (Kaggle `mlprojectfleet2`)

ML course project. For 20,000 vehicles from a new depot, decide whether to **flag** each one for inspection and what spare-parts **budget** to set, minimising total cost:

- not flagged: cost = repair cost if it fails, else 0
- flagged: 10 + 0.20·max(c−b, 0) + 0.02·max(b−c, 0)

Course rules: classical ML only (XGBoost / LightGBM / sklearn), no deep learning, AutoML, pretrained models, LLM predictions or external data.

## Pipeline (current candidates)

1. **Failure probability:** XGBoost classifier (300 trees, depth 4, lr 0.05) on all sensors, after the v1 data fixes (s054–s063 histogram realigned in test; s047, s146 dropped). `followup_days` is never a feature.
2. **Calibration:** Platt scaling fitted on out-of-fold predictions.
3. **Depot-shift correction:** the new depot fails far more often. Probabilities are shifted by an odds factor k, which is **not** a fixed guess: it is a *posterior* over k estimated from all public scores (mean k ≈ 2.9, 90% interval 2–4), and the decision averages the shifted probability over that posterior.
4. **Repair-cost model:** XGBoost (1,000 trees, depth 2) on log(repair cost), failures only, using sensors (without the four drifted ones) + histogram-shape features of six hidden 10-bin histograms. Per-vehicle cost distribution = prediction + out-of-fold residual quantiles.
5. **Decision:** budget = newsvendor quantile (critical ratio 0.20/0.22 = 0.909, adjusted for p); flag when the expected cost of flagging is below p × expected repair cost (≈ p > 2.5%).
6. **Variance reduction (v13):** classifier and cost model averaged over seeds.

## Decisions along each version

Every row is one decision we took, **why**, and what the evidence was. "Judged on" says what we used to decide: cross-validated *cost* (never AUC or log-loss alone: several AUC gains made cost worse), or the public leaderboard where only the test depot can answer. Details, tables and caveats: [`reports/findings.md`](reports/findings.md); every run: [`reports/experiments.csv`](reports/experiments.csv); notebook per version: [`notebooks/README.md`](notebooks/README.md).

| Version | Problem we saw | What we changed | Evidence | Decision | Public score (total, lower is better) |
|---|---|---|---|---|---|
| **v0** Baselines | Need reference points | Flag nobody, flag everybody, oracle | Flag nobody 3.487/vehicle, oracle 0.071 | Fixes the scale of what is achievable | n/a |
| **v1** EDA | Train and test look nothing alike (adversarial AUC 1.0) | Drop `s047` (all NaN in test) and `s146` (constant); realign the s054–s063 histogram, which is shifted by one bin in test; keep NaNs (missingness predicts failure); never use `followup_days` | Bin-shifted histogram matches train after realigning (KS ≤ 0.07) | Keep the fixes. Four sensors (s120, s122, s126, s128) still drift | n/a |
| **v2** Naive XGBoost | A starting model | XGBoost, flag if p > 0.5 | CV 2.501/vehicle, AUC 0.984 | A 0.5 threshold is wrong for this cost structure | 74,933 |
| **v3** Cost-derived rule | Threshold ignores the real costs | Derive: budget = newsvendor quantile, flag iff p·μ > 12.5 (≈ p > 2.5%) | CV 1.242 | **Largest single lever.** Keep | 27,760 |
| **v3b** Drop drifted sensors | Do the four drifted sensors hurt? | v3 without them | CV −0.03 (inside noise) but public +6% worse | **Reverted.** CV and public disagreed, no evidence it helps | 29,530 |
| **v4** Calibration | Raw p is under-confident exactly near the flag threshold | Platt (and isotonic) on out-of-fold p | Reliability plot; CV 1.107 (Platt); hold-out (79 failures) disagreed | Keep Platt (smoother than isotonic with few failures); public confirmed the direction | 23,876 |
| **v5** Per-vehicle repair cost | One global cost distribution for every vehicle | XGBoost on log(cost), failures only; budget from per-vehicle quantile with out-of-fold residuals | R² 0.68; 0.909 quantile covers 91.5% of held-out failures; 5/5 folds better; CV 0.904 | **Keep.** Replaces v3/v4 as the base | 26,061 raw / **22,969** Platt |
| **v6** Follow-up, missingness | 22% of healthy vehicles have < 365 days of follow-up (noisy labels) | NaN-count features, dropping or down-weighting short follow-up | No variant beat fold noise; the apparent gains came from one lucky fold | **Reverted.** Failure model unchanged | not uploaded |
| **v7** Depot shift | Model predicts ~3× more failures on test | Importance weights (ESS ~1%, **rejected**); prior-shift correction p' = kp/(1−p+kp), k from EM (validated on simulated shift) and from consistency with public scores | EM k = 6.5; public-score fit k ≈ 4; minimax regret k = 4; classifier tuning improved log-loss but made cost worse in 5/5 folds (**rejected**) | **Correct for the shift** (under-correcting is expensive, over-correcting is cheap). Classifier unchanged | v7_k4: 19,067 |
| **v8** Hidden histograms | Cost model underfit | Found six hidden 10-bin histograms (exact least-squares fit); added shape features | Cost model R² 0.66 → 0.79 (15/15 folds); same features **hurt** the classifier | **Keep for the cost model, reject for the classifier** | n/a (superseded by v9) |
| **v9** Cost-model capacity, k averaging | Cost model over-shrinks expensive repairs (the costly direction) | 1,000 trees depth 2; per-vehicle spread (rejected); decide with p averaged over a belief about k | Budget loss −3.5%; spread signal too weak (coverage fell below target) | **Keep 1,000 trees depth 2**, reject per-vehicle spread. Averaging over k is the principled way to set k | k=1 19,601 · EM 18,059 · **k=4 17,014** |
| **v10** Posterior over k | Which k does the data support? | Joint posterior over k from all public scores, simulating each file's public score | Posterior mean ≈ 3.0 (90% 2–4.5); EM's 6.5 has ~0 mass (label-shift assumption biased by covariate shift) | **Bayes decision** = average p over the posterior | 17,349 |
| **v11** Ideas from the structure | Loss is dominated by expensive failures | Cost-weighted classifier, direct expected-cost (Tweedie) model, blend, quantile-regression budgets | All worse than baseline in paired CV (e.g. Tweedie +0.30, weighted +0.011 to +0.025) | **All rejected.** The simple pipeline stands | not uploaded |
| **v12** Cost-model inputs | Cost model might extrapolate on drifted sensors | Same as v10 but cost model excludes s120/s122/s126/s128; cost decomposition and model families checked | Dropping them is free in CV (R² 0.801 vs 0.802) but moves test predictions ~12%; boosted depth-2 trees beat linear/additive/other families | **Keep** as free insurance. Linear models blow up on outliers | **16,840** (best so far) |
| **v13** Seed averaging | v12's gain may be a lucky random draw (single-fit test predictions differ ≈ 0.13 in mean absolute log-cost between seeds, more than the sensor effect ≈ 0.11) | Average classifier over 5 seeds and cost model over 8 seeds, nothing else | Public got **worse** (+1,135), so v12's lead was largely seed luck | **Keep averaging as the default for finals** (lower variance for the private 70%); do not read gaps of ~1,000 as real | 17,975 |
| **v14** Rescale s120/s122 | These two sensors look like pure instrument changes (ratio ≈ ×2.7 at every quantile) and carry 16% of the classifier's gain | Quantile-map s120/s122 to the training distribution for the classifier; s126/s128 left alone (only their upper tail differs: looks like real wear); flag count matched so only ranking changes | −599 vs v13 | Promising, but see v15 | 17,376 |
| **v15** Replicate v14 | Is v14's gain the mapping or the base? | Same mapping on the v12_robust base | +218 vs v12: opposite sign to v14 | **Mapping has no reliable effect → dropped.** Public differences below ~600 between sibling files are noise | 17,058 |
| **v16 / v17** Level probes | Level (flag count, budget size) moves the score by thousands; ranking changes by hundreds | On the v12_robust base: flagged budgets ×1.25 (`v17_bud125`), and k ×1.35 for ~7% more flags (`v17_k135`); v16 = same on the v15 ranking, not uploaded | −14 and +68 vs v12_robust; the model itself expected +275 and +117 | **Both ties: the level is at its optimum.** Keep newsvendor budgets and the k posterior. The remaining gap can only come from ranking (the classifier) | 16,826 / 16,908 |

**Standing rules these decisions follow**
- Judge a change by **paired fold differences in cost** on repeated CV; AUC and log-loss gains were repeatedly misleading (v7 tuning, v8 features in the classifier, v11).
- Public leaderboard = 30% of test, ~6,000 vehicles with ~1% failures. It resolves changes of thousands of points (the flag level, the cost model), not hundreds. We use it for the **level** of the shift and never to pick between close siblings.
- Reject ideas honestly and keep them documented (`submissions/rejected/`, findings log), including our own wrong readings (v12's gain, v14's gain).

## Results (CV cost per vehicle on the 80% dev split; lower is better)

| Version | Change | CV cost | Public |
|---|---|---|---|
| v0 | flag nobody | 3.487 | n/a |
| v2 | XGBoost, flag if p > 0.5 | 2.501 | 74,933 |
| v3 | cost-based flag + newsvendor budget | 1.242 | 27,760 |
| v4 | + Platt calibration | 1.107 | 23,876 |
| v5 | + per-vehicle cost model | 0.904 | 22,969 |
| v6 | follow-up handling, missingness features | no gain (reverted) | n/a |
| v7 | prior-shift correction (test-only, no CV) | n/a | 19,067 (k=4) |
| v8 | histogram-shape features in the cost model (R² 0.66 → 0.79) | 0.866 (3×5 CV) | superseded |
| v9 | cost model 1,000 trees depth 2; k as belief | 0.862 (3×5 CV) | 17,014 (k=4) |
| v10 | posterior over k, Bayes decision | n/a (test-only) | 17,349 |
| v12 | cost model without drifted sensors | R² 0.801 | **16,840** |
| v13 / v14 / v15 | seed averaging; s120/s122 mapping | variance / rank probes | 17,975 / 17,376 / 17,058 |
| v17 | budget ×1.25; k ×1.35 | level probes | 16,826 / 16,908 |

The shift correction is test-only: the dev set has no shift by construction, so only the public score and a simulation can judge it.

## Course submission requirements

- **One notebook per version** (including every uploaded submission): see [`notebooks/README.md`](notebooks/README.md).
- **Report notebook:** [`notebooks/fleet_maintenance.ipynb`](notebooks/fleet_maintenance.ipynb) is the running codebase for the three selected models (it writes `notebooks/outputs/submission_model{1,2,3}.csv`) with the analysis in markdown cells. Rebuild it from `notebooks/build_notebook.py`.
- **Every Kaggle submission is tagged:** run `tools/tag_submission.sh <csv> <notebook> "<what changed>" "<score>"` right after uploading; the log is [`SUBMISSIONS.md`](SUBMISSIONS.md).
- **Individual contributions:** each member commits their own work from their own account; summary in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
- **Only the last submission before 27 Nov counts**, so the final upload must be the chosen model.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Download `train.csv`, `test.csv`, `sample_submission.csv` from the Kaggle competition into `data/`. Scripts run from `src/`:

```bash
cd src && ../.venv/bin/python v2_v3.py     # base classifier (writes reports/p_oof_v2.npy, p_te_v2.npy)
../.venv/bin/python v12_build.py           # a current candidate -> submissions/submission_v12_robust.csv
```

## Layout

- `src/score.py`: exact competition cost. `src/common.py`: data loading + fixes, locked 20% hold-out, fixed 5 folds, submission writer, experiment logger.
- `src/decide.py`: flag/budget decision rules. `src/calib.py`: Platt / isotonic. `src/features.py`: histogram-shape features.
- `src/vN_*.py`: one script per version/experiment.
- `notebooks/`: one notebook per version + the report notebook and its generators. `tools/`: submission tagging.
- `submissions/`: ready-to-upload files (`rejected/` holds files built on rejected models).

## Working rules

- The 20% hold-out is locked (used only at v0, v4, final).
- Choose final submissions by CV + hold-out + evidence agreement, not by the public leaderboard alone.
