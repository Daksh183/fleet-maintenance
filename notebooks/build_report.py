"""Builds final_report.ipynb: the final project report notebook (course instructions, section 5a).
It contains only (i) the running code that regenerates the three selected Kaggle CSVs (v3, v34, v51), (ii) the analysis of the
improvements from the baseline to the final models, and (iii) markdown cells that record the intermediate results, organised by the
evaluation scheme (part 2 of the instructions).
Run: ../.venv/bin/python build_report.py, then execute the notebook from this folder (about 5 minutes)."""
import os
from pathlib import Path

os.environ["ONLY"] = "__report_import_only__"          # import the shared cells of build_part7 without rewriting its notebooks
import build_part7 as B7                               # noqa: E402
from nbkit import NB, SETUP, FUNCS                     # noqa: E402

del os.environ["ONLY"]
V34_GP = (Path(__file__).resolve().parents[1] / "src" / "v34_hist_gp.py").read_text()          # Daksh's v34 GP, inlined verbatim
V34_GP = V34_GP[V34_GP.index("class GaussianSeverity:"):].replace(
    "class GaussianSeverity:", "class GaussianSeveritySD:   # Daksh's v34 class from src/v34_hist_gp.py, verbatim apart from its name", 1)

n = NB("Fleet Maintenance Planning: final report")

# ------------------------------------------------------------------ 0. title
n.md(r"""
# Fleet Maintenance Planning: final report

**Competition:** Kaggle `mlprojectfleet2`, Problem 2 (CSE 343 / ECE 363). **Team:** Group 16. **Repository:** `Daksh183/fleet-maintenance`.

For each of 20,000 vehicles from a **new depot**, decide whether to **flag** it for inspection and what spare-parts **budget** to set,
so that the total cost is as low as possible.

**The three selected models** (the three submissions selected on Kaggle). Running this notebook top to bottom regenerates their CSVs
in `outputs/` and checks each one against the uploaded file.

| Model | Version | Idea | Kaggle file | Public score |
|---|---|---|---|---|
| **Model 1** | v3 | XGBoost failure probability with a flag rule and newsvendor budget derived from the cost function | `submission_v3.csv` | 27,760.14 |
| **Model 2** | v34 | v12-style flags with Gaussian-process repair budgets that use each vehicle's own uncertainty and a context-corrected failure probability | `submission_v34_budget.csv` | 16,587.74 |
| **Model 3** | v51 | Bayes decision under a posterior over test-depot hypotheses, with a reduced-input Gaussian-process repair-cost model (**best public score**) | `submission_v51_gp_reduced.csv` | **16,388.51** |

Public scores are total cost on the public 30% of test (about 6,000 vehicles); lower is better. The baseline (v2: plain XGBoost with a
0.5 threshold) scored 74,933 on the same board. **Leaderboard: rank 13.** The final upload, v82, scored 16,440.10; our best public
score is 16,388.51 (Model 3).

**How this notebook maps to the evaluation scheme (instructions, part 2)**

| Criterion | Where |
|---|---|
| (a) Leaderboard ranking | Sections 0 and 9 |
| (b) Performance gains across the three model versions | Sections 5–8 |
| (c) Data and error analyses behind the improvements; limitations and capabilities of the best model | Sections 2, 3, 4 and 9 |
| (d) Individual contributions | Section 10 |
| (e) Understanding of the models, and the reasoning behind the choice of models and features | Sections 1, 4 and 5–7 (each model is explained before its code) |

**To run:** put `train.csv` and `test.csv` from the competition in `../data/` and run from the `notebooks/` folder. Models 2 and 3
load saved artifacts from `../reports/` (probabilities, out-of-fold predictions, public-score posteriors), as the course asked on 26 Nov
("save your trained artifacts"); each artifact is listed with the script in `../src/` that produced it. Libraries: numpy, pandas,
scipy, scikit-learn, xgboost.
""")
n.code(SETUP)
n.code(FUNCS)

# ------------------------------------------------------------------ 1. problem
n.md(r"""
## 1. The cost function and the two decisions it implies

For a vehicle with repair cost $c$ ($c = 0$ if it does not fail):

- not flagged: cost $= c$
- flagged with budget $b$: cost $= 10 + 0.20\,(c-b)^+ + 0.02\,(b-c)^+$

**Budget (newsvendor).** A unit of shortfall costs 0.20 and a unit of unused budget 0.02, so the best budget is the
$q^* = 0.20/0.22 = 0.909$ quantile of the vehicle's cost distribution. A vehicle fails with probability $p$, so its cost is 0 with
probability $1-p$, and
$b^* = 0$ if $p \le 0.0909$, else $b^* = F^{-1}\!\left(1 - 0.0909/p\right)$, where $F$ is the repair-cost distribution given failure.

**Flag.** Flag when the expected cost of flagging (with $b^*$) is below the expected cost of not flagging, $p\,\mu$. With $b^* = 0$
this means $p\,\mu > 12.5$; with the mean training repair cost $\mu \approx 492$, **flag when $p \gtrsim 2.5\%$**, far below 0.5.

So every model needs two things per vehicle: a **failure probability** $p$ and a **repair-cost distribution** $F$. Each model below
improves one or both, or the way the decision combines them.
""")

# ------------------------------------------------------------------ 2. data
n.md(r"""
## 2. Data analysis: what the data forced on us

Findings from the exploratory analysis (v1) and the depot-shift work (v7, v22–v26). The cell below recomputes the main ones.

1. **Rare, heavy-tailed target.** 397 failures out of 56,000 (0.71%). Repair cost: median 176, mean 492, maximum 8,000; the top 10% of
   failures carry 53% of all repair cost.
2. **A shifted histogram.** Sensors s054–s063 are a 10-bin histogram that is shifted by one bin in test: test bin $k+1$ matches train
   bin $k$. Every model realigns it. s047 (all missing in test) and s146 (constant) are dropped.
3. **A new depot.** After that fix a classifier still separates train from test (adversarial AUC 0.9999). s120 runs about +16M higher,
   s122 about ×2.7 at every quantile, and wear (s126, s128) is higher. Importance weighting is impossible (effective sample size
   0.3% of rows).
4. **A much higher failure rate at test.** The classifier predicts about 3× more failures on test; the label-shift EM estimate and the
   public scores put the test failure odds at roughly 3–6× training. This is why the flag rule had to be recalibrated (v7–v10).
5. **Censoring.** 22% of healthy vehicles were followed for less than 365 days. `followup_days` is partly determined by the outcome, so
   it is never a feature.
6. **Missing values predict failure** (most-missing quartile fails at 2.06%), so NaNs are kept and handled by XGBoost.
""")
n.code(r'''
raw_tr, raw_te, _ = load(raw=True)
ks_shift = [ks_2samp(raw_tr[f"s{54 + k:03d}"].dropna(), raw_te[f"s{55 + k:03d}"].dropna()).statistic for k in range(9)]
ks_same = [ks_2samp(raw_tr[f"s{54 + k:03d}"].dropna(), raw_te[f"s{54 + k:03d}"].dropna()).statistic for k in range(9)]
fc = tr.repair_cost[tr.failure == 1]
print(f"failures {int(tr.failure.sum())} of {len(tr)} ({tr.failure.mean():.2%}); repair cost median {fc.median():.0f}, mean {fc.mean():.0f}, max {fc.max():.0f}")
print(f"histogram s054-s063, KS(train bin k, test bin k):   {np.round(ks_same, 3)}")
print(f"histogram s054-s063, KS(train bin k, test bin k+1): {np.round(ks_shift, 3)}")
q = [0.1, 0.5, 0.9]
for s in ("s120", "s122", "s126"):
    print(f"{s}: train quantiles {np.round(np.nanquantile(tr[s], q) / 1e6, 2)}M, test {np.round(np.nanquantile(te[s], q) / 1e6, 2)}M")
cens = (tr.followup_days < 365) & (tr.failure == 0)
print(f"healthy vehicles followed < 365 days: {cens.sum()} ({cens.sum() / (tr.failure == 0).sum():.0%})")
''')

# ------------------------------------------------------------------ 3. baseline -> final
n.md(r"""
## 3. From the baseline to the final models

Our uploads in order. Each changed one thing, so the public score shows what that change was worth (the public difference between two
close files has noise of a few hundred points, so small differences are not evidence on their own). Rows marked ★ are the selected models.

| # | Version | What changed | Public |
|---|---|---|---|
| 1 | v2 | **Baseline:** XGBoost failure classifier, flag if p > 0.5, budget = mean repair cost | 74,933 |
| 2 | ★ **v3 (Model 1)** | Flag rule and budget derived from the cost function (Section 1) | **27,760** |
| 3 | v3b | v3 without the four drifting sensors; reverted | 29,530 |
| 4 | v4 | Platt calibration of the probabilities | 23,876 |
| 5–6 | v5 | Per-vehicle repair-cost model (XGBoost on log cost) for budgets and flags | 26,061 / 22,969 |
| 7–10 | v7, v9 | Depot shift: failure odds raised by a factor k (EM estimate); bigger cost model with histogram-shape features | 19,601 → 17,014 |
| 11 | v10 | k averaged over its posterior given the public scores | 17,349 |
| 12 | v12 | Repair-cost model without the four drifted sensors | 16,840 |
| 13–17 | v13–v17 | Seed averaging, quantile-mapping s120/s122, budget and flag-level probes | 16,826 – 17,975 |
| 18–20 | v18, v22, v24 | Rankings that correct for censoring or drop s120/s122: all worse publicly | 17,173 – 17,624 |
| 21 | v29 | **Harness:** Bayes decision under a posterior over test-depot hypotheses fitted to the public scores | 17,196 |
| 22–23 | v32, v33 | Histogram-kernel and **Gaussian-process repair-cost** budgets on v12-style flags (Daksh) | 16,992 / 16,588 |
| – | ★ **v34 (Model 2)** | v33 with budgets from each vehicle's GP uncertainty and a context-corrected failure probability (Daksh) | 16,588 |
| 24 | v35 | Harness flags with Gaussian-process budgets | 16,878 |
| 25 | v37 | Gaussian-process costs with wear and sensor groups, used for flags *and* budgets | 16,410 |
| 26 | v42 | Re-selected sensor group, cost spread by predicted-cost band | 16,487 |
| 27 | ★ **v51 (Model 3)** | Gaussian process with fewer inputs (ablation) | **16,388** |
| 28 | v82 (final upload) | Model 3's costs; decision hedged over all ranking families, incl. survival and CatBoost rankings | 16,440 |

**The steps that mattered.**

1. **Deriving the decision from the cost (v2 → v3, −63%).** A missed failure costs about 50 inspections, so the right threshold is near
   p = 2.5%, not 0.5. This was the largest single lever, and it is Model 1.
2. **Calibration and a per-vehicle cost model (v3 → v5, −17%).** Budgets became vehicle-specific instead of one global quantile.
3. **The depot shift (v5 → v12, −27%).** The test depot fails 3–6× more often. Raising the failure odds by k (EM label-shift estimate,
   then a posterior over k from the public scores) fixed the flag count. Removing the drifted sensors from the repair-cost model made
   the cost predictions robust to the new depot.
4. **A Gaussian-process repair-cost model (v12 → v33/v34, −1.5%).** A GP on the shapes of the sensor histograms predicts repair cost
   better than trees (out-of-fold R² 0.81 vs 0.80, budget loss −22%). Model 2 refines its budgets.
5. **The GP inside a Bayes decision fitted to the public scores (v29 → v51, −5% from v29).** The harness weighs hypotheses about the
   test depot by our earlier public scores; v37, v42 and v51 improve the GP inputs and spread. Model 3 is the result.

**What did not help** (each tried and measured; details in `../reports/findings.md`): dropping the drifted sensors from the classifier
(v3b, v22), censoring corrections (v18, v24), seed averaging (v13), quantile-mapping the drifted sensors (v14, v15), other model families
(CatBoost, LightGBM, histogram-gradient boosting, kernel methods), cost-weighted training, survival (AFT) rankings, and many repair-cost
variants (Section 9).
""")

# ------------------------------------------------------------------ 4. reasoning: models and features
n.md(r"""
## 4. Why these models and these features

Every choice below was made by measuring it: on the exact competition cost in cross-validation, on the locked hold-out where noted,
and on the public board when two files differed by one change.

### 4.1 The decision layer: derived, not learned

The cost is asymmetric (a missed failure costs about 50 inspections, a unit of shortfall 10× a unit of unused budget), so we do not
tune a classification threshold or a budget multiplier. We compute, per vehicle, the flag and budget that minimise **expected cost**
(Section 1). A model therefore only has to supply a calibrated failure probability and a repair-cost distribution, and each can be judged
on its own. This choice alone took the public score from 74,933 to 27,760 (Model 1).

### 4.2 The failure model: XGBoost on the raw sensors

**Why gradient-boosted trees (XGBoost).**
- 168 numeric sensors with heavy tails, counters on very different scales, and histogram bins. Trees are unaffected by monotone
  transforms, so no scaling or log choices are needed, and they learn interactions between bins and counters without hand-made features.
- **Missing values carry signal** (vehicles in the most-missing quartile fail at 2.06%). XGBoost learns a default direction for missing
  values at each split, so NaNs are kept rather than imputed.
- Only 397 failures (0.71%), so the model is kept small: 300 trees, depth 4, learning rate 0.05, row and column subsampling 0.8. A sweep
  of 11 regularisation variants (depth 2/3/6, slower learning, larger leaves, L1/L2, `max_delta_step`, `scale_pos_weight`, gamma) found
  none better under seven validation designs (`src/v79_regular.py`).
- No class reweighting (`scale_pos_weight`): it distorts the probabilities that the decision needs, and it was worse in validation.

**Alternatives tried and rejected.** Logistic regression and kernel models (+0.73 cost per vehicle, `src/v37_riskkernel.py`); CatBoost,
LightGBM and sklearn's histogram gradient boosting alone (worse in most validation designs, HistGB up to +55%); seed averaging (v13,
worse publicly). Survival (AFT) models that use `followup_days` as a training target were better offline (−2% to −12%), but they rank the
test depot very differently and the public scores cannot tell which is right, so they enter only as one hypothesis among many (v82).

**Probability calibration (Platt).** The flag decision depends on $p$ itself around 2.5%, and the raw model is under-confident exactly
there: for raw $p$ between 0.003 and 0.028 the observed failure rate is 2.5% against 1.0% predicted. Platt scaling (a logistic fit on the
logit, two parameters) fixes this and cut CV cost from 1.242 to 1.107 per vehicle (4/5 folds). Isotonic regression did slightly better in
CV but is the riskier choice with 318 development failures and was the worse of the two on the hold-out, so Platt was kept.
Public: 27,760 → 23,876 (v4).

**The depot shift (prior shift).** The test depot fails 3–6× more often. We assume the sensors-given-outcome relationship carries over
and only the failure rate changes (label shift), and raise the failure odds by a factor $k$: estimated by EM (Saerens et al.), validated on
simulated shifts inside the development set (it recovers $k$ = 1, 2, 4, 6 as 0.99, 1.93, 3.89, 5.91), then averaged over a posterior from the public
scores. Importance weighting, the usual alternative for covariate shift, was impossible: train and test overlap so little that the
effective sample size was 0.3% of rows.

### 4.3 Failure-model features

- **All sensors, after three data fixes:** the s054–s063 histogram realigned (without it, test bin $k$ would be read as train bin $k$:
  KS up to 0.98 instead of ≤ 0.07); s047 dropped (all missing in test); s146 dropped (constant).
- **The drifted sensors (s120, s122, s126, s128) are kept** in the classifier. Dropping them looked neutral in CV but was worse publicly
  (v3b: +1,770; v22: +333), and quantile-mapping them to the training distribution was also worse (v14, v15). Part of the drift is genuine
  risk (wear is higher at the new depot). How much weight they deserve is left to the harness, whose hypotheses include rankings with and
  without them and explicit wear tilts.
- **`followup_days` is never a feature:** it is missing from test and partly determined by the outcome (records stop near failures).
- **Engineered features did not help the classifier:** missing-value counts (v6), histogram shares, means, entropies and tail shares
  (v8), per-usage rates, monotone constraints and linear terms (v60), and a duplicate-row flag (v72) were all within noise or worse.

### 4.4 The repair-cost model: a Gaussian process on histogram shapes

**Why a Gaussian process.**
- Repair cost is known only for the 397 failures. A GP is a smooth non-parametric regressor with a handful of hyperparameters (one
  amplitude per input group and a noise level), fitted by maximum marginal likelihood, so it overfits less on 397 points than boosted
  trees, and it gives a predictive distribution, not just a mean.
- **The target is log cost** (costs run from 5 to 8,000, heavily skewed). Budgets come from quantiles of out-of-fold residuals, so there is
  no back-transform bias.
- **The inputs are histogram shapes.** Repair cost tracks the *shape* of the sensor histograms (s080–s085 correlate about 0.55 with log
  cost). Each histogram becomes square-root bin shares (a Hellinger-type distance compares shapes, not raw counts, which scale with
  usage) plus cumulative shares (where in the histogram the mass sits). One RBF kernel per group, added together, lets each histogram
  contribute its own smooth effect, and the fitted amplitudes act as automatic relevance weights.
- Evidence: out-of-fold R² 0.814 against 0.798 for the tree model, and budget loss −22% (v35); ridge, SVR, kNN and gradient boosting
  were 56–113% worse on budget loss, and every blend with the GP was worse (`src/v84_costens.py`).

**Cost-model inputs.**
- **The drifted sensors are excluded** from the cost model: costs predicted from them extrapolate at the new depot (v12: public −509).
- **Wear group** (log s126, s149, s123, Model 3): more-worn vehicles cost more to repair, and the test depot is more worn.
- **Cost sensors** (s116, s142, s105, s104, s078, Model 3): chosen by a greedy search on budget loss, and kept only if stable between
  depots (KS ≤ 0.032), so they do not shift at the new depot.
- **Removed by ablation** (Model 3): the usage group and the s023 and s129 histograms made budgets worse when removed one at a time;
  s080, s054, s037 and s154 are essential (removing them costs +7% to +38%).
- **Spread of the cost distribution:** Model 2 scales each vehicle's residuals by the GP's own predictive uncertainty; Model 3 takes
  residual quantiles from failures with similar predicted cost (three bands), which was more stable in our checks (the per-vehicle
  uncertainty barely tracked the actual errors, correlation 0.02–0.05).

### 4.5 Why these three models

- **Model 1 (v3)** isolates the most important idea: the decision derived from the cost, on a plain classifier.
- **Model 2 (v34)** is the strongest version of the single-classifier pipeline: calibration, depot shift, and Gaussian-process budgets
  with per-vehicle uncertainty, with the flags fixed from v33.
- **Model 3 (v51)** adds a decision fitted to the evidence about the test depot and the refined cost model, and has our best public score.
""")

# ------------------------------------------------------------------ 4. Model 1: v3
n.md(r"""
## 5. Model 1: v3, the decision derived from the cost (public 27,760.14)

**Idea.** The baseline v2 used a reasonable classifier with the wrong decision: flag if p > 0.5 and a fixed budget. v3 keeps the same
classifier and replaces the decision with the rule of Section 1. Here $F$ is **global**: the empirical cost distribution of the
training failures, the same for every vehicle.

- **Classifier:** XGBoost (300 trees, depth 4, learning rate 0.05) on all sensors after the v1 data fixes.
- **Validation:** 5 fixed stratified folds of the 80% development split; the 20% hold-out stays locked. Cost per vehicle with the exact
  competition cost. In each fold the budget distribution uses only the training folds' failures.

The cell reports the cross-validated cost of the v2 rule and of v3 on the same folds, then writes the test file.
""")
n.code(r'''
oof = fit_oof(dev[S])
p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
naive = fold_cost((oof > 0.5).astype(int), np.full(len(y), c[c > 0].mean()))
fl, bu = np.zeros(len(y), int), np.zeros(len(y))
for a, b in F:
    fl[b], bu[b] = decide_global(oof[b], c[a][c[a] > 0])
v3cv = fold_cost(fl, bu)
print(f"v2 rule (p > 0.5): {naive.mean():.3f} ± {naive.std():.3f} per vehicle")
print(f"v3 rule          : {v3cv.mean():.3f} ± {v3cv.std():.3f} per vehicle; paired change per fold {np.round(v3cv - naive, 3)}; flagged {fl.mean():.2%}")
f3, b3 = decide_global(p_te, c_all[c_all > 0])
write_submission(te.id, f3, b3, "v3")
SUMMARY = {"Model 1 (v3)": dict(flags=int(f3.sum()), mean_budget=b3[f3 == 1].mean())}
''')
n.md(r"""
**Intermediate results.**

| | CV cost / vehicle | Public score |
|---|---|---|
| v2 (p > 0.5) | 2.50 | 74,933 |
| **v3** | **1.24** | **27,760** (−63%) |

The cost halves in every fold, and the public score falls by 63%. About 5.4% of test vehicles are flagged (v2: 1.6%), as expected from
a threshold near 2.5%.

**What v3 gets wrong, which motivated the next versions.** The decision depends on $p$ itself near 2.5%, so it needs **calibrated**
probabilities (v4). $F$ is the same for every vehicle, although repair cost is predictable from the sensors (v5). And it assumes the
training depot's failure rate, while the new depot fails 3–6× more often (v7–v10).
""")

# ------------------------------------------------------------------ 5. Model 2: v34
n.md(r"""
## 6. Model 2: v34, Gaussian-process budgets with per-vehicle uncertainty (Daksh)

**Starting point: v33.** v33 keeps the v12-era decisions for *which* vehicles to flag: Platt-calibrated XGBoost probabilities $q$, the
depot-shift correction $p = 3q/(1 - q + 3q)$ (failure odds ×3), and the tree repair-cost model. It replaces the **budgets** with a
Gaussian-process repair-cost model, multiplied by 1.25 (16,588 public, −252 against v12-style files).

**The Gaussian process (v33).** Fitted on the 397 training failures, on log repair cost. Each of the six sensor histograms becomes a shape
vector (square-root bin shares and cumulative shares); the kernel is a sum of one Gaussian (RBF) kernel per histogram plus a usage total,
with amplitudes and noise fitted by maximum marginal likelihood and lengthscales set by the median pairwise distance. Vehicles with too
few observed histograms are priced by the tree model.

**v34 changes only the budgets of the 1,780 flagged vehicles**, in two ways:

1. **Per-vehicle uncertainty.** The GP also returns each vehicle's predictive standard deviation $\sigma_i$. The budget becomes
   $\exp(\mu_i + \sigma_i\,z_\ell)$, where $z_\ell$ is the newsvendor-level quantile of the *standardised* out-of-fold residuals, so a vehicle
   unlike the training failures gets a wider budget.
2. **A context correction to the failure probability, for budgets only.** A small penalised logistic regression with logit$(q)$ as an
   offset, on wear (s126), usage (s080–s089), their difference, s123, s149 and the missing-value fraction, shifts each vehicle's
   probability by at most ±1.5 on the logit scale. The budget level $1 - 1/(11p)$ depends on $p$, so this changes budgets but not flags.

**Intermediate results** (Daksh's validation, `reports/v34_implementation_plan.md`; 44,800 development vehicles, 5 outer folds × 3
repeats, simulated depot with 3× failure weight and the test depot's wear mix):

| check | result against v33 |
|---|---|
| per-vehicle uncertainty alone | −1.54% simulated depot cost |
| budget-only policy (v34), ordinary cost | 0.8288 → 0.8246 per vehicle |
| budget-only policy (v34), simulated depot cost | 1.9807 → 1.9421 (−1.96%, −1.65%, −2.24% over 3 repeats) |
| extrapolation: train without the top usage decile | +0.42% (worse) |
| extrapolation: train without the top wear decile | −0.14% |
| test-like wear × usage mix | −0.76% |
| rejected along the way | learned lengthscales +1.39%, histogram interactions +1.15%, log-ratio kernels +1.71%, no ×1.25 budget multiplier +1.65% |

The full policy (changing flags too) cost more on the hold-out, so v34 keeps v33's flags exactly.

**Artifacts:** `../reports/v33_baseline.npz` (v12 probabilities $q$, tree log-cost and its residuals; `src/v33_build.py`),
`../reports/p_oof_v2.npy` (out-of-fold classifier probabilities for the correction), and `../submissions/submission_v33_gp_budget.csv`,
whose budgets v34 keeps for unflagged vehicles (the scorer ignores those). Builder: `src/v34_build.py`.
""")
n.code(B7.GP_CLASS)
n.code("from scipy.special import expit, logit as slogit\n\n" + V34_GP.strip())
n.code(r'''
REP = Path("../reports")
def quantile_budget(prob, mean, residuals, spread=1.0):
    out = np.zeros(len(prob)); pos = prob > 1 / 11; lev = 1 - 1 / (11 * prob[pos])
    out[pos] = np.exp(mean[pos] + np.broadcast_to(spread, prob.shape)[pos] * np.quantile(residuals, lev)); return out

# repair costs: GP mean and sd for every test vehicle; standardised out-of-fold residuals (5 folds, seed 42, as v33)
fails = tr[tr.failure == 1]; Xf = extract(fails); lyf = np.log(fails.repair_cost.to_numpy())
gm, gs = np.full(len(lyf), np.nan), np.full(len(lyf), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(lyf):
    gm[b], gs[b] = GaussianSeveritySD().fit(select(Xf, a), lyf[a]).predict(select(Xf, b))
okf = np.isfinite(gm); res34 = lyf[okf] - gm[okf]; std34 = res34 / gs[okf]
mean34, sigma34 = GaussianSeveritySD().fit(Xf, lyf).predict(extract(te)); sup34 = np.isfinite(mean34)
print(f"GP: {okf.sum()} out-of-fold failures, R2 {1 - (res34 ** 2).sum() / ((lyf[okf] - lyf[okf].mean()) ** 2).sum():.4f}; "
      f"corr(|residual|, sd) {np.corrcoef(np.abs(res34), gs[okf])[0, 1]:.3f}; test vehicles priced by the GP {sup34.sum()}")

# flags exactly as v33: v12 probabilities with the prior shift k = 3 and the tree cost model
base = np.load(REP / "v33_baseline.npz"); assert np.array_equal(base["ids"], te.id)
q33, tree33, tres33 = base["q"], base["tree"], base["residual"]
p33 = 3 * q33 / (1 - q33 + 3 * q33)
fl34, _ = decide_pv(p33, tree33, tres33); insp = fl34 == 1
print(f"{insp.sum()} flags (the v33 inspection set)")

# change 1: context correction to the failure odds (wear, usage, missingness), used for budgets only
MISS = [f"s{i:03d}" for i in range(170) if i not in {47, 63, 120, 122, 146}]; USAGE = [f"s{i:03d}" for i in range(80, 90)]
def context(d):
    wear = np.log1p(d.s126.to_numpy(float)); use = np.log1p(d[USAGE].sum(axis=1, min_count=10).to_numpy(float))
    return np.column_stack([wear, use, wear - use, np.log1p(d.s123), np.log1p(d.s149), d.reindex(columns=MISS).isna().mean(axis=1)])
raw_te2 = raw_te.copy(); raw_te2[HISTB[:-1]] = raw_te2[HISTB[1:]].to_numpy(); raw_te2[HISTB[-1]] = np.nan
dev_raw = raw_tr.iloc[dev_idx].reset_index(drop=True)
p_oof = np.load(REP / "p_oof_v2.npy"); qdev = Platt().fit(p_oof, y).predict(p_oof)
Fd = context(dev_raw); lo, med, hi = np.nanquantile(Fd, [0.01, 0.5, 0.99], axis=0)
fill = lambda v: np.clip(np.where(np.isfinite(v), v, med), lo, hi)
Td = fill(Fd); cen, scl = Td.mean(0), np.maximum(Td.std(0), 0.001)
design = lambda v, t: np.column_stack([np.ones(len(v)), (t - cen) / scl, np.isnan(v[:, :2])])
Xd = design(Fd, Td); off = slogit(np.clip(qdev, 1e-6, 1 - 1e-6))
def nll(cf):
    s = off + Xd @ cf; return np.sum(np.logaddexp(0, s) - y * s) + 8 * cf @ cf, Xd.T @ (expit(s) - y) + 16 * cf
coef = minimize(nll, np.zeros(Xd.shape[1]), jac=True, method="L-BFGS-B").x
Ft = context(raw_te2); eta = np.clip(design(Ft, fill(Ft)) @ coef, -1.5, 1.5)
p_budget = expit(slogit(np.clip(p33, 1e-9, 1 - 1e-9)) + eta)
print(f"context correction on flagged vehicles: logit shift mean {eta[insp].mean():+.3f}, range {eta[insp].min():+.2f} to {eta[insp].max():+.2f}")

# change 2: each vehicle's GP sd scales the standardised residual quantiles; budgets x1.25 as in v33
bu34 = np.zeros(len(te)); act = insp & sup34; fb = insp & ~sup34
bu34[act] = (1.25 * quantile_budget(p_budget[act], mean34[act], std34, sigma34[act])).round(2)
bu34[fb] = (1.25 * quantile_budget(p_budget[fb], tree33[fb], tres33)).round(2)
v33file = pd.read_csv(Path("../submissions/submission_v33_gp_budget.csv")).set_index("id").loc[te.id]
bu34[~insp] = v33file.budget.to_numpy()[~insp]          # unflagged rows keep v33's values (the scorer ignores budgets of unflagged vehicles)
print(f"mean budget on flagged vehicles {bu34[insp].mean():.0f} (v33: {v33file.budget.to_numpy()[insp].mean():.0f})")
write_submission(te.id, fl34, bu34, "v34_budget")
SUMMARY["Model 2 (v34)"] = dict(flags=int(insp.sum()), mean_budget=bu34[insp].mean())
''')

# ------------------------------------------------------------------ 6. Model 3: v51
n.md(r"""
## 7. Model 3: v51, a Bayes decision fitted to the public scores, with a reduced-input Gaussian process (public 16,388.51, best)

Model 3 changes both halves of the decision.

**Failure probabilities: the harness.** Which ranking holds at the test depot, and how much more often it fails, cannot be learned from
training data. The harness considers a family of hypotheses about the test depot:

$$p_i = \sigma\!\left(a + b\,L_i + c\,z_i\right),$$

where $L_i$ blends the logits of ten classifier rankings built earlier (v12, no-s122, soft labels, censoring ensembles,
instrument-corrected, quantile-mapped, ...), $z_i$ is a wear score (s120, s122, s126 or s149) with tilt $c$, $b$ is a confidence slope,
$a$ the overall level, and a repair-cost level $\alpha$ scales costs. Each hypothesis is weighted by the **exact likelihood of our earlier
public scores**: for a hypothesis we know the mean and covariance of every uploaded file's score on a random 30% of test, so the observed
scores give a Gaussian likelihood. The decision is the **Bayes decision**: for each vehicle, the flag and newsvendor budget that minimise
the posterior-expected cost. Budgets are capped at 8,000.

**Repair costs: three refinements of the v33 Gaussian process**, each checked with honest cross-fitted budget loss (budget quantiles
estimated on other folds' failures, weighted over the newsvendor levels the flagged test vehicles need):

| step | change | evidence | public |
|---|---|---|---|
| v37 | add a wear group (s126, s149, s123) and six cost sensors; use GP costs for flags too | R² 0.814 → 0.832, budget loss −4.6% | 16,410 |
| v42 | sensors re-selected (s116, s142, s105, s104, s078); residual spread by predicted-cost third | −7.9%, −6.6%, −6.2% on 3 fold seeds | 16,487 |
| **v51** | **drop the usage group and the s023 and s129 histograms; average fits with and without s123** | **−7.4% (10/10 fresh seeds); end-to-end CV −2.15% (18/25 folds)** | **16,388** |

The v51 inputs came from an ablation (`src/v47_ablation.py`, 26 variants, 5 seeds): removing histogram s080 costs +38% budget loss, s054
+20%, s037 +10%, s154 +7%, the sensor group +6%; Matérn or rational-quadratic kernels, raw bin levels, other cost transforms, other
sensors, principal components and a missing-value count were no better; removing the usage group and the s023 and s129 histograms helped.
The groups were chosen on the same failures, so the real gain is probably smaller than measured. End-to-end, GP costs also beat tree
costs in the flag decision: −3.3% cost per vehicle (11/15 folds, `src/v45_flagmu.py`).

**Artifacts** (`../reports/`): `v18_qm.npy` (v12 test probabilities and tree log-cost, `src/v18_build.py`), `v18_resid.npy`, the ranking
probabilities `v19_q.npy`, `v21_H.npy`, `v22_q.npy`, `v24_q.npy`, `v24_pu_x2_q.npy`, `v18_ens10_q.npy`, `v18_ens_q.npy`, `p_oof_v2.npy`,
and the posterior `v51_post.csv` (26 public scores, `src/v51_harness.py`).
""")
n.code(B7.RANKS)
n.code(B7.DECIDE)
n.code(r'''
class GPAll(GaussianSeverity):
    """predicts every row; which rows count is decided below, the same way for both fits"""
    def predict(self, features):
        return np.clip(np.einsum("i,ijk->jk", self.amplitude, self.blocks(features)) @ self.alpha + self.center, *self.bounds)

f = tr[tr.failure == 1].reset_index(drop=True); ly = np.log(f.repair_cost.to_numpy())
START6 = (23, 37, 54, 80, 129, 154); KEEP = (37, 54, 80, 154)
SENS = ["s116", "s142", "s105", "s104", "s078"]
def z(d, cols):                                   # log1p, standardised with the training failures' mean and sd
    R = np.log1p(np.maximum(f[cols].to_numpy(float), 0))
    return (np.log1p(np.maximum(d[cols].to_numpy(float), 0)) - np.nanmean(R, 0)) / (np.nanstd(R, 0) + 1e-9)
def feats(d, wear):
    H = extract(d)
    return [H[START6.index(a)] for a in KEEP] + [z(d, wear), z(d, SENS)]
# rows the Gaussian process may price: enough of the original blocks observed
valid = sum(np.isfinite(x).all(1).astype(int) for x in extract(f)[:6]) >= 3
sup = sum(np.isfinite(x).all(1).astype(int) for x in extract(te) + [z(te, ["s126", "s149", "s123"])]) >= 3
oofs, sevs = [], []
for wear in (["s126", "s149", "s123"], ["s126", "s149"]):
    X = feats(f, wear); o = np.full(len(ly), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=42).split(ly):
        o[b] = GPAll().fit(select(X, a), ly[a]).predict(select(X, b))
    o[~valid] = np.nan; s = GPAll().fit(X, ly).predict(feats(te, wear)); s[~sup] = np.nan
    oofs.append(o); sevs.append(s)
oof, sev = (oofs[0] + oofs[1]) / 2, (sevs[0] + sevs[1]) / 2
ok = np.isfinite(oof); gres = ly[ok] - oof[ok]; sup = np.isfinite(sev)
print(f"GP out-of-fold R2 {1 - (gres ** 2).sum() / ((ly[ok] - ly[ok].mean()) ** 2).sum():.4f}; residual sd {gres.std():.3f}; test supported {sup.sum()}")
''')
n.code(r'''
P = load_posterior("v51_post.csv")
mu_oof = oof[ok]; edges = np.quantile(mu_oof, [1 / 3, 2 / 3])
band_of_failure = np.searchsorted(edges, mu_oof); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
QB = np.stack([np.quantile(gres[band_of_failure == j], grid) for j in range(3)])
draws = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v51_gp_reduced")
SUMMARY["Model 3 (v51)"] = dict(flags=int(fl.sum()), mean_budget=bu[fl == 1].mean())
''')

# ------------------------------------------------------------------ 7. gains + verification
n.md(r"""
## 8. Performance across the three models, and a check against the uploaded files

The table is computed from the runs above. The check confirms that each regenerated CSV is identical to the file uploaded to Kaggle
(same ids and flags, budgets equal to the cent).
""")
n.code(r'''
PUBLIC = {"Model 1 (v3)": 27760.14, "Model 2 (v34)": 16587.74, "Model 3 (v51)": 16388.51}     # public scores from Kaggle
R = pd.DataFrame(SUMMARY).T; R["public"] = pd.Series(PUBLIC)
R["public vs Model 1"] = (R.public / R.public.iloc[0] - 1).map(lambda v: f"{v:+.1%}" if v == v else "n/a")
print(R.round({"mean_budget": 0, "public": 2}).to_string()); print()
for name in ("v3", "v34_budget", "v51_gp_reduced"):
    a, b = pd.read_csv(OUT / f"submission_{name}.csv"), pd.read_csv(Path("../submissions") / f"submission_{name}.csv")
    same = a.id.equals(b.id) and (a.flag == b.flag).all() and np.allclose(a.budget, b.budget, atol=0.015)
    print(f"{'MATCH ' if same else 'DIFFER'} {name}: flags differ {int((a.flag != b.flag).sum())}, max budget difference {np.abs(a.budget - b.budget).max():.2f}")
''')
n.md(r"""
**Reading the gains.**

- **Model 1 → Model 2.** The decision rule (Model 1) is the foundation. Model 2 adds everything that followed it: calibrated
  probabilities, the depot-shift correction (odds ×3), a per-vehicle repair-cost model, and on top of that the Gaussian-process budgets
  with per-vehicle uncertainty. Together these cut the public score from 27,760 to 16,588, −40%.
- **Model 2 → Model 3.** Model 3 changes *which* vehicles are flagged (about 1,430 instead of 1,780, chosen by a posterior fitted to the
  public scores) and uses a better GP (R² 0.84 against 0.81, with banded spread and reduced inputs). Offline the three GP refinements
  (v37, v42, v51) cut budget loss by roughly 15–18% relative to v33's GP. Publicly it is our best file: 16,388, −200 (−1.2%) against
  Model 2.
- On the public board, differences below about 300 are inside the noise: only about 190 public vehicles fail, with heavy-tailed costs.
""")

# ------------------------------------------------------------------ 8. error analysis, limitations
n.md(r"""
## 9. Error analysis of the best model, its limitations and capabilities, and the leaderboard

The cell below breaks Model 3's expected cost on the public 30% into its parts, under its own posterior, and compares it with two
oracles: knowing which vehicles fail, and knowing each vehicle's repair cost if it fails.
""")
n.code(r'''
PB, Wa = posterior_means(P); wsum = sum(Wa.values()); flg = fl == 1
parts = dict.fromkeys(["inspections", "unused budget, flagged vehicles that do not fail", "shortfall, flagged failures",
                       "unused budget, flagged failures", "missed failures (not flagged)"], 0.0)
extra = dict.fromkeys(["fail_flagged", "fail_missed", "oracle_fail", "oracle_cost"], 0.0)
for al in ALPHAS:
    if Wa[al] == 0: continue
    w = Wa[al] / wsum; p = PB[al] / Wa[al]; C = np.clip(draws * al, 5, 8000); mu = C.mean(1)
    sh = (UNDER * np.maximum(C - bu[:, None], 0)).mean(1); ov = (OVER * np.maximum(bu[:, None] - C, 0)).mean(1)
    parts["inspections"] += w * INSPECT * flg.sum()
    parts["unused budget, flagged vehicles that do not fail"] += w * (OVER * bu * (1 - p))[flg].sum()
    parts["shortfall, flagged failures"] += w * (p * sh)[flg].sum()
    parts["unused budget, flagged failures"] += w * (p * ov)[flg].sum()
    parts["missed failures (not flagged)"] += w * (p * mu)[~flg].sum()
    extra["fail_flagged"] += w * p[flg].sum(); extra["fail_missed"] += w * p[~flg].sum()
    extra["oracle_fail"] += w * (p * (INSPECT + sh + ov)).sum()                                  # flag exactly the failures, same budgets
    best = np.minimum(p[:, None] * UNDER * C, (1 - p)[:, None] * OVER * C).mean(1)              # know c if it fails: budget 0 or c
    extra["oracle_cost"] += w * (INSPECT * flg.sum() + (p * mu)[~flg].sum() + best[flg].sum())
tot = sum(parts.values())
print("Model 3, expected cost on the public 30% (actual public score 16,388.51):")
for k, v in parts.items(): print(f"  {k:50s} {0.3 * v:7.0f}  ({v / tot:5.1%})")
print(f"  {'total':50s} {0.3 * tot:7.0f}")
print(f"\nexpected failures on the public 30%: {0.3 * (extra['fail_flagged'] + extra['fail_missed']):.0f}, of which flagged "
      f"{extra['fail_flagged'] / (extra['fail_flagged'] + extra['fail_missed']):.0%}; flagged vehicles {0.3 * flg.sum():.0f}")
print(f"oracle, knowing which vehicles fail: {0.3 * extra['oracle_fail']:.0f};  oracle, knowing each repair cost: {0.3 * extra['oracle_cost']:.0f}")
''')
n.md(r"""
**What the breakdown says.** About a third of the expected cost is inspections, about half is budget error (shortfall plus unused
budget), and about a sixth is missed failures. The model catches about 90% of expected failures. The two oracles bound what is left:
knowing which vehicles fail would save roughly 6,000 on the public 30%; knowing repair costs exactly, about 4,000.

**Capabilities of the best model.**
- Decisions follow from the cost function: a 2.5% flag threshold and per-vehicle newsvendor budgets, not a 0.5 cut-off and a fixed budget.
- It adapts to the new depot: the failure level, the ranking and the wear tilts are weighted by how well they explain the public scores,
  and the decision hedges over all of them instead of betting on one.
- Its repair-cost model (R² about 0.84 on log cost, out of fold) beats every alternative we tried by a wide margin: ridge, SVR, kNN,
  gradient boosting and all blends with the GP were 16–113% worse on budget loss (`src/v84_costens.py`).

**Limitations.**
- **Which test vehicles fail is the largest open problem, and the available data cannot resolve it.** Rankings from different model
  families disagree strongly on the test depot (CatBoost against the classifier: rank correlation 0.58), yet they fit the public scores
  equally well, and offline validation runs on the training depot. Seven validation designs (plain, wear-weighted, complete cases,
  instrument scenario, cluster hold-out, wear and usage hold-out) were backtested against five changes with known public results; none
  predicted them reliably (`src/v67_valmethods.py`). Our final upload, v82, hedges over these rankings for this reason.
- **The public score is noisy.** It covers about 6,000 vehicles and about 190 failures with heavy-tailed costs; one 8,000 repair moves a
  score by thousands. Under the model a single file's public score has a standard deviation of about 1,850, and all our files scored
  above their expectation by a common amount, consistent with a few expensive failures in the public 30%.
- **The cost model rests on 397 failures**, and its inputs were chosen on them, so some measured gains are selection.
- **The posterior is fitted to the public scores**, so part of its edge may be fitting the public subset rather than the private 70%.

**Leaderboard (criterion a).** Public leaderboard: **rank 13**. Best public score 16,388.51 (Model 3); the final upload, v82 (the Bayes
decision hedged over every ranking family, with Model 3's repair costs), scored 16,440.10, +52 against Model 3 and within the noise.
The private ranking (70% of test) decides the final standing.
""")

# ------------------------------------------------------------------ 9. contributions
n.md(r"""
## 10. Individual contributions

Summarised from the commit history of `Daksh183/fleet-maintenance` and `CONTRIBUTIONS.md`. Each member committed their own work from
their own account.

| Member | GitHub | Contributions |
|---|---|---|
| Vaibhav Sahni | `vaibhav-sahni` | Repository, validation protocol (locked hold-out, fixed folds, exact cost scorer) and submission log; v8 candidate comparisons, versions v9–v31 and v35–v82: depot-shift posterior (v10), robust cost model (v12), level probes, censoring and ranking probes (v13–v24), the public-score **harness** and Bayes decision (v28–v29), and the Gaussian-process refinements leading to **Model 3** (v37, v42, v51) with their ablations; rebuild notebooks, findings log and this report |
| Daksh Sheoran | `Daksh183` | Cost scorer, baselines and v1 sensor analysis (v0–v1); v2 classifier and the **cost-based decision rule of Model 1** (v3) and v3b; v0–v5 notebook builder and histogram discovery scripts; **v32 histogram kernel, the v33 Gaussian-process repair-cost model and Model 2 (v34)**; the GP is the basis of Models 2 and 3 |
| Shivanshu Kumar Singh | `Shivanshu-Kr-Singh` | **Probability calibration** pipeline (v4: Platt and isotonic, `src/calib.py`, notebook kit); **depot-shift correction** (v7: label-shift EM, its simulation check, SHAP and regret analysis; public 19,067); follow-up and missingness variants tested and rejected (v6); feature-drift validation tool (v8) |
| Naveen Jangra | `Naveen23-hub` | **Per-vehicle repair-cost model** (v5) used for budgets and flags; histogram-shape features and their repair-cost validation over 15 folds (v8) |
""")

n.save("final_report.ipynb")
