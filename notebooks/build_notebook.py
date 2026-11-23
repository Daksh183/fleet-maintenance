"""Builds fleet_maintenance.ipynb (the report notebook). Run: ../.venv/bin/python build_notebook.py"""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Fleet Maintenance Planning: ML Project (Kaggle `mlprojectfleet2`)

For each of 20,000 vehicles from a **new depot**, decide whether to **flag** it for inspection and what spare-parts **budget** to set, so that the total cost is as low as possible.

This notebook is the running codebase for the **three selected model versions**. Running it top to bottom regenerates the three Kaggle CSVs in `outputs/`, and it documents the analysis behind each improvement.

| Model | Idea | Kaggle file |
|---|---|---|
| **Model 1** | XGBoost failure probability + cost-derived flag rule + newsvendor budget | `outputs/submission_model1.csv` |
| **Model 2** | + probability calibration (Platt) + per-vehicle repair-cost model | `outputs/submission_model2.csv` |
| **Model 3** | + histogram-shape features in the cost model + depot-shift (prior) correction | `outputs/submission_model3.csv` |

Data: put `train.csv` and `test.csv` from the competition in `../data/`. Allowed tools only: classical ML (XGBoost, scikit-learn). Run time: about 5 minutes on a laptop.
""")

code(r"""
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from scipy.optimize import nnls
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_predict

DATA, OUT = Path("../data"), Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42
INSPECT, UNDER, OVER = 10.0, 0.20, 0.02        # inspection cost, shortage cost per unit, unused-parts cost per unit
Q_STAR = UNDER / (UNDER + OVER)                # newsvendor critical ratio = 0.909
""")

md(r"""
## 1. The cost function and the two decisions it implies

For a vehicle with repair cost $c$ ($c=0$ if it does not fail):

- not flagged: cost $= c$
- flagged with budget $b$: cost $= 10 + 0.20\,(c-b)^+ + 0.02\,(b-c)^+$

**Budget (newsvendor).** A unit of shortage costs 0.20 and a unit of unused budget costs 0.02, so the optimal budget is the $q^*=0.20/0.22=0.909$ quantile of the vehicle's cost distribution. With failure probability $p$, $c$ is 0 with probability $1-p$, so
$b^* = 0$ if $p \le 0.0909$, else $b^* = F^{-1}\!\left(1 - 0.0909/p\right)$, where $F$ is the repair-cost distribution *given* failure.

**Flag.** Flag when $E[\text{cost}\mid\text{flag}, b^*] < p\,\mu$ (the expected cost of not flagging). With $b^*=0$ this is $p\,\mu > 12.5$; with the mean repair cost $\mu\approx492$ that means **flag when $p \gtrsim 2.5\%$**, far below 0.5. A missed failure costs ~50× an inspection.

Everything below is scored with the exact competition cost, reported **per vehicle** (total ÷ number of vehicles) so that folds, the hold-out and test are comparable.
""")

code(r"""
def vehicle_cost(c, flag, b):
    return np.where(flag == 1, INSPECT + UNDER * np.maximum(c - b, 0) + OVER * np.maximum(b - c, 0), c)

def per_vehicle(c, flag, b):
    return float(vehicle_cost(np.asarray(c, float), np.asarray(flag), np.asarray(b, float)).mean())

def decide_global(p, fail_costs):
    # Same repair-cost distribution F (empirical training failure costs) for every vehicle.
    p = np.clip(np.asarray(p, float), 0, 1); fc = np.sort(np.asarray(fail_costs, float)); mu = fc.mean()
    b = np.zeros_like(p); need = p > 1 - Q_STAR
    b[need] = np.quantile(fc, 1 - (1 - Q_STAR) / p[need])
    fail = (UNDER * np.maximum(fc[None, :] - b[:, None], 0) + OVER * np.maximum(b[:, None] - fc[None, :], 0)).mean(1)
    return ((INSPECT + p * fail + (1 - p) * OVER * b) < p * mu).astype(int), b

def decide_pv(p, m, resid, K=99):
    # Per-vehicle repair-cost distribution: log c | fail ~ m_i + out-of-fold residuals (no exp(mean) bias).
    p, m = np.clip(np.asarray(p, float), 0, 1), np.asarray(m, float)
    M = np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + 0.5) / K)[None, :]); mu = M.mean(1)
    b = np.zeros_like(p); need = p > 1 - Q_STAR
    b[need] = np.exp(m[need] + np.quantile(resid, 1 - (1 - Q_STAR) / p[need]))
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return ((INSPECT + p * fail + (1 - p) * OVER * b) < p * mu).astype(int), b

# sanity check: the worked example in the problem statement totals 1,241
ex = vehicle_cost(np.array([400, 0, 1200, 0.]), np.array([1, 1, 0, 0]), np.array([300, 50, 0, 0.])).sum()
assert ex == 1241.0
print("worked example total:", ex)
""")

md(r"""
## 2. Data analysis and the fixes it led to

Key findings (full log: `reports/findings.md`):

- **Severe imbalance:** 397 failures in 56,000 vehicles (0.71%). Repair cost is heavy-tailed (median 176, mean 492, max 8,000; the top 10% of failures carry 53% of the cost).
- **Censoring:** 22% of healthy vehicles were followed for < 365 days, and the failure rate rises with follow-up. `followup_days` is absent from test and partly *caused by* the outcome, so it is **never used as a feature**.
- **Depot shift:** a classifier separates train from test perfectly (adversarial AUC 1.0). Two causes are data errors we can fix:
  1. `s047` is 100% missing in test → dropped (and `s146` is constant → dropped).
  2. The 10-bin histogram `s054–s063` is **shifted by one bin in test**: test bin $k+1$ matches train bin $k$ (checked below). Realigned.
- What remains (`s120`, `s122`, and a diffuse shift elsewhere) is a genuine difference between depots; it is handled in Model 3.
""")

code(r"""
tr_raw, te_raw = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
H = [f"s{i:03d}" for i in range(54, 64)]
same = [ks_2samp(tr_raw[H[i]].dropna(), te_raw[H[i]].dropna()).statistic for i in range(9)]
shifted = [ks_2samp(tr_raw[H[i]].dropna(), te_raw[H[i + 1]].dropna()).statistic for i in range(9)]
print("KS train bin k vs test bin k   :", np.round(same, 3))
print("KS train bin k vs test bin k+1 :", np.round(shifted, 3), " <- matches: test is shifted by one bin")

def load():
    tr, te = tr_raw.copy(), te_raw.copy()
    te[H[:-1]] = te[H[1:]].to_numpy(); te[H[-1]] = np.nan          # realign histogram
    tr, te = tr.drop(columns=["s047", "s146"]), te.drop(columns=["s047", "s146"])
    return tr, te, [c for c in tr.columns if c.startswith("s")]

tr, te, S = load()
c_all = tr.repair_cost.fillna(0).to_numpy() * tr.failure.to_numpy()
print(f"train {tr.shape}, test {te.shape}, sensors used {len(S)}; failures {tr.failure.sum()} ({tr.failure.mean():.2%})")
""")

md(r"""
## 3. Validation protocol and baselines

- A **locked 20% hold-out** (stratified) is set aside; everything is tuned on the 80% dev split with **5 fixed stratified folds**.
- Fold costs are very noisy (one 8,000-unit failure moves a fold), so versions are compared with **paired fold differences**, never by eyeballing mean ± std.
""")

code(r"""
dev_idx, ho_idx = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=tr.failure, random_state=SEED)
dev = tr.iloc[dev_idx].reset_index(drop=True)
y = dev.failure.to_numpy(); c = dev.repair_cost.fillna(0).to_numpy() * y
F = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
fold_cost = lambda flag, b: np.array([per_vehicle(c[v], flag[v], b[v]) for _, v in F])

base = {"flag nobody": fold_cost(np.zeros(len(y), int), np.zeros(len(y))),
        "flag all, budget 0": fold_cost(np.ones(len(y), int), np.zeros(len(y))),
        "oracle (perfect foresight)": fold_cost(y, c)}
for k, v in base.items(): print(f"{k:28s} {v.mean():.3f} ± {v.std():.3f} per vehicle")
""")

md(r"""
**Bar to beat: 3.49 per vehicle (flag nobody).** Flagging everyone is 3× worse, because 99.3% of inspections would be wasted. The oracle (0.07) shows how much room a model has.

## 4. Model 1: XGBoost + cost-derived decisions

A gradient-boosted classifier estimates $p(\text{fail})$ from the sensors (missing values left as NaN, which XGBoost handles natively). The decision layer turns $p$ into flag and budget using Section 1, with **one global** repair-cost distribution $F$ (the training failures).

We also show the naive version (flag if $p>0.5$, budget = mean cost) to quantify how much of the gain comes from the decision rule alone.
""")

code(r"""
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=1, n_jobs=-1, random_state=SEED)
oof = np.zeros(len(dev))
for a, b in F:
    oof[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1]
p_te = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
print(f"OOF AUC {roc_auc_score(y, oof):.4f}   PR-AUC {average_precision_score(y, oof):.4f}")

naive = fold_cost((oof > 0.5).astype(int), np.full(len(y), c[c > 0].mean()))
m1_flag, m1_b = np.zeros(len(y), int), np.zeros(len(y))
for a, b in F:
    m1_flag[b], m1_b[b] = decide_global(oof[b], c[a][c[a] > 0])
m1 = fold_cost(m1_flag, m1_b)
print(f"naive (p>0.5)          : {naive.mean():.3f} ± {naive.std():.3f}")
print(f"Model 1 (cost rule)    : {m1.mean():.3f} ± {m1.std():.3f}   paired gain vs naive {np.round(m1 - naive, 3)}")

f1, b1 = decide_global(p_te, c_all[c_all > 0])
pd.DataFrame({"id": te.id, "flag": f1, "budget": b1.round(2)}).to_csv(OUT / "submission_model1.csv", index=False)
print(f"Model 1 test: flagged {f1.mean():.2%}")
""")

md(r"""
**Analysis.** Same classifier, different decision rule: the cost-derived rule halves the cost (2.50 → 1.24; public score 74,933 → 27,760, −63%). The 0.5 threshold is the single biggest mistake one can make on this problem.

**Error analysis that motivated Model 2:**
1. *Calibration:* the decision depends on $p$ itself near 2.5%, but in exactly that range the raw model is **under-confident** (raw $p$ 0.3–2.8%: 2.5% observed vs 1.0% predicted). It under-flags.
2. *One cost distribution for everyone:* a vehicle's repair cost turns out to be predictable from its sensors (below), so a global $F$ wastes information in both the budget and the flag decision.
""")

code(r"""
bins = [0, 0.003, 0.028, 0.15, 1]
cut = pd.cut(oof, bins)
print(pd.DataFrame({"predicted": pd.Series(oof).groupby(cut, observed=True).mean(),
                    "observed": pd.Series(y).groupby(cut, observed=True).mean(),
                    "n": pd.Series(y).groupby(cut, observed=True).size()}).round(4))
""")

md(r"""
## 5. Model 2: calibration + per-vehicle repair-cost model

- **Platt scaling** maps raw scores to calibrated probabilities. It is cross-fitted: each fold's calibrator is fitted on the *other* folds' out-of-fold predictions.
- **Repair-cost model:** XGBoost regression on log(repair cost), trained on failing vehicles only. Each vehicle's cost distribution is its prediction plus the model's *out-of-fold* residuals (honest spread, no exp-of-mean bias). The budget becomes the per-vehicle 0.909-quantile and the flag rule uses the per-vehicle mean.
- Nested: inside each outer fold, the cost model and its residuals use only that fold's training failures.
""")

code(r"""
class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(p), y); return self
    def predict(self, p): return self.m.predict_proba(self._x(p))[:, 1]
    @staticmethod
    def _x(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))[:, None]

CM = dict(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
          min_child_weight=5, n_jobs=-1, random_state=SEED)
inner = KFold(5, shuffle=True, random_state=SEED)

def cost_model(params, Xa, lc, Xb):
    resid = lc - cross_val_predict(xgb.XGBRegressor(**params), Xa, lc, cv=inner)
    return xgb.XGBRegressor(**params).fit(Xa, lc).predict(Xb), resid

m2, r2_base = [], []
for a, b in F:
    fa, fb = a[y[a] == 1], b[y[b] == 1]
    mb, resid = cost_model(CM, dev.loc[fa, S], np.log(c[fa]), dev.loc[b, S])
    r2_base.append(r2_score(np.log(c[fb]), mb[y[b] == 1]))
    fl, bu = decide_pv(Platt().fit(oof[a], y[a]).predict(oof[b]), mb, resid)
    m2.append(per_vehicle(c[b], fl, bu))
m2 = np.array(m2)
print(f"cost model R2 (log scale, held-out failures): {np.mean(r2_base):.3f}")
print(f"Model 2: {m2.mean():.3f} ± {m2.std():.3f}   paired vs Model 1 {np.round(m2 - m1, 3)}  better {(m2 < m1).sum()}/5")

pl = Platt().fit(oof, y); q_te = pl.predict(p_te)
fail_tr = tr[tr.failure == 1]; lc_all = np.log(fail_tr.repair_cost.to_numpy())
m_te2, resid2 = cost_model(CM, fail_tr[S], lc_all, te[S])
f2, b2 = decide_pv(q_te, m_te2, resid2)
pd.DataFrame({"id": te.id, "flag": f2, "budget": b2.round(2)}).to_csv(OUT / "submission_model2.csv", index=False)
print(f"Model 2 test: flagged {f2.mean():.2%}")
""")

md(r"""
**Analysis.** Model 2 beats Model 1 in every fold (1.24 → 0.90 per vehicle, −27%); public score 27,760 → 22,969 (−17%). Ablations (in `reports/findings.md`): most of the gain is the per-vehicle **flag** decision; calibration helps on top (and on the public set, raw vs Platt: 26,061 vs 22,969).

**Error analysis that motivated Model 3:**
1. *The public scores are ~2× higher than our model expects* if the public set is 30% of test, and the model already predicts ~3× more failures on test than in training (mean $p$ 0.018 vs 0.006). The new depot appears to **fail more often**, which a model trained on the old depot cannot know.
2. *Repair-cost sensors are histograms.* The cost drivers `s080–s085` belong to a hidden 10-bin histogram; raw bin counts mix "how much the truck was used" with "how its usage was distributed", which trees struggle to separate.
""")

md(r"""
## 6. Model 3: histogram-shape features + depot-shift correction

### 6a. Hidden histograms
A non-negative least-squares fit of the known histogram's total on all other sensors is **exact** and exposes five more 10-bin histograms. Each one's total is a fixed fraction of the reference total for every vehicle: they are histograms over a shared operating time. None is shifted in test.
""")

code(r"""
U = tr[H].sum(axis=1, min_count=10)
cand = [s for s in S if s not in H]
ratio = tr[cand].div(U.replace(0, np.nan), axis=0)
cand = [s for s in cand if (ratio[s].dropna() <= 1.0001).mean() > 0.999 and tr[s].notna().mean() > 0.8]
d = pd.concat([tr[cand], U.rename("U")], axis=1).dropna().sample(15000, random_state=0)
coef, rn = nnls(d[cand].to_numpy(float), d.U.to_numpy(float))
print(f"relative residual of the fit: {rn / np.linalg.norm(d.U):.6f}")
groups = pd.Series(coef, index=cand).round(3); groups = groups[groups > 0.05]
for v, g in groups.groupby(groups): print(f"  equal coefficient {v}: {list(g.index)}")

HIST = {a: [f"s{i:03d}" for i in range(a, a + 10)] for a in (23, 37, 54, 80, 129, 154)}
for a, g in HIST.items():
    r = (tr[g].sum(axis=1, min_count=10) / U).dropna()
    print(f"histogram s{a:03d}-s{a+9:03d}: total / reference total = {r.median():.3f} (5-95%: {r.quantile(.05):.3f}-{r.quantile(.95):.3f})")
""")

code(r"""
def shape(df):
    out = {"logU": np.log1p(df[HIST[54][:9]].sum(axis=1, min_count=9))}
    for a, g in HIST.items():
        g = g[:9] if a == 54 else g                     # test s063 is NaN after realignment
        x = df[g].to_numpy(float); tot = x.sum(1, keepdims=True)
        sh = np.where(tot > 0, x / np.where(tot > 0, tot, 1), np.nan); k = np.arange(len(g)); mean = (sh * k).sum(1)
        out.update({f"h{a}_sh{i}": sh[:, i] for i in range(len(g))})
        out[f"h{a}_mean"] = np.where(tot[:, 0] > 0, mean, np.nan)
        out[f"h{a}_std"] = np.where(tot[:, 0] > 0, np.sqrt(np.maximum((sh * k ** 2).sum(1) - mean ** 2, 0)), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"h{a}_ent"] = np.where(tot[:, 0] > 0, -np.nansum(np.where(sh > 0, sh * np.log(sh), 0), 1), np.nan)
        out[f"h{a}_lo3"], out[f"h{a}_hi3"] = sh[:, :3].sum(1), sh[:, -3:].sum(1)
    return pd.DataFrame(out, index=df.index)

Xdev = pd.concat([dev[S], shape(dev)], axis=1)
CM3 = {**CM, "n_estimators": 1000, "max_depth": 2}       # more capacity: removes shrinkage on expensive repairs
m3k1, r2_shape = [], []
for a, b in F:
    fa, fb = a[y[a] == 1], b[y[b] == 1]
    mb, resid = cost_model(CM3, Xdev.iloc[fa], np.log(c[fa]), Xdev.iloc[b])
    r2_shape.append(r2_score(np.log(c[fb]), mb[y[b] == 1]))
    fl, bu = decide_pv(Platt().fit(oof[a], y[a]).predict(oof[b]), mb, resid)
    m3k1.append(per_vehicle(c[b], fl, bu))
m3k1 = np.array(m3k1)
print(f"cost model R2: base features {np.mean(r2_base):.3f}  ->  + histogram shape {np.mean(r2_shape):.3f}   (per fold {np.round(np.array(r2_shape) - r2_base, 3)})")
print(f"Model 3 without shift correction: {m3k1.mean():.3f} ± {m3k1.std():.3f}   paired vs Model 2 {np.round(m3k1 - m2, 3)}")
""")

md(r"""
### 6b. Depot-shift (prior) correction

The training depot has no shift, so CV cannot measure this part; its evidence is below.

- **EM prior estimation** (Saerens et al., 2002) estimates the test failure rate from the test predictions alone. Validated on *simulated* shifts inside dev, it recovered the true odds factor almost exactly (true 4 → 3.9; true 6 → 5.9) and captured ~95% of the oracle's gain, costing only +0.009 when there was no shift (`src/v7_emsim.py`).
- On test it gives an odds factor **k ≈ 6.5** (5.7 even without the drifting sensors). EM assumes the sensors' meaning is unchanged, which the remaining covariate shift violates, so this is most likely an over-estimate.
- The public scores so far are best explained by **k ≈ 4**, and k = 4 also minimises the worst-case regret over k ∈ [1, 10]: under-correcting is far more expensive than over-correcting.

Each calibrated probability is shifted by the odds factor: $p' = kp / (1 - p + kp)$. `K_SHIFT` is the single setting chosen from this evidence and updated as Kaggle feedback arrives (see `SUBMISSIONS.md`).
""")

code(r"""
def em_odds_factor(p, pi_tr, it=300):
    pi = pi_tr
    for _ in range(it):
        a = pi / pi_tr * p
        pi = (a / (a + (1 - pi) / (1 - pi_tr) * (1 - p))).mean()
    return (pi / (1 - pi)) / (pi_tr / (1 - pi_tr))

shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
print(f"mean calibrated p: dev {pl.predict(oof).mean():.4f}   test {q_te.mean():.4f}")
print(f"EM odds factor on test: {em_odds_factor(q_te, y.mean()):.2f}")

K_SHIFT = 4
fail_X = pd.concat([fail_tr[S], shape(fail_tr)], axis=1)
m_te3, resid3 = cost_model(CM3, fail_X, lc_all, pd.concat([te[S], shape(te)], axis=1))
f3, b3 = decide_pv(shift(q_te, K_SHIFT), m_te3, resid3)
pd.DataFrame({"id": te.id, "flag": f3, "budget": b3.round(2)}).to_csv(OUT / "submission_model3.csv", index=False)
print(f"Model 3 test (k={K_SHIFT}): flagged {f3.mean():.2%}")
""")

md(r"""
## 7. Results across the three versions

| | CV cost / vehicle (5 folds) | Public score |
|---|---|---|
| Flag nobody (baseline) | 3.49 | n/a |
| Naive XGBoost (p > 0.5) | 2.50 | 74,933 |
| **Model 1** | 1.24 | 27,760 |
| **Model 2** | 0.90 | 22,969 |
| **Model 3** | 0.85 *(without the shift correction, which is test-only)* | *pending* |

Model 3's cost-model change is consistent (R² better in every fold), but its *total-cost* gain without the shift is within fold noise (paired differences mixed); across 3×5 repeated CV it was −0.025. Its main gain is expected on test, where the shift correction applies.

**Experiments that did not help** (all judged on paired CV cost, details in `reports/findings.md`): dropping the drifting sensors (public score worse), isotonic calibration, down-weighting short follow-up, missing-value count features, classifier tuning, 3-seed averaging, LightGBM and blends, histogram-shape features in the classifier, a vehicle-specific cost spread, importance weighting (effective sample size ~1%). Several of these *improved AUC or log-loss while making cost worse*: the cost depends on the few vehicles near the 2.5% flag threshold and on the expensive failures, not on overall ranking.

## 8. Capabilities and limitations of the best model

**Capabilities:** ranks failures well (AUC 0.984); repair cost is predictable from the sensors (R² ≈ 0.8 on log cost); budgets hit their target coverage (~91% of failures covered, as the cost ratio requires).

**Limitations:**
- Only 397 failures: fold-to-fold noise is large (±0.2 per vehicle) and gains below ~0.03 cannot be resolved.
- The depot-shift correction rests on one number, k, that cannot be validated on labelled data; it assumes the extra failures are spread proportionally over all vehicles.
- Labels are censored (short follow-up) and repair costs are simulated, which likely inflates how predictable cost looks.
- The histogram realignment and histogram structure are inferred from the data, not documented.

## 9. Individual contributions

See `CONTRIBUTIONS.md` and the commit history of each member.
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "fleet_maintenance.ipynb")
print("wrote fleet_maintenance.ipynb with", len(C), "cells")
