"""Builds the v13, v14 and v15 notebooks (variance reduction and the s120/s122 quantile mapping)."""
from nbkit import NB, SHAPE

POSTERIOR = r'''
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}                      # v9 cost model
DRIFT = ["s120", "s122", "s126", "s128"]; MAP = ["s120", "s122"]
keep = [s for s in S if s not in DRIFT]
f_all = tr[tr.failure == 1]; lc_all = np.log(f_all.repair_cost.to_numpy())
Xf, Xt = pd.concat([f_all[keep], shape(f_all)], axis=1), pd.concat([te[keep], shape(te)], axis=1)   # cost model: no drifted sensors (v12)
# posterior over the depot shift k from the v10 notebook (nine public scores), full precision
GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0]
W = [0.008172761879954695, 0.13116481847950048, 0.26721015776942314, 0.267056932957599, 0.1710924552535038, 0.0956936236265262,
     0.03459275261241694, 0.016549303552639335, 0.005725443007721319, 0.002368204370941116, 0.00031775367779660887,
     5.437858042612226e-05, 1.3770078232241008e-06, 3.722372808678639e-08]
pavg = lambda qq, lam=1.0: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))   # posterior-averaged shifted probability
'''

COSTBAG = r'''
NK = 8
m_seeds, mo = [], np.zeros(len(f_all))                                   # cost model: NK seeds, bagged inner out-of-fold residuals
for s in range(NK):
    prm = {**CM9, "random_state": SEED + s}
    mo += cross_val_predict(xgb.XGBRegressor(**prm), Xf, lc_all, cv=inner) / NK
    m_seeds.append(xgb.XGBRegressor(**prm).fit(Xf, lc_all).predict(Xt))
m_bag, resid_bag = np.mean(m_seeds, axis=0), lc_all - mo
'''

QMAP = r'''
def qmap(col):
    """Map each test value to the training value at the same quantile (rank-preserving)."""
    x = te[col]; r = x.rank(pct=True, method="average"); out = x.copy(); ok = x.notna()
    out[ok] = np.quantile(tr[col].dropna(), np.clip((r[ok] * ok.sum() - 0.5) / ok.sum(), 0, 1)); return out
te_map = te.copy()
for s in MAP: te_map[s] = qmap(s)
'''

BISECT = r'''
def match_flag_count(qq, m, resid, target):
    """Scale the shift (k -> lambda * k) until the number of flagged vehicles equals `target`."""
    lo, hi = 0.5, 4.0
    for _ in range(25):
        lam = (lo + hi) / 2
        if decide_pv(pavg(qq, lam), m, resid)[0].sum() < target: lo = lam
        else: hi = lam
    return lam
'''

# ----------------------------------------------------------------------------------------------- v13
n = NB("v13: Seed-averaging the models (variance reduction)")
n.header("v13", "`v13_bag` → `outputs/submission_v13_bag.csv` (single change vs `v12_robust`)",
         "**Question:** v12_robust beat v10_bayes by 509 public points. Is that a real gain from the cost-model inputs, or a lucky draw of "
         "XGBoost's random column/row sampling? A single fit's test predictions change as much between seeds as between the two versions "
         "we compared. We therefore average the classifier over 5 seeds and the repair-cost model over 8 seeds and change nothing else.")
n.code(SHAPE); n.code(POSTERIOR)
n.md("### 1. How large is seed-to-seed noise compared with the effect we were trying to measure?")
n.code(COSTBAG + r'''
dm = lambda a, b: float(np.abs(a - b).mean())
pairs = [dm(m_seeds[i], m_seeds[j]) for i in range(NK) for j in range(i + 1, NK)]
m_drift_all, _ = cost_model(CM9, pd.concat([f_all[S], shape(f_all)], axis=1), lc_all, pd.concat([te[S], shape(te)], axis=1))
print(f"mean abs difference of test log-cost predictions between two single fits (different seeds): {np.mean(pairs):.3f}")
print(f"... between the bag of {NK} and one single fit: {np.mean([dm(m_bag, ms) for ms in m_seeds]):.3f}")
print(f"... between a fit with and without the four drifted sensors (the v10 -> v12 change): {dm(m_drift_all, m_seeds[0]):.3f}")
''')
n.md("### 2. Classifier bag: out-of-fold probabilities for Platt calibration and test probabilities, averaged over 5 seeds")
n.code(r'''
NC = 5
oof, pte = np.zeros(len(dev)), np.zeros(len(te))
for s in range(NC):
    prm = {**PARAMS, "random_state": SEED + s}
    o = fit_oof(dev[S], prm); p = xgb.XGBClassifier(**prm).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
    if s == 0: o0, p0 = o, p                                           # seed 0 == the single-fit v2 classifier used before
    oof += o / NC; pte += p / NC
q, q1 = Platt().fit(oof, y).predict(pte), Platt().fit(o0, y).predict(p0)
print(f"mean calibrated test p: bag {q.mean():.4f} vs single fit {q1.mean():.4f}; correlation {np.corrcoef(q, q1)[0, 1]:.4f}")
print(f"OOF AUC bag {roc_auc_score(y, oof):.4f} vs single {roc_auc_score(y, o0):.4f}")
''')
n.md("### 3. Decisions: same posterior-averaged shift, newsvendor budgets and flag rule as `v12_robust`")
n.code(r'''
fl, bu = decide_pv(pavg(q), m_bag, resid_bag)
v12 = pd.read_csv("../submissions/submission_v12_robust.csv"); both = (v12.flag.to_numpy() == 1) & (fl == 1)
print(f"flag rate {fl.mean():.3%} (v12_robust {v12.flag.mean():.3%}); flags differing {int((v12.flag.to_numpy() != fl).sum())}; "
      f"budget change on commonly flagged: mean abs {np.abs(bu[both] - v12.budget.to_numpy()[both]).mean():.1f}")
write_submission(te.id, fl, bu, "v13_bag")
ref = pd.read_csv("../submissions/submission_v13_bag.csv")
assert (ref.flag.to_numpy() == fl).all() and np.allclose(ref.budget, bu.round(2), atol=0.011), "v13_bag not reproduced"
''')
n.md(r"""
## Results and decisions

- **Seed noise is as large as the effect we measured.** Two single cost-model fits differ by ≈ 0.133 in mean absolute log-cost on the test depot (a bag of 8 is 0.088 from a single fit), more than the ≈ 0.111 that dropping the four drifted sensors changed (see v12). A single public-score difference between a one-seed v10_bayes and a one-seed v12_robust therefore cannot be attributed to the sensors.
- **`v13_bag`** changes only the averaging: 86 flags differ from v12_robust, mean absolute budget change ≈ 59 on commonly flagged vehicles. The classifier bag changes the ranking little (correlation 0.996 with the single fit).
- **Public score 17,974.55 vs 16,839.86 for v12_robust (+1,135).** With the same model family and inputs, averaging made the public score worse. The honest reading: v12_robust's lead over v10_bayes was largely a favourable draw of one seed, not a robust gain, and public differences of a few hundred to ≈ 1,000 points between sibling files are within noise at ~6,000 public vehicles.
- **Decision:** keep seed-averaging as the default for any final candidate (lower variance is what we want on the private 70%), but do **not** read the v12 → v13 score gap as evidence that averaging hurts either; it is one more draw.

**Caveats**
- Averaging reduces model variance only; it cannot remove the bias from the depot shift or the unknown test cost level.
- The shift posterior is reused from v10 (nine scores).
""")
n.save("v13_seed_averaging.ipynb")

# ----------------------------------------------------------------------------------------------- v14
n = NB("v14: Rescale the instrument-like drifted sensors (quantile mapping)")
n.header("v14", "`v14_map` → `outputs/submission_v14_map.csv` (single change vs `v13_bag`, flag count matched)",
         "**Question:** four sensors (s120, s122, s126, s128) are distributed very differently at the new depot. We removed them from the "
         "repair-cost model in v12. The classifier still uses s120 and s122 (≈ 16% of its gain). If their drift is only an instrument/scale "
         "change, rescaling them onto the training distribution should make the classifier's test probabilities more trustworthy than leaving "
         "the shifted values in.")
n.code(SHAPE); n.code(POSTERIOR)
n.md("### 1. Which of the four drifted sensors look like a pure scale change?")
n.code(r'''
qs = [.05, .25, .5, .75, .95]
for s in DRIFT:
    a, b = np.quantile(tr[s].dropna(), qs), np.quantile(te[s].dropna(), qs)
    print(f"{s}: train {np.round(a, 1).tolist()}\n      test  {np.round(b, 1).tolist()}\n      ratio {np.round(b / np.where(a == 0, np.nan, a), 2).tolist()}")
''')
n.md(r"""
- **s122:** the test/train ratio is ≈ 2.5–2.8 at every quantile → a multiplicative change (units or calibration).
- **s120:** a floor of ≈ 16M in test, ≈ ×2 at the top → offset plus scale.
- **s126 / s128:** identical in the bulk (ratio 1.01–1.05); only the upper tail is heavier (×1.8 at the 95th percentile). That looks like genuine wear in a harsher depot, not an instrument change, and mapping it would erase a real signal. **We map only s120 and s122.**
""")
n.md("### 2. The v13 pipeline (seed-averaged), scoring test both with raw and with mapped s120/s122")
n.code(COSTBAG + r'''
NC = 5
oof, pr, pm, gain = np.zeros(len(dev)), np.zeros(len(te)), np.zeros(len(te)), 0.0
''' + QMAP + r'''
for s in range(NC):
    prm = {**PARAMS, "random_state": SEED + s}
    oof += fit_oof(dev[S], prm) / NC
    mdl = xgb.XGBClassifier(**prm).fit(tr[S], tr.failure)
    pr += mdl.predict_proba(te[S])[:, 1] / NC; pm += mdl.predict_proba(te_map[S])[:, 1] / NC
    g = pd.Series(mdl.get_booster().get_score(importance_type="total_gain")); gain += g.reindex(MAP).fillna(0).sum() / g.sum() / NC
pl = Platt().fit(oof, y); q, qm = pl.predict(pr), pl.predict(pm)
from scipy.stats import spearmanr
topk = int(0.088 * len(te)); ov = len(set(np.argsort(-q)[:topk]) & set(np.argsort(-qm)[:topk]))
print(f"classifier gain share of s120+s122: {gain:.3f}")
print(f"mean test p: raw {q.mean():.4f} mapped {qm.mean():.4f}; Spearman {spearmanr(q, qm)[0]:.4f}; overlap of the top-{topk}: {ov}")
''')
n.md("### 3. Decisions: the v13 baseline, then the mapped version with the *number of flagged vehicles matched*")
n.code(BISECT + r'''
fl13, bu13 = decide_pv(pavg(q), m_bag, resid_bag)
r13 = pd.read_csv("../submissions/submission_v13_bag.csv")
assert (r13.flag.to_numpy() == fl13).all() and np.allclose(r13.budget, bu13.round(2), atol=0.011), "v13_bag not reproduced"
target = int(fl13.sum()); lam = match_flag_count(qm, m_bag, resid_bag, target)
fl, bu = decide_pv(pavg(qm, lam), m_bag, resid_bag); both = (fl13 == 1) & (fl == 1)
print(f"lambda {lam:.3f} (mean p {pavg(qm, lam).mean():.4f} vs v13 {pavg(q).mean():.4f}); flags {int(fl.sum())} (v13 {target}); "
      f"differing {int((fl13 != fl).sum())}; budget mean abs change on commonly flagged {np.abs(bu[both] - bu13[both]).mean():.1f}")
write_submission(te.id, fl, bu, "v14_map")
ref = pd.read_csv("../submissions/submission_v14_map.csv")
assert (ref.flag.to_numpy() == fl).all() and np.allclose(ref.budget, bu.round(2), atol=0.011), "v14_map not reproduced"
''')
n.md(r"""
## Results and decisions

- Mapping lowers the mean test probability (0.0203 → 0.0183), keeps the ranking mostly intact (Spearman 0.986, 1,670 of the top 1,760 unchanged) and changes 222 flags.
- **Why match the flag count:** mapping lowers probabilities, which alone would cut the flag rate to 7.9%. Flag *level* moves the score far more than ranking (k=1 → k=4 moved it ≈ 2,500), so we scale k by λ = 1.72 to flag exactly as many vehicles as v13_bag. The comparison is then about *which* vehicles, not how many. It is only approximate: matching raised the mean probability by ≈ 10%, so budgets rise slightly too.
- **Public score 17,375.65 vs 17,974.55 for v13_bag (−599).** In the direction the instrument-shift idea predicts. See v15: the same mapping on the v12_robust base gave **+218**, so the two effects have opposite signs and the mapping's effect on the public score is inconclusive.

**Caveats**
- Assumes the s120/s122 differences are purely instrumental. Any genuine wear difference carried by those sensors is removed.
- No labels at the new depot; only the public score can judge it, with ≈ 600-point noise between sibling files.
""")
n.save("v14_quantile_map.ipynb")

# ----------------------------------------------------------------------------------------------- v15
n = NB("v15: The s120/s122 mapping on the v12_robust base (replication)")
n.header("v15", "`v15_map_v12` → `outputs/submission_v15_map_v12.csv` (single change vs `v12_robust`, flag count matched)",
         "**Question:** v14 gave −599 on top of the seed-averaged v13. Is that a property of the mapping or of that particular base? We "
         "repeat the same change on the best-scoring base, `v12_robust` (single-seed classifier and cost model), as a replication.")
n.code(SHAPE); n.code(POSTERIOR); n.code(QMAP)
n.md("### 1. v12_robust base, scored with raw and with mapped s120/s122")
n.code(r'''
Pl = Platt().fit(fit_oof(dev[S]), y)
mdl = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
q, qm = Pl.predict(mdl.predict_proba(te[S])[:, 1]), Pl.predict(mdl.predict_proba(te_map[S])[:, 1])
m, resid = cost_model(CM9, Xf, lc_all, Xt)
fl12, bu12 = decide_pv(pavg(q), m, resid)
r12 = pd.read_csv("../submissions/submission_v12_robust.csv")
assert (r12.flag.to_numpy() == fl12).all() and np.allclose(r12.budget, bu12.round(2), atol=0.01), "v12_robust not reproduced"
print(f"mean test p raw {q.mean():.4f}, mapped {qm.mean():.4f}")
''')
n.md("### 2. Decisions with the flag count matched to v12_robust")
n.code(BISECT + r'''
target = int(fl12.sum()); lam = match_flag_count(qm, m, resid, target)
fl, bu = decide_pv(pavg(qm, lam), m, resid); both = (fl12 == 1) & (fl == 1)
print(f"lambda {lam:.3f} (mean p {pavg(qm, lam).mean():.4f} vs v12 {pavg(q).mean():.4f}); flags {int(fl.sum())} (v12 {target}); "
      f"differing {int((fl12 != fl).sum())}; budget mean abs change on commonly flagged {np.abs(bu[both] - bu12[both]).mean():.1f}")
write_submission(te.id, fl, bu, "v15_map_v12")
ref = pd.read_csv("../submissions/submission_v15_map_v12.csv")
assert (ref.flag.to_numpy() == fl).all() and np.allclose(ref.budget, bu.round(2), atol=0.01), "v15 not reproduced"
''')
n.md(r"""
## Results and decisions

- 269 flags differ from v12_robust; mean absolute budget change ≈ 87 on commonly flagged vehicles; 1,760 flags (v12: 1,761).
- **Public score 17,058.28 vs 16,839.86 for v12_robust (+218).** The same change on the seed-averaged base was −599. Opposite signs, both within the ≈ 600-point noise of sibling files.
- **Decision:** the s120/s122 mapping has **no reliable effect** on the public score, so it is not part of the pipeline. We keep the v12 cost-model inputs and drop the mapping from further candidates; the analysis stays in the report as a hypothesis that the data neither supports nor refutes.
- **Methodological lesson:** a single public-score comparison on 30% of test (≈ 6,000 vehicles, ~1% failures) cannot resolve changes below a few hundred points. Level changes (k, budget scale) move the score by thousands and are the only ones we can read reliably.

**Caveats:** same as v14 (assumes the drift is purely instrumental; approximate isolation because flag-count matching shifts mean probabilities).
""")
n.save("v15_quantile_map_v12_base.ipynb")
