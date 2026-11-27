"""Builds the notebooks for uploads v18, v22, v24, v29, v35, v37 and v42.
Each notebook is self-contained in the sense that all code that makes the CSV is in the notebook; it loads the two data files plus the
saved artifacts under ../reports/ that it names (rankings, the public-score posterior, out-of-fold predictions)."""
from pathlib import Path

from nbkit import NB

GP_CLASS = (Path(__file__).resolve().parents[1] / "src" / "v33_hist_gp.py").read_text()   # inlined verbatim (written by Daksh183, v33)

BASE_ART = r'''
REP, SUB = Path("../reports"), Path("../submissions")
qv12, m = np.load(REP / "v18_qm.npy"); resid = np.load(REP / "v18_resid.npy")     # v12 test probabilities, tree log-cost, its OOF residuals
GRID, W = np.load(REP / "v10_posterior.npy")                                       # v10 posterior over the prior-shift factor k
pavg = lambda q, lam=1.0: sum(w * shift(q, k * lam) for w, k in zip(W, GRID))

def match_flags(q, target, lo, hi, iters):
    """odds multiplier lambda such that the posterior-averaged rule flags `target` vehicles"""
    for _ in range(iters):
        lam = (lo + hi) / 2
        if decide_pv(pavg(q, lam), m, resid)[0].sum() < target: lo = lam
        else: hi = lam
    return lam, hi
'''

# ------------------------------------------------------------------ v18
n = NB("v18: censoring-aware classifier ensemble")
n.header("v18", "#18, `submission_v18_ens10.csv`, 24 Nov, public 17,624",
         "**Question.** Healthy-labelled vehicles followed for less than a year may hide failures. Does a classifier that corrects for "
         "this censoring (inverse-probability-of-censoring weights, ten XGBoost variants averaged) rank test vehicles better than v12's? "
         "The flag count is matched to v12, so only the ranking changes.\n\n**Artifacts used:** `reports/v18_qm.npy`, `v18_resid.npy`, "
         "`v10_posterior.npy`, and the variants' out-of-fold predictions `reports/v18_search/ipcw*.npy` (3 x 5-fold, made by `src/v18_search.py`).")
n.code(BASE_ART)
n.code(r'''
def km_censor(t, y, H=365):
    """G(s) = P(not censored before s); the censoring event is 'healthy with follow-up below H days'."""
    cens = (y == 0) & (t < H); G = 1.0; out = {}
    for s in np.unique(t[cens]):
        G *= 1 - ((t == s) & cens).sum() / (t >= s).sum(); out[s] = G
    keys = np.array(list(out)); vals = np.array(list(out.values()))
    return lambda u: np.where(u <= keys[0], 1.0, vals[np.searchsorted(keys, u, side="right") - 1].clip(min=0))

SNO = [s for s in S if s not in ["s120", "s122", "s126", "s128"]]
D6 = {"max_depth": 6, "min_child_weight": 5}; SL = {"n_estimators": 1000, "learning_rate": 0.02, "colsample_bytree": 0.5}
VAR = {"ipcw": ({}, S), "ipcw_nodrift": ({}, SNO), "ipcw_d6_mcw5": (D6, S), "ipcw_slow": (SL, S), "ipcw_nodrift_d6": (D6, SNO),
       "ipcw_nodrift_slow": (SL, SNO), "ipcw_d6_slow": ({**D6, **SL}, S), "ipcw_nodrift_d6_slow": ({**D6, **SL}, SNO),
       "ipcw_s1": ({"random_state": 1}, S), "ipcw_nodrift_s1": ({"random_state": 1}, SNO)}
Y, T, t = tr.failure.to_numpy(), tr.followup_days.to_numpy(), dev.followup_days.to_numpy()
G = km_censor(T, Y); cm = (Y == 1) | (T >= 365)                                    # vehicles whose one-year outcome is known
w = 1 / G(np.where(Y == 1, np.minimum(T, 365) - 1, 364)[cm])
pte = np.mean([xgb.XGBClassifier(**{**PARAMS, **kw}).fit(tr.loc[cm, cols], Y[cm], sample_weight=w).predict_proba(te[cols])[:, 1]
               for kw, cols in VAR.values()], axis=0)
complete = (y == 1) | (t >= 365)
oof = np.mean([np.load(REP / f"v18_search/{k}.npy")[0] for k in VAR], axis=0)
q = Platt().fit(oof[complete], y[complete]).predict(pte)
target = int(pd.read_csv(SUB / "submission_v12_robust.csv").flag.sum())
lam, _ = match_flags(q, target, 0.2, 4.0, 30)
fl, bu = decide_pv(pavg(q, lam), m, resid)
print(f"{int(fl.sum())} flags (v12 {target}), lambda {lam:.3f}")
write_submission(te.id, fl, bu, "v18_ens10")
''')
n.md("**Result.** Better in cross-validation on complete cases, worse on the public board (17,624 against v12's 16,840). The test depot's "
     "labels behave as if censored like the training labels, so the correction was dropped.")
n.save("v18_censoring_ensemble.ipynb")

# ------------------------------------------------------------------ v22
n = NB("v22: classifier without the two mis-scaled depot sensors")
n.header("v22", "`submission_v22_nos122.csv`, 25 Nov, public 17,172.57",
         "**Question.** s120 (offset) and s122 (scale about x2.55) read differently at the test depot. Does a classifier without them rank "
         "better there? Flag count matched to v12.\n\n**Artifacts used:** `reports/v18_qm.npy`, `v18_resid.npy`, `v10_posterior.npy`.")
n.code(BASE_ART)
n.code(r'''
cols = [s for s in S if s not in ("s063", "s120", "s122")]
oof = np.zeros(len(dev))
for a, b in F: oof[b] = xgb.XGBClassifier(**PARAMS).fit(dev.loc[a, cols], y[a]).predict_proba(dev.loc[b, cols])[:, 1]
raw = xgb.XGBClassifier(**PARAMS).fit(tr[cols], tr.failure).predict_proba(te[cols])[:, 1]
q = Platt().fit(oof, y).predict(raw)
target = int(decide_pv(pavg(qv12), m, resid)[0].sum())
_, hi = match_flags(q, target, 0.2, 8.0, 40)
fl, bu = decide_pv(pavg(q, hi), m, resid); bu = np.minimum(bu, 8000)
print(f"{int(fl.sum())} flags (v12 {target}), lambda {hi:.3f}")
write_submission(te.id, fl, bu, "v22_nos122")
''')
n.md("**Result.** Neutral to better in cross-validation, worse on the public board (17,172.57). On its own the sensor set is not the lever; "
     "it is kept as one of the ranking hypotheses that later versions weigh.")
n.save("v22_drop_depot_sensors.ipynb")

# ------------------------------------------------------------------ v24
n = NB("v24: soft labels for censored vehicles")
n.header("v24", "`submission_v24_pu.csv`, 25 Nov, public 17,575",
         "**Question.** Instead of dropping censored healthy vehicles (v18), give each a soft label: the failure probability from a model "
         "trained only on vehicles with a known one-year outcome. Flag count matched to v12.\n\n**Artifacts used:** `reports/v18_qm.npy`, "
         "`v18_resid.npy`, `v10_posterior.npy`.")
n.code(BASE_ART)
n.code(r'''
def fit_pu(D):
    yy = D.failure.to_numpy(); t = D.followup_days.to_numpy(); cens = (yy == 0) & (t < 365)
    A = xgb.XGBClassifier(**PARAMS).fit(D.loc[~cens, S], yy[~cens])
    pi = np.clip(A.predict_proba(D.loc[cens, S])[:, 1], 0, 1)
    X = pd.concat([D.loc[~cens, S], D.loc[cens, S], D.loc[cens, S]])             # censored rows twice: weight pi as failed, 1 - pi as healthy
    Y = np.r_[yy[~cens], np.ones(cens.sum()), np.zeros(cens.sum())]
    Wt = np.r_[np.ones((~cens).sum()), pi, 1 - pi]
    return xgb.XGBClassifier(**PARAMS).fit(X, Y, sample_weight=Wt), pi.sum()

oof = np.zeros(len(dev))
for a, b in F: oof[b] = fit_pu(dev.loc[a].reset_index(drop=True))[0].predict_proba(dev.loc[b, S])[:, 1]
mdl, hidden = fit_pu(tr); print(f"expected hidden failures in train: {hidden:.1f} (recorded {int(tr.failure.sum())})")
q = Platt().fit(oof, y).predict(mdl.predict_proba(te[S])[:, 1])
target = int(decide_pv(pavg(qv12), m, resid)[0].sum())
_, hi = match_flags(q, target, 0.1, 8.0, 40)
fl, bu = decide_pv(pavg(q, hi), m, resid); bu = np.minimum(bu, 8000)
print(f"{int(fl.sum())} flags, lambda {hi:.3f}")
write_submission(te.id, fl, bu, "v24_pu")
''')
n.md("**Result.** Worse on the public board (17,575). Second independent sign that correcting for censoring does not help at the test depot.")
n.save("v24_soft_labels.ipynb")

# ------------------------------------------------------------------ harness pieces
RANKS = r'''
REP, SUB = Path("../reports"), Path("../submissions")
qv12, m = np.load(REP / "v18_qm.npy"); resid = np.load(REP / "v18_resid.npy")
sig = lambda v: 1 / (1 + np.exp(-v)); logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
def zs(x): x = np.log1p(np.nan_to_num(np.maximum(x, 0), nan=np.nanmedian(x))); return (x - x.mean()) / x.std()
v19 = np.load(REP / "v19_q.npy"); H21 = np.load(REP / "v21_H.npy")
te_map = te.copy()
for col in ("s120", "s122"):                                                        # quantile-map the two test sensors onto train
    r = te[col].rank(pct=True); te_map[col] = np.nanquantile(tr[col], r.fillna(0.5).clip(0, 1)); te_map.loc[te[col].isna(), col] = np.nan
mdl = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
q_map = Platt().fit(np.load(REP / "p_oof_v2.npy"), y).predict(mdl.predict_proba(te_map[S])[:, 1])
RANK = {"v12": qv12, "nos122": np.load(REP / "v22_q.npy"), "pu": np.load(REP / "v24_q.npy"), "pu_x2": np.load(REP / "v24_pu_x2_q.npy"),
        "ens10": np.load(REP / "v18_ens10_q.npy"), "ens4": np.load(REP / "v18_ens_q.npy"), "fix120": v19[0], "fix": v19[1],
        "nodrift": H21[2], "map": q_map}                                            # ten ranking hypotheses
RANK = {k: logit(np.ravel(v)[-len(te):]) for k, v in RANK.items()}
Z = {"none": np.zeros(len(te)), **{s: zs(te[s].to_numpy()) for s in ("s120", "s122", "s126", "s149")}}   # wear scores

def prob(r):
    """failure probabilities of one 'truth': blend of rankings, wear tilts, confidence slope b, level a"""
    g = lambda k, d: r[k] if k in r and isinstance(r[k], (int, float, str)) and r[k] == r[k] else d
    w2, R2, c2, z2 = g("w2", 0.0), g("R2", "v12"), g("c2", 0.0), g("z2", "none")
    L = (1 - r["w"] - w2) * RANK["v12"] + r["w"] * RANK[r["R"]] + w2 * RANK[R2]
    return r["fl0"] + (1 - r["fl0"]) * sig(r["a"] + r["b"] * L + r["c"] * Z[r["z"]] + c2 * Z[z2])
ALPHAS = (1.0, 1.15, 1.3, 1.45); Kd = 99; grid = (np.arange(Kd) + .5) / Kd          # repair-cost level multipliers, 99 cost draws
'''

DECIDE = r'''
def posterior_means(T):
    """posterior-weighted failure probability and weight, per cost level alpha"""
    T = T.copy(); T.wt /= T.wt.sum()
    PB = {al: sum((r.wt * prob(r._asdict()) for r in T[T.alpha == al].itertuples()), np.zeros(len(te))) for al in ALPHAS}
    return PB, {al: T[T.alpha == al].wt.sum() for al in ALPHAS}

def bayes_decide(PB, Wa, draws):
    """flag and budget that minimise posterior-expected cost; budgets capped at 8,000 (the largest repair cost seen)"""
    MX = {al: np.clip(draws * al, 5, 8000) for al in ALPHAS}
    wsum = sum(Wa.values()); p = sum(PB.values()) / wsum
    Mx = np.concatenate([MX[al] for al in ALPHAS if Wa[al] > 0], 1)
    wcol = np.concatenate([PB[al][:, None] / np.maximum(p[:, None] * wsum, 1e-12) * np.ones((1, Kd)) / Kd for al in ALPHAS if Wa[al] > 0], 1)
    o = np.argsort(Mx, 1); Ms = np.take_along_axis(Mx, o, 1); cw = np.cumsum(np.take_along_axis(wcol, o, 1), 1)
    Q = 1 - (1 - Q_STAR) / np.maximum(p, 1e-9)
    idx = np.clip((cw < np.clip(Q, 0, 1)[:, None]).sum(1), 0, Ms.shape[1] - 1)
    b = np.minimum(np.where(Q > 0, Ms[np.arange(len(p)), idx], 0.0), 8000)
    fail = ((UNDER * np.maximum(Mx - b[:, None], 0) + OVER * np.maximum(b[:, None] - Mx, 0)) * wcol).sum(1); mu = (Mx * wcol).sum(1)
    fl = (INSPECT + p * fail + (1 - p) * OVER * b < p * mu).astype(int); return fl, np.where(fl == 1, b, 0.0)

def load_posterior(path):
    P = pd.read_csv(REP / path)
    print(f"{len(P)} truths; weight by wear tilt:", P.groupby("z").wt.sum().round(3).to_dict())
    return P[P.wt > 1e-4]
'''

HARNESS_TEXT = ("**The harness.** A family of hypotheses about the test depot (which classifier ranking is right, how failure odds tilt "
                "with wear, overall level, repair-cost level) is weighted by the exact likelihood of the public scores of our earlier "
                "uploads. The decision then minimises posterior-expected cost. ")
ART = ("`reports/v18_qm.npy`, `v18_resid.npy`, `v19_q.npy`, `v21_H.npy`, `v22_q.npy`, `v24_q.npy`, `v24_pu_x2_q.npy`, "
       "`v18_ens10_q.npy`, `v18_ens_q.npy`, `p_oof_v2.npy`")

# ------------------------------------------------------------------ v29
n = NB("v29: Bayes decision under a posterior fitted to the public scores")
n.header("v29", "`submission_v29_bayes.csv`, 25 Nov, public 17,196",
         HARNESS_TEXT + "Repair costs come from v12's tree model.\n\n**Artifacts used:** " + ART + ", and the posterior "
         "`reports/v29_post.csv` (20 public scores; `python src/v28_harness.py`, then `python src/v29_harness.py`).")
n.code(RANKS); n.code(DECIDE)
n.code(r'''
P = load_posterior("v29_post.csv")
draws = np.exp(m[:, None] + np.quantile(resid, grid)[None, :])
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v29_bayes")
''')
n.md("**Result.** 17,196 public, worse than v12. The posterior was only as good as its repair-cost model; it improved once the cost model did (v35 onward).")
n.save("v29_harness_bayes.ipynb")

# ------------------------------------------------------------------ v35
n = NB("v35: harness flags with Gaussian-process repair budgets")
n.header("v35", "`submission_v35_combo.csv`, 27 Nov, public 16,878",
         "**Question.** Daksh's v33 Gaussian-process repair-cost model scored well. Does it help when its budgets are put on the harness's "
         "flags (those posterior truths that include the no-s122 ranking)?\n\n**Artifacts used:** " + ART + ", and `reports/v30_post.csv` "
         "(21 public scores; `python src/v30_harness.py`). The Gaussian-process class is the one from v33 (`src/v33_hist_gp.py`, by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS)
n.code(r'''
P = load_posterior("v30_post.csv")
TN = P[(P.R == "nos122") | (P.R2 == "nos122")].copy()
PB, Wa = posterior_means(TN)
fl, _ = bayes_decide(PB, Wa, np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))     # flags: tree costs
ph = sum(PB.values()) / sum(Wa.values()); alpha = sum(al * w for al, w in Wa.items()) / sum(Wa.values())

f = tr[tr.failure == 1]; ly = np.log(f.repair_cost.to_numpy()); X = extract(f)
oof = np.full(len(ly), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(ly):
    oof[b] = GaussianSeverity().fit(select(X, a), ly[a]).predict(select(X, b))
ok = np.isfinite(oof); gres = ly[ok] - oof[ok]
sev = GaussianSeverity().fit(X, ly).predict(extract(te)); sup = np.isfinite(sev)
print(f"GP out-of-fold R2 {1 - (gres ** 2).sum() / ((ly[ok] - ly[ok].mean()) ** 2).sum():.3f}; test vehicles supported {sup.sum()}")

Q = 1 - (1 - Q_STAR) / np.maximum(ph, 1e-9); lev = np.clip(Q, 0, 1)                      # newsvendor level per vehicle
bq = np.where(sup, np.exp(np.where(sup, sev, 0) + np.quantile(gres, lev)), np.exp(m + np.quantile(resid, lev)))
bu = np.where(fl == 1, np.where(Q > 0, np.minimum(bq * alpha, 8000), 0.0), 0.0)
print(f"{int(fl.sum())} flags, cost level {alpha:.3f}, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v35_combo")
''')
n.md("**Result.** 16,878 public. Mixing flags from one cost model with budgets from another is weaker than deciding both under the same costs (v37).")
n.save("v35_gp_budget_combo.ipynb")

# ------------------------------------------------------------------ v37 / v42
GPFIT = r'''
WEAR = ["s126", "s149", "s123"]
f = tr[tr.failure == 1].reset_index(drop=True); ly = np.log(f.repair_cost.to_numpy())
def zfit(cols):
    X = np.log1p(np.maximum(f[cols].to_numpy(float), 0)); return np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
ZW, ZS = zfit(WEAR), zfit(SENS)
def feats(d):
    zw = (np.log1p(np.maximum(d[WEAR].to_numpy(float), 0)) - ZW[0]) / ZW[1]
    zz = (np.log1p(np.maximum(d[SENS].to_numpy(float), 0)) - ZS[0]) / ZS[1]
    return extract(d) + [zw, zz]
X = feats(f); oof = np.full(len(ly), np.nan)
for a, b in KFold(5, shuffle=True, random_state=42).split(ly):
    oof[b] = GaussianSeverity().fit(select(X, a), ly[a]).predict(select(X, b))
ok = np.isfinite(oof); gres = ly[ok] - oof[ok]
sev = GaussianSeverity().fit(X, ly).predict(feats(te)); sup = np.isfinite(sev)
print(f"GP out-of-fold R2 {1 - (gres ** 2).sum() / ((ly[ok] - ly[ok].mean()) ** 2).sum():.4f}; residual sd {gres.std():.3f}; test supported {sup.sum()}")
'''

n = NB("v37: Gaussian-process repair costs with wear and cost sensors, used for flags and budgets")
n.header("v37", "`submission_v37_bayes_gp2.csv`, 27 Nov, public 16,410",
         "**Question.** The GP of v33 sees only the six histograms. Adding a wear group (s126, s149, s123) and six cost-relevant sensors, "
         "does the joint flag-and-budget decision improve?\n\n" + HARNESS_TEXT + "\n\n**Artifacts used:** " + ART + ", and "
         "`reports/v36_post.csv` (24 public scores; `python src/v36_harness.py`). GP class from v33 (by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS)
n.code('SENS = ["s049", "s116", "s142", "s105", "s010", "s104"]\n' + GPFIT.strip())
n.code(r'''
P = load_posterior("v36_post.csv")
draws = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + np.quantile(gres, grid)[None, :]),    # GP where supported
                 np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))                                    # tree model elsewhere
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v37_bayes_gp2")
''')
n.md("**Result.** 16,410 public, the best public score at the time. The repair-cost model, not the classifier, was the lever.")
n.save("v37_gp_sensor_costs.ipynb")

n = NB("v42: repair-cost spread that depends on the predicted cost")
n.header("v42", "`submission_v42_gp_bands.csv`, 27 Nov, public 16,487",
         "**Question.** v37 gives every vehicle the same spread of cost errors. Using a re-selected sensor group (s116, s142, s105, s104, "
         "s078) and taking residual quantiles from three predicted-cost bands, honest cross-fitted budget loss fell by 6 to 8% on three "
         "fold seeds (`src/v39_costsearch.py`, `src/v42b_combined.py`).\n\n" + HARNESS_TEXT + "\n\n**Artifacts used:** " + ART +
         ", and `reports/v38_post.csv` (25 public scores; `python src/v38_harness.py`). GP class from v33 (by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS)
n.code('SENS = ["s116", "s142", "s105", "s104", "s078"]\n' + GPFIT.strip())
n.code(r'''
P = load_posterior("v38_post.csv")
mu_oof = oof[ok]; edges = np.quantile(mu_oof, [1 / 3, 2 / 3])
band_of_failure = np.searchsorted(edges, mu_oof); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
QB = np.stack([np.quantile(gres[band_of_failure == j], grid) for j in range(3)])
draws = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v42_gp_bands")
''')
n.md("**Result.** 16,487 public, 77 above v37. The two files differ mostly in budgets, which only the few failing public vehicles feel; "
     "the noise of that difference is about 450, so the public board cannot separate them.")
n.save("v42_gp_banded_spread.ipynb")

# ------------------------------------------------------------------ v51
n = NB("v51: repair-cost model with fewer inputs")
n.header("v51", "`submission_v51_gp_reduced.csv`, #27, 27 Nov, public 16,388",
         "**Question.** Do all inputs of v42's Gaussian process earn their place? Removing one input at a time showed that the usage "
         "(exposure) group and the s023 and s129 histograms make budgets worse. This model drops them, and averages the fits with and "
         "without wear sensor s123 (`src/v47_ablation.py` to `src/v52_flagcv.py`). Honest cross-fitted budget loss fell about 7% "
         "(10 of 10 fresh fold seeds); end-to-end shift-aware CV cost per vehicle fell 2.1% (18 of 25 folds). The removed inputs were "
         "chosen on the same 397 failures, so the real gain is probably smaller.\n\n" + HARNESS_TEXT + "\n\n**Artifacts used:** " + ART +
         ", and `reports/v51_post.csv` (26 public scores; `python src/v51_harness.py`). GP class from v33 (by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS)
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
# rows the Gaussian process may price (as in v42): enough of the original blocks observed
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
''')
n.md("**Evidence.** Better than v42 in both cross-validation checks. The public board cannot confirm it: the posterior-expected difference "
     "to v42 is under 100 points on the public 30%, far below the noise between two such files. It then scored 16,388 public, 99 below v42 and the best "
     "public score so far; that gap is also within the noise.")
n.save("v51_gp_reduced_inputs.ipynb")

# ------------------------------------------------------------------ v66
GP51 = [c for c in n.cells if "class GPAll" in c.source][0].source                      # the v51 repair-cost fit, reused verbatim
n = NB("v66: survival-model failure rankings added to the harness")
n.header("v66", "`submission_v66_aft.csv`, built 27 Nov (see `../SUBMISSIONS.md` for whether it was uploaded)",
         "**Question.** Does a survival model, which learns time to failure from `followup_days` (training target only, never an input), "
         "rank vehicles better than the failure classifier? On fresh folds with validation weighted to the test depot's wear, XGBoost "
         "accelerated-failure-time models cost up to 2.6% less per vehicle (`src/v63_survival.py`, `v64_aft.py`); on the locked 20% "
         "hold-out their ensemble cost 2.7% less and the depth-6 member 6.6% less (`src/v65_aft_holdout.py`, 79 failures).\n\n" + HARNESS_TEXT +
         "Here the survival rankings join the family, and hypotheses that use them get four times the prior weight, because the "
         "cross-validation evidence for them is not part of the public-score likelihood.\n\n**Artifacts used:** " + ART +
         ", `reports/v61_q.npz`, `v65_q.npz`, and `reports/v66_post.csv` (27 public scores; `python src/v66_harness.py`). Repair costs "
         "as in v51. GP class from v33 (by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS); n.code(GP51)
n.code(r'''
AFT = ["aft", "aftd6", "clfaft"]
for path in ("v61_q.npz", "v65_q.npz"):
    z_ = np.load(REP / path); RANK.update({k: logit(z_[k]) for k in z_.files})
P = load_posterior("v66_post.csv").copy(); P.wt /= P.wt.sum()
P.loc[P.R.isin(AFT) | P.R2.isin(AFT), "wt"] *= 4; P.wt /= P.wt.sum()
mu_oof = oof[ok]; edges = np.quantile(mu_oof, [1 / 3, 2 / 3])
band_of_failure = np.searchsorted(edges, mu_oof); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
QB = np.stack([np.quantile(gres[band_of_failure == j], grid) for j in range(3)])
draws = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v66_aft")
''')
n.md("**Evidence.** The cross-validation and hold-out results favour the survival rankings. The public-score posterior gives them modest "
     "weight, and expects this file to differ from v51 by only a few tens of points on the public 30%.")
n.save("v66_survival_rankings.ipynb")

# ------------------------------------------------------------------ v69
n = NB("v69: survival-model ranking replaces the classifier")
n.header("v69", "`submission_v69_aftfull.csv`, built 27 Nov (see `../SUBMISSIONS.md` for whether it was uploaded)",
         "**Question.** v66 gave the survival rankings only part of the posterior weight. What if every hypothesis uses a survival "
         "ranking? Seven validation setups (`src/v67_valmethods.py`: plain folds, wear-weighted folds, completely followed vehicles, "
         "instrument-fault scenario, clustered folds, hold-out of the most worn and of the most used) all put the survival ensemble "
         "ahead of the classifier, by 2.4% to 11.8% of cost per vehicle. A check of those setups against five earlier ranking changes "
         "found that none of them predicts public results reliably.\n\n" + HARNESS_TEXT +
         "Here the family holds only the three survival rankings, and level, confidence, wear tilts and repair-cost level are fitted "
         "to the public scores.\n\n**Artifacts used:** " + ART +
         ", `reports/v61_q.npz`, `v65_q.npz`, and `reports/v69_post.csv` (27 public scores; `python src/v69_aftfull.py`). Repair costs "
         "as in v51. GP class from v33 (by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS); n.code(GP51)
n.code(r'''
for path in ("v61_q.npz", "v65_q.npz"):
    z_ = np.load(REP / path); RANK.update({k: logit(z_[k]) for k in z_.files})
P = load_posterior("v69_post.csv")
mu_oof = oof[ok]; edges = np.quantile(mu_oof, [1 / 3, 2 / 3])
band_of_failure = np.searchsorted(edges, mu_oof); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
QB = np.stack([np.quantile(gres[band_of_failure == j], grid) for j in range(3)])
draws = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v69_aftfull")
''')
n.md("**Evidence.** The offline setups favour the survival rankings. The public scores do not separate them from the classifier: the "
     "best survival-only hypothesis fits as well as the best of all, but under the full v66 posterior this file is expected to cost about "
     "130 more than v51 on the public 30%.")
n.save("v69_aftfull.ipynb")

# ------------------------------------------------------------------ v82
n = NB("v82: Bayes decision hedged over every ranking family")
n.header("v82", "`submission_v82_hedge.csv`, built 27 Nov (see `../SUBMISSIONS.md` for whether it was uploaded)",
         "**Question.** The failure rankings disagree a lot on the test depot (CatBoost against the v12 classifier: Spearman 0.58), and "
         "neither validation nor the locked hold-out nor the public scores can say which is right. The safest decision averages over all "
         "of them. This adds two ranking families to v66's: CatBoost without s120/s122, alone and averaged with the survival model "
         "(`src/v72_newmodels.py`, `v73_catboost.py`), and survival models trained with failures weighted by sqrt(repair cost) "
         "(`src/v80_costweight.py`, `v81_aftcw.py`).\n\n" + HARNESS_TEXT +
         "Every hypothesis is weighted only by the public-score likelihood (no extra prior).\n\n**Artifacts used:** " + ART +
         ", `reports/v61_q.npz`, `v65_q.npz`, `v73_q.npz`, `v81_q.npz`, and `reports/v82all_post.csv` (27 public scores; "
         "`python src/v66_harness.py`, `v74_harness_cat.py`, `v82_harness_aftcw.py`, `v82_hedge.py`). Repair costs as in v51. "
         "GP class from v33 (by Daksh183).")
n.code(RANKS); n.code(DECIDE); n.code(GP_CLASS); n.code(GP51)
n.code(r'''
for path in ("v61_q.npz", "v65_q.npz", "v73_q.npz", "v81_q.npz"):
    z_ = np.load(REP / path); RANK.update({k: logit(z_[k]) for k in z_.files})
P = pd.read_csv(REP / "v82all_post.csv"); P = P[P.wt > 1e-5].copy(); P.wt /= P.wt.sum()
print(f"{len(P)} hypotheses carry the posterior; weight by ranking family:",
      P.assign(f=np.where(P.R.str.startswith("catNO"), "catboost", np.where(P.R.str.startswith("aft") | P.R.isin(["clfaft"]) | P.R2.isin(["aft", "aftd6", "clfaft"]), "survival", "classifier"))).groupby("f").wt.sum().round(3).to_dict())
mu_oof = oof[ok]; edges = np.quantile(mu_oof, [1 / 3, 2 / 3])
band_of_failure = np.searchsorted(edges, mu_oof); band = np.searchsorted(edges, np.where(sup, sev, 0.0))
QB = np.stack([np.quantile(gres[band_of_failure == j], grid) for j in range(3)])
draws = np.where(sup[:, None], np.exp(np.where(sup, sev, m)[:, None] + QB[band]), np.exp(m[:, None] + np.quantile(resid, grid)[None, :]))
fl, bu = bayes_decide(*posterior_means(P), draws)
print(f"{int(fl.sum())} flags, mean budget {bu[fl == 1].mean():.0f}")
write_submission(te.id, fl, bu, "v82_hedge")
''')
n.md("**Evidence.** Under every ranking family the posterior expects this file within about -42 to +14 of v51 on the public 30% "
     "(-11 overall). The public difference between two files this close has noise of about ±300.")
n.save("v82_hedge.ipynb")
