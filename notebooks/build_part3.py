"""Builds the v10 notebook: posterior over the depot-shift factor k from all public scores."""
from nbkit import NB, SHAPE

n = NB("v10: What shift do the public scores support? (posterior over k)")
n.header("v10", "ready to upload on 22 Nov: `v10_bayes`, `v10_k2` (and the earlier `v9_k4`) → `outputs/submission_v10_*.csv`",
         "**Question:** after nine public scores, how large is the failure-rate shift between depots (the odds factor *k*), and which decision "
         "should we upload? Instead of reading one score at a time, the whole set of scores is used jointly: for every candidate *k* we simulate "
         "what **each uploaded file would have scored**, then ask which *k* makes the observed vector of scores most plausible.")
n.code(SHAPE)
n.code(r'''
from scipy.stats import multivariate_normal
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}                      # v9 cost model
Pl = Platt().fit(fit_oof(dev[S]), y)
p_te_raw = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure).predict_proba(te[S])[:, 1]
q = Pl.predict(p_te_raw)                                                # calibrated test probabilities (old-depot scale)
f_all = tr[tr.failure == 1]; lc_all = np.log(f_all.repair_cost.to_numpy())
m_te, resid = cost_model(CM9, pd.concat([f_all[S], shape(f_all)], axis=1), lc_all, pd.concat([te[S], shape(te)], axis=1))
K = 99
M = np.exp(m_te[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :]); mu = M.mean(1)   # (vehicle, cost draw)
nte = len(te)
''')
n.md(r"""
### 1. The generative model of a public score
For a true shift $k$: vehicle $i$ fails with probability $\text{shift}(q_i,k)$ ($q_i$ = calibrated probability, $\text{shift}(p,k)=kp/(1-p+kp)$), its repair cost is drawn from the v9 per-vehicle cost distribution, and the public score is the total cost of a random 30% of the test vehicles. All uploaded files are scored on the **same** subset and the **same** failures, so their scores are strongly correlated; the simulation captures that. A multivariate normal is fitted to the simulated score vector for each $k$, a relative model-error term `eps` is added to its covariance (our probabilities and cost model are not exact), and the observed vector is evaluated.
""")
n.code(r'''
OBS = {"v2": 74933.1668, "v3": 27760.1404, "v3b": 29529.7466, "v4_platt": 23875.6666, "v5_raw": 26060.5718,
       "v5_platt": 22968.5508, "v9_k1": 19600.792, "v7_k4": 19067.165, "v9_em": 18058.5558}
files = {k: pd.read_csv(f"../submissions/submission_{k}.csv") for k in OBS}
Fm = np.column_stack([files[k].flag.to_numpy() for k in OBS]); Bm = np.column_stack([files[k].budget.to_numpy() for k in OBS])
obs = np.array(list(OBS.values())); R = 1200; rng = np.random.default_rng(SEED)
GRID = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12]
mean, cov = {}, {}
for k in GRID:
    p = shift(q, k); T = np.zeros((R, len(OBS)))
    for r in range(R):
        fail = rng.random(nte) < p
        c_sim = np.where(fail, M[np.arange(nte), rng.integers(0, K, nte)], 0.0)
        pub = rng.random(nte) < 0.3; cc = c_sim[pub][:, None]; Fp, Bp = Fm[pub], Bm[pub]
        T[r] = np.where(Fp == 1, INSPECT + UNDER * np.maximum(cc - Bp, 0) + OVER * np.maximum(Bp - cc, 0), cc).sum(0)
    mean[k], cov[k] = T.mean(0), np.cov(T.T)
sim = pd.DataFrame({k: mean[k] for k in GRID}, index=list(OBS)).round(-2).astype(int); sim.insert(0, "observed", obs.round(0).astype(int))
print("simulated mean public score of each uploaded file under each true k:"); print(sim.to_string())
''')
n.md("### 2. Posterior over k (uniform prior on the grid), and how sensitive it is to the model-error term")
n.code(r'''
def posterior(eps):
    L = np.array([multivariate_normal(mean=mean[k], cov=cov[k] + np.diag((eps * mean[k]) ** 2), allow_singular=True).logpdf(obs) for k in GRID])
    w = np.exp(L - L.max()); return w / w.sum(), L
post = {}
for eps in (0.03, 0.06, 0.10, 0.15):
    w, L = posterior(eps); post[eps] = w; cdf = np.cumsum(w)
    print(f"eps {eps:.2f}: posterior mean k {np.dot(w, GRID):.2f}   90% interval [{GRID[int(np.searchsorted(cdf, .05))]}, {GRID[min(int(np.searchsorted(cdf, .95)), len(GRID) - 1)]}]   mode {GRID[int(w.argmax())]}")
W = post[0.06]
print("\nposterior over k (eps 0.06):", {k: round(float(x), 3) for k, x in zip(GRID, W)})
kb = GRID[int(W.argmax())]
z = (obs - mean[kb]) / np.sqrt(np.diag(cov[kb]) + (0.06 * mean[kb]) ** 2)
print(f"standardised residuals at k={kb} (all within ~1 = the model explains the nine scores):", dict(zip(OBS, np.round(z, 2))))
''')
n.md(r"""
### 3. Decisions: the posterior-averaged probability, and the cost of each candidate under the posterior
For a fixed decision the expected cost is linear in $p$, so the Bayes decision under uncertainty about $k$ is made with $\bar p=\sum_k w_k\,\text{shift}(q,k)$. Expected cost per vehicle on the **whole** test set is computed exactly (not by simulation) for each true $k$.
""")
n.code(r'''
def ec(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return float(np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).mean())
pbar = sum(w * shift(q, k) for w, k in zip(W, GRID))
bay, k2 = decide_pv(pbar, m_te, resid), decide_pv(shift(q, 2), m_te, resid)
rows = {"v10_bayes": bay, "v10_k2": k2}
for nm in ("v9_k1", "v9_k4", "v9_em", "v5_platt"):
    d = pd.read_csv(f"../submissions/submission_{nm}.csv"); rows[nm] = (d.flag.to_numpy(), d.budget.to_numpy())
truth = (2, 2.5, 3, 3.5, 4, 5, 6.5)
tab = pd.DataFrame({f"true {k}": {nm: ec(*fb, shift(q, k)) for nm, fb in rows.items()} for k in truth})
tab["posterior avg"] = {nm: sum(w * ec(*fb, shift(q, k)) for w, k in zip(W, GRID)) for nm, fb in rows.items()}
tab["vs v9_em"] = tab["posterior avg"] / tab.loc["v9_em", "posterior avg"] - 1
print(f"posterior mean k {np.dot(W, GRID):.2f}; mean posterior p on test {pbar.mean():.4f}\nexpected cost / vehicle on the whole test set:")
print(tab.round(3).to_string()); print("test flag rates:", {nm: round(float(fb[0].mean()), 4) for nm, fb in rows.items()})
''')
n.md("### 4. Files")
n.code(r'''
write_submission(te.id, bay[0], bay[1], "v10_bayes")
write_submission(te.id, k2[0], k2[1], "v10_k2")
''')
n.md(r"""
## Results and decisions

- **The nine scores support k ≈ 3** (posterior mean 3.0, 90% interval 2-4.5; it moves between 2.5 and 3.2 as the model-error term goes from 0.15 to 0.03). All standardised residuals are within about ±1, so the simulation explains the whole set of scores. **The EM estimate (6.5) has essentially zero posterior mass** (EM assumes only the failure rate changed; the extra covariate shift between depots biases it upward, as suspected in v7), and k = 4 sits at the edge (probability ~0.10).
- **Decisions.** The posterior-averaged decision (`v10_bayes`, behaves like k ≈ 3.0, flags 8.9% of test) has the lowest expected cost: **5.1% below `v9_em`** (the best upload so far), 4.4% for `v9_k4`, 3.9% for `v10_k2`. If the public set behaves like the model, that is roughly **17,100-17,300** public points against 18,059.
- **Over-correcting is cheaper than under-correcting, but not free:** `v9_em` (k=6.53) costs about 5% more than the Bayes build in expectation; `v9_k1` (no shift) costs 4.9% *more than `v9_em`*, and `v5_platt` (old cost model, no shift) 17.9% more.

**Caveats**
- All of this rests on our own model of the test depot: failures ~ Bernoulli(shift(q, k)) with **one constant k for all vehicles** and the v9 cost model. If the extra failures are concentrated in some vehicle types, or repair costs are higher at the new depot (which would act like a higher k), the inferred k is an *effective* value, not a measured failure rate.
- The public subset is ~6,000 vehicles with few failures. The model-error term `eps` is a crude stand-in for everything the simulation gets wrong; the posterior shifts a little with it (above) but the conclusion that k is well below 6.5 does not.
- **Public scores cannot separate the close candidates** (`v10_bayes`, `v9_k4`, `v10_k2` differ by ~1-2% in expected cost, i.e. ~150-350 points, below the public noise). Uploading them anyway gives the score-vs-k curve (k = 1, 2, 3, 4, 6.5 with the v9 cost model) for the report, but the final choice should rest on the posterior, not on which of them scores lowest on the public subset.
- Remaining improvement is more likely in the models (cost model, failure ranking) than in fine-tuning k.
""")
n.save("v10_shift_posterior.ipynb")
