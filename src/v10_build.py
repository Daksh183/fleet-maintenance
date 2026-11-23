"""v10: build the posterior-averaged decision and bracketing files from the saved posterior over k."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from calib import make
from common import ROOT, SEED, load, log_result, split_holdout, write_submission
from decide import decide_pv
from features import shape
from score import INSPECT, OVER, UNDER
from v5_percost_cfg import CM
CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
q = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y).predict(np.load(ROOT / "reports/p_te_v2.npy"))
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy())
Xa, Xt = pd.concat([f[S], shape(f)], axis=1), pd.concat([te[S], shape(te)], axis=1)
resid = lc - cross_val_predict(xgb.XGBRegressor(**CM9), Xa, lc, cv=KFold(5, shuffle=True, random_state=SEED))
m = xgb.XGBRegressor(**CM9).fit(Xa, lc).predict(Xt)
M = np.exp(m[:, None] + np.quantile(resid, (np.arange(99) + .5) / 99)[None, :]); mu = M.mean(1)
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy")
def ec(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return float(np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * mu).mean())
pbar = sum(w * shift(q, k) for w, k in zip(W, GRID))
print(f"posterior mean k {np.dot(W, GRID):.2f}; posterior-mean p on test {pbar.mean():.4f} (= single k of about "
      f"{GRID[np.argmin([abs(shift(q, k).mean() - pbar.mean()) for k in GRID])]})")
bay = decide_pv(pbar, m, resid); k2 = decide_pv(shift(q, 2), m, resid)
rows = {"v10_bayes": bay, "v10_k2": k2}
for nm in ("v9_k1", "v9_k4", "v9_em", "v5_platt"):
    d = pd.read_csv(ROOT / f"submissions/submission_{nm}.csv"); rows[nm] = (d.flag.to_numpy(), d.budget.to_numpy())
truth = (2, 2.5, 3, 3.5, 4, 5, 6.5)
tab = pd.DataFrame({f"true {k}": {n: ec(*fb, shift(q, k)) for n, fb in rows.items()} for k in truth})
tab["posterior avg"] = {n: sum(w * ec(*fb, shift(q, k)) for w, k in zip(W, GRID)) for n, fb in rows.items()}
tab["vs v9_em"] = tab["posterior avg"] / tab.loc["v9_em", "posterior avg"] - 1
print(tab.round(3).to_string()); print("flag rates:", {n: round(float(fb[0].mean()), 4) for n, fb in rows.items()})
for nm, (fl, bu) in (("v10_bayes", bay), ("v10_k2", k2)):
    write_submission(te, fl, bu, nm)
    log_result("v10", f"{nm}: v9 cost model, shift averaged over the posterior of k" if nm == "v10_bayes" else f"{nm}: v9 cost model, shift k=2",
               np.nan, np.nan, notes=f"test flag rate {fl.mean():.3%}")
