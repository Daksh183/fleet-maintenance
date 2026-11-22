"""v14: v13_bag with ONE change: the two instrument-like drifted test sensors (s120: offset+scale, s122: x2.7; s126/s128 differ only in the upper tail, which looks like a real shift and is left alone) are quantile-mapped onto the training
distribution before the classifier scores the test vehicles. The number of flagged vehicles is matched to v13_bag (scale lambda on k),
so the comparison is about RANKING, not level.  (The cost model already excludes these sensors.)"""
import numpy as np, pandas as pd, xgboost as xgb
from scipy.optimize import brentq
from sklearn.model_selection import KFold
from calib import make
from common import ROOT, SEED, folds, load, log_result, split_holdout, write_submission
from decide import decide_pv
from features import shape
from v2_params import PARAMS
from v5_percost_cfg import CM
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); F = folds(y)
DRIFT = ["s120", "s122", "s126", "s128"]; MAP = ["s120", "s122"]; CM9 = {**CM, "n_estimators": 1000, "max_depth": 2}; NC, NK = 5, 8
qs = [.05, .25, .5, .75, .95]
print("quantiles train | test | ratio (test/train):")
for s in DRIFT:
    a, b = np.quantile(tr[s].dropna(), qs), np.quantile(te[s].dropna(), qs)
    print(f"  {s}: train {np.round(a, 3).tolist()}  test {np.round(b, 3).tolist()}  ratio {np.round(b / np.where(a == 0, np.nan, a), 2).tolist()}  NaN share train {tr[s].isna().mean():.3f} test {te[s].isna().mean():.3f}")
def qmap(col):                                   # test value -> training value at the same quantile
    x = te[col]; r = x.rank(pct=True, method="average"); out = x.copy(); ok = x.notna()
    out[ok] = np.quantile(tr[col].dropna(), np.clip((r[ok] * ok.sum() - 0.5) / ok.sum(), 0, 1)); return out
te_map = te.copy()
for s in MAP: te_map[s] = qmap(s)
oof = np.zeros(len(dev)); pr = np.zeros(len(te)); pm = np.zeros(len(te)); gain = 0
for s in range(NC):
    prm = {**PARAMS, "random_state": SEED + s}
    for a, b in F: oof[b] += xgb.XGBClassifier(**prm).fit(dev.loc[a, S], y[a]).predict_proba(dev.loc[b, S])[:, 1] / NC
    mdl = xgb.XGBClassifier(**prm).fit(tr[S], tr.failure)
    pr += mdl.predict_proba(te[S])[:, 1] / NC; pm += mdl.predict_proba(te_map[S])[:, 1] / NC
    g = pd.Series(mdl.get_booster().get_score(importance_type="total_gain")); gain += g.reindex(MAP).fillna(0).sum() / g.sum() / NC
    print("classifier seed", s, flush=True)
print(f"classifier gain share of the two mapped sensors: {gain:.3f}")
pl = make("platt").fit(oof, y); q, qm = pl.predict(pr), pl.predict(pm)
from scipy.stats import spearmanr
topk = int(0.088 * len(te)); ov = len(set(np.argsort(-q)[:topk]) & set(np.argsort(-qm)[:topk]))
print(f"mean test p: raw {q.mean():.4f} mapped {qm.mean():.4f}; Spearman {spearmanr(q, qm)[0]:.4f}; overlap of top-{topk}: {ov}")
f = tr[tr.failure == 1]; lc = np.log(f.repair_cost.to_numpy()); keep = [s for s in S if s not in DRIFT]
Xa, Xt = pd.concat([f[keep], shape(f)], axis=1), pd.concat([te[keep], shape(te)], axis=1)
inner = KFold(5, shuffle=True, random_state=SEED); m = np.zeros(len(te)); mo = np.zeros(len(f))
for s in range(NK):
    prm = {**CM9, "random_state": SEED + s}
    for a, b in inner.split(Xa): mo[b] += xgb.XGBRegressor(**prm).fit(Xa.iloc[a], lc[a]).predict(Xa.iloc[b]) / NK
    m += xgb.XGBRegressor(**prm).fit(Xa, lc).predict(Xt) / NK
resid = lc - mo
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy")
pavg = lambda qq, lam: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))
ref = pd.read_csv(ROOT / "submissions/submission_v13_bag.csv"); target = int(ref.flag.sum())
lo, hi = 0.5, 4.0                                   # match the NUMBER of flagged vehicles to v13_bag by bisection on lambda
for _ in range(25):
    lam = (lo + hi) / 2
    if decide_pv(pavg(qm, lam), m, resid)[0].sum() < target: lo = lam
    else: hi = lam
print(f"flag-count matching: k scaled by lambda = {lam:.3f} (mean p {pavg(qm, lam).mean():.4f} vs v13_bag {pavg(q, 1.0).mean():.4f}); target {target} flags")
fl, bu = decide_pv(pavg(qm, lam), m, resid)
both = (ref.flag.to_numpy() == 1) & (fl == 1)
print(f"flag rate {fl.mean():.3%} (v13_bag {ref.flag.mean():.3%}); flags differing from v13_bag {int((ref.flag.to_numpy() != fl).sum())}; "
      f"budget change on commonly flagged: mean abs {np.abs(bu[both] - ref.budget.to_numpy()[both]).mean():.1f}")
write_submission(te, fl, bu, "v14_map")
log_result("v14", "v14_map: v13_bag with test s120/s122 quantile-mapped to the training distribution for the classifier (flag count matched)", np.nan, np.nan,
           notes=f"test flag rate {fl.mean():.3%}; {int((ref.flag.to_numpy() != fl).sum())} flags differ from v13_bag; lambda {lam:.3f}")
