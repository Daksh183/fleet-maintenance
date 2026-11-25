"""v19: v12_robust with the instrument faults of the test depot removed from the classifier's inputs (a v1-style data fix).
 - s120 is the same counter as s126 (train: s120 = 5.9321*s126 exactly). In test s120 = 5.9321*s126 + 16,023,789 (residual 1e-5):
   a constant offset. Fix: subtract it. This keeps the depot's genuinely heavier wear (unlike quantile mapping in v14/v15).
 - s122 tracks s126 (Spearman 0.96); relative to s126 it is scaled by ~2.55 in test (median s122/s126 0.0546 vs 0.0214). Fix: divide.
Single-seed classifier exactly as v12 (v2 params, fit on all of train); cost model unchanged (it excludes these sensors)."""
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from calib import make
from common import ROOT, load, split_holdout, write_submission, log_result
from decide import decide_pv
from v2_params import PARAMS
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy()
x = te[["s120", "s126"]].dropna(); a, off = np.linalg.lstsq(np.c_[x.s126, np.ones(len(x))], x.s120, rcond=None)[0]
xr = tr[["s122", "s126"]].dropna(); xt = te[["s122", "s126"]].dropna(); xr, xt = xr[xr.s126 > 0], xt[xt.s126 > 0]
sc = np.median(xt.s122 / xt.s126) / np.median(xr.s122 / xr.s126)
print(f"s120 offset {off:,.0f} (slope {a:.4f}); s122 scale {sc:.3f}")
te_fix = te.copy(); te_fix["s120"] = te.s120 - off; te_fix["s122"] = te.s122 / sc
mdl = xgb.XGBClassifier(**PARAMS).fit(tr[S], tr.failure)
Pl = make("platt").fit(np.load(ROOT / "reports/p_oof_v2.npy"), y)
q0 = Pl.predict(mdl.predict_proba(te[S])[:, 1]); q1 = Pl.predict(mdl.predict_proba(te_fix[S])[:, 1])
qa = Pl.predict(mdl.predict_proba(te_fix.assign(s122=te.s122)[S])[:, 1])          # s120 fix only
qv12, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
print(f"refit reproduces v12 q: max |diff| {np.abs(q0 - qv12).max():.2e}")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda qq, lam=1.0: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))
ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv"); fl0, bu0 = decide_pv(pavg(qv12), m, resid); target = int(fl0.sum())
def matched(qq):
    lo, hi = 0.2, 6.0
    for _ in range(30):
        lam = (lo + hi) / 2
        if decide_pv(pavg(qq, lam), m, resid)[0].sum() < target: lo = lam
        else: hi = lam
    return lam, *decide_pv(pavg(qq, lam), m, resid)
for nm, qq in (("v19_fix120", qa), ("v19_fix", q1)):
    lam, fl, bu = matched(qq); both = (fl == 1) & (fl0 == 1)
    fl_pos, _ = decide_pv(pavg(qq), m, resid)
    print(f"{nm}: mean q {qq.mean():.4f} (v12 {qv12.mean():.4f}); Spearman {spearmanr(qv12, qq)[0]:.4f}; flags at the v10 posterior {int(fl_pos.sum())}; "
          f"matched lambda {lam:.3f}; flags differ from v12 {int((fl != fl0).sum())}; budget mean abs change {np.abs(bu[both]-bu0[both]).mean():.1f}")
    write_submission(te, fl, bu, nm)
    log_result("v19", f"{nm}: v12_robust with test instrument faults fixed for the classifier ({'s120 offset' if nm=='v19_fix120' else 's120 offset + s122 scale'}), flag count matched",
               np.nan, np.nan, notes=f"{int(fl.sum())} flags; lambda {lam:.3f}")
np.save(ROOT / "reports/v19_q.npy", np.vstack([qa, q1]))
