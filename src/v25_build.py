"""v25_tilt: v12_robust ranking tilted toward high-wear vehicles: q' = sigmoid(logit(q_v12) + c * z_s120), c = 0.6 (best fit to the
20 public scores, v25_tilt.py; +2.7 log-lik vs no tilt). Flag count matched to v12 (1,761), v12 cost model, budgets capped at 8000.
Prints the model-expected public difference vs v20_cap if the truth has tilt 0 / 0.3 / 0.6 / 0.9."""
import numpy as np, pandas as pd
from common import ROOT, load, log_result, write_submission
from decide import decide_pv
from score import INSPECT, OVER, UNDER
tr, te, S = load()
q, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
x = np.log1p(np.nan_to_num(np.maximum(te.s120.to_numpy(), 0), nan=np.nanmedian(te.s120))); z = (x - x.mean()) / x.std()
sig = lambda v: 1 / (1 + np.exp(-v)); lg = np.log(q / (1 - q))
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); pavg = lambda qq, lam=1.0: sum(w * shift(qq, k * lam) for w, k in zip(W, GRID))
ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv"); f0, b0 = ref.flag.to_numpy(), ref.budget.to_numpy(); target = int(f0.sum())
def matched(qq):
    lo, hi = 0.05, 10.0
    for _ in range(40):
        lam = (lo + hi) / 2
        if decide_pv(pavg(qq, lam), m, resid)[0].sum() < target: lo = lam
        else: hi = lam
    fl, bu = decide_pv(pavg(qq, hi), m, resid); return fl, np.minimum(bu, 8000), hi
Kd = 99; M = np.clip(np.exp(m[:, None] + np.quantile(resid, (np.arange(Kd) + .5) / Kd)[None, :]), 5, 8000)
def ec(fl, b, p):
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * M.mean(1))
out = {}
for c in (0.3, 0.6, 0.9):
    fl, bu, lam = matched(sig(lg + c * z)); out[c] = (fl, bu)
    d = []
    for ct in (0.0, 0.3, 0.6, 0.9):
        pt = pavg(sig(lg + ct * z)); pt = pt * (pavg(q).mean() / pt.mean())          # truth with tilt ct, same mean failure rate
        d.append(0.3 * (ec(fl, bu, pt) - ec(f0, b0, pt)).sum())
    print(f"tilt {c}: lambda {lam:.3f}, {int(fl.sum())} flags, {int((fl != f0).sum())} differ from v20_cap; expected public diff if true tilt "
          f"0/0.3/0.6/0.9: " + " ".join(f"{v:+.0f}" for v in d), flush=True)
fl, bu = out[0.6]
print("added flags: median test s126", np.nanmedian(te.s126[(fl == 1) & (f0 == 0)]), " removed:", np.nanmedian(te.s126[(fl == 0) & (f0 == 1)]))
write_submission(te, fl, bu, "v25_tilt")
log_result("v25", "v25_tilt: v12_robust ranking tilted toward high wear (logit + 0.6*z_s120, fitted to 20 public scores), flag count matched, budgets capped",
           np.nan, np.nan, notes=f"{int(fl.sum())} flags; {int((fl != f0).sum())} differ from v20_cap")
