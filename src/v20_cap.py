"""v20: respect the repair-cost range [5, 8000] (8000 is the cap in training; no repair costs more).
 A) v20_cap:  v12_robust with budget = min(budget, 8000). A budget above 8000 wastes 0.02*(b-8000) whether or not the vehicle
    fails, so this is a certain saving (4 vehicles, 187 on the whole test). Flags unchanged.
 B) v20_clip: v12_robust re-decided with the per-vehicle cost draws clipped to [5, 8000] and budgets capped (decide_pv logic,
    with M clipped), all inputs identical. Changes mu for very expensive vehicles slightly. Found by an independent review."""
import numpy as np, pandas as pd
from common import ROOT, load, write_submission, log_result
from score import INSPECT, OVER, UNDER
from decide import decide_pv, Q_STAR
tr, te, S = load()
q, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); p = np.clip(sum(w * shift(q, k) for w, k in zip(W, GRID)), 0, 1)
ref = pd.read_csv(ROOT / "submissions/submission_v12_robust.csv")
fl0, bu0 = decide_pv(p, m, resid); assert (fl0 == ref.flag.to_numpy()).all() and np.allclose(bu0.round(2), ref.budget, atol=0.011)
write_submission(te, ref.flag.to_numpy(), np.minimum(ref.budget.to_numpy(), 8000), "v20_cap")
K = 99; M = np.clip(np.exp(m[:, None] + np.quantile(resid, (np.arange(K) + .5) / K)[None, :]), 5, 8000); mu = M.mean(1)
b = np.zeros_like(p); need = p > 1 - Q_STAR
b[need] = np.minimum(np.clip(np.exp(m[need] + np.quantile(resid, 1 - (1 - Q_STAR) / p[need])), 5, 8000), 8000)
fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
fl = ((INSPECT + p * fail + (1 - p) * OVER * b) < p * mu).astype(int)
print(f"v20_clip: flags {fl.sum()} (v12 {fl0.sum()}), differ {int((fl != fl0).sum())}; budgets changed {int((np.abs(b - bu0) > 0.01).sum())}; max budget {b.max():.0f}")
write_submission(te, fl, b, "v20_clip")
log_result("v20", "v20_cap: v12_robust with budgets capped at 8000 (max repair cost)", np.nan, np.nan, notes="4 budgets capped; flags identical")
log_result("v20", "v20_clip: v12_robust re-decided with cost draws clipped to [5, 8000]", np.nan, np.nan, notes=f"{int(fl.sum())} flags; {int((fl != fl0).sum())} differ")
