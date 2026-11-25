"""v23: repair costs at the new depot higher than in training (level alpha). Consistent re-decision on the v12_robust base:
cost draws exp(m + resid) * alpha (clipped to [5, 8000]) feed BOTH the budget (newsvendor) and the flag rule (more flags, as
missed failures cost more). Classifier and k posterior unchanged. Prints the model-expected public difference vs v20_cap
under several true alphas, to show how informative an upload is."""
import numpy as np, pandas as pd
from common import ROOT, load, log_result, write_submission
from score import INSPECT, OVER, UNDER
tr, te, S = load()
q, m = np.load(ROOT / "reports/v18_qm.npy"); resid = np.load(ROOT / "reports/v18_resid.npy")
shift = lambda p, k: k * p / (1 - p) / (1 + k * p / (1 - p))
GRID, W = np.load(ROOT / "reports/v10_posterior.npy"); p = sum(w * shift(q, k) for w, k in zip(W, GRID))
K = 199; R = np.quantile(resid, (np.arange(K) + .5) / K)
def draws(al): return np.clip(np.exp(m[:, None] + R[None, :]) * al, 5, 8000)
def decide(al):
    M = draws(al); Q = 1 - (1 - UNDER / (UNDER + OVER)) / np.maximum(p, 1e-9)
    b = np.where(Q > 0, np.array([np.quantile(M[i], min(max(Q[i], 0), 1)) for i in range(len(p))]), 0.0)
    b = np.minimum(b, 8000)
    fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    fl = (INSPECT + p * fail + (1 - p) * OVER * b < p * M.mean(1)).astype(int)
    return fl, np.where(fl == 1, b, 0.0)
def exp_cost(fl, b, al):
    M = draws(al); fail = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(1)
    return np.where(fl == 1, INSPECT + p * fail + (1 - p) * OVER * b, p * M.mean(1))
ref = pd.read_csv(ROOT / "submissions/submission_v20_cap.csv"); f0, b0 = ref.flag.to_numpy(), ref.budget.to_numpy()
files = {"v20_cap": (f0, b0)}
for al in (1.0, 1.25, 1.5, 2.0): files[f"a{al}"] = decide(al)
print("alpha=1 re-decision vs v20_cap: flags differ", int((files["a1.0"][0] != f0).sum()))
print(f"{'file':8s} {'flags':>6s} {'mean b(flag)':>12s} | model-expected public diff vs v20_cap if true alpha = 1.0 / 1.25 / 1.5 / 2.0")
for nm, (fl, b) in files.items():
    d = [0.3 * (exp_cost(fl, b, t) - exp_cost(f0, b0, t)).sum() for t in (1.0, 1.25, 1.5, 2.0)]
    print(f"{nm:8s} {int(fl.sum()):6d} {b[fl == 1].mean():12.0f} | " + "  ".join(f"{x:+8.0f}" for x in d))
for al in (1.5,):
    fl, b = files[f"a{al}"]; write_submission(te, fl, b, "v23_alpha15")
    log_result("v23", "v23_alpha15: v12_robust re-decided with test repair costs x1.5 (budgets and flags), draws clipped [5, 8000]",
               np.nan, np.nan, notes=f"{int(fl.sum())} flags; {int((fl != f0).sum())} differ from v20_cap")
