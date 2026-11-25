"""v31: textbook label-shift estimators for the test depot's failure rate (literature check of our k posterior).
 - BCTS + EM/MLLS (Alexandari et al. 2020; Saerens et al. 2002): calibrate logits on dev OOF with temperature T and bias b
   (binary: sigmoid(T*z + b), fitted by NLL), then EM re-estimates the target prior on test.
 - BBSE (Lipton et al. 2018): hard predictions at a threshold; pi_test = (rate_test - FPR) / (TPR - FPR) from dev confusion matrix.
Each estimate is converted to the odds factor k (test odds / train odds) and compared with the harness posterior."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from common import ROOT, load, split_holdout
tr, te, S = load(); dev, _ = split_holdout(tr); y = dev.failure.to_numpy(); pi_tr = y.mean()
oof = np.load(ROOT / "reports/p_oof_v2.npy"); pte = np.load(ROOT / "reports/p_te_v2.npy")
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); sig = lambda v: 1 / (1 + np.exp(-v))
def bcts_fit(z, y):
    nll = lambda th: -np.mean(y * np.log(sig(th[0] * z + th[1]) + 1e-12) + (1 - y) * np.log(1 - sig(th[0] * z + th[1]) + 1e-12))
    return minimize(nll, [1.0, 0.0], method="Nelder-Mead").x
def em_prior(p, pi_src, iters=500):
    pi = pi_src
    for _ in range(iters):
        w1, w0 = pi / pi_src, (1 - pi) / (1 - pi_src); post = w1 * p / (w1 * p + w0 * (1 - p)); pi_new = post.mean()
        if abs(pi_new - pi) < 1e-10: break
        pi = pi_new
    return pi
odds = lambda a: a / (1 - a); K = lambda pt: odds(pt) / odds(pi_tr)
rows = []
for nm, (zo, zt) in {"v2 (v12 classifier)": (lg(oof), lg(pte)), "nos122 (v22)": (None, None)}.items():
    if zo is None: continue
    T, b = bcts_fit(zo, y); pd_ = sig(T * zo + b); pt = sig(T * zt + b)
    print(f"{nm}: BCTS T={T:.3f} b={b:+.3f}; calibrated dev mean {pd_.mean():.4f} (true {pi_tr:.4f}); test mean calibrated p {pt.mean():.4f}")
    pi_em = em_prior(pt, pd_.mean()); rows.append(("BCTS+EM", nm, pi_em, K(pi_em)))
    for thr in (0.02, 0.05, 0.1, 0.2, 0.5):
        tpr = (pd_[y == 1] >= thr).mean(); fpr = (pd_[y == 0] >= thr).mean(); rt = (pt >= thr).mean()
        pi_b = (rt - fpr) / (tpr - fpr); rows.append((f"BBSE thr {thr}", nm, pi_b, K(pi_b) if 0 < pi_b < 1 else np.nan))
R = pd.DataFrame(rows, columns=["method", "classifier", "test_failure_rate", "k_odds_factor"]); print(R.round(4).to_string(index=False))
# what the harness posterior implies: mean test failure rate under the v30 posterior
P = pd.read_csv(ROOT / "reports/v30_post.csv"); print(f"\ntrain failure rate {pi_tr:.4f}; k posterior from public scores (v10): ~3 (90% 2-4.5)")
R.to_csv(ROOT / "reports/v31_labelshift.csv", index=False)
