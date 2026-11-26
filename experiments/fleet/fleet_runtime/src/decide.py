"""Decision layer: (p(fail), repair-cost distribution) -> (flag, budget).

Budget (newsvendor): under-cost 0.20, over-cost 0.02 -> critical ratio q* = 0.20/0.22 = 0.909.
  Mixture CDF of c is (1-p) + p*F(b), so b* = 0 if p <= 1-q*, else F^{-1}(1 - (1-q*)/p).
Flag: flag iff E[cost | flag, b*] < p*mu  (with b*=0 this is p*mu > 12.5).
"""
import numpy as np

from score import INSPECT, OVER, UNDER

Q_STAR = UNDER / (UNDER + OVER)


def decide_pv(p, m, resid, fail_costs=None, K=99, scale=None):
    """Per-vehicle repair-cost distribution: log c | fail ~ m_i + resid (empirical residuals).

    m: predicted log repair cost per vehicle. resid: out-of-fold log residuals of the cost model.
    fail_costs: if given, the BUDGET comes from this global distribution (ablation: per-vehicle
    flag decision, global budget); otherwise the budget is the per-vehicle newsvendor quantile.
    Back-transform uses residual quantiles (not exp(mean)), so there is no log-bias.
    """
    p = np.clip(np.asarray(p, float), 0, 1)
    m = np.asarray(m, float)
    # scale: optional per-vehicle spread (v9). Then `resid` are STANDARDISED residuals and the
    # log-cost distribution of vehicle i is m_i + scale_i * resid.
    sc = np.ones_like(m) if scale is None else np.asarray(scale, float)
    qs = np.quantile(resid, (np.arange(K) + 0.5) / K)
    M = np.exp(m[:, None] + sc[:, None] * qs[None, :])               # (n, K) cost draws given failure
    mu = M.mean(axis=1)
    b = np.zeros_like(p)
    need = p > 1 - Q_STAR
    level = 1 - (1 - Q_STAR) / p[need]
    if fail_costs is None:
        b[need] = np.exp(m[need] + sc[need] * np.quantile(resid, level))
    else:
        b[need] = np.quantile(np.asarray(fail_costs, float), level)
    fail_part = (UNDER * np.maximum(M - b[:, None], 0) + OVER * np.maximum(b[:, None] - M, 0)).mean(axis=1)
    flag_cost = INSPECT + p * fail_part + (1 - p) * OVER * b
    return (flag_cost < p * mu).astype(int), b


def decide_global(p, fail_costs):
    """Same cost distribution F for every vehicle (empirical failure costs `fail_costs`)."""
    p = np.clip(np.asarray(p, float), 0, 1)
    fc = np.sort(np.asarray(fail_costs, float))
    mu = fc.mean()
    b = np.zeros_like(p)
    need = p > 1 - Q_STAR
    b[need] = np.quantile(fc, 1 - (1 - Q_STAR) / p[need])
    # E over failure draws of 0.2(c-b)+ + 0.02(b-c)+, per vehicle (fc is small, so broadcast)
    fail_part = (UNDER * np.maximum(fc[None, :] - b[:, None], 0)
                 + OVER * np.maximum(b[:, None] - fc[None, :], 0)).mean(axis=1)
    flag_cost = INSPECT + p * fail_part + (1 - p) * OVER * b
    return (flag_cost < p * mu).astype(int), b
